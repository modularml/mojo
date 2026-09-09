# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import hashlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.util import spec_from_file_location
from pathlib import Path

from .paths import MojoCompilationError, MojoModulePath, find_mojo_module_in_dir
from .run import subprocess_run_mojo

# Name of the cache directory co-located with Mojo sources. This is the default
# location, used whenever the source tree is writable.
_IN_TREE_CACHE_DIR_NAME = "__mojocache__"

# Subdirectory used when the cache is redirected out of the source tree (e.g.
# into the Modular cache folder for a read-only `site-packages`). Compiled
# extensions are namespaced by their fully-qualified module name underneath
# this directory so that packages sharing a root-file stem (every package's
# root is `__init__`) do not collide in a shared cache directory.
_REDIRECTED_CACHE_SUBDIR = "python_extensions"

# ---------------------------------------
# Helper Functions
# ---------------------------------------


def _calculate_mojo_source_hash(mojo_dir: Path) -> str:
    """Calculates a truncated SHA256 hash of all .mojo files in a directory."""
    # Find all .mojo files recursively
    source_files = sorted(mojo_dir.rglob("*.mojo"))

    if not source_files:
        # This should be unreachable if the caller validates that mojo_dir
        # contains Mojo source files before calling this function.
        raise ImportError(
            f"Internal Error: No .mojo files found in directory '{mojo_dir}'"
            " for hashing."
        )

    hasher = hashlib.sha256()
    for file_path in source_files:
        try:
            # Add file path to hash to distinguish identical content in different files
            hasher.update(str(file_path.relative_to(mojo_dir)).encode("utf-8"))
            # Add file content to hash
            with open(file_path, "rb") as f:
                hasher.update(f.read())
        except (ValueError, UnicodeError, OSError) as e:
            raise ImportError(
                f"Could not process Mojo source file '{file_path}' for hashing"
            ) from e

    # Return only the first 16 characters of the hex digest, since the full
    # hash is quite long and this is just a best-effort heuristic to check for
    # changes.
    return hasher.hexdigest()[:16]


def _compile_mojo_to_so(root_mojo_path: Path, output_so_path: Path) -> None:
    """Compiles a Mojo file to a shared object library."""
    # Assertions from _build_mojo_file_to_python_extension_module
    assert root_mojo_path.is_file()
    assert output_so_path.suffix == ".so"

    mojo_cli_args = [
        # First arg is implicitly the `mojo` executable (handled by subprocess_run_mojo)
        "build",
        str(root_mojo_path),
        "--emit",
        "shared-lib",
        "-o",
        str(output_so_path),
    ]

    try:
        # Run the `mojo` that's embedded in the `max` package layout via subprocess_run_mojo.
        subprocess_run_mojo(mojo_cli_args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        error = MojoCompilationError.from_subprocess_error(
            root_mojo_path, mojo_cli_args, e
        )
        logging.error(str(error))
        # Propagate compilation errors as ImportError
        raise ImportError(
            "Import of Mojo module failed due to compilation error."
        ) from error
    except FileNotFoundError:
        # Handle case where mojo executable is not found
        raise ImportError(
            "Mojo executable not found via subprocess_run_mojo."
        ) from None


# TODO: Instead of being careful about only deleting old files, we could just
#   delete all files in the cache directory?
def _delete_matching_cached_files(
    cache_dir: Path, *, stem: str, ext: str
) -> None:
    """Removes outdated cache files for a given Mojo module."""
    if not cache_dir.is_dir():
        return

    for old_cache_file in cache_dir.glob(f"{stem}.*.{ext}"):
        os.remove(old_cache_file)


def _modular_cache_root() -> Path | None:
    """Resolves the Modular cache folder by querying the bundled `mojo`.

    Returns the directory reported by ``mojo --print-cache-location``, which
    honors the standard Modular configuration: the ``cache_dir`` key in
    ``modular.cfg``, the ``MODULAR_CACHE_DIR`` and ``MODULAR_HOME`` environment
    variables, and the XDG base directory specification. Delegating to `mojo`
    avoids duplicating that resolution logic here.

    Returns `None` if the location cannot be determined (e.g. the `mojo`
    executable is missing or too old to support the flag), in which case the
    caller falls back to the in-tree cache.

    This is not memoized on purpose: it is only reached on the read-only
    fallback path, where the cost of one extra subprocess is negligible next to
    the compile it gates. Re-resolving each time also keeps the result
    consistent with the live environment.
    """
    try:
        result = subprocess_run_mojo(
            ["--print-cache-location"], capture_output=True, check=True
        )
    # `subprocess_run_mojo` raises `RuntimeError` when the `mojo` executable
    # cannot be located, before it ever spawns a subprocess; catch it alongside
    # the subprocess/OS errors so this stays best-effort and the caller can fall
    # back to the in-tree cache (preserving the original read-only error).
    except (subprocess.SubprocessError, OSError, RuntimeError):
        return None

    location = result.stdout.decode().strip()
    return Path(location) if location else None


def _cache_dir_is_writable(in_tree_cache_dir: Path, mojo_dir: Path) -> bool:
    """Returns True if the in-tree cache directory can be written to.

    If the cache directory already exists, its own writability is checked;
    otherwise we check whether it could be created inside `mojo_dir`.
    """
    target = in_tree_cache_dir if in_tree_cache_dir.is_dir() else mojo_dir
    # `os.access` is a deliberate best-effort heuristic, not a guarantee: it
    # uses the real uid/gid and can disagree with actual writability on NFS,
    # ACL, or overlay mounts, and always returns True for root. Both failure
    # modes are benign here -- a false positive reverts to the previous behavior
    # (an `OSError` at `makedirs`), and a false negative merely redirects the
    # cache unnecessarily.
    return os.access(target, os.W_OK)


def _resolve_cache_dir(
    name: str, mojo_dir: Path, *, cache_filename: str
) -> Path:
    """Determines where the compiled extension for `name` should be cached.

    Resolution order:

    1. The in-tree `__mojocache__` directory next to the Mojo sources, when it
       already holds a matching artifact or is writable. This is the default
       and leaves existing behavior unchanged.
    2. The Modular cache folder (see `_modular_cache_root`) when the in-tree
       directory is read-only, as happens for packages installed into a
       read-only `site-packages`. This location is configurable via the
       `cache_dir` key in `modular.cfg` or the `MODULAR_CACHE_DIR` environment
       variable.

    In the redirected case (2) the path is namespaced by the fully-qualified
    module `name` so that packages sharing a root stem do not collide in a
    shared cache directory.
    """
    in_tree_cache_dir = mojo_dir / _IN_TREE_CACHE_DIR_NAME

    # Prefer the in-tree cache when a prebuilt artifact is already present or
    # the directory is writable. This keeps the default behavior unchanged.
    if (in_tree_cache_dir / cache_filename).is_file() or _cache_dir_is_writable(
        in_tree_cache_dir, mojo_dir
    ):
        return in_tree_cache_dir

    # The in-tree directory is read-only; redirect to the Modular cache folder.
    if cache_root := _modular_cache_root():
        return cache_root / _REDIRECTED_CACHE_SUBDIR / name.replace(".", "/")

    # As a last resort fall back to the in-tree directory; this preserves the
    # original read-only error if the directory ultimately cannot be created.
    return in_tree_cache_dir


# ---------------------------------------
# Define custom importer
# ---------------------------------------


# Resources:
#    https://docs.python.org/3/reference/import.html#the-meta-path
#    https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.find_spec
#    https://docs.python.org/3/library/importlib.html#importlib.machinery.ExtensionFileLoader
#    https://peps.python.org/pep-0489/#module-creation-phase
class MojoImporter:
    def find_spec(  # noqa: ANN201
        self,
        name: str,
        import_path: Sequence[str] | None,
        target: object | None = None,
    ):
        name_path = name.replace(".", "/")

        mojo_module: MojoModulePath | None = None
        # Search sys.path for the Mojo source file or package
        for path_entry in sys.path:
            # Use the helper function to check this directory
            mojo_module = find_mojo_module_in_dir(Path(path_entry), name_path)

            if mojo_module:
                break  # Found the source, stop searching sys.path

        # If no Mojo source found, let other importers handle it.
        if not mojo_module:
            return None

        # `root_mojo_path` is the path to the specific Mojo source file.
        root_mojo_path = mojo_module.path

        # `mojo_dir` is the directory containing the Mojo source file(s); it is
        # the directory that will be hashed to check for changes.
        mojo_dir = root_mojo_path.parent

        # Calculate hash.
        current_hash = _calculate_mojo_source_hash(mojo_dir)

        cache_filename = f"{root_mojo_path.stem}.hash-{current_hash}.so"

        # Determine the cache location. This defaults to an in-tree
        # `__mojocache__` directory but is redirected to a writable location
        # when the source tree is read-only or an override is configured.
        cache_dir = _resolve_cache_dir(
            name, mojo_dir, cache_filename=cache_filename
        )

        expected_cache_file = cache_dir / cache_filename

        # Compile if cache doesn't exist or is invalid
        if not expected_cache_file.is_file():
            # No matching cached file exists, so compile the Mojo source file.
            os.makedirs(cache_dir, exist_ok=True)
            # Delete any non-matching cached .so's, to prevent the number of
            # stale cached files from growing without bound.
            _delete_matching_cached_files(
                cache_dir, stem=root_mojo_path.stem, ext="so"
            )
            _compile_mojo_to_so(root_mojo_path, expected_cache_file)

        # If we reach here, expected_cache_file should exist (either pre-existing or just compiled)
        assert expected_cache_file.is_file()

        # Constructs an ExtensionFileLoader automatically based on the .so
        # file extension.
        return spec_from_file_location(
            name, str(expected_cache_file), submodule_search_locations=None
        )


# -------------------------------------------------------
# Side Effect: Add custom importer to the Python metapath
# -------------------------------------------------------

sys.meta_path.append(MojoImporter())
