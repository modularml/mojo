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

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from mojo.importer import _resolve_cache_dir

_CACHE_FILENAME = "__init__.hash-0123456789abcdef.so"


def test_in_tree_cache_dir_is_default_when_writable() -> None:
    """A writable source tree uses the co-located `__mojocache__` directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mojo_dir = Path(tmp_dir)

        result = _resolve_cache_dir(
            "pkg.mod", mojo_dir, cache_filename=_CACHE_FILENAME
        )

    assert result == mojo_dir / "__mojocache__"


def test_read_only_tree_redirects_to_modular_cache_root() -> None:
    """A read-only source tree falls back to the Modular cache folder.

    The fallback location is whatever the Modular configuration resolves to
    (the `cache_dir` key in `modular.cfg` or the `MODULAR_CACHE_DIR`
    environment variable), namespaced by the fully-qualified module name.
    """
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        tempfile.TemporaryDirectory() as cache_root,
        patch("mojo.importer._cache_dir_is_writable", return_value=False),
        patch(
            "mojo.importer._modular_cache_root",
            return_value=Path(cache_root),
        ),
    ):
        result = _resolve_cache_dir(
            "max._kv_cache_ops",
            Path(tmp_dir),
            cache_filename=_CACHE_FILENAME,
        )

    assert (
        result
        == Path(cache_root) / "python_extensions" / "max" / "_kv_cache_ops"
    )


def test_prebuilt_artifact_in_read_only_tree_is_used() -> None:
    """A prebuilt `.so` in a read-only tree is reused, not redirected.

    This protects installations that ship precompiled artifacts inside a
    read-only `site-packages`: we must not recompile into the cache folder
    when a valid in-tree artifact already exists.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mojo_dir = Path(tmp_dir)
        in_tree_cache = mojo_dir / "__mojocache__"
        in_tree_cache.mkdir()
        (in_tree_cache / _CACHE_FILENAME).touch()

        with (
            patch("mojo.importer._cache_dir_is_writable", return_value=False),
            patch(
                "mojo.importer._modular_cache_root",
                return_value=Path("/should/not/be/used"),
            ),
        ):
            result = _resolve_cache_dir(
                "pkg.mod", mojo_dir, cache_filename=_CACHE_FILENAME
            )

    assert result == in_tree_cache


def test_read_only_tree_without_cache_root_falls_back_in_tree() -> None:
    """If the Modular cache folder can't be resolved, keep the in-tree path.

    Returning the in-tree directory preserves the original read-only error
    rather than masking it with an unrelated failure.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mojo_dir = Path(tmp_dir)

        with (
            patch("mojo.importer._cache_dir_is_writable", return_value=False),
            patch("mojo.importer._modular_cache_root", return_value=None),
        ):
            result = _resolve_cache_dir(
                "pkg.mod", mojo_dir, cache_filename=_CACHE_FILENAME
            )

    assert result == mojo_dir / "__mojocache__"
