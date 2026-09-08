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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/black/cache.py
#
# ===----------------------------------------------------------------------=== #

"""Caching of formatted files with feature-based invalidation."""

import os
import pickle
import tempfile
from collections.abc import Iterable
from pathlib import Path

from _mblack_version import version as __version__
from platformdirs import user_cache_dir

from mblack.mode import Mode

# types
Timestamp = float
FileSize = int
CacheInfo = tuple[Timestamp, FileSize]
Cache = dict[str, CacheInfo]


def get_cache_dir() -> Path:
    """Get the cache directory used by black.

    Users can customize this directory on all systems using `MBLACK_CACHE_DIR`
    environment variable. By default, the cache directory is the user cache directory
    under the black application.

    This result is immediately set to a constant `mblack.cache.CACHE_DIR` as to avoid
    repeated calls.
    """
    # NOTE: Function mostly exists as a clean way to test getting the cache directory.
    default_cache_dir = user_cache_dir("mblack", version=__version__)
    cache_dir = Path(os.environ.get("MBLACK_CACHE_DIR", default_cache_dir))
    return cache_dir


CACHE_DIR = get_cache_dir()


def read_cache(mode: Mode) -> Cache:
    """Read the cache if it exists and is well formed.

    If it is not well formed, the call to write_cache later should resolve the issue.
    """
    cache_file = get_cache_file(mode)
    if not cache_file.exists():
        return {}

    with cache_file.open("rb") as fobj:
        try:
            cache: Cache = pickle.load(fobj)
        except (pickle.UnpicklingError, ValueError, IndexError):
            return {}

    return cache


def get_cache_file(mode: Mode) -> Path:
    return CACHE_DIR / f"cache.{mode.get_cache_key()}.pickle"


def get_cache_info(path: Path) -> CacheInfo:
    """Return the information used to check if a file is already formatted or not."""
    stat = path.stat()
    return stat.st_mtime, stat.st_size


def filter_cached(
    cache: Cache, sources: Iterable[Path]
) -> tuple[set[Path], set[Path]]:
    """Split an iterable of paths in `sources` into two sets.

    The first contains paths of files that modified on disk or are not in the
    cache. The other contains paths to non-modified files.
    """
    todo, done = set(), set()
    for src in sources:
        res_src = src.resolve()
        if cache.get(str(res_src)) != get_cache_info(res_src):
            todo.add(src)
        else:
            done.add(src)
    return todo, done


def write_cache(cache: Cache, sources: Iterable[Path], mode: Mode) -> None:
    """Update the cache file."""
    cache_file = get_cache_file(mode)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        new_cache = {
            **cache,
            **{str(src.resolve()): get_cache_info(src) for src in sources},
        }
        with tempfile.NamedTemporaryFile(
            dir=str(cache_file.parent), delete=False
        ) as f:
            pickle.dump(new_cache, f, protocol=4)
        os.replace(f.name, cache_file)
    except OSError:
        pass
