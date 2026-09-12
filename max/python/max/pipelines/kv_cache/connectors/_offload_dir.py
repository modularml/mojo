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

"""Lifetime ownership of the tiered connector's disk offload directory.

The disk tier is ephemeral scratch -- it starts empty every run and never
warm-starts from a directory a previous process left -- so its directory is
MAX's to delete. Deleting it from a shutdown hook is not enough on its own:
the model worker is torn down with ``SIGTERM``, which Python does not handle,
so its ``AsyncExitStack`` never unwinds (see
:py:func:`max.serve.process_control.run_subprocess`), and a SIGKILL or
OOM-kill can land at any time regardless.

So ownership is published to the filesystem rather than tracked only in
process memory: an auto-created directory holds an exclusive ``flock`` on a
lock file for as long as its owning process lives. The kernel drops that lock
however the process dies, which turns "is anyone still using this directory?"
into a question the next startup can answer -- and so reclaim what it finds.
"""

from __future__ import annotations

import fcntl
import logging
import shutil
import tempfile
from pathlib import Path
from typing import IO

logger = logging.getLogger("max.pipelines")

# Prefix for auto-created tiered-connector disk offload directories. Owned by
# this module, which is the only thing that creates, claims, and deletes them.
KV_OFFLOAD_DIR_PREFIX = "max_kv_tiered_"

# Lock file naming the live owner of an auto-created offload directory.
_OWNER_LOCK_NAME = ".max_kv_owner.lock"


class OffloadDirectory:
    """The disk offload directory this process uses, and its cleanup.

    Construct one with :py:func:`acquire_offload_dir`. Hold it for as long as
    the disk tier is in use and call :py:meth:`release` once the tier has
    stopped writing.
    """

    def __init__(self, path: str, owner_lock: IO[bytes] | None) -> None:
        """Binds ``path`` to the ownership lock held over it, if any.

        Args:
            path: The offload directory, which already exists.
            owner_lock: The flocked lock file naming this process the
                directory's live owner, or ``None`` for a directory a later run
                cannot reclaim (see :py:func:`acquire_offload_dir`). Held here
                only to keep the lock: closing the file releases it.
        """
        self._path = path
        self._owner_lock = owner_lock

    @property
    def path(self) -> str:
        """The directory the disk tier reads and writes."""
        return self._path

    def release(self) -> None:
        """Deletes the directory and drops the ownership lock."""
        try:
            shutil.rmtree(self._path)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(
                "Failed to remove KV cache offload directory %s: %s",
                self._path,
                error,
            )
        if self._owner_lock is not None:
            self._owner_lock.close()
            self._owner_lock = None


def acquire_offload_dir(configured_dir: str | None) -> OffloadDirectory:
    """Returns the offload directory to use, auto-creating one if unset.

    A single connector serves every DP replica, so the directory is acquired
    once per process, not per replica.

    An auto-created directory is claimed with an ownership lock and so can be
    reclaimed by a later run if this process dies without releasing it. A
    configured one is not: it has no naming convention a later run could
    recognize it by, and its path may be shared with things that are not MAX.

    Args:
        configured_dir: ``kv_connector_config.disk_offload_dir``. ``None``
            auto-creates a directory under the system temp dir.
    """
    if configured_dir is not None:
        # A configured dir need not exist yet, and callers stat it for free
        # space, so create it before handing it over.
        Path(configured_dir).mkdir(parents=True, exist_ok=True)
        _reap_abandoned_offload_dirs(configured_dir)
        return OffloadDirectory(configured_dir, owner_lock=None)

    path = tempfile.mkdtemp(prefix=KV_OFFLOAD_DIR_PREFIX)
    logger.info("Tiered connector: auto-created disk offload dir %s", path)
    owner_lock = _claim(path)
    _reap_abandoned_offload_dirs(path)
    return OffloadDirectory(path, owner_lock)


def _claim(directory: str) -> IO[bytes] | None:
    """Takes the ownership lock on ``directory``, or ``None`` if it can't.

    Failing to claim only costs a later run the ability to reclaim this
    directory automatically, so it is a warning rather than a startup failure.
    """
    owner_lock = None
    try:
        owner_lock = open(Path(directory) / _OWNER_LOCK_NAME, "a+b")
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if owner_lock is not None:
            owner_lock.close()
        logger.warning(
            "Could not claim KV cache offload directory %s (%s); a forceful "
            "shutdown will leave it behind for an operator to delete.",
            directory,
            error,
        )
        return None
    return owner_lock


def _reap_abandoned_offload_dirs(active_dir: str) -> None:
    """Deletes auto-created offload directories no live process still owns.

    Args:
        active_dir: The directory this run uses. Its siblings matching
            ``{KV_OFFLOAD_DIR_PREFIX}*`` are the reap candidates.
    """
    parent = Path(active_dir).parent
    try:
        candidates = sorted(
            p
            for p in parent.glob(f"{KV_OFFLOAD_DIR_PREFIX}*")
            if p.is_dir() and str(p) != active_dir
        )
    except OSError as error:
        logger.warning(
            "Could not scan %s for abandoned KV cache offload directories: %s",
            parent,
            error,
        )
        return

    reaped: list[str] = []
    unclaimed: list[str] = []
    for candidate in candidates:
        if not (candidate / _OWNER_LOCK_NAME).is_file():
            unclaimed.append(str(candidate))
        elif _delete_if_unowned(candidate):
            reaped.append(str(candidate))

    if reaped:
        logger.info(
            "Reclaimed %d abandoned KV cache offload director%s in %s:\n  %s",
            len(reaped),
            "y" if len(reaped) == 1 else "ies",
            parent,
            "\n  ".join(reaped),
        )
    if unclaimed:
        logger.warning(
            "Found %d KV cache offload director%s in %s with no ownership "
            "lock:\n  %s\n"
            "These predate MAX Serve's automatic cleanup, or were not created "
            "by MAX Serve. If no process is currently using them, delete them "
            "to reclaim disk space.",
            len(unclaimed),
            "y" if len(unclaimed) == 1 else "ies",
            parent,
            "\n  ".join(unclaimed),
        )


def _delete_if_unowned(directory: Path) -> bool:
    """Deletes ``directory`` unless a live process holds its ownership lock.

    Taking the lock is what proves the owner is gone: the kernel drops it when
    that process exits, however it exits, and refuses it to us while it lives.
    """
    lock_path = directory / _OWNER_LOCK_NAME
    try:
        owner_lock = open(lock_path, "r+b")
    except FileNotFoundError:
        # The caller saw the file a moment ago, so another reaper got here
        # first and has already removed the directory.
        logger.debug("%s already reclaimed by another process", directory)
        return False
    except OSError as error:
        logger.warning(
            "Cannot check ownership of KV cache offload directory %s: %s",
            directory,
            error,
        )
        return False
    with owner_lock:
        try:
            fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        try:
            shutil.rmtree(directory)
        except OSError as error:
            logger.warning(
                "Failed to reclaim abandoned KV cache offload directory %s: %s",
                directory,
                error,
            )
            return False
    return True
