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
"""Tests for KV cache offload directory ownership and startup reclamation."""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from max.pipelines.kv_cache.connectors._offload_dir import (
    _OWNER_LOCK_NAME,
    KV_OFFLOAD_DIR_PREFIX,
    _reap_abandoned_offload_dirs,
    acquire_offload_dir,
)


def _unclaimed_dir(parent: Path, suffix: str) -> Path:
    """An offload directory with no ownership lock, as older MAX left behind."""
    path = parent / f"{KV_OFFLOAD_DIR_PREFIX}{suffix}"
    path.mkdir()
    return path


def _abandoned_dir(parent: Path, suffix: str) -> Path:
    """An offload directory whose owning process is gone."""
    path = _unclaimed_dir(parent, suffix)
    (path / _OWNER_LOCK_NAME).touch()
    (path / "00").mkdir()
    return path


def test_abandoned_dirs_are_reclaimed(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Unowned offload dirs are deleted; the active one is left alone."""
    active = _abandoned_dir(tmp_path, "current")
    abandoned = _abandoned_dir(tmp_path, "old")

    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        _reap_abandoned_offload_dirs(str(active))

    assert not abandoned.exists()
    assert active.is_dir()
    assert str(abandoned) in caplog.text


def test_dirs_owned_by_a_live_process_are_kept(tmp_path: Path) -> None:
    """A directory whose ownership lock is still held is not touched.

    ``flock`` conflicts between open file descriptions rather than between
    processes, so holding the lock on a separate ``open`` of the same file
    reproduces a second live server without spawning one.
    """
    active = _abandoned_dir(tmp_path, "current")
    in_use = _abandoned_dir(tmp_path, "in_use")

    owner = open(in_use / _OWNER_LOCK_NAME, "a+b")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        _reap_abandoned_offload_dirs(str(active))
        assert in_use.is_dir()
    finally:
        owner.close()

    # With the lock dropped, the next run reclaims it.
    _reap_abandoned_offload_dirs(str(active))
    assert not in_use.exists()


def test_a_killed_process_leaves_a_reclaimable_dir(tmp_path: Path) -> None:
    """The bug this all exists for: SIGKILL still releases the ownership lock.

    No in-process cleanup hook survives a SIGKILL, so this is the only thing
    standing between a hard-killed server and a leaked offload directory.
    """
    killed = _abandoned_dir(tmp_path, "killed_run")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import fcntl, sys
                lock = open({str(killed / _OWNER_LOCK_NAME)!r}, "a+b")
                fcntl.flock(lock, fcntl.LOCK_EX)
                sys.stdout.write("locked")
                sys.stdout.flush()
                sys.stdin.read()
            """),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert holder.stdout is not None
    assert holder.stdout.read(len(b"locked")) == b"locked"
    holder.kill()
    holder.wait()

    _reap_abandoned_offload_dirs(str(_abandoned_dir(tmp_path, "next_run")))

    assert not killed.exists()


def test_unclaimed_dirs_are_warned_about_not_deleted(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Dirs with no ownership lock predate this cleanup, so only warn."""
    active = _abandoned_dir(tmp_path, "current")
    legacy = _unclaimed_dir(tmp_path, "legacy")

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        _reap_abandoned_offload_dirs(str(active))

    assert legacy.is_dir()
    assert str(legacy) in caplog.text


def test_reap_leaves_unrelated_directories_alone(tmp_path: Path) -> None:
    """Only directories matching the offload prefix are candidates."""
    active = _abandoned_dir(tmp_path, "current")
    unrelated = tmp_path / "someone_elses_scratch"
    unrelated.mkdir()
    (unrelated / _OWNER_LOCK_NAME).touch()

    _reap_abandoned_offload_dirs(str(active))

    assert unrelated.is_dir()
    assert os.listdir(unrelated) == [_OWNER_LOCK_NAME]


def test_missing_parent_is_silent(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A non-existent parent directory doesn't raise or warn."""
    missing = tmp_path / "does_not_exist" / f"{KV_OFFLOAD_DIR_PREFIX}x"

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        _reap_abandoned_offload_dirs(str(missing))

    assert not caplog.text


def test_acquire_creates_and_releases_a_claimed_dir() -> None:
    """An auto-created dir exists while held, carries a lock, and is deleted."""
    offload_dir = acquire_offload_dir(None)
    path = Path(offload_dir.path)
    try:
        assert path.is_dir()
        assert path.name.startswith(KV_OFFLOAD_DIR_PREFIX)
        assert (path / _OWNER_LOCK_NAME).is_file()
    finally:
        offload_dir.release()
    assert not path.exists()


def test_acquire_keeps_explicit_path(tmp_path: Path) -> None:
    """A configured dir is used as given and created if it doesn't exist yet."""
    explicit = tmp_path / "kv-offload"
    offload_dir = acquire_offload_dir(str(explicit))
    try:
        assert offload_dir.path == str(explicit)
        assert explicit.is_dir()
    finally:
        offload_dir.release()


def test_release_is_quiet_when_the_dir_is_already_gone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Something else clearing the temp dir first isn't an error."""
    offload_dir = acquire_offload_dir(None)
    shutil.rmtree(offload_dir.path)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        offload_dir.release()

    assert not caplog.text
