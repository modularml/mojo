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
"""Tests for the stale KV cache offload directory startup warning."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pytest
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.kv_cache.connectors.rust_tier_connector import (
    KV_OFFLOAD_DIR_PREFIX,
    _resolve_disk_offload_dir,
    warn_stale_offload_dirs,
)


def _make_dir(parent: str, suffix: str) -> str:
    path = os.path.join(parent, f"{KV_OFFLOAD_DIR_PREFIX}{suffix}")
    os.makedirs(path, exist_ok=True)
    return path


def test_warns_about_stale_offload_dirs(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Leftover offload-dir siblings are warned about, the active one isn't."""
    parent = str(tmp_path)
    current = _make_dir(parent, "current")
    stale1 = _make_dir(parent, "old1")
    stale2 = _make_dir(parent, "old2")

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        warn_stale_offload_dirs(current)

    text = caplog.text
    assert stale1 in text
    assert stale2 in text
    # The directory this run is actively using must not be flagged as stale.
    assert current not in text
    # The message explains graceful vs forceful shutdown cleanup.
    assert "forceful shutdown" in text.lower()


def test_no_warning_when_no_stale_dirs(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """With only the active dir present, nothing is warned."""
    current = _make_dir(str(tmp_path), "current")

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        warn_stale_offload_dirs(current)

    assert "leftover" not in caplog.text.lower()


def test_missing_parent_is_silent(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A non-existent parent directory doesn't raise or warn."""
    missing = os.path.join(
        str(tmp_path), "does_not_exist", f"{KV_OFFLOAD_DIR_PREFIX}x"
    )

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        warn_stale_offload_dirs(missing)

    assert "leftover" not in caplog.text.lower()


def test_resolve_disk_offload_dir_does_not_mutate_frozen_config() -> None:
    cfg = KVConnectorConfig()
    assert cfg.disk_offload_dir is None
    resolved = _resolve_disk_offload_dir(cfg)
    try:
        assert Path(resolved).is_dir()
        assert cfg.disk_offload_dir is None
    finally:
        shutil.rmtree(resolved, ignore_errors=True)


def test_resolve_disk_offload_dir_keeps_explicit_path(tmp_path: Path) -> None:
    explicit = str(tmp_path / "kv-offload")
    os.makedirs(explicit)
    cfg = KVConnectorConfig(disk_offload_dir=explicit)
    assert _resolve_disk_offload_dir(cfg) == explicit
    assert cfg.disk_offload_dir == explicit
