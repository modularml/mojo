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

"""Unit tests for the KV cache host and disk capacity preflights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.pipelines.kv_cache.connectors import rust_tier_connector


def test_host_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rust_tier_connector.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1024),
    )

    with pytest.raises(RuntimeError, match="host_offload_max_gb"):
        rust_tier_connector._check_host_memory_capacity(2048)


def test_host_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rust_tier_connector.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    rust_tier_connector._check_host_memory_capacity(4096)


def test_host_capacity_skips_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _raise() -> None:
        raise OSError("host memory unavailable")

    monkeypatch.setattr(rust_tier_connector.psutil, "virtual_memory", _raise)

    rust_tier_connector._check_host_memory_capacity(1 << 60)
    assert "skipping KV cache host capacity preflight" in caplog.text


def test_disk_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rust_tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=1024),
    )

    with pytest.raises(RuntimeError, match="disk_offload_max_gb"):
        rust_tier_connector._check_disk_capacity("/tmp", 2048)


def test_disk_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rust_tier_connector.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=4096),
    )

    rust_tier_connector._check_disk_capacity("/tmp", 4096)
