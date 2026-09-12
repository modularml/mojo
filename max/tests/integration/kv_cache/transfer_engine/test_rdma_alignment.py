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

"""Tests the AMD RDMA start-address check the transfer engine applies.

The check reads only ``Device.api``, so these run on CPU-only workers against
a stand-in device rather than needing an AMD host.
"""

from __future__ import annotations

from typing import cast

import pytest
from max.driver import Device
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    _AMD_RDMA_ALIGNED_SIZE_FACTOR,
    _AMD_RDMA_START_ALIGNMENT,
    _check_rdma_start_alignment,
)

_ALIGNED_SIZE = _AMD_RDMA_START_ALIGNMENT * _AMD_RDMA_ALIGNED_SIZE_FACTOR


class _StubDevice:
    def __init__(self, api: str) -> None:
        self.api = api


def _device(api: str) -> Device:
    return cast(Device, _StubDevice(api))


def test_amd_rejects_misaligned_start() -> None:
    base_addr = _AMD_RDMA_START_ALIGNMENT + 256
    with pytest.raises(ValueError, match=hex(base_addr)):
        _check_rdma_start_alignment(
            base_addr, _ALIGNED_SIZE, _device("hip"), "agent-0", 0
        )


def test_amd_accepts_aligned_start() -> None:
    _check_rdma_start_alignment(
        4 * _AMD_RDMA_START_ALIGNMENT,
        _ALIGNED_SIZE,
        _device("hip"),
        "agent-0",
        0,
    )


def test_amd_start_below_the_aligned_size_is_unconstrained() -> None:
    """A region too small to have earned the coarse start is left alone.

    ``MemoryManager`` only aligns blocks at ``_ALIGNED_SIZE`` and above, so a
    small buffer is misaligned by design. Raising on one deadlocked
    ``test_send_recv_multi_thread_gpu``, whose peer threads rendezvous through
    an unbounded queue after building their engines.
    """
    _check_rdma_start_alignment(
        _AMD_RDMA_START_ALIGNMENT + 256,
        _ALIGNED_SIZE - 1,
        _device("hip"),
        "agent-0",
        0,
    )


@pytest.mark.parametrize("api", ["cuda", "metal", "cpu"])
def test_non_amd_start_is_unconstrained(api: str) -> None:
    _check_rdma_start_alignment(
        _AMD_RDMA_START_ALIGNMENT + 256,
        _ALIGNED_SIZE,
        _device(api),
        "agent-0",
        0,
    )
