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

"""CPU unit tests for the tiered connector's host block row width.

``host_bytes_per_page`` sizes one jenga leaf's shared pinned host row, so
getting it wrong truncates a stored page or wastes host memory. Each layout is
pinned to a hand-computed byte count.

These run on CPU because the only GPU coverage
(``test_mla_offload_roundtrip_distributed_gpu``) needs 4 GPUs and is skipped on
every 1- and 2-GPU platform, including CI's.
"""

from __future__ import annotations

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheBuffer, KVCacheMemory

_PAGES = 8


def _unit(
    bytes_per_page: int, *, shards: int, replicated: bool = False
) -> KVCacheMemory:
    """One KV memory unit of ``shards`` uint8 page views."""
    return KVCacheMemory(
        replicated=replicated,
        buffers=[
            Buffer(
                shape=(_PAGES, bytes_per_page),
                dtype=DType.uint8,
                device=CPU(),
            )
            for _ in range(shards)
        ],
    )


def test_tp1_values_only() -> None:
    """A single unquantized shard contributes its stride once."""
    assert _unit(256, shards=1).host_bytes_per_page == 256


def test_values_and_scales_size_their_own_leaf() -> None:
    """Quantized TP==1: values and scales are distinct jenga leaves.

    ``KVCacheParams.leaves`` gives a quantized cache's scales their own leaf
    id, so each unit sizes its own host row rather than sharing one.
    """
    assert _unit(256, shards=1).host_bytes_per_page == 256
    assert _unit(16, shards=1).host_bytes_per_page == 16


def test_sharded_tp2_stores_every_shard() -> None:
    """MHA TP==2: each shard holds distinct bytes, so all of them are stored."""
    assert _unit(256, shards=2).host_bytes_per_page == 2 * 256
    assert _unit(16, shards=2).host_bytes_per_page == 2 * 16


def test_replicated_tp2_stores_one_shard_per_unit() -> None:
    """MLA TP==2: shards are byte-identical, so a unit is stored once.

    Counting the broadcast peers would double the host allocation for every MLA
    deployment.
    """
    assert _unit(256, shards=2, replicated=True).host_bytes_per_page == 256
    assert _unit(16, shards=2, replicated=True).host_bytes_per_page == 16


def test_row_width_is_shard_count_times_stride_for_sharded() -> None:
    """Widening the TP degree of a sharded unit widens the row linearly."""
    widths = [_unit(128, shards=tp).host_bytes_per_page for tp in (1, 2, 4, 8)]
    assert widths == [128, 256, 512, 1024]


def test_replicated_row_width_is_flat_in_tp_degree() -> None:
    """Widening the TP degree of a replicated unit does not widen the row."""
    widths = [
        _unit(128, shards=tp, replicated=True).host_bytes_per_page
        for tp in (2, 4, 8)
    ]
    assert widths == [128, 128, 128]


@pytest.mark.parametrize("replicated", [False, True])
def test_matches_kv_cache_buffer_to_memory(replicated: bool) -> None:
    """The row width agrees with what the producer actually authors.

    A change to ``to_memory``'s unit shape would otherwise shift the host row
    without failing anything.
    """
    tp = 2
    kv_buffer = KVCacheBuffer(
        replicates_kv_across_tp=replicated,
        # bfloat16 values fold to 2 bytes per element in the uint8 page view.
        values=[
            Buffer(shape=(_PAGES, 64), dtype=DType.bfloat16, device=CPU())
            for _ in range(tp)
        ],
        scales=[
            Buffer(shape=(_PAGES, 4), dtype=DType.float32, device=CPU())
            for _ in range(tp)
        ],
    )

    values_bytes, scales_bytes = 64 * 2, 4 * 4
    stored_per_unit = 1 if replicated else tp
    values_mem, scales_mem = kv_buffer.to_memory()
    assert values_mem.host_bytes_per_page == stored_per_unit * values_bytes
    assert scales_mem.host_bytes_per_page == stored_per_unit * scales_bytes
