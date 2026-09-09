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

import pytest
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheBuffer, KVCacheMemory


def _buffer(
    num_pages: int,
    bytes_per_page: int,
    dtype: DType = DType.uint8,
    device: Device | None = None,
) -> Buffer:
    """Allocate a 2-D buffer on CPU for memory-layout tests."""
    return Buffer(
        shape=(num_pages, bytes_per_page),
        dtype=dtype,
        device=device if device is not None else CPU(),
    )


def _uint8_buffer(num_pages: int, bytes_per_page: int) -> Buffer:
    """Allocate a 2-D ``uint8`` buffer on CPU for memory-layout tests."""
    return _buffer(num_pages, bytes_per_page)


# ==================== KVCacheMemory Tests ====================


def test_kv_cache_memory_valid_buffers() -> None:
    """2-D uint8 shards of a matching shape pass validation."""
    memory = KVCacheMemory(
        replicated=False,
        buffers=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
    )
    assert memory.total_num_pages == 4
    assert memory.bytes_per_page == 256


def test_kv_cache_memory_empty_buffers_fails() -> None:
    """A unit with no shards raises ValueError."""
    with pytest.raises(
        ValueError, match="KVCacheMemory must have at least one buffer"
    ):
        KVCacheMemory(replicated=False, buffers=[])


def test_kv_cache_memory_non_2d_buffer_fails() -> None:
    """A buffer that is not 2-D raises ValueError."""
    buffer = Buffer(shape=(4,), dtype=DType.uint8, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have 2 dimensions"
    ):
        KVCacheMemory(replicated=False, buffers=[buffer])


def test_kv_cache_memory_non_uint8_buffer_fails() -> None:
    """A buffer with a non-uint8 dtype raises ValueError."""
    buffer = Buffer(shape=(4, 256), dtype=DType.bfloat16, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have dtype uint8"
    ):
        KVCacheMemory(replicated=False, buffers=[buffer])


def test_kv_cache_memory_mismatched_shard_shape_fails() -> None:
    """Shards that disagree on shape raise ValueError.

    ``bytes_per_page`` / ``total_num_pages`` are read off shard 0, so a
    mismatched shard would silently report the wrong stride.
    """
    with pytest.raises(ValueError, match="must share a shape"):
        KVCacheMemory(
            replicated=False,
            buffers=[_uint8_buffer(4, 256), _uint8_buffer(4, 512)],
        )


def test_kv_cache_memory_replicated_single_shard_fails() -> None:
    """replicated=True with a single shard raises ValueError."""
    with pytest.raises(
        ValueError, match="replicated=True requires at least 2 TP-shard buffers"
    ):
        KVCacheMemory(replicated=True, buffers=[_uint8_buffer(4, 256)])


def test_kv_cache_memory_buffers_are_shard_ordered() -> None:
    """``buffers[s]`` is shard ``s`` whether or not the unit is replicated.

    Consumers index by shard rather than branching on ``replicated`` to find a
    shard, so this correspondence is the type's core contract.
    """
    shards = [_uint8_buffer(4, 256) for _ in range(3)]

    for replicated in (False, True):
        memory = KVCacheMemory(replicated=replicated, buffers=list(shards))
        assert memory.buffers == shards
        assert memory.replicated is replicated


# ==================== KVCacheBuffer Tests ====================


def test_kv_cache_buffer_empty_values_fails() -> None:
    """An empty values list raises ValueError before the TP-shard check."""
    with pytest.raises(ValueError, match="List of values must be non-empty"):
        KVCacheBuffer(
            values=[],
            replicates_kv_across_tp=True,
        )


def test_kv_cache_buffer_replicated_single_shard_fails() -> None:
    """replicates_kv_across_tp=True with a single shard raises ValueError."""
    with pytest.raises(ValueError, match="requires at least 2 TP shards"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256)],
            replicates_kv_across_tp=True,
        )


def test_kv_cache_buffer_replicated_multi_shard_marks_replicated() -> None:
    """replicates_kv_across_tp=True yields one replicated unit of all shards."""
    kv_buffer = KVCacheBuffer(
        values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
        replicates_kv_across_tp=True,
    )
    memory = kv_buffer.to_memory()
    assert len(memory) == 1
    assert memory[0].replicated
    assert len(memory[0].buffers) == 2


def test_kv_cache_buffer_sharded_single_shard_valid() -> None:
    """A non-replicated single-shard buffer validates and exposes its pages."""
    kv_buffer = KVCacheBuffer(
        values=[_uint8_buffer(4, 256)],
        replicates_kv_across_tp=False,
    )
    assert kv_buffer.total_num_pages == 4
    assert kv_buffer.all_buffers == kv_buffer.values


def test_kv_cache_buffer_mismatched_value_dtypes_fails() -> None:
    """Values that disagree on dtype raise ValueError."""
    with pytest.raises(ValueError, match="All values must have the same dtype"):
        KVCacheBuffer(
            values=[
                _buffer(4, 256, dtype=DType.uint8),
                _buffer(4, 256, dtype=DType.bfloat16),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_value_shapes_fails() -> None:
    """Values that disagree on shape raise ValueError."""
    with pytest.raises(ValueError, match="All values must have the same shape"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 512)],
            replicates_kv_across_tp=False,
        )


# ==================== KVCacheBuffer + scales Tests ====================


def test_kv_cache_buffer_with_scales_valid() -> None:
    """Values and scales of equal length and page count validate."""
    kv_buffer = KVCacheBuffer(
        values=[_buffer(4, 256, dtype=DType.uint8)],
        scales=[_buffer(4, 8, dtype=DType.float32)],
        replicates_kv_across_tp=False,
    )
    assert kv_buffer.scales is not None
    assert kv_buffer.total_num_pages == 4
    assert kv_buffer.all_buffers == [*kv_buffer.values, *kv_buffer.scales]


def test_kv_cache_buffer_with_scales_to_memory() -> None:
    """to_memory emits one unit per kind, each carrying every TP shard."""
    kv_buffer = KVCacheBuffer(
        values=[_buffer(4, 256, dtype=DType.uint8) for _ in range(2)],
        scales=[_buffer(4, 8, dtype=DType.float32) for _ in range(2)],
        replicates_kv_across_tp=False,
    )
    memory = kv_buffer.to_memory()

    # values before scales, each with both TP shards folded into uint8 pages.
    assert len(memory) == 2
    assert not any(m.replicated for m in memory)
    assert [len(m.buffers) for m in memory] == [2, 2]
    assert [m.bytes_per_page for m in memory] == [256, 8 * 4]


def test_kv_cache_buffer_scales_length_mismatch_fails() -> None:
    """A scales list shorter than values raises ValueError."""
    with pytest.raises(
        ValueError, match="Scales must be the same length as values"
    ):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[_buffer(4, 8, dtype=DType.float32)],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_scale_dtypes_fails() -> None:
    """Scales that disagree on dtype raise ValueError."""
    with pytest.raises(ValueError, match="All scales must have the same dtype"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[
                _buffer(4, 8, dtype=DType.float32),
                _buffer(4, 8, dtype=DType.bfloat16),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_scale_shapes_fails() -> None:
    """Scales that disagree on shape raise ValueError."""
    with pytest.raises(ValueError, match="All scales must have the same shape"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[
                _buffer(4, 8, dtype=DType.float32),
                _buffer(4, 16, dtype=DType.float32),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_value_scale_page_count_mismatch_fails() -> None:
    """Values and scales with differing page counts raise ValueError."""
    with pytest.raises(
        ValueError, match="Values and scales must have the same number of pages"
    ):
        KVCacheBuffer(
            values=[_buffer(4, 256, dtype=DType.uint8)],
            scales=[_buffer(8, 8, dtype=DType.float32)],
            replicates_kv_across_tp=False,
        )
