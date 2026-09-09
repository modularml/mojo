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

from std.math import ceildiv
from std.random import seed

from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import *
from layout._utils import ManagedLayoutTensor, UNKNOWN_VALUE
from std.memory import unsafe_memset_zero
from nn.kv_cache_ragged import kv_cache_store_padded, kv_cache_store_ragged
from std.testing import assert_almost_equal

from std.utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime kv_params_test = KVCacheStaticParams(num_heads=4, head_size=64)
comptime dtype = DType.float32


def test_kv_cache_store_ragged_basic(ctx: DeviceContext) raises:
    comptime dtype = DType.float32
    comptime page_size = 128
    comptime num_kv_heads = 2
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=64
    )
    comptime num_layers = 2
    comptime batch_size = 3
    var valid_lengths: List[Int] = [100, 200, 300]
    var cache_lengths: List[Int] = [100, 200, 300]

    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    var cache_lengths_table = CacheLengthsTable.build(
        valid_lengths, cache_lengths, ctx
    )

    var total_length = cache_lengths_table.total_length
    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_managed = ManagedLayoutTensor[dtype, kv_block_layout](
        kv_block_runtime_layout, ctx
    )
    var kv_block_tensor = kv_block_managed.tensor()

    var paged_lut = PagedLookupTable[page_size].build(
        valid_lengths,
        cache_lengths,
        max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    var kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_managed.device_tensor(),
        cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var q_shape = IndexList[3](total_length, num_kv_heads, kv_params.head_size)
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, num_kv_heads, kv_params.head_size
    )
    var q_runtime_layout = RuntimeLayout[q_layout].row_major(q_shape)
    var q_managed = ManagedLayoutTensor[dtype, q_layout](q_runtime_layout, ctx)
    var q_tensor = q_managed.tensor()

    # Fill input data for testing
    var current_offset = 0
    for batch_idx in range(batch_size):
        var seq_len = valid_lengths[batch_idx]
        for token_idx in range(seq_len):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    # Calculate expected value
                    var global_token_idx = current_offset + token_idx
                    var expected_linear_idx = (
                        global_token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    q_tensor[
                        global_token_idx, head_idx, head_dim_idx
                    ] = Float32(expected_linear_idx)
        current_offset += seq_len

    var q_device_tensor = q_managed.device_tensor()

    @__parameter
    @always_inline
    @__copy_capture(q_device_tensor)
    def input_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3]) -> SIMD[dtype, width]:
        return q_device_tensor.load[width](idx)

    var k_cache_device = kv_collection_paged_device.get_key_cache(0)
    kv_cache_store_ragged[input_fn=input_fn, target="gpu"](
        k_cache_device,
        q_shape,
        cache_lengths_table.input_row_offsets.device_tensor(),
        ctx,
    )
    ctx.synchronize()

    var kv_collection_paged_host = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_managed.tensor(),
        cache_lengths_table.cache_lengths.host_tensor(),
        paged_lut.host_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )
    var k_cache_host = kv_collection_paged_host.get_key_cache(0)

    # Verify the data was stored correctly
    current_offset = 0
    for batch_idx in range(batch_size):
        var seq_len = valid_lengths[batch_idx]
        for token_idx in range(seq_len):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    # Calculate expected value
                    var global_token_idx = current_offset + token_idx
                    var expected_linear_idx = (
                        global_token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    var expected_value = Float32(expected_linear_idx)

                    # Get actual value from cache
                    var cache_token_idx = token_idx + cache_lengths[batch_idx]
                    var actual_value = k_cache_host.load[width=1](
                        batch_idx,
                        head_idx,
                        cache_token_idx,
                        head_dim_idx,
                    )
                    # Verify the values match
                    assert_almost_equal(
                        actual_value,
                        expected_value,
                        rtol=1e-5,
                        atol=1e-6,
                        msg="Mismatch at batch="
                        + String(batch_idx)
                        + ", token="
                        + String(token_idx)
                        + ", head="
                        + String(head_idx)
                        + ", head_dim="
                        + String(head_dim_idx)
                        + ": expected "
                        + String(expected_value)
                        + ", got "
                        + String(actual_value),
                    )
        current_offset += seq_len

    _ = cache_lengths_table^
    _ = paged_lut^


def test_kv_cache_store_padded_basic(ctx: DeviceContext) raises:
    comptime dtype = DType.float32
    comptime page_size = 128
    comptime num_kv_heads = 2
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=64
    )
    comptime num_layers = 2
    comptime batch_size = 3
    var valid_lengths: List[Int] = [3, 1, 4]
    var cache_lengths: List[Int] = [5, 2, 0]

    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    var cache_lengths_table = CacheLengthsTable.build(
        valid_lengths, cache_lengths, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_managed = ManagedLayoutTensor[dtype, kv_block_layout](
        kv_block_runtime_layout, ctx
    )
    var kv_block_tensor = kv_block_managed.tensor()
    unsafe_memset_zero(kv_block_tensor.ptr, kv_block_shape.flattened_length())

    var paged_lut = PagedLookupTable[page_size].build(
        valid_lengths,
        cache_lengths,
        max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    var kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_managed.device_tensor().ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                kv_block_managed.device_tensor().runtime_layout.shape.value
            ),
        ),
        cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var q_shape = IndexList[4](
        batch_size,
        max_seq_length_batch,
        num_kv_heads,
        kv_params.head_size,
    )
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_kv_heads, kv_params.head_size
    )
    var q_runtime_layout = RuntimeLayout[q_layout].row_major(q_shape)
    var q_managed = ManagedLayoutTensor[dtype, q_layout](q_runtime_layout, ctx)
    var q_tensor = q_managed.tensor()

    for batch_idx in range(batch_size):
        for token_idx in range(max_seq_length_batch):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    var expected_linear_idx = (
                        batch_idx
                        * max_seq_length_batch
                        * num_kv_heads
                        * kv_params.head_size
                        + token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    q_tensor[
                        batch_idx, token_idx, head_idx, head_dim_idx
                    ] = Float32(expected_linear_idx)

    var valid_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with valid_lengths_device.map_to_host() as valid_lengths_host:
        for i in range(batch_size):
            valid_lengths_host[i] = UInt32(valid_lengths[i])

    var valid_lengths_tensor = LayoutTensor[
        .uint32,
        Layout.row_major(UNKNOWN_VALUE),
    ](
        valid_lengths_device,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
    )

    var q_device_tensor = q_managed.device_tensor()

    @__parameter
    @always_inline
    @__copy_capture(q_device_tensor)
    def input_fn[
        width: Int, alignment: Int
    ](idx: IndexList[4]) -> SIMD[dtype, width]:
        return q_device_tensor.load[width](idx)

    var k_cache_device = kv_collection_paged_device.get_key_cache(0)
    kv_cache_store_padded[input_fn=input_fn, target="gpu"](
        k_cache_device, q_shape, valid_lengths_tensor, ctx
    )
    ctx.synchronize()

    var kv_collection_paged_host = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_managed.tensor().ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                kv_block_managed.tensor().runtime_layout.shape.value
            ),
        ),
        cache_lengths_table.cache_lengths.host_tensor(),
        paged_lut.host_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )
    var k_cache_host = kv_collection_paged_host.get_key_cache(0)

    for batch_idx in range(batch_size):
        var seq_len = valid_lengths[batch_idx]
        for token_idx in range(seq_len):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    var expected_linear_idx = (
                        batch_idx
                        * max_seq_length_batch
                        * num_kv_heads
                        * kv_params.head_size
                        + token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    var expected_value = Float32(expected_linear_idx)

                    var cache_token_idx = token_idx + cache_lengths[batch_idx]
                    var actual_value = k_cache_host.load[width=1](
                        batch_idx,
                        head_idx,
                        cache_token_idx,
                        head_dim_idx,
                    )

                    assert_almost_equal(
                        actual_value,
                        expected_value,
                        rtol=1e-5,
                        atol=1e-6,
                        msg="Mismatch at batch="
                        + String(batch_idx)
                        + ", token="
                        + String(token_idx)
                        + ", head="
                        + String(head_idx)
                        + ", head_dim="
                        + String(head_dim_idx)
                        + ": expected "
                        + String(expected_value)
                        + ", got "
                        + String(actual_value),
                    )

        for token_idx in range(seq_len, max_seq_length_batch):
            for head_idx in range(num_kv_heads):
                var cache_token_idx = token_idx + cache_lengths[batch_idx]
                var actual_value = k_cache_host.load[width=1](
                    batch_idx,
                    head_idx,
                    cache_token_idx,
                    0,
                )
                assert_almost_equal(
                    actual_value,
                    Float32(0.0),
                    rtol=1e-5,
                    atol=1e-6,
                    msg="Unexpected write at batch="
                    + String(batch_idx)
                    + ", token="
                    + String(token_idx)
                    + ", head="
                    + String(head_idx),
                )

    _ = cache_lengths_table^
    _ = paged_lut^


def main() raises:
    seed(42)  # Set seed for reproducible tests

    with DeviceContext() as ctx:
        test_kv_cache_store_ragged_basic(ctx)
        test_kv_cache_store_padded_basic(ctx)
