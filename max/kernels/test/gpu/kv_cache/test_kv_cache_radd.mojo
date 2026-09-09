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

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from nn.kv_cache_ragged import generic_kv_cache_radd_dispatch

from std.utils import Index, IndexList

from kv_cache_test_utils import PagedLookupTable


def test_kv_cache_radd[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    ctx: DeviceContext,
) raises:
    comptime num_layers = 2
    assert (
        num_active_loras <= batch_size
    ), "num_active_loras must be less than or equal to batch_size"
    var cache_lengths = ManagedLayoutTensor[.uint32, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
        ctx,
    )
    var input_row_offsets_slice = ManagedLayoutTensor[
        .uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            Index(num_active_loras + 1)
        ),
        ctx,
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    var input_row_offsets_slice_host = input_row_offsets_slice.tensor[
        update=False
    ]()
    var num_active_loras_slice_start = batch_size - num_active_loras
    var running_total = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i >= num_active_loras_slice_start:
            input_row_offsets_slice_host[
                i - num_active_loras_slice_start
            ] = UInt32(running_total)
            total_slice_length += prompt_lens[i]

        running_total += prompt_lens[i]

    input_row_offsets_slice_host[num_active_loras] = UInt32(running_total)

    var num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        num_heads,
        head_dim,
    )
    var kv_block_paged = ManagedLayoutTensor[dtype, Layout.row_major[6]()](
        RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_paged_shape),
        ctx,
    )
    var kv_block_paged_host = kv_block_paged.tensor[update=False]()
    for i in range(kv_block_paged_shape.flattened_length()):
        kv_block_paged_host.ptr[i] = Scalar[dtype](1)

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_lens, max_full_context_length, num_paged_blocks, ctx
    )

    var kv_collection_device = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        page_size,
    ](
        kv_block_paged.device_tensor(),
        cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var a_shape = IndexList[2](total_slice_length, num_heads * head_dim * 2)
    var a = ManagedLayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim * 2)
    ](
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim * 2)
        ].row_major(a_shape),
        ctx,
    )
    var a_host = a.tensor[update=False]()
    for i in range(a_shape.flattened_length()):
        a_host.ptr[i] = Scalar[dtype](i)

    var layer_idx = 1
    generic_kv_cache_radd_dispatch[target="gpu"](
        LayoutTensor[
            dtype,
            Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim * 2),
            MutAnyOrigin,
        ](a.device_tensor().ptr, a.device_tensor().runtime_layout),
        kv_collection_device,
        input_row_offsets_slice.device_tensor(),
        UInt32(num_active_loras_slice_start),
        UInt32(layer_idx),
        ctx,
    )
    ctx.synchronize()
    kv_block_paged_host = kv_block_paged.tensor()
    a_host = a.tensor()

    var kv_collection_host = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        page_size,
    ](
        kv_block_paged_host,
        cache_lengths.tensor(),
        paged_lut.host_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # first check that we didn't augment previous cache entries
    for i in range(batch_size):
        for c in range(cache_lens[i]):
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, c, d)
                    var v_val = v_cache_host.load[width=1](i, h, c, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )

    # now check that we augmented the correct entries
    # the first elements in the batch should not be lora-augmented
    for i in range(batch_size - num_active_loras):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )

    # now check that the lora-augmented entries are correct
    var arange_counter = 0
    for i in range(batch_size - num_active_loras, batch_size):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var expected_k_val = 1 + arange_counter
                    if k_val != Scalar[dtype](expected_k_val):
                        raise Error(
                            "Mismatch in output for k, expected "
                            + String(expected_k_val)
                            + ", got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    arange_counter += 1
            for h in range(num_heads):
                for d in range(head_dim):
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    var expected_v_val = 1 + arange_counter
                    if v_val != Scalar[dtype](expected_v_val):
                        raise Error(
                            "Mismatch in output for v, expected "
                            + String(expected_v_val)
                            + ", got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    arange_counter += 1

    # Keep helper ownership explicit until helper internals are migrated.
    _ = paged_lut^


def main() raises:
    with DeviceContext() as ctx:
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[1](10),
            IndexList[1](40),
            1,
            ctx,
        )
