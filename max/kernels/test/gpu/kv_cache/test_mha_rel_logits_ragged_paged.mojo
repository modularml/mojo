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
"""Runs flash attention through `RelativeLogitsMask` over a paged
ragged KV cache and checks it against `mha_gpu_naive`, which calls the same
mask functor -- a cross-implementation check of the mask wired into the
production kernel path, not just the mask struct in isolation.
"""

from std.math import align_up, ceildiv, rsqrt
from std.random import seed

from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    RelativeLogitsMask,
    SlidingWindowCausalMask,
)
from nn.attention.mha_utils import as_dynamic_row_major_1d
from std.testing import assert_almost_equal

from std.utils import IndexList


def execute_rel_logits_flash_attention_test[
    V: MHAMask,
    //,
    visibility: V,
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    page_size: Int,
    extent: Int,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    label: String,
    ctx: DeviceContext,
    num_partitions: Optional[Int] = None,
) raises:
    comptime dtype = DType.bfloat16
    comptime group = num_q_heads // kv_params.num_heads
    comptime head_size = kv_params.head_size
    comptime num_layers = 1
    comptime layer_idx = 0

    var batch_size = len(valid_lengths)
    var cache_table = CacheLengthsTable.build(valid_lengths, cache_lengths, ctx)
    var total_length = cache_table.total_length
    var max_full_context_length = cache_table.max_full_context_length
    var max_valid_length = max(1, cache_table.max_seq_length_batch)

    # The low-level kernels address the full final 64-row output tile.
    var padded_total_length = align_up(total_length, 64)

    comptime tensor_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime kv_block_6d_layout = Layout.row_major[6]()
    comptime rel_logits_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, extent
    )

    var scale = rsqrt(Float32(head_size))

    seed(0x9F3A + label.byte_length())

    var tensor_runtime_layout = RuntimeLayout[tensor_layout].row_major(
        IndexList[3](padded_total_length, num_q_heads, head_size)
    )
    var q = ManagedLayoutTensor[dtype, tensor_layout](
        tensor_runtime_layout, ctx
    )
    random(q.tensor[update=False]())
    var q_device = q.device_tensor().as_imm()

    var rel_logits_runtime_layout = RuntimeLayout[rel_logits_layout].row_major(
        IndexList[3](total_length, num_q_heads, extent)
    )
    var rel_logits = ManagedLayoutTensor[dtype, rel_logits_layout](
        rel_logits_runtime_layout, ctx
    )
    random(rel_logits.tensor[update=False]())
    var rel_logits_device = rel_logits.device_tensor().as_imm()

    var mask = RelativeLogitsMask[visibility](
        rel_logits_device,
        as_dynamic_row_major_1d(cache_table.cache_lengths.device_tensor()),
        as_dynamic_row_major_1d(cache_table.input_row_offsets.device_tensor()),
    )

    var num_paged_blocks = 0
    for i in range(batch_size):
        num_paged_blocks += ceildiv(
            cache_lengths[i] + valid_lengths[i], page_size
        )
    num_paged_blocks += 4

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_size,
    )
    var kv_blocks = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
        ctx,
    )
    random(kv_blocks.tensor[update=False]())
    var kv_blocks_device = kv_blocks.device_tensor()

    var paged_lut = PagedLookupTable[page_size].build(
        valid_lengths,
        cache_lengths,
        max_full_context_length,
        num_paged_blocks,
        ctx,
    )

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_blocks_device.as_unsafe_any_origin(),
        cache_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_valid_length),
        UInt32(max_full_context_length),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    var test_output = ManagedLayoutTensor[dtype, tensor_layout](
        tensor_runtime_layout, ctx
    )
    var test_output_device = test_output.device_tensor[update=False]()
    flash_attention[ragged=True](
        test_output_device,
        q_device,
        k_cache,
        v_cache,
        mask,
        cache_table.input_row_offsets.device_tensor(),
        scale,
        ctx,
        num_partitions=num_partitions,
    )

    var reference_output = ManagedLayoutTensor[dtype, tensor_layout](
        tensor_runtime_layout, ctx
    )
    var reference_output_device = reference_output.device_tensor[update=False]()
    mha_gpu_naive[ragged=True](
        q_device,
        k_cache,
        v_cache,
        mask,
        reference_output_device,
        cache_table.input_row_offsets.device_tensor(),
        scale,
        batch_size,
        max_valid_length,
        max_full_context_length,
        num_q_heads,
        head_size,
        group,
        ctx,
    )

    var actual = test_output.tensor()
    var expected = reference_output.tensor()
    for i in range(total_length * num_q_heads * head_size):
        assert_almost_equal(actual.ptr[i], expected.ptr[i], atol=0.06)

    print("PASSED [", label, "]", sep="")


def main() raises:
    with DeviceContext() as ctx:
        # Prefill, global, GQA kv_heads=8, ragged batch of 2 unequal
        # sequences, extent < the longest sequence.
        execute_rel_logits_flash_attention_test[
            CausalMask(),
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=8, head_size=64),
            page_size=128,
            extent=8,
        ]([5, 9], [0, 0], "prefill+global+kv8", ctx)

        # Long cached prefixes exercise a nonzero start tile.
        execute_rel_logits_flash_attention_test[
            SlidingWindowCausalMask[6](),
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=16, head_size=64),
            page_size=128,
            extent=4,
        ]([1, 1], [258, 513], "decode+window+kv16+long_prefix", ctx)

        execute_rel_logits_flash_attention_test[
            CausalMask(),
            num_q_heads=32,
            kv_params=KVCacheStaticParams(num_heads=8, head_size=64),
            page_size=128,
            extent=0,
        ]([1], [0], "prefill+empty_extent", ctx)

        # TODO: Investigate MI355
        comptime if ctx.default_device_info == B200:
            execute_rel_logits_flash_attention_test[
                SlidingWindowCausalMask[512](),
                num_q_heads=32,
                kv_params=KVCacheStaticParams(num_heads=8, head_size=128),
                page_size=128,
                extent=512,
            ](
                [1, 1],
                [2048, 3072],
                "decode+window+splitk_p8",
                ctx,
                num_partitions=8,
            )

            execute_rel_logits_flash_attention_test[
                CausalMask(),
                num_q_heads=32,
                kv_params=KVCacheStaticParams(num_heads=8, head_size=128),
                page_size=128,
                extent=1024,
            ](
                [1, 1],
                [2048, 4096],
                "decode+global+splitk_p4",
                ctx,
                num_partitions=4,
            )
