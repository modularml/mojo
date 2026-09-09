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

from std.math import ceildiv, rsqrt
from std.random import seed

from max.gpu.host import DeviceContext
from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout._fillers import random
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal
from std.utils import Index, IndexList

comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    valid_length: LayoutTensor[.uint32, Layout(UNKNOWN_VALUE), _],
    max_seq_len: Int,
    num_layers: Int,
    layer_idx: Int,
    cache_valid_length: LayoutTensor[.uint32, Layout(UNKNOWN_VALUE), _],
    ctx: DeviceContext,
) raises:
    comptime page_size = 128
    var pages_per_seq = ceildiv(max_seq_len, page_size)
    # Twice what the batch needs, so the lookup table indexes sparsely into the
    # pool instead of covering a dense prefix of it.
    var num_blocks = 2 * batch_size * pages_per_seq

    var max_prompt_len = 0
    var max_context_len = 0

    for i in range(batch_size):
        max_prompt_len = max(max_prompt_len, Int(valid_length[i]))
        max_context_len = max(
            max_context_len, Int(cache_valid_length[i] + valid_length[i])
        )

    # Define layouts for q tensor
    comptime q_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    var q_shape = IndexList[4](
        batch_size, max_prompt_len, num_q_heads, kv_params.head_size
    )
    var q_runtime_layout = RuntimeLayout[q_static_layout].row_major(q_shape)

    var q = ManagedLayoutTensor[dtype, q_static_layout](q_runtime_layout, ctx)
    random(q.tensor())

    var valid_lengths = ManagedLayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size)
        ),
        ctx,
    )
    var valid_lengths_host = valid_lengths.tensor[update=False]()
    for i in range(batch_size):
        valid_lengths_host[i] = valid_length[i]

    # Define layouts for output tensors
    comptime output_static_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[4](
        batch_size, max_prompt_len, num_q_heads, kv_params.head_size
    )
    var output_runtime_layout = RuntimeLayout[output_static_layout].row_major(
        output_shape
    )

    var ref_output = ManagedLayoutTensor[dtype, output_static_layout](
        output_runtime_layout, ctx
    )
    var test_output = ManagedLayoutTensor[dtype, output_static_layout](
        output_runtime_layout, ctx
    )

    # initialize our KVCache
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = cache_valid_length[i]

    var cache_lengths_device = cache_lengths_managed.device_tensor()

    # Define layouts for kv_block tensor
    comptime kv_block_static_layout = Layout.row_major[6]()
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_layout
    ].row_major(kv_block_shape)

    var kv_block = ManagedLayoutTensor[dtype, kv_block_static_layout](
        kv_block_runtime_layout, ctx
    )

    # Initialize kv_block once via host view.
    var kv_block_host_tensor = kv_block.tensor()
    random(kv_block_host_tensor)

    # Create lookup table
    comptime lut_layout = Layout.row_major[2]()
    var lookup_table = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, pages_per_seq)
        ),
        ctx,
    )

    # Initialize lookup table
    var lookup_table_host = lookup_table.tensor[update=False]()
    # Every page of every sequence gets a distinct physical block, so an
    # off-by-one in the page lookup reads another sequence's data rather than
    # aliasing back onto the correct row.
    var lut_blocks = random_distinct(num_blocks, batch_size * pages_per_seq)
    for batch_idx in range(batch_size):
        for page_idx in range(pages_per_seq):
            lookup_table_host[batch_idx, page_idx] = UInt32(
                lut_blocks[batch_idx * pages_per_seq + page_idx]
            )

    # Create layout tensors for GPU operations
    var q_tensor = q.device_tensor()
    var valid_lengths_tensor = valid_lengths.device_tensor()
    var ref_output_tensor = ref_output.device_tensor()
    var test_output_tensor = test_output.device_tensor()
    var kv_block_tensor = kv_block.device_tensor()
    var lookup_table_tensor = lookup_table.device_tensor()

    var kv_collection_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_tensor,
        cache_lengths_device,
        lookup_table_tensor,
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    flash_attention(
        test_output_tensor,
        q_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        valid_lengths_tensor,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    mha_gpu_naive(
        q_tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        ref_output_tensor,
        valid_lengths_tensor,
        rsqrt(Float32(kv_params.head_size)),
        batch_size,
        max_prompt_len,
        max_context_len,
        num_q_heads,
        kv_params.head_size,
        num_q_heads // kv_params.num_heads,
        ctx,
    )

    # Verify results
    var rtol = 8e-3
    var test_out_tensor = test_output.tensor()
    var ref_out_tensor = ref_output.tensor()
    for bs in range(batch_size):
        for s in range(Int(valid_length[bs])):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    assert_almost_equal(
                        ref_out_tensor[bs, s, h, hd],
                        test_out_tensor[bs, s, h, hd],
                        atol=1e-5,
                        rtol=rtol,
                    )


def execute_flash_attention_suite(ctx: DeviceContext) raises:
    # comptime dtypes = (DType.float32, DType.bfloat16)
    comptime dtypes = (DType.bfloat16,)
    var bs = 2
    var valid_length_managed = ManagedLayoutTensor[
        .uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(bs)),
        ctx,
    )
    var valid_length = valid_length_managed.tensor[update=False]()

    var cache_valid_length_managed = ManagedLayoutTensor[
        .uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(Index(bs)),
        ctx,
    )
    var cache_valid_length = cache_valid_length_managed.tensor[update=False]()

    comptime for dtype_idx in range(len(dtypes)):
        comptime dtype = rebind[DType](dtypes[dtype_idx])
        # Replit context encoding [testing even query valid lengths].
        valid_length[0] = 128
        valid_length[1] = 64
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, dtype, kv_params_replit](
            bs, valid_length, 1024, 4, 3, cache_valid_length, ctx
        )

        # Replit context encoding [testing odd query valid length].
        valid_length[0] = 128
        valid_length[1] = 65
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, dtype, kv_params_replit](
            bs, valid_length, 1024, 4, 0, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 256

        execute_flash_attention[replit_num_q_heads, dtype, kv_params_replit](
            bs, valid_length, 1024, 4, 1, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 255

        execute_flash_attention[replit_num_q_heads, dtype, kv_params_replit](
            bs, valid_length, 1024, 4, 2, cache_valid_length, ctx
        )


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
