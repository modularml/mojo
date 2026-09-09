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

from std.random import random_ui64, seed
from std.math.uutils import udivmod

from max.gpu.host import DeviceContext
from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    Idx,
    UNKNOWN_VALUE,
    row_major,
)
from layout._utils import ManagedLayoutTensor
from layout._fillers import random
from linalg.matmul.gpu import _matmul_gpu
from nn.kv_cache import _fused_qkv_matmul_kv_cache_impl
from std.testing import assert_almost_equal

from std.utils import IndexList

comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_fused_qkv_matmul[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, ...
    ]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Define layouts
    comptime hidden_state_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, hidden_size
    )
    comptime weight_layout = Layout.row_major(fused_hidden_size, hidden_size)
    comptime ref_output_layout = Layout.row_major[2]()
    comptime test_output_layout = Layout.row_major[3]()
    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)

    # Define shapes
    var hidden_state_shape = IndexList[3](batch_size, prompt_len, hidden_size)
    var weight_shape = IndexList[2](fused_hidden_size, hidden_size)
    var ref_output_shape = IndexList[2](
        batch_size * prompt_len, fused_hidden_size
    )
    var test_output_shape = IndexList[3](batch_size, prompt_len, hidden_size)
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_len,
        kv_params.num_heads,
        kv_params.head_size,
    )

    # Initialize hidden state
    var hidden_state = ManagedLayoutTensor[dtype, hidden_state_layout](
        RuntimeLayout[hidden_state_layout].row_major(hidden_state_shape), ctx
    )
    var hidden_state_host = hidden_state.tensor()
    random(hidden_state_host)

    var hidden_state_device_2d = TileTensor(
        hidden_state.device_tensor().ptr,
        row_major(batch_size * prompt_len, Idx[hidden_size]),
    )

    # Keep matmul weights on a direct device buffer; _matmul_gpu expects this
    # static layout path and currently does not compose well with managed views.
    var weight_device = ctx.enqueue_create_buffer[dtype](
        weight_shape.flattened_length()
    )
    with weight_device.map_to_host() as weight_host_ptr:
        var weight_host = LayoutTensor[dtype, weight_layout](
            weight_host_ptr,
            RuntimeLayout[weight_layout].row_major(weight_shape),
        )
        random(weight_host)

    # Initialize reference output
    var ref_output = ManagedLayoutTensor[dtype, ref_output_layout](
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape), ctx
    )

    # Initialize test output
    var test_output = ManagedLayoutTensor[dtype, test_output_layout](
        RuntimeLayout[test_output_layout].row_major(test_output_shape), ctx
    )

    # Initialize our KVCache
    var is_context_encoding = True
    var cache_lengths = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_sizes[i])
        if cache_lengths_host[i] != 0:
            is_context_encoding = False

    var kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape), ctx
    )
    var kv_block_host = kv_block.tensor()

    var lookup_table = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_table_host = lookup_table.tensor[update=False]()

    # Assign each batch entry a distinct block. `random_ui64` is inclusive, so
    # the original draw range `[0, num_blocks - 1]` is a population of
    # `num_blocks` blocks.
    var lut_blocks = random_distinct(num_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(lut_blocks[idx])

    var kv_collection_device = CollectionType(
        kv_block.device_tensor(),
        cache_lengths.device_tensor(),
        lookup_table.device_tensor(),
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )

    # Create device tensors for kernel calls
    var hidden_state_device_tensor = hidden_state.device_tensor()
    var weight_device_tensor = LayoutTensor[dtype, weight_layout](
        weight_device,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    var test_output_device_tensor = test_output.device_tensor()

    # Create valid_lengths - all sequences have full prompt_len valid
    var valid_lengths = ManagedLayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
        ctx,
    )
    var valid_lengths_host = valid_lengths.tensor[update=False]()
    for i in range(batch_size):
        valid_lengths_host[i] = UInt32(prompt_len)
    var valid_lengths_tensor = valid_lengths.device_tensor()

    _fused_qkv_matmul_kv_cache_impl[target="gpu"](
        hidden_state_device_tensor,
        weight_device_tensor,
        kv_collection_device,
        UInt32(layer_idx),
        valid_lengths_tensor,
        test_output_device_tensor,
        ctx,
    )

    var ref_output_device_ndbuffer = TileTensor(
        ref_output.device_tensor().ptr,
        row_major(ref_output_shape[0], Idx[fused_hidden_size]),
    )
    var weight_device_ndbuffer = TileTensor(
        weight_device,
        row_major(Idx[fused_hidden_size], Idx[hidden_size]),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device_ndbuffer,
        hidden_state_device_2d,
        weight_device_ndbuffer,
        ctx,
    )

    var kv_block_host_after = kv_block.tensor()
    var test_output_host = test_output.tensor()
    var ref_output_host = ref_output.tensor()
    var kv_collection_host = CollectionType(
        kv_block_host_after,
        cache_lengths_host,
        lookup_table_host,
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)
    for bs in range(batch_size):
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                assert_almost_equal(
                    ref_output_host[bs * prompt_len + s, q_dim],
                    test_output_host[bs, s, q_dim],
                )

            for k_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(k_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[bs * prompt_len + s, hidden_size + k_dim],
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )

            for v_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(v_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[
                        bs * prompt_len + s,
                        hidden_size + kv_hidden_size + v_dim,
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )


def execute_fused_matmul_suite(ctx: DeviceContext) raises:
    comptime dtypes = (DType.float32, DType.bfloat16)

    comptime for dtype_idx in range(2):
        comptime dtype = rebind[DType](dtypes[dtype_idx])
        for bs in [1, 16]:
            var ce_cache_sizes = List[Int]()
            var tg_cache_sizes = List[Int]()
            for _ in range(bs):
                tg_cache_sizes.append(Int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 128, 1024, ce_cache_sizes, 4, 1, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 512, 1024, ce_cache_sizes, 4, 0, ctx)

            # llama3 token gen
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 3, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 0, ctx)


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
