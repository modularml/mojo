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
from std.math.uutils import udivmod
from std.random import random_ui64, seed

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    TileTensor,
    Coord,
    Idx,
    row_major,
)
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from std.memory import unsafe_memcpy
from nn.kv_cache_ragged import (
    _matmul_k_cache_ragged_scale_impl,
)
from std.testing import assert_almost_equal

from std.utils import IndexList

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


comptime block_scale = 128


def _initialize_ragged_inputs[
    dtype: DType, hidden_size: Int
](
    input_row_offsets_host_ptr: MutPointer[UInt32, _],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) raises -> Tuple[
    DeviceBuffer[.uint32],
    DeviceBuffer[dtype],
    DeviceBuffer[dtype],
    Int,  # total_length
    Int,  # max_seq_length_batch
]:
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    var total_length = 0
    var max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(total_length)

        var curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host_ptr[batch_size] = UInt32(total_length)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    # Initialize ragged hidden state.
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    var ragged_size = total_length * hidden_size
    var hidden_state_ragged_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        ragged_size
    )
    ctx.synchronize()
    var hidden_state_ragged_host = LayoutTensor[dtype, hidden_state_layout](
        hidden_state_ragged_host_ptr,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](total_length, hidden_size)
        ),
    )
    random(hidden_state_ragged_host)

    var hidden_state_ragged_device = ctx.enqueue_create_buffer[dtype](
        ragged_size
    )
    ctx.enqueue_copy(hidden_state_ragged_device, hidden_state_ragged_host_ptr)

    # Initialize padded hidden state.
    var padded_size = batch_size * max_seq_length_batch * hidden_size
    var hidden_state_padded_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        padded_size
    )
    ctx.synchronize()

    # Copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        var unpadded_seq_len = prompt_lens[bs]
        var ragged_start_idx = Int(input_row_offsets_host_ptr[bs])
        for s in range(unpadded_seq_len):
            var padded_ptr = (
                hidden_state_padded_host_ptr.unsafe_ptr()
                + (bs * max_seq_length_batch + s) * hidden_size
            )
            var ragged_ptr = (
                hidden_state_ragged_host_ptr.unsafe_ptr()
                + (ragged_start_idx + s) * hidden_size
            )
            unsafe_memcpy(dest=padded_ptr, src=ragged_ptr, count=hidden_size)

    var hidden_state_padded_device = ctx.enqueue_create_buffer[dtype](
        padded_size
    )
    ctx.enqueue_copy(hidden_state_padded_device, hidden_state_padded_host_ptr)

    # Sync here so that HtoD transfers complete prior to host buffer dtor.
    ctx.synchronize()

    return (
        input_row_offsets_device,
        hidden_state_ragged_device,
        hidden_state_padded_device,
        total_length,
        max_seq_length_batch,
    )


def execute_matmul_k_cache_ragged_scale[
    num_q_heads: Int,
    dtype: DType,
    weight_dtype: DType,
    scale_dtype: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
    atol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    """Tests the scaled KV cache matmul for key projections.

    This test follows the same pattern as execute_matmul_k_cache_ragged but
    includes input_scale and weight_scale parameters for scaled FP8 operations.
    """
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size

    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime CollectionType = PagedKVCacheCollection[
        dtype, kv_params, page_size, ...
    ]
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major[6]()
    comptime hidden_state_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime weight_layout = Layout.row_major(kv_hidden_size, hidden_size)
    comptime input_scale_rows = ceildiv(hidden_size, block_scale)
    comptime weight_scale_rows: Int = ceildiv(kv_hidden_size, block_scale)
    comptime weight_scale_cols = ceildiv(hidden_size, block_scale)
    comptime input_scale_layout = Layout.row_major(
        input_scale_rows, UNKNOWN_VALUE
    )
    comptime weight_scale_layout = Layout.row_major(
        weight_scale_rows, weight_scale_cols
    )
    comptime ref_output_layout = Layout.row_major(UNKNOWN_VALUE, kv_hidden_size)

    var batch_size = len(prompt_lens)

    assert len(prompt_lens) == len(
        cache_sizes
    ), "expected prompt_lens and cache_sizes size to be equal"

    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block = ManagedLayoutTensor[dtype, kv_block_layout](
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
        ctx,
    )

    var cache_lengths_table = CacheLengthsTable.build(
        prompt_lens, cache_sizes, ctx
    )

    var max_full_context_length = cache_lengths_table.max_full_context_length
    var max_seq_length_batch = cache_lengths_table.max_seq_length_batch

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_sizes, max_full_context_length, num_paged_blocks, ctx
    )

    var kv_collection_device = CollectionType(
        kv_block.device_tensor(),
        cache_lengths_table.cache_lengths.device_tensor(),
        paged_lut.device_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)

    var kv_collection_host = CollectionType(
        kv_block.tensor(),
        cache_lengths_table.cache_lengths.host_tensor(),
        paged_lut.host_tensor(),
        UInt32(max_seq_length_batch),
        UInt32(max_full_context_length),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)

    # Initialize input row offsets and hidden states.
    var input_row_offsets_host_ptr = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](batch_size + 1)
    ctx.synchronize()
    var init_result = _initialize_ragged_inputs[weight_dtype, hidden_size](
        input_row_offsets_host_ptr.unsafe_ptr(), batch_size, prompt_lens, ctx
    )
    var input_row_offsets_device = init_result[0]
    var hidden_state_ragged_device = init_result[1]
    var hidden_state_padded_device = init_result[2]
    var ragged_total_length = init_result[3]
    var init_max_seq_length_batch = init_result[4]

    # Initialize the weights.
    var weight_size = kv_hidden_size * hidden_size
    var weight_shape = IndexList[2](kv_hidden_size, hidden_size)
    var weight_host_ptr = ctx.enqueue_create_host_buffer[weight_dtype](
        weight_size
    )
    ctx.synchronize()
    var weight_host = LayoutTensor[weight_dtype, weight_layout](
        weight_host_ptr,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    random(weight_host)
    var weight_device = ctx.enqueue_create_buffer[weight_dtype](weight_size)
    ctx.enqueue_copy(weight_device, weight_host_ptr)

    # Initialize scales for blockwise scaling.
    var input_scale_cols = ragged_total_length
    var input_scale_shape = IndexList[2](input_scale_rows, input_scale_cols)
    var weight_scale_shape = IndexList[2](weight_scale_rows, weight_scale_cols)

    var input_scale = ManagedLayoutTensor[scale_dtype, input_scale_layout](
        RuntimeLayout[input_scale_layout].row_major(input_scale_shape),
        ctx,
    )
    var input_scale_host = input_scale.tensor[update=False]()
    var weight_scale = ManagedLayoutTensor[scale_dtype, weight_scale_layout](
        ctx
    )
    var weight_scale_host = weight_scale.tensor[update=False]()

    random(input_scale_host)
    random(weight_scale_host)

    # Initialize reference output.
    var ref_output_size = ragged_total_length * kv_hidden_size
    var ref_output_shape = IndexList[2](ragged_total_length, kv_hidden_size)
    var ref_output = ManagedLayoutTensor[dtype, ref_output_layout](
        RuntimeLayout[ref_output_layout].row_major(ref_output_shape),
        ctx,
    )
    var ref_output_host = ref_output.tensor[update=False]()

    # Create device LayoutTensors for kernel calls
    var hidden_state_ragged_tensor = LayoutTensor[
        weight_dtype, hidden_state_layout
    ](
        hidden_state_ragged_device,
        RuntimeLayout[hidden_state_layout].row_major(
            IndexList[2](ragged_total_length, hidden_size)
        ),
    )
    var input_row_offsets_tensor = LayoutTensor[
        mut=False,
        .uint32,
        layout_1d,
    ](
        input_row_offsets_device,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var weight_device_tensor = LayoutTensor[
        weight_dtype,
        weight_layout,
    ](
        weight_device,
        RuntimeLayout[weight_layout].row_major(weight_shape),
    )
    var input_scale_device_tensor = LayoutTensor[
        scale_dtype,
        input_scale_layout,
    ](
        input_scale.device_tensor().ptr,
        input_scale.device_tensor().runtime_layout,
    )
    var weight_scale_device_tensor = LayoutTensor[
        scale_dtype,
        weight_scale_layout,
    ](
        weight_scale.device_tensor().ptr,
    )

    # Execute test with scaled implementation.
    _matmul_k_cache_ragged_scale_impl[
        target="gpu",
        scales_granularity_mnk=IndexList[3](1, block_scale, block_scale),
    ](
        hidden_state_ragged_tensor,
        input_row_offsets_tensor,
        weight_device_tensor,
        input_scale_device_tensor,
        weight_scale_device_tensor,
        k_cache_device,
        ctx,
    )

    # Execute reference using naive blockwise scaled matmul.
    # Create TileTensors for naive_blockwise_scaled_fp8_matmul
    var ref_output_tt = TileTensor(
        ref_output.device_tensor[update=False]().ptr,
        row_major(Coord(Int(ragged_total_length), Idx[kv_hidden_size])),
    )
    var hidden_state_ragged_tt = TileTensor(
        hidden_state_ragged_device,
        row_major(Coord(Int(ragged_total_length), Idx[hidden_size])),
    )
    var weight_ref_tt = TileTensor(
        weight_device,
        row_major(Coord(Idx[kv_hidden_size], Idx[hidden_size])),
    )
    var ref_input_scale_tt = TileTensor(
        input_scale.device_tensor[update=False]().ptr,
        row_major(Coord(Idx[input_scale_rows], Int(input_scale_cols))),
    )
    var ref_weight_scale_tt = TileTensor(
        weight_scale.device_tensor[update=False]().ptr,
        row_major(Coord(Idx[weight_scale_rows], Idx[weight_scale_cols])),
    )

    # Use naive blockwise scaled matmul as reference
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=IndexList[3](1, block_scale, block_scale),
    ](
        ref_output_tt.to_layout_tensor(),
        hidden_state_ragged_tt.to_layout_tensor(),
        weight_ref_tt.to_layout_tensor(),
        ref_input_scale_tt.to_layout_tensor(),
        ref_weight_scale_tt.to_layout_tensor(),
        ctx,
    )

    var kv_block_host = kv_block.tensor()
    ref_output_host = ref_output.tensor()

    # Verify results
    for bs in range(batch_size):
        var prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = divmod(k_dim, kv_params.head_size)
                var a = ref_output_host[
                    Int(input_row_offsets_host_ptr[bs]) + s, k_dim
                ]
                var b = k_cache_host.load[width=1](
                    bs,
                    head_idx,
                    cache_sizes[bs] + s,
                    head_dim_idx,
                )
                assert_almost_equal(a, b, atol=atol, rtol=rtol)

    # Cleanup device buffers
    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_device^
    _ = input_row_offsets_device^

    # Cleanup managed objects.
    _ = cache_lengths_table^
    _ = paged_lut^


def execute_fused_matmul_suite_float8_e4m3fn(ctx: DeviceContext) raises:
    """Test suite specifically for FP8 scaled matmul operations."""
    comptime dtype = DType.float8_e4m3fn
    comptime rtol = 1e-2
    comptime atol = 1e-2
    for bs in [1, 16]:
        var ce_cache_sizes = List[Int]()
        var ce_seq_lens = List[Int]()
        var tg_cache_sizes = List[Int]()
        var tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            # TODO increase sizes here to ensure we cross page boundary.
            tg_cache_sizes.append(Int(random_ui64(512, 700)))
            ce_seq_lens.append(Int(random_ui64(512, 700)))
            ce_cache_sizes.append(0)

        # Context encoding test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)

        # Token generation test
        execute_matmul_k_cache_ragged_scale[
            llama_num_q_heads,
            DType.float32,
            dtype,
            dtype,
            kv_params_llama3,
            rtol,
            atol,
        ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite_float8_e4m3fn(ctx)
