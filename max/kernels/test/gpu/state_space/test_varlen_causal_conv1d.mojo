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

from std.math import ceildiv, exp

from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_fwd_gpu,
    causal_conv1d_varlen_fwd_seqparallel_gpu,
    causal_conv1d_varlen_update_gpu,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index, IndexList


# Constants
comptime PAD_SLOT_ID: Int32 = -1


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_varlen_causal_conv1d_fwd_gpu[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seq_lengths: IndexList,
    width: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    nonzero_initial_state: Bool = False,
) raises:
    """Test varlen causal conv1d forward GPU kernel against CPU reference.

    Also cross-checks the sequence-parallel prefill kernel
    (`causal_conv1d_varlen_fwd_seqparallel_gpu`) against both the CPU
    reference and the serial GPU kernel's output and final `conv_states`.
    """
    # Calculate total_seqlen (sum of all sequence lengths)
    var total_seqlen = 0
    for i in range(batch):
        total_seqlen += seq_lengths[i]

    # Allocate host memory as TileTensors over their backing heaps.
    # x: (dim, total_seqlen) for varlen - sequences concatenated
    var x_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var x_h = TileTensor(x_heap, row_major(dim, total_seqlen))

    # weight: (dim, width)
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_h = TileTensor(weight_heap, row_major(dim, width))

    # bias: (dim,)
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_h = TileTensor(
        bias_heap,
        row_major(
            dim,
        ),
    )

    # query_start_loc: (batch + 1,) - cumulative sequence lengths
    var query_start_loc_heap = List(length=batch + 1, fill=Int32(0))
    var query_start_loc_h = TileTensor(
        query_start_loc_heap,
        row_major(
            batch + 1,
        ),
    )
    var cumsum = 0
    query_start_loc_h.raw_store(0, Int32(0))
    for i in range(batch):
        cumsum += seq_lengths[i]
        query_start_loc_h.raw_store(i + 1, Int32(cumsum))

    # cache_indices: (batch,) - identity mapping
    var cache_indices_heap = List(length=batch, fill=Int32(0))
    var cache_indices_h = TileTensor(
        cache_indices_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        cache_indices_h.raw_store(i, Int32(i))

    # has_initial_state: (batch,) - all False
    var has_initial_state_heap = List(length=batch, fill=Scalar[.bool](False))
    var has_initial_state_h = TileTensor(
        has_initial_state_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        has_initial_state_h.raw_store(i, Scalar[.bool](nonzero_initial_state))

    # conv_states: (batch, dim, width - 1)
    var state_len = width - 1
    var conv_states_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_states_h = TileTensor(
        conv_states_heap,
        row_major(batch, dim, state_len),
    )
    for i in range(batch * dim * state_len):
        conv_states_h.raw_store(i, Scalar[dtype](0))
    if nonzero_initial_state:
        # Non-zero seed so the "read initial conv_states" gather path (both
        # kernels) is actually exercised, not just the all-zero default.
        rand[dtype](conv_states_h._storage, batch * dim * state_len)

    # The seed, kept separately: the CPU reference call below overwrites
    # `conv_states_h` in place, and the write-back check needs what the pool
    # held BEFORE the kernel ran.
    var conv_states_seed_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    for i in range(batch * dim * state_len):
        conv_states_seed_heap[i] = conv_states_h.raw_load(i)

    # output: (dim, total_seqlen)
    var output_gpu_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var output_gpu_h = TileTensor(output_gpu_heap, row_major(dim, total_seqlen))

    var output_cpu_heap = List(length=dim * total_seqlen, fill=Scalar[dtype](0))
    var output_cpu_h = TileTensor(output_cpu_heap, row_major(dim, total_seqlen))

    # Initialize input data
    rand[dtype](x_h._storage, dim * total_seqlen)
    rand[dtype](weight_h._storage, dim * width)
    rand[dtype](bias_h._storage, dim)

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var query_start_loc_buf = query_start_loc_h
    var cache_indices_buf = cache_indices_h
    var has_initial_state_buf = has_initial_state_h
    var conv_states_buf = conv_states_h
    var output_gpu_buf = output_gpu_h
    var output_cpu_buf = output_cpu_h

    # Strides for row-major layout
    var x_dim_stride: UInt32 = UInt32(total_seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_dim_stride: UInt32 = UInt32(total_seqlen)
    var out_seqlen_stride: UInt32 = 1
    var conv_states_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_states_dim_stride: UInt32 = UInt32(state_len)
    var conv_states_width_stride: UInt32 = 1

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[dtype](dim * total_seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var query_start_loc_device = ctx.enqueue_create_buffer[.int32](batch + 1)
    var cache_indices_device = ctx.enqueue_create_buffer[.int32](batch)
    var has_initial_state_device = ctx.enqueue_create_buffer[.bool](batch)
    var conv_states_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var output_device = ctx.enqueue_create_buffer[dtype](dim * total_seqlen)

    # Copy data to device
    ctx.enqueue_copy(x_device, x_buf._storage)
    ctx.enqueue_copy(weight_device, weight_buf._storage)
    ctx.enqueue_copy(bias_device, bias_buf._storage)
    ctx.enqueue_copy(query_start_loc_device, query_start_loc_buf._storage)
    ctx.enqueue_copy(cache_indices_device, cache_indices_buf._storage)
    ctx.enqueue_copy(has_initial_state_device, has_initial_state_buf._storage)
    ctx.enqueue_copy(conv_states_device, conv_states_buf._storage)

    # Create TileTensors for GPU kernel
    var x_device_tt = TileTensor(
        x_device,
        row_major(dim, total_seqlen),
    )
    var weight_device_tt = TileTensor(
        weight_device,
        row_major(dim, width),
    )
    var bias_device_tt = TileTensor(
        bias_device,
        row_major(
            dim,
        ),
    )
    var query_start_loc_device_tt = TileTensor(
        query_start_loc_device,
        row_major(
            batch + 1,
        ),
    )
    var cache_indices_device_tt = TileTensor(
        cache_indices_device,
        row_major(
            batch,
        ),
    )
    var has_initial_state_device_tt = TileTensor(
        has_initial_state_device,
        row_major(
            batch,
        ),
    )
    var conv_states_device_tt = TileTensor(
        conv_states_device,
        row_major(batch, dim, state_len),
    )
    var output_device_tt = TileTensor(
        output_device,
        row_major(dim, total_seqlen),
    )

    # Run GPU kernel
    comptime BLOCK_DIM = 128
    comptime BLOCK_SEQ = 1

    if width == 1:
        comptime kWidth = 1
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                DType.bool,
                dtype,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
                x_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                query_start_loc_device_tt.LayoutType,
                cache_indices_device_tt.LayoutType,
                has_initial_state_device_tt.LayoutType,
                conv_states_device_tt.LayoutType,
                output_device_tt.LayoutType,
                x_device_tt.Engine,
                weight_device_tt.Engine,
                bias_device_tt.Engine,
                query_start_loc_device_tt.Engine,
                cache_indices_device_tt.Engine,
                has_initial_state_device_tt.Engine,
                conv_states_device_tt.Engine,
                output_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(dim),
            Int32(total_seqlen),
            Int32(batch),
            x_device_tt,
            weight_device_tt,
            bias_device_tt,
            query_start_loc_device_tt,
            cache_indices_device_tt,
            has_initial_state_device_tt,
            conv_states_device_tt,
            output_device_tt,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            out_dim_stride,
            out_seqlen_stride,
            conv_states_batch_stride,
            conv_states_dim_stride,
            conv_states_width_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_cache_indices
            Int8(1),  # has_initial_state_flag
            Int8(1),  # has_conv_states
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_SEQ),
        )
    elif width == 2:
        comptime kWidth = 2
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                DType.bool,
                dtype,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
                x_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                query_start_loc_device_tt.LayoutType,
                cache_indices_device_tt.LayoutType,
                has_initial_state_device_tt.LayoutType,
                conv_states_device_tt.LayoutType,
                output_device_tt.LayoutType,
                x_device_tt.Engine,
                weight_device_tt.Engine,
                bias_device_tt.Engine,
                query_start_loc_device_tt.Engine,
                cache_indices_device_tt.Engine,
                has_initial_state_device_tt.Engine,
                conv_states_device_tt.Engine,
                output_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(dim),
            Int32(total_seqlen),
            Int32(batch),
            x_device_tt,
            weight_device_tt,
            bias_device_tt,
            query_start_loc_device_tt,
            cache_indices_device_tt,
            has_initial_state_device_tt,
            conv_states_device_tt,
            output_device_tt,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            out_dim_stride,
            out_seqlen_stride,
            conv_states_batch_stride,
            conv_states_dim_stride,
            conv_states_width_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_cache_indices
            Int8(1),  # has_initial_state_flag
            Int8(1),  # has_conv_states
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_SEQ),
        )
    elif width == 3:
        comptime kWidth = 3
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                DType.bool,
                dtype,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
                x_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                query_start_loc_device_tt.LayoutType,
                cache_indices_device_tt.LayoutType,
                has_initial_state_device_tt.LayoutType,
                conv_states_device_tt.LayoutType,
                output_device_tt.LayoutType,
                x_device_tt.Engine,
                weight_device_tt.Engine,
                bias_device_tt.Engine,
                query_start_loc_device_tt.Engine,
                cache_indices_device_tt.Engine,
                has_initial_state_device_tt.Engine,
                conv_states_device_tt.Engine,
                output_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(dim),
            Int32(total_seqlen),
            Int32(batch),
            x_device_tt,
            weight_device_tt,
            bias_device_tt,
            query_start_loc_device_tt,
            cache_indices_device_tt,
            has_initial_state_device_tt,
            conv_states_device_tt,
            output_device_tt,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            out_dim_stride,
            out_seqlen_stride,
            conv_states_batch_stride,
            conv_states_dim_stride,
            conv_states_width_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_cache_indices
            Int8(1),  # has_initial_state_flag
            Int8(1),  # has_conv_states
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_SEQ),
        )
    elif width == 4:
        comptime kWidth = 4
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                DType.bool,
                dtype,
                kWidth,
                BLOCK_DIM,
                BLOCK_SEQ,
                x_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                query_start_loc_device_tt.LayoutType,
                cache_indices_device_tt.LayoutType,
                has_initial_state_device_tt.LayoutType,
                conv_states_device_tt.LayoutType,
                output_device_tt.LayoutType,
                x_device_tt.Engine,
                weight_device_tt.Engine,
                bias_device_tt.Engine,
                query_start_loc_device_tt.Engine,
                cache_indices_device_tt.Engine,
                has_initial_state_device_tt.Engine,
                conv_states_device_tt.Engine,
                output_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(dim),
            Int32(total_seqlen),
            Int32(batch),
            x_device_tt,
            weight_device_tt,
            bias_device_tt,
            query_start_loc_device_tt,
            cache_indices_device_tt,
            has_initial_state_device_tt,
            conv_states_device_tt,
            output_device_tt,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            out_dim_stride,
            out_seqlen_stride,
            conv_states_batch_stride,
            conv_states_dim_stride,
            conv_states_width_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_cache_indices
            Int8(1),  # has_initial_state_flag
            Int8(1),  # has_conv_states
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_SEQ),
        )
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    # Copy GPU results back to host
    ctx.enqueue_copy(output_gpu_buf._storage, output_device)
    ctx.synchronize()

    # --- Seq-parallel prefill kernel: run independently, cross-check ---
    # Copy back the serial kernel's final conv_states now, before the CPU
    # reference below mutates the shared host seed buffer (`conv_states_buf`)
    # in place.
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    var conv_states_serial_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_states_serial_h = LayoutTensor[dtype, layout_3d, _](
        conv_states_serial_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )
    with ctx.push_context():
        ctx.enqueue_copy(conv_states_serial_h.ptr, conv_states_device)
    ctx.synchronize()

    # Check the state against the CONTRACT, not against the other kernel: after
    # consuming a chunk the pool holds the last `state_len` tokens of everything
    # seen, so entries the chunk is too short to supply come from the seed. The
    # seq-parallel-vs-serial comparison further down cannot see a violation both
    # kernels share, which is exactly how a missing carry-over survived here.
    if nonzero_initial_state:
        for b in range(batch):
            var seq_start = Int(query_start_loc_h.raw_load(b))
            var seqlen = Int(query_start_loc_h.raw_load(b + 1)) - seq_start
            for d in range(dim):
                for s in range(state_len):
                    var offset = seqlen - state_len + s
                    var expected: Scalar[dtype]
                    if offset >= 0:
                        expected = x_h.raw_load(
                            UInt32(d) * UInt32(total_seqlen)
                            + UInt32(seq_start + offset)
                        )
                    else:
                        expected = conv_states_seed_heap[
                            (b * dim + d) * state_len + state_len + offset
                        ]
                    assert_almost_equal(
                        conv_states_serial_h.ptr[(b * dim + d) * state_len + s],
                        expected,
                        rtol=rtol,
                    )

    # Fresh device buffers so the seq-parallel run starts from the same
    # (still-unmutated) initial conv_states as the serial run above.
    var conv_states_seqpar_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var output_seqpar_device = ctx.enqueue_create_buffer[dtype](
        dim * total_seqlen
    )
    with ctx.push_context():
        ctx.enqueue_copy(conv_states_seqpar_device, conv_states_buf._storage)
    var conv_states_seqpar_device_tt = TileTensor(
        conv_states_seqpar_device,
        row_major(batch, dim, state_len),
    )
    var output_seqpar_device_tt = TileTensor(
        output_seqpar_device,
        row_major(dim, total_seqlen),
    )

    comptime TILE_SEQ = 128

    @__parameter
    @always_inline
    def launch_seqpar_gpu[kWidth: Int]() raises:
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_fwd_seqparallel_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                DType.bool,
                dtype,
                kWidth,
                BLOCK_DIM,
                TILE_SEQ,
                x_device_tt.LayoutType,
                weight_device_tt.LayoutType,
                bias_device_tt.LayoutType,
                query_start_loc_device_tt.LayoutType,
                cache_indices_device_tt.LayoutType,
                has_initial_state_device_tt.LayoutType,
                conv_states_seqpar_device_tt.LayoutType,
                output_seqpar_device_tt.LayoutType,
                x_device_tt.Engine,
                weight_device_tt.Engine,
                bias_device_tt.Engine,
                query_start_loc_device_tt.Engine,
                cache_indices_device_tt.Engine,
                has_initial_state_device_tt.Engine,
                conv_states_seqpar_device_tt.Engine,
                output_seqpar_device_tt.Engine,
            ]
        ]()
        with ctx.push_context():
            ctx.enqueue_function(
                compiled_func,
                Int32(dim),
                Int32(total_seqlen),
                Int32(batch),
                x_device_tt,
                weight_device_tt,
                bias_device_tt,
                query_start_loc_device_tt,
                cache_indices_device_tt,
                has_initial_state_device_tt,
                conv_states_seqpar_device_tt,
                output_seqpar_device_tt,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation_int8,
                PAD_SLOT_ID,
                Int8(1),  # has_cache_indices
                Int8(1),  # has_initial_state_flag
                Int8(1),  # has_conv_states
                Int8(1),  # has_bias
                grid_dim=(
                    batch,
                    ceildiv(dim, BLOCK_DIM),
                    ceildiv(total_seqlen, TILE_SEQ),
                ),
                block_dim=(BLOCK_DIM, 1),
            )

    if width == 1:
        launch_seqpar_gpu[1]()
    elif width == 2:
        launch_seqpar_gpu[2]()
    elif width == 3:
        launch_seqpar_gpu[3]()
    elif width == 4:
        launch_seqpar_gpu[4]()
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    var output_seqpar_heap = ctx.enqueue_create_host_buffer[dtype](
        dim * total_seqlen
    )
    var output_seqpar_h = LayoutTensor[dtype, layout_2d, _](
        output_seqpar_heap,
        RuntimeLayout[layout_2d].row_major(Index(dim, total_seqlen)),
    )
    var conv_states_seqpar_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_states_seqpar_h = LayoutTensor[dtype, layout_3d, _](
        conv_states_seqpar_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )
    with ctx.push_context():
        ctx.enqueue_copy(output_seqpar_h.ptr, output_seqpar_device)
        ctx.enqueue_copy(conv_states_seqpar_h.ptr, conv_states_seqpar_device)
    ctx.synchronize()

    # Create TileTensors for CPU reference
    var x_cpu_tt = TileTensor(x_buf._storage, row_major(dim, total_seqlen))
    var weight_cpu_tt = TileTensor(weight_buf._storage, row_major(dim, width))
    var bias_cpu_tt = TileTensor(
        bias_buf._storage,
        row_major(
            dim,
        ),
    )
    var query_start_loc_cpu_tt = TileTensor(
        query_start_loc_buf._storage,
        row_major(
            batch + 1,
        ),
    )
    var cache_indices_cpu_tt = TileTensor(
        cache_indices_buf._storage,
        row_major(
            batch,
        ),
    )
    var has_initial_state_cpu_tt = TileTensor(
        has_initial_state_buf._storage,
        row_major(
            batch,
        ),
    )
    var conv_states_cpu_tt = TileTensor(
        conv_states_buf._storage,
        row_major(batch, dim, state_len),
    )
    var output_cpu_tt = TileTensor(
        output_cpu_buf._storage, row_major(dim, total_seqlen)
    )

    # Run CPU reference
    causal_conv1d_varlen_fwd_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        DType.int32,
        DType.int32,
        DType.bool,
        dtype,
    ](
        dim,
        total_seqlen,
        width,
        batch,
        x_cpu_tt,
        weight_cpu_tt,
        bias_cpu_tt,
        query_start_loc_cpu_tt,
        cache_indices_cpu_tt,
        has_initial_state_cpu_tt,
        conv_states_cpu_tt,
        output_cpu_tt,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        out_dim_stride,
        out_seqlen_stride,
        conv_states_batch_stride,
        conv_states_dim_stride,
        conv_states_width_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_cache_indices
        True,  # has_initial_state_flag
        True,  # has_conv_states
        True,  # has_bias
    )

    # Compare results
    var flattened_size = dim * total_seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_gpu_h._storage[i],
            output_cpu_h._storage[i],
            rtol=rtol,
        )

    # Cross-check the seq-parallel prefill kernel: output vs the CPU
    # reference, and final conv_states vs the serial GPU kernel (the serial
    # kernel is the trusted baseline for the recurrent conv_states contract;
    # the CPU reference's conv_states buffer was already overwritten in
    # place by the CPU call above, which is why we compare against the
    # earlier-captured `conv_states_serial_h` instead).
    for i in range(flattened_size):
        assert_almost_equal(
            output_seqpar_h.ptr[i],
            output_cpu_h._storage[i],
            rtol=rtol,
        )
    var state_flattened_size = batch * dim * state_len
    for i in range(state_flattened_size):
        assert_almost_equal(
            conv_states_seqpar_h.ptr[i],
            conv_states_serial_h.ptr[i],
            rtol=rtol,
        )

    # Host buffers are List-owned: RAII frees them on scope exit, including on an
    # early assertion raise above. Consume them here so they outlive the async
    # H2D copies until ctx.synchronize() (each List's borrow is released at its
    # host TileTensor's last use, well before this point).
    _ = x_heap^
    _ = weight_heap^
    _ = bias_heap^
    _ = query_start_loc_heap^
    _ = cache_indices_heap^
    _ = has_initial_state_heap^
    _ = conv_states_heap^
    _ = output_gpu_heap^
    _ = output_cpu_heap^


def run_varlen_causal_conv1d_update_gpu[
    dtype: DType,
    activation: StaticString,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test varlen causal conv1d update GPU kernel against CPU reference."""
    # Allocate host memory as TileTensors over their backing heaps.
    # x: (batch, dim, seqlen)
    var x_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var x_h = TileTensor(x_heap, row_major(batch, dim, seqlen))

    # weight: (dim, width)
    var weight_heap = List(length=dim * width, fill=Scalar[dtype](0))
    var weight_h = TileTensor(weight_heap, row_major(dim, width))

    # bias: (dim,)
    var bias_heap = List(length=dim, fill=Scalar[dtype](0))
    var bias_h = TileTensor(
        bias_heap,
        row_major(
            dim,
        ),
    )

    # conv_state: (batch, dim, state_len)
    var conv_state_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_h = TileTensor(
        conv_state_heap,
        row_major(batch, dim, state_len),
    )

    # cache_seqlens: (batch,) - all zeros
    var cache_seqlens_heap = List(length=batch, fill=Int32(0))
    var cache_seqlens_h = TileTensor(
        cache_seqlens_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        cache_seqlens_h.raw_store(i, Int32(0))

    # conv_state_indices: (batch,) - identity mapping
    var conv_state_indices_heap = List(length=batch, fill=Int32(0))
    var conv_state_indices_h = TileTensor(
        conv_state_indices_heap,
        row_major(
            batch,
        ),
    )
    for i in range(batch):
        conv_state_indices_h.raw_store(i, Int32(i))

    # output: (batch, dim, seqlen)
    var output_gpu_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var output_gpu_h = TileTensor(
        output_gpu_heap, row_major(batch, dim, seqlen)
    )

    var output_cpu_heap = List(
        length=batch * dim * seqlen, fill=Scalar[dtype](0)
    )
    var output_cpu_h = TileTensor(
        output_cpu_heap, row_major(batch, dim, seqlen)
    )

    # Copy of conv_state for CPU and GPU
    var conv_state_cpu_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_cpu_h = TileTensor(
        conv_state_cpu_heap,
        row_major(batch, dim, state_len),
    )

    var conv_state_gpu_heap = List(
        length=batch * dim * state_len, fill=Scalar[dtype](0)
    )
    var conv_state_gpu_h = TileTensor(
        conv_state_gpu_heap,
        row_major(batch, dim, state_len),
    )

    # Initialize input data
    rand[dtype](x_h._storage, batch * dim * seqlen)
    rand[dtype](conv_state_h._storage, batch * dim * state_len)
    rand[dtype](weight_h._storage, dim * width)
    rand[dtype](bias_h._storage, dim)

    # Copy conv_state for CPU and GPU
    for i in range(batch * dim * state_len):
        conv_state_cpu_h._storage[i] = conv_state_h._storage[i]
        conv_state_gpu_h._storage[i] = conv_state_h._storage[i]

    var x_buf = x_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var conv_state_cpu_buf = conv_state_cpu_h
    var conv_state_gpu_buf = conv_state_gpu_h
    var cache_seqlens_buf = cache_seqlens_h
    var conv_state_indices_buf = conv_state_indices_h
    var output_gpu_buf = output_gpu_h
    var output_cpu_buf = output_cpu_h

    # Strides for row-major layout
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_dim_stride: UInt32 = UInt32(seqlen)
    var x_seqlen_stride: UInt32 = 1
    var weight_dim_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var conv_state_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_state_dim_stride: UInt32 = UInt32(state_len)
    var conv_state_seqlen_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_dim_stride: UInt32 = UInt32(seqlen)
    var out_seqlen_stride: UInt32 = 1

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var conv_state_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var cache_seqlens_device = ctx.enqueue_create_buffer[.int32](batch)
    var conv_state_indices_device = ctx.enqueue_create_buffer[.int32](batch)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    ctx.enqueue_copy(x_device, x_buf._storage)
    ctx.enqueue_copy(weight_device, weight_buf._storage)
    ctx.enqueue_copy(bias_device, bias_buf._storage)
    ctx.enqueue_copy(conv_state_device, conv_state_gpu_buf._storage)
    ctx.enqueue_copy(cache_seqlens_device, cache_seqlens_buf._storage)
    ctx.enqueue_copy(conv_state_indices_device, conv_state_indices_buf._storage)

    # Create TileTensors for GPU kernel
    var x_upd_device_tt = TileTensor(
        x_device,
        row_major(batch, dim, seqlen),
    )
    var weight_upd_device_tt = TileTensor(
        weight_device,
        row_major(dim, width),
    )
    var bias_upd_device_tt = TileTensor(
        bias_device,
        row_major(
            dim,
        ),
    )
    var conv_state_upd_device_tt = TileTensor(
        conv_state_device,
        row_major(batch, dim, state_len),
    )
    var cache_seqlens_device_tt = TileTensor(
        cache_seqlens_device,
        row_major(
            batch,
        ),
    )
    var conv_state_indices_device_tt = TileTensor(
        conv_state_indices_device,
        row_major(
            batch,
        ),
    )
    var output_upd_device_tt = TileTensor(
        output_device,
        row_major(batch, dim, seqlen),
    )

    # Run GPU kernel
    comptime BLOCK_DIM = 128

    if width == 1:
        comptime kWidth = 1
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                kWidth,
                BLOCK_DIM,
                x_upd_device_tt.LayoutType,
                weight_upd_device_tt.LayoutType,
                bias_upd_device_tt.LayoutType,
                conv_state_upd_device_tt.LayoutType,
                cache_seqlens_device_tt.LayoutType,
                conv_state_indices_device_tt.LayoutType,
                output_upd_device_tt.LayoutType,
                x_upd_device_tt.Engine,
                weight_upd_device_tt.Engine,
                bias_upd_device_tt.Engine,
                conv_state_upd_device_tt.Engine,
                cache_seqlens_device_tt.Engine,
                conv_state_indices_device_tt.Engine,
                output_upd_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(dim),
            Int32(seqlen),
            Int32(state_len),
            x_upd_device_tt,
            weight_upd_device_tt,
            bias_upd_device_tt,
            conv_state_upd_device_tt,
            cache_seqlens_device_tt,
            conv_state_indices_device_tt,
            output_upd_device_tt,
            x_batch_stride,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            conv_state_batch_stride,
            conv_state_dim_stride,
            conv_state_seqlen_stride,
            out_batch_stride,
            out_dim_stride,
            out_seqlen_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_conv_state_indices
            Int8(1),  # has_cache_seqlens
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM),
        )
    elif width == 2:
        comptime kWidth = 2
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                kWidth,
                BLOCK_DIM,
                x_upd_device_tt.LayoutType,
                weight_upd_device_tt.LayoutType,
                bias_upd_device_tt.LayoutType,
                conv_state_upd_device_tt.LayoutType,
                cache_seqlens_device_tt.LayoutType,
                conv_state_indices_device_tt.LayoutType,
                output_upd_device_tt.LayoutType,
                x_upd_device_tt.Engine,
                weight_upd_device_tt.Engine,
                bias_upd_device_tt.Engine,
                conv_state_upd_device_tt.Engine,
                cache_seqlens_device_tt.Engine,
                conv_state_indices_device_tt.Engine,
                output_upd_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(dim),
            Int32(seqlen),
            Int32(state_len),
            x_upd_device_tt,
            weight_upd_device_tt,
            bias_upd_device_tt,
            conv_state_upd_device_tt,
            cache_seqlens_device_tt,
            conv_state_indices_device_tt,
            output_upd_device_tt,
            x_batch_stride,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            conv_state_batch_stride,
            conv_state_dim_stride,
            conv_state_seqlen_stride,
            out_batch_stride,
            out_dim_stride,
            out_seqlen_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_conv_state_indices
            Int8(1),  # has_cache_seqlens
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM),
        )
    elif width == 3:
        comptime kWidth = 3
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                kWidth,
                BLOCK_DIM,
                x_upd_device_tt.LayoutType,
                weight_upd_device_tt.LayoutType,
                bias_upd_device_tt.LayoutType,
                conv_state_upd_device_tt.LayoutType,
                cache_seqlens_device_tt.LayoutType,
                conv_state_indices_device_tt.LayoutType,
                output_upd_device_tt.LayoutType,
                x_upd_device_tt.Engine,
                weight_upd_device_tt.Engine,
                bias_upd_device_tt.Engine,
                conv_state_upd_device_tt.Engine,
                cache_seqlens_device_tt.Engine,
                conv_state_indices_device_tt.Engine,
                output_upd_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(dim),
            Int32(seqlen),
            Int32(state_len),
            x_upd_device_tt,
            weight_upd_device_tt,
            bias_upd_device_tt,
            conv_state_upd_device_tt,
            cache_seqlens_device_tt,
            conv_state_indices_device_tt,
            output_upd_device_tt,
            x_batch_stride,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            conv_state_batch_stride,
            conv_state_dim_stride,
            conv_state_seqlen_stride,
            out_batch_stride,
            out_dim_stride,
            out_seqlen_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_conv_state_indices
            Int8(1),  # has_cache_seqlens
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM),
        )
    elif width == 4:
        comptime kWidth = 4
        var compiled_func = ctx.compile_function[
            causal_conv1d_varlen_update_gpu[
                dtype,
                dtype,
                dtype,
                dtype,
                dtype,
                DType.int32,
                DType.int32,
                kWidth,
                BLOCK_DIM,
                x_upd_device_tt.LayoutType,
                weight_upd_device_tt.LayoutType,
                bias_upd_device_tt.LayoutType,
                conv_state_upd_device_tt.LayoutType,
                cache_seqlens_device_tt.LayoutType,
                conv_state_indices_device_tt.LayoutType,
                output_upd_device_tt.LayoutType,
                x_upd_device_tt.Engine,
                weight_upd_device_tt.Engine,
                bias_upd_device_tt.Engine,
                conv_state_upd_device_tt.Engine,
                cache_seqlens_device_tt.Engine,
                conv_state_indices_device_tt.Engine,
                output_upd_device_tt.Engine,
            ]
        ]()
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(dim),
            Int32(seqlen),
            Int32(state_len),
            x_upd_device_tt,
            weight_upd_device_tt,
            bias_upd_device_tt,
            conv_state_upd_device_tt,
            cache_seqlens_device_tt,
            conv_state_indices_device_tt,
            output_upd_device_tt,
            x_batch_stride,
            x_dim_stride,
            x_seqlen_stride,
            weight_dim_stride,
            weight_width_stride,
            conv_state_batch_stride,
            conv_state_dim_stride,
            conv_state_seqlen_stride,
            out_batch_stride,
            out_dim_stride,
            out_seqlen_stride,
            silu_activation_int8,
            PAD_SLOT_ID,
            Int8(1),  # has_conv_state_indices
            Int8(1),  # has_cache_seqlens
            Int8(1),  # has_bias
            grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
            block_dim=(BLOCK_DIM),
        )
    else:
        raise Error(
            "Unsupported kernel width: only widths 1, 2, 3, 4 are supported"
        )

    # Copy results back from device
    ctx.enqueue_copy(output_gpu_buf._storage, output_device)
    ctx.enqueue_copy(conv_state_gpu_buf._storage, conv_state_device)
    ctx.synchronize()

    # Create TileTensors for CPU reference
    var x_upd_cpu_tt = TileTensor(x_buf._storage, row_major(batch, dim, seqlen))
    var weight_upd_cpu_tt = TileTensor(
        weight_buf._storage, row_major(dim, width)
    )
    var bias_upd_cpu_tt = TileTensor(
        bias_buf._storage,
        row_major(
            dim,
        ),
    )
    var conv_state_upd_cpu_tt = TileTensor(
        conv_state_cpu_buf._storage,
        row_major(batch, dim, state_len),
    )
    var cache_seqlens_cpu_tt = TileTensor(
        cache_seqlens_buf._storage,
        row_major(
            batch,
        ),
    )
    var conv_state_indices_cpu_tt = TileTensor(
        conv_state_indices_buf._storage,
        row_major(
            batch,
        ),
    )
    var output_upd_cpu_tt = TileTensor(
        output_cpu_buf._storage, row_major(batch, dim, seqlen)
    )

    # Run CPU reference
    causal_conv1d_varlen_update_cpu[
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        DType.int32,
        DType.int32,
    ](
        batch,
        dim,
        seqlen,
        width,
        state_len,
        x_upd_cpu_tt,
        weight_upd_cpu_tt,
        bias_upd_cpu_tt,
        conv_state_upd_cpu_tt,
        cache_seqlens_cpu_tt,
        conv_state_indices_cpu_tt,
        output_upd_cpu_tt,
        x_batch_stride,
        x_dim_stride,
        x_seqlen_stride,
        weight_dim_stride,
        weight_width_stride,
        conv_state_batch_stride,
        conv_state_dim_stride,
        conv_state_seqlen_stride,
        out_batch_stride,
        out_dim_stride,
        out_seqlen_stride,
        silu_activation,
        PAD_SLOT_ID,
        True,  # has_conv_state_indices
        True,  # has_cache_seqlens
        True,  # has_bias
    )

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            output_gpu_h._storage[i],
            output_cpu_h._storage[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_gpu_h._storage[i],
            conv_state_cpu_h._storage[i],
            rtol=rtol,
        )

    # Host buffers are List-owned: RAII frees them on scope exit, including on an
    # early assertion raise above. Consume them here so they outlive the async
    # H2D copies until ctx.synchronize() (each List's borrow is released at its
    # host TileTensor's last use, well before this point).
    _ = x_heap^
    _ = weight_heap^
    _ = bias_heap^
    _ = conv_state_heap^
    _ = cache_seqlens_heap^
    _ = conv_state_indices_heap^
    _ = output_gpu_heap^
    _ = output_cpu_heap^
    _ = conv_state_cpu_heap^
    _ = conv_state_gpu_heap^


# =============================================================================
# Test functions for varlen causal conv1d forward on GPU
# =============================================================================


def test_varlen_causal_conv1d_fwd_gpu_equal_lengths() raises:
    """Test varlen causal conv1d forward GPU with equal-length sequences."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3, ctx=ctx
    )


def test_varlen_causal_conv1d_fwd_gpu_variable_lengths() raises:
    """Test varlen causal conv1d forward GPU with variable-length sequences."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=3, dim=4, seq_lengths=Index(10, 6, 1), width=3, ctx=ctx
    )


def test_varlen_causal_conv1d_fwd_gpu_with_silu() raises:
    """Test varlen causal conv1d forward GPU with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "silu"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=3, ctx=ctx
    )


def test_varlen_causal_conv1d_fwd_gpu_various_widths() raises:
    """Test varlen causal conv1d forward GPU with various kernel widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=2, ctx=ctx
    )
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2, dim=4, seq_lengths=Index(8, 8), width=4, ctx=ctx
    )


# =============================================================================
# Test functions for the seq-parallel prefill kernel
# (causal_conv1d_varlen_fwd_seqparallel_gpu, cross-checked inside
# run_varlen_causal_conv1d_fwd_gpu against both the CPU reference and the
# serial GPU kernel).
# =============================================================================


def test_varlen_causal_conv1d_fwd_gpu_seqparallel_prefill_shapes() raises:
    """Single-sequence prefill at production-relevant lengths, spanning
    multiple TILE_SEQ=128 tiles and exact tile boundaries (255/256/257), for
    each kernel width the models use (2, 3, 4; width 1 also dispatches but no
    model uses it)."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    var seqlens: List[Int] = [255, 256, 257, 512, 1024, 4032]
    var widths: List[Int] = [2, 3, 4]
    for seqlen in seqlens:
        for width in widths:
            run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
                batch=1,
                dim=8,
                seq_lengths=Index(seqlen),
                width=width,
                ctx=ctx,
            )


def test_varlen_causal_conv1d_fwd_gpu_seqparallel_ragged_batch() raises:
    """Ragged varlen batch mixing multiple lengths (including a short,
    sub-tile sequence and long multi-tile sequences) in one packed call.
    Every position is independently cross-checked against the CPU reference,
    which validates there is no cross-sequence bleed across the shared `x`
    buffer."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=4,
        dim=8,
        seq_lengths=Index(1024, 512, 4032, 1),
        width=3,
        ctx=ctx,
    )


def test_varlen_causal_conv1d_fwd_gpu_seqparallel_nonzero_initial_state() raises:
    """Long prefill sequences with a non-zero initial conv_states pool (the
    `has_initial_state`-gated read path), at the narrowest and widest width."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2,
        dim=8,
        seq_lengths=Index(1024, 257),
        width=2,
        ctx=ctx,
        nonzero_initial_state=True,
    )
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2,
        dim=8,
        seq_lengths=Index(4032, 256),
        width=4,
        ctx=ctx,
        nonzero_initial_state=True,
    )


def test_varlen_causal_conv1d_fwd_gpu_chunk_shorter_than_width() raises:
    """Chunks too short to supply the whole new state, with a state to continue.

    The regime every decode step is in, and the one the long-sequence
    `nonzero_initial_state` cases above never reach: with `seqlen < width - 1`
    the pool's older entries can only come from the state being continued.
    `Index(1, 1)` is pure decode (the serial kernel in production);
    `Index(2, 2)` keeps `total_seqlen > batch`, so it is the seq-parallel
    kernel's version of the same partial write.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2,
        dim=8,
        seq_lengths=Index(1, 1),
        width=4,
        ctx=ctx,
        nonzero_initial_state=True,
    )
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=2,
        dim=8,
        seq_lengths=Index(2, 2),
        width=4,
        ctx=ctx,
        nonzero_initial_state=True,
    )
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=3,
        dim=8,
        seq_lengths=Index(1, 1, 1),
        width=3,
        ctx=ctx,
        nonzero_initial_state=True,
    )


def test_varlen_causal_conv1d_fwd_gpu_seqparallel_with_silu() raises:
    """The seq-parallel kernel's silu path over multi-tile sequences.

    The activation is a production path, not a hypothetical: Nemotron-H calls
    this kernel with `activation="silu"` while Inkling passes `"none"`, and the
    single-tile `with_silu` case above is the only other silu coverage."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "silu"](
        batch=3,
        dim=8,
        seq_lengths=Index(512, 255, 1024),
        width=4,
        ctx=ctx,
        nonzero_initial_state=True,
    )


def test_varlen_causal_conv1d_fwd_gpu_seqparallel_zero_length_sequence() raises:
    """A zero-length sequence packed alongside a long prefill sequence must
    still reach the tail-tile epilogue (conv_states zeroing) rather than
    early-returning out of every z-tile block."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_fwd_gpu[.float32, "none"](
        batch=3,
        dim=8,
        seq_lengths=Index(0, 1024, 0),
        width=3,
        ctx=ctx,
    )


# =============================================================================
# Test functions for varlen causal conv1d update on GPU
# =============================================================================


def test_varlen_causal_conv1d_update_gpu_basic() raises:
    """Test basic varlen causal conv1d update on GPU."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[.float32, "none"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4, ctx=ctx
    )


def test_varlen_causal_conv1d_update_gpu_with_silu() raises:
    """Test varlen causal conv1d update GPU with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[.float32, "silu"](
        batch=2, dim=4, seqlen=1, width=3, state_len=4, ctx=ctx
    )


def test_varlen_causal_conv1d_update_gpu_seqlen_gt_1() raises:
    """Test varlen causal conv1d update GPU with seqlen > 1."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[.float32, "none"](
        batch=2, dim=4, seqlen=4, width=3, state_len=4, ctx=ctx
    )


def test_varlen_causal_conv1d_update_gpu_various_widths() raises:
    """Test varlen causal conv1d update GPU with various kernel widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_varlen_causal_conv1d_update_gpu[.float32, "none"](
        batch=2, dim=4, seqlen=1, width=2, state_len=3, ctx=ctx
    )
    run_varlen_causal_conv1d_update_gpu[.float32, "none"](
        batch=2, dim=4, seqlen=1, width=4, state_len=5, ctx=ctx
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
