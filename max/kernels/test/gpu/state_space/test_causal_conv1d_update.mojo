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
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_cpu_no_bias,
    causal_conv1d_update_gpu,
    causal_conv1d_update_gpu_no_bias,
)
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@always_inline
def silu_ref[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = Float32(1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


def run_causal_conv1d_update_gpu[
    dtype: DType,
    has_bias: Bool,
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
    """Test causal conv1d update GPU kernel against CPU reference."""
    # Allocate host memory
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    # Input x: (B, C, L)
    var input_heap = ctx.enqueue_create_host_buffer[dtype](batch * dim * seqlen)
    var input_h = LayoutTensor[dtype, layout_3d, _](
        input_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Conv state: (B, C, S)
    var conv_state_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_h = LayoutTensor[dtype, layout_3d, _](
        conv_state_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Weight: (C, W)
    var weight_heap = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var weight_h = LayoutTensor[dtype, layout_2d, _](
        weight_heap, RuntimeLayout[layout_2d].row_major(Index(dim, width))
    )

    # Bias: (C,)
    var bias_heap = ctx.enqueue_create_host_buffer[dtype](dim)
    var bias_h = LayoutTensor[dtype, layout_1d, _](
        bias_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    # Output: (B, C, L)
    var result_gpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_gpu_h = LayoutTensor[dtype, layout_3d, _](
        result_gpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var result_cpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * seqlen
    )
    var result_cpu_h = LayoutTensor[dtype, layout_3d, _](
        result_cpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    # Copy of conv_state for CPU reference
    var conv_state_cpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_cpu_h = LayoutTensor[dtype, layout_3d, _](
        conv_state_cpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Copy of conv_state for GPU
    var conv_state_gpu_heap = ctx.enqueue_create_host_buffer[dtype](
        batch * dim * state_len
    )
    var conv_state_gpu_h = LayoutTensor[dtype, layout_3d, _](
        conv_state_gpu_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, state_len)),
    )

    # Initialize input data
    rand[dtype](input_h.ptr, input_h.size())
    rand[dtype](conv_state_h.ptr, conv_state_h.size())
    rand[dtype](weight_h.ptr, weight_h.size())
    rand[dtype](bias_h.ptr, bias_h.size())

    # Copy conv_state for CPU and GPU
    for i in range(batch * dim * state_len):
        conv_state_cpu_h.ptr[i] = conv_state_h.ptr[i]
        conv_state_gpu_h.ptr[i] = conv_state_h.ptr[i]

    var input_buf = input_h
    var conv_state_cpu_buf = conv_state_cpu_h
    var conv_state_gpu_buf = conv_state_gpu_h
    var weight_buf = weight_h
    var bias_buf = bias_h
    var result_gpu_buf = result_gpu_h
    var result_cpu_buf = result_cpu_h

    # Strides for channel-first layout (B, C, L)
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_c_stride: UInt32 = UInt32(seqlen)
    var x_l_stride: UInt32 = 1

    var conv_state_batch_stride: UInt32 = UInt32(dim * state_len)
    var conv_state_c_stride: UInt32 = UInt32(state_len)
    var conv_state_l_stride: UInt32 = 1

    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1

    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_c_stride: UInt32 = UInt32(seqlen)
    var out_l_stride: UInt32 = 1

    var silu_activation = activation == "silu"
    var silu_activation_int8 = Int8(silu_activation)

    # Allocate device buffers
    var input_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var conv_state_device = ctx.enqueue_create_buffer[dtype](
        batch * dim * state_len
    )
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    # Copy data to device
    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_buf.ptr)
        ctx.enqueue_copy(conv_state_device, conv_state_gpu_buf.ptr)
        ctx.enqueue_copy(weight_device, weight_buf.ptr)
        ctx.enqueue_copy(bias_device, bias_buf.ptr)

    # Create TileTensors for GPU kernel
    var input_device_tt = TileTensor(
        input_device,
        row_major(batch, dim, seqlen),
    )
    var conv_state_device_tt = TileTensor(
        conv_state_device,
        row_major(batch, dim, state_len),
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
    var output_device_tt = TileTensor(
        output_device,
        row_major(batch, dim, seqlen),
    )

    # Run GPU kernel
    comptime kNThreads = 128
    with ctx.push_context():
        if has_bias:
            var compiled_func = ctx.compile_function[
                causal_conv1d_update_gpu[
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    input_device_tt.LayoutType,
                    conv_state_device_tt.LayoutType,
                    weight_device_tt.LayoutType,
                    output_device_tt.LayoutType,
                    bias_device_tt.LayoutType,
                    input_device_tt.Engine,
                    conv_state_device_tt.Engine,
                    weight_device_tt.Engine,
                    output_device_tt.Engine,
                    bias_device_tt.Engine,
                ]
            ]()
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                Int32(state_len),
                input_device_tt,
                conv_state_device_tt,
                weight_device_tt,
                output_device_tt,
                bias_device_tt,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8,
                grid_dim=(batch, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )
        else:
            var compiled_func = ctx.compile_function[
                causal_conv1d_update_gpu_no_bias[
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    kNThreads,
                    input_device_tt.LayoutType,
                    conv_state_device_tt.LayoutType,
                    weight_device_tt.LayoutType,
                    output_device_tt.LayoutType,
                    input_device_tt.Engine,
                    conv_state_device_tt.Engine,
                    weight_device_tt.Engine,
                    output_device_tt.Engine,
                ]
            ]()
            ctx.enqueue_function(
                compiled_func,
                Int32(batch),
                Int32(dim),
                Int32(seqlen),
                Int32(width),
                Int32(state_len),
                input_device_tt,
                conv_state_device_tt,
                weight_device_tt,
                output_device_tt,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation_int8,
                grid_dim=(batch, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )

    # Copy results back from device
    with ctx.push_context():
        ctx.enqueue_copy(result_gpu_buf.ptr, output_device)
        ctx.enqueue_copy(conv_state_gpu_buf.ptr, conv_state_device)
    ctx.synchronize()

    # Create TileTensors for CPU reference
    var input_tt = TileTensor(input_buf.ptr, row_major(batch, dim, seqlen))
    var conv_state_cpu_tt = TileTensor(
        conv_state_cpu_buf.ptr,
        row_major(batch, dim, state_len),
    )
    var weight_tt = TileTensor(weight_buf.ptr, row_major(dim, width))
    var bias_tt = TileTensor(
        bias_buf.ptr,
        row_major(
            dim,
        ),
    )
    var result_cpu_tt = TileTensor(
        result_cpu_buf.ptr, row_major(batch, dim, seqlen)
    )

    # Run CPU reference
    if has_bias:
        causal_conv1d_update_cpu[
            dtype,
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_tt,
            conv_state_cpu_tt,
            weight_tt,
            result_cpu_tt,
            bias_tt,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )
    else:
        causal_conv1d_update_cpu_no_bias[
            dtype,
            dtype,
            dtype,
            dtype,
        ](
            batch,
            dim,
            seqlen,
            width,
            state_len,
            input_tt,
            conv_state_cpu_tt,
            weight_tt,
            result_cpu_tt,
            x_batch_stride,
            x_c_stride,
            x_l_stride,
            conv_state_batch_stride,
            conv_state_c_stride,
            conv_state_l_stride,
            weight_c_stride,
            weight_width_stride,
            out_batch_stride,
            out_c_stride,
            out_l_stride,
            silu_activation,
        )

    # Compare results
    var flattened_size = batch * dim * seqlen
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h.ptr[i],
            result_cpu_h.ptr[i],
            rtol=rtol,
        )

    # Compare conv_state updates
    var conv_state_size = batch * dim * state_len
    for i in range(conv_state_size):
        assert_almost_equal(
            conv_state_gpu_h.ptr[i],
            conv_state_cpu_h.ptr[i],
            rtol=rtol,
        )


def test_gpu_causal_conv1d_update_basic() raises:
    """Test basic GPU causal conv1d update with bias."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_with_silu() raises:
    """Test GPU causal conv1d update with SiLU activation."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_update_gpu[.float32, True, "silu"](2, 8, 1, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_without_bias() raises:
    """Test GPU causal conv1d update without bias."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_update_gpu[.float32, False, "none"](
        2, 8, 1, 3, 4, ctx=ctx
    )


def test_gpu_causal_conv1d_update_seqlen_greater_than_one() raises:
    """Test GPU causal conv1d update with seqlen > 1."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 4, 3, 4, ctx=ctx)


def test_gpu_causal_conv1d_update_various_widths() raises:
    """Test GPU causal conv1d update with various kernel widths."""
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 2, 3, ctx=ctx)
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 3, 4, ctx=ctx)
    run_causal_conv1d_update_gpu[.float32, True, "none"](2, 8, 1, 4, 5, ctx=ctx)
