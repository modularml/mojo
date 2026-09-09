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
"""SM100 Q-slice 3D conv direct-dispatch tests.

These tests invoke `dispatch_qslice_conv3d_sm100` directly, bypassing
the runtime `_is_sm10x_gpu` guard in `conv_gpu`. That makes the test
file itself SM100-only: the dispatcher pulls in `dispatch_sm100_conv2d`,
which instantiates tcgen05 kernels that only compile for sm_100a /
sm_101a. Bazel gates this file to b200_gpu so it never compiles for
H100 or other non-Blackwell targets.

Shape coverage matches the WAN VAE Phase B profile — mid_res 3x3x3,
time_conv 3x1x1, and the C_out F-padding paths (64/192/320).
"""

from std.random import rand
from std.sys import align_of

from layout import (
    LTToTTLayout,
    Layout,
    LayoutTensor,
    TileTensor,
)
from max.gpu.host import DeviceContext
from nn.conv.conv import Naive2dConvolution
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from nn.conv.gpu.nvidia.sm100.qslice_conv3d import (
    dispatch_qslice_conv3d_sm100,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_conv3d_qslice_direct[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    pad_h: Int,
    pad_w: Int,
    with_epilogue: Bool,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
](ctx: DeviceContext) raises:
    """Drive `dispatch_qslice_conv3d_sm100` directly on a WAN-eligible
    3D conv shape (bf16, stride=1, dilation=1, groups=1, zero temporal
    padding, C_in%64==0, C_out%64==0). Validates correctness against
    the CPU naive reference. When C_out is 64-aligned but not
    128-aligned, the dispatcher's F-padding path is exercised. With
    `with_epilogue=True`, also covers the epilogue that fires on the
    final fp32→dtype cast.
    """
    print(
        "test_conv3d_qslice_direct: with_epilogue=",
        with_epilogue,
        " pad_h=",
        pad_h,
        " pad_w=",
        pad_w,
    )
    comptime N = Int(input_layout.shape[0])
    comptime D = Int(input_layout.shape[1])
    comptime H = Int(input_layout.shape[2])
    comptime W = Int(input_layout.shape[3])
    comptime C = Int(input_layout.shape[4])

    comptime Q = Int(filter_layout.shape[0])
    comptime R = Int(filter_layout.shape[1])
    comptime S = Int(filter_layout.shape[2])
    comptime F = Int(filter_layout.shape[4])

    comptime stride = IndexList[3](1, 1, 1)
    comptime dilation = IndexList[3](1, 1, 1)
    comptime pad = IndexList[3](0, pad_h, pad_w)

    comptime D_out = D - Q + 1
    comptime H_out = H + 2 * pad_h - R + 1
    comptime W_out = W + 2 * pad_w - S + 1

    comptime input_size = input_layout.size()
    comptime filter_size = filter_layout.size()
    comptime output_layout_ = Layout.row_major(N, D_out, H_out, W_out, F)
    comptime output_size = output_layout_.size()

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, D_out, H_out, W_out, F),
        Index(N, D, H, W, C),
        Index(Q, R, S, C, F),
        IndexList[2](0, 0),
        IndexList[2](pad_h, pad_h),
        IndexList[2](pad_w, pad_w),
        stride,
        dilation,
        1,
    )
    var scale = Scalar[accum_dtype](2.0) if with_epilogue else Scalar[
        accum_dtype
    ](1.0)
    for i in range(output_size):
        output_ref_host[i] = (output_ref_accum_host[i] * scale).cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    var output_lt = LayoutTensor[dtype, output_layout_](output_dev.unsafe_ptr())
    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), LTToTTLayout[input_layout]()
    )
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), LTToTTLayout[filter_layout]()
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), LTToTTLayout[output_layout_]()
    )

    comptime if with_epilogue:

        @__parameter
        @always_inline
        @__copy_capture(output_lt)
        def scale_epilogue[
            _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            var scaled = (val.cast[.float32]() * 2.0).cast[dtype]()
            output_lt.store[
                width=_width, store_alignment=align_of[dtype]() * _alignment
            ](
                rebind[IndexList[5]](coords),
                rebind[SIMD[dtype, _width]](scaled),
            )

        var handled = dispatch_qslice_conv3d_sm100[
            maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
                scale_epilogue
            ),
        ](
            input_tt,
            filter_tt,
            output_tt,
            stride,
            dilation,
            pad,
            1,
            ctx,
        )
        if not handled:
            print("SKIP: qslice dispatcher declined this shape")
            _ = input_dev^
            _ = filter_dev^
            _ = output_dev^
            return
    else:
        var handled = dispatch_qslice_conv3d_sm100(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            dilation,
            pad,
            1,
            ctx,
        )
        if not handled:
            print("SKIP: qslice dispatcher declined this shape")
            _ = input_dev^
            _ = filter_dev^
            _ = output_dev^
            return

    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    try:
        for i in range(output_size):
            assert_almost_equal(
                output_ref_host[i], output_gpu_host[i], rtol=rtol, atol=atol
            )
        print("RESULT: PASS - All elements match within tolerance")
    except e:
        print("RESULT: FAIL - ", String(e))
        raise e^
    finally:
        _ = input_dev^
        _ = filter_dev^
        _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Q=2 mid_res shape (small D) via fp32 accumulator.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 2, 32, 32, 384),
            Layout.row_major(2, 3, 3, 384, 384),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Q=3 mid_res shape (D_out=1).
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 32, 32, 384),
            Layout.row_major(3, 3, 3, 384, 384),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 32, 32, 384),
            Layout.row_major(3, 3, 3, 384, 384),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Q=3 time_conv 3x1x1 shape.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 16, 16, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            pad_h=0,
            pad_w=0,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Q=3 with D_out=4, exercises batch>1 per 2D call.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 6, 16, 16, 384),
            Layout.row_major(3, 3, 3, 384, 384),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Q=3 C_out=192 (64-aligned, not 128-aligned) — exercises the
        # F-padding path (Phase C). C_out pads to 256 internally; the
        # final write strides back to 192-wide user output.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 16, 16, 192),
            Layout.row_major(3, 3, 3, 192, 192),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 16, 16, 192),
            Layout.row_major(3, 3, 3, 192, 192),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # C_out=64 — smallest valid F-padding target (64 pads to 128).
        # Exercises the one-tile-wide case.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 16, 16, 128),
            Layout.row_major(3, 3, 3, 128, 64),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # C_out=320 — non-power-of-2 padded target (320 pads to 384).
        # Exercises the 3-tile-wide F-padded write.
        test_conv3d_qslice_direct[
            Layout.row_major(1, 3, 16, 16, 128),
            Layout.row_major(3, 3, 3, 128, 320),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # batch=2 with F-padding — covers the outer batch loop's
        # per-n accumulator offset math under padded_run.
        test_conv3d_qslice_direct[
            Layout.row_major(2, 3, 16, 16, 192),
            Layout.row_major(3, 3, 3, 192, 192),
            DType.bfloat16,
            pad_h=1,
            pad_w=1,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)
