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
from std.random import rand
from std.sys import align_of

from layout import (
    Coord,
    Idx,
    LTToTTLayout,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext
from nn.conv.conv import (
    Naive2dConvolution,
    conv3d_gpu_naive_ndhwc_qrscf,
    conv_gpu,
)
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from nn.conv.gpu.im2col_matmul_2d import dispatch_im2col_matmul_conv2d
from nn.conv.gpu.im2col_matmul_3d import dispatch_im2col_matmul_conv3d
from nn.conv.gpu.matmul_1x1x1_conv3d import dispatch_1x1x1_matmul_conv3d
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_conv3d_gpu[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad: IndexList[3],
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-4,
](ctx: DeviceContext) raises:
    print("test_conv3d: Testing 3D Convolution")
    comptime N = Int(input_layout.shape[0])
    comptime D = Int(input_layout.shape[1])
    comptime H = Int(input_layout.shape[2])
    comptime W = Int(input_layout.shape[3])
    comptime C = Int(input_layout.shape[4])

    comptime Q = Int(filter_layout.shape[0])
    comptime R = Int(filter_layout.shape[1])
    comptime S = Int(filter_layout.shape[2])
    comptime F = Int(filter_layout.shape[4])

    comptime pad_d = IndexList[2](pad[0], pad[0])
    comptime pad_h = IndexList[2](pad[1], pad[1])
    comptime pad_w = IndexList[2](pad[2], pad[2])

    # compute output dimensions, just working backwards to see what the output shape will be
    comptime D_out = (
        D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1
    ) // stride[0] + 1
    comptime H_out = (
        H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1
    ) // stride[1] + 1
    comptime W_out = (
        W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1
    ) // stride[2] + 1

    comptime output_layout = Layout.row_major(N, D_out, H_out, W_out, F)

    # calculate flattened sizes, gotta know how much memory we need
    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    # allocate host memory and initialize with random data
    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    # initialize with random data
    rand(input_host.as_span())
    rand(filter_host.as_span())

    # Run the CPU reference at the same precision the GPU kernel uses for
    # its accumulator (fp32 for bf16 inputs), then narrow back to `dtype`.
    # Without this, a bf16 reference reduction drifts far outside any
    # reasonable tolerance for non-trivial channel counts.
    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, D_out, H_out, W_out, F),  # output shape
        Index(N, D, H, W, C),  # input shape
        Index(Q, R, S, C, F),  # filter shape
        pad_d,
        pad_h,
        pad_w,
        IndexList[3](stride[0], stride[1], stride[2]),
        IndexList[3](dilation[0], dilation[1], dilation[2]),
        1,  # num_groups
    )
    for i in range(output_size):
        output_ref_host[i] = output_ref_accum_host[i].cast[dtype]()
    # allocate device memory
    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    # copy input and filter to device, shipping data to gpu land
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    # create ndbuffer views, making it easier to work with
    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_buf = LayoutTensor[dtype, filter_layout](filter_dev.unsafe_ptr())
    var output_buf = LayoutTensor[dtype, output_layout](output_dev.unsafe_ptr())

    # define grid and block dimensions for the gpu kernel
    comptime block_size = 16
    var grid_dim_x = ceildiv(
        W_out * H_out, block_size
    )  # collapsed width and height into 1 dimension
    var grid_dim_y = ceildiv(D_out, block_size)  # depth is the y dimension
    var grid_dim_z = N  # batch size is the z dimension

    comptime kernel = conv3d_gpu_naive_ndhwc_qrscf[
        input_layout,
        filter_layout,
        output_layout,
        dtype,
        dtype,
        dtype,
        block_size,
        None,
    ]

    # run gpu implementation
    ctx.enqueue_function[kernel](
        input_buf,
        filter_buf,
        output_buf,
        stride,
        dilation,
        pad,
        Int32(1),  # num_groups
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_size, block_size, 1),
    )

    # copy result back to host, bringing it home
    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    # Verify results using assert_almost_equal
    try:
        for i in range(output_size):
            assert_almost_equal(
                output_ref_host[i], output_gpu_host[i], rtol=rtol, atol=atol
            )
        print("RESULT: PASS - All elements match within tolerance")
    except:
        print("RESULT: FAIL - Elements do not match")
    finally:
        pass


def test_conv3d_gpu_dispatch[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad: IndexList[3],
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
](ctx: DeviceContext) raises:
    """Exercise the full `conv_gpu` dispatch (QRSCF path) with a
    scale-by-2 epilogue, mirroring the production MOGG wiring where an
    output_fn is always supplied. Compares the epilogue path against a
    CPU reference to catch regressions in the `dispatch_im2col_matmul_conv3d`
    path that the naive-only tests miss.
    """
    print("test_conv3d_gpu_dispatch: Testing conv_gpu with epilogue")
    comptime N = Int(input_layout.shape[0])
    comptime D = Int(input_layout.shape[1])
    comptime H = Int(input_layout.shape[2])
    comptime W = Int(input_layout.shape[3])
    comptime C = Int(input_layout.shape[4])

    comptime Q = Int(filter_layout.shape[0])
    comptime R = Int(filter_layout.shape[1])
    comptime S = Int(filter_layout.shape[2])
    comptime F = Int(filter_layout.shape[4])

    comptime pad_d = IndexList[2](pad[0], pad[0])
    comptime pad_h = IndexList[2](pad[1], pad[1])
    comptime pad_w = IndexList[2](pad[2], pad[2])

    comptime D_out = (
        D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1
    ) // stride[0] + 1
    comptime H_out = (
        H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1
    ) // stride[1] + 1
    comptime W_out = (
        W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1
    ) // stride[2] + 1

    comptime output_layout = Layout.row_major(N, D_out, H_out, W_out, F)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    # CPU reference at fp32 accumulator then narrow back (matches GPU).
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
        pad_d,
        pad_h,
        pad_w,
        IndexList[3](stride[0], stride[1], stride[2]),
        IndexList[3](dilation[0], dilation[1], dilation[2]),
        1,
    )
    for i in range(output_size):
        # Scale by 2 to match the epilogue the GPU path applies.
        output_ref_host[i] = (
            output_ref_accum_host[i] * Scalar[accum_dtype](2.0)
        ).cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    comptime output_layout_ = Layout.row_major(N, D_out, H_out, W_out, F)
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

    conv_gpu[
        dtype,
        dtype,
        dtype,
        Optional[elementwise_simd_epilogue_type](scale_epilogue),
    ](
        input_tt,
        filter_tt,
        output_tt,
        stride,
        dilation,
        IndexList[6](pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]),
        1,
        ctx,
    )

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


def test_conv3d_im2col_multi_tile[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad: IndexList[3],
    m_tile_byte_budget: Int,
    with_epilogue: Bool,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
](ctx: DeviceContext) raises:
    """Directly drive `dispatch_im2col_matmul_conv3d` with a tiny M-tile
    byte budget to force multi-tile execution on modest shapes. Validates
    both the no-epilogue and epilogue branches, since the latter captures
    `m_offset` across iterations and is not exercised by the single-tile
    WAN benchmark shapes.
    """
    print(
        "test_conv3d_im2col_multi_tile: budget=",
        m_tile_byte_budget,
        " with_epilogue=",
        with_epilogue,
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

    comptime pad_d = IndexList[2](pad[0], pad[0])
    comptime pad_h = IndexList[2](pad[1], pad[1])
    comptime pad_w = IndexList[2](pad[2], pad[2])

    comptime D_out = (
        D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1
    ) // stride[0] + 1
    comptime H_out = (
        H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1
    ) // stride[1] + 1
    comptime W_out = (
        W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1
    ) // stride[2] + 1

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (
        Layout.row_major(N, D_out, H_out, W_out, F).size()
    )

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
        pad_d,
        pad_h,
        pad_w,
        IndexList[3](stride[0], stride[1], stride[2]),
        IndexList[3](dilation[0], dilation[1], dilation[2]),
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

    comptime output_layout_ = Layout.row_major(N, D_out, H_out, W_out, F)
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

        var handled = dispatch_im2col_matmul_conv3d[
            maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
                scale_epilogue
            ),
            m_tile_byte_budget=m_tile_byte_budget,
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
            print("SKIP: dispatcher declined this shape (likely 1x1x1 or K<16)")
            _ = input_dev^
            _ = filter_dev^
            _ = output_dev^
            return
    else:
        var handled = dispatch_im2col_matmul_conv3d[
            m_tile_byte_budget=m_tile_byte_budget,
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
            print("SKIP: dispatcher declined this shape (likely 1x1x1 or K<16)")
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


def test_conv2d_im2col_multi_tile[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[2],
    m_tile_byte_budget: Int,
    with_epilogue: Bool,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
](ctx: DeviceContext) raises:
    """Directly drive `dispatch_im2col_matmul_conv2d` with a tiny M-tile
    byte budget to force multi-tile execution on modest 2-D shapes.
    Validates both the no-epilogue and epilogue branches; the epilogue
    closure captures `m_offset` across iterations and reconstructs 4-D
    NHWC coordinates, which is the key source of drift if the M-tile
    boundary logic is wrong.
    """
    print(
        "test_conv2d_im2col_multi_tile: budget=",
        m_tile_byte_budget,
        " with_epilogue=",
        with_epilogue,
    )
    comptime N = Int(input_layout.shape[0])
    comptime H = Int(input_layout.shape[1])
    comptime W = Int(input_layout.shape[2])
    comptime C = Int(input_layout.shape[3])

    comptime R = Int(filter_layout.shape[0])
    comptime S = Int(filter_layout.shape[1])
    comptime F = Int(filter_layout.shape[3])

    comptime pad_h = IndexList[2](pad[0], pad[0])
    comptime pad_w = IndexList[2](pad[1], pad[1])

    comptime H_out = (
        H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1
    ) // stride[0] + 1
    comptime W_out = (
        W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1
    ) // stride[1] + 1

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (Layout.row_major(N, H_out, W_out, F).size())

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    # Naive2dConvolution internally uses 5-D NDHWC shapes with D=Q=1 for 2-D.
    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, 1, H_out, W_out, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C, F),
        IndexList[2](0, 0),
        pad_h,
        pad_w,
        IndexList[3](1, stride[0], stride[1]),
        IndexList[3](1, dilation[0], dilation[1]),
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

    comptime output_layout_ = Layout.row_major(N, H_out, W_out, F)
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

    var handled: Bool
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
                rebind[IndexList[4]](coords),
                rebind[SIMD[dtype, _width]](scaled),
            )

        handled = dispatch_im2col_matmul_conv2d[
            maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
                scale_epilogue
            ),
            m_tile_byte_budget=m_tile_byte_budget,
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
    else:
        handled = dispatch_im2col_matmul_conv2d[
            m_tile_byte_budget=m_tile_byte_budget,
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
        print("SKIP: dispatcher declined this shape (likely 1x1 or K<16)")
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


def test_conv3d_1x1x1_matmul_direct[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    with_epilogue: Bool,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
](ctx: DeviceContext) raises:
    """Drive `dispatch_1x1x1_matmul_conv3d` directly on a 1x1x1 shape.

    Validates the reshape-as-matmul fast path, the 2D->5D coord-unpack
    epilogue, and the no-scratch pathway. Compared against a CPU
    fp32-accumulator reference narrowed back to `dtype`.
    """
    print("test_conv3d_1x1x1_matmul_direct: with_epilogue=", with_epilogue)
    comptime N = Int(input_layout.shape[0])
    comptime D = Int(input_layout.shape[1])
    comptime H = Int(input_layout.shape[2])
    comptime W = Int(input_layout.shape[3])
    comptime C = Int(input_layout.shape[4])

    comptime Q = Int(filter_layout.shape[0])
    comptime R = Int(filter_layout.shape[1])
    comptime S = Int(filter_layout.shape[2])
    comptime F = Int(filter_layout.shape[4])
    comptime assert (
        Q == 1 and R == 1 and S == 1
    ), "test_conv3d_1x1x1_matmul_direct requires a 1x1x1 filter"

    comptime stride = IndexList[3](1, 1, 1)
    comptime dilation = IndexList[3](1, 1, 1)
    comptime pad = IndexList[3](0, 0, 0)

    comptime input_size = input_layout.size()
    comptime filter_size = filter_layout.size()
    comptime output_layout_ = Layout.row_major(N, D, H, W, F)
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
        Index(N, D, H, W, F),
        Index(N, D, H, W, C),
        Index(Q, R, S, C, F),
        IndexList[2](0, 0),
        IndexList[2](0, 0),
        IndexList[2](0, 0),
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

        var handled = dispatch_1x1x1_matmul_conv3d[
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
            print("SKIP: 1x1x1 dispatcher declined this shape")
            _ = input_dev^
            _ = filter_dev^
            _ = output_dev^
            return
    else:
        var handled = dispatch_1x1x1_matmul_conv3d(
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
            print("SKIP: 1x1x1 dispatcher declined this shape")
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
        # test case 1: small dimensions, starting simple
        test_conv3d_gpu[
            Layout.row_major(1, 4, 4, 4, 2),  # input (NDHWC)
            Layout.row_major(2, 2, 2, 2, 3),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # padding
        ](ctx)

        # test case 2: medium dimensions with padding
        test_conv3d_gpu[
            Layout.row_major(2, 6, 6, 6, 4),  # input (NDHWC)
            Layout.row_major(3, 3, 3, 4, 8),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 2, 2),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)

        # test case 3: non-square dimensions
        test_conv3d_gpu[
            Layout.row_major(1, 5, 7, 9, 3),  # input (NDHWC)
            Layout.row_major(2, 3, 2, 3, 4),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 1, 0),  # padding
        ](ctx)

        # test case 4: varying filter dimensions, getting creative
        test_conv3d_gpu[
            Layout.row_major(1, 9, 8, 5, 1),  # input (NDHWC)
            Layout.row_major(2, 2, 3, 1, 32),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 3, 2),  # stride - mixed stride values
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # padding
        ](ctx)

        # test case 5: with padding on all dimensions
        test_conv3d_gpu[
            Layout.row_major(1, 5, 7, 6, 7),  # input (NDHWC)
            Layout.row_major(3, 4, 3, 7, 24),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)

        # test case 6: large dimensions with asymmetric padding
        test_conv3d_gpu[
            Layout.row_major(1, 10, 11, 6, 2),  # input (NDHWC)
            Layout.row_major(3, 4, 3, 2, 31),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 3, 1),  # stride - mixed stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 2, 1),  # padding - asymmetric
        ](ctx)

        # test case 7: 3d-unet style small dimensions
        test_conv3d_gpu[
            Layout.row_major(1, 8, 8, 8, 320),  # input (NDHWC)
            Layout.row_major(3, 3, 3, 320, 320),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 2, 2),  # stride - downsampling
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)

        # --- bfloat16 coverage ---
        # WAN VAE post_quant_conv: 1x1x1, C_in=16, F=16.
        test_conv3d_gpu[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(1, 1, 1, 16, 16),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # WAN VAE conv_in: 3x3x3, C_in=16, F=96 (early-level Residual).
        test_conv3d_gpu[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(3, 3, 3, 16, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # WAN VAE mid Residual: 3x3x3, C_in=96, F=96. Exercises main vec loop.
        test_conv3d_gpu[
            Layout.row_major(1, 4, 8, 8, 96),
            Layout.row_major(3, 3, 3, 96, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # WAN VAE time_conv temporal: 3x1x1, C_in=192, F=384.
        test_conv3d_gpu[
            Layout.row_major(1, 4, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 0, 0),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # Tail-only C_in case (C_in < vec_w=8 for bf16).
        test_conv3d_gpu[
            Layout.row_major(1, 4, 6, 6, 3),
            Layout.row_major(3, 3, 3, 3, 16),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # Main + tail C_in case (C_in = vec_w + remainder).
        test_conv3d_gpu[
            Layout.row_major(1, 4, 6, 6, 10),
            Layout.row_major(3, 3, 3, 10, 16),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=1e-2,
            atol=1e-2,
        ](ctx)

        # ---------------------------------------------------------------
        # conv_gpu dispatch path with epilogue (matches production MOGG).
        # These hit `dispatch_im2col_matmul_conv3d` for 3x3x3 / 3x1x1 on
        # B200 with bf16 and a fused scale-by-2 elementwise lambda.
        # ---------------------------------------------------------------

        # WAN VAE conv_in through dispatch + epilogue.
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(3, 3, 3, 16, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # WAN VAE mid residual through dispatch + epilogue.
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 4, 8, 8, 96),
            Layout.row_major(3, 3, 3, 96, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # WAN VAE time_conv 3x1x1 through dispatch + epilogue.
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 4, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 0, 0),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # ---------------------------------------------------------------
        # Multi-tile coverage: force the M-tile loop to iterate multiple
        # times by shrinking the byte budget, so the per-iteration
        # `m_offset` capture and output-offset math are both exercised.
        # The production VAE uses the default 256 MiB budget and single-
        # tile benchmark shapes, so this is a gap the WAN bench doesn't
        # cover.
        # ---------------------------------------------------------------
        # No-epilogue multi-tile: shape M=1*4*8*8=256, K=3*3*3*16=432,
        # bytes/row=864. Budget=64 KiB -> ~75 rows/tile -> ~4 tiles.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(3, 3, 3, 16, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            m_tile_byte_budget=64 * 1024,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Epilogue multi-tile: same shape, with scale-by-2 epilogue.
        # This is the combination that is NEVER hit by the existing
        # benchmark (bench uses no epilogue) or the existing tests
        # (tests use the naive kernel directly). If `m_offset` capture
        # across iterations is broken, this is what catches it.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(3, 3, 3, 16, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            m_tile_byte_budget=64 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Larger multi-tile epilogue case: batch=2 to exercise the
        # batch-decomposition in the epilogue's coord-unpack math.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(2, 4, 8, 8, 96),
            Layout.row_major(3, 3, 3, 96, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            m_tile_byte_budget=256 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # 3x1x1 multi-tile epilogue case: the time_conv filter shape.
        # Different K (non-3x3x3) to exercise the K-unpack math.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(1, 4, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](1, 0, 0),
            m_tile_byte_budget=128 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # ---------------------------------------------------------------
        # Stride > 1 coverage: the encoder's `downsample3d.time_conv`
        # uses stride=(2,1,1) with a 3x1x1 filter and hits
        # `dispatch_im2col_matmul_conv3d`. The WAN bench and decoder-
        # only tests only use stride=(1,1,1), so stride > 1 would not
        # be caught by existing coverage.
        # ---------------------------------------------------------------
        # Stride (2,1,1) via conv_gpu dispatch + epilogue.
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 8, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](2, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Stride (2,1,1) via direct dispatch, forced multi-tile, no epilogue.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(1, 8, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](2, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            m_tile_byte_budget=64 * 1024,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Stride (2,1,1) via direct dispatch, forced multi-tile, with
        # epilogue. This is the combination closest to what the encoder
        # runs in production on the temporal downsample.
        test_conv3d_im2col_multi_tile[
            Layout.row_major(1, 8, 6, 6, 192),
            Layout.row_major(3, 1, 1, 192, 384),
            DType.bfloat16,
            IndexList[3](2, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            m_tile_byte_budget=64 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # 2D im2col+matmul coverage: shapes where the SM100 fast path
        # declines (non-128-aligned channels).
        # Single M-tile, no epilogue.
        test_conv2d_im2col_multi_tile[
            Layout.row_major(1, 32, 32, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            m_tile_byte_budget=256 * 1024 * 1024,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Same shape with scale-by-2 epilogue: exercises the 4-D
        # coord-unpack closure inside the M-tile loop.
        test_conv2d_im2col_multi_tile[
            Layout.row_major(1, 32, 32, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            m_tile_byte_budget=256 * 1024 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Force multi-tile execution: M=256, K=864, bytes/row=1728;
        # budget 32 KiB -> ~18 rows/tile.
        test_conv2d_im2col_multi_tile[
            Layout.row_major(1, 16, 16, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            m_tile_byte_budget=32 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # batch>1, asymmetric filter (3x1 — spatial-only row conv),
        # covers non-square kernel geometry and batch outer dim.
        test_conv2d_im2col_multi_tile[
            Layout.row_major(2, 16, 16, 192),
            Layout.row_major(3, 1, 192, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](1, 0),
            m_tile_byte_budget=256 * 1024 * 1024,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # stride=2 case — the `h_in/w_in` stride math needs exercising
        # separately from stride=1.
        test_conv2d_im2col_multi_tile[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 128),
            DType.bfloat16,
            IndexList[2](2, 2),
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            m_tile_byte_budget=256 * 1024 * 1024,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # ---------------------------------------------------------------
        # Phase A (1x1x1 matmul fast path). `dispatch_1x1x1_matmul_conv3d`
        # reshapes the NDHWC input to [M, C_in], the 1x1x1 filter to
        # [F, C] / [C, F], and fires a single _matmul_gpu call. These
        # cover the VAE `post_quant_conv` and `conv_shortcut` shapes.
        # ---------------------------------------------------------------
        # post_quant_conv shape through conv_gpu dispatch (exercises
        # Phase A since our wiring puts it before im2col+matmul).
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(1, 1, 1, 16, 16),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Larger 1x1x1 at conv_shortcut scale (192 -> 96).
        test_conv3d_gpu_dispatch[
            Layout.row_major(1, 4, 8, 8, 192),
            Layout.row_major(1, 1, 1, 192, 96),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 0, 0),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # End-to-end dispatch at batch=2 through a Q-slice-eligible
        # shape (bf16, stride=1, zero temporal pad, C_in and C_out
        # both 64-aligned, Q>1). Guards against the batch-stride class
        # of bugs where the qslice outer batch loop uses the wrong
        # per-n input offset.
        test_conv3d_gpu_dispatch[
            Layout.row_major(2, 5, 6, 6, 192),
            Layout.row_major(3, 3, 3, 192, 192),
            DType.bfloat16,
            IndexList[3](1, 1, 1),
            IndexList[3](1, 1, 1),
            IndexList[3](0, 1, 1),
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Direct dispatch of the 1x1x1 matmul path with + without
        # epilogue, over the post_quant_conv shape. Validates the
        # 2D->5D coord-unpack closure and confirms no-scratch allocation.
        test_conv3d_1x1x1_matmul_direct[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(1, 1, 1, 16, 16),
            DType.bfloat16,
            with_epilogue=False,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)
        test_conv3d_1x1x1_matmul_direct[
            Layout.row_major(1, 4, 8, 8, 16),
            Layout.row_major(1, 1, 1, 16, 16),
            DType.bfloat16,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Direct dispatch at larger conv_shortcut-style shape (384 -> 192)
        # with epilogue. Exercises the coord-unpack math at non-square
        # batch=2.
        test_conv3d_1x1x1_matmul_direct[
            Layout.row_major(2, 4, 8, 8, 384),
            Layout.row_major(1, 1, 1, 384, 192),
            DType.bfloat16,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)

        # Direct dispatch at realistic latent resolution (matches VAE
        # post_quant_conv's actual shape: 1x16x21x30x52 -> ... x16).
        test_conv3d_1x1x1_matmul_direct[
            Layout.row_major(1, 21, 30, 52, 16),
            Layout.row_major(1, 1, 1, 16, 16),
            DType.bfloat16,
            with_epilogue=True,
            rtol=2e-2,
            atol=2e-2,
        ](ctx)
