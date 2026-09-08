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
"""SM100 Conv2D Test - Validation of TMA im2col conv2d kernel.

This test file validates the SM100 conv2d fprop kernel using hardware TMA
im2col transformation. The kernel computes C[M,N] = A[M,K] @ B[N,K]^T
which maps to convolution as:
- M = batch * out_h * out_w (output spatial)
- N = out_channels (filters)
- K = in_channels * filter_h * filter_w (reduction)

Test coverage:
- 3x3 and 1x1 convolutions
- 1-SM and 2-SM cluster modes
- Epilogue lambda fusion (bias addition)
- conv_gpu scale/additive epilogues via dispatch path
- Native TMA residual add via conv_gpu dispatch path

Usage:
    bazel test //max/kernels/test/gpu/linalg:test_conv2d_sm100 --config=b200
"""

from std.collections import Optional
from std.sys import align_of
from std.testing import assert_false

import linalg.matmul.vendor.blas as vendor_blas
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils import assert_almost_equal
from nn.conv.conv import conv_gpu
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from std.random import rand
from std.utils.index import IndexList
from nn.conv.gpu.nvidia.sm100.conv2d import (
    conv2d_fprop,
    conv2d_fprop_with_residual,
    im2col,
)
from nn.conv.gpu.nvidia.sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)
from linalg.utils import elementwise_compute_lambda_type


def test_conv2d_implicit_im2col[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d using implicit im2col (4D NHWC API).

    This tests the full conv2d kernel with TMA im2col transformation:
    - Activation: [N, H, W, C] (NHWC)
    - Filter: [K, R, S, C] (KRSC)
    - Output: [N, H_out, W_out, K] (NHWC)

    The reference is computed using explicit im2col + conv2d_fprop_gemm.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    print(
        "[IMPLICIT IM2COL] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = ctx.enqueue_create_host_buffer[act_type](act_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_size
    )
    var out_host_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)
    var out_host_ref_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)

    # TileTensor shapes with dynamic dimensions
    var act_shape = row_major(
        Coord(Int(batch), Int(in_h), Int(in_w), Int(in_c))
    )
    var filter_shape = row_major(
        Coord(
            Int(out_c),
            Int(filter_h),
            Int(filter_w),
            Int(in_c),
        )
    )
    var out_shape = row_major(
        Coord(Int(batch), Int(out_h), Int(out_w), Int(out_c))
    )

    var act_host = TileTensor(act_host_ptr, act_shape)

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = TileTensor(act_device, act_shape)
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = TileTensor(filter_device, filter_shape)
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = TileTensor(out_device, out_shape)

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host_ptr.as_span())
    rand(filter_host_ptr.as_span())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)

    # Run conv2d with implicit im2col
    conv2d_fprop(
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM
    # Allocate im2col buffer [M, K]
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    # Perform im2col on host
    var im2col_host_ptr = ctx.enqueue_create_host_buffer[act_type](im2col_size)
    var im2col_host = TileTensor(
        im2col_host_ptr, row_major(Coord(Int(M), Int(K)))
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    var im2col_device_nd = TileTensor(im2col_device, row_major(M, K))
    var filter_2d_device_nd = TileTensor(filter_device, row_major(N, K))
    var out_2d_ref_nd = TileTensor(out_device_ref, row_major(M, N))

    # Reference: cuBLAS GEMM (transpose_b=True for NK layout)
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr.unsafe_ptr(),
        out_host_ref_ptr.unsafe_ptr(),
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")


def test_conv2d_1sm[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_pipeline_stages_override: Int = 0,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with 1-SM configuration (cta_group=1).

    This uses the same logic as test_conv2d_implicit_im2col but with
    the 1-SM configuration matching the CUTLASS example.

    Parameters:
        act_type: Activation dtype.
        filter_type: Filter dtype.
        out_type: Output dtype.
        swizzle: Activation/filter TMA swizzle. SWIZZLE_128B requires
            C*sizeof(dtype) to be 128B-aligned (e.g. bf16 C=64/128).
            SWIZZLE_64B only requires 64B alignment (e.g. bf16 C=96).
        num_pipeline_stages_override: If > 0, pin num_pipeline_stages to
            this value instead of the config's auto-sizer. Needed for the
            SWIZZLE_64B path (BK=32) because the auto-sizer over-estimates
            the stage budget there (doesn't account for conv-specific
            SMEM like SourceTiles). Default 0 = use auto-sizer.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get 1-SM config (must be comptime for kernel parameters)
    # The config uses the dtype template parameters, so it works for both BF16 and FP16
    comptime config = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16_1sm[
        swizzle=swizzle,
        num_pipeline_stages_override=num_pipeline_stages_override,
    ]()

    print(
        "[1-SM MODE] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ") cta_group=",
        config.cta_group,
        " stages=",
        config.num_pipeline_stages,
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = ctx.enqueue_create_host_buffer[act_type](act_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_size
    )
    var out_host_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)
    var out_host_ref_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)

    # TileTensor shapes with dynamic dimensions
    var act_shape = row_major(
        Coord(Int(batch), Int(in_h), Int(in_w), Int(in_c))
    )
    var filter_shape = row_major(
        Coord(
            Int(out_c),
            Int(filter_h),
            Int(filter_w),
            Int(in_c),
        )
    )
    var out_shape = row_major(
        Coord(Int(batch), Int(out_h), Int(out_w), Int(out_c))
    )

    var act_host = TileTensor(act_host_ptr, act_shape)

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = TileTensor(act_device, act_shape)
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = TileTensor(filter_device, filter_shape)
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = TileTensor(out_device, out_shape)

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host_ptr.as_span())
    rand(filter_host_ptr.as_span())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)

    # Run conv2d with 1-SM config
    conv2d_fprop[config=config](
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    var im2col_host_ptr = ctx.enqueue_create_host_buffer[act_type](im2col_size)
    var im2col_host = TileTensor(
        im2col_host_ptr, row_major(Coord(Int(M), Int(K)))
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    var _dynamic_a_ref_shape = IndexList[2](M, K)
    var _dynamic_b_ref_shape = IndexList[2](N, K)
    var _dynamic_c_ref_shape = IndexList[2](M, N)
    var im2col_device_nd = TileTensor(im2col_device, row_major(M, K))
    var filter_2d_device_nd = TileTensor(filter_device, row_major(N, K))
    var out_2d_ref_nd = TileTensor(out_device_ref, row_major(M, N))

    # Reference: cuBLAS GEMM
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr.unsafe_ptr(),
        out_host_ref_ptr.unsafe_ptr(),
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")


def test_conv2d_epilogue_lambda[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with epilogue lambda for bias addition.

    This tests the epilogue fusion infrastructure by adding a bias tensor
    to the convolution output using an epilogue lambda.

    The operation is: output = conv2d(activation, filter) + bias
    where bias is broadcast over the spatial dimensions.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get config
    comptime config = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16_1sm()

    print(
        "[EPILOGUE LAMBDA] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c
    var bias_size = out_c

    # Host allocations
    var act_host_ptr = ctx.enqueue_create_host_buffer[act_type](act_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[filter_type](
        filter_size
    )
    var out_host_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)
    var out_host_ref_ptr = ctx.enqueue_create_host_buffer[out_type](out_size)
    var bias_host_ptr = ctx.enqueue_create_host_buffer[out_type](bias_size)

    # TileTensor shapes with dynamic dimensions
    var act_shape = row_major(
        Coord(Int(batch), Int(in_h), Int(in_w), Int(in_c))
    )
    var filter_shape = row_major(
        Coord(
            Int(out_c),
            Int(filter_h),
            Int(filter_w),
            Int(in_c),
        )
    )
    var out_shape = row_major(
        Coord(Int(batch), Int(out_h), Int(out_w), Int(out_c))
    )

    var act_host = TileTensor(act_host_ptr, act_shape)

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = TileTensor(act_device, act_shape)
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = TileTensor(filter_device, filter_shape)
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = TileTensor(out_device, out_shape)
    var bias_device = ctx.enqueue_create_buffer[out_type](bias_size)

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host_ptr.as_span())
    rand(filter_host_ptr.as_span())
    rand(bias_host_ptr.as_span())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)
    ctx.enqueue_copy(bias_device, bias_host_ptr)

    # Create bias tensor view for epilogue lambda
    # Bias is 1D [out_c], needs to be broadcast over [M, N] output
    var bias_tensor = TileTensor(bias_device, row_major(out_c))

    # Define epilogue lambda that adds bias (broadcast over M dimension)
    # Output shape is [M, N] where N = out_channels
    # Bias is [N], so we index by idx[1] (the column/channel index)
    @__parameter
    @always_inline
    @__copy_capture(bias_tensor)
    def epilogue_add_bias[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # Load bias value for this channel and broadcast to SIMD width
        # Note: For width > 1, consecutive columns may have different biases
        # so we need to load a vector of biases
        var bias_val = bias_tensor.load[width=width]((idx[1],)).cast[_dtype]()
        return val + bias_val

    # Create optional lambda
    comptime optional_lambda = Optional[elementwise_compute_lambda_type](
        epilogue_add_bias
    )

    # Run conv2d with epilogue lambda
    conv2d_fprop[
        config=config,
        elementwise_compute_lambda_fn=optional_lambda,
    ](
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM (without bias)
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    var im2col_host_ptr = ctx.enqueue_create_host_buffer[act_type](im2col_size)
    var im2col_host = TileTensor(
        im2col_host_ptr, row_major(Coord(Int(M), Int(K)))
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    var im2col_device_nd = TileTensor(im2col_device, row_major(M, K))
    var filter_2d_device_nd = TileTensor(filter_device, row_major(N, K))
    var out_2d_ref_nd = TileTensor(out_device_ref, row_major(M, N))

    # Reference: cuBLAS GEMM
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Apply bias on CPU to reference output
    # Reference output is [M, N], bias is [N]
    for m in range(M):
        for n in range(N):
            var idx = m * N + n
            out_host_ref_ptr[idx] = out_host_ref_ptr[idx] + bias_host_ptr[n]

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr.unsafe_ptr(),
        out_host_ref_ptr.unsafe_ptr(),
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")


def test_conv2d_bias_fusion[
    dtype: DType,
    use_1sm: Bool,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with fused bias addition - typical FLUX pattern.

    This is the most common fusion pattern in neural networks:
    output = conv2d(input, weights) + bias

    The bias is 1D [out_channels] and broadcasts over spatial dimensions.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Select config based on parameter
    comptime if use_1sm:
        comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()
    else:
        comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16()

    var mode_str = "1-SM" if use_1sm else "2-SM"
    print(
        "[CONV+BIAS ",
        mode_str,
        "] ",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        " -> ",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        sep="",
    )

    # Allocate memory
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    var act_host = ctx.enqueue_create_host_buffer[dtype](act_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](out_c)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_ref_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    rand(act_host.as_span())
    rand(filter_host.as_span())
    rand(bias_host.as_span())

    var act_dev = ctx.enqueue_create_buffer[dtype](act_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var bias_dev = ctx.enqueue_create_buffer[dtype](out_c)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var im2col_dev = ctx.enqueue_create_buffer[dtype](M * K)

    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    # Create TileTensors
    var act_shape = row_major(
        Coord(Int(batch), Int(in_h), Int(in_w), Int(in_c))
    )
    var filter_shape = row_major(
        Coord(
            Int(out_c),
            Int(filter_h),
            Int(filter_w),
            Int(in_c),
        )
    )
    var out_shape = row_major(
        Coord(Int(batch), Int(out_h), Int(out_w), Int(out_c))
    )
    var act_nd = TileTensor(act_dev, act_shape)
    var filter_nd = TileTensor(filter_dev, filter_shape)
    var out_nd = TileTensor(out_dev, out_shape)

    # Create bias tensor for capture
    var bias_tensor = TileTensor(bias_dev, row_major(out_c))

    # Epilogue lambda: add bias (idx[1] = channel index in [M, N] output)
    @__parameter
    @always_inline
    @__copy_capture(bias_tensor)
    def add_bias[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + bias_tensor.load[width=width]((idx[1],)).cast[_dtype]()

    comptime bias_lambda = Optional[elementwise_compute_lambda_type](add_bias)

    # Run conv2d with fused bias
    comptime if use_1sm:
        conv2d_fprop[
            config=Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm(),
            elementwise_compute_lambda_fn=bias_lambda,
        ](
            out_nd,
            act_nd,
            filter_nd,
            problem,
            ctx,
        )
    else:
        conv2d_fprop[
            config=Conv2dConfig[dtype, dtype, dtype].default_bf16(),
            elementwise_compute_lambda_fn=bias_lambda,
        ](
            out_nd,
            act_nd,
            filter_nd,
            problem,
            ctx,
        )

    # Reference: im2col + GEMM + bias (CPU bias add)
    var act_host_nd = TileTensor(act_host, act_shape)
    var im2col_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var im2col_host_nd = TileTensor(
        im2col_host, row_major(Coord(Int(M), Int(K)))
    )
    im2col(im2col_host_nd, act_host_nd, problem)
    ctx.enqueue_copy(im2col_dev, im2col_host)

    var im2col_nd = TileTensor(im2col_dev, row_major(M, K))
    var filter_2d_nd = TileTensor(filter_dev, row_major(N, K))
    var out_ref_nd = TileTensor(out_ref_dev, row_major(M, N))

    vendor_blas.matmul(
        ctx,
        out_ref_nd,
        im2col_nd,
        filter_2d_nd,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Copy results
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.synchronize()

    # Apply bias to reference on CPU
    for m in range(M):
        for n in range(N):
            out_ref_host[m * N + n] = out_ref_host[m * N + n] + bias_host[n]

    # Validate
    assert_almost_equal(
        out_host.unsafe_ptr(),
        out_ref_host.unsafe_ptr(),
        out_size,
        atol=0.0001,
        rtol=1e-2,
    )
    print("    PASSED")
    _ = act_dev^
    _ = filter_dev^
    _ = bias_dev^
    _ = out_dev^
    _ = out_ref_dev^
    _ = im2col_dev^


def test_conv2d_residual_api[
    dtype: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d_fprop_with_residual API.

    This tests the residual add: D = Conv(A,B) + beta*C

    Tests:
    1. has_residual=False should fall back to standard conv2d.
    2. beta=0 should fall back to standard conv2d.
    3. Full residual path: validates D = Conv(A,B) + beta*C against
       cuBLAS GEMM reference with host-side residual add.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get 1-SM config
    comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()

    print(
        "[RESIDUAL API] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = ctx.enqueue_create_host_buffer[dtype](act_size)
    var filter_host_ptr = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var out_host_ptr = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](out_size)
    var source_host_ptr = ctx.enqueue_create_host_buffer[dtype](out_size)

    # TileTensor shapes with dynamic dimensions
    var act_shape = row_major(
        Coord(Int(batch), Int(in_h), Int(in_w), Int(in_c))
    )
    var filter_shape = row_major(
        Coord(
            Int(out_c),
            Int(filter_h),
            Int(filter_w),
            Int(in_c),
        )
    )
    var out_shape = row_major(
        Coord(Int(batch), Int(out_h), Int(out_w), Int(out_c))
    )

    var act_host = TileTensor(act_host_ptr, act_shape)

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[dtype](act_size)
    var act_device_nd = TileTensor(act_device, act_shape)
    var filter_device = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_device_nd = TileTensor(filter_device, filter_shape)
    var out_device = ctx.enqueue_create_buffer[dtype](out_size)
    var out_device_nd = TileTensor(out_device, out_shape)
    var source_device = ctx.enqueue_create_buffer[dtype](out_size)
    var source_device_nd = TileTensor(source_device, out_shape)

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[dtype](out_size)

    # Initialize with random data
    rand(act_host_ptr.as_span())
    rand(filter_host_ptr.as_span())
    rand(source_host_ptr.as_span())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)
    ctx.enqueue_copy(source_device, source_host_ptr)

    # Test 1: has_residual=False should fall back to standard conv2d
    print("  Test 1: has_residual=False fallback...")
    conv2d_fprop_with_residual[config=config, has_residual=False](
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        source_device_nd,  # Ignored when has_residual=False
        Float32(1.0),  # Beta (ignored)
        problem,
        ctx,
    )

    # Test 2: beta=0 should fall back to standard conv2d
    print("  Test 2: beta=0 fallback...")
    conv2d_fprop_with_residual[config=config, has_residual=True](
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        source_device_nd,
        Float32(0.0),  # Beta=0 means no residual
        problem,
        ctx,
    )

    # Test 3: source provided with beta!=0
    # Full residual path: D = Conv(A,B) + beta*C
    comptime test_beta = Float32(1.0)
    print("  Test 3: source + beta (residual add)...")
    conv2d_fprop_with_residual[config=config, has_residual=True](
        out_device_nd,
        act_device_nd,
        filter_device_nd,
        source_device_nd,
        test_beta,  # Beta=1.0 for skip connection
        problem,
        ctx,
    )

    # Reference: compute Conv(A,B) via cuBLAS GEMM
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[dtype](im2col_size)

    var im2col_host_ptr = ctx.enqueue_create_host_buffer[dtype](im2col_size)
    var im2col_host = TileTensor(
        im2col_host_ptr, row_major(Coord(Int(M), Int(K)))
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    var im2col_device_nd = TileTensor(im2col_device, row_major(M, K))
    var filter_2d_device_nd = TileTensor(filter_device, row_major(N, K))
    var out_2d_ref_nd = TileTensor(out_device_ref, row_major(M, N))

    # Reference: cuBLAS GEMM (conv2d only)
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Add residual to reference on host: ref = Conv(A,B) + beta * C
    for i in range(out_size):
        out_host_ref_ptr[i] = (
            out_host_ref_ptr[i].cast[.float32]()
            + test_beta * source_host_ptr[i].cast[.float32]()
        ).cast[dtype]()

    # Validate: D = Conv(A,B) + beta*C
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr.unsafe_ptr(),
        out_host_ref_ptr.unsafe_ptr(),
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")


def test_conv2d_problem_shape():
    """Test Conv2dProblemShape computations."""
    print("Testing Conv2dProblemShape...")

    # Test 1: 3x3 conv with padding (common VAE pattern)
    var problem = Conv2dProblemShape(
        batch=1,
        in_height=64,
        in_width=64,
        in_channels=128,
        out_channels=256,
        filter_h=3,
        filter_w=3,
        pad_h=1,
        pad_w=1,
    )

    # With padding=1, stride=1, output size == input size
    var expected_out_h = 64
    var expected_out_w = 64
    if problem.out_height() != expected_out_h:
        print(
            "FAILED: out_height expected",
            expected_out_h,
            "got",
            problem.out_height(),
        )
        return
    if problem.out_width() != expected_out_w:
        print(
            "FAILED: out_width expected",
            expected_out_w,
            "got",
            problem.out_width(),
        )
        return

    # GEMM dimensions
    var expected_m = 1 * 64 * 64  # batch * out_h * out_w = 4096
    var expected_n = 256  # out_channels
    var expected_k = 128 * 3 * 3  # in_channels * R * S = 1152

    if problem.gemm_m() != expected_m:
        print("FAILED: gemm_m expected", expected_m, "got", problem.gemm_m())
        return
    if problem.gemm_n() != expected_n:
        print("FAILED: gemm_n expected", expected_n, "got", problem.gemm_n())
        return
    if problem.gemm_k() != expected_k:
        print("FAILED: gemm_k expected", expected_k, "got", problem.gemm_k())
        return

    print("  Conv2dProblemShape: PASSED\n")

    # Test 2: 1x1 conv (pointwise, no padding)
    var problem2 = Conv2dProblemShape(
        batch=2,
        in_height=32,
        in_width=32,
        in_channels=512,
        out_channels=256,
        filter_h=1,
        filter_w=1,
        pad_h=0,
        pad_w=0,
    )

    # 1x1 conv: output size == input size
    if problem2.out_height() != 32 or problem2.out_width() != 32:
        print("FAILED: 1x1 conv output size incorrect")
        return

    # GEMM: M = 2*32*32 = 2048, N = 256, K = 512*1*1 = 512
    if (
        problem2.gemm_m() != 2048
        or problem2.gemm_n() != 256
        or problem2.gemm_k() != 512
    ):
        print("FAILED: 1x1 conv GEMM dimensions incorrect")
        return

    print("  1x1 Conv: PASSED\n")


# ============================================================
# conv_gpu dispatch-level epilogue and residual tests
# ============================================================


def test_conv_gpu_scale_epilogue[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test conv_gpu with a scale-by-2 epilogue fused into the kernel."""
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1
    comptime dtype = DType.bfloat16
    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print("  ", name, sep="")

    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var out_epilogue_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_ref_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var out_epilogue_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    comptime input_tt_layout = row_major[N, H, W, C_in]()
    comptime filter_tt_layout = row_major[R, S, C_in, C_out]()
    comptime output_tt_layout = row_major[N, Hout, Wout, C_out]()
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_epilogue_tt = TileTensor(out_epilogue_dev, output_tt_layout)
    var out_ref_tt = TileTensor(out_ref_dev, output_tt_layout)

    @__parameter
    @always_inline
    @__copy_capture(out_epilogue_tt)
    def scale_epilogue[
        _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        var scaled = (val.cast[.float32]() * 2.0).cast[dtype]()
        out_epilogue_tt.store[
            width=_width, alignment=align_of[dtype]() * _alignment
        ](
            Coord(coords[0], coords[1], coords[2], coords[3]),
            scaled,
        )

    conv_gpu[
        dtype,
        dtype,
        dtype,
        Optional[elementwise_simd_epilogue_type](scale_epilogue),
    ](
        input_tt,
        filter_tt,
        out_epilogue_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
    )

    conv_gpu[
        dtype,
        dtype,
        dtype,
    ](
        input_tt,
        filter_tt,
        out_ref_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_epilogue_host, out_epilogue_dev)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var errors = 0
    for i in range(out_size):
        var epilogue_val = out_epilogue_host[i].cast[.float32]()
        var ref_val = out_ref_host[i].cast[.float32]()
        var expected = (ref_val * 2.0).cast[.bfloat16]().cast[.float32]()
        var diff = abs(epilogue_val - expected)
        if diff > max_diff:
            max_diff = diff
        var scale = max(abs(expected), Float32(1e-6))
        if diff / scale > 0.02:
            errors += 1

    if errors > 0:
        print("    FAILED: ", errors, " errors, max_diff=", max_diff)
    else:
        print("    PASSED (max_diff=", max_diff, ")")
    assert_false(errors > 0, "conv_gpu scale epilogue mismatch")

    _ = input_dev^
    _ = filter_dev^
    _ = out_epilogue_dev^
    _ = out_ref_dev^


def test_conv_gpu_additive_epilogue[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test conv_gpu with an additive bias epilogue."""
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1
    comptime dtype = DType.bfloat16
    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print("  ", name, sep="")

    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var out_epilogue_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_ref_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    for i in range(out_size):
        bias_host[i] = Scalar[dtype](1.0)

    rand(input_host.as_span())
    rand(filter_host.as_span())

    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var out_epilogue_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(out_epilogue_dev, bias_host)

    comptime input_tt_layout = row_major[N, H, W, C_in]()
    comptime filter_tt_layout = row_major[R, S, C_in, C_out]()
    comptime output_tt_layout = row_major[N, Hout, Wout, C_out]()
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_epilogue_tt = TileTensor(out_epilogue_dev, output_tt_layout)
    var out_ref_tt = TileTensor(out_ref_dev, output_tt_layout)

    @__parameter
    @always_inline
    @__copy_capture(out_epilogue_tt)
    def add_bias_epilogue[
        _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        var coord = Coord(coords[0], coords[1], coords[2], coords[3])
        var existing = out_epilogue_tt.load[
            width=_width, alignment=align_of[_dtype]() * _alignment
        ](coord)
        var result = (val.cast[.float32]() + existing.cast[.float32]()).cast[
            dtype
        ]()
        out_epilogue_tt.store[
            width=_width, alignment=align_of[_dtype]() * _alignment
        ](coord, result)

    conv_gpu[
        dtype,
        dtype,
        dtype,
        Optional[elementwise_simd_epilogue_type](add_bias_epilogue),
    ](
        input_tt,
        filter_tt,
        out_epilogue_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
    )

    conv_gpu[
        dtype,
        dtype,
        dtype,
    ](
        input_tt,
        filter_tt,
        out_ref_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_epilogue_host, out_epilogue_dev)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var errors = 0
    for i in range(out_size):
        var epilogue_val = out_epilogue_host[i].cast[.float32]()
        var ref_val = out_ref_host[i].cast[.float32]()
        var expected = (
            (ref_val + Float32(1.0)).cast[.bfloat16]().cast[.float32]()
        )
        var diff = abs(epilogue_val - expected)
        if diff > max_diff:
            max_diff = diff
        var scale = max(abs(expected), Float32(1e-6))
        if diff / scale > 0.02:
            errors += 1

    if errors > 0:
        print("    FAILED: ", errors, " errors, max_diff=", max_diff)
    else:
        print("    PASSED (max_diff=", max_diff, ")")
    assert_false(errors > 0, "conv_gpu additive epilogue mismatch")

    _ = input_dev^
    _ = filter_dev^
    _ = out_epilogue_dev^
    _ = out_ref_dev^


def test_conv_gpu_residual[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test conv_gpu with native TMA-based residual add."""
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1
    comptime dtype = DType.bfloat16
    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print("  ", name, sep="")

    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var source_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_residual_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_ref_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    rand(input_host.as_span())
    rand(filter_host.as_span())
    rand(source_host.as_span())

    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var source_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_residual_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(source_dev, source_host)

    comptime input_tt_layout = row_major[N, H, W, C_in]()
    comptime filter_tt_layout = row_major[R, S, C_in, C_out]()
    comptime output_tt_layout = row_major[N, Hout, Wout, C_out]()
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_residual_tt = TileTensor(out_residual_dev, output_tt_layout)
    var out_ref_tt = TileTensor(out_ref_dev, output_tt_layout)

    conv_gpu[
        dtype,
        dtype,
        dtype,
        has_residual=True,
    ](
        input_tt,
        filter_tt,
        out_residual_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
        source_dev.unsafe_ptr().as_unsafe_any_origin(),
        Float32(1.0),
    )

    conv_gpu[
        dtype,
        dtype,
        dtype,
    ](
        input_tt,
        filter_tt,
        out_ref_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_residual_host, out_residual_dev)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var errors = 0
    for i in range(out_size):
        var residual_val = out_residual_host[i].cast[.float32]()
        var ref_val = out_ref_host[i].cast[.float32]()
        var src_val = source_host[i].cast[.float32]()
        var expected = (ref_val + src_val).cast[.bfloat16]().cast[.float32]()
        var diff = abs(residual_val - expected)
        if diff > max_diff:
            max_diff = diff
        var scale = max(abs(expected), Float32(1e-6))
        if diff / scale > 0.02:
            errors += 1

    if errors > 0:
        print("    FAILED: ", errors, " errors, max_diff=", max_diff)
    else:
        print("    PASSED (max_diff=", max_diff, ")")
    assert_false(errors > 0, "conv_gpu residual output mismatch")

    _ = input_dev^
    _ = filter_dev^
    _ = source_dev^
    _ = out_residual_dev^
    _ = out_ref_dev^


def test_conv_gpu_residual_diag_no_lambda[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Residual-only diagnostic: zero input/filter, position-encoded source.

    Bypasses the elementwise_lambda_fn branch — calls conv_gpu with
    `has_residual=True` and no epilogue lambda. Expected output = source
    (since Conv = 0). Any wrong-source read shows up directly as a
    per-element mismatch — random-input tests mask this via magnitude.
    """
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1
    comptime dtype = DType.bfloat16
    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print("  ", name, sep="")

    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var source_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    for i in range(in_size):
        input_host[i] = Scalar[dtype](0.0)
    for i in range(filter_size):
        filter_host[i] = Scalar[dtype](0.0)
    for b in range(N):
        for h in range(Hout):
            for w in range(Wout):
                for c in range(C_out):
                    var idx = ((b * Hout + h) * Wout + w) * C_out + c
                    source_host[idx] = Scalar[dtype](Float32(c) * 0.01)

    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var source_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(source_dev, source_host)

    comptime input_tt_layout = row_major[N, H, W, C_in]()
    comptime filter_tt_layout = row_major[R, S, C_in, C_out]()
    comptime output_tt_layout = row_major[N, Hout, Wout, C_out]()
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_tt = TileTensor(out_dev, output_tt_layout)

    conv_gpu[
        dtype,
        dtype,
        dtype,
        has_residual=True,
    ](
        input_tt,
        filter_tt,
        out_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
        source_dev.unsafe_ptr().as_unsafe_any_origin(),
        Float32(1.0),
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var errors = 0
    var first_err_coords = IndexList[4](-1, -1, -1, -1)
    var first_err_got: Float32 = 0.0
    var first_err_expected: Float32 = 0.0
    for b in range(N):
        for h in range(Hout):
            for w in range(Wout):
                for c in range(C_out):
                    var idx = ((b * Hout + h) * Wout + w) * C_out + c
                    var got = out_host[idx].cast[.float32]()
                    var expected = (
                        (Float32(c) * 0.01).cast[.bfloat16]().cast[.float32]()
                    )
                    var diff = abs(got - expected)
                    if diff > max_diff:
                        max_diff = diff
                    if diff > 0.05:
                        if errors == 0:
                            first_err_coords = IndexList[4](b, h, w, c)
                            first_err_got = got
                            first_err_expected = expected
                        errors += 1

    if errors > 0:
        print(
            "    FAILED: ",
            errors,
            "/",
            out_size,
            " errors, max_diff=",
            max_diff,
            " (first @ ",
            first_err_coords,
            ": got=",
            first_err_got,
            " want=",
            first_err_expected,
            ")",
            sep="",
        )
    else:
        print("    PASSED (max_diff=", max_diff, ")")
    assert_false(errors > 0, "conv_gpu residual no-lambda mismatch")

    _ = input_dev^
    _ = filter_dev^
    _ = source_dev^
    _ = out_dev^


def test_conv_gpu_residual_with_bias[
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_out: Int,
    pad: Int,
    name: StringLiteral,
](ctx: DeviceContext) raises:
    """Test conv_gpu with residual add AND a void elementwise bias lambda.

    Validates the FLUX VAE pattern `D = Conv(A,B) + bias + beta*C` end-to-end.
    Source values are position-encoded against zero-input/filter so the
    expected output reduces to `bias + source` element-wise; any read of
    the wrong source position (the MODELS-1484 source-loading bug) is
    immediately visible per element.
    """
    comptime Hout = H + 2 * pad - R + 1
    comptime Wout = W + 2 * pad - S + 1
    comptime dtype = DType.bfloat16
    comptime in_size = N * H * W * C_in
    comptime filter_size = R * S * C_in * C_out
    comptime out_size = N * Hout * Wout * C_out

    print("  ", name, sep="")

    var input_host = ctx.enqueue_create_host_buffer[dtype](in_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var source_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](out_size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](out_size)

    # Zero input + filter so Conv = 0; expected = bias + source.
    for i in range(in_size):
        input_host[i] = Scalar[dtype](0.0)
    for i in range(filter_size):
        filter_host[i] = Scalar[dtype](0.0)

    # Position-encoded source: source[b,h,w,c] = c * 0.01. Reading the
    # wrong channel gives a visibly different value.
    for b in range(N):
        for h in range(Hout):
            for w in range(Wout):
                for c in range(C_out):
                    var idx = ((b * Hout + h) * Wout + w) * C_out + c
                    source_host[idx] = Scalar[dtype](Float32(c) * 0.01)

    for i in range(out_size):
        bias_host[i] = Scalar[dtype](1.0)

    var input_dev = ctx.enqueue_create_buffer[dtype](in_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var source_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(source_dev, source_host)
    # Pre-load output with bias; the void lambda reads it back and adds val.
    ctx.enqueue_copy(out_dev, bias_host)

    comptime input_tt_layout = row_major[N, H, W, C_in]()
    comptime filter_tt_layout = row_major[R, S, C_in, C_out]()
    comptime output_tt_layout = row_major[N, Hout, Wout, C_out]()
    var input_tt = TileTensor(input_dev, input_tt_layout)
    var filter_tt = TileTensor(filter_dev, filter_tt_layout)
    var out_tt = TileTensor(out_dev, output_tt_layout)

    @__parameter
    @always_inline
    @__copy_capture(out_tt)
    def add_bias_epilogue[
        _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        var coord = Coord(coords[0], coords[1], coords[2], coords[3])
        var existing = out_tt.load[
            width=_width, alignment=align_of[_dtype]() * _alignment
        ](coord)
        var result = (val.cast[.float32]() + existing.cast[.float32]()).cast[
            dtype
        ]()
        out_tt.store[width=_width, alignment=align_of[_dtype]() * _alignment](
            coord, result
        )

    conv_gpu[
        dtype,
        dtype,
        dtype,
        Optional[elementwise_simd_epilogue_type](add_bias_epilogue),
        has_residual=True,
    ](
        input_tt,
        filter_tt,
        out_tt,
        IndexList[2](1, 1),
        IndexList[2](1, 1),
        IndexList[4](pad, pad, pad, pad),
        1,
        ctx,
        source_dev.unsafe_ptr().as_unsafe_any_origin(),
        Float32(1.0),
    )

    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var errors = 0
    var first_err_coords = IndexList[4](-1, -1, -1, -1)
    var first_err_got: Float32 = 0.0
    var first_err_expected: Float32 = 0.0
    for b in range(N):
        for h in range(Hout):
            for w in range(Wout):
                for c in range(C_out):
                    var idx = ((b * Hout + h) * Wout + w) * C_out + c
                    var got = out_host[idx].cast[.float32]()
                    var expected = (
                        (Float32(1.0) + Float32(c) * 0.01)
                        .cast[.bfloat16]()
                        .cast[.float32]()
                    )
                    var diff = abs(got - expected)
                    if diff > max_diff:
                        max_diff = diff
                    if diff > 0.05:
                        if errors == 0:
                            first_err_coords = IndexList[4](b, h, w, c)
                            first_err_got = got
                            first_err_expected = expected
                        errors += 1

    if errors > 0:
        print(
            "    FAILED: ",
            errors,
            "/",
            out_size,
            " errors, max_diff=",
            max_diff,
            " (first @ ",
            first_err_coords,
            ": got=",
            first_err_got,
            " want=",
            first_err_expected,
            ")",
            sep="",
        )
    else:
        print("    PASSED (max_diff=", max_diff, ")")
    assert_false(errors > 0, "conv_gpu residual+bias output mismatch")

    _ = input_dev^
    _ = filter_dev^
    _ = source_dev^
    _ = out_dev^


def main() raises:
    print("=" * 60)
    print("SM100 CONV2D TEST")
    print("=" * 60)
    print()

    # Test problem shape computations (CPU)
    test_conv2d_problem_shape()

    # Test GPU kernels
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16

        # ============================================================
        # Test 1: 3x3 conv with padding
        # batch=1, 16x16 spatial, 64 in_channels, 256 out_channels
        # M = 256, N = 256, K = 64*3*3 = 576
        # ============================================================
        print("--- Test 1: 3x3 conv with padding ---")
        test_conv2d_implicit_im2col[
            dtype,
            dtype,
            DType.bfloat16,
        ](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=64,
            out_c=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 2: 1x1 pointwise conv
        # batch=1, 32x32 spatial, 256 in_channels, 256 out_channels
        # M = 1024, N = 256, K = 256
        # ============================================================
        print("--- Test 2: 1x1 pointwise conv ---")
        test_conv2d_implicit_im2col[
            dtype,
            dtype,
            DType.bfloat16,
        ](
            ctx,
            batch=1,
            in_h=32,
            in_w=32,
            in_c=256,
            out_c=256,
            filter_h=1,
            filter_w=1,
            pad_h=0,
            pad_w=0,
        )

        # ============================================================
        # Test 3: 1-SM mode (cta_group=1)
        # Same as Test 1 but with 1-SM configuration
        # ============================================================
        print("--- Test 3: 1-SM mode (3x3 conv) ---")
        test_conv2d_1sm[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 3b: 1-SM mode with SWIZZLE_64B, C_in=96 (bf16)
        # C_in=96 * 2 bytes = 192 B per row: 64B-aligned but not 128B-aligned.
        # SWIZZLE_128B cannot represent this row; SWIZZLE_64B can.
        # BK=32 halves per-stage SMEM, but the auto-sizer currently ignores
        # conv-specific fixed overhead (SourceTiles) so we pin stages=6.
        # ============================================================
        print("--- Test 3b: 1-SM mode, SWIZZLE_64B, C_in=96 (3x3) ---")
        test_conv2d_1sm[
            dtype,
            dtype,
            DType.bfloat16,
            swizzle=TensorMapSwizzle.SWIZZLE_64B,
            num_pipeline_stages_override=6,
        ](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=96,
            out_c=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # 1x1 pointwise variant at C_in=96 (also routes via SWIZZLE_64B).
        print("--- Test 3c: 1-SM mode, SWIZZLE_64B, C_in=96 (1x1) ---")
        test_conv2d_1sm[
            dtype,
            dtype,
            DType.bfloat16,
            swizzle=TensorMapSwizzle.SWIZZLE_64B,
            num_pipeline_stages_override=6,
        ](
            ctx,
            batch=1,
            in_h=32,
            in_w=32,
            in_c=96,
            out_c=128,
            filter_h=1,
            filter_w=1,
            pad_h=0,
            pad_w=0,
        )

        # ============================================================
        # Test 3d: 1-SM, out_c=96 (N-tail), C_in=128 (SWIZZLE_128B)
        # Exercises TMA OOB handling on the output side:
        #   - filter TMA zero-fills rows 96..127 on load,
        #   - output TMA drops stores to cols 96..127.
        # Grid_y = ceildiv(96, MMA_N=128) = 1; one block handles the tail.
        # ============================================================
        print("--- Test 3d: 1-SM mode, out_c=96, C_in=128 (3x3) ---")
        test_conv2d_1sm[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 3e: 1-SM, out_c=96 AND C_in=96 (both-dim tail on SWIZZLE_64B)
        # Worst-case shape for the native path: both N and K dims are non-128
        # aligned. Confirms C_in tail (via SWIZZLE_64B / BK=32) and N tail
        # (via TMA OOB) compose cleanly.
        # ============================================================
        print(
            "--- Test 3e: 1-SM mode, SWIZZLE_64B, C_in=96, out_c=96 (3x3) ---"
        )
        test_conv2d_1sm[
            dtype,
            dtype,
            DType.bfloat16,
            swizzle=TensorMapSwizzle.SWIZZLE_64B,
            num_pipeline_stages_override=6,
        ](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=96,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 3f: 2-SM, out_c=96, C_in=128 via implicit_im2col
        # MMA_N=256 in 2-SM mode: even more aggressive N-tail (96/256 block)
        # Tests TMA OOB on the 2-SM dispatch path.
        # ============================================================
        print("--- Test 3f: 2-SM mode, out_c=96, C_in=128 (3x3) ---")
        test_conv2d_implicit_im2col[
            dtype,
            dtype,
            DType.bfloat16,
        ](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 3g: 1-SM, C_in=192, out_c=96 (SWIZZLE_128B)
        # in_c=192 is 128B-aligned but non-power-of-two; K = 192*9 = 1728
        # is still a multiple of BK=64 (1728/64=27 iterations). Combines
        # with out_c=96 to cover "larger non-standard in_c with N-tail".
        # ============================================================
        print("--- Test 3g: 1-SM mode, C_in=192, out_c=96 (3x3) ---")
        test_conv2d_1sm[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=192,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 3h: 1-SM, C_in=192, out_c=192 (SWIZZLE_128B)
        # out_c=192 spans >1 MMA_N block in 1-SM mode:
        #   grid_y = ceildiv(192, MMA_N=128) = 2.
        # Block 0 covers N=[0, 128); block 1 covers N=[128, 256) with cols
        # 192..255 being TMA-dropped tail (64 cols of OOB per output tile).
        # Validates that multi-block N with a per-block-partial tail works.
        # ============================================================
        print("--- Test 3h: 1-SM mode, C_in=192, out_c=192 (3x3) ---")
        test_conv2d_1sm[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=192,
            out_c=192,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 4: Epilogue lambda (bias addition)
        # Tests the epilogue fusion infrastructure
        # ============================================================
        print("--- Test 4: Epilogue lambda (bias add) ---")
        test_conv2d_epilogue_lambda[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 5: Conv2d + bias fusion (2-SM mode)
        # Focused test for the common FLUX pattern
        # ============================================================
        print("--- Test 5: Conv2d + bias fusion (2-SM) ---")
        test_conv2d_bias_fusion[dtype, use_1sm=False](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 6: Conv2d + bias fusion (1-SM mode)
        # Same pattern with 1-SM configuration
        # ============================================================
        print("--- Test 6: Conv2d + bias fusion (1-SM) ---")
        test_conv2d_bias_fusion[dtype, use_1sm=True](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 6b: Conv2d + bias fusion at out_c=96 (1-SM)
        # Exercises the TileWriter's compute-lambda path with a
        # non-MMA_N-aligned bias tensor. Bias has exactly out_c=96
        # elements; safety relies on the TMA-store dropping OOB cols
        # (the compute-lambda store goes through the TMA path, not the
        # direct-store path).
        # ============================================================
        print("--- Test 6b: Conv2d + bias fusion, out_c=96 (1-SM) ---")
        test_conv2d_bias_fusion[dtype, use_1sm=True](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 7: Conv2d with residual API
        # Tests the conv2d_fprop_with_residual API
        # ============================================================
        print("--- Test 7: Conv2d with residual API ---")
        test_conv2d_residual_api[dtype](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 7b: Conv2d residual API at out_c=96
        # Source-C TMA load zero-fills cols 96..127; add to accumulator;
        # output TMA drops the tail. Validates residual add semantics
        # when the residual tensor is itself non-MMA_N-aligned.
        # ============================================================
        print("--- Test 7b: Conv2d with residual API, out_c=96 ---")
        test_conv2d_residual_api[dtype](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=96,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # NOTE: FP16 tests require additional stdlib changes beyond TMA:
        # - max/gpu/compute/mma.mojo st_matrix() also only supports BF16/F32
        # - Full FP16 support would require updates across multiple files
        # For now, CUTLASS comparison requires modifying CUTLASS to use BF16

        # ============================================================
        # Tests 8-12: Scale-by-2 epilogue on FLUX layer shapes
        # ============================================================
        print("--- Scale-by-2 Epilogue ---")
        test_conv_gpu_scale_epilogue[1, 16, 16, 512, 3, 3, 512, 1, "3x3_512"](
            ctx
        )
        test_conv_gpu_scale_epilogue[
            1, 32, 32, 512, 3, 3, 256, 1, "3x3_512to256"
        ](ctx)
        test_conv_gpu_scale_epilogue[
            1, 32, 32, 512, 1, 1, 256, 0, "1x1_shortcut"
        ](ctx)
        test_conv_gpu_scale_epilogue[
            1, 64, 64, 256, 3, 3, 128, 1, "3x3_256to128"
        ](ctx)
        test_conv_gpu_scale_epilogue[
            1, 16, 32, 512, 3, 3, 512, 1, "3x3_nonsquare"
        ](ctx)
        # out_c=96 via the high-level conv_gpu entry point. This is the
        # end-to-end check that the relaxed gate in conv.mojo (previously
        # `out_c % 128 == 0`, now `out_c * sizeof % 4 == 0`) actually routes
        # the non-MMA_N-aligned N through the SM100 path rather than the
        # im2col_matmul fallback.
        test_conv_gpu_scale_epilogue[
            1, 16, 16, 128, 3, 3, 96, 1, "3x3_128to96"
        ](ctx)
        test_conv_gpu_scale_epilogue[1, 16, 16, 96, 3, 3, 96, 1, "3x3_96to96"](
            ctx
        )
        # in_c=192 is 128B-aligned (192*2=384B = 3*128) so uses SWIZZLE_128B
        # with BK=64. K=192*9=1728 is a multiple of BK. Combined with
        # out_c=96 this exercises an in_c > 128 non-power-of-two case.
        test_conv_gpu_scale_epilogue[
            1, 16, 16, 192, 3, 3, 96, 1, "3x3_192to96"
        ](ctx)
        # in_c=192, out_c=192: out_c > MMA_N=128 so grid_y spans 2 blocks
        # in 1-SM mode, the second of which has a 64-col N-tail dropped by
        # TMA. Exercises the multi-N-block partial-tail path through the
        # full conv_gpu dispatch.
        test_conv_gpu_scale_epilogue[
            1, 16, 16, 192, 3, 3, 192, 1, "3x3_192to192"
        ](ctx)

        # ============================================================
        # Tests 13-14: Additive bias epilogue
        # ============================================================
        print("\n--- Additive Bias Epilogue ---")
        test_conv_gpu_additive_epilogue[
            1, 16, 16, 512, 3, 3, 512, 1, "3x3_512_bias"
        ](ctx)
        test_conv_gpu_additive_epilogue[
            1, 64, 64, 256, 3, 3, 128, 1, "3x3_256to128_bias"
        ](ctx)

        # ============================================================
        # Tests 15-18: Native TMA residual add
        # ============================================================
        print("\n--- Native TMA Residual Add ---")
        test_conv_gpu_residual[1, 16, 16, 512, 3, 3, 512, 1, "3x3_512_res"](ctx)
        test_conv_gpu_residual[
            1, 32, 32, 512, 3, 3, 256, 1, "3x3_512to256_res"
        ](ctx)
        test_conv_gpu_residual[
            1, 32, 32, 512, 1, 1, 256, 0, "1x1_shortcut_res"
        ](ctx)
        test_conv_gpu_residual[
            1, 64, 64, 256, 3, 3, 128, 1, "3x3_256to128_res"
        ](ctx)

        # ============================================================
        # Tests 19+: Residual add + void bias lambda — the FLUX VAE
        # `conv2d_residual_add` pattern (MODELS-1484). These tests use
        # zero-input/filter so any wrong-source read is visible per
        # element; they would have caught the source-loading bug that
        # only manifests under matching conv/source magnitudes.
        # ============================================================
        # Diagnostic without the elementwise lambda — isolates source loading.
        print(
            "\n--- Residual No-Lambda Diag (zero-conv + position-encoded"
            " src) ---"
        )
        test_conv_gpu_residual_diag_no_lambda[
            1, 16, 16, 128, 3, 3, 128, 1, "3x3_128_no_lambda"
        ](ctx)

        print(
            "\n--- Residual + Bias Lambda (zero-conv + position-encoded"
            " src) ---"
        )
        # Small / coverage shapes.
        test_conv_gpu_residual_with_bias[
            1, 16, 16, 128, 3, 3, 128, 1, "3x3_128_diag"
        ](ctx)
        test_conv_gpu_residual_with_bias[
            1, 16, 16, 512, 3, 3, 512, 1, "3x3_512_diag"
        ](ctx)
        # FLUX2 VAE decoder shapes (block_out_channels = [128, 256, 512, 512];
        # decoder spatial dims 128 -> 256 -> 512 -> 1024 for a 1024px output).
        # We test at reduced spatial dims to keep runtime manageable while
        # still exercising every (in_c == out_c) channel count used by FLUX2.
        test_conv_gpu_residual_with_bias[
            1, 32, 32, 512, 3, 3, 512, 1, "flux_mid_512"
        ](ctx)
        test_conv_gpu_residual_with_bias[
            1, 32, 32, 256, 3, 3, 256, 1, "flux_up2_256"
        ](ctx)
        test_conv_gpu_residual_with_bias[
            1, 32, 32, 128, 3, 3, 128, 1, "flux_up3_128"
        ](ctx)

    print("=" * 60)
    print("ALL CONV2D TESTS PASSED!")
    print("=" * 60)
