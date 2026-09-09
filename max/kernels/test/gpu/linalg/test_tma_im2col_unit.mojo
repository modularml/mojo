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
"""Unit tests for TMA im2col - ported from CUTLASS test/unit/conv/device_3x/.

These tests verify the TMA im2col transformation in isolation, matching
CUTLASS's conv_problem_sizes.hpp test cases for 2D fprop.

Test cases from CUTLASS (simplest first):
1. 1x1 filter, no padding: {1, 8, 8, 64} NHWC, {64, 1, 1, 64} KRSC
2. 3x3 filter, no padding: {2, 8, 8, 64} NHWC, {256, 3, 3, 64} KRSC
3. 3x3 filter, symmetric padding: {2, 8, 8, 32} NHWC, {256, 3, 3, 32} KRSC, pad=(1,1)
"""

from std.sys import size_of
from layout import Layout, LayoutTensor
from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from std.testing import assert_false
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from layout import Layout, LayoutTensor

from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTileIm2col,
    create_tensor_tile_im2col,
)
from std.memory import alloc
from std.utils.index import Index, IndexList


# ============================================================================
# Test kernel: Load a single tile using TMA im2col and copy to output
# ============================================================================


@__llvm_arg_metadata(act_tma_op, `nvvm.grid_constant`)
def im2col_load_kernel[
    dtype: DType,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
    BM: Int,
    BK: Int,
](
    act_tma_op: TMATensorTileIm2col[dtype, tile_rank, tile_shape, desc_shape],
    output_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    k_coord: Int,
    m_coord: Int,
):
    """Kernel that loads one tile using im2col TMA and copies to global memory.
    """
    comptime tile_size = BM * BK
    comptime tile_bytes = tile_size * size_of[dtype]()

    # Allocate shared memory for the tile + barrier
    # Barrier needs to come after tile data (aligned to 128B)
    comptime barrier_offset = (tile_bytes + 127) // 128 * 128

    var smem_ptr = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=128,
        name="im2col_smem",
    ]()

    var barrier_ptr = (
        external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
            name="im2col_smem",
        ]()
        + barrier_offset
    ).bitcast[SharedMemBarrier]()

    # Thread 0 initializes barrier and issues TMA load
    if thread_idx.x == 0:
        barrier_ptr[].init()
        barrier_ptr[].expect_bytes(tile_bytes)

        # Create shared memory tile view
        comptime smem_layout = Layout.row_major(BM, BK)
        comptime smem_tile_t = LayoutTensor[
            dtype,
            smem_layout,
            MutAnyOrigin,
            address_space=.SHARED,
            alignment=128,
        ]
        var smem_tile = smem_tile_t(smem_ptr)

        # Issue im2col TMA load using async_copy
        act_tma_op.async_copy[cta_group=1](
            smem_tile,
            barrier_ptr[],
            (k_coord, m_coord),
        )

    barrier()
    if thread_idx.x == 0:
        barrier_ptr[].wait(0)
    barrier()

    # Copy loaded data to output (all threads participate)
    comptime for i in range(tile_size // 128 + 1):
        var idx = thread_idx.x + i * 128
        if idx < tile_size:
            output_ptr[idx] = smem_ptr[idx]


# ============================================================================
# Reference im2col implementation (CPU)
# ============================================================================


def im2col_reference[
    dtype: DType,
](
    output: MutPointer[Scalar[dtype], _],
    input: ImmPointer[Scalar[dtype], _],  # NHWC (flat pointer)
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    out_height: Int,
    out_width: Int,
    # Tile parameters
    m_start: Int,  # Starting M coordinate
    k_start: Int,  # Starting K coordinate
    BM: Int,  # M tile size
    BK: Int,  # K tile size
):
    """Compute reference im2col output for a specific tile.

    The im2col transformation maps:
    - M dimension: batch * out_height * out_width (output spatial locations)
    - K dimension: filter_h * filter_w * in_channels (filter elements)

    K decomposition: K = r * filter_w * in_channels + s * in_channels + c
    M decomposition: M = n * out_height * out_width + oh * out_width + ow
    """
    var hw = out_height * out_width

    for m_local in range(BM):
        var m = m_start + m_local
        if m >= batch * hw:
            # Out of bounds - fill with zeros
            for k_local in range(BK):
                output[m_local * BK + k_local] = Scalar[dtype](0)
            continue

        # Decompose M into (n, oh, ow)
        var n, m_rem = divmod(m, hw)
        var oh, ow = divmod(m_rem, out_width)

        for k_local in range(BK):
            var k = k_start + k_local
            if k >= filter_h * filter_w * in_channels:
                # Out of bounds
                output[m_local * BK + k_local] = Scalar[dtype](0)
                continue

            # Decompose K into (r, s, c)
            # K = r * filter_w * in_channels + s * in_channels + c
            var filter_idx, c = divmod(k, in_channels)
            var r, s = divmod(filter_idx, filter_w)

            # Compute input coordinates
            var ih = oh * stride_h + r - pad_h
            var iw = ow * stride_w + s - pad_w

            # Check bounds and load value
            var val: Scalar[dtype] = 0
            if ih >= 0 and ih < in_height and iw >= 0 and iw < in_width:
                var input_idx = (
                    n * in_height * in_width * in_channels
                    + ih * in_width * in_channels
                    + iw * in_channels
                    + c
                )
                val = input.load(input_idx)

            output[m_local * BK + k_local] = val


# ============================================================================
# Test runner
# ============================================================================


def run_im2col_test[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
    BM: Int,
    BK: Int,
](ctx: DeviceContext, test_name: String,) raises -> Bool:
    """Run a single im2col test case.

    Returns True if test passes, False otherwise.
    """
    print("  Testing:", test_name)

    # Compute output dimensions (using comptime parameters)
    comptime out_height = in_height + 2 * pad_h - filter_h + 1
    comptime out_width = in_width + 2 * pad_w - filter_w + 1

    # GEMM dimensions
    comptime M = batch * out_height * out_width
    comptime K_gemm = filter_h * filter_w * in_channels

    print(
        "    Input: [",
        batch,
        ",",
        in_height,
        ",",
        in_width,
        ",",
        in_channels,
        "] NHWC",
    )
    print(
        "    Filter: [",
        out_channels,
        ",",
        filter_h,
        ",",
        filter_w,
        ",",
        in_channels,
        "] KRSC",
    )
    print("    Output spatial: [", out_height, ",", out_width, "]")
    print("    GEMM: M=", M, " K=", K_gemm)
    print("    Tile: BM=", BM, " BK=", BK)

    # Allocate input tensor
    comptime input_size = batch * in_height * in_width * in_channels
    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)

    # Initialize with sequential pattern
    for i in range(input_size):
        input_host[i] = Scalar[dtype](i % 256)  # Keep values small for bf16

    # Copy to device
    var input_device = ctx.enqueue_create_buffer[dtype](input_size)
    ctx.enqueue_copy(input_device, input_host)

    # Create LayoutTensor view with compile-time static shape for TMA
    comptime input_layout = Layout.row_major(
        batch, in_height, in_width, in_channels
    )
    var input_tensor = LayoutTensor[dtype, input_layout, MutAnyOrigin](
        input_device.unsafe_ptr()
    )

    # Create im2col TMA descriptor
    # Corner offsets for TMA (CUTLASS Fprop convention with dilation=1)
    # From CUTLASS detail.hpp compute_upper_corner_whd:
    #   lower_corner = -pad
    #   upper_corner = pad - (filter - 1) * dilation = pad - (filter - 1) for dilation=1
    comptime lower_corner_h = -pad_h
    comptime lower_corner_w = -pad_w
    comptime upper_corner_h = pad_h - (filter_h - 1)
    comptime upper_corner_w = pad_w - (filter_w - 1)

    # Debug: print descriptor parameters
    print("    Descriptor params:")
    print("      lower_corner: (", lower_corner_h, ",", lower_corner_w, ")")
    print("      upper_corner: (", upper_corner_h, ",", upper_corner_w, ")")
    print("      channels_per_pixel: 64 (128B swizzle / 2B bf16)")
    print("      pixels_per_column: 1")

    var act_tma = create_tensor_tile_im2col[
        dtype,
        tile_shape=Index(BM, BK),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](
        ctx,
        input_tensor,
        Int(lower_corner_h),
        Int(lower_corner_w),
        Int(upper_corner_h),
        Int(upper_corner_w),
        Int(out_height),
        Int(out_width),
        Int(filter_h),
        Int(filter_w),
    )

    # Allocate output buffer
    comptime tile_size = BM * BK
    var output_host = ctx.enqueue_create_host_buffer[dtype](tile_size)
    var output_device = ctx.enqueue_create_buffer[dtype](tile_size)
    var ref_host = ctx.enqueue_create_host_buffer[dtype](tile_size)

    # Compute reference on CPU for tile at (k=0, m=0)
    im2col_reference[dtype](
        ref_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        batch,
        in_height,
        in_width,
        in_channels,
        filter_h,
        filter_w,
        pad_h,
        pad_w,
        1,
        1,  # stride_h, stride_w
        out_height,
        out_width,
        0,
        0,  # m_start, k_start
        BM,
        BK,
    )

    # Run kernel
    comptime smem_bytes = tile_size * size_of[dtype]() + 256

    comptime kernel = im2col_load_kernel[
        dtype,
        type_of(act_tma).rank,
        type_of(act_tma).tile_shape,
        type_of(act_tma).desc_shape,
        BM,
        BK,
    ]

    ctx.enqueue_function[kernel, dump_asm=False](
        act_tma,
        output_device,
        0,  # k_coord
        0,  # m_coord
        grid_dim=(1, 1, 1),
        block_dim=128,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_bytes),
    )
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    # Compare results with 128B swizzle compensation
    # 128B swizzle for bf16 (2 bytes): XOR offset = (m % 8) * 8 elements
    # This means data at linear position k is at swizzled position k XOR ((m % 8) * 8)
    var errors = 0
    var first_error_m = -1
    var first_error_k = -1
    var first_actual: Float32 = 0
    var first_expected: Float32 = 0

    for m in range(BM):
        # 128B swizzle XOR pattern for bf16: (m % 8) * 8 elements
        var xor_offset = (m % 8) * 8
        for k in range(BK):
            var swizzled_k = k ^ xor_offset  # Apply XOR to get swizzled index
            # Wrap within BK
            swizzled_k = swizzled_k % BK
            var actual_idx = m * BK + k
            var expected_idx = m * BK + swizzled_k
            var actual = Float32(output_host[actual_idx])
            var expected = Float32(ref_host[expected_idx])
            if actual != expected:
                errors += 1
                if first_error_m < 0:
                    first_error_m = m
                    first_error_k = k
                    first_actual = actual
                    first_expected = expected

    # Report results
    if errors == 0:
        print("    PASSED")
    else:
        print("    FAILED:", errors, "errors out of", tile_size)
        print("      First error at m=", first_error_m, " k=", first_error_k)
        print("      Got:", first_actual, " Expected:", first_expected)
        print("      (Swizzle: 128B, XOR pattern = (m % 8) * 8)")

        # Print first few values for debugging
        print("      First 8 actual values:", end="")
        for i in range(min(8, tile_size)):
            print(" ", Float32(output_host[i]), end="")
        print()
        print("      First 8 expected values:", end="")
        for i in range(min(8, tile_size)):
            print(" ", Float32(ref_host[i]), end="")
        print()

    # Cleanup
    _ = input_device
    _ = output_device

    return errors == 0


def main() raises:
    print("=" * 70)
    print("TMA Im2Col Unit Tests (ported from CUTLASS)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        var all_passed = True

        # ================================================================
        # Test 1: Simplest case - 1x1 filter, no padding
        # CUTLASS: {1, 8, 8, 64} NHWC, {64, 1, 1, 64} KRSC
        # ================================================================
        print("\n--- Test 1: 1x1 filter, no padding ---")
        # batch=1, H=8, W=8, C=64, out_C=64, filter=1x1, pad=0, BM=64, BK=64
        var test1_passed = run_im2col_test[
            dtype, 1, 8, 8, 64, 64, 1, 1, 0, 0, 64, 64
        ](ctx, "CUTLASS case 1: 1x8x8x64, 1x1 filter")
        all_passed = all_passed and test1_passed

        # ================================================================
        # Test 2: Smaller tile to test basic functionality
        # ================================================================
        print("\n--- Test 2: 1x1 filter, small tile ---")
        # batch=1, H=4, W=4, C=64, out_C=64, filter=1x1, pad=0, BM=8, BK=64
        var test2_passed = run_im2col_test[
            dtype, 1, 4, 4, 64, 64, 1, 1, 0, 0, 8, 64
        ](ctx, "Small: 1x4x4x64, 1x1 filter, BM=8")
        all_passed = all_passed and test2_passed

        # ================================================================
        # Test 3: 3x3 filter, no padding
        # CUTLASS: {1, 8, 8, 64} NHWC, {256, 3, 3, 64} KRSC
        # ================================================================
        print("\n--- Test 3: 3x3 filter, no padding ---")
        # batch=1, H=8, W=8, C=64, out_C=256, filter=3x3, pad=0, BM=32, BK=64
        var test3_passed = run_im2col_test[
            dtype, 1, 8, 8, 64, 256, 3, 3, 0, 0, 32, 64
        ](ctx, "3x3 filter, no padding")
        all_passed = all_passed and test3_passed

        # ================================================================
        # Test 4: 3x3 filter, symmetric padding (same output size)
        # CUTLASS: {1, 8, 8, 64} NHWC, {256, 3, 3, 64} KRSC, pad=(1,1)
        # Note: Use C=64 and BK=64 to align with 128B swizzle (64 bf16 elements)
        # ================================================================
        print("\n--- Test 4: 3x3 filter, symmetric padding ---")
        # batch=1, H=8, W=8, C=64, out_C=256, filter=3x3, pad=1, BM=32, BK=64
        var test4_passed = run_im2col_test[
            dtype, 1, 8, 8, 64, 256, 3, 3, 1, 1, 32, 64
        ](ctx, "3x3 filter, pad=1 (same size)")
        all_passed = all_passed and test4_passed

        # ================================================================
        # Summary
        # ================================================================
        print()
        print("=" * 70)
        if all_passed:
            print("ALL TESTS PASSED!")
        else:
            print("SOME TESTS FAILED")
        print("=" * 70)

        assert_false(not all_passed, "TMA im2col tests failed")
