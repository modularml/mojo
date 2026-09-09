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

"""Tests for fused RMSNorm + FP8 quantization kernel."""

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TileTensor,
    row_major,
)
from std.memory import bitcast

from std.utils.index import Index, IndexList
from std.math import rsqrt
from std.utils.numerics import max_finite, min_finite

from nn.normalization import rms_norm_fused_fp8


def initialize_test_data[
    dtype: DType
](data: MutPointer[Scalar[dtype], _], size: Int):
    """Initialize test data with diverse positive values to avoid FP8 saturation.

    Creates a mix of small, medium, and large magnitudes (0.05 to 3.0) that will
    produce a good distribution after RMSNorm without hitting FP8 edge cases.
    """
    for i in range(size):
        var pattern = i % 20
        var val: Float32

        # Use a wider range [0.05, 3.0] with varied spacing - all positive
        if pattern < 5:
            # Small values: 0.05, 0.1, 0.15, 0.2, 0.25
            val = Float32(0.05 + Float64(pattern) * 0.05)
        elif pattern < 10:
            # Medium values: 0.3, 0.4, 0.5, 0.6, 0.7
            val = Float32(0.3 + Float64((pattern - 5)) * 0.1)
        elif pattern < 15:
            # Medium-large values: 0.8, 1.0, 1.2, 1.4, 1.6
            val = Float32(0.8 + Float64((pattern - 10)) * 0.2)
        else:
            # Large values: 1.8, 2.1, 2.4, 2.7, 3.0
            val = Float32(1.8 + Float64((pattern - 15)) * 0.3)

        data[i] = val.cast[dtype]()


def compute_reference_dynamic_scaling[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
](
    input_data: Pointer[Scalar[in_dtype], _],
    gamma_data: Pointer[Scalar[in_dtype], _],
    output_data: MutPointer[Scalar[out_dtype], _],
    scales_data: MutPointer[Scalar[scales_dtype], _],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    weight_offset: Float32,
    scale_ub: Float32,
):
    """Compute reference RMSNorm + dynamic FP8 quantization on host."""
    var fp8_max = Float32(max_finite[out_dtype]())
    var fp8_min = Float32(min_finite[out_dtype]())

    # Allocate temporary storage for normalized values
    var temp_storage = List(length=cols, fill=Float32(0))

    for row in range(rows):
        # Step 1: Compute mean square for RMSNorm
        var mean_square = Float32(0.0)
        for col in range(cols):
            var val = input_data[row * cols + col].cast[.float32]()
            mean_square += val * val
        mean_square = mean_square / Float32(cols)

        # Step 2: Compute normalization factor
        var norm_factor = rsqrt(mean_square + epsilon)

        # Step 3: Normalize and apply gamma (store in temp buffer)
        var row_max = Float32(0.0)
        for col in range(cols):
            var val = input_data[row * cols + col].cast[.float32]()
            var normalized = val * norm_factor
            var gamma_val = gamma_data[col].cast[.float32]()
            var scaled_val = normalized * (gamma_val + weight_offset)

            # Track max for dynamic scaling
            row_max = max(row_max, abs(scaled_val))

            # Store normalized value in temporary buffer
            temp_storage[col] = scaled_val

        # Step 4: Compute scale and quantize (clamping row_max to scale_ub)
        # Match kernel precision: scale is computed in scales_dtype
        var clamped_max_sd = min(
            row_max.cast[scales_dtype](),
            scale_ub.cast[scales_dtype](),
        )
        var scale_sd = (
            clamped_max_sd / max_finite[out_dtype]().cast[scales_dtype]()
        )
        scales_data[row] = scale_sd
        var scale_factor_recip = (
            Float32(0.0) if scale_sd == 0.0 else 1.0 / scale_sd.cast[.float32]()
        )

        for col in range(cols):
            var scaled_val = temp_storage[col]
            var quantized = scaled_val * scale_factor_recip
            quantized = max(fp8_min, min(fp8_max, quantized))
            output_data[row * cols + col] = quantized.cast[out_dtype]()


def test_dynamic[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    weight_offset: Float32 = 0.0,
    scale_ub: Float32 = 448.0,
) raises:
    """Test arbitrary rank tensor with dynamic scaling."""
    var input_size = shape.flattened_length()
    var cols = shape[rank - 1]
    var rows = input_size // cols

    # Allocate and initialize host memory
    var in_host = ctx.enqueue_create_host_buffer[in_dtype](input_size)
    var out_host = ctx.enqueue_create_host_buffer[out_dtype](input_size)
    var gamma_host = ctx.enqueue_create_host_buffer[in_dtype](cols)
    var scales_host = ctx.enqueue_create_host_buffer[scales_dtype](rows)
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](input_size)
    var expected_scales_host = ctx.enqueue_create_host_buffer[scales_dtype](
        rows
    )

    # Initialize with diverse values to avoid FP8 saturation
    initialize_test_data(in_host.unsafe_ptr(), input_size)
    for i in range(cols):
        # Gamma values between 0.5 and 1.5 to create variety after normalization
        gamma_host[i] = Scalar[in_dtype](0.5 + Float64((i % 11)) * 0.1)

    # epsilon is Float32 (the kernel signature), so the reference and kernel
    # both consume it directly. weight_offset is cast to in_dtype (matching its
    # kernel signature) and back to Float32 so reference and kernel see the
    # same value.
    var epsilon = Float32(1e-5)
    var weight_offset_id = Scalar[in_dtype](weight_offset)
    var weight_offset_f32 = weight_offset_id.cast[.float32]()

    # Compute reference
    compute_reference_dynamic_scaling[in_dtype, out_dtype](
        in_host.unsafe_ptr(),
        gamma_host.unsafe_ptr(),
        expected_host.unsafe_ptr(),
        expected_scales_host.unsafe_ptr(),
        rows,
        cols,
        epsilon,
        weight_offset_f32,
        scale_ub,
    )

    # Setup GPU
    var in_device = ctx.enqueue_create_buffer[in_dtype](input_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](input_size)
    var gamma_device = ctx.enqueue_create_buffer[in_dtype](cols)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](rows)

    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(gamma_device, gamma_host)

    var param_shape = Index(cols)
    var gamma_tensor = TileTensor(gamma_device, row_major(Coord(param_shape)))

    var scale_shape = shape
    scale_shape[rank - 1] = 1

    var in_ptr = in_device.unsafe_ptr()

    @__copy_capture(in_ptr)
    @always_inline
    @__parameter
    def input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var linear_idx = Int(0)
        var stride = 1
        for i in reversed(range(_rank)):
            linear_idx += idx[i] * stride
            stride *= shape[i]
        return in_ptr.load[width=width, alignment=width](linear_idx)

    var out_tile = TileTensor(out_device, row_major(Coord(shape)))
    var scale_tile = TileTensor(scales_device, row_major(Coord(scale_shape)))

    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        scales_dtype,
        rank,
        input_fn,
        target="gpu",
    ](
        shape,
        out_tile,
        gamma_tensor,
        epsilon,
        weight_offset_id,
        ctx,
        scale_ub,
        scale_tile,
    )

    # Verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.enqueue_copy(scales_host, scales_device)
    ctx.synchronize()

    # `input_fn` captures only the raw `in_ptr`, not `in_device`, so without an
    # explicit keep-alive Mojo's ASAP destruction frees `in_device` right after
    # its last use (`unsafe_ptr()`) — before the kernel that reads it runs. The
    # caching allocator masks the resulting use-after-free, but the sanitizer's
    # 1:1 allocator (`--//:gpu_disable_memory_manager`) reports it as an OOB read
    # of the input. Keep the buffer alive through the launch (mirrors the
    # `_ = data_d` guard in test_rms_norm.mojo).
    _ = in_device

    var num_mismatches = 0
    for i in range(input_size):
        if bitcast[.uint8](out_host[i]) != bitcast[.uint8](expected_host[i]):
            num_mismatches += 1
            if num_mismatches <= 5:
                print("Mismatch at", i)

    if num_mismatches > 0:
        raise Error("Higher rank tensor test failed")

    print(
        "✓ Rank-",
        rank,
        " tensor test passed (dynamic, weight_offset=",
        weight_offset,
        ") ",
        shape,
    )


def main() raises:
    print("Running fused RMSNorm + FP8 tests...")
    var ctx = DeviceContext()

    comptime for scales_dtype in [DType.float32, DType.bfloat16]:
        print("\nTesting scales dtype: ", scales_dtype)
        # Rank-2 tests: Small sizes (warp-tiling)
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(32, 256)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(8, 512)
        )

        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(64, 4096)
        )

        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 8192)
        )

        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 16384)
        )

        # Rank-2 tests: Large (block-tiling)
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 32768)
        )

        # Rank-2 tests: Non-power-of-2 dimensions
        print("\nTesting non-power-of-2 dimensions...")
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(13, 97)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(7, 333)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(17, 513)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(23, 1001)
        )

        # Rank-3 and Rank-4 tests
        print("\nTesting higher rank tensors...")
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(2, 16, 256)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(3, 5, 97)
        )  # Non-power-of-2

        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 4, 8, 128)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 3, 5, 64)
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 3, 7, 97)
        )  # Non-power-of-2

        # Tests with nonzero weight_offset
        print("\nTesting nonzero weight_offset...")
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128), weight_offset=0.5
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128), weight_offset=0.1
        )

        # Tests with low scale_ub to exercise the clamping path
        print("\nTesting low scale_ub...")
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128), scale_ub=0.5
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(32, 256), scale_ub=0.1
        )
        test_dynamic[.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128), scale_ub=0.25
        )

    print("\n✅ All tests passed!")
