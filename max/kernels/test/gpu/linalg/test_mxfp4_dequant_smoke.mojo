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
"""Smoke test for MXFP4 dequantization kernel on H100 (SM90).

Validates dequant_mxfp4 by comparing GPU output against a CPU reference
for several shapes and scale values. Target runtime: < 30s on H100.
"""

from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.fp4_utils import E2M1_TO_FLOAT32


def _pack_fp4_pair(low: UInt8, high: UInt8) -> UInt8:
    """Packs two 4-bit FP4 values into one uint8 byte."""
    return (high & UInt8(0x0F)) << UInt8(4) | (low & UInt8(0x0F))


def _e8m0_to_float32(bits: UInt8) -> Float32:
    """Converts float8_e8m0fnu scale byte to float32: 2^(exp-127)."""
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[.float32](f32_bits)


def _cpu_dequant_mxfp4[
    out_dtype: DType = .bfloat16
](
    expected: MutPointer[Scalar[out_dtype], _],
    input_data: ImmPointer[UInt8, _],
    scales_data: ImmPointer[UInt8, _],
    num_rows: Int,
    num_cols: Int,
):
    """CPU reference: dequant MXFP4 packed uint8 + E8M0 scales."""
    var packed_cols = num_cols // 2
    var scale_cols = ceildiv(num_cols, 32)

    for row in range(num_rows):
        for col in range(num_cols):
            var packed_col = col // 2
            var packed_byte = input_data[row * packed_cols + packed_col]
            var nibble_shift = UInt8((col % 2) * 4)
            var fp4_bits = Int((packed_byte >> nibble_shift) & UInt8(0x0F))
            var fp32_val = E2M1_TO_FLOAT32[fp4_bits]

            var scale_col = col // 32
            var scale_byte = scales_data[row * scale_cols + scale_col]
            var scale_f32 = _e8m0_to_float32(scale_byte)

            var result = (fp32_val * scale_f32).cast[out_dtype]()
            expected[row * num_cols + col] = result


def test_mxfp4_dequant[
    num_rows: Int,
    num_cols: Int,
    out_dtype: DType = .bfloat16,
](ctx: DeviceContext, scale_exp: UInt8) raises:
    """Tests MXFP4 dequant kernel for compile-time shape and runtime scale."""
    comptime packed_cols = num_cols // 2
    comptime scale_cols = ceildiv(num_cols, 32)

    # FP8 has lower precision; use a wider tolerance.
    comptime tol = Float32(
        0.1
    ) if out_dtype == DType.float8_e4m3fn else Float32(0.01)

    var scale_f32 = _e8m0_to_float32(scale_exp)
    print(
        "  rows=",
        num_rows,
        " cols=",
        num_cols,
        " dtype=",
        out_dtype,
        " scale_exp=",
        scale_exp,
        " (scale=",
        scale_f32,
        ")",
    )

    comptime in_size = num_rows * packed_cols
    comptime scales_size = num_rows * scale_cols
    comptime out_size = num_rows * num_cols

    # Allocate and fill host input
    var in_host = ctx.enqueue_create_host_buffer[.uint8](in_size)
    var scales_host = ctx.enqueue_create_host_buffer[.uint8](scales_size)
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    for i in range(scales_size):
        scales_host[i] = scale_exp

    for row in range(num_rows):
        for col in range(packed_cols):
            var low = UInt8((col * 2) % 16)
            var high = UInt8((col * 2 + 1) % 16)
            in_host[row * packed_cols + col] = _pack_fp4_pair(low, high)

    # CPU reference
    _cpu_dequant_mxfp4[out_dtype](
        expected_host.unsafe_ptr(),
        in_host.unsafe_ptr(),
        scales_host.unsafe_ptr(),
        num_rows,
        num_cols,
    )

    # Device buffers
    var in_device = ctx.enqueue_create_buffer[.uint8](in_size)
    var scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](scales_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)

    # Upload input via host buffers
    var in_host_buf = ctx.enqueue_create_host_buffer[.uint8](in_size)
    var scales_host_buf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        scales_size
    )

    for i in range(in_size):
        in_host_buf[i] = in_host[i]
    for i in range(scales_size):
        scales_host_buf[i] = bitcast[.float8_e8m0fnu](scales_host[i])

    ctx.enqueue_copy(in_device, in_host_buf)
    ctx.enqueue_copy(scales_device, scales_host_buf)
    ctx.synchronize()

    # Create TileTensors with compile-time row-major layouts
    var in_tt = TileTensor(in_device, row_major[num_rows, packed_cols]())
    var scales_tt = TileTensor(scales_device, row_major[num_rows, scale_cols]())
    var out_tt = TileTensor(out_device, row_major[num_rows, num_cols]())

    # Run GPU kernel
    dequant_mxfp4(
        ctx,
        out_tt,
        in_tt,
        scales_tt,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    ctx.synchronize()

    # Copy output back
    var out_host_buf = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()

    # Compare
    var max_err = Float32(0.0)
    var num_mismatches = 0
    for i in range(out_size):
        var got = out_host_buf[i].cast[.float32]()
        var exp = expected_host[i].cast[.float32]()
        var err = abs(got - exp)
        max_err = max(max_err, err)
        if err > tol:
            if num_mismatches < 5:
                var row = i // num_cols
                var col = i % num_cols
                print(
                    "    MISMATCH [",
                    row,
                    ",",
                    col,
                    "]: got=",
                    got,
                    " expected=",
                    exp,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL: ",
            num_mismatches,
            " mismatches, max_err=",
            max_err,
        )
        raise Error("MXFP4 dequant test failed")

    print("    PASS max_err=", max_err)


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP4 Dequant Smoke Tests (H100 SM90)")
        print("======================================")

        # Scale = 1.0 (exponent 127)
        print("-- Scale = 1.0 --")
        test_mxfp4_dequant[64, 64](ctx, UInt8(127))
        test_mxfp4_dequant[128, 512](ctx, UInt8(127))
        test_mxfp4_dequant[256, 2880](ctx, UInt8(127))
        test_mxfp4_dequant[100, 192](ctx, UInt8(127))

        # Scale = 2.0 (exponent 128)
        print("-- Scale = 2.0 --")
        test_mxfp4_dequant[64, 64](ctx, UInt8(128))
        test_mxfp4_dequant[128, 512](ctx, UInt8(128))

        # Scale = 0.5 (exponent 126)
        print("-- Scale = 0.5 --")
        test_mxfp4_dequant[128, 512](ctx, UInt8(126))

        # Large shape (gpt-oss-20b MoE dimensions)
        print("-- Large shape --")
        test_mxfp4_dequant[2880, 2880](ctx, UInt8(127))

        # FP8 output (the path used by mxfp4_matmul_sm90)
        print("-- FP8 output --")
        test_mxfp4_dequant[64, 64, out_dtype=.float8_e4m3fn](ctx, UInt8(127))
        test_mxfp4_dequant[128, 512, out_dtype=.float8_e4m3fn](ctx, UInt8(127))

        print("======================================")
        print("ALL TESTS PASSED")
