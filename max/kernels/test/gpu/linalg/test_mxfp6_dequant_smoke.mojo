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
"""Smoke test for the MXFP6 dequantization kernel.

Validates `dequant_mxfp6` against a CPU reference across shapes, scales, both
FP6 encodings, and both output dtypes. The kernel is vendor-neutral, so this
runs on NVIDIA and AMD alike.

The CPU reference deliberately re-derives each 6-bit code straight from the
byte stream instead of calling `unpack_fp6_x4`. That makes it an independent
check that `pack_fp6_x4` and `unpack_fp6_x32` together implement a genuine
contiguous little-endian 6-bit stream, rather than merely inverting each other
under some shared private convention.
"""

from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from linalg.fp6_utils import FP6Format, fp6_reference_table, pack_fp6_x4
from linalg.mxfp6_dequant import dequant_mxfp6


def _e8m0_to_float32(bits: UInt8) -> Float32:
    """Converts a float8_e8m0fnu scale byte to float32: `2^(exp - 127)`."""
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[.float32](f32_bits)


def _code_at(
    stream: ImmPointer[UInt8, _],
    row_base: Int,
    row_bytes: Int,
    col: Int,
) -> Int:
    """Reads element `col` from a packed FP6 row as a raw 6-bit code."""
    var bit_offset = col * 6
    var byte_index = bit_offset // 8
    var shift = bit_offset % 8
    var low = UInt32(stream[unsafe_offset=row_base + byte_index]) >> UInt32(
        shift
    )
    # A code straddles into the next byte only when it starts past bit 2.
    if shift > 2 and byte_index + 1 < row_bytes:
        low |= UInt32(
            stream[unsafe_offset=row_base + byte_index + 1]
        ) << UInt32(8 - shift)
    return Int(low & UInt32(0x3F))


def _cpu_dequant_mxfp6[
    fmt: FP6Format, out_dtype: DType
](
    expected: MutPointer[Scalar[out_dtype], _],
    input_data: ImmPointer[UInt8, _],
    scales_data: ImmPointer[UInt8, _],
    num_rows: Int,
    num_cols: Int,
):
    """CPU reference: unpack the 6-bit stream, decode by table, apply scale."""
    comptime table = fp6_reference_table[fmt]()
    var row_bytes = (num_cols * 6) // 8
    var scale_cols = ceildiv(num_cols, 32)

    for row in range(num_rows):
        var row_base = row * row_bytes
        for col in range(num_cols):
            var code = _code_at(input_data, row_base, row_bytes, col)
            var scale_byte = scales_data[
                unsafe_offset=row * scale_cols + col // 32
            ]
            var value = table[code] * _e8m0_to_float32(scale_byte)
            expected[unsafe_offset=row * num_cols + col] = value.cast[
                out_dtype
            ]()


def test_mxfp6_dequant[
    fmt: FP6Format,
    num_rows: Int,
    num_cols: Int,
    out_dtype: DType = .bfloat16,
](ctx: DeviceContext, scale_exp: UInt8) raises:
    """Tests the MXFP6 dequant kernel for one shape, format and scale."""
    comptime assert num_cols % 32 == 0, "num_cols must be a multiple of 32"
    comptime packed_cols = (num_cols * 6) // 8
    comptime scale_cols = ceildiv(num_cols, 32)

    # Every FP6 value carries at most 3 mantissa bits, so a power-of-two scale
    # keeps the product exact in both bfloat16 and float8_e4m3fn. Any nonzero
    # error here is a real bug, not rounding.
    comptime tol = Float32(0.0)

    print(
        "  fmt=",
        "E2M3" if fmt == FP6Format.E2M3 else "E3M2",
        " rows=",
        num_rows,
        " cols=",
        num_cols,
        " dtype=",
        out_dtype,
        " scale_exp=",
        scale_exp,
    )

    comptime in_size = num_rows * packed_cols
    comptime scales_size = num_rows * scale_cols
    comptime out_size = num_rows * num_cols

    var in_host = ctx.enqueue_create_host_buffer[.uint8](in_size)
    var scales_host = ctx.enqueue_create_host_buffer[.uint8](scales_size)
    var expected_host = ctx.enqueue_create_host_buffer[out_dtype](out_size)

    for i in range(scales_size):
        scales_host[i] = scale_exp

    # Cycle through all 64 codes so every encoding path is exercised, and
    # offset per row so a row-indexing bug cannot alias into a pass.
    for row in range(num_rows):
        for group in range(packed_cols // 3):
            var quad = SIMD[.uint8, 4](0)
            for j in range(4):
                quad[j] = UInt8((group * 4 + j + row) % 64)
            var packed = pack_fp6_x4(quad)
            for b in range(3):
                in_host[row * packed_cols + group * 3 + b] = UInt8(
                    (packed >> UInt32(b * 8)) & UInt32(0xFF)
                )

    _cpu_dequant_mxfp6[fmt, out_dtype](
        expected_host.unsafe_ptr(),
        in_host.unsafe_ptr(),
        scales_host.unsafe_ptr(),
        num_rows,
        num_cols,
    )

    var in_device = ctx.enqueue_create_buffer[.uint8](in_size)
    var scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](scales_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](out_size)

    var scales_host_buf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        scales_size
    )
    for i in range(scales_size):
        scales_host_buf[i] = bitcast[.float8_e8m0fnu](scales_host[i])

    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(scales_device, scales_host_buf)
    ctx.synchronize()

    var in_tt = TileTensor(in_device, row_major[num_rows, packed_cols]())
    var scales_tt = TileTensor(scales_device, row_major[num_rows, scale_cols]())
    var out_tt = TileTensor(out_device, row_major[num_rows, num_cols]())

    dequant_mxfp6[fmt](
        ctx,
        out_tt,
        in_tt,
        scales_tt,
        num_rows=num_rows,
        num_cols=num_cols,
    )
    ctx.synchronize()

    var out_host_buf = ctx.enqueue_create_host_buffer[out_dtype](out_size)
    ctx.enqueue_copy(out_host_buf, out_device)
    ctx.synchronize()

    var max_err = Float32(0.0)
    var num_mismatches = 0
    for i in range(out_size):
        var got = out_host_buf[i].cast[.float32]()
        var want = expected_host[i].cast[.float32]()
        var err = abs(got - want)
        max_err = max(max_err, err)
        if err > tol:
            if num_mismatches < 5:
                print(
                    "    MISMATCH [",
                    i // num_cols,
                    ",",
                    i % num_cols,
                    "]: got=",
                    got,
                    " expected=",
                    want,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print("    FAIL: ", num_mismatches, " mismatches, max_err=", max_err)
        raise Error("MXFP6 dequant test failed")

    print("    PASS max_err=", max_err)


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP6 Dequant Smoke Tests")
        print("=========================")

        print("-- E2M3, scale = 1.0 --")
        test_mxfp6_dequant[FP6Format.E2M3, 64, 64](ctx, UInt8(127))
        test_mxfp6_dequant[FP6Format.E2M3, 128, 512](ctx, UInt8(127))
        test_mxfp6_dequant[FP6Format.E2M3, 256, 2880](ctx, UInt8(127))
        test_mxfp6_dequant[FP6Format.E2M3, 100, 192](ctx, UInt8(127))

        print("-- E3M2, scale = 1.0 --")
        test_mxfp6_dequant[FP6Format.E3M2, 64, 64](ctx, UInt8(127))
        test_mxfp6_dequant[FP6Format.E3M2, 128, 512](ctx, UInt8(127))

        print("-- Non-unit scales --")
        test_mxfp6_dequant[FP6Format.E2M3, 64, 256](ctx, UInt8(128))
        test_mxfp6_dequant[FP6Format.E2M3, 64, 256](ctx, UInt8(120))
        test_mxfp6_dequant[FP6Format.E3M2, 64, 256](ctx, UInt8(133))

        print("-- float8_e4m3fn output --")
        test_mxfp6_dequant[FP6Format.E2M3, 64, 256, .float8_e4m3fn](
            ctx, UInt8(127)
        )
        test_mxfp6_dequant[FP6Format.E3M2, 64, 256, .float8_e4m3fn](
            ctx, UInt8(127)
        )

        print("All MXFP6 dequant smoke tests passed.")
