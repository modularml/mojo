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
"""Tests for AMD MXFP4 activation quantization kernel (MI355X / CDNA4).

Round-trip test: quantize BF16 -> MXFP4 (packed uint8 + E8M0 scales),
then dequant on CPU and compare against the original BF16 values.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.block_scaled_quantization import quantize_mx_amd
from linalg.fp4_utils import E2M1_TO_FLOAT32, MXFP4_SF_VECTOR_SIZE
from linalg.mxfp4_dequant import dequant_mxfp4


def _e8m0_to_float32(bits: UInt8) -> Float32:
    """Convert E8M0 exponent byte to float32 power-of-two.

    E8M0 biased exponent 0 is technically 2^-127, not zero. We map it to 0.0
    because the quantization kernel only stores e8m0=0 for all-zero blocks.
    """
    if bits == UInt8(0):
        return Float32(0.0)
    var f32_bits = UInt32(bits) << UInt32(23)
    return bitcast[.float32](f32_bits)


def _dequant_element(packed_byte: UInt8, low: Bool, scale: Float32) -> Float32:
    """Unpack one FP4 nibble from a packed byte and dequantize it."""
    var nibble: UInt8
    if low:
        nibble = packed_byte & 0x0F
    else:
        nibble = (packed_byte >> 4) & 0x0F
    return E2M1_TO_FLOAT32[Int(nibble)] * scale


def test_quantize_roundtrip[M: Int, K: Int](ctx: DeviceContext) raises:
    """Quantize BF16 -> MXFP4 on GPU, dequant on CPU, check round-trip."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, MXFP4_SF_VECTOR_SIZE)

    print("  M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with input_dev.map_to_host() as h:
        for i in range(M * K):
            h[i] = random_float64(-3.0, 3.0).cast[.bfloat16]()

    var output_dev = ctx.enqueue_create_buffer[.uint8](M * packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_K)

    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_K])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var input_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var output_host = ctx.enqueue_create_host_buffer[.uint8](M * packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M * scale_K
    )
    ctx.enqueue_copy(input_host, input_dev)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    for row in range(M):
        for block in range(scale_K):
            var scale_bits = scales_host[row * scale_K + block]
            var scale_f32 = _e8m0_to_float32(bitcast[.uint8](scale_bits))

            for elem in range(MXFP4_SF_VECTOR_SIZE):
                var col = block * MXFP4_SF_VECTOR_SIZE + elem
                if col >= K:
                    break
                var byte_idx = row * packed_K + col // 2
                var dequanted = _dequant_element(
                    output_host[byte_idx], col % 2 == 0, scale_f32
                )
                var original = input_host[row * K + col].cast[.float32]()
                if abs(original) >= 0.5:
                    var rel_err = abs(dequanted - original) / abs(original)
                    max_rel_err = max(max_rel_err, rel_err)

    print("    max_rel_err=", max_rel_err)

    # MXFP4 uses block-scaled FP4: each 32-element block gets an E8M0
    # (power-of-2) scale factor. With seeded inputs in [-3, 3], observed
    # max relative error is ~20% for |value| >= 0.5.
    assert_true(
        max_rel_err < 0.25,
        "max relative error " + String(max_rel_err) + " exceeds 25%",
    )


def test_quantize_dequant_gpu_roundtrip[
    M: Int, K: Int
](ctx: DeviceContext) raises:
    """Quantize BF16 -> MXFP4 on GPU, dequant MXFP4 -> BF16 on GPU via
    dequant_mxfp4, compare against original. Validates layout compatibility
    with the matmul pipeline's dequant kernel."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, MXFP4_SF_VECTOR_SIZE)

    print("  M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with input_dev.map_to_host() as h:
        for i in range(M * K):
            h[i] = random_float64(-3.0, 3.0).cast[.bfloat16]()

    var packed_dev = ctx.enqueue_create_buffer[.uint8](M * packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_K)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var packed_tt = TileTensor(packed_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_K])))
    quantize_mx_amd(ctx, packed_tt, scales_tt, input_tt)

    var output_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[K])))
    dequant_mxfp4(ctx, output_tt, packed_tt, scales_tt, num_rows=M, num_cols=K)

    var input_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var output_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    ctx.enqueue_copy(input_host, input_dev)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    for i in range(M * K):
        var original = input_host[i].cast[.float32]()
        var dequanted = output_host[i].cast[.float32]()
        if abs(original) >= 0.5:
            var rel_err = abs(dequanted - original) / abs(original)
            max_rel_err = max(max_rel_err, rel_err)

    print("    max_rel_err=", max_rel_err)
    assert_true(
        max_rel_err < 0.25,
        "max relative error " + String(max_rel_err) + " exceeds 25%",
    )


def test_quantize_all_zeros[M: Int, K: Int](ctx: DeviceContext) raises:
    """Verify all-zero input produces zero nibbles and e8m0=0 scale."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, MXFP4_SF_VECTOR_SIZE)

    print("  M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](M * K)
    with input_dev.map_to_host() as h:
        for i in range(M * K):
            h[i] = BFloat16(0.0)

    var output_dev = ctx.enqueue_create_buffer[.uint8](M * packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * scale_K)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_K])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var output_host = ctx.enqueue_create_host_buffer[.uint8](M * packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        M * scale_K
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    for row in range(M):
        for block in range(scale_K):
            var s = bitcast[.uint8](scales_host[row * scale_K + block])
            assert_true(
                s == UInt8(0),
                "expected e8m0 scale=0 for all-zero block, got "
                + String(Int(s)),
            )

    for i in range(M * packed_K):
        assert_true(
            output_host[i] == UInt8(0),
            "expected packed byte=0, got " + String(Int(output_host[i])),
        )


def test_quantize_known_scales(ctx: DeviceContext) raises:
    """Verify exact E8M0 scale values for blocks with known max values.

    Block 0: all 1.0 -> scale = ceil_pow2(1/6) = 0.25 (e8m0=125)
    Block 1: all 3.0 -> scale = ceil_pow2(3/6) = 0.5  (e8m0=126)
    Block 2: all 6.0 -> scale = ceil_pow2(6/6) = 1.0  (e8m0=127)
    """
    comptime M = 1
    comptime K = 96  # 3 blocks of 32
    comptime packed_K = K // 2
    comptime scale_K = 3

    print("  known_scales test M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](K)
    with input_dev.map_to_host() as h:
        for i in range(32):
            h[i] = BFloat16(1.0)
        for i in range(32, 64):
            h[i] = BFloat16(3.0)
        for i in range(64, 96):
            h[i] = BFloat16(6.0)

    var output_dev = ctx.enqueue_create_buffer[.uint8](packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](scale_K)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[scale_K])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var output_host = ctx.enqueue_create_host_buffer[.uint8](packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](scale_K)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    @__parameter
    def _check_block(
        blk: Int, expected_e8m0: Int, expected_val: Float32
    ) raises:
        var bits = Int(bitcast[.uint8](scales_host[blk]))
        var scale_f32 = _e8m0_to_float32(UInt8(bits))
        print("    block", blk, ": e8m0=", bits, " scale=", scale_f32)
        assert_true(
            bits == expected_e8m0,
            "block "
            + String(blk)
            + ": expected e8m0="
            + String(expected_e8m0)
            + " got "
            + String(bits),
        )
        for elem in range(32):
            var col = blk * 32 + elem
            var dequanted = _dequant_element(
                output_host[col // 2], col % 2 == 0, scale_f32
            )
            assert_true(
                dequanted == expected_val,
                "block "
                + String(blk)
                + " elem "
                + String(elem)
                + ": expected "
                + String(expected_val)
                + " got "
                + String(dequanted),
            )

    _check_block(0, 125, 1.0)
    _check_block(1, 126, 3.0)
    _check_block(2, 127, 6.0)


def test_quantize_saturation(ctx: DeviceContext) raises:
    """Verify values beyond FP4 max (6.0) get correct scales."""
    comptime M = 1
    comptime K = 32
    comptime packed_K = K // 2

    print("  saturation test M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](K)
    with input_dev.map_to_host() as h:
        for i in range(K):
            if i % 3 == 0:
                h[i] = BFloat16(6.0)
            elif i % 3 == 1:
                h[i] = BFloat16(12.0)
            else:
                h[i] = BFloat16(24.0)

    var output_dev = ctx.enqueue_create_buffer[.uint8](packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](1)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[1])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var input_host = ctx.enqueue_create_host_buffer[.bfloat16](K)
    var output_host = ctx.enqueue_create_host_buffer[.uint8](packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](1)
    ctx.enqueue_copy(input_host, input_dev)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    # ceil_pow2(24/6) = 4.0 = 2^2 -> e8m0 = 129
    var scale_bits = Int(bitcast[.uint8](scales_host[0]))
    print("    scale_e8m0=", scale_bits)
    assert_true(
        scale_bits == 129,
        "expected e8m0=129 (scale=4.0), got " + String(scale_bits),
    )

    var scale_f32 = _e8m0_to_float32(UInt8(scale_bits))
    var max_rel_err = Float32(0.0)
    for col in range(K):
        var dequanted = _dequant_element(
            output_host[col // 2], col % 2 == 0, scale_f32
        )
        var original = input_host[col].cast[.float32]()
        var rel_err = abs(dequanted - original) / abs(original)
        max_rel_err = max(max_rel_err, rel_err)

    print("    max_rel_err=", max_rel_err)
    assert_true(
        max_rel_err < 0.25,
        "max relative error " + String(max_rel_err) + " exceeds 25%",
    )


def test_quantize_negative(ctx: DeviceContext) raises:
    """Verify all-negative input preserves sign and gets correct scale."""
    comptime M = 1
    comptime K = 32
    comptime packed_K = K // 2

    print("  negative test M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](K)
    with input_dev.map_to_host() as h:
        for i in range(K):
            var val = -Float64(0.5) * Float64((i % 12) + 1)
            h[i] = val.cast[.bfloat16]()

    var output_dev = ctx.enqueue_create_buffer[.uint8](packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](1)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[1])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var output_host = ctx.enqueue_create_host_buffer[.uint8](packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](1)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    var scale_bits = Int(bitcast[.uint8](scales_host[0]))
    print("    scale_e8m0=", scale_bits)
    assert_true(
        scale_bits == 127,
        "expected e8m0=127 (scale=1.0), got " + String(scale_bits),
    )

    var scale_f32 = _e8m0_to_float32(UInt8(scale_bits))
    var sign_errors = 0
    for col in range(K):
        var dequanted = _dequant_element(
            output_host[col // 2], col % 2 == 0, scale_f32
        )
        if dequanted > 0.0:
            sign_errors += 1

    print("    sign_errors=", sign_errors)
    assert_true(
        sign_errors == 0,
        String(sign_errors) + " values have wrong sign (expected all <= 0)",
    )


def test_quantize_even_mode_boundary(ctx: DeviceContext) raises:
    """Probe a block where even-mode scale and ceil(max/6) diverge."""
    comptime M = 1
    comptime K = 32
    comptime packed_K = K // 2

    print("  even-mode boundary test M=", M, " K=", K)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](K)
    with input_dev.map_to_host() as h:
        h[0] = Float64(1.6).cast[.bfloat16]()
        for i in range(1, K):
            h[i] = Float64(0.4).cast[.bfloat16]()

    var output_dev = ctx.enqueue_create_buffer[.uint8](packed_K)
    var scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](1)
    var input_tt = TileTensor(input_dev, row_major((Idx[M], Idx[K])))
    var output_tt = TileTensor(output_dev, row_major((Idx[M], Idx[packed_K])))
    var scales_tt = TileTensor(scales_dev, row_major((Idx[M], Idx[1])))
    quantize_mx_amd(ctx, output_tt, scales_tt, input_tt)

    var output_host = ctx.enqueue_create_host_buffer[.uint8](packed_K)
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](1)
    ctx.enqueue_copy(output_host, output_dev)
    ctx.enqueue_copy(scales_host, scales_dev)
    ctx.synchronize()

    var scale_bits = Int(bitcast[.uint8](scales_host[0]))
    var scale_f32 = _e8m0_to_float32(UInt8(scale_bits))
    var max_dequanted = _dequant_element(output_host[0], True, scale_f32)
    var small_dequanted = _dequant_element(output_host[0], False, scale_f32)
    print(
        "    scale_e8m0=",
        scale_bits,
        " scale=",
        scale_f32,
        " max_dequant=",
        max_dequanted,
        " small_dequant=",
        small_dequanted,
    )
    assert_true(
        scale_bits == 125,
        "expected even-mode e8m0=125 (scale=0.25), got " + String(scale_bits),
    )
    assert_true(
        abs(max_dequanted - 1.5) < 0.001,
        "expected max value to dequantize to 1.5, got " + String(max_dequanted),
    )
    assert_true(
        abs(small_dequanted - 0.375) < 0.001,
        "expected small value to dequantize to 0.375, got "
        + String(small_dequanted),
    )


def main() raises:
    seed(42)
    var ctx = DeviceContext()

    print("test_quantize_roundtrip (CPU dequant):")
    test_quantize_roundtrip[4, 64](ctx)
    test_quantize_roundtrip[1, 128](ctx)
    test_quantize_roundtrip[16, 256](ctx)
    test_quantize_roundtrip[32, 1024](ctx)
    test_quantize_roundtrip[4, 7168](ctx)
    test_quantize_roundtrip[1, 2048](ctx)

    print("test_quantize_all_zeros:")
    test_quantize_all_zeros[4, 64](ctx)
    test_quantize_all_zeros[1, 32](ctx)

    print("test_quantize_known_scales:")
    test_quantize_known_scales(ctx)

    print("test_quantize_saturation:")
    test_quantize_saturation(ctx)

    print("test_quantize_negative:")
    test_quantize_negative(ctx)

    print("test_quantize_even_mode_boundary:")
    test_quantize_even_mode_boundary(ctx)

    print("test_quantize_min_k (K=32):")
    test_quantize_roundtrip[1, 32](ctx)
    test_quantize_roundtrip[4, 32](ctx)

    print("test_quantize_dequant_gpu_roundtrip:")
    test_quantize_dequant_gpu_roundtrip[4, 64](ctx)
    test_quantize_dequant_gpu_roundtrip[16, 1024](ctx)
    test_quantize_dequant_gpu_roundtrip[4, 7168](ctx)
    print("PASS")
