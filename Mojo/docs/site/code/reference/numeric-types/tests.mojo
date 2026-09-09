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
# tests.mojo
# Tests for numeric-types.mdx code examples.
#
# Not tested (no runnable behavior to assert from this file):
#   - `reflect[type_of(a)].name()` prints (Int naming demos)
#   - `Float8` arithmetic (requires GPU; covered in the support-matrix
#     table; CPU execution is unsupported)
#   - `Float4_e2m1fn` operations (requires NVIDIA Blackwell)
#   - `BFloat16` arithmetic on Apple Silicon (unsupported per doc)
#   - `comptime big = 2 ** 200` materialization (IntLiteral arbitrary
#     precision; no straightforward runtime assertion)
from std.testing import assert_equal
from std.sys import bit_width_of

# --- SIMD construction and broadcast multiply ---


def test_simd_broadcast_mul() raises:
    var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var doubled = v * 2.0
    assert_equal(doubled[0], 2.0)
    assert_equal(doubled[1], 4.0)
    assert_equal(doubled[2], 6.0)
    assert_equal(doubled[3], 8.0)


# --- Element access by lane ---


def test_simd_element_access() raises:
    var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_equal(v[0], 1.0)
    v[0] = 5.0
    assert_equal(v[0], 5.0)


# --- Element-wise arithmetic ---


def test_simd_arithmetic() raises:
    var a = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0)
    var sum = a + b
    assert_equal(sum[0], 6.0)
    assert_equal(sum[3], 12.0)
    var prod = a * b
    assert_equal(prod[0], 5.0)
    assert_equal(prod[3], 32.0)


# --- Reductions ---


def test_simd_reductions() raises:
    var a = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_equal(a.reduce_add(), 10.0)
    assert_equal(a.reduce_max(), 4.0)
    assert_equal(a.reduce_min(), 1.0)


# --- Casting between SIMD dtypes ---


def test_simd_cast() raises:
    var a = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var ints = a.cast[.int32]()
    assert_equal(ints[0], 1)
    assert_equal(ints[3], 4)
    var wide = a.cast[.float64]()
    assert_equal(wide[0], 1.0)
    assert_equal(wide[3], 4.0)
    var tiny = a.cast[.float16]()
    assert_equal(tiny[0], 1.0)
    assert_equal(tiny[3], 4.0)


# --- Clamp (both bounds inclusive) ---


def test_simd_clamp() raises:
    var a = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var clamped = a.clamp(1.5, 3.5)
    assert_equal(clamped[0], 1.5)
    assert_equal(clamped[1], 2.0)
    assert_equal(clamped[2], 3.0)
    assert_equal(clamped[3], 3.5)


# --- Free-function min / max ---


def test_min_max() raises:
    var a = SIMD[.float32, 4](1.0, 2.0, 7.0, 4.0)
    var b = SIMD[.float32, 4](5.0, 1.0, 3.0, 8.0)
    var minimum = min(a, b)
    assert_equal(minimum[0], 1.0)
    assert_equal(minimum[2], 3.0)
    var maximum = max(a, b)
    assert_equal(maximum[0], 5.0)
    assert_equal(maximum[2], 7.0)


# --- Scalar aliases ---


def test_scalar_aliases() raises:
    var a: Scalar[.float32] = 3.14
    var b: Float32 = 3.14
    var c: SIMD[.float32, 1] = 3.14
    assert_equal(a, b)
    assert_equal(b, c)


# --- Parameterized numeric function using DType ---


def double[T: DType](x: Scalar[T]) -> Scalar[T]:
    return x * UInt8(2).cast[T]()


def test_dtype_parameterized() raises:
    assert_equal(double[.float32](3.5), 7.0)
    assert_equal(double[.int16](21), 42)


# --- Int constants ---


def test_int_constants() raises:
    # BITWIDTH is platform-dependent; on supported platforms today it's 64.
    assert_equal(bit_width_of[DType.int](), 64)
    assert_equal(Int.MAX, 9223372036854775807)
    assert_equal(Int.MIN, -9223372036854775808)


# --- Sized integer bounds ---


def test_sized_int_bounds() raises:
    assert_equal(UInt8.MAX, 255)
    assert_equal(Int8.MIN, -128)
    assert_equal(UInt32.MAX, 4294967295)
    assert_equal(Int32.MIN, -2147483648)
    assert_equal(SIMD[.int16, 1].MIN, -32768)


# --- Byte is UInt8 ---


def test_byte_alias() raises:
    var buf: List[Byte] = [0x48, 0x65, 0x6C, 0x6C, 0x6F]
    assert_equal(len(buf), 5)
    assert_equal(buf[0], 0x48)
    # Byte and UInt8 are the same type
    var b: Byte = 65
    var u: UInt8 = 65
    assert_equal(b, u)


# --- Float MAX / MIN / MAX_FINITE / MIN_FINITE ---


def test_float_bounds() raises:
    # Float32.MAX_FINITE is the largest finite Float32 value
    var max_f = Float32.MAX_FINITE
    var min_f = Float32.MIN_FINITE
    assert_equal(max_f > 0.0, True)
    assert_equal(min_f < 0.0, True)
    # MIN_FINITE is the most negative finite value, equal to -MAX_FINITE
    assert_equal(min_f, -max_f)


# --- Floating-point rounding error (comptime vs runtime) ---


def test_float_rounding() raises:
    comptime exact = 3.0 * (4.0 / 3.0 - 1.0)
    var three = 3.0
    var finite = three * (4.0 / three - 1.0)
    # Comptime arithmetic is exact, so this is exactly 1.0
    assert_equal(exact, 1.0)
    # Runtime arithmetic accumulates rounding error
    assert_equal(exact == finite, False)


# --- IntLiteral materialization into different types ---


def test_int_literal_materialization() raises:
    var a: Int = 42
    var b: Int8 = 42
    var c: Float32 = 42
    var d: UInt64 = 1_000_000
    assert_equal(a, 42)
    assert_equal(b, 42)
    assert_equal(c, 42.0)
    assert_equal(d, 1_000_000)


# --- FloatLiteral materialization into different float types ---


def test_float_literal_materialization() raises:
    var x: Float32 = 3.14
    var y: Float64 = 3.14
    var z: BFloat16 = 0.5
    # Approximate equality across widths: round-trip through Float64
    assert_equal(x.cast[.float64]() > 3.13, True)
    assert_equal(x.cast[.float64]() < 3.15, True)
    assert_equal(y, 3.14)
    assert_equal(z, 0.5)


# --- FloatLiteral special-value constants ---


def test_float_literal_specials() raises:
    # FloatLiteral exposes nan, infinity, negative_infinity, negative_zero
    assert_equal(FloatLiteral.nan.is_nan(), True)
    assert_equal(FloatLiteral.negative_zero.is_neg_zero(), True)
    # infinity > any finite Float64
    var inf: Float64 = FloatLiteral.infinity
    assert_equal(inf > 1.0e308, True)
    # negative_infinity < any finite Float64
    var neg_inf: Float64 = FloatLiteral.negative_infinity
    assert_equal(neg_inf < -1.0e308, True)


# --- Literals adapt to context type ---


def test_literal_adapts() raises:
    var x = Float32(1.0)
    var y = x * 0.5  # 0.5 becomes Float32
    var z = x + 2  # 2 becomes Float32
    assert_equal(y, 0.5)
    assert_equal(z, 3.0)


# --- Explicit constructor conversions ---


def test_explicit_conversions() raises:
    var i = 42  # Int
    var f = Float32(i)
    var u = UInt64(i)
    var narrow = Int8(i)
    assert_equal(f, 42.0)
    assert_equal(u, 42)
    assert_equal(narrow, 42)


# --- SIMD .cast[] between dtypes ---


def test_cast_method() raises:
    var a = Float32(3.14)
    var b = a.cast[.int32]()  # Float32 → Int32 (truncates)
    var c = a.cast[.float64]()
    assert_equal(b, 3)
    # Float32 → Float64 widens; some bit-pattern drift is acceptable
    assert_equal(c > 3.13, True)
    assert_equal(c < 3.15, True)


# --- Int ↔ SIMD-based via constructors ---


def test_int_simd_roundtrip() raises:
    var i = 42  # Int
    var s = Int64(i)
    var back = Int(s)
    assert_equal(s, 42)
    assert_equal(back, 42)


# --- Sharp edge: integer overflow wraps (two's complement) ---


def test_int_overflow_wraps() raises:
    var x = Int8(127)
    var y = x + Int8(1)  # wraps to -128
    assert_equal(y, -128)
    var u = UInt8(255)
    var v = u + UInt8(1)  # wraps to 0
    assert_equal(v, 0)


# --- Sharp edge: float-to-int truncates toward zero ---


def test_float_to_int_truncates() raises:
    assert_equal(Int(Float32(3.9)), 3)
    assert_equal(Int(Float32(-3.9)), -3)


# --- Sharp edge: NaN comparison always returns False ---


def test_nan_comparisons() raises:
    var x = Float32.MAX_FINITE * 2.0  # +inf (overflow)
    var nan = x - x  # inf - inf = NaN
    # NaN is never equal to itself
    assert_equal(nan == nan, False)
    # NaN comparisons of all kinds return False
    assert_equal(nan < 0.0, False)
    assert_equal(nan > 0.0, False)


def main() raises:
    test_simd_broadcast_mul()
    test_simd_element_access()
    test_simd_arithmetic()
    test_simd_reductions()
    test_simd_cast()
    test_simd_clamp()
    test_min_max()
    test_scalar_aliases()
    test_dtype_parameterized()
    test_int_constants()
    test_sized_int_bounds()
    test_byte_alias()
    test_float_bounds()
    test_float_rounding()
    test_int_literal_materialization()
    test_float_literal_materialization()
    test_float_literal_specials()
    test_literal_adapts()
    test_explicit_conversions()
    test_cast_method()
    test_int_simd_roundtrip()
    test_int_overflow_wraps()
    test_float_to_int_truncates()
    test_nan_comparisons()
