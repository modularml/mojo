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

from std.testing import assert_equal, TestSuite

from std.utils.numerics import isnan


def test_abs() raises:
    assert_equal(0, abs(0))
    assert_equal(1, abs(1))
    assert_equal(1, abs(-1))

    var lhs = SIMD[.int32, 4](1, -2, 3, -4)
    var expected = SIMD[.int32, 4](1, 2, 3, 4)
    assert_equal(expected, abs(lhs))


def test_divmod() raises:
    var t = divmod(0, 1)
    assert_equal(0, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 1)
    assert_equal(1, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 2)
    assert_equal(0, t[0])
    assert_equal(1, t[1])
    t = divmod(4, 3)
    assert_equal(1, t[0])
    assert_equal(1, t[1])


def test_min() raises:
    assert_equal(-2, min(-2, -1))
    assert_equal(-1, min(0, -1))
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    assert_equal(UInt(0), min(UInt(0), UInt(1)))
    assert_equal(
        UInt(1),
        min(UInt(1), UInt(42)),
    )

    comptime F = SIMD[.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(min(f, F(-9.0, -6.0, -4.0, 10.5)), F(-10.5, -6.0, -4.0, 10.0))
    assert_equal(min(f, -4.0), F(-10.5, -5.0, -4.0, -4.0))

    comptime I = SIMD[.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(min(i, I(-9, -6, -4, 11)), I(-10, -6, -4, 10))
    assert_equal(min(i, -4), I(-10, -5, -4, -4))

    assert_equal(min(1), 1)
    assert_equal(min(1, 2, 3, 4), 1)
    assert_equal(min(500, 1, 2, 3, 4), 1)


def test_max() raises:
    assert_equal(-1, max(-2, -1))
    assert_equal(0, max(0, -1))
    assert_equal(1, max(0, 1))
    assert_equal(2, max(2, 1))

    assert_equal(UInt(1), max(UInt(0), UInt(1)))
    assert_equal(UInt(2), max(UInt(1), UInt(2)))

    comptime F = SIMD[.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(max(f, F(-9.0, -6.0, -4.0, 10.5)), F(-9.0, -5.0, 5.0, 10.5))
    assert_equal(max(f, -4.0), F(-4.0, -4.0, 5.0, 10.0))

    comptime I = SIMD[.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(max(i, I(-9, -6, -4, 11)), I(-9, -5, 5, 11))
    assert_equal(max(i, -4), I(-4, -4, 5, 10))

    assert_equal(max(1), 1)
    assert_equal(max(1, 2, 3, 4), 4)
    assert_equal(max(-10, 2, 3, 4, -10), 4)


def test_round() raises:
    assert_equal(0.0, round(0.0))
    assert_equal(1.0, round(1.0))
    assert_equal(1.0, round(1.1))
    assert_equal(1.0, round(1.4))
    assert_equal(2.0, round(1.5))
    assert_equal(2.0, round(2.0))
    assert_equal(1.0, round(1.4, 0))
    assert_equal(2.0, round(2.5))
    assert_equal(-2.0, round(-2.5))

    assert_equal(1.5, round(1.5, 1))
    assert_equal(1.61, round(1.613, 2))

    var lhs = SIMD[.float32, 4](1.1, 1.5, 1.9, 2.0)
    var expected = SIMD[.float32, 4](1.0, 2.0, 2.0, 2.0)
    assert_equal(expected, round(lhs))

    # Ensure that round works on float literal
    comptime r1 = round(2.3)
    assert_equal(r1, 2.0)
    comptime r2 = round(2.3324, 2)
    assert_equal(r2, 2.33)


def test_pow() raises:
    comptime F = SIMD[.float32, 4]
    var base = F(0.0, 1.0, 2.0, 3.0)
    assert_equal(pow(base, Float32(2.0)), F(0.0, 1.0, 4.0, 9.0))
    assert_equal(pow(base, Int(2)), F(0.0, 1.0, 4.0, 9.0))
    comptime I = SIMD[.int32, 4]
    assert_equal(pow(I(0, 1, 2, 3), Int(2)), I(0, 1, 4, 9))


def test_isnan() raises:
    # Check that we can run llvm intrinsics returning bool at comptime.
    comptime x1 = isnan(SIMD[.float32, 4](SIMD[.float64, 4](1.0)))
    assert_equal(x1, SIMD[.bool, 4](fill=False))

    comptime x2 = isnan(SIMD[.float32, 4](FloatLiteral.nan))
    assert_equal(x2, SIMD[.bool, 4](fill=True))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
