# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from testing import assert_equal


def test_abs():
    assert_equal(0, abs(0))
    assert_equal(1, abs(1))
    assert_equal(1, abs(-1))

    var lhs = SIMD[DType.int32, 4](1, -2, 3, -4)
    var expected = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(expected, abs(lhs))


def test_divmod():
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


def test_min():
    assert_equal(-2, min(-2, -1))
    assert_equal(-1, min(0, -1))
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    assert_equal(UInt(0), min(UInt(0), UInt(1)))
    assert_equal(UInt(1), min(UInt(1), UInt(42)))

    alias F = SIMD[DType.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(min(f, F(-9.0, -6.0, -4.0, 10.5)), F(-10.5, -6.0, -4.0, 10.0))
    assert_equal(min(f, -4.0), F(-10.5, -5.0, -4.0, -4.0))

    alias I = SIMD[DType.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(min(i, I(-9, -6, -4, 11)), I(-10, -6, -4, 10))
    assert_equal(min(i, -4), I(-10, -5, -4, -4))


def test_max():
    assert_equal(-1, max(-2, -1))
    assert_equal(0, max(0, -1))
    assert_equal(1, max(0, 1))
    assert_equal(2, max(2, 1))

    assert_equal(UInt(1), max(UInt(0), UInt(1)))
    assert_equal(UInt(2), max(UInt(1), UInt(2)))

    alias F = SIMD[DType.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(max(f, F(-9.0, -6.0, -4.0, 10.5)), F(-9.0, -5.0, 5.0, 10.5))
    assert_equal(max(f, -4.0), F(-4.0, -4.0, 5.0, 10.0))

    alias I = SIMD[DType.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(max(i, I(-9, -6, -4, 11)), I(-9, -5, 5, 11))
    assert_equal(max(i, -4), I(-4, -4, 5, 10))


def test_round():
    assert_equal(0, round(0.0))
    assert_equal(1, round(1.0))
    assert_equal(1, round(1.1))
    assert_equal(1, round(1.4))
    assert_equal(2, round(1.5))
    assert_equal(2, round(2.0))
    assert_equal(1, round(1.4, 0))
    assert_equal(2, round(2.5))
    assert_equal(1.5, round(1.5, 1))
    assert_equal(1.61, round(1.613, 2))

    var lhs = SIMD[DType.float32, 4](1.1, 1.5, 1.9, 2.0)
    var expected = SIMD[DType.float32, 4](1.0, 2.0, 2.0, 2.0)
    assert_equal(expected, round(lhs))

    # Ensure that round works on float literal
    alias r1: FloatLiteral = round(2.3)
    assert_equal(r1, 2.0)
    alias r2: FloatLiteral = round(2.3324, 2)
    assert_equal(r2, 2.33)


def test_pow():
    alias F = SIMD[DType.float32, 4]
    var base = F(0.0, 1.0, 2.0, 3.0)
    assert_equal(pow(base, 2.0), F(0.0, 1.0, 4.0, 9.0))
    assert_equal(pow(base, int(2)), F(0.0, 1.0, 4.0, 9.0))
    alias I = SIMD[DType.int32, 4]
    assert_equal(pow(I(0, 1, 2, 3), int(2)), I(0, 1, 4, 9))


def main():
    test_abs()
    test_divmod()
    test_max()
    test_min()
    test_round()
    test_pow()
