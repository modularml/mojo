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
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](0, 1, 3, 4)
    assert_equal(expected, lhs.min(rhs))
    assert_equal(expected, rhs.min(lhs))


def test_max():
    assert_equal(1, max(0, 1))
    assert_equal(2, max(1, 2))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](1, 2, 5, 7)
    assert_equal(expected, lhs.max(rhs))
    assert_equal(expected, rhs.max(lhs))


def test_round():
    assert_equal(0, round(0.0))
    assert_equal(1, round(1.0))
    assert_equal(1, round(1.1))
    assert_equal(1, round(1.4))
    assert_equal(2, round(1.5))
    assert_equal(2, round(2.0))
    assert_equal(1, round(1.4, 0))
    # FIXME: round(2.5) is 2.0 (roundeven) but it's using roundhalfup
    # Fix when the math libray is open sourced
    # assert_equal(2, round(2.5))
    # assert_equal(1.5, round(1.5, 1))
    # assert_equal(1.61, round(1.613, 2))

    var lhs = SIMD[DType.float32, 4](1.1, 1.5, 1.9, 2.0)
    var expected = SIMD[DType.float32, 4](1.0, 2.0, 2.0, 2.0)
    assert_equal(expected, round(lhs))


def main():
    test_abs()
    test_divmod()
    test_max()
    test_min()
    test_round()
