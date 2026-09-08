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

from std.math.uutils import ufloordiv, umod
from std.testing import TestSuite, assert_equal


def test_umod_int() raises:
    assert_equal(umod(0, 3), 0)
    assert_equal(umod(7, 3), 1)
    assert_equal(umod(9, 3), 0)
    assert_equal(umod(123, 100), 23)


def test_ufloordiv_int() raises:
    assert_equal(ufloordiv(0, 3), 0)
    assert_equal(ufloordiv(7, 3), 2)
    assert_equal(ufloordiv(9, 3), 3)
    assert_equal(ufloordiv(123, 100), 1)


def test_umod_scalar() raises:
    # The SIMD overload matches signed `%` for non-negative operands while
    # performing the reduction as unsigned.
    assert_equal(umod(Int32(7), Int32(3)), Int32(1))
    assert_equal(umod(Int64(123), Int64(100)), Int64(23))
    assert_equal(umod(UInt32(9), UInt32(3)), UInt32(0))


def test_umod_simd() raises:
    var a = SIMD[.int32, 4](0, 7, 9, 123)
    var b = SIMD[.int32, 4](3, 3, 3, 100)
    var expected = SIMD[.int32, 4](0, 1, 0, 23)
    assert_equal(umod(a, b), expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
