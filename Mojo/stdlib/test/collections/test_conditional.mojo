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

from std.collections._conditional import _ComptimeConditional
from std.sys import size_of

from std.testing import *
from std.testing import TestSuite


def test_engaged_init_and_getitem() raises:
    var c = _ComptimeConditional[Int, engaged=True](42)
    assert_equal(c[], 42)


def test_engaged_implicit_conversion() raises:
    var c: _ComptimeConditional[Int, engaged=True] = 99
    assert_equal(c[], 99)


def test_engaged_getitem_returns_ref() raises:
    var c = _ComptimeConditional[Int, engaged=True](10)
    assert_equal(c[], 10)
    c[] = 20
    assert_equal(c[], 20)


def test_disengaged_default_construction() raises:
    var _c = _ComptimeConditional[Int, engaged=False]()


def test_engaged_size_equals_inner_type() raises:
    assert_equal(
        size_of[_ComptimeConditional[Byte, engaged=True]](),
        size_of[Byte](),
    )
    assert_equal(
        size_of[_ComptimeConditional[Float64, engaged=True]](),
        size_of[Float64](),
    )


def test_disengaged_size_is_zero() raises:
    assert_equal(
        size_of[_ComptimeConditional[Byte, engaged=False]](),
        0,
    )
    assert_equal(
        size_of[_ComptimeConditional[Float64, engaged=False]](),
        0,
    )


def test_engaged_copy() raises:
    var a = _ComptimeConditional[Int, engaged=True](7)
    var b = a
    assert_equal(a[], 7)
    assert_equal(b[], 7)
    # Mutating b should not affect a.
    b[] = 100
    assert_equal(a[], 7)
    assert_equal(b[], 100)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
