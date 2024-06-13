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

from testing import assert_equal, assert_false, assert_true, assert_not_equal


def test_simple_uint():
    assert_equal(str(UInt(32)), "32")

    assert_equal(str(UInt(0)), "0")
    assert_equal(str(UInt()), "0")

    # (2 ** 64) - 1
    # TODO: raise an error in the future when
    # https://github.com/modularml/mojo/issues/2933 is fixed
    assert_equal(str(UInt(-1)), "18446744073709551615")

    assert_equal(str(UInt(18446744073709551615)), "18446744073709551615")


def test_uint_representation():
    assert_equal(repr(UInt(32)), "UInt(32)")

    assert_equal(repr(UInt(0)), "UInt(0)")
    assert_equal(repr(UInt()), "UInt(0)")

    assert_equal(repr(UInt(18446744073709551615)), "UInt(18446744073709551615)")


def test_equality():
    assert_equal(UInt(32), UInt(32))
    assert_equal(UInt(0), UInt(0))
    assert_equal(UInt(), UInt(0))
    assert_equal(UInt(18446744073709551615), UInt(18446744073709551615))
    assert_equal(
        UInt(18446744073709551615 - 10), UInt(18446744073709551615 - 10)
    )

    assert_true(UInt(32).__eq__(UInt(32)))
    assert_true(UInt(0).__eq__(UInt(0)))
    assert_true(UInt().__eq__(UInt(0)))
    assert_true(UInt(18446744073709551615).__eq__(UInt(18446744073709551615)))
    assert_true(
        UInt(18446744073709551615 - 10).__eq__(UInt(18446744073709551615 - 10))
    )

    assert_false(UInt(32).__eq__(UInt(0)))
    assert_false(UInt(0).__eq__(UInt(32)))
    assert_false(UInt(0).__eq__(UInt(18446744073709551615)))
    assert_false(UInt(18446744073709551615).__eq__(UInt(0)))
    assert_false(
        UInt(18446744073709551615).__eq__(UInt(18446744073709551615 - 10))
    )


def test_inequality():
    assert_not_equal(UInt(32), UInt(0))
    assert_not_equal(UInt(0), UInt(32))
    assert_not_equal(UInt(0), UInt(18446744073709551615))
    assert_not_equal(UInt(18446744073709551615), UInt(0))
    assert_not_equal(
        UInt(18446744073709551615), UInt(18446744073709551615 - 10)
    )

    assert_false(UInt(32).__ne__(UInt(32)))
    assert_false(UInt(0).__ne__(UInt(0)))
    assert_false(UInt().__ne__(UInt(0)))
    assert_false(UInt(18446744073709551615).__ne__(UInt(18446744073709551615)))
    assert_false(
        UInt(18446744073709551615 - 10).__ne__(UInt(18446744073709551615 - 10))
    )

    assert_true(UInt(32).__ne__(UInt(0)))
    assert_true(UInt(0).__ne__(UInt(32)))
    assert_true(UInt(0).__ne__(UInt(18446744073709551615)))
    assert_true(UInt(18446744073709551615).__ne__(UInt(0)))
    assert_true(
        UInt(18446744073709551615).__ne__(UInt(18446744073709551615 - 10))
    )


def main():
    test_simple_uint()
    test_uint_representation()
    test_equality()
    test_inequality()
