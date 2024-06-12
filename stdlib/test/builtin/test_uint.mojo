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

from testing import assert_equal, assert_true, assert_false

# TODO: compute those with ** when IntLiteral supports __pow__()
alias ABOVE_32_BITS = 4398046511104  # 2**42
alias CLOSE_TO_UINTMAX = 18446744073709551605  # 2**64 - 10
alias UINTMAX = 18446744073709551615  # 2**64 - 1


def test_uint_lower_than():
    assert_true(UInt(1) < UInt(2))
    assert_false(UInt(2) < UInt(1))

    assert_true(UInt(0) < UInt(1))
    assert_false(UInt(1) < UInt(0))

    assert_true(UInt(0) < UInt(UINTMAX))
    assert_false(UInt(UINTMAX) < UInt(0))

    assert_true(UInt(1) < UInt(UINTMAX))
    assert_false(UInt(UINTMAX) < UInt(1))

    assert_true(UInt(1) < UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) < UInt(1))

    assert_true(UInt(0) < UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) < UInt(0))

    assert_true(UInt(CLOSE_TO_UINTMAX) < UInt(UINTMAX))
    assert_false(UInt(UINTMAX) < UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(ABOVE_32_BITS) < UInt(UINTMAX))
    assert_false(UInt(UINTMAX) < UInt(ABOVE_32_BITS))

    assert_false(UInt(0) < UInt(0))
    assert_false(UInt(1) < UInt(1))
    assert_false(UInt(ABOVE_32_BITS) < UInt(ABOVE_32_BITS))
    assert_false(UInt(UINTMAX) < UInt(UINTMAX))


def test_uint_lower_equal():
    assert_true(UInt(1) <= UInt(2))
    assert_false(UInt(2) <= UInt(1))

    assert_true(UInt(0) <= UInt(1))
    assert_false(UInt(1) <= UInt(0))

    assert_true(UInt(0) <= UInt(UINTMAX))
    assert_false(UInt(UINTMAX) <= UInt(0))

    assert_true(UInt(1) <= UInt(UINTMAX))
    assert_false(UInt(UINTMAX) <= UInt(1))

    assert_true(UInt(1) <= UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) <= UInt(1))

    assert_true(UInt(0) <= UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) <= UInt(0))

    assert_true(UInt(CLOSE_TO_UINTMAX) <= UInt(UINTMAX))
    assert_false(UInt(UINTMAX) <= UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(ABOVE_32_BITS) <= UInt(UINTMAX))
    assert_false(UInt(UINTMAX) <= UInt(ABOVE_32_BITS))

    assert_true(UInt(0) <= UInt(0))
    assert_true(UInt(1) <= UInt(1))
    assert_true(UInt(ABOVE_32_BITS) <= UInt(ABOVE_32_BITS))
    assert_true(UInt(UINTMAX) <= UInt(UINTMAX))


def test_uint_greater_than():
    assert_true(UInt(2) > UInt(1))
    assert_false(UInt(1) > UInt(2))

    assert_true(UInt(1) > UInt(0))
    assert_false(UInt(0) > UInt(1))

    assert_true(UInt(UINTMAX) > UInt(0))
    assert_false(UInt(0) > UInt(UINTMAX))

    assert_true(UInt(UINTMAX) > UInt(1))
    assert_false(UInt(1) > UInt(UINTMAX))

    assert_true(UInt(CLOSE_TO_UINTMAX) > UInt(1))
    assert_false(UInt(1) > UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(CLOSE_TO_UINTMAX) > UInt(0))
    assert_false(UInt(0) > UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(UINTMAX) > UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) > UInt(UINTMAX))

    assert_true(UInt(UINTMAX) > UInt(ABOVE_32_BITS))
    assert_false(UInt(ABOVE_32_BITS) > UInt(UINTMAX))

    assert_false(UInt(0) > UInt(0))
    assert_false(UInt(1) > UInt(1))
    assert_false(UInt(ABOVE_32_BITS) > UInt(ABOVE_32_BITS))
    assert_false(UInt(UINTMAX) > UInt(UINTMAX))


def test_uint_greater_equal():
    assert_true(UInt(2) >= UInt(1))
    assert_false(UInt(1) >= UInt(2))

    assert_true(UInt(1) >= UInt(0))
    assert_false(UInt(0) >= UInt(1))

    assert_true(UInt(UINTMAX) >= UInt(0))
    assert_false(UInt(0) >= UInt(UINTMAX))

    assert_true(UInt(UINTMAX) >= UInt(1))
    assert_false(UInt(1) >= UInt(UINTMAX))

    assert_true(UInt(CLOSE_TO_UINTMAX) >= UInt(1))
    assert_false(UInt(1) >= UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(CLOSE_TO_UINTMAX) >= UInt(0))
    assert_false(UInt(0) >= UInt(CLOSE_TO_UINTMAX))

    assert_true(UInt(UINTMAX) >= UInt(CLOSE_TO_UINTMAX))
    assert_false(UInt(CLOSE_TO_UINTMAX) >= UInt(UINTMAX))

    assert_true(UInt(UINTMAX) >= UInt(ABOVE_32_BITS))
    assert_false(UInt(ABOVE_32_BITS) >= UInt(UINTMAX))

    assert_true(UInt(0) >= UInt(0))
    assert_true(UInt(1) >= UInt(1))
    assert_true(UInt(ABOVE_32_BITS) >= UInt(ABOVE_32_BITS))
    assert_true(UInt(UINTMAX) >= UInt(UINTMAX))


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


def main():
    test_uint_lower_than()
    test_uint_lower_equal()
    test_uint_greater_than()
    test_uint_greater_equal()
    test_simple_uint()
    test_uint_representation()
