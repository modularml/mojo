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

from testing import (
    assert_equal,
    assert_false,
    assert_true,
    assert_not_equal,
    assert_almost_equal,
)
from utils.numerics import isinf


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


def test_uint_add():
    assert_equal(UInt(32).__add__(UInt(32)), UInt(64))
    assert_equal(UInt(0).__add__(UInt(0)), UInt(0))
    assert_equal(UInt().__add__(UInt(0)), UInt(0))

    assert_equal(UInt(3).__add__(UInt(2)), UInt(5))
    assert_equal(UInt(3).__add__(UInt(0)), UInt(3))
    assert_equal(UInt(154324).__add__(UInt(27435)), UInt(181759))


def test_uint_sub():
    assert_equal(UInt(21).__sub__(UInt(21)), UInt(0))
    assert_equal(UInt(0).__sub__(UInt(0)), UInt(0))
    assert_equal(UInt().__sub__(UInt(0)), UInt(0))

    assert_equal(UInt(3).__sub__(UInt(2)), UInt(1))
    assert_equal(UInt(3).__sub__(UInt(0)), UInt(3))
    assert_equal(UInt(154324).__sub__(UInt(27435)), UInt(126889))


def test_uint_mul():
    assert_equal(UInt(21).__mul__(UInt(21)), UInt(441))
    assert_equal(UInt(0).__mul__(UInt(0)), UInt(0))
    assert_equal(UInt().__mul__(UInt(0)), UInt(0))

    assert_equal(UInt(3).__mul__(UInt(2)), UInt(6))
    assert_equal(UInt(3).__mul__(UInt(0)), UInt(0))
    assert_equal(UInt(154324).__mul__(UInt(27435)), UInt(4233878940))


def test_uint_truediv():
    assert_equal(UInt(21).__truediv__(UInt(21)), Float64(1.0))
    assert_equal(UInt(0).__truediv__(UInt(47)), Float64(0.0))
    assert_equal(UInt().__truediv__(UInt(47)), Float64(0.0))

    assert_almost_equal(UInt(3).__truediv__(UInt(2)), Float64(1.5))
    assert_almost_equal(
        UInt(1).__truediv__(UInt(3)), Float64(0.3333333333333333333)
    )
    assert_almost_equal(
        UInt(154324).__truediv__(UInt(27435)), Float64(5.625077455804629)
    )

    assert_true(
        isinf(UInt(3).__truediv__(UInt(0))),
        msg="UInt(3).__truediv__(UInt(0)) is not infinite",
    )
    assert_true(
        isinf(UInt(154324).__truediv__(UInt(0))),
        msg="UInt(154324).__truediv__(UInt(0)) is not infinite",
    )
    assert_false(
        isinf(UInt(0).__truediv__(UInt(0))),
        msg="UInt(0).__truediv__(UInt(0)) should not be infinite",
    )


def main():
    test_simple_uint()
    test_uint_representation()
    test_equality()
    test_inequality()
    test_uint_add()
    test_uint_sub()
    test_uint_mul()
    test_uint_truediv()
