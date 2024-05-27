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

from testing import assert_equal, assert_false, assert_true

from utils import StaticTuple, StaticIntTuple, InlineArray


def test_static_tuple():
    var tup1 = StaticTuple[Int, 1](1)
    assert_equal(tup1[0], 1)

    var tup2 = StaticTuple[Int, 2](1, 1)
    assert_equal(tup2[0], 1)
    assert_equal(tup2[1], 1)

    var tup3 = StaticTuple[Int, 3](1, 2, 3)
    assert_equal(tup3[0], 1)
    assert_equal(tup3[1], 2)
    assert_equal(tup3[2], 3)

    assert_equal(tup3[0], 1)
    assert_equal(tup3[Int(0)], 1)
    assert_equal(tup3[Int64(0)], 1)


def test_static_int_tuple():
    assert_equal(str(StaticIntTuple[1](1)), "(1,)")

    assert_equal(str(StaticIntTuple[3](2)), "(2, 2, 2)")

    assert_equal(
        str(StaticIntTuple[3](1, 2, 3) * StaticIntTuple[3](4, 5, 6)),
        "(4, 10, 18)",
    )

    assert_equal(
        str(StaticIntTuple[4](1, 2, 3, 4) - StaticIntTuple[4](4, 5, 6, 7)),
        "(-3, -3, -3, -3)",
    )

    assert_equal(
        str(StaticIntTuple[2](10, 11) // StaticIntTuple[2](3, 4)), "(3, 2)"
    )

    # Note: index comparison is intended for access bound checking, which is
    #  usually all-element semantic, i.e. true if true for all positions.
    assert_true(
        StaticIntTuple[5](1, 2, 3, 4, 5) < StaticIntTuple[5](4, 5, 6, 7, 8)
    )

    assert_false(
        StaticIntTuple[4](3, 5, -1, -2) > StaticIntTuple[4](0, 0, 0, 0)
    )

    assert_equal(len(StaticIntTuple[4](3, 5, -1, -2)), 4)

    assert_equal(str(StaticIntTuple[2]((1, 2))), "(1, 2)")

    assert_equal(str(StaticIntTuple[4]((1, 2, 3, 4))), "(1, 2, 3, 4)")


def test_tuple_literal():
    assert_equal(len((1, 2, (3, 4), 5)), 4)
    assert_equal(len(()), 0)


def test_array_get_reference_unsafe():
    # Negative indexing is undefined behavior with _get_reference_unsafe
    # so there are not test cases for it.
    var arr = InlineArray[Int, 3](0, 0, 0)

    assert_equal(arr._get_reference_unsafe(0)[], 0)
    assert_equal(arr._get_reference_unsafe(1)[], 0)
    assert_equal(arr._get_reference_unsafe(2)[], 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr._get_reference_unsafe(0)[], 1)
    assert_equal(arr._get_reference_unsafe(1)[], 2)
    assert_equal(arr._get_reference_unsafe(2)[], 3)


def test_array_int():
    var arr = InlineArray[Int, 3](0, 0, 0)

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 0)
    assert_equal(arr[2], 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr[0], 1)
    assert_equal(arr[1], 2)
    assert_equal(arr[2], 3)

    # test negative indexing
    assert_equal(arr[-1], 3)
    assert_equal(arr[-2], 2)

    # test negative indexing with dynamic index
    var i = -1
    assert_equal(arr[i], 3)
    i -= 1
    assert_equal(arr[i], 2)

    var copy = arr
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = InlineArray[Int, 3](5)
    assert_equal(arr2[0], 5)
    assert_equal(arr2[1], 5)
    assert_equal(arr2[2], 5)

    var arr3 = InlineArray[Int, 1](5)
    assert_equal(arr3[0], 5)

    var arr4 = InlineArray[UInt8, 1](42)
    assert_equal(arr4[0], 42)


def test_array_str():
    var arr = InlineArray[String, 3]("hi", "hello", "hey")

    assert_equal(arr[0], "hi")
    assert_equal(arr[1], "hello")
    assert_equal(arr[2], "hey")

    # Test mutating an array through its __getitem__
    arr[0] = "howdy"
    arr[1] = "morning"
    arr[2] = "wazzup"

    assert_equal(arr[0], "howdy")
    assert_equal(arr[1], "morning")
    assert_equal(arr[2], "wazzup")

    # test negative indexing
    assert_equal(arr[-1], "wazzup")
    assert_equal(arr[-2], "morning")

    var copy = arr
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = InlineArray[String, 3]("hi")
    assert_equal(arr2[0], "hi")
    assert_equal(arr2[1], "hi")
    assert_equal(arr2[2], "hi")

    # size 1 array to prevent regressions in the constructors
    var arr3 = InlineArray[String, 1]("hi")
    assert_equal(arr3[0], "hi")


def test_array_int_pointer():
    var arr = InlineArray[Int, 3](0, 10, 20)

    var ptr = arr.unsafe_ptr()
    assert_equal(ptr[0], 0)
    assert_equal(ptr[1], 10)
    assert_equal(ptr[2], 20)

    ptr[0] = 0
    ptr[1] = 1
    ptr[2] = 2

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 1)
    assert_equal(arr[2], 2)

    assert_equal(ptr[0], 0)
    assert_equal(ptr[1], 1)
    assert_equal(ptr[2], 2)

    # We make sure it lives long enough
    _ = arr


def test_array_contains():
    var arr = InlineArray[String, 3]("hi", "hello", "hey")
    assert_true(str("hi") in arr)
    assert_true(not str("greetings") in arr)


def main():
    test_static_tuple()
    test_static_int_tuple()
    test_tuple_literal()
    test_array_get_reference_unsafe()
    test_array_int()
    test_array_str()
    test_array_int_pointer()
    test_array_contains()
