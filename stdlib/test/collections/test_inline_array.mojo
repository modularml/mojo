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

from collections import InlineArray

from memory import UnsafePointer
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from test_utils import ValueDestructorRecorder
from testing import assert_equal, assert_false, assert_true


def test_array_unsafe_get():
    # Negative indexing is undefined behavior with unsafe_get
    # so there are not test cases for it.
    var arr = InlineArray[Int, 3](0, 0, 0)

    assert_equal(arr.unsafe_get(0), 0)
    assert_equal(arr.unsafe_get(1), 0)
    assert_equal(arr.unsafe_get(2), 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr.unsafe_get(0), 1)
    assert_equal(arr.unsafe_get(1), 2)
    assert_equal(arr.unsafe_get(2), 3)


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


def test_array_unsafe_assume_initialized_constructor_string():
    var maybe_uninitialized_arr = InlineArray[
        UnsafeMaybeUninitialized[String], 3
    ](unsafe_uninitialized=True)
    maybe_uninitialized_arr[0].write("hello")
    maybe_uninitialized_arr[1].write("mojo")
    maybe_uninitialized_arr[2].write("world")

    var initialized_arr = InlineArray[String, 3](
        unsafe_assume_initialized=maybe_uninitialized_arr^
    )

    assert_equal(initialized_arr[0], "hello")
    assert_equal(initialized_arr[1], "mojo")
    assert_equal(initialized_arr[2], "world")

    # trigger a move
    var initialized_arr2 = initialized_arr^

    assert_equal(initialized_arr2[0], "hello")
    assert_equal(initialized_arr2[1], "mojo")
    assert_equal(initialized_arr2[2], "world")

    # trigger a copy
    var initialized_arr3 = InlineArray(other=initialized_arr2)

    assert_equal(initialized_arr3[0], "hello")
    assert_equal(initialized_arr3[1], "mojo")
    assert_equal(initialized_arr3[2], "world")

    # We assume the destructor was called correctly, but one
    # might want to add a test for that in the future.


def test_array_contains():
    var arr = InlineArray[String, 3]("hi", "hello", "hey")
    assert_true(str("hi") in arr)
    assert_true(not str("greetings") in arr)


def test_inline_array_runs_destructors():
    """Ensure we delete the right number of elements."""
    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer.address_of(
        destructor_counter
    )
    alias capacity = 32
    var inline_list = InlineArray[
        ValueDestructorRecorder, 4, run_destructors=True
    ](
        ValueDestructorRecorder(0, pointer_to_destructor_counter),
        ValueDestructorRecorder(10, pointer_to_destructor_counter),
        ValueDestructorRecorder(20, pointer_to_destructor_counter),
        ValueDestructorRecorder(30, pointer_to_destructor_counter),
    )
    _ = inline_list
    # This is the last use of the inline list, so it should be destroyed here,
    # along with each element.
    assert_equal(len(destructor_counter), 4)
    assert_equal(destructor_counter[0], 0)
    assert_equal(destructor_counter[1], 10)
    assert_equal(destructor_counter[2], 20)
    assert_equal(destructor_counter[3], 30)


def main():
    test_array_unsafe_get()
    test_array_int()
    test_array_str()
    test_array_int_pointer()
    test_array_unsafe_assume_initialized_constructor_string()
    test_array_contains()
    test_inline_array_runs_destructors()
