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

from memory.unsafe import UnsafeMaybeUninitialized
from testing import assert_equal
from test_utils import MoveCounter, CopyCounter


@value
struct ValueToCountDestructor(CollectionElement):
    var value: Int
    var destructor_counter: UnsafePointer[List[Int]]

    fn __del__(owned self):
        self.destructor_counter[].append(self.value)


def test_maybe_uninitialized():
    # Every time an Int is destroyed, it's going to be reccorded here.
    var destructor_counter = List[Int]()

    var a = UnsafeMaybeUninitialized[ValueToCountDestructor]()
    a.write(
        ValueToCountDestructor(42, UnsafePointer.address_of(destructor_counter))
    )

    assert_equal(a.assume_initialized().value, 42)
    assert_equal(len(destructor_counter), 0)

    assert_equal(a.unsafe_ptr()[].value, 42)
    assert_equal(len(destructor_counter), 0)

    a.assume_initialized_destroy()
    assert_equal(len(destructor_counter), 1)
    assert_equal(destructor_counter[0], 42)
    _ = a

    # Last use of a, but the destructor should not have run
    # since we asssume uninitialized memory
    assert_equal(len(destructor_counter), 1)


@value
struct ImpossibleToDestroy(CollectionElement):
    var value: Int

    fn __del__(owned self):
        abort("We should never call the destructor of ImpossibleToDestroy")


def test_write_does_not_trigger_destructor():
    var a = UnsafeMaybeUninitialized[ImpossibleToDestroy]()
    a.write(ImpossibleToDestroy(42))

    # Using the initializer should not trigger the destructor too.
    var b = UnsafeMaybeUninitialized[ImpossibleToDestroy](
        ImpossibleToDestroy(42)
    )

    # The destructor of a and b have already run at this point, and it shouldn't have
    # caused a crash since we assume uninitialized memory.


def test_maybe_uninitialized_move():
    var a = UnsafeMaybeUninitialized[MoveCounter[Int]](MoveCounter(10))
    assert_equal(a.assume_initialized().move_count, 1)

    var b = UnsafeMaybeUninitialized[MoveCounter[Int]]()
    # b is uninitialized here.
    b.move_from(a)
    # a is uninitialized now.
    assert_equal(b.assume_initialized().move_count, 2)
    b.assume_initialized_destroy()


def test_maybe_uninitialized_copy():
    var a = UnsafeMaybeUninitialized[CopyCounter]()
    a.write(CopyCounter())
    assert_equal(a.assume_initialized().copy_count, 0)

    var b = UnsafeMaybeUninitialized[CopyCounter]()
    assert_equal(a.assume_initialized().copy_count, 0)

    # b is uninitialized here.
    b.copy_from(a)
    a.assume_initialized_destroy()

    assert_equal(b.assume_initialized().copy_count, 1)
    b.assume_initialized_destroy()


def main():
    test_maybe_uninitialized()
    test_write_does_not_trigger_destructor()
    test_maybe_uninitialized_move()
    test_maybe_uninitialized_copy()
