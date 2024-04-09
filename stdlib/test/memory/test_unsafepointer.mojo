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
# RUN: %mojo -debug-level full %s | FileCheck %s --dump-input=always

from memory.anypointer import *
from test_utils import MoveCounter
from testing import assert_equal, assert_not_equal, assert_true


struct MoveOnlyType(Movable):
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __moveinit__(inout self, owned existing: Self):
        self.value = existing.value
        print("moved", self.value)

    fn __del__(owned self):
        print("deleted", self.value)


fn test_unsafepointer_of_move_only_type():
    # CHECK-LABEL: === test_unsafepointer
    print("=== test_unsafepointer")

    var ptr = UnsafePointer[MoveOnlyType].alloc(1)
    # CHECK: moved 42
    initialize_pointee(ptr, MoveOnlyType(42))
    # CHECK: moved 42
    var value = move_from_pointee(ptr)
    # CHECK: value 42
    print("value", value.value)
    # CHECK: deleted 42
    ptr.free()


def test_unsafepointer_move_pointee_move_count():
    var ptr = UnsafePointer[MoveCounter[Int]].alloc(1)

    var value = MoveCounter(5)
    assert_equal(0, value.move_count)
    initialize_pointee(ptr, value^)

    # -----
    # Test that `UnsafePointer.move_pointee` performs exactly one move.
    # -----

    assert_equal(1, ptr[].move_count)

    var ptr_2 = UnsafePointer[MoveCounter[Int]].alloc(1)

    move_pointee(src=ptr, dst=ptr_2)

    assert_equal(2, ptr_2[].move_count)


def test_refitem():
    var ptr = UnsafePointer[Int].alloc(1)
    ptr[0] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)
    ptr.free()


def test_refitem_offset():
    var ptr = UnsafePointer[Int].alloc(5)
    for i in range(5):
        ptr[i] = i
    for i in range(5):
        assert_equal(ptr[i], i)
    ptr.free()


def test_address_of():
    var local = 1
    assert_not_equal(0, int(UnsafePointer[Int].address_of(local)))


def test_bitcast():
    var local = 1
    var ptr = AnyPointer[Int].address_of(local)
    var aliased_ptr = ptr.bitcast_element[SIMD[DType.uint8, 4]]()

    assert_equal(int(ptr), int(ptr.bitcast_element[Int]()))

    assert_equal(int(ptr), int(aliased_ptr))


def test_unsafepointer_string():
    var nullptr = UnsafePointer[Int]()
    assert_equal(str(nullptr), "0x0")

    var ptr = UnsafePointer[Int].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_eq():
    var local = 1
    var p1 = UnsafePointer[Int].address_of(local)
    var p2 = p1
    assert_equal(p1, p2)

    var other_local = 2
    var p3 = UnsafePointer[Int].address_of(other_local)
    assert_not_equal(p1, p3)

    var p4 = UnsafePointer[Int].address_of(local)
    assert_equal(p1, p4)


def test_comparisons():
    var p1 = UnsafePointer[Int].alloc(1)

    assert_true((p1 - 1) < p1)
    assert_true((p1 - 1) <= p1)
    assert_true(p1 <= p1)
    assert_true((p1 + 1) > p1)
    assert_true((p1 + 1) >= p1)
    assert_true(p1 >= p1)

    p1.free()


def test_unsafepointer_address_space():
    var p1 = UnsafePointer[Int, AddressSpace(0)].alloc(1)
    p1.free()

    var p2 = UnsafePointer[Int, AddressSpace.GENERIC].alloc(1)
    p2.free()


def main():
    test_address_of()

    test_refitem()
    test_refitem_offset()

    test_unsafepointer_of_move_only_type()
    test_unsafepointer_move_pointee_move_count()

    test_bitcast()
    test_unsafepointer_string()
    test_eq()
    test_comparisons()

    test_unsafepointer_address_space()
