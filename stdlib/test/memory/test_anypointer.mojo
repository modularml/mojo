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

from memory.anypointer import AnyPointer
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


fn test_anypointer_of_move_only_type():
    # CHECK-LABEL: === test_anypointer
    print("=== test_anypointer")

    var ptr = AnyPointer[MoveOnlyType].alloc(1)
    # CHECK: moved 42
    ptr.emplace_value(MoveOnlyType(42))
    # CHECK: moved 42
    var value = ptr.take_value()
    # CHECK: value 42
    print("value", value.value)
    # CHECK: deleted 42
    ptr.free()


def test_anypointer_move_into_move_count():
    var ptr = AnyPointer[MoveCounter[Int]].alloc(1)

    var value = MoveCounter(5)
    assert_equal(0, value.move_count)
    ptr.emplace_value(value^)

    # -----
    # Test that `AnyPointer.move_into` performs exactly one move.
    # -----

    assert_equal(1, ptr[].move_count)

    var ptr_2 = AnyPointer[MoveCounter[Int]].alloc(1)

    ptr.move_into(ptr_2)

    assert_equal(2, ptr_2[].move_count)


def test_refitem():
    var ptr = AnyPointer[Int].alloc(1)
    ptr[0] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)
    ptr.free()


def test_refitem_offset():
    var ptr = AnyPointer[Int].alloc(5)
    for i in range(5):
        ptr[i] = i
    for i in range(5):
        assert_equal(ptr[i], i)
    ptr.free()


def test_address_of():
    var local = 1
    assert_not_equal(0, int(AnyPointer[Int].address_of(local)))


def test_bitcast():
    var local = 1
    var ptr = AnyPointer[Int].address_of(local)
    var aliased_ptr = ptr.bitcast[SIMD[DType.uint8, 4]]()

    assert_equal(int(ptr), int(ptr.bitcast[Int]()))

    assert_equal(int(ptr), int(aliased_ptr))


def test_anypointer_string():
    var nullptr = AnyPointer[Int]()
    assert_equal(str(nullptr), "0x0")

    var ptr = AnyPointer[Int].alloc(1)
    assert_true(str(ptr).startswith("0x"))
    assert_not_equal(str(ptr), "0x0")
    ptr.free()


def test_eq():
    var local = 1
    var p1 = AnyPointer[Int].address_of(local)
    var p2 = p1
    assert_equal(p1, p2)

    var other_local = 2
    var p3 = AnyPointer[Int].address_of(other_local)
    assert_not_equal(p1, p3)

    var p4 = AnyPointer[Int].address_of(local)
    assert_equal(p1, p4)


def test_comparisons():
    var p1 = AnyPointer[Int].alloc(1)

    assert_true((p1 - 1) < p1)
    assert_true((p1 - 1) <= p1)
    assert_true(p1 <= p1)
    assert_true((p1 + 1) > p1)
    assert_true((p1 + 1) >= p1)
    assert_true(p1 >= p1)

    p1.free()


def main():
    test_address_of()

    test_refitem()
    test_refitem_offset()

    test_anypointer_of_move_only_type()
    test_anypointer_move_into_move_count()

    test_bitcast()
    test_anypointer_string()
    test_eq()
    test_comparisons()
