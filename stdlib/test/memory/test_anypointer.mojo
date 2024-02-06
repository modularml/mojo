# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from memory.anypointer import AnyPointer
from testing.testing import _MoveCounter

from testing import *


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

    let ptr = AnyPointer[MoveOnlyType].alloc(1)
    # CHECK: moved 42
    ptr.emplace_value(MoveOnlyType(42))
    # CHECK: moved 42
    let value = ptr.take_value()
    # NOTE: Destructor is called before `print`.
    # CHECK: deleted 42
    # CHECK: value 42
    print("value", value.value)
    ptr.free()


fn test_anypointer_move_into_move_count():
    let ptr = AnyPointer[_MoveCounter[Int]].alloc(1)

    let value = _MoveCounter(5)
    # CHECK: 0
    print(value.move_count)
    ptr.emplace_value(value ^)

    # -----
    # Test that `AnyPointer.move_into` performs exactly one move.
    # -----

    # CHECK: 1
    print(__get_address_as_lvalue(ptr.value).move_count)

    let ptr_2 = AnyPointer[_MoveCounter[Int]].alloc(1)

    ptr.move_into(ptr_2)

    # CHECK: 2
    print(__get_address_as_lvalue(ptr_2.value).move_count)


def test_refitem():
    let ptr = AnyPointer[Int].alloc(1)
    ptr[0] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)
    ptr.free()


def test_refitem_offset():
    let ptr = AnyPointer[Int].alloc(5)
    for i in range(5):
        ptr[i] = i
    for i in range(5):
        assert_equal(ptr[i], i)
    ptr.free()


def main():
    test_refitem()
    test_refitem_offset()

    test_anypointer_of_move_only_type()
    test_anypointer_move_into_move_count()
