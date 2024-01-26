# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from memory.anypointer import AnyPointer


struct MoveOnlyType(Movable):
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __moveinit__(inout self, owned existing: Self):
        self.value = existing.value
        print("moved", self.value)

    fn __del__(owned self):
        print("deleted", self.value)


fn main():
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
