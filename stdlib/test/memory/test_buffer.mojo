# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory.buffer import Buffer

from utils.list import Dim


# CHECK-LABEL: test_buffer
fn test_buffer():
    print("== test_buffer")

    alias vec_size = 4
    var data = Pointer[Float32].alloc(vec_size)

    var b1 = Buffer[DType.float32, 4](data)
    var b2 = Buffer[DType.float32, 4](data, 4)
    var b3 = Buffer[DType.float32](data, 4)

    # CHECK: 4 4 4
    print(len(b1), len(b2), len(b3))

    data.free()


fn main():
    test_buffer()
