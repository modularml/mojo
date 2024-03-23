# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.intrinsics import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from memory.unsafe import DTypePointer


# CHECK-LABEL: test_strided_load
fn test_strided_load():
    print("== test_strided_load")

    alias size = 16
    var vector = DTypePointer[DType.float32]().alloc(size)

    for i in range(size):
        vector[i] = i

    # CHECK: [0.0, 4.0, 8.0, 12.0]
    var s = strided_load[DType.float32, 4](vector, 4)
    print(s)
    vector.free()


# CHECK-LABEL: test_strided_store
fn test_strided_store():
    print("== test_strided_store")

    alias size = 8
    var vector = DTypePointer[DType.float32]().alloc(size)
    memset_zero(vector, size)

    strided_store(SIMD[DType.float32, 4](99, 12, 23, 56), vector, 2)
    # CHECK: 99.0
    # CHECK: 0.0
    # CHECK: 12.0
    # CHECK: 0.0
    # CHECK: 23.0
    # CHECK: 0.0
    # CHECK: 56.0
    # CHECK: 0.0
    for i in range(size):
        print(vector[i])
    vector.free()


fn main():
    test_strided_load()
    test_strided_store()
