# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota
from sys.intrinsics import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from memory.buffer import Buffer


# CHECK-LABEL: test_masked_load
fn test_masked_load():
    print("== test_masked_load")

    let vector = Buffer[5, DType.float32].stack_allocation()
    vector.fill(1)

    # CHECK: [1.0, 1.0, 1.0, 1.0]
    print(masked_load[4](vector.data, iota[DType.float32, 4]() < 5, 0))

    # CHECK: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    print(masked_load[8](vector.data, iota[DType.float32, 8]() < 5, 0))

    # CHECK: [1.0, 1.0, 1.0, 1.0, 1.0, 15.0, 9.0, 3.0]
    print(
        masked_load[8](
            vector.data,
            iota[DType.float32, 8]() < 5,
            SIMD[DType.float32, 8](43, 321, 12, 312, 323, 15, 9, 3),
        )
    )

    # CHECK: [1.0, 1.0, 12.0, 312.0, 323.0, 15.0, 9.0, 3.0]
    print(
        masked_load[8](
            vector.data,
            iota[DType.float32, 8]() < 2,
            SIMD[DType.float32, 8](43, 321, 12, 312, 323, 15, 9, 3),
        )
    )


# CHECK-LABEL: test_masked_store
fn test_masked_store():
    print("== test_masked_store")

    let vector = Buffer[5, DType.float32].stack_allocation()
    vector.fill(0)

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    masked_store[4](
        iota[DType.float32, 4](), vector.data, iota[DType.float32, 4]() < 5
    )
    print(vector.simd_load[4](0))

    # CHECK: [0.0, 1.0, 2.0, 3.0, 4.0, 33.0, 33.0, 33.0]
    masked_store[8](
        iota[DType.float32, 8](), vector.data, iota[DType.float32, 8]() < 5
    )
    print(masked_load[8](vector.data, iota[DType.float32, 8]() < 5, 33))


# CHECK-LABEL: test_compressed_store
fn test_compressed_store():
    print("== test_compressed_store")

    let vector = Buffer[4, DType.float32].stack_allocation()
    vector.fill(0)

    # CHECK: [2.0, 3.0, 0.0, 0.0]
    compressed_store(
        iota[DType.float32, 4](), vector.data, iota[DType.float32, 4]() >= 2
    )
    print(vector.simd_load[4](0))

    # Just clear the buffer.
    vector.simd_store[4](0, 0)

    # CHECK: [1.0, 3.0, 0.0, 0.0]
    let val = SIMD[DType.float32, 4](0.0, 1.0, 3.0, 0.0)
    compressed_store(val, vector.data, val != 0)
    print(vector.simd_load[4](0))


# CHECK-LABEL: test_strided_load
fn test_strided_load():
    print("== test_strided_load")

    alias size = 16
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i

    # CHECK: [0.0, 4.0, 8.0, 12.0]
    let s = strided_load[DType.float32, 4](vector.data, 4)
    print(s)


# CHECK-LABEL: test_strided_store
fn test_strided_store():
    print("== test_strided_store")

    alias size = 8
    let vector = Buffer[size, DType.float32].stack_allocation()
    vector.fill(0)

    strided_store(SIMD[DType.float32, 4](99, 12, 23, 56), vector.data, 2)
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


fn main():
    test_masked_load()
    test_masked_store()
    test_compressed_store()
    test_strided_load()
    test_strided_store()
