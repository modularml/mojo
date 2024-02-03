# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota
from sys.intrinsics import PrefetchOptions

from memory import memcmp, memset_zero
from memory.buffer import NDBuffer, _compute_ndbuffer_offset
from tensor import Tensor

from utils.index import Index, StaticIntTuple
from utils.list import DimList


# CHECK-LABEL: test_ndbuffer
fn test_ndbuffer():
    print("== test_ndbuffer")
    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix = NDBuffer[
        DType.index,
        2,
        DimList(4, 4),
    ].stack_allocation()

    matrix[StaticIntTuple[2](0, 0)] = 0
    matrix[StaticIntTuple[2](0, 1)] = 1
    matrix[StaticIntTuple[2](0, 2)] = 2
    matrix[StaticIntTuple[2](0, 3)] = 3
    matrix[StaticIntTuple[2](1, 0)] = 4
    matrix[StaticIntTuple[2](1, 1)] = 5
    matrix[StaticIntTuple[2](1, 2)] = 6
    matrix[StaticIntTuple[2](1, 3)] = 7
    matrix[StaticIntTuple[2](2, 0)] = 8
    matrix[StaticIntTuple[2](2, 1)] = 9
    matrix[StaticIntTuple[2](2, 2)] = 10
    matrix[StaticIntTuple[2](2, 3)] = 11
    matrix[StaticIntTuple[2](3, 0)] = 12
    matrix[StaticIntTuple[2](3, 1)] = 13
    matrix[StaticIntTuple[2](3, 2)] = 14
    matrix[StaticIntTuple[2](3, 3)] = 15

    # CHECK: 11
    print(
        _compute_ndbuffer_offset[
            DType.index,
            2,
            DimList(4, 4),
        ](matrix, StaticIntTuple[2](2, 3))
    )

    # CHECK: 14
    print(
        _compute_ndbuffer_offset[
            DType.index,
            2,
            DimList(4, 4),
        ](matrix, StaticIntTuple[2](3, 2))
    )

    # CHECK: 15
    print(
        _compute_ndbuffer_offset[
            DType.index,
            2,
            DimList(4, 4),
        ](matrix, StaticIntTuple[2](3, 3))
    )

    # CHECK: 2
    print(matrix.get_rank())

    # CHECK: 16
    print(matrix.size())

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 1
    print(matrix[0, 1])

    # CHECK: 2
    print(matrix[0, 2])

    # CHECK: 3
    print(matrix[0, 3])

    # CHECK: 4
    print(matrix[1, 0])

    # CHECK: 5
    print(matrix[1, 1])

    # CHECK: 6
    print(matrix[1, 2])

    # CHECK: 7
    print(matrix[1, 3])

    # CHECK: 8
    print(matrix[2, 0])

    # CHECK: 9
    print(matrix[2, 1])

    # CHECK: 10
    print(matrix[2, 2])

    # CHECK: 11
    print(matrix[2, 3])

    # CHECK: 12
    print(matrix[3, 0])

    # CHECK: 13
    print(matrix[3, 1])

    # CHECK: 14
    print(matrix[3, 2])

    # CHECK: 15
    print(matrix[3, 3])


# CHECK-LABEL: test_fill
fn test_fill():
    print("== test_fill")

    var buf = NDBuffer[
        DType.index,
        2,
        DimList(3, 3),
    ].stack_allocation()
    buf[StaticIntTuple[2](0, 0)] = 1
    buf[StaticIntTuple[2](0, 1)] = 1
    buf[StaticIntTuple[2](0, 2)] = 1
    buf[StaticIntTuple[2](1, 0)] = 1
    buf[StaticIntTuple[2](1, 1)] = 1
    buf[StaticIntTuple[2](1, 2)] = 1
    buf[StaticIntTuple[2](2, 0)] = 1
    buf[StaticIntTuple[2](2, 1)] = 1
    buf[StaticIntTuple[2](2, 2)] = 1

    var filled = NDBuffer[
        DType.index,
        2,
        DimList(3, 3),
    ].stack_allocation()
    filled.fill(1)

    var err = memcmp(buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)

    memset_zero(filled.data, filled.num_elements())
    filled.fill(1)
    err = memcmp[DType.index](buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)

    memset_zero(buf.data, buf.num_elements())
    filled.simd_fill[4](0)
    err = memcmp[DType.index](buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)


# CHECK-LABEL: test_ndbuffer_prefetch
fn test_ndbuffer_prefetch():
    print("== test_ndbuffer_prefetch")
    # Create a matrix of the form
    # [[0, 1, 2],
    #  [3, 4, 5]]
    var matrix = NDBuffer[
        DType.index,
        2,
        DimList(2, 3),
    ].stack_allocation()

    # Prefetch for write
    for i0 in range(2):
        for j0 in range(3):
            matrix.prefetch[PrefetchOptions().high_locality().for_write()](
                i0, j0
            )

    # Set values
    for i1 in range(2):
        for j1 in range(3):
            matrix[Index(i1, j1)] = i1 * 3 + j1

    # Prefetch for read
    for i2 in range(2):
        for j2 in range(3):
            matrix.prefetch[PrefetchOptions().high_locality().for_read()](
                i2, j2
            )

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 1
    print(matrix[0, 1])

    # CHECK: 2
    print(matrix[0, 2])

    # CHECK: 3
    print(matrix[1, 0])

    # CHECK: 4
    print(matrix[1, 1])

    # CHECK: 5
    print(matrix[1, 2])


# CHECK-LABEL: test_aligned_load_store
fn test_aligned_load_store():
    print("== test_aligned_load_store")
    var matrix = NDBuffer[
        DType.index,
        2,
        DimList(4, 4),
    ].aligned_stack_allocation[128]()

    # Set values
    for i1 in range(4):
        for j1 in range(4):
            matrix[Index(i1, j1)] = i1 * 4 + j1

    # CHECK: [0, 1, 2, 3]
    print(matrix.aligned_simd_load[4, 16](0, 0))

    # CHECK: [12, 13, 14, 15]
    print(matrix.aligned_simd_load[4, 16](3, 0))

    # CHECK: [0, 1, 2, 3]
    matrix.aligned_simd_store[4, 32](Index(3, 0), iota[DType.index, 4]())
    print(matrix.aligned_simd_load[4, 32](3, 0))


fn test_get_nd_index():
    print("== test_get_nd_index\n")
    var matrix0 = NDBuffer[
        DType.index,
        2,
        DimList(2, 3),
    ].stack_allocation()

    var matrix1 = NDBuffer[
        DType.index,
        3,
        DimList(3, 5, 7),
    ].stack_allocation()

    # CHECK: (0, 0)
    print(matrix0.get_nd_index(0))

    # CHECK: (0, 1)
    print(matrix0.get_nd_index(1))

    # CHECK: (1, 0)
    print(matrix0.get_nd_index(3))

    # CHECK: (1, 2)
    print(matrix0.get_nd_index(5))

    # CHECK: (0, 2, 6)
    print(matrix1.get_nd_index(20))

    # CHECK: (2, 4, 6)
    print(matrix1.get_nd_index(104))


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")
    # CHECK{LITERAL}: NDBuffer([[[0, 1, 2],
    # CHECK{LITERAL}: [3, 4, 5]],
    # CHECK{LITERAL}: [[6, 7, 8],
    # CHECK{LITERAL}: [9, 10, 11]]], dtype=index, shape=2x2x3)
    let tensor = Tensor[DType.index](2, 2, 3)
    iota(tensor.data(), tensor.num_elements())

    let buffer = NDBuffer[DType.index, 3, DimList(2, 2, 3)](tensor.data())

    print(str(buffer))
    _ = tensor ^


fn main():
    test_ndbuffer()
    test_fill()
    test_ndbuffer_prefetch()
    test_aligned_load_store()
    test_get_nd_index()
    test_print()
