# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import stack_allocation
from memory.buffer import NDBuffer

from utils.index import Index
from utils.list import DimList


# CHECK-LABEL: test_sub_matrix
fn test_sub_matrix():
    print("== test_sub_matrix")
    alias num_row = 4
    alias num_col = 4

    # Create a 4x4 matrix.
    var matrix = NDBuffer[
        2,
        DimList(num_row, num_col),
        DType.float32,
    ].stack_allocation()
    for i in range(num_row):
        for j in range(num_col):
            matrix[Index(i, j)] = Float32(i * num_col + j).value

    # Extract a sub-matrix 2x2 at (1,1).
    var sub_matrix0 = NDBuffer[2, DimList(2, 2), DType.float32](
        matrix.data.offset(5),
        DimList(2, 2),
        Index(4, 1),
    )

    # CHECK: 4
    print(sub_matrix0.stride(0))
    # CHECK: 1
    print(sub_matrix0.stride(1))
    # CHECK: False
    print(sub_matrix0.is_contiguous)
    # CHECK: 6.0
    print(sub_matrix0[Index(0, 1)])
    # CHECK: 10.0
    print(sub_matrix0[Index(1, 1)])

    # Extract a sub-matrix 2x2 at (1,1) with discontiguous last dim.
    # It includes (1,1) (1,3) (3,1) (3,3) of the original matrix.
    var sub_matrix1 = NDBuffer[2, DimList(2, 2), DType.float32](
        matrix.data.offset(1),
        DimList(2, 2),
        Index(8, 2),
    )

    # CHECK: 3.0
    print(sub_matrix1[Index(0, 1)])
    # CHECK: 9.0
    print(sub_matrix1[Index(1, 0)])

    # Extract a contiguous 2x2 buffer starting at (1,1).
    # It includes (1,1) (1,2) (1,3) (2,1) of the original matrix.
    var sub_matrix2 = NDBuffer[2, DimList(2, 2), DType.float32](
        matrix.data.offset(5),
        DimList(2, 2),
        Index(2, 1),
    )

    # CHECK: True
    print(sub_matrix2.is_contiguous)
    # CHECK: 8.0
    print(sub_matrix2[Index(1, 1)])


# CHECK-LABEL: test_broadcast
fn test_broadcast():
    print("== test_broadcast")

    # Create a buffer holding a single value with zero stride.
    var ptr = stack_allocation[1, DType.float32, 1]()
    var stride_buf = NDBuffer[1, DimList(100), DType.float32](
        ptr,
        DimList(100),
        Index(0),
    )

    # CHECK: 2.0
    stride_buf[0] = 2.0
    print(stride_buf[13])
    # CHECK: 2.0
    stride_buf[41] = 2.0
    print(stride_buf[99])


fn main():
    test_sub_matrix()
    test_broadcast()
