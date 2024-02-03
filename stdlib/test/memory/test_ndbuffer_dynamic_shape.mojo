# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import stack_allocation
from memory.buffer import NDBuffer, _compute_ndbuffer_offset

from utils.list import Dim, DimList


# CHECK-LABEL: test_ndbuffer_dynamic_shape
fn test_ndbuffer_dynamic_shape():
    print("== test_ndbuffer_dynamic_shape")

    # Create a buffer of size 16
    var buffer = stack_allocation[16, DType.index, 1]()

    var matrix = NDBuffer[DType.index, 2, DimList.create_unknown[2]()](
        buffer.address, DimList(4, 4)
    )

    matrix.dynamic_shape[0] = 42
    matrix.dynamic_shape[1] = 43

    # CHECK: 42
    print(matrix.dim[0]())
    # CHECK: 43
    print(matrix.dim[1]())

    # Mix static and dynamic shape.
    var matrix2 = NDBuffer[
        DType.index,
        2,
        DimList(42, Dim()),
    ](buffer.address, DimList(42, 1))

    matrix2.dynamic_shape[1] = 43

    # CHECK: 42
    print(matrix2.dim[0]())
    # CHECK: 43
    print(matrix2.dim[1]())


fn main():
    test_ndbuffer_dynamic_shape()
