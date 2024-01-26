# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import stack_allocation
from memory.buffer import Buffer, NDBuffer

from utils.list import DimList


# CHECK-LABEL: test_ndbuffer_indexing
fn test_ndbuffer_indexing():
    print("== test_ndbuffer_indexing")

    # The total amount of data to allocate
    alias total_buffer_size: Int = 2 * 3 * 4 * 5 * 6

    # Create a buffer for indexing test:
    var _data = stack_allocation[
        total_buffer_size,
        DType.index,
        1,
    ]()

    # Fill data with increasing order, so that the value of each element in
    #  the test buffer is equal to it's linear index.:
    var fillBufferView = Buffer[
        total_buffer_size,
        DType.index,
    ](_data)

    for fillIdx in range(total_buffer_size):
        fillBufferView[fillIdx] = fillIdx

    # ===------------------------------------------------------------------=== #
    # Test 1DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView1D = NDBuffer[
        1,
        DimList(6),
        DType.index,
    ](_data)

    # Try to access element[5]
    # CHECK: 5
    print[1, DType.index](bufferView1D[5])

    # ===------------------------------------------------------------------=== #
    # Test 2DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView2D = NDBuffer[
        2,
        DimList(5, 6),
        DType.index,
    ](_data)

    # Try to access element[4,5]
    # Result should be 4*6+5 = 29
    # CHECK: 29
    print[1, DType.index](bufferView2D[4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 3DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView3D = NDBuffer[
        3,
        DimList(4, 5, 6),
        DType.index,
    ](_data)

    # Try to access element[3,4,5]
    # Result should be 3*(5*6)+4*6+5 = 119
    # CHECK: 119
    print[1, DType.index](bufferView3D[3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 4DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView4D = NDBuffer[
        4,
        DimList(3, 4, 5, 6),
        DType.index,
    ](_data)

    # Try to access element[2,3,4,5]
    # Result should be 2*4*5*6+3*5*6+4*6+5 = 359
    # CHECK: 359
    print[1, DType.index](bufferView4D[2, 3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 5DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView5D = NDBuffer[
        5,
        DimList(2, 3, 4, 5, 6),
        DType.index,
    ](_data)

    # Try to access element[1,2,3,4,5]
    # Result should be 1*3*4*5*6+2*4*5*6+3*5*6+4*6+5 = 719
    # CHECK: 719
    print[1, DType.index](bufferView5D[1, 2, 3, 4, 5])


fn main():
    test_ndbuffer_indexing()
