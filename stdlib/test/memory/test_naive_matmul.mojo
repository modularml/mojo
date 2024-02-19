# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This implements a naive matrix multiplication function as an example of using
# Lit and its standard library.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


# Corresponds to the following Python code:
#
# def nd_buffer(size):
#     return [[0.0] * size for i in range(size)]
#
#
# def fill_a(buf):
#     for i in range(len(buf)):
#         for j in range(len(buf[i])):
#             buf[i][j] = i + 2.0 * j
#
#
# def fill_b(buf):
#     for i in range(len(buf)):
#         for j in range(len(buf[i])):
#             buf[i][j] = float(i // (j + 1) + j)
#
#
# def print_matrix(buf):
#     for i in range(len(buf)):
#         for j in range(len(buf[i])):
#             print(buf[i][j])
#
#
# def test_my_naive_matmul(c, a, b):
#     for m in range(len(c)):
#         for n in range(len(c[0])):
#             c_val = 0.0
#             for k in range(len(a[0])):
#                 c_val += a[m][k] * b[k][n]
#             c[m][n] = c_val
#
#
# def test_naive_matmul(size):
#     print("== test_naive_matmul")
#     c = nd_buffer(size)
#
#     b = nd_buffer(size)
#     fill_b(b)
#
#     a = nd_buffer(size)
#     fill_a(a)
#
#     test_my_naive_matmul(c, a, b)
#     print_matrix(c)


from sys.info import sizeof

from memory.buffer import NDBuffer

from utils.index import StaticIntTuple
from utils.list import DimList


fn test_my_naive_matmul[
    shape: DimList, type: DType
](
    c: NDBuffer[type, 2, shape],
    a: NDBuffer[type, 2, shape],
    b: NDBuffer[type, 2, shape],
):
    """Computes matrix multiplication with a naive algorithm.

    Args:
        c: Buffer with allocated output space.
        a: Buffer containing matrix operand A.
        b: Buffer containing matrix operand B.
    """
    for m in range(c.dim[0]()):
        for n in range(c.dim[1]()):
            var c_val: SIMD[type, 1] = 0
            for k in range(a.dim[1]()):
                c_val += a[m, k] * b[k, n]
            c[StaticIntTuple[2](m, n)] = c_val


fn fill_a[size: Int](buf: NDBuffer[DType.float32, 2, DimList(size, size)]):
    """Fills the matrix with the values `row + 2*col`."""

    for i in range(size):
        for j in range(size):
            var val = Float32(i + 2 * j)
            buf[StaticIntTuple[2](i, j)] = val.value


fn fill_b[size: Int](buf: NDBuffer[DType.float32, 2, DimList(size, size)]):
    """Fills the matrix with the values `row/(col + 1) + col`."""

    for i in range(size):
        for j in range(size):
            var val = Float32(i // (j + 1) + j)
            buf[StaticIntTuple[2](i, j)] = val.value


fn print_matrix[
    size: Int
](buf: NDBuffer[DType.float32, 2, DimList(size, size),]):
    """Prints each element of the input matrix, element-wise."""
    for i in range(size):
        for j in range(size):
            print(buf[i, j])


# CHECK-LABEL: test_naive_matmul
fn test_naive_matmul[size: Int]():
    print("== test_naive_matmul")
    var c = NDBuffer[
        DType.float32,
        2,
        DimList(size, size),
    ].stack_allocation()
    c.fill(0)

    var b = NDBuffer[
        DType.float32,
        2,
        DimList(size, size),
    ].stack_allocation()
    fill_b[size](b)

    var a = NDBuffer[
        DType.float32,
        2,
        DimList(size, size),
    ].stack_allocation()
    fill_a[size](a)

    test_my_naive_matmul[
        DimList(size, size),
        DType.float32,
    ](c, a, b)

    print_matrix[size](c)


fn main():
    # CHECK: 4.0
    test_naive_matmul[2]()
    # CHECK: 72.0
    test_naive_matmul[4]()
    # CHECK: 784.0
    test_naive_matmul[8]()
    # CHECK: 7200.0
    test_naive_matmul[16]()
