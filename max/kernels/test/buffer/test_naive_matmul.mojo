# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# This implements a naive matrix multiplication function as an example of using
# Lit and its standard library.
#
# ===----------------------------------------------------------------------=== #

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
#             buf[i][j] = Float64(i // (j + 1) + j)
#
#
# def print_matrix(buf):
#     for i in range(len(buf)):
#         for j in range(len(buf[i])):
#             print(buf[i][j])
#
#
# def _test_my_naive_matmul(c, a, b):
#     for m in range(len(c)):
#         for n in range(len(c[0])):
#             c_val = 0.0
#             for k in range(len(a[0])):
#                 c_val += a[m][k] * b[k][n]
#             c[m][n] = c_val
#
#
# def _test_naive_matmul(size):
#     print("== _test_naive_matmul")
#     c = nd_buffer(size)
#
#     b = nd_buffer(size)
#     fill_b(b)
#
#     a = nd_buffer(size)
#     fill_a(a)
#
#     _test_my_naive_matmul(c, a, b)
#     print_matrix(c)


from std.testing import TestSuite


def _test_my_naive_matmul[
    size: Int, dtype: DType
](
    c: MutPointer[Scalar[dtype], _],
    a: ImmPointer[Scalar[dtype], _],
    b: ImmPointer[Scalar[dtype], _],
):
    """Computes matrix multiplication with a naive algorithm.

    Args:
        c: Pointer to output space (size x size row-major).
        a: Pointer to matrix operand A (size x size row-major).
        b: Pointer to matrix operand B (size x size row-major).
    """
    for m in range(size):
        for n in range(size):
            var c_val: Scalar[dtype] = 0
            for k in range(size):
                c_val += a[m * size + k] * b[k * size + n]
            c[m * size + n] = c_val


def fill_a[size: Int](buf: MutPointer[Float32, _]):
    """Fills the matrix with the values `row + 2*col`."""

    for i in range(size):
        for j in range(size):
            buf[i * size + j] = Float32(i + 2 * j)


def fill_b[size: Int](buf: MutPointer[Float32, _]):
    """Fills the matrix with the values `row/(col + 1) + col`."""

    for i in range(size):
        for j in range(size):
            buf[i * size + j] = Float32(i // (j + 1) + j)


def print_matrix[size: Int](buf: ImmPointer[Float32, _]):
    """Prints each element of the input matrix, element-wise."""
    for i in range(size):
        for j in range(size):
            print(buf[i * size + j])


# CHECK-LABEL: _test_naive_matmul
def _test_naive_matmul[size: Int]():
    print("== _test_naive_matmul")
    var c_stack = Array[Float32, size * size](fill=0)
    var c = c_stack.unsafe_ptr()

    var b_stack = Array[Float32, size * size](fill={})
    var b = b_stack.unsafe_ptr()
    fill_b[size](b)

    var a_stack = Array[Float32, size * size](fill={})
    var a = a_stack.unsafe_ptr()
    fill_a[size](a)

    _test_my_naive_matmul[size, DType.float32](c, a, b)

    print_matrix[size](c)


def test_naive_matmul_2() raises:
    # CHECK: 4.0
    _test_naive_matmul[2]()


def test_naive_matmul_4() raises:
    # CHECK: 72.0
    _test_naive_matmul[4]()


def test_naive_matmul_8() raises:
    # CHECK: 784.0
    _test_naive_matmul[8]()


def test_naive_matmul_16() raises:
    # CHECK: 7200.0
    _test_naive_matmul[16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
