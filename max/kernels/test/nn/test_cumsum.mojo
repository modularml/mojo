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


from std.math import iota, isclose

from layout import TileTensor, row_major
from nn.cumsum import cumsum


# CHECK-LABEL: test_cumsum_1d
# CHECK: 1.0 ,3.0 ,6.0 ,10.0 ,15.0 ,
def test_cumsum_1d():
    print("== test_cumsum_1d")
    comptime exclusive = False
    comptime reverse = False
    comptime axis = 0

    var matrix_stack = Array[Float64, 5](fill={})
    var matrix = TileTensor(matrix_stack, row_major[5]())

    iota(matrix._storage, 5, 1)

    var cumsum_stack = Array[Float64, 5](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[5]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(5):
        print(cumsum_matrix[i], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_1d_precision
# CHECK: Passed
def test_cumsum_1d_precision():
    print("== test_cumsum_1d_precision")
    comptime exclusive = False
    comptime reverse = False
    comptime axis = 0
    comptime size = 1024

    var f32_stack = Array[Float32, size](fill=1.1)
    var f32_matrix = TileTensor(f32_stack, row_major[size]())

    var f64_stack = Array[Float64, size](fill=1.1)
    var f64_matrix = TileTensor(f64_stack, row_major[size]())

    var cumsum_f32_stack = Array[Float32, size](fill={})
    var cumsum_f32 = TileTensor(cumsum_f32_stack, row_major[size]())
    var cumsum_f64_stack = Array[Float64, size](fill={})
    var cumsum_f64 = TileTensor(cumsum_f64_stack, row_major[size]())

    cumsum[.float32, exclusive, reverse, axis=axis](cumsum_f32, f32_matrix)
    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_f64, f64_matrix)

    var passed = True
    for i in range(size):
        var f64_cast = cumsum_f64[i].cast[.float32]()
        if not isclose(cumsum_f32[i], f64_cast, atol=1e-6, rtol=1e-6):
            passed = False
            break

    print("Passed" if passed else "Failed")


# CHECK-LABEL: test_cumsum_1d_exclusive
# CHECK: 0.0 ,1.0 ,3.0 ,6.0 ,10.0 ,
def test_cumsum_1d_exclusive():
    print("== test_cumsum_1d_exclusive")
    comptime exclusive = True
    comptime reverse = False
    comptime axis = 0

    var matrix_stack = Array[Float64, 5](fill={})
    var matrix = TileTensor(matrix_stack, row_major[5]())

    iota(matrix._storage, 5, 1)

    var cumsum_stack = Array[Float64, 5](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[5]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(5):
        print(cumsum_matrix[i], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_1d_reverse
# CHECK: 15.0 ,14.0 ,12.0 ,9.0 ,5.0 ,
def test_cumsum_1d_reverse():
    print("== test_cumsum_1d_reverse")
    comptime exclusive = False
    comptime reverse = True
    comptime axis = 0

    var matrix_stack = Array[Float64, 5](fill={})
    var matrix = TileTensor(matrix_stack, row_major[5]())

    iota(matrix._storage, 5, 1)

    var cumsum_stack = Array[Float64, 5](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[5]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(5):
        print(cumsum_matrix[i], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_1d_reverse_exclusive
# CHECK: 14.0 ,12.0 ,9.0 ,5.0 ,0.0 ,
def test_cumsum_1d_reverse_exclusive():
    print("== test_cumsum_1d_reverse_exclusive")
    comptime exclusive = True
    comptime reverse = True
    comptime axis = 0

    var matrix_stack = Array[Float64, 5](fill={})
    var matrix = TileTensor(matrix_stack, row_major[5]())

    iota(matrix._storage, 5, 1)

    var cumsum_stack = Array[Float64, 5](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[5]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(5):
        print(cumsum_matrix[i], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_2d_axis_0
# CHECK: 1.0 ,2.0 ,3.0 ,5.0 ,7.0 ,9.0 ,
def test_cumsum_2d_axis_0():
    print("== test_cumsum_2d_axis_0")
    comptime exclusive = False
    comptime reverse = False
    comptime axis = 0

    var matrix_stack = Array[Float64, 6](fill={})
    var matrix = TileTensor(matrix_stack, row_major[2, 3]())

    iota(matrix._storage, 6, 1)

    var cumsum_stack = Array[Float64, 6](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[2, 3]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(2):
        for j in range(3):
            print(cumsum_matrix[i, j], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_2d_axis_1
# CHECK: 1.0 ,3.0 ,6.0 ,4.0 ,9.0 ,15.0 ,
def test_cumsum_2d_axis_1():
    print("== test_cumsum_2d_axis_1")
    comptime exclusive = False
    comptime reverse = False
    comptime axis = 1

    var matrix_stack = Array[Float64, 6](fill={})
    var matrix = TileTensor(matrix_stack, row_major[2, 3]())

    iota(matrix._storage, 6, 1)

    var cumsum_stack = Array[Float64, 6](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[2, 3]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(2):
        for j in range(3):
            print(cumsum_matrix[i, j], ",", end="")
    print()


# CHECK-LABEL: test_cumsum_2d_negative_axis
# CHECK: 1.0 ,3.0 ,6.0 ,4.0 ,9.0 ,15.0 ,
def test_cumsum_2d_negative_axis():
    print("== test_cumsum_2d_negative_axis")
    comptime exclusive = False
    comptime reverse = False
    comptime axis = -1

    var matrix_stack = Array[Float64, 6](fill={})
    var matrix = TileTensor(matrix_stack, row_major[2, 3]())

    iota(matrix._storage, 6, 1)

    var cumsum_stack = Array[Float64, 6](fill={})
    var cumsum_matrix = TileTensor(cumsum_stack, row_major[2, 3]())

    cumsum[.float64, exclusive, reverse, axis=axis](cumsum_matrix, matrix)

    for i in range(2):
        for j in range(3):
            print(cumsum_matrix[i, j], ",", end="")
    print()


def main():
    test_cumsum_1d()
    test_cumsum_1d_precision()
    test_cumsum_1d_exclusive()
    test_cumsum_1d_reverse()
    test_cumsum_1d_reverse_exclusive()
    test_cumsum_2d_axis_0()
    test_cumsum_2d_axis_1()
    test_cumsum_2d_negative_axis()
