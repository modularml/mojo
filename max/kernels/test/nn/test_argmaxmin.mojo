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

from layout import TileTensor, row_major
from nn.argmaxmin import argmax, argmin


# CHECK-LABEL: test_argn
def test_argn() raises:
    print("== test_argn")

    comptime size = 93

    comptime vector_shape = row_major[size]()
    var vector_stack = Array[Int32, vector_shape.product()](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    argmax(
        vector.make_dynamic[.int64](),
        0,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 92
    print("argmax = ", output[0])

    argmin(
        vector.make_dynamic[.int64](),
        0,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_2
def test_argn_2() raises:
    print("== test_argn_2")

    comptime batch_size = 4
    comptime size = 91

    comptime vector_shape = row_major[batch_size, size]()
    var vector_stack = Array[Float32, vector_shape.product()](
        fill_with=lambda (idx: Int) -> Float32: Float32(idx % size)
    )
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    argmax(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    # CHECK: argmax = 90
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_2_test_2
def test_argn_2_test_2() raises:
    print("== test_argn_2_test_2")

    comptime batch_size = 2
    comptime size = 3

    comptime vector_shape = row_major[batch_size, size]()
    var vector_stack = Array[Float32, vector_shape.product()](
        fill_with=lambda (idx: Int) -> Float32: (
            Float32(idx) if (idx // size) % 2 == 0 else Float32(-idx)
        )
    )
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    argmax(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_2_neg_axis
def test_argn_2_neg_axis() raises:
    print("== test_argn_2_neg_axis")

    comptime batch_size = 2
    comptime size = 3

    comptime vector_shape = row_major[batch_size, size]()
    var vector_stack = Array[Float32, vector_shape.product()](
        fill_with=lambda (idx: Int) -> Float32: (
            Float32(idx) if (idx // size) % 2 == 0 else Float32(-idx)
        )
    )
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    argmax(
        vector.make_dynamic[.int64](),
        -1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 2
    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(
        vector.make_dynamic[.int64](),
        -1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 2
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_test_zeros
def test_argn_test_zeros() raises:
    print("== test_argn_test_zeros")

    comptime batch_size = 1
    comptime size = 16

    comptime vector_shape = row_major[batch_size, size]()
    var vector_stack = Array[Float32, vector_shape.product()](fill=0)
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    argmax(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 0
    for i in range(batch_size):
        print("argmax = ", output[i, 0])

    argmin(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_test_identity
def test_argn_test_identity() raises:
    print("== test_argn_test_identity")

    comptime batch_size = 3
    comptime size = 5

    comptime vector_shape = row_major[batch_size, size]()
    var vector_stack = Array[Int64, vector_shape.product()](fill=0)
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    vector[1, 4] = 1
    vector[2, 3] = 1
    vector[2, 4] = 1

    argmax(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 0
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 4
    print("argmax = ", output[1, 0])
    # CHECK: argmax = 3
    print("argmax = ", output[2, 0])

    argmin(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        print("argmin = ", output[i, 0])


# CHECK-LABEL: test_argn_3d_identity
def test_argn_3d_identity() raises:
    print("== test_argn_3d_identity")

    comptime batch_size = 2
    comptime seq_len = 2
    comptime hidden_dim = 5

    comptime vector_shape = row_major[batch_size, seq_len, hidden_dim]()
    var vector_stack = Array[Int64, vector_shape.product()](fill=0)
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, seq_len, 1]()
    var output_stack = Array[Int, output_shape.product()](fill=0)
    var output = TileTensor(output_stack, output_shape)

    vector[0, 1, 4] = 1
    vector[1, 0, 1] = 1
    vector[1, 0, 2] = 1
    vector[1, 1, 3] = 1

    argmax(
        vector.make_dynamic[.int64](),
        2,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 0
    print("argmax = ", output[0, 0, 0])
    # CHECK: argmax = 4
    print("argmax = ", output[0, 1, 0])
    # CHECK: argmax = 1
    print("argmax = ", output[1, 0, 0])
    # CHECK: argmax = 3
    print("argmax = ", output[1, 1, 0])

    argmin(
        vector.make_dynamic[.int64](),
        2,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    # CHECK: argmin = 0
    for i in range(batch_size):
        for j in range(seq_len):
            print("argmin = ", output[i, j, 0])


def test_argn_less_than_simd() raises:
    print("== test_argn_less_than_simd")

    comptime batch_size = 2
    comptime hidden_dim = 3  # assumes simd_width of 4

    comptime vector_shape = row_major[batch_size, hidden_dim]()
    var vector_stack = Array[Int64, vector_shape.product()](fill=0)
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill=0)
    var output = TileTensor(output_stack, output_shape)

    vector[0, 0] = 0
    vector[0, 1] = 1
    vector[0, 2] = 2
    vector[1, 0] = 5
    vector[1, 1] = 4
    vector[1, 2] = 3

    argmax(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 2
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 0
    print("argmax = ", output[1, 0])

    argmin(
        vector.make_dynamic[.int64](),
        1,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 0
    print("argmin = ", output[0, 0])
    # CHECK: argmin = 2
    print("argmin = ", output[1, 0])


# CHECK-LABEL: test_argn_simd_edge_case
def test_argn_simd_index_order() raises:
    print("== test_argn_simd_edge_case")

    # Checks the case where the maximal value is found in two simd_chunks, where
    # the index of the maximal value in the second simd_chunk is earlier than in the first.
    # ex:
    #   simd_width = 4
    #   [0, 0, 1, 0, 0, 1, 0, 0, 0]
    #   <--------->  <-------->  <>
    #          ^        ^
    comptime size = 17

    comptime vector_shape = row_major[size]()
    var vector_stack = Array[Int32, vector_shape.product()](fill=0)
    var vector = TileTensor(vector_stack, vector_shape)

    comptime output_shape = row_major[1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    vector[5] = 1
    vector[4] = -1
    vector[8] = -1
    vector[9] = 1

    argmax(
        vector.make_dynamic[.int64](),
        0,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmax = 5
    print("argmax = ", output[0])

    argmin(
        vector.make_dynamic[.int64](),
        0,
        output.make_dynamic[.int64](),
    )

    # CHECK: argmin = 4
    print("argmin = ", output[0])


# CHECK-LABEL: test_argn_parallelize
def test_argn_parallelize() raises:
    print("== test_argn_parallelize")

    # Checks argn's performance when the size of the TileTensor exceeds the threshold to enable parallelism
    comptime batch_size = 8
    comptime hidden_dim = 16384

    comptime input_shape = row_major[batch_size, hidden_dim]()
    var input_ptr = List(length=input_shape.product(), fill=Float32(0))
    var input = TileTensor(input_ptr, input_shape)

    for i in range(batch_size):
        for j in range(hidden_dim):
            input[i, j] = 0

    comptime output_shape = row_major[batch_size, 1]()
    var output_stack = Array[Int, output_shape.product()](fill={})
    var output = TileTensor(output_stack, output_shape)

    input[0, 10] = 100
    input[0, 100] = -100
    input[1, 20] = -100
    input[1, 200] = 100
    input[2, 30] = 100
    input[2, 300] = -100
    input[3, 40] = -100
    input[3, 400] = 100
    input[4, 10] = 100
    input[4, 100] = -100
    input[5, 20] = 100
    input[5, 200] = -100
    input[6, 30] = 100
    input[6, 300] = -100
    input[7, 40] = -100
    input[7, 400] = 100

    argmax(input.make_dynamic[.int64](), 1, output.make_dynamic[.int64]())

    # CHECK: argmax = 10
    print("argmax = ", output[0, 0])
    # CHECK: argmax = 200
    print("argmax = ", output[1, 0])
    # CHECK: argmax = 30
    print("argmax = ", output[2, 0])
    # CHECK: argmax = 400
    print("argmax = ", output[3, 0])
    # CHECK: argmax = 10
    print("argmax = ", output[4, 0])
    # CHECK: argmax = 20
    print("argmax = ", output[5, 0])
    # CHECK: argmax = 30
    print("argmax = ", output[6, 0])
    # CHECK: argmax = 400
    print("argmax = ", output[7, 0])

    argmin(input.make_dynamic[.int64](), 1, output.make_dynamic[.int64]())

    # CHECK: argmin = 100
    print("argmin = ", output[0, 0])
    # CHECK: argmin = 20
    print("argmin = ", output[1, 0])
    # CHECK: argmin = 300
    print("argmin = ", output[2, 0])
    # CHECK: argmin = 40
    print("argmin = ", output[3, 0])
    # CHECK: argmin = 100
    print("argmin = ", output[4, 0])
    # CHECK: argmin = 200
    print("argmin = ", output[5, 0])
    # CHECK: argmin = 300
    print("argmin = ", output[6, 0])
    # CHECK: argmin = 40
    print("argmin = ", output[7, 0])
    _ = input_ptr^


def main() raises:
    test_argn()
    test_argn_2()
    test_argn_2_test_2()
    test_argn_2_neg_axis()
    test_argn_test_zeros()
    test_argn_test_identity()
    test_argn_3d_identity()
    test_argn_less_than_simd()
    test_argn_simd_index_order()
    test_argn_parallelize()
