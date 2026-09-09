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

from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from linalg.transpose import (
    _simplify_transpose_perms,
    transpose,
    transpose_inplace,
)

from std.utils.index import IndexList


# CHECK-LABEL: test_transpose_4x4_tiletensor
def test_transpose_4x4_tiletensor():
    print("== test_transpose_4x4_tiletensor")

    var matrix = stack_allocation[dtype=DType.int](row_major[4, 4]())

    matrix[0, 0] = 0
    matrix[0, 1] = 1
    matrix[0, 2] = 2
    matrix[0, 3] = 3
    matrix[1, 0] = 4
    matrix[1, 1] = 5
    matrix[1, 2] = 6
    matrix[1, 3] = 7
    matrix[2, 0] = 8
    matrix[2, 1] = 9
    matrix[2, 2] = 10
    matrix[2, 3] = 11
    matrix[3, 0] = 12
    matrix[3, 1] = 13
    matrix[3, 2] = 14
    matrix[3, 3] = 15

    transpose_inplace[4, 4, DType.int](matrix)

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 4
    print(matrix[0, 1])

    # CHECK: 8
    print(matrix[0, 2])

    # CHECK: 12
    print(matrix[0, 3])

    # CHECK: 1
    print(matrix[1, 0])

    # CHECK: 5
    print(matrix[1, 1])

    # CHECK: 9
    print(matrix[1, 2])

    # CHECK: 13
    print(matrix[1, 3])

    # CHECK: 2
    print(matrix[2, 0])

    # CHECK: 6
    print(matrix[2, 1])

    # CHECK: 10
    print(matrix[2, 2])

    # CHECK: 14
    print(matrix[2, 3])

    # CHECK: 3
    print(matrix[3, 0])

    # CHECK: 7
    print(matrix[3, 1])

    # CHECK: 11
    print(matrix[3, 2])

    # CHECK: 15
    print(matrix[3, 3])


# CHECK-LABEL: test_transpose_8x8_tiletensor
def test_transpose_8x8_tiletensor():
    print("== test_transpose_8x8_tiletensor")

    comptime num_rows: Int = 8
    comptime num_cols: Int = 8

    var matrix = stack_allocation[dtype=DType.int](
        row_major[num_rows, num_cols]()
    )

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[i, j] = Int(val)

    transpose_inplace[num_rows, num_cols, DType.int](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j])
            # CHECK-NOT: Transpose 8x8 failed
            if expected != actual:
                print("Transpose 8x8 failed")


# CHECK-LABEL: test_transpose_16x16
def test_transpose_16x16_tiletensor():
    print("== test_transpose_16x16_tiletensor")

    comptime num_rows: Int = 16
    comptime num_cols: Int = 16

    var matrix = stack_allocation[dtype=DType.int](
        row_major[num_rows, num_cols]()
    )

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[i, j] = Int(val)

    transpose_inplace[num_rows, num_cols, DType.int](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j])
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed")


# CHECK-LABEL: test_transpose_2d_identity_tiletensor
def test_transpose_2d_identity_tiletensor() raises:
    print("== test_transpose_2d_identity_tiletensor")

    var input = stack_allocation[dtype=DType.int](row_major[3, 3]())
    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[1, 0] = 4
    input[1, 1] = 5
    input[1, 2] = 6
    input[2, 0] = 7
    input[2, 1] = 8
    input[2, 2] = 9

    var perm: Array[Int, 2] = [0, 1]

    var output = stack_allocation[dtype=DType.int](row_major[3, 3]())
    _ = output.fill(0)

    transpose(output, input, perm.unsafe_ptr())

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 2
    print(output[0, 1])
    # CHECK: 3
    print(output[0, 2])
    # CHECK: 4
    print(output[1, 0])
    # CHECK: 5
    print(output[1, 1])
    # CHECK: 6
    print(output[1, 2])
    # CHECK: 7
    print(output[2, 0])
    # CHECK: 8
    print(output[2, 1])
    # CHECK: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_2d_tiletensor
def test_transpose_2d_tiletensor() raises:
    print("== test_transpose_2d_tiletensor")

    var input = stack_allocation[dtype=DType.int](row_major[3, 3]())
    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[1, 0] = 4
    input[1, 1] = 5
    input[1, 2] = 6
    input[2, 0] = 7
    input[2, 1] = 8
    input[2, 2] = 9

    var perm: Array[Int, 2] = [1, 0]

    var output = stack_allocation[dtype=DType.int](row_major[3, 3]())
    _ = output.fill(0)

    transpose(output, input, perm.unsafe_ptr())

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 4
    print(output[0, 1])
    # CHECK: 7
    print(output[0, 2])
    # CHECK: 2
    print(output[1, 0])
    # CHECK: 5
    print(output[1, 1])
    # CHECK: 8
    print(output[1, 2])
    # CHECK: 3
    print(output[2, 0])
    # CHECK: 6
    print(output[2, 1])
    # CHECK: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_3d_identity_tiletensor
def test_transpose_3d_identity_tiletensor() raises:
    print("== test_transpose_3d_identity_tiletensor")

    var input = stack_allocation[dtype=DType.int](row_major[2, 2, 3]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 0, 2] = 3
    input[0, 1, 0] = 4
    input[0, 1, 1] = 5
    input[0, 1, 2] = 6
    input[1, 0, 0] = 7
    input[1, 0, 1] = 8
    input[1, 0, 2] = 9
    input[1, 1, 0] = 10
    input[1, 1, 1] = 11
    input[1, 1, 2] = 12

    var perm: Array[Int, 3] = [0, 1, 2]

    var output = stack_allocation[dtype=DType.int](row_major[2, 2, 3]())
    _ = output.fill(0)

    transpose(output, input, perm.unsafe_ptr())

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 0, 1])
    # CHECK: 3
    print(output[0, 0, 2])
    # CHECK: 4
    print(output[0, 1, 0])
    # CHECK: 5
    print(output[0, 1, 1])
    # CHECK: 6
    print(output[0, 1, 2])
    # CHECK: 7
    print(output[1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1])
    # CHECK: 9
    print(output[1, 0, 2])
    # CHECK: 10
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 12
    print(output[1, 1, 2])


# CHECK-LABEL: test_transpose_3d_tiletensor
def test_transpose_3d_tiletensor() raises:
    print("== test_transpose_3d_tiletensor")

    var input = stack_allocation[dtype=DType.int](row_major[2, 2, 3]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 0, 2] = 3
    input[0, 1, 0] = 4
    input[0, 1, 1] = 5
    input[0, 1, 2] = 6
    input[1, 0, 0] = 7
    input[1, 0, 1] = 8
    input[1, 0, 2] = 9
    input[1, 1, 0] = 10
    input[1, 1, 1] = 11
    input[1, 1, 2] = 12

    var perm: Array[Int, 3] = [2, 0, 1]

    var output = stack_allocation[dtype=DType.int](row_major[3, 2, 2]())
    _ = output.fill(0)

    transpose(output, input, perm.unsafe_ptr())

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 4
    print(output[0, 0, 1])
    # CHECK: 7
    print(output[0, 1, 0])
    # CHECK: 10
    print(output[0, 1, 1])
    # CHECK: 2
    print(output[1, 0, 0])
    # CHECK: 5
    print(output[1, 0, 1])
    # CHECK: 8
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 3
    print(output[2, 0, 0])
    # CHECK: 6
    print(output[2, 0, 1])
    # CHECK: 9
    print(output[2, 1, 0])
    # CHECK: 12
    print(output[2, 1, 1])


# CHECK-LABEL: test_transpose_si64_tiletensor
def test_transpose_si64_tiletensor() raises:
    print("== test_transpose_si64_tiletensor")

    var input = stack_allocation[dtype=DType.int64](row_major[2, 2, 3]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 0, 2] = 3
    input[0, 1, 0] = 4
    input[0, 1, 1] = 5
    input[0, 1, 2] = 6
    input[1, 0, 0] = 7
    input[1, 0, 1] = 8
    input[1, 0, 2] = 9
    input[1, 1, 0] = 10
    input[1, 1, 1] = 11
    input[1, 1, 2] = 12

    var perm: Array[Int, 3] = [2, 1, 0]

    var output = stack_allocation[dtype=DType.int64](row_major[3, 2, 2]())
    _ = output.fill(0)

    transpose(output, input, perm.unsafe_ptr())

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 7
    print(output[0, 0, 1])
    # CHECK: 4
    print(output[0, 1, 0])
    # CHECK: 10
    print(output[0, 1, 1])
    # CHECK: 2
    print(output[1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1])
    # CHECK: 5
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 3
    print(output[2, 0, 0])
    # CHECK: 9
    print(output[2, 0, 1])
    # CHECK: 6
    print(output[2, 1, 0])
    # CHECK: 12
    print(output[2, 1, 1])


# CHECK-LABEL: test_simplify_perm
def test_simplify_perm():
    print("== test_simplify_perm")
    var perm = IndexList[4](0, 2, 3, 1)
    var shape = IndexList[4](8, 3, 200, 200)
    var rank = 4
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1
    print(perm)
    # CHECK: 8, 3, 40000
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 3, 1)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0
    print(perm)
    # CHECK: 3, 40000
    print(shape)
    # CHECK: 2
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](8, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1, 3
    print(perm)
    # CHECK: 8, 3, 200, 200
    print(shape)
    # CHECK: 4
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0, 2
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](2, 1, 0, 3)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0, 2
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](3, 2, 1, 0)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 2, 1, 0
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](1, 3, 1, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0
    print(perm)
    # CHECK: 600
    print(shape)
    # CHECK: 1
    print(rank)

    rank = 4
    perm = IndexList[4](0, 3, 1, 2)
    shape = IndexList[4](9, 1, 2, 3)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1
    print(perm)
    # CHECK: 9, 2, 3
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 2
    var perm2 = IndexList[2](0, 1)
    var shape2 = IndexList[2](20, 30)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 0
    print(perm2)
    # CHECK: 600
    print(shape2)
    # CHECK: 1
    print(rank)

    rank = 2
    perm2 = IndexList[2](1, 0)
    shape2 = IndexList[2](20, 30)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 1, 0
    print(perm2)
    # CHECK: 20, 30
    print(shape2)
    # CHECK: 2
    print(rank)

    rank = 2
    perm2 = IndexList[2](1, 0)
    shape2 = IndexList[2](20, 1)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 0
    print(perm2)
    # CHECK: 20
    print(shape2)
    # CHECK: 1
    print(rank)


# CHECK-LABEL: test_transpose_4x4
def test_transpose_4x4():
    print("== test_transpose_4x4")

    var matrix = stack_allocation[dtype=DType.int](row_major[4, 4]())

    matrix[0, 0] = 0
    matrix[0, 1] = 1
    matrix[0, 2] = 2
    matrix[0, 3] = 3
    matrix[1, 0] = 4
    matrix[1, 1] = 5
    matrix[1, 2] = 6
    matrix[1, 3] = 7
    matrix[2, 0] = 8
    matrix[2, 1] = 9
    matrix[2, 2] = 10
    matrix[2, 3] = 11
    matrix[3, 0] = 12
    matrix[3, 1] = 13
    matrix[3, 2] = 14
    matrix[3, 3] = 15

    transpose_inplace[4, 4](matrix)

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 4
    print(matrix[0, 1])

    # CHECK: 8
    print(matrix[0, 2])

    # CHECK: 12
    print(matrix[0, 3])

    # CHECK: 1
    print(matrix[1, 0])

    # CHECK: 5
    print(matrix[1, 1])

    # CHECK: 9
    print(matrix[1, 2])

    # CHECK: 13
    print(matrix[1, 3])

    # CHECK: 2
    print(matrix[2, 0])

    # CHECK: 6
    print(matrix[2, 1])

    # CHECK: 10
    print(matrix[2, 2])

    # CHECK: 14
    print(matrix[2, 3])

    # CHECK: 3
    print(matrix[3, 0])

    # CHECK: 7
    print(matrix[3, 1])

    # CHECK: 11
    print(matrix[3, 2])

    # CHECK: 15
    print(matrix[3, 3])


# CHECK-LABEL: test_transpose_8x8
def test_transpose_8x8():
    print("== test_transpose_8x8")

    comptime num_rows: Int = 8
    comptime num_cols: Int = 8

    var matrix = stack_allocation[dtype=DType.int](
        row_major[num_rows, num_cols]()
    )

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[i, j] = Int(val)

    transpose_inplace[num_rows, num_cols](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j][0])
            # CHECK-NOT: Transpose 8x8 failed
            if expected != actual:
                print("Transpose 8x8 failed")


# CHECK-LABEL: test_transpose_16x16
def test_transpose_16x16():
    print("== test_transpose_16x16")

    comptime num_rows: Int = 16
    comptime num_cols: Int = 16

    var matrix = stack_allocation[dtype=DType.int](
        row_major[num_rows, num_cols]()
    )

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[i, j] = Int(val)

    transpose_inplace[num_rows, num_cols](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j][0])
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed")


def main() raises:
    test_transpose_4x4_tiletensor()
    test_transpose_8x8_tiletensor()
    test_transpose_16x16_tiletensor()
    test_transpose_2d_identity_tiletensor()
    test_transpose_2d_tiletensor()
    test_transpose_3d_identity_tiletensor()
    test_transpose_3d_tiletensor()
    test_transpose_si64_tiletensor()
    test_simplify_perm()

    test_transpose_4x4()
    test_transpose_8x8()
    test_transpose_16x16()
