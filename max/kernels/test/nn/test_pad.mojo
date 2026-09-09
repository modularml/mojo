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
from nn.pad import pad_constant, pad_reflect, pad_repeat
from std.testing import assert_equal


# CHECK-LABEL: test_pad_1d
def test_pad_1d() raises:
    print("== test_pad_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: Array[Int, 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create a padding array of the form
    # [1, 2]
    var paddings_stack: Array[Int, 2] = [1, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0]
    var output_stack = Array[Int, 6](fill=0)
    var output = TileTensor(output_stack, row_major[6]())

    var constant = Int(5)

    # pad
    pad_constant(output, input, paddings._storage, constant)

    # output should have form
    # [5, 1, 2, 3, 5, 5]

    assert_equal(output[0], 5)
    assert_equal(output[1], 1)
    assert_equal(output[2], 2)
    assert_equal(output[3], 3)
    assert_equal(output[4], 5)
    assert_equal(output[5], 5)


# CHECK-LABEL: test_pad_reflect_1d
def test_pad_reflect_1d() raises:
    print("== test_pad_reflect_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: Array[Int, 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = Array[Int, 8](fill=0)
    var output = TileTensor(output_stack, row_major[8]())

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack: Array[Int, 2] = [3, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # pad
    pad_reflect(output, input, paddings._storage)

    # output should have form
    # [2, 3, 2, 1, 2, 3, 2, 1]

    assert_equal(output[0], 2)
    assert_equal(output[1], 3)
    assert_equal(output[2], 2)
    assert_equal(output[3], 1)
    assert_equal(output[4], 2)
    assert_equal(output[5], 3)
    assert_equal(output[6], 2)
    assert_equal(output[7], 1)


# CHECK-LABEL: test_pad_repeat_1d
def test_pad_repeat_1d() raises:
    print("== test_pad_repeat_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: Array[Int, 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = Array[Int, 8](fill=0)
    var output = TileTensor(output_stack, row_major[8]())

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack: Array[Int, 2] = [3, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # pad
    pad_repeat(output, input, paddings._storage)

    # output should have form
    # [1, 1, 1, 1, 2, 3, 3, 3]

    assert_equal(output[0], 1)
    assert_equal(output[1], 1)
    assert_equal(output[2], 1)
    assert_equal(output[3], 1)
    assert_equal(output[4], 2)
    assert_equal(output[5], 3)
    assert_equal(output[6], 3)
    assert_equal(output[7], 3)


# CHECK-LABEL: test_pad_2d
def test_pad_2d() raises:
    print("== test_pad_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # Create a padding array of the form
    # [1, 0, 1, 1]
    var paddings_stack: Array[Int, 4] = [1, 0, 1, 1]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0, 0, 0, 0]
    #  [0, 0, 0, 0]
    #  [0, 0, 0, 0]]
    var output_stack = Array[Int, 12](fill=0)
    var output = TileTensor(output_stack, row_major[3, 4]())

    var constant = Int(6)

    # pad
    pad_constant(output, input, paddings._storage, constant)

    # output should have form
    # [[6, 6, 6, 6]
    #  [6, 1, 2, 6]
    #  [6, 3, 4, 6]]

    assert_equal(output[0, 0], 6)
    assert_equal(output[0, 1], 6)
    assert_equal(output[0, 2], 6)
    assert_equal(output[0, 3], 6)
    assert_equal(output[1, 0], 6)
    assert_equal(output[1, 1], 1)
    assert_equal(output[1, 2], 2)
    assert_equal(output[1, 3], 6)
    assert_equal(output[2, 0], 6)
    assert_equal(output[2, 1], 3)
    assert_equal(output[2, 2], 4)
    assert_equal(output[2, 3], 6)


# CHECK-LABEL: test_pad_reflect_2d
def test_pad_reflect_2d() raises:
    print("== test_pad_reflect_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack: Array[Int, 4] = [2, 2, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = Array[Int, 18](fill=0)
    var output = TileTensor(output_stack, row_major[6, 3]())

    # pad
    pad_reflect(output, input, paddings._storage)

    # output should have form
    # [[2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]]

    assert_equal(output[0, 0], 2)
    assert_equal(output[0, 1], 1)
    assert_equal(output[0, 2], 2)
    assert_equal(output[1, 0], 4)
    assert_equal(output[1, 1], 3)
    assert_equal(output[1, 2], 4)
    assert_equal(output[2, 0], 2)
    assert_equal(output[2, 1], 1)
    assert_equal(output[2, 2], 2)
    assert_equal(output[3, 0], 4)
    assert_equal(output[3, 1], 3)
    assert_equal(output[3, 2], 4)
    assert_equal(output[4, 0], 2)
    assert_equal(output[4, 1], 1)
    assert_equal(output[4, 2], 2)
    assert_equal(output[5, 0], 4)
    assert_equal(output[5, 1], 3)
    assert_equal(output[5, 2], 4)


# CHECK-LABEL: test_pad_repeat_2d
def test_pad_repeat_2d() raises:
    print("== test_pad_repeat_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack: Array[Int, 4] = [2, 2, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = Array[Int, 18](fill=0)
    var output = TileTensor(output_stack, row_major[6, 3]())

    # pad
    pad_repeat(output, input, paddings._storage)

    # output should have form
    # [[1, 1, 2],
    #  [1, 1, 2],
    #  [1, 1, 2],
    #  [3, 3, 4],
    #  [3, 3, 4],
    #  [3, 3, 4]]

    assert_equal(output[0, 0], 1)
    assert_equal(output[0, 1], 1)
    assert_equal(output[0, 2], 2)
    assert_equal(output[1, 0], 1)
    assert_equal(output[1, 1], 1)
    assert_equal(output[1, 2], 2)
    assert_equal(output[2, 0], 1)
    assert_equal(output[2, 1], 1)
    assert_equal(output[2, 2], 2)
    assert_equal(output[3, 0], 3)
    assert_equal(output[3, 1], 3)
    assert_equal(output[3, 2], 4)
    assert_equal(output[4, 0], 3)
    assert_equal(output[4, 1], 3)
    assert_equal(output[4, 2], 4)
    assert_equal(output[5, 0], 3)
    assert_equal(output[5, 1], 3)
    assert_equal(output[5, 2], 4)


# CHECK-LABEL: test_pad_3d
def test_pad_3d() raises:
    print("== test_pad_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[1, 2, 2]())

    # Create a padding array of the form
    # [1, 0, 0, 1, 1, 0]
    var paddings_stack: Array[Int, 6] = [1, 0, 0, 1, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output matrix of the form
    # [[[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]
    #  [[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]]
    var output_stack = Array[Int, 18](fill=0)
    var output = TileTensor(output_stack, row_major[2, 3, 3]())

    var constant = Int(7)

    # pad
    pad_constant(output, input, paddings._storage, constant)

    # output should have form
    # [[[7, 7, 7]
    #   [7, 7, 7]
    #   [7, 7, 7]]
    #  [[7, 1, 2]
    #   [7, 3, 4]
    #   [7, 7, 7]]]

    assert_equal(output[0, 0, 0], 7)
    assert_equal(output[0, 0, 1], 7)
    assert_equal(output[0, 0, 2], 7)
    assert_equal(output[0, 1, 0], 7)
    assert_equal(output[0, 1, 1], 7)
    assert_equal(output[0, 1, 2], 7)
    assert_equal(output[0, 2, 0], 7)
    assert_equal(output[0, 2, 1], 7)
    assert_equal(output[0, 2, 2], 7)
    assert_equal(output[1, 0, 0], 7)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 7)
    assert_equal(output[1, 1, 1], 3)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 7)
    assert_equal(output[1, 2, 1], 7)
    assert_equal(output[1, 2, 2], 7)


# CHECK-LABEL: test_pad_reflect_3d
def test_pad_reflect_3d() raises:
    print("== test_pad_reflect_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4, 1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack: Array[Int, 6] = [1, 1, 0, 1, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output matrix of the form
    # [[[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]]
    var output_stack = Array[Int, 36](fill=0)
    var output = TileTensor(output_stack, row_major[4, 3, 3]())

    # pad
    pad_reflect(output, input, paddings._storage)

    # output should have form
    # [[[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]]

    assert_equal(output[0, 0, 0], 2)
    assert_equal(output[0, 0, 1], 1)
    assert_equal(output[0, 0, 2], 2)
    assert_equal(output[0, 1, 0], 4)
    assert_equal(output[0, 1, 1], 3)
    assert_equal(output[0, 1, 2], 4)
    assert_equal(output[0, 2, 0], 2)
    assert_equal(output[0, 2, 1], 1)
    assert_equal(output[0, 2, 2], 2)
    assert_equal(output[1, 0, 0], 2)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 4)
    assert_equal(output[1, 1, 1], 3)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 2)
    assert_equal(output[1, 2, 1], 1)
    assert_equal(output[1, 2, 2], 2)
    assert_equal(output[2, 0, 0], 2)
    assert_equal(output[2, 0, 1], 1)
    assert_equal(output[2, 0, 2], 2)
    assert_equal(output[2, 1, 0], 4)
    assert_equal(output[2, 1, 1], 3)
    assert_equal(output[2, 1, 2], 4)
    assert_equal(output[2, 2, 0], 2)
    assert_equal(output[2, 2, 1], 1)
    assert_equal(output[2, 2, 2], 2)
    assert_equal(output[3, 0, 0], 2)
    assert_equal(output[3, 0, 1], 1)
    assert_equal(output[3, 0, 2], 2)
    assert_equal(output[3, 1, 0], 4)
    assert_equal(output[3, 1, 1], 3)
    assert_equal(output[3, 1, 2], 4)
    assert_equal(output[3, 2, 0], 2)
    assert_equal(output[3, 2, 1], 1)
    assert_equal(output[3, 2, 2], 2)


# CHECK-LABEL: test_pad_reflect_3d_singleton
def test_pad_reflect_3d_singleton() raises:
    print("== test_pad_reflect_3d_singleton")

    # Create an input matrix of the form
    # [[[1]]]
    var input_stack: Array[Int, _] = [1]
    var input = TileTensor(input_stack, row_major[1, 1, 1]())

    # Create a padding array of the form
    # [1, 0, 0, 1, 2, 2]
    var paddings_stack: Array[Int, 6] = [
        1,
        0,
        0,
        1,
        2,
        2,
    ]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output matrix of the form
    # [[[0 0 0 0 0]
    #   [0 0 0 0 0]]
    #  [[0 0 0 0 0]
    #   [0 0 0 0 0]]]
    var output_stack = Array[Int, 20](fill=0)
    var output = TileTensor(output_stack, row_major[2, 2, 5]())

    # pad
    pad_reflect(output, input, paddings._storage)

    # output should have the form
    # [[[1 1 1 1 1]
    #   [1 1 1 1 1]]
    #  [[1 1 1 1 1]
    #   [1 1 1 1 1]]]

    assert_equal(output[0, 0, 0], 1)
    assert_equal(output[0, 0, 1], 1)
    assert_equal(output[0, 0, 2], 1)
    assert_equal(output[0, 0, 3], 1)
    assert_equal(output[0, 0, 4], 1)
    assert_equal(output[0, 1, 0], 1)
    assert_equal(output[0, 1, 1], 1)
    assert_equal(output[0, 1, 2], 1)
    assert_equal(output[0, 1, 3], 1)
    assert_equal(output[0, 1, 4], 1)
    assert_equal(output[1, 0, 0], 1)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 1)
    assert_equal(output[1, 0, 3], 1)
    assert_equal(output[1, 0, 4], 1)
    assert_equal(output[1, 1, 0], 1)
    assert_equal(output[1, 1, 1], 1)
    assert_equal(output[1, 1, 2], 1)
    assert_equal(output[1, 1, 3], 1)
    assert_equal(output[1, 1, 4], 1)


# CHECK-LABEL: test_pad_reflect_4d_big_input
def test_pad_reflect_4d_big_input() raises:
    print("== test_pad_reflect_4d_big_input")

    comptime in_size = 1 * 1 * 512 * 512
    comptime out_size = 2 * 3 * 1024 * 1024

    # create a big input matrix and fill it with ones
    var input_ptr = List(length=in_size, fill=Int(1))
    var input = TileTensor(input_ptr, row_major[1, 1, 512, 512]())

    # create a padding array of the form
    # [1, 0, 1, 1, 256, 256, 256, 256]
    var paddings_stack: Array[Int, 8] = [
        1,
        0,
        1,
        1,
        256,
        256,
        256,
        256,
    ]
    var paddings = TileTensor(paddings_stack, row_major[8]())

    # create an even bigger output matrix and fill it with zeros
    var output_ptr = List(length=out_size, fill=Int(0))
    var output = TileTensor(output_ptr, row_major[2, 3, 1024, 1024]())

    # pad
    pad_reflect(output, input, paddings._storage)

    assert_equal(output[0, 0, 0, 0], 1)
    _ = output_ptr^
    _ = input_ptr^


# CHECK-LABEL: test_pad_repeat_3d
def test_pad_repeat_3d() raises:
    print("== test_pad_repeat_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack: Array[Int, _] = [1, 2, 3, 4, 1, 2, 3, 4]
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack: Array[Int, 6] = [1, 2, 0, 2, 0, 1]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output array equivalent to np.zeros((5, 4, 3))
    var output_stack = Array[Int, 60](fill=0)
    var output = TileTensor(output_stack, row_major[5, 4, 3]())

    # pad
    pad_repeat(output, input, paddings._storage)

    # output should have form
    # [[[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]]]

    assert_equal(output[0, 0, 0], 1)
    assert_equal(output[0, 0, 1], 2)
    assert_equal(output[0, 0, 2], 2)
    assert_equal(output[0, 1, 0], 3)
    assert_equal(output[0, 1, 1], 4)
    assert_equal(output[0, 1, 2], 4)
    assert_equal(output[0, 2, 0], 3)
    assert_equal(output[0, 2, 1], 4)
    assert_equal(output[0, 2, 2], 4)
    assert_equal(output[0, 3, 0], 3)
    assert_equal(output[0, 3, 1], 4)
    assert_equal(output[0, 3, 2], 4)
    assert_equal(output[1, 0, 0], 1)
    assert_equal(output[1, 0, 1], 2)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 3)
    assert_equal(output[1, 1, 1], 4)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 3)
    assert_equal(output[1, 2, 1], 4)
    assert_equal(output[1, 2, 2], 4)
    assert_equal(output[1, 3, 0], 3)
    assert_equal(output[1, 3, 1], 4)
    assert_equal(output[1, 3, 2], 4)
    assert_equal(output[2, 0, 0], 1)
    assert_equal(output[2, 0, 1], 2)
    assert_equal(output[2, 0, 2], 2)
    assert_equal(output[2, 1, 0], 3)
    assert_equal(output[2, 1, 1], 4)
    assert_equal(output[2, 1, 2], 4)
    assert_equal(output[2, 2, 0], 3)
    assert_equal(output[2, 2, 1], 4)
    assert_equal(output[2, 2, 2], 4)
    assert_equal(output[2, 3, 0], 3)
    assert_equal(output[2, 3, 1], 4)
    assert_equal(output[2, 3, 2], 4)
    assert_equal(output[3, 0, 0], 1)
    assert_equal(output[3, 0, 1], 2)
    assert_equal(output[3, 0, 2], 2)
    assert_equal(output[3, 1, 0], 3)
    assert_equal(output[3, 1, 1], 4)
    assert_equal(output[3, 1, 2], 4)
    assert_equal(output[3, 2, 0], 3)
    assert_equal(output[3, 2, 1], 4)
    assert_equal(output[3, 2, 2], 4)
    assert_equal(output[3, 3, 0], 3)
    assert_equal(output[3, 3, 1], 4)
    assert_equal(output[3, 3, 2], 4)
    assert_equal(output[4, 0, 0], 1)
    assert_equal(output[4, 0, 1], 2)
    assert_equal(output[4, 0, 2], 2)
    assert_equal(output[4, 1, 0], 3)
    assert_equal(output[4, 1, 1], 4)
    assert_equal(output[4, 1, 2], 4)
    assert_equal(output[4, 2, 0], 3)
    assert_equal(output[4, 2, 1], 4)
    assert_equal(output[4, 2, 2], 4)
    assert_equal(output[4, 3, 0], 3)
    assert_equal(output[4, 3, 1], 4)
    assert_equal(output[4, 3, 2], 4)


def main() raises:
    test_pad_1d()
    test_pad_reflect_1d()
    test_pad_repeat_1d()
    test_pad_2d()
    test_pad_reflect_2d()
    test_pad_repeat_2d()
    test_pad_3d()
    test_pad_reflect_3d()
    test_pad_reflect_3d_singleton()
    test_pad_reflect_4d_big_input()
    test_pad_repeat_3d()
