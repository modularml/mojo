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
from nn.broadcast import broadcast


# CHECK-LABEL: test_broadcast_empty_shape
def test_broadcast_empty_shape():
    print("== test_broadcast_empty_shape")

    # parameters
    comptime input_layout = row_major[1]()
    comptime output_layout = row_major[0]()

    # Create a 1D tensor of layout (1), of the form [1]
    var input_stack = Array[Int, input_layout.product()](fill=1)
    var input = TileTensor(input_stack, input_layout)

    # Create a 1D tensor of shape (0)
    # Note: output_layout.product() is 0, but we need to allocate a buffer for the
    # output tensor.
    var output_stack = Array[Int, 1](fill={})
    var output = TileTensor(output_stack, output_layout)

    broadcast(output, input)
    # output tensor will have the form:
    # []

    # CHECK: 1
    print(input[0])

    # test shouldn't crash


# CHECK-LABEL: test_broadcast_same_shape
def test_broadcast_same_shape():
    print("== test_broadcast_same_shape")

    # parameters
    comptime input_layout = row_major[1, 2, 1]()
    comptime output_layout = row_major[1, 2, 1]()

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input_stack = Array[Int, input_layout.product()](
        fill_with=lambda (i: Int) -> Int: i + 1
    )
    var input = TileTensor(input_stack, input_layout)

    # Create a 3D tensor of shape (1, 2, 1)
    var output_stack = Array[Int, output_layout.product()](fill=0)
    var output = TileTensor(output_stack, output_layout)

    broadcast(output, input)
    # output tensor will have the form:
    # [[[1], [2]]]

    # CHECK: 1
    print(input[0, 0, 0])
    # CHECK: 2
    print(input[0, 1, 0])

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0])


# CHECK-LABEL: test_broadcast_single_axis
def test_broadcast_single_axis():
    print("== test_broadcast_single_axis")

    # parameters
    comptime input_layout = row_major[1, 2]()
    comptime output_layout = row_major[3, 2]()

    # Create a 2D tensor of shape (1, 2), of the form
    # [[1, 2]]
    var input_stack = Array[Int, input_layout.product()](
        fill_with=lambda (i: Int) -> Int: i + 1
    )
    var input = TileTensor(input_stack, input_layout)

    # Create a 2D tensor of shape (3, 2)
    var output_stack = Array[Int, output_layout.product()](fill=0)
    var output = TileTensor(output_stack, output_layout)

    broadcast(output, input)
    # output tensor will have the form:
    # [[1, 2], [1, 2], [1, 2]]

    # CHECK: 1
    print(input[0, 0])
    # CHECK: 2
    print(input[0, 1])

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 2
    print(output[0, 1])
    # CHECK: 1
    print(output[1, 0])
    # CHECK: 2
    print(output[1, 1])
    # CHECK: 1
    print(output[2, 0])
    # CHECK: 2
    print(output[2, 1])


# CHECK-LABEL: test_broadcast_multi_axes
def test_broadcast_multi_axes():
    print("== test_broadcast_multi_axes")

    # parameters
    comptime input_layout = row_major[1, 2, 1]()
    comptime output_layout = row_major[2, 2, 3]()

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input_stack = Array[Int, input_layout.product()](
        fill_with=lambda (i: Int) -> Int: i + 1
    )
    var input = TileTensor(input_stack, input_layout)

    # Create a 3D tensor of shape (2, 2, 3)
    var output_stack = Array[Int, output_layout.product()](fill=0)
    var output = TileTensor(output_stack, output_layout)

    broadcast(output, input)
    # output tensor will have the form:
    # [[[1, 1, 1], [2, 2, 2]],
    #  [[1, 1, 1], [2, 2, 2]]]

    # CHECK: 1
    print(input[0, 0, 0])
    # CHECK: 2
    print(input[0, 1, 0])

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0])
    # CHECK: 1
    print(output[0, 0, 1])
    # CHECK: 2
    print(output[0, 1, 1])
    # CHECK: 1
    print(output[0, 0, 2])
    # CHECK: 2
    print(output[0, 1, 2])
    # CHECK: 1
    print(output[1, 0, 0])
    # CHECK: 2
    print(output[1, 1, 0])
    # CHECK: 1
    print(output[1, 0, 1])
    # CHECK: 2
    print(output[1, 1, 1])
    # CHECK: 1
    print(output[1, 0, 2])
    # CHECK: 2
    print(output[1, 1, 2])


def test_broadcast_multi_axes_nested():
    # parameters
    comptime input_layout = row_major[2, 1, 2, 1, 2]()
    comptime output_layout = row_major[2, 2, 2, 2, 2]()

    # Create a 5D tensor of shape (2, 1, 2, 1, 2), of the form
    # [[[[[1, 2]], [[3, 4]]]], [[[[5, 6]], [[7, 8]]]]]
    var input_stack = Array[Int, input_layout.product()](
        fill_with=lambda (i: Int) -> Int: i + 1
    )
    var input = TileTensor(input_stack, input_layout)

    # Create a 5D tensor of shape (2, 2, 2, 2, 2)
    var output_stack = Array[Int, output_layout.product()](fill=0)
    var output = TileTensor(output_stack, output_layout)

    broadcast(output, input)

    # CHECK: 1
    print(output[0, 0, 0, 0, 0])
    # CHECK: 2
    print(output[0, 0, 0, 0, 1])
    # CHECK: 1
    print(output[0, 0, 0, 1, 0])
    # CHECK: 2
    print(output[0, 0, 0, 1, 1])
    # CHECK: 3
    print(output[0, 0, 1, 0, 0])
    # CHECK: 4
    print(output[0, 0, 1, 0, 1])
    # CHECK: 3
    print(output[0, 0, 1, 1, 0])
    # CHECK: 4
    print(output[0, 0, 1, 1, 1])

    # CHECK: 1
    print(output[0, 1, 0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0, 0, 1])
    # CHECK: 1
    print(output[0, 1, 0, 1, 0])
    # CHECK: 2
    print(output[0, 1, 0, 1, 1])
    # CHECK: 3
    print(output[0, 1, 1, 0, 0])
    # CHECK: 4
    print(output[0, 1, 1, 0, 1])
    # CHECK: 3
    print(output[0, 1, 1, 1, 0])
    # CHECK: 4
    print(output[0, 1, 1, 1, 1])

    # CHECK: 5
    print(output[1, 0, 0, 0, 0])
    # CHECK: 6
    print(output[1, 0, 0, 0, 1])
    # CHECK: 5
    print(output[1, 0, 0, 1, 0])
    # CHECK: 6
    print(output[1, 0, 0, 1, 1])
    # CHECK: 7
    print(output[1, 0, 1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1, 0, 1])
    # CHECK: 7
    print(output[1, 0, 1, 1, 0])
    # CHECK: 8
    print(output[1, 0, 1, 1, 1])

    # CHECK: 5
    print(output[1, 1, 0, 0, 0])
    # CHECK: 6
    print(output[1, 1, 0, 0, 1])
    # CHECK: 5
    print(output[1, 1, 0, 1, 0])
    # CHECK: 6
    print(output[1, 1, 0, 1, 1])
    # CHECK: 7
    print(output[1, 1, 1, 0, 0])
    # CHECK: 8
    print(output[1, 1, 1, 0, 1])
    # CHECK: 7
    print(output[1, 1, 1, 1, 0])
    # CHECK: 8
    print(output[1, 1, 1, 1, 1])


def main():
    test_broadcast_empty_shape()
    test_broadcast_same_shape()
    test_broadcast_single_axis()
    test_broadcast_multi_axes()
    test_broadcast_multi_axes_nested()
