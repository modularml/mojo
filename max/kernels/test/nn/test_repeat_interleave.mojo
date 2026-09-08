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
from max.gpu.host import DeviceContext
from nn.repeat_interleave import _collapse_dims_around_axis, repeat_interleave

from std.utils.index import IndexList


def test_collapse_dims_around_axis() raises:
    # CHECK-LABEL: test_collapse_dims_around_axis
    print("test_collapse_dims_around_axis")

    # CHECK: (1, 100, 1)
    print(_collapse_dims_around_axis(IndexList[1](100), 0))

    # CHECK: (1, 17, 23)
    print(_collapse_dims_around_axis(IndexList[2](17, 23), 0))

    # CHECK: (17, 23, 1)
    print(_collapse_dims_around_axis(IndexList[2](17, 23), 1))

    # CHECK: (1, 2, 16)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 0))

    # CHECK: (4, 2, 4)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 2))

    # CHECK: (16, 2, 1)
    print(_collapse_dims_around_axis(IndexList[5](2, 2, 2, 2, 2), 4))


def test_repeat_interleave_1d(ctx: DeviceContext) raises:
    # CHECK-LABEL: test_repeat_interleave_1d
    print("test_repeat_interleave_1d")

    comptime rank = 1
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[4]())

    # rank_repeats is always 1
    comptime rank_repeats = 1
    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [1, 2, 3, 4]
    var repeats = TileTensor(repeats_stack, row_major[4,]())

    var output_stack = Array[Scalar[type], 10](fill={})
    var output = TileTensor(output_stack, row_major[10]())

    repeat_interleave(input, repeats, 0, output, ctx)

    # CHECK: 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0
    print()
    for i in range(10):
        print(output[i], ", ", sep="", end="")
    print()


def test_repeat_interleave_1d_broadcast_repeats(ctx: DeviceContext) raises:
    # CHECK-LABEL: test_repeat_interleave_1d_broadcast_repeats
    print("test_repeat_interleave_1d_broadcast_repeats")

    comptime rank = 1
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[4]())

    # rank_repeats is always 1
    comptime rank_repeats = 1
    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2]
    var repeats = TileTensor(repeats_stack, row_major[1]())

    var output_stack = Array[Scalar[type], 8](fill={})
    var output = TileTensor(output_stack, row_major[8]())

    repeat_interleave(input, repeats, 0, output, ctx)

    # CHECK: 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
    print()
    for i in range(8):
        print(output[i], ", ", sep="", end="")
    print()


def test_repeat_interleave_2d_axis_0(ctx: DeviceContext) raises:
    # CHECK-LABEL: test_repeat_interleave_2d_axis_0
    print("test_repeat_interleave_2d_axis_0")

    comptime rank = 2
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # rank_repeats is always 1
    comptime rank_repeats = 1
    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack = Array[Scalar[type_repeats], 4](fill={})
    var repeats = TileTensor(repeats_stack, row_major[2]())

    repeats[0] = 2
    repeats[1] = 3

    # Result is 2x5
    var output_stack = Array[Scalar[type], 10](fill={})
    var output = TileTensor(output_stack, row_major[5, 2]())

    repeat_interleave(input, repeats, 0, output, ctx)

    # CHECK: 0.0, 1.0,
    # CHECK: 0.0, 1.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    print()
    for i in range(5):
        for j in range(2):
            print(output[i, j], ", ", sep="", end="")
        print()
    print()


def test_repeat_interleave_2d_axis_1(ctx: DeviceContext) raises:
    # CHECK-LABEL: test_repeat_interleave_2d_axis_1
    print("test_repeat_interleave_2d_axis_1")

    comptime rank = 2
    comptime type = DType.float32

    var input_stack: Array[Scalar[type], _] = [0, 1, 2, 3]
    var input = TileTensor(input_stack, row_major[2, 2]())

    # rank_repeats is always 1
    comptime rank_repeats = 1
    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 3]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Result is 2x5
    var output_stack = Array[Scalar[type], 10](fill={})
    var output = TileTensor(output_stack, row_major[2, 5]())

    repeat_interleave(input, repeats, 1, output, ctx)

    # CHECK: 0.0, 0.0, 1.0, 1.0, 1.0
    # CHECK: 2.0, 2.0, 3.0, 3.0, 3.0
    print()
    for i in range(2):
        for j in range(5):
            print(output[i, j], ", ", sep="", end="")
        print()
    print()


def test_repeat_interleave_3d(ctx: DeviceContext) raises:
    # CHECK-LABEL: test_repeat_interleave_3d
    print("test_repeat_interleave_3d")

    comptime rank = 3
    comptime type = DType.float32

    var input_stack = Array[Scalar[type], 8](
        fill_with=lambda (i: Int) -> Scalar[type]: Scalar[type](i)
    )
    var input = TileTensor(input_stack, row_major[2, 2, 2]())

    # rank_repeats is always 1
    comptime rank_repeats = 1
    # type_repeats is always .int64
    comptime type_repeats = DType.int64

    var repeats_stack: Array[Scalar[type_repeats], _] = [2, 3]
    var repeats = TileTensor(repeats_stack, row_major[2]())

    # Result is 2x5
    var output_stack = Array[Scalar[type], 20](fill={})
    var output = TileTensor(output_stack, row_major[2, 5, 2]())

    repeat_interleave(input, repeats, 1, output, ctx)

    # CHECK: 0.0, 1.0,
    # CHECK: 0.0, 1.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: 2.0, 3.0,
    # CHECK: =====
    # CHECK: 4.0, 5.0,
    # CHECK: 4.0, 5.0,
    # CHECK: 6.0, 7.0,
    # CHECK: 6.0, 7.0,
    # CHECK: 6.0, 7.0,
    # CHECK: =====

    print()
    for i in range(2):
        for j in range(5):
            for k in range(2):
                print(output[i, j, k], ", ", sep="", end="")
            print()
        print("=====")
    print()


def main() raises:
    with DeviceContext(api="cpu") as ctx:
        test_collapse_dims_around_axis()
        test_repeat_interleave_1d(ctx)
        test_repeat_interleave_1d_broadcast_repeats(ctx)
        test_repeat_interleave_2d_axis_0(ctx)
        test_repeat_interleave_2d_axis_1(ctx)
        test_repeat_interleave_3d(ctx)
