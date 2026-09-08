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

# Test gather_2D_input_1D_indices_axis_0.
# This test verifies that the prefetch function in `gather` passes
# compilation. The test can also be used to check the assembly to see
# if compiler generates proper SIMD instructions and unrolling.

from std.sys import simd_width_of

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.gather_scatter import gather


# CHECK-LABEL: test_gather
def test_gather() raises:
    print("== test_gather")

    @always_inline
    @__parameter
    def _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4

        # Setup input.
        var input_stack = Array[Float32, num_rows * row_size](
            fill_with=lambda (idx: Int) -> Float32: Float32(idx // row_size)
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size]())

        # Setup indices.
        comptime num_indices = 16
        var indices_stack = Array[Scalar[indices_type], num_indices](
            fill_with=lambda (i: Int) -> Scalar[indices_type]: Scalar[
                indices_type
            ](i // 2)
        )
        var indices = TileTensor(indices_stack, row_major[num_indices]())

        indices[0] = -1
        indices[1] = -num_rows

        # create output
        var output_stack = Array[Float32, num_indices * row_size](fill={})
        var output = TileTensor(
            output_stack, row_major[num_indices, row_size]()
        )

        # Test gather
        comptime simd_width = simd_width_of[__mlir_type.`!kgen.scalar<f32>`]()

        gather[axis=0](
            output.make_dynamic[.int64](),
            input.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            context=DeviceContext(api="cpu"),
        )

        print(output[0, 0])
        print(output[1, 0])
        print(output[2, 0])
        print(output[6, 0])
        print(output[15, 0])

    # CHECK: 15.0
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int64]()


def test_gather_3d() raises:
    print("== test_gather_3d\n")

    @always_inline
    @__parameter
    def _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4

        # Setup input.
        var input_stack = Array[Float32, num_rows * row_size * 1](
            fill_with=lambda (idx: Int) -> Float32: Float32(idx // row_size)
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size, 1]())

        # Setup indices.
        comptime num_indices = 16
        var indices_stack = Array[Scalar[indices_type], num_indices * 1](
            fill_with=lambda (i: Int) -> Scalar[indices_type]: Scalar[
                indices_type
            ](i // 2)
        )
        var indices = TileTensor(indices_stack, row_major[num_indices, 1]())

        # create output
        var output_stack = Array[Float32, num_indices * 1 * row_size * 1](
            fill={}
        )
        var output = TileTensor(
            output_stack, row_major[num_indices, 1, row_size, 1]()
        )

        # Test gather
        comptime simd_width = simd_width_of[DType.float32]()

        gather[axis=0](
            output.make_dynamic[.int64](),
            input.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            context=DeviceContext(api="cpu"),
        )

        print(output[0, 0, 0, 0])
        print(output[2, 0, 0, 0])
        print(output[6, 0, 0, 0])
        print(output[15, 0, 0, 0])

    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int64]()


# CHECK-LABEL: test_gather_empty_indices
def test_gather_empty_indices() raises:
    print("== test_gather_empty_indices")

    @always_inline
    @__parameter
    def _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4
        comptime num_indices = 0

        # Setup input.
        var input_stack = Array[Float32, num_rows * row_size](
            fill_with=lambda (idx: Int) -> Float32: Float32(idx // row_size)
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size]())

        # Setup indices.
        # There isn't a way to represent a stack size of 0 with Array
        # so we use 1 here
        var indices_stack = Array[Scalar[indices_type], 1](fill={})
        var indices = TileTensor(indices_stack, row_major[num_indices]())

        for i in range(num_indices):
            indices[i] = Scalar[indices_type](i // 2)

        # create output
        var output_stack = Array[Float32, num_rows * row_size](fill={})
        var output = TileTensor(
            output_stack, row_major[num_indices, row_size]()
        )

        # Test gather
        comptime simd_width = simd_width_of[DType.float32]()

        gather[axis=0](
            output.make_dynamic[.int64](),
            input.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            context=DeviceContext(api="cpu"),
        )

    _test_gather[.int32]()
    _test_gather[.int64]()


def main() raises:
    test_gather()
    test_gather_3d()
    test_gather_empty_indices()
