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
from nn.gather_scatter import gather_elements
from std.testing import assert_equal


def main() raises:
    var ctx = DeviceContext(api="cpu")

    def test_gather_ax1(ctx: DeviceContext) raises:
        print("== test_gather_ax1")

        var data_stack: Array[Float32, 4] = [Float32(1), 2, 3, 4]
        var data = TileTensor(data_stack, row_major[2, 2]())

        var indices_stack: Array[Int32, 4] = [Int32(0), 0, 1, 0]
        var indices = TileTensor(indices_stack, row_major[2, 2]())

        var output_stack = Array[Float32, 4](fill={})
        var output = TileTensor(output_stack, row_major[2, 2]())

        gather_elements(data, indices, 1, output, ctx)

        assert_equal(output[0, 0], Float32(1))
        assert_equal(output[0, 1], Float32(1))
        assert_equal(output[1, 0], Float32(4))
        assert_equal(output[1, 1], Float32(3))

    # CHECK-LABEL: test_gather_ax1
    # CHECK-NOT: FAIL
    test_gather_ax1(ctx)

    def test_gather_ax0(ctx: DeviceContext) raises:
        print("== test_gather_ax0")

        var data_stack: Array[Float32, 9] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]
        var data = TileTensor(data_stack, row_major[3, 3]())

        var indices_stack: Array[Int32, 6] = [Int32(1), 2, 0, 2, 0, 0]
        var indices = TileTensor(indices_stack, row_major[2, 3]())

        var output_stack = Array[Float32, 6](fill={})
        var output = TileTensor(output_stack, row_major[2, 3]())

        gather_elements(data, indices, 0, output, ctx)

        assert_equal(output[0, 0], Float32(4))
        assert_equal(output[0, 1], Float32(8))
        assert_equal(output[0, 2], Float32(3))
        assert_equal(output[1, 0], Float32(7))
        assert_equal(output[1, 1], Float32(2))
        assert_equal(output[1, 2], Float32(3))

    # CHECK-LABEL: test_gather_ax0
    # CHECK-NOT: FAIL
    test_gather_ax0(ctx)

    def test_gather_neg_indices(ctx: DeviceContext) raises:
        print("== test_gather_neg_indices")

        var data_stack: Array[Float32, 9] = [
            Float32(1),
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]
        var data = TileTensor(data_stack, row_major[3, 3]())

        var indices_stack: Array[Int32, 6] = [Int32(-1), -2, 0, -2, 0, 0]
        var indices = TileTensor(indices_stack, row_major[2, 3]())

        var output_stack = Array[Float32, 6](fill={})
        var output = TileTensor(output_stack, row_major[2, 3]())

        gather_elements(data, indices, 0, output, ctx)

        assert_equal(output[0, 0], Float32(7))
        assert_equal(output[0, 1], Float32(5))
        assert_equal(output[0, 2], Float32(3))
        assert_equal(output[1, 0], Float32(4))
        assert_equal(output[1, 1], Float32(2))
        assert_equal(output[1, 2], Float32(3))

    # CHECK-LABEL: test_gather_neg_indices
    # CHECK-NOT: FAIL
    test_gather_neg_indices(ctx)
