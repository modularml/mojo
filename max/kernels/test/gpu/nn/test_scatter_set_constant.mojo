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

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.gather_scatter import scatter_set_constant


def test_scatter_set_constant(ctx: DeviceContext) raises:
    # TODO not sure why this doesn't work with Array?
    var data_stack = Array[Float32, 9](fill=0.0)
    var data = TileTensor(data_stack, row_major[3, 3]())
    var data_ptr_gpu = ctx.enqueue_create_buffer[.float32](3 * 3)
    ctx.enqueue_copy(data_ptr_gpu, Span(data_stack))

    var data_gpu = TileTensor(data_ptr_gpu, row_major[3, 3]())

    # One (row, col) index pair per scatter row.
    var array: Array[Int32, _] = [0, 1, 1, 2, 1, 3, 2, 0]
    var indices = TileTensor(array, row_major[4, 2]())

    var indices_ptr_gpu = ctx.enqueue_create_buffer[.int32](4 * 2)
    ctx.enqueue_copy(indices_ptr_gpu, indices._storage)
    var indices_gpu = TileTensor(indices_ptr_gpu, row_major[4, 2]())

    var fill_value: Float32 = 5.0
    var expected_stack = Array[Float32, 9](fill=0.0)
    var expected_output = TileTensor(expected_stack, row_major[3, 3]())

    expected_output[0, 1] = 5.0
    expected_output[1, 2] = 5.0
    expected_output[1, 3] = 5.0
    expected_output[2, 0] = 5.0

    var ctx_ptr = ctx

    scatter_set_constant[target="gpu",](
        data_gpu, indices_gpu, fill_value, ctx_ptr
    )

    ctx.enqueue_copy(Span(data_stack), data_ptr_gpu)
    ctx.synchronize()

    for i in range(3):
        for j in range(3):
            if data[i, j] != expected_output[i, j]:
                raise Error(
                    "data[",
                    i,
                    ", ",
                    j,
                    "] = ",
                    data[i, j],
                    " != ",
                    expected_output[i, j],
                )


def main() raises:
    with DeviceContext() as ctx:
        test_scatter_set_constant(ctx)
