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
# DOC: max/tile-tensor/tensors.mdx

# start-initialize-tensor-from-cpu-example
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, stack_allocation, row_major
from std.sys import has_accelerator


def initialize_tensor_from_cpu_example() raises:
    comptime dtype = DType.float32
    comptime rows = 32
    comptime cols = 8
    comptime block_size = 8
    comptime row_blocks = rows // block_size
    comptime col_blocks = cols // block_size
    comptime input_layout = row_major[rows, cols]()
    comptime size: Int = rows * cols

    def kernel(tensor: TileTensor[dtype, type_of(input_layout), MutAnyOrigin]):
        if global_idx.y < Int(tensor.dim[0]()) and global_idx.x < Int(
            tensor.dim[1]()
        ):
            tensor[global_idx.y, global_idx.x] = (
                tensor[global_idx.y, global_idx.x] + 1
            )

    var ctx = DeviceContext()
    var host_buf = ctx.enqueue_create_host_buffer[dtype](size)
    var dev_buf = ctx.enqueue_create_buffer[dtype](size)
    ctx.synchronize()

    var expected_values = List[Scalar[dtype]](length=size, fill=0)

    for i in range(size):
        host_buf[i] = Scalar[dtype](i)
        expected_values[i] = Scalar[dtype](i + 1)
    ctx.enqueue_copy(dev_buf, host_buf)
    var tensor = TileTensor(dev_buf, input_layout)

    ctx.enqueue_function[kernel](
        tensor,
        grid_dim=(col_blocks, row_blocks),
        block_dim=(block_size, block_size),
    )
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(rows * cols):
        if host_buf[i] != expected_values[i]:
            raise Error(
                String("Error at position {} expected {} got {}").format(
                    i, expected_values[i], host_buf[i]
                )
            )


# end-initialize-tensor-from-cpu-example


def main() raises:
    if has_accelerator():
        initialize_tensor_from_cpu_example()
    else:
        print("No accelerator, skipping examples that require a GPU.")
