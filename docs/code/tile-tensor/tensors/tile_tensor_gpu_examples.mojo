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
from max.gpu import (
    thread_idx,
    block_idx,
    global_idx,
    block_dim,
    grid_dim,
    lane_id,
    WARP_SIZE,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer, get_gpu_target
from layout import Coord, Idx, stack_allocation, TileTensor
from layout.tile_layout import row_major, blocked_product
from std.sys import has_accelerator
from std.sys.info import (
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_apple_gpu,
    is_nvidia_gpu,
    simd_width_of,
)
from std.testing import assert_equal, assert_false, assert_true
from std.sys import exit


def shared_memory_alloc_example() raises:
    comptime dtype = DType.float32
    comptime in_size = 128
    comptime block_size = 16
    comptime num_blocks = in_size // block_size  # number of block in one dimension
    comptime input_layout = row_major[in_size, in_size]()

    def kernel(tensor: TileTensor[dtype, type_of(input_layout), MutAnyOrigin]):
        # extract a tile from the input tensor.
        var global_tile = tensor.tile[block_size, block_size](
            Int(block_idx.y), Int(block_idx.x)
        )
        # start-shared-memory-alloc-example
        comptime tile_layout = row_major[block_size, block_size]()
        var shared_tile = stack_allocation[dtype, address_space=.SHARED](
            tile_layout
        )
        # end-shared-memory-alloc-example

        # Copy one element from the global tile to the shared tile.
        shared_tile[thread_idx.y, thread_idx.x] = global_tile[
            thread_idx.y, thread_idx.x
        ]
        barrier()

        # Put some data into the shared tile that we can verify on the host.
        if global_idx.x < in_size and global_idx.y < in_size:
            shared_tile[thread_idx.y, thread_idx.x] = Float32(
                global_idx.y * in_size + global_idx.x
            )

        barrier()
        global_tile[thread_idx.y, thread_idx.x] = shared_tile[
            thread_idx.y, thread_idx.x
        ]

    try:
        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](in_size * in_size)
        var dev_buf = ctx.enqueue_create_buffer[dtype](in_size * in_size)
        ctx.enqueue_memset(dev_buf, 0.0)
        var tensor = TileTensor(dev_buf, input_layout)

        ctx.enqueue_function[kernel](
            tensor,
            grid_dim=(num_blocks, num_blocks),
            block_dim=(block_size, block_size),
        )
        ctx.enqueue_copy(host_buf, dev_buf)
        ctx.synchronize()
        for i in range(in_size * in_size):
            if host_buf[i] != Float32(i):
                raise Error(
                    String("Error at position {} expected {} got {}").format(
                        i, i, host_buf[i]
                    )
                )
        print("success")
    except error:
        print(error)


def tile_tensor_vectorized_example() raises:
    comptime dtype = DType.int32
    comptime vector_width = 4

    comptime rows = 64
    comptime columns = 64
    comptime layout = row_major[rows, columns]()
    var storage = Array[Scalar[dtype], rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Int32(i)
    var tensor = TileTensor(storage, layout)
    # start-vectorize-tensor-example
    var vectorized_tensor = tensor.vectorize[1, 4]()
    # end-vectorize-tensor-example
    var values = vectorized_tensor[0, 0]
    # The SIMD width could be anywhere from 4 to 16 (possibly  more in the future)
    # So just test a single value.
    assert_equal(values[3], SIMD[dtype, 1](3))


def tile_tensor_distribute_example() raises:
    comptime rows = 4
    comptime columns = 8
    comptime layout = row_major[rows, columns]()
    comptime dtype = DType.int32

    def kernel(tensor: TileTensor[dtype, type_of(layout), MutAnyOrigin]):
        # start-vectorize-distribute-example
        comptime simd_size = 4
        var fragment = tensor.vectorize[1, simd_size]().distribute[
            thread_layout=row_major[2, simd_size]()
        ](lane_id())
        # end-vectorize-distribute-example
        _ = fragment

    try:
        var ctx = DeviceContext()
        var dev_buf = ctx.enqueue_create_buffer[.int32](rows * columns)
        var host_buf = ctx.enqueue_create_host_buffer[.int32](rows * columns)
        for i in range(rows * columns):
            host_buf[i] = Int32(i)
        var tensor = TileTensor(dev_buf, layout)
        ctx.enqueue_copy(dev_buf, host_buf)
        ctx.enqueue_function[kernel](
            tensor,
            grid_dim=(1, 1),
            block_dim=(8, 1),
        )

    except error:
        print(error)


# TODO: Add simple copy example to doc
def simple_copy_example():
    comptime dtype = DType.float32
    comptime rows = 128
    comptime cols = 128
    comptime block_size = 16
    comptime num_row_blocks = rows // block_size
    comptime num_col_blocks = cols // block_size
    comptime input_layout = row_major[rows, cols]()

    def kernel(tensor: TileTensor[dtype, type_of(input_layout), MutAnyOrigin]):
        # extract a tile from the input tensor.
        var global_tile = tensor.tile[block_size, block_size](
            Int(block_idx.y), Int(block_idx.x)
        )
        comptime tile_layout = row_major[block_size, block_size]()
        var shared_tile = stack_allocation[dtype, address_space=.SHARED](
            tile_layout
        )

        if global_idx.y < rows and global_idx.x < cols:
            shared_tile[thread_idx.y, thread_idx.x] = global_tile[
                thread_idx.y, thread_idx.x
            ]

        barrier()

        # Put some data into the shared tile that we can verify on the host.
        if global_idx.y < rows and global_idx.x < cols:
            shared_tile[thread_idx.y, thread_idx.x] = (
                shared_tile[thread_idx.y, thread_idx.x] * 2
            )
        barrier()

        if global_idx.y < rows and global_idx.x < cols:
            global_tile[thread_idx.y, thread_idx.x] = shared_tile[
                thread_idx.y, thread_idx.x
            ]

    try:
        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](rows * cols)
        var dev_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
        for i in range(rows * cols):
            host_buf[i] = Float32(i)
        ctx.enqueue_copy(dev_buf, host_buf)
        var tensor = TileTensor(dev_buf, input_layout)

        ctx.enqueue_function[kernel](
            tensor,
            grid_dim=(num_row_blocks, num_col_blocks),
            block_dim=(block_size, block_size),
        )
        ctx.enqueue_copy(host_buf, dev_buf)
        ctx.synchronize()
        for i in range(rows * cols):
            if host_buf[i] != Float32(i * 2):
                raise Error(
                    String("Unexpected value ", host_buf[i], " at position ", i)
                )
    except error:
        print(error)


def main() raises:
    if has_accelerator():
        shared_memory_alloc_example()
        tile_tensor_vectorized_example()
        tile_tensor_distribute_example()
        simple_copy_example()

    else:
        print("No accelerator, skipping examples that require a GPU.")
