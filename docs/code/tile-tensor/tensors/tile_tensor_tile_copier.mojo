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
    WARP_SIZE,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import async_copy_commit_group, async_copy_wait_all
from layout import TileTensor, stack_allocation, row_major
from layout.tile_io import (
    GenericToSharedAsyncTileCopier,
    SharedToGenericTileCopier,
)
from std.sys import has_accelerator


def tile_copier_example() raises:
    comptime dtype = DType.float32
    comptime rows = 128
    comptime cols = 128
    comptime block_size = 16
    comptime num_row_blocks = rows // block_size
    comptime num_col_blocks = cols // block_size
    comptime input_layout = row_major[rows, cols]()
    comptime simd_width = 4

    def kernel(tensor: TileTensor[dtype, type_of(input_layout), MutAnyOrigin]):
        var global_tile = tensor.tile[block_size, block_size](
            Int(block_idx.y), Int(block_idx.x)
        )
        comptime tile_layout = row_major[block_size, block_size]()
        var shared_tile = stack_allocation[dtype, address_space=.SHARED](
            tile_layout
        )

        comptime thread_layout = row_major[
            WARP_SIZE // simd_width, simd_width
        ]()

        GenericToSharedAsyncTileCopier[thread_layout]().copy(
            shared_tile.vectorize[1, simd_width](),
            global_tile.vectorize[1, simd_width](),
        )
        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        if global_idx.y < rows and global_idx.x < cols:
            shared_tile[thread_idx.y, thread_idx.x] = (
                shared_tile[thread_idx.y, thread_idx.x] + 1
            )
        barrier()

        SharedToGenericTileCopier[thread_layout]().copy(
            global_tile.vectorize[1, simd_width](),
            shared_tile.vectorize[1, simd_width](),
        )

    var ctx = DeviceContext()
    var host_buf = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var dev_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
    for i in range(rows * cols):
        host_buf[i] = Float32(i)
    var tensor = TileTensor(dev_buf, input_layout)
    ctx.enqueue_copy(dev_buf, host_buf)
    ctx.enqueue_function[kernel](
        tensor,
        grid_dim=(num_row_blocks, num_col_blocks),
        block_dim=(block_size, block_size),
    )
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()
    for i in range(rows * cols):
        if host_buf[i] != Float32(i + 1):
            raise Error(
                String("Unexpected value ", host_buf[i], " at position ", i)
            )


def main() raises:
    if has_accelerator():
        tile_copier_example()
    else:
        print("No accelerator, skipping examples that require a GPU.")
