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

# DOC: max/layout/tensors.mdx

from max.gpu import (
    thread_idx,
    block_idx,
    global_idx,
    lane_id,
    WARP_SIZE,
)
from max.gpu.sync import barrier
from max.gpu.memory import async_copy_wait_all
from max.gpu.host import DeviceContext, DeviceBuffer, get_gpu_target
from layout import Layout, LayoutTensor, print_layout
from layout.layout_tensor import copy_sram_to_local
from std.memory import Pointer
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


# start-initialize-tensor-from-cpu-example
def initialize_tensor_from_cpu_example() raises:
    comptime dtype = DType.float32
    comptime rows = 32
    comptime cols = 8
    comptime block_size = 8
    comptime row_blocks = rows // block_size
    comptime col_blocks = cols // block_size
    comptime input_layout = Layout.row_major(rows, cols)
    comptime size: Int = rows * cols

    def kernel(tensor: LayoutTensor[dtype, input_layout, MutAnyOrigin]):
        if (
            global_idx.y < tensor.shape[0]()
            and global_idx.x < tensor.shape[1]()
        ):
            tensor[global_idx.y, global_idx.x] = (
                tensor[global_idx.y, global_idx.x] + 1
            )

    try:
        var ctx = DeviceContext()
        var host_buf = ctx.enqueue_create_host_buffer[dtype](size)
        var dev_buf = ctx.enqueue_create_buffer[dtype](size)
        ctx.synchronize()

        var expected_values = List[Scalar[dtype]](length=size, fill=0)

        for i in range(size):
            host_buf[i] = Scalar[dtype](i)
            expected_values[i] = Scalar[dtype](i + 1)
        ctx.enqueue_copy(dev_buf, host_buf)
        var tensor = LayoutTensor[dtype, input_layout](dev_buf)

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
    except error:
        print(error)


# end-initialize-tensor-from-cpu-example


def shared_memory_alloc_example() raises:
    comptime dtype = DType.float32
    comptime in_size = 128
    comptime block_size = 16
    comptime num_blocks = in_size // block_size  # number of block in one dimension
    comptime input_layout = Layout.row_major(in_size, in_size)

    def kernel(tensor: LayoutTensor[dtype, input_layout, MutAnyOrigin]):
        # extract a tile from the input tensor.
        var global_tile = tensor.tile[block_size, block_size](
            block_idx.y, block_idx.x
        )
        # start-shared-memory-alloc-example
        comptime tile_layout = Layout.row_major(block_size, block_size)
        var shared_tile = LayoutTensor[
            dtype,
            tile_layout,
            MutAnyOrigin,
            address_space=.SHARED,
        ].stack_allocation()
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
        var tensor = LayoutTensor[dtype, input_layout](dev_buf)

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
    except error:
        print(error)


def simd_width_example():
    # start-simd-width-example
    from std.sys.info import simd_width_of
    from max.gpu.host.compile import get_gpu_target

    comptime simd_width = simd_width_of[DType.float32, get_gpu_target()]
    # end-simd-width-example


def layout_tensor_vectorized_example() raises:
    comptime dtype = DType.int32
    comptime vector_width = 4

    comptime rows = 64
    comptime columns = 64
    comptime layout = Layout.row_major(rows, columns)
    var storage = Array[Scalar[dtype], rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Int32(i)
    var tensor = LayoutTensor[dtype, layout](storage)
    # start-vectorize-tensor-example
    var vectorized_tensor = tensor.vectorize[1, vector_width]()
    # end-vectorize-tensor-example
    var values = vectorized_tensor[0, 0]
    # The SIMD width could be anywhere from 4 to 16 (possibly  more in the future)
    # So just test a single value.
    assert_equal(
        rebind[SIMD[dtype, vector_width]](values)[3], SIMD[dtype, 1](3)
    )


def layout_tensor_distribute_example():
    comptime rows = 4
    comptime columns = 8
    comptime layout = Layout.row_major(rows, columns)
    comptime dtype = DType.int32

    def kernel(tensor: LayoutTensor[dtype, layout, MutAnyOrigin]):
        var fragment = tensor.vectorize[1, 4]().distribute[
            Layout.row_major(2, 2)
        ](lane_id())
        _ = fragment

    try:
        var ctx = DeviceContext()
        var dev_buf = ctx.enqueue_create_buffer[.int32](rows * columns)
        var host_buf = ctx.enqueue_create_host_buffer[.int32](rows * columns)
        for i in range(rows * columns):
            host_buf[i] = Int32(i)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
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
    comptime input_layout = Layout.row_major(rows, cols)

    def kernel(tensor: LayoutTensor[dtype, input_layout, MutAnyOrigin]):
        # extract a tile from the input tensor.
        var global_tile = tensor.tile[block_size, block_size](
            block_idx.y, block_idx.x
        )
        comptime tile_layout = Layout.row_major(block_size, block_size)
        var shared_tile = LayoutTensor[
            dtype,
            tile_layout,
            MutAnyOrigin,
            address_space=.SHARED,
        ].stack_allocation()

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
        var tensor = LayoutTensor[dtype, input_layout](dev_buf)

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


# TODO: improve thread layout example and explanations
# start-copy-from-async-example
def copy_from_async_example():
    comptime if not has_apple_gpu_accelerator():
        comptime dtype = DType.float32
        comptime rows = 128
        comptime cols = 128
        comptime block_size = 16
        comptime num_row_blocks = rows // block_size
        comptime num_col_blocks = cols // block_size
        comptime input_layout = Layout.row_major(rows, cols)
        comptime simd_width = 4

        def kernel(tensor: LayoutTensor[dtype, input_layout, MutAnyOrigin]):
            # extract a tile from the input tensor.
            var global_tile = tensor.tile[block_size, block_size](
                block_idx.y, block_idx.x
            )
            comptime tile_layout = Layout.row_major(block_size, block_size)
            var shared_tile = LayoutTensor[
                dtype,
                tile_layout,
                MutAnyOrigin,
                address_space=.SHARED,
            ].stack_allocation()

            # Create thread layouts for copying
            comptime thread_layout = Layout.row_major(
                WARP_SIZE // simd_width, simd_width
            )
            var global_fragment = global_tile.vectorize[
                1, simd_width
            ]().distribute[thread_layout](lane_id())
            var shared_fragment = shared_tile.vectorize[
                1, simd_width
            ]().distribute[thread_layout](lane_id())

            shared_fragment.copy_from_async(global_fragment)

            comptime if is_nvidia_gpu():
                async_copy_wait_all()
            barrier()

            # Put some data into the shared tile that we can verify on the host.
            if global_idx.y < rows and global_idx.x < cols:
                shared_tile[thread_idx.y, thread_idx.x] = (
                    shared_tile[thread_idx.y, thread_idx.x] + 1
                )
            barrier()
            global_fragment.copy_from(shared_fragment)

        try:
            var ctx = DeviceContext()
            var host_buf = ctx.enqueue_create_host_buffer[dtype](rows * cols)
            var dev_buf = ctx.enqueue_create_buffer[dtype](rows * cols)
            for i in range(rows * cols):
                host_buf[i] = Float32(i)
            var tensor = LayoutTensor[dtype, input_layout](dev_buf)
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
                        String(
                            "Unexpected value ", host_buf[i], " at position ", i
                        )
                    )
        except error:
            print(error)


# end-copy-from-async-example

# TODO: Currently doesn't run on Apple silicon GPU


def main() raises:
    if has_accelerator():
        initialize_tensor_from_cpu_example()
        shared_memory_alloc_example()
        layout_tensor_vectorized_example()
        layout_tensor_distribute_example()
        simple_copy_example()
        copy_from_async_example()

    else:
        print("No accelerator, skipping examples that require a GPU.")
