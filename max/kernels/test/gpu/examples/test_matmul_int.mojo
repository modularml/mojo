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

from std.math import ceildiv
from std.math.uutils import udivmod

from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import (
    unsafe_memset_zero,
    unsafe_stack_allocation,
)
from layout import Coord, Idx, TileTensor, row_major

comptime TILE_SZ_A = 128
comptime TILE_SZ_B = 16
comptime TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B


def matmul(
    a_ptr: MutPointer[Int, MutAnyOrigin],
    b_ptr: MutPointer[Int, MutAnyOrigin],
    c_ptr: MutPointer[Int, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var a = TileTensor(a_ptr, row_major(Coord(m, k)))
    var b = TileTensor(b_ptr, row_major(Coord(k, n)))
    var c = TileTensor(c_ptr, row_major(Coord(m, n)))

    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    # Allocate B array into shared memory for tiling.
    var b_shared = unsafe_stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.int,
        address_space=.SHARED,
    ]()

    # Thread indexing offsets.
    var row = global_idx.x
    var col = block_idx.y * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = unsafe_stack_allocation[TILE_SZ_B, DType.int]()

    unsafe_memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i, j = udivmod(thread_idx.x, TILE_SZ_B)

        # Load the B matrix into shared memory.
        var b_val = Int(b[tile_idx * TILE_SZ_RATIO + i, col + j])
        b_shared[i * TILE_SZ_B + j] = Int(b_val)

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Int
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = Int(a[row, tile_idx * TILE_SZ_RATIO + idx])
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg[out_idx] += (
                    Int(a_reg) * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            c[row, col + out_idx] = c_reg.load(out_idx)


def run_matmul(ctx: DeviceContext) raises:
    print("== run_matmul")

    comptime m = 512
    comptime n = 512
    comptime k = 512

    var a_host_ptr = alloc[Int](m * k)
    var b_host_ptr = alloc[Int](k * n)
    var c_host_ptr = alloc[Int](m * n)

    var a_host = TileTensor(a_host_ptr, row_major[m, k]())
    var b_host = TileTensor(b_host_ptr, row_major[k, n]())
    var c_host = TileTensor(c_host_ptr, row_major[m, n]())

    for i in range(m):
        for j in range(k):
            a_host[i, j] = 1

    for i in range(k):
        for j in range(n):
            b_host[i, j] = 1

    for i in range(m):
        for j in range(n):
            c_host[i, j] = 0

    var a_device = ctx.enqueue_create_buffer[.int](m * k)
    var b_device = ctx.enqueue_create_buffer[.int](k * n)
    var c_device = ctx.enqueue_create_buffer[.int](m * n)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_function[matmul](
        a_device,
        b_device,
        c_device,
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    for i in range(10):
        for j in range(10):
            print("at index = [", i, ",", j, "]the value is", c_host[i, j])


def main() raises:
    with DeviceContext() as ctx:
        run_matmul(ctx)
