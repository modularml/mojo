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

from std.math import align_down, ceildiv

from std.algorithm.functional import tile_and_unswitch
from max.gpu import global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, Coord, Idx, row_major
from std.memory import unsafe_stack_allocation
from std.testing import assert_false


# Tile size for tiling in shared memory.
# Thread block would have shape (tile_size, tile_size, 1)
comptime tile_size = 32


def matmul_sram(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
):
    """Matrix Multiplication using shared memory.
    This version loads blocks of size tile_size x tile_size from A and B
    and updates a tile_size x tile_size in C.

    The thread block should have shape (tile_size, tile_size, 1). Each
    thread is mapped one element in C. The grid should have shape
    (N/tile_size, M/tile_size, 1). N is the first dimension for coalesced
    access.
    """

    # `Int` is not device-passable; widen the fixed-width args.
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)
    var a = TileTensor(a_ptr, row_major(Coord(M, K)))
    var b = TileTensor(b_ptr, row_major(Coord(K, N)))
    var c = TileTensor(c_ptr, row_major(Coord(M, N)))

    # Allocate A, B tile in shared memory.
    var a_shared = unsafe_stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space=.SHARED,
    ]()
    var b_shared = unsafe_stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space=.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    var col = global_idx.x
    var row = global_idx.y

    # Local index in the c sub-matrix updated by current block.
    var localCol = thread_idx.x
    var localRow = thread_idx.y

    # Result of current thread in C.
    var result = Float32(0.0)

    var K_roundbytile = align_down(K, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    var K_remainder = K - K_roundbytile if K - K_roundbytile > 0 else 1

    @always_inline
    def update_tile[
        full_tile: Bool
    ](offset: Int, end: Int, tile_size: Int) {
        var localCol,
        var a,
        var row,
        var a_shared,
        var localRow,
        var col,
        var b,
        var b_shared,
        mut result,
        imm,
    }:
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: Float32

        comptime if not full_tile:
            a_val = a.load[width=1](Coord(row, offset + localCol)) if (
                row < M and offset + localCol < K
            ) else 0.0
        else:
            a_val = (
                a.load[width=1](Coord(row, offset + localCol)) if row
                < M else 0.0
            )
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Float32

        comptime if not full_tile:
            b_val = b.load[width=1](Coord(offset + localRow, col)) if (
                col < N and offset + localRow < K
            ) else 0.0
        else:
            b_val = (
                b.load[width=1](Coord(offset + localRow, col)) if col
                < N else 0.0
            )
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for k in range(tile_size):
            result += a_shared.load(localRow * tile_size + k) * b_shared.load(
                k * tile_size + localCol
            )

        barrier()

    tile_and_unswitch(
        0, K, tile_size, K_remainder, workgroup_function=update_tile
    )

    if row < M and col < N:
        c.store(Coord(row, col), result)


def run_matmul(ctx: DeviceContext) raises:
    print("== run_matmul_sram")

    # Should be able to handle non-divisible values.
    comptime M = 513
    comptime N = 502
    comptime K = 511

    var a_host_ptr = alloc[Float32](M * K)
    var a_host = TileTensor(a_host_ptr, row_major[M, K]())
    var b_host_ptr = alloc[Float32](K * N)
    var b_host = TileTensor(b_host_ptr, row_major[K, N]())
    var c_host_ptr = alloc[Float32](M * N)
    var c_host = TileTensor(c_host_ptr, row_major[M, N]())

    _ = a_host.fill(Float32(1))
    _ = b_host.fill(Float32(1))
    _ = c_host.fill(Float32(0))

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_function[matmul_sram](
        a_device,
        b_device,
        c_device,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(N, tile_size), ceildiv(M, tile_size)),
        block_dim=(tile_size, tile_size),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    var failed = False
    for i in range(M - 10, M):
        for j in range(N - 10, N):
            var val = c_host.load[width=1](Coord(i, j))
            if val != Float32(K):
                print(
                    "Fail at index = [",
                    i,
                    ",",
                    j,
                    "] the value is",
                    val,
                    "the golden value is",
                    K,
                )
                failed = True

    assert_false(failed)
    if not failed:
        print("succeed")


def main() raises:
    with DeviceContext() as ctx:
        run_matmul(ctx)
