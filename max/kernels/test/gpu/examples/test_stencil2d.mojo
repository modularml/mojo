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

from max.gpu import global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from layout import TileTensor, Coord, Idx, row_major

comptime BLOCK_DIM = 4


def stencil2d(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    arr_size_dev: Int32,
    num_rows_dev: Int32,
    num_cols_dev: Int32,
    coeff0_dev: Int32,
    coeff1_dev: Int32,
    coeff2_dev: Int32,
    coeff3_dev: Int32,
    coeff4_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var arr_size = Int(arr_size_dev)
    var num_rows = Int(num_rows_dev)
    var num_cols = Int(num_cols_dev)
    var coeff0 = Int(coeff0_dev)
    var coeff1 = Int(coeff1_dev)
    var coeff2 = Int(coeff2_dev)
    var coeff3 = Int(coeff3_dev)
    var coeff4 = Int(coeff4_dev)
    var tidx = global_idx.x
    var tidy = global_idx.y

    var a = TileTensor(a_ptr, row_major(Coord(Int(arr_size))))
    var b = TileTensor(b_ptr, row_major(Coord(Int(arr_size))))

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        var idx = tidy * num_cols + tidx
        b.store(
            Coord(idx),
            Float32(coeff0) * a.load[width=1](Coord(idx - 1))
            + Float32(coeff1) * a.load[width=1](Coord(idx))
            + Float32(coeff2) * a.load[width=1](Coord(idx + 1))
            + Float32(coeff3)
            * a.load[width=1](Coord((tidy - 1) * num_cols + tidx))
            + Float32(coeff4)
            * a.load[width=1](Coord((tidy + 1) * num_cols + tidx)),
        )


def stencil2d_smem(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    arr_size_dev: Int32,
    num_rows_dev: Int32,
    num_cols_dev: Int32,
    coeff0_dev: Int32,
    coeff1_dev: Int32,
    coeff2_dev: Int32,
    coeff3_dev: Int32,
    coeff4_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var arr_size = Int(arr_size_dev)
    var num_rows = Int(num_rows_dev)
    var num_cols = Int(num_cols_dev)
    var coeff0 = Int(coeff0_dev)
    var coeff1 = Int(coeff1_dev)
    var coeff2 = Int(coeff2_dev)
    var coeff3 = Int(coeff3_dev)
    var coeff4 = Int(coeff4_dev)
    var tidx = global_idx.x
    var tidy = global_idx.y
    var lindex_x = thread_idx.x + 1
    var lindex_y = thread_idx.y + 1

    var a = TileTensor(a_ptr, row_major(Coord(Int(arr_size))))
    var b = TileTensor(b_ptr, row_major(Coord(Int(arr_size))))

    var a_shared_ptr = unsafe_stack_allocation[
        (BLOCK_DIM + 2) * (BLOCK_DIM + 2),
        DType.float32,
        address_space=.SHARED,
    ]()
    var a_shared = TileTensor(
        a_shared_ptr, row_major[BLOCK_DIM + 2, BLOCK_DIM + 2]()
    )

    # Each element is loaded in shared memory.
    a_shared.store(
        Coord(lindex_y, lindex_x),
        a.load[width=1](Coord(tidy * num_cols + tidx)),
    )

    # First column also loads elements left and right to the block.
    if thread_idx.x == 0:
        var idx = tidy * num_cols + (tidx - 1)
        a_shared.store(
            Coord(lindex_y, Idx[0]),
            a.load[width=1](Coord(idx)) if 0 <= idx < arr_size else 0,
        )

        idx = tidy * num_cols + tidx + BLOCK_DIM
        a_shared.store(
            Coord(lindex_y, BLOCK_DIM + 1),
            a.load[width=1](Coord(idx)) if 0 <= idx < arr_size else 0,
        )

    # First row also loads elements above and below the block.
    if thread_idx.y == 0:
        var idx = (tidy - 1) * num_cols + tidx
        a_shared.store(
            Coord(Idx[0], lindex_x),
            a.load[width=1](Coord(idx)) if 0 < idx < arr_size else 0,
        )

        idx = (tidy + BLOCK_DIM) * num_cols + tidx
        a_shared.store(
            Coord(BLOCK_DIM + 1, lindex_x),
            a.load[width=1](Coord(idx)) if 0 <= idx < arr_size else 0,
        )

    barrier()

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        b.store(
            Coord(tidy * num_cols + tidx),
            Float32(coeff0)
            * a_shared.load[width=1](Coord(lindex_y, lindex_x - 1))
            + Float32(coeff1)
            * a_shared.load[width=1](Coord(lindex_y, lindex_x))
            + Float32(coeff2)
            * a_shared.load[width=1](Coord(lindex_y, lindex_x + 1))
            + Float32(coeff3)
            * a_shared.load[width=1](Coord(lindex_y - 1, lindex_x))
            + Float32(coeff4)
            * a_shared.load[width=1](Coord(lindex_y + 1, lindex_x)),
        )


# CHECK-LABEL: run_stencil2d
def run_stencil2d[smem: Bool](ctx: DeviceContext) raises:
    print("== run_stencil2d")

    comptime m = 64
    comptime coeff0 = 3
    comptime coeff1 = 2
    comptime coeff2 = 4
    comptime coeff3 = 1
    comptime coeff4 = 5
    comptime iterations = 4

    comptime num_rows = 8
    comptime num_cols = 8

    var a_host = alloc[Float32](m)
    var b_host = alloc[Float32](m)

    for i in range(m):
        a_host[i] = Float32(i)
        b_host[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float32](m)
    var b_device = ctx.enqueue_create_buffer[.float32](m)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime func_select = stencil2d_smem if smem == True else stencil2d

    for _ in range(iterations):
        ctx.enqueue_function[func_select](
            a_device,
            b_device,
            Int32(m),
            Int32(num_rows),
            Int32(num_cols),
            Int32(coeff0),
            Int32(coeff1),
            Int32(coeff2),
            Int32(coeff3),
            Int32(coeff4),
            grid_dim=(
                ceildiv(num_rows, BLOCK_DIM),
                ceildiv(num_cols, BLOCK_DIM),
            ),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

        var tmp_ptr = b_device
        b_device = a_device
        a_device = tmp_ptr

    ctx.enqueue_copy(b_host, b_device)
    ctx.synchronize()

    # CHECK: 37729.0 ,52628.0 ,57021.0 ,60037.0 ,58925.0 ,39597.0 ,
    # CHECK: 57888.0 ,80505.0 ,86322.0 ,89682.0 ,86994.0 ,57818.0 ,
    # CHECK: 76680.0 ,106488.0 ,113400.0 ,116775.0 ,112182.0 ,73933.0 ,
    # CHECK: 95424.0 ,132408.0 ,140400.0 ,143775.0 ,137262.0 ,89925.0 ,
    # CHECK: 91968.0 ,135753.0 ,144450.0 ,147450.0 ,138642.0 ,81842.0 ,
    # CHECK: 50277.0 ,73628.0 ,81985.0 ,83565.0 ,71417.0 ,43229.0 ,
    for i in range(1, num_rows - 1):
        for j in range(1, num_cols - 1):
            print(b_host[i * num_cols + j], ",", end="")
        print()


def main() raises:
    with DeviceContext() as ctx:
        run_stencil2d[False](ctx)
        run_stencil2d[True](ctx)
