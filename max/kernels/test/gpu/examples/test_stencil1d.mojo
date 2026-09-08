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

comptime BLOCK_DIM = 8


def stencil1d(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    arr_size_dev: Int32,
    coeff0_dev: Int32,
    coeff1_dev: Int32,
    coeff2_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var arr_size = Int(arr_size_dev)
    var coeff0 = Int(coeff0_dev)
    var coeff1 = Int(coeff1_dev)
    var coeff2 = Int(coeff2_dev)
    var tid = global_idx.x

    var a = TileTensor(a_ptr, row_major(Coord(Int(arr_size))))
    var b = TileTensor(b_ptr, row_major(Coord(Int(arr_size))))

    if 0 < tid < arr_size - 1:
        b.store(
            Coord(tid),
            Float32(coeff0) * a.load[width=1](Coord(tid - 1))
            + Float32(coeff1) * a.load[width=1](Coord(tid))
            + Float32(coeff2) * a.load[width=1](Coord(tid + 1)),
        )


def stencil1d_smem(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    arr_size_dev: Int32,
    coeff0_dev: Int32,
    coeff1_dev: Int32,
    coeff2_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var arr_size = Int(arr_size_dev)
    var coeff0 = Int(coeff0_dev)
    var coeff1 = Int(coeff1_dev)
    var coeff2 = Int(coeff2_dev)
    var tid = global_idx.x
    var lindex = thread_idx.x + 1

    var a = TileTensor(a_ptr, row_major(Coord(Int(arr_size))))
    var b = TileTensor(b_ptr, row_major(Coord(Int(arr_size))))

    var a_shared = unsafe_stack_allocation[
        BLOCK_DIM + 2, DType.float32, address_space=.SHARED
    ]()

    a_shared[lindex] = a.load[width=1](Coord(tid))
    if thread_idx.x == 0:
        a_shared[lindex - 1] = (
            a.load[width=1](Coord(tid - 1)) if 0 <= tid - 1 < arr_size else 0
        )

        var idx = tid + Int(BLOCK_DIM)
        a_shared[lindex + BLOCK_DIM] = (
            a.load[width=1](Coord(idx)) if idx < arr_size else 0
        )

    barrier()

    if 0 < tid < arr_size - 1:
        b.store(
            Coord(tid),
            Float32(coeff0) * a_shared[lindex - 1]
            + Float32(coeff1) * a_shared[lindex]
            + Float32(coeff2) * a_shared[lindex + 1],
        )


# CHECK-LABEL: run_stencil1d
def run_stencil1d[smem: Bool](ctx: DeviceContext) raises:
    print("== run_stencil1d")

    comptime m = 64
    comptime coeff0 = 3
    comptime coeff1 = 2
    comptime coeff2 = 4
    comptime iterations = 4

    var a_host = alloc[Float32](m)
    var b_host = alloc[Float32](m)

    for i in range(m):
        a_host[i] = Float32(i)
        b_host[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float32](m)
    var b_device = ctx.enqueue_create_buffer[.float32](m)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime func_select = stencil1d_smem if smem == True else stencil1d

    for _ in range(iterations):
        ctx.enqueue_function[func_select](
            a_device,
            b_device,
            Int32(m),
            Int32(coeff0),
            Int32(coeff1),
            Int32(coeff2),
            grid_dim=(ceildiv(m, BLOCK_DIM)),
            block_dim=(BLOCK_DIM),
        )

        var tmp_ptr = b_device
        b_device = a_device
        a_device = tmp_ptr

    ctx.enqueue_copy(b_host, b_device)

    ctx.synchronize()

    # CHECK: 912.0 ,1692.0 ,2430.0 ,3159.0 ,3888.0 ,4617.0 ,5346.0 ,6075.0 ,
    # CHECK: 6804.0 ,7533.0 ,8262.0 ,8991.0 ,9720.0 ,10449.0 ,11178.0 ,11907.0 ,
    # CHECK: 12636.0 ,13365.0 ,14094.0 ,14823.0 ,15552.0 ,16281.0 ,17010.0 ,
    # CHECK: 17739.0 ,18468.0 ,19197.0 ,19926.0 ,20655.0 ,21384.0 ,22113.0 ,
    # CHECK: 22842.0 ,23571.0 ,24300.0 ,25029.0 ,25758.0 ,26487.0 ,27216.0 ,
    # CHECK: 27945.0 ,28674.0 ,29403.0 ,30132.0 ,30861.0 ,31590.0 ,32319.0 ,
    # CHECK: 33048.0 ,33777.0 ,34506.0 ,35235.0 ,35964.0 ,36693.0 ,37422.0 ,
    # CHECK: 38151.0 ,38880.0 ,39609.0 ,40338.0 ,41067.0 ,41796.0 ,42525.0 ,
    # CHECK: 43254.0 ,43983.0 ,35624.0 ,20665.0 ,
    for i in range(1, m - 1):
        print(b_host[i], ",", end="")
    print()


def main() raises:
    with DeviceContext() as ctx:
        run_stencil1d[False](ctx)
        run_stencil1d[True](ctx)
