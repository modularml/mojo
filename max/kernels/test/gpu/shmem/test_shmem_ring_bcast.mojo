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
# REQUIRES: NVIDIA-GPU
# RUN: %mojo %s

from std.ffi import c_int, c_size_t

from max.gpu.host import DeviceBuffer
from shmem import *
from std.testing import assert_equal


def ring_bcast(
    data: Pointer[c_int, MutAnyOrigin],
    nelem_dev: Int32,
    root: c_int,
    psync: Pointer[UInt64, MutAnyOrigin],
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var nelem = Int(nelem_dev)
    var mype = shmem_my_pe()
    var npes = shmem_n_pes()
    var peer = (mype + 1) % npes

    if mype == root:
        psync[0] = 1

    shmem_signal_wait_until(psync, SHMEM_CMP_NE, 0)

    if mype == npes - 1:
        return

    shmem_put(data, data, c_size_t(nelem), peer)
    shmem_fence()
    shmem_signal_op(psync, 1, SHMEM_SIGNAL_SET, peer)

    psync[0] = 0


def test_ring_bcast(ctx: SHMEMContext) raises:
    comptime data_len = 32
    var destination = ctx.enqueue_create_buffer[.int32](1)

    var data = ctx.enqueue_create_buffer[.int32](data_len)
    var data_h = alloc[Int32](data_len)
    var psync = shmem_calloc[.uint64](1)

    for i in range(data_len):
        data_h[i] = shmem_my_pe() + Int32(i)

    data.enqueue_copy_from(data_h)

    var root: Int32 = 0
    ctx.barrier_all()
    ctx.enqueue_function_collective_checked[ring_bcast](
        data,
        Int32(data_len),
        root,
        DeviceBuffer[.uint64](ctx._ctx, psync, 1, owning=False),
        grid_dim=1,
        block_dim=1,
    )
    ctx.barrier_all()

    data.enqueue_copy_to(data_h)
    ctx.synchronize()

    var mype = shmem_my_pe()
    for i in range(data_len):
        assert_equal(
            data_h[i],
            Int32(i),
            String(
                "PE",
                mype,
                "error, data[",
                i,
                "] = ",
                data_h[i],
                " expected ",
                i,
            ),
        )


def main() raises:
    shmem_launch[test_ring_bcast]()
