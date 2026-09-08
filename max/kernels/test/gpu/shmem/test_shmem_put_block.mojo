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
# RUN: %mojo %s
from max.gpu import block_dim, block_idx, global_idx
from std.memory import Pointer, alloc
from std.ffi import c_size_t
from shmem import *
from std.testing import assert_equal


def set_and_shift_kernel(
    send_data: Pointer[Float32, MutAnyOrigin],
    recv_data: Pointer[Float32, MutAnyOrigin],
    num_elems: Int,
    mype: Int32,
    npes: Int32,
    use_nbi: Int,
):
    var thread_idx = global_idx.x

    # set the corresponding element of send_data
    if thread_idx < num_elems:
        send_data[thread_idx] = Float32(mype)

    var peer = (mype + 1) % npes
    var block_offset = block_idx.x * block_dim.x

    # Every thread in block 0 calls shmem_put_block. shmem_p over IB will use
    # one RMA message for every element, and it cannot leverage multiple threads
    # to copy the data to the destination GPU.

    if use_nbi == 1:
        shmem_put_nbi[SHMEMScope.block](
            recv_data + block_offset,
            send_data + block_offset,
            c_size_t(min(block_dim.x, num_elems - block_offset)),
            peer,
        )
    else:
        shmem_put[SHMEMScope.block](
            recv_data + block_offset,
            send_data + block_offset,
            c_size_t(min(block_dim.x, num_elems - block_offset)),
            peer,
        )


def test_shmem_put[use_nbi: Bool](ctx: SHMEMContext) raises:
    comptime num_elems: Int = 8192
    comptime threads_per_block: Int = 256
    assert (
        num_elems % threads_per_block == 0
    ), "num_elems must be divisible by threads_per_block"
    comptime num_blocks = num_elems // threads_per_block

    var mype = shmem_my_pe()
    var npes = shmem_n_pes()

    var send_data = ctx.enqueue_create_buffer[.float32](num_elems)
    var recv_data = ctx.enqueue_create_buffer[.float32](num_elems)

    ctx.barrier_all()

    ctx.enqueue_function[set_and_shift_kernel](
        send_data,
        recv_data,
        num_elems,
        mype,
        npes,
        Int(use_nbi),
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    var host = alloc[Float32](num_elems)
    recv_data.enqueue_copy_to(host)

    # The completion of the non-blocking version of `shmem_put` is
    # guaranteed by the `nvshmem_barrier_all_on_stream` call.
    ctx.barrier_all()
    ctx.synchronize()

    # Verify the result
    var expected = Float32((mype - 1 + npes) % npes)

    for i in range(num_elems):
        assert_equal(
            host[i],
            expected,
            String(t"unexpected value on PE: {mype} at idx: {i}"),
        )

    print("[", mype, "of", npes, "] run complete. use_nbi=", use_nbi)


def main() raises:
    def test_both(ctx: SHMEMContext) raises:
        test_shmem_put[False](ctx)
        # Test the non-blocking version of `shmem_put` primitive, which returns
        # after initiating the operation.
        test_shmem_put[True](ctx)

    # FIXME SERVOPT-873: This test times out on CI on H100
    # shmem_launch[test_both]()
