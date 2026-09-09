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

from std.sys import size_of

from max.gpu import thread_idx, block_dim, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_reduce_global_shared_cta,
    cp_async_bulk_shared_cluster_global,
    fence_async_view_proxy,
    fence_mbarrier_init,
    ReduceOp,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from layout.tma_async import SharedMemBarrier


# ---------------------------------------------------------------------------
# Test: global -> shared::cta  (mbarrier completion)
# ---------------------------------------------------------------------------


def kernel_bulk_g2s[
    NUM_ELEMS: Int
](
    src: ImmPointer[Float32, ImmutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    comptime BYTES = NUM_ELEMS * size_of[Float32]()

    var smem = unsafe_stack_allocation[
        NUM_ELEMS, Float32, address_space=.SHARED
    ]()
    var mbar = unsafe_stack_allocation[
        1, SharedMemBarrier, address_space=.SHARED
    ]()

    var tid = thread_idx.x

    if warp_id() == 0 and elect_one_sync():
        mbar[].init()
        fence_mbarrier_init()

        mbar[].expect_bytes(Int32(BYTES))

        cp_async_bulk_shared_cluster_global(
            smem, src, Int32(BYTES), mbar[].unsafe_ptr()
        )

        mbar[].wait()

    barrier()
    dst[tid] = smem[tid]


def test_bulk_g2s[NUM_ELEMS: Int](ctx: DeviceContext) raises:
    var in_host = alloc[Float32](NUM_ELEMS)
    var out_host = alloc[Float32](NUM_ELEMS)

    for i in range(NUM_ELEMS):
        in_host[i] = Float32(i + 1)
        out_host[i] = 0

    var in_dev = ctx.enqueue_create_buffer[.float32](NUM_ELEMS)
    var out_dev = ctx.enqueue_create_buffer[.float32](NUM_ELEMS)

    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(out_dev, out_host)

    ctx.enqueue_function[kernel_bulk_g2s[NUM_ELEMS]](
        in_dev,
        out_dev,
        grid_dim=(1,),
        block_dim=(NUM_ELEMS,),
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(NUM_ELEMS):
        assert_equal(out_host[i], Float32(i + 1))

    in_host.free()
    out_host.free()


# ---------------------------------------------------------------------------
# Test: shared::cta -> global  (bulk_group completion)
# ---------------------------------------------------------------------------


def kernel_bulk_s2g[
    NUM_ELEMS: Int
](dst: MutPointer[Float32, MutAnyOrigin],):
    comptime BYTES = NUM_ELEMS * size_of[Float32]()

    var smem = unsafe_stack_allocation[
        NUM_ELEMS, Float32, address_space=.SHARED
    ]()

    var tid = thread_idx.x
    smem[tid] = Float32(tid + 1)
    barrier()

    if warp_id() == 0 and elect_one_sync():
        fence_async_view_proxy()
        cp_async_bulk_global_shared_cta(dst, smem, Int32(BYTES))
        cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()


def test_bulk_s2g[NUM_ELEMS: Int](ctx: DeviceContext) raises:
    var out_host = alloc[Float32](NUM_ELEMS)
    for i in range(NUM_ELEMS):
        out_host[i] = 0

    var out_dev = ctx.enqueue_create_buffer[.float32](NUM_ELEMS)
    ctx.enqueue_copy(out_dev, out_host)

    ctx.enqueue_function[kernel_bulk_s2g[NUM_ELEMS]](
        out_dev,
        grid_dim=(1,),
        block_dim=(NUM_ELEMS,),
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(NUM_ELEMS):
        assert_equal(out_host[i], Float32(i + 1))

    out_host.free()


# ---------------------------------------------------------------------------
# Test: shared::cta -> global reduce-add  (bulk_group completion)
# ---------------------------------------------------------------------------


def kernel_bulk_reduce_s2g[
    NUM_ELEMS: Int
](dst: MutPointer[Float32, MutAnyOrigin],):
    comptime BYTES = NUM_ELEMS * size_of[Float32]()

    var smem = unsafe_stack_allocation[
        NUM_ELEMS, Float32, address_space=.SHARED
    ]()

    var tid = thread_idx.x
    smem[tid] = Float32(tid + 1)
    barrier()

    if warp_id() == 0 and elect_one_sync():
        fence_async_view_proxy()
        cp_async_bulk_reduce_global_shared_cta[
            DType.float32, reduction_kind=ReduceOp.ADD
        ](dst, smem, Int32(BYTES))
        cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()


def test_bulk_reduce_s2g[NUM_ELEMS: Int](ctx: DeviceContext) raises:
    # Fill dst with known values; expect dst[i] = init + (i + 1) after
    # reduce-add.
    comptime init = Float32(100.0)

    var out_host = alloc[Float32](NUM_ELEMS)
    for i in range(NUM_ELEMS):
        out_host[i] = init

    var out_dev = ctx.enqueue_create_buffer[.float32](NUM_ELEMS)
    ctx.enqueue_copy(out_dev, out_host)

    ctx.enqueue_function[kernel_bulk_reduce_s2g[NUM_ELEMS]](
        out_dev,
        grid_dim=(1,),
        block_dim=(NUM_ELEMS,),
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(NUM_ELEMS):
        assert_equal(out_host[i], init + Float32(i + 1))

    out_host.free()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    with DeviceContext() as ctx:
        test_bulk_g2s[16](ctx)
        test_bulk_g2s[32](ctx)
        test_bulk_g2s[64](ctx)
        test_bulk_g2s[128](ctx)
        test_bulk_g2s[256](ctx)

        test_bulk_s2g[16](ctx)
        test_bulk_s2g[32](ctx)
        test_bulk_s2g[64](ctx)
        test_bulk_s2g[128](ctx)
        test_bulk_s2g[256](ctx)

        test_bulk_reduce_s2g[16](ctx)
        test_bulk_reduce_s2g[32](ctx)
        test_bulk_reduce_s2g[64](ctx)
        test_bulk_reduce_s2g[128](ctx)
        test_bulk_reduce_s2g[256](ctx)
