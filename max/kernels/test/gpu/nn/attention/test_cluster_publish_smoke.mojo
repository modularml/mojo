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

"""Partial-warp cluster-mbarrier publish spike for SM100 FA4 1Q split-K (M4).

Target hardware family: NVIDIA SM100 (B200) — thread-block clusters, distributed
shared memory (DSMEM), and cluster-scoped mbarriers are a B200 feature.

## What this guards

The M4 in-cluster O combine runs entirely inside `fa4_softmax`, i.e. on the
softmax warps (0-7) only. The other warps (8-15: correction / mma / load / idle)
are off doing unrelated work and there is NO mid-kernel all-warps reconvergence
point — so the cross-CTA *publish* that makes each partition's `(max,sum)` + O
visible to its peers CANNOT be an all-warps `cluster_sync()` (a partial-CTA
`cluster_sync` deadlocks). It must be a **partial-warp cluster mbarrier**: each
CTA's WG0 rows arrive on every peer's barrier via
`mbarrier.arrive.shared::cluster` (wrapped by `SharedMemBarrier.arrive_cluster`),
and all softmax threads wait on the local barrier. Warps 8-15 never touch it and
proceed straight to the terminal `cluster_sync()`.

This test reproduces exactly that structure in isolation, before wiring it into
the dense kernel:

  * 16 warps launched per CTA (matching FA4); only warps 0-7 (tid < 256) publish.
  * each CTA writes a rank-distinct pattern into its shared memory,
  * every WG0 row (tid < BM) arrives on **every** peer's publish mbarrier
    (init count = BM * P, one arrival per (row, CTA)); all softmax threads then
    `wait`,
  * after the wait, the leader reads **every** peer's shared memory via DSMEM
    (`load_cluster_smem`) — this both checks the cross-CTA release/acquire
    ordering the mbarrier must provide and that no partition raced ahead,
  * warps 8-15 skip the publish entirely and fall through,
  * ALL 16 warps reach a terminal `cluster_sync()` (the keep-alive) — a deadlock
    here would mean the partial-warp publish broke warp reconvergence.

A hang ⇒ the publish mechanism deadlocks (wrong arrival count, or partial-warp
mbarrier interacts badly with the terminal `cluster_sync`). A value mismatch ⇒
the mbarrier did not establish the cross-CTA happens-before that the combine
relies on. Passing for `P in {2, 4, 8}` clears the M4 #1 risk.
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext, Dim
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils.static_tuple import StaticTuple

from layout.tma_async import SharedMemBarrier
from max.gpu.primitives.cluster import load_cluster_smem


comptime BASE = 1000
comptime RANK_STRIDE = 100
comptime W = 8  # words per CTA (also the SIMD load width)
comptime SENTINEL = UInt32(0xFFFFFFFF)
comptime NUM_WARPS = 16  # match FA4 (warps 0-7 softmax, 8-15 other)
comptime SOFTMAX_THREADS = 256  # warps 0-7
comptime BM = 128  # WG0 rows that each arrive (the production combine warpgroup)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
def publish_smoke_kernel[
    P: Int, cluster_shape: StaticTuple[Int32, 3]
](output: MutPointer[UInt32, MutAnyOrigin]):
    # Static shared scratch, identically offset in every CTA — `mapa` rebases it
    # onto a peer's window.
    var smem = unsafe_stack_allocation[
        W, DType.uint32, address_space=.SHARED, alignment=16
    ]()
    # The publish mbarrier (one SharedMemBarrier = one 8-byte slot).
    var mbar = unsafe_stack_allocation[
        1, DType.int64, address_space=.SHARED, alignment=8
    ]().bitcast[SharedMemBarrier]()

    var me = block_rank_in_cluster()
    var tid = Int(thread_idx.x)

    # Init the publish barrier to expect `BM * P` arrivals: every WG0 row of
    # every CTA arrives on its own behalf (BM rows × P partitions), mirroring the
    # dense kernel's per-row publish. The cluster_sync below guarantees every CTA
    # finished init before any peer arrives on it (in the dense kernel this
    # ordering is provided by the kernel-prologue mbarrier init + the work before
    # the publish).
    if tid == 0:
        mbar[].init(Int32(BM * P))

    # Each CTA writes its own rank-distinct pattern.
    if tid == 0:
        comptime for i in range(W):
            smem[i] = UInt32(BASE) + me * UInt32(RANK_STRIDE) + UInt32(i)

    cluster_sync()

    # ---- Partial-warp publish: ONLY warps 0-7 (tid < SOFTMAX_THREADS). ----
    if tid < SOFTMAX_THREADS:
        # Per-row publish: every WG0 row (tid < BM) arrives on every peer's
        # barrier (including its own). No CTA-local barrier collects rows first —
        # each row's arrive stands on its own (here trivially, since each row's
        # work preceded the cluster_sync; in the dense kernel it follows the
        # row's own staging write). Barrier count BM * P resolves when all rows
        # of all CTAs have arrived.
        if tid < BM:
            comptime for p in range(P):
                mbar[].arrive_cluster(UInt32(p))
        # All softmax threads wait on the local barrier (phase 0).
        mbar[].wait(UInt32(0))

        # After the publish, the leader reads every peer's smem via DSMEM and
        # dumps to out[me, r, :]. Correct values ⇒ the mbarrier established the
        # cross-CTA happens-before.
        if tid == 0:
            var me_i = Int(me)
            comptime for r in range(P):
                var v = load_cluster_smem[.uint32, W](smem, UInt32(r))
                comptime for w in range(W):
                    output[(me_i * P + r) * W + w] = v[w]

    # ALL warps (0-15) reach the terminal keep-alive sync. Warps 8-15 arrive
    # here without having touched the publish mbarrier; a deadlock would mean
    # the partial-warp publish broke reconvergence.
    cluster_sync()


def run_publish_test[P: Int](ctx: DeviceContext) raises:
    var n = P * P * W
    var out_dev = ctx.enqueue_create_buffer[.uint32](n)

    with out_dev.map_to_host() as h:
        for i in range(n):
            h[i] = SENTINEL

    comptime cluster_shape = StaticTuple[Int32, 3](Int32(P), Int32(1), Int32(1))
    comptime kernel = publish_smoke_kernel[P, cluster_shape]
    ctx.enqueue_function[kernel](
        out_dev,
        grid_dim=(P, 1, 1),
        block_dim=(NUM_WARPS * 32, 1, 1),
        cluster_dim=Dim(P, 1, 1),
    )
    ctx.synchronize()

    with out_dev.map_to_host() as h:
        for s in range(P):
            for r in range(P):
                for w in range(W):
                    var got = h[(s * P + r) * W + w]
                    var want = UInt32(BASE + r * RANK_STRIDE + w)
                    assert_equal(
                        got,
                        want,
                        msg=String(
                            "cluster publish mismatch P=",
                            P,
                            " reader=",
                            s,
                            " peer=",
                            r,
                            " word=",
                            w,
                        ),
                    )

    print("== cluster publish smoke P=", P, " OK")


def main() raises:
    with DeviceContext() as ctx:
        run_publish_test[2](ctx)
        run_publish_test[4](ctx)
        run_publish_test[8](ctx)
