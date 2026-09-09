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

"""DYNAMIC-cluster-dim DSMEM spike for SM100 FA4 1Q split-K (§6 investigation).

Target hardware family: NVIDIA SM100 (B200) — thread-block clusters and
distributed shared memory (DSMEM) are a B200 feature.

## What this guards — the linchpin for single-kernel runtime-P split-K

`test_dsmem_cluster_smoke.mojo` proved the DSMEM transport works, but it sets
**both** the static `@__llvm_metadata(`nvvm.cluster_dim`=…)` kernel metadata
**and** the launch-time `cluster_dim=` (the production FA4 form). That leaves the
open question (plan §6 "Dynamic cluster dimensions"): can we compile the split-K
1Q kernel **once** — with **NO** static `nvvm.cluster_dim` metadata — and choose
the cluster size `P` purely at launch via `cluster_dim=Dim(P,1,1)`?

If yes, we can make `P` a runtime value (single kernel, runtime cluster) instead
of instantiating one kernel per static `P ∈ {1,2,4,8}`. If no (driver rejects the
launch, or `mapa`/`cluster_sync` misbehave without the static metadata), the
route collapses to per-`P` static instantiation.

This kernel is the runtime-P analogue of `dsmem_smoke_kernel`:
  * **NO** `nvvm.cluster_dim` decorator (the whole point),
  * **NO** `P` type parameter — the partition count arrives as the runtime arg
    `p_count`, and the peer-read loop is a runtime `for r in range(p_count)`,
  * one compiled kernel is launched three times with `cluster_dim=Dim(P,1,1)`
    for `P ∈ {2,4,8}`.

Verification is identical to the static spike: each CTA writes a rank-distinct
pattern, a `cluster_sync()` publishes it, every CTA reads **every** peer's smem
(off-diagonal included) via `load_cluster_smem`, and the host checks the full
`[P,P,W]` matrix against the sentinel-filled buffer.
"""

from max.gpu import thread_idx
from max.gpu.primitives.id import cluster_dim as rt_cluster_dim
from max.gpu.host import DeviceContext, Dim
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from max.gpu.primitives.cluster import load_cluster_smem


comptime BASE = 1000
comptime RANK_STRIDE = 100
comptime W = 8  # words per CTA (also the SIMD load width)
comptime SENTINEL = UInt32(0xFFFFFFFF)


# NOTE: deliberately NO `@__llvm_metadata(`nvvm.cluster_dim`=…)` — the cluster
# size is supplied only at launch (`cluster_dim=`). NO `P` type parameter — the
# count is the runtime arg `p_count`, so a single compiled kernel serves every P.
def dyn_dsmem_kernel(
    output: MutPointer[UInt32, MutAnyOrigin],
    cdim_out: MutPointer[UInt32, MutAnyOrigin],
    p_count: UInt32,
):
    var smem = unsafe_stack_allocation[
        W, DType.uint32, address_space=.SHARED, alignment=16
    ]()

    var me = block_rank_in_cluster()

    # Runtime cluster size read from the hardware special register
    # (`cluster.nctaid.x`) — Stage B uses this as `num_partitions` instead of a
    # comptime config value. Verify it returns the launched `P` even with NO
    # static `nvvm.cluster_dim` metadata (the whole point of the dynamic path).
    if thread_idx.x == 0:
        cdim_out[Int(me)] = UInt32(rt_cluster_dim.x)

    # Phase 1: each CTA writes its own rank-distinct pattern.
    if thread_idx.x == 0:
        comptime for i in range(W):
            smem[i] = UInt32(BASE) + me * UInt32(RANK_STRIDE) + UInt32(i)

    # Publish the writes to the whole cluster (release + acquire).
    cluster_sync()

    # Phase 2: read every peer's shared memory (runtime peer count) and dump to
    # gmem out[me, r, :]. The peer index `r` is a RUNTIME loop bound here.
    if thread_idx.x == 0:
        var me_i = Int(me)
        var p_i = Int(p_count)
        for r in range(p_i):
            var v = load_cluster_smem[.uint32, W](smem, UInt32(r))
            comptime for w in range(W):
                output[(me_i * p_i + r) * W + w] = v[w]

    # All peers must finish reading before any CTA exits and reclaims its smem.
    cluster_sync()


def run_dyn_dsmem_test[P: Int](ctx: DeviceContext) raises:
    var n = P * P * W
    var out_dev = ctx.enqueue_create_buffer[.uint32](n)

    # Sentinel-fill so a missing write is caught (not mistaken for a valid 0).
    with out_dev.map_to_host() as h:
        for i in range(n):
            h[i] = SENTINEL

    var cdim_dev = ctx.enqueue_create_buffer[.uint32](P)
    with cdim_dev.map_to_host() as h:
        for i in range(P):
            h[i] = SENTINEL

    # Single compiled kernel (no P type param); cluster size chosen at launch.
    comptime kernel = dyn_dsmem_kernel
    ctx.enqueue_function[kernel](
        out_dev,
        cdim_dev,
        UInt32(P),
        grid_dim=(P, 1, 1),
        block_dim=(32, 1, 1),
        cluster_dim=Dim(P, 1, 1),
    )
    ctx.synchronize()

    # Every CTA must observe the launched cluster size via the runtime register.
    with cdim_dev.map_to_host() as h:
        for r in range(P):
            assert_equal(
                h[r],
                UInt32(P),
                msg=String(
                    "cluster_dim.x mismatch (no static metadata) P=",
                    P,
                    " rank=",
                    r,
                ),
            )

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
                            "dyn DSMEM peer-read mismatch P=",
                            P,
                            " reader=",
                            s,
                            " peer=",
                            r,
                            " word=",
                            w,
                        ),
                    )

    print("== dyn dsmem cluster smoke (no static metadata) P=", P, " OK")


def main() raises:
    with DeviceContext() as ctx:
        run_dyn_dsmem_test[2](ctx)
        run_dyn_dsmem_test[4](ctx)
        run_dyn_dsmem_test[8](ctx)
