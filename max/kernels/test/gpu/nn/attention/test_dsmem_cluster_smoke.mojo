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

"""DSMEM verification spike for SM100 FA4 1Q split-K (de-risks M3/M4).

Target hardware family: NVIDIA SM100 (B200) — thread-block clusters and
distributed shared memory (DSMEM) are a B200 feature.

## What this guards

M3/M4 combine the per-partition split-K results in-cluster: each CTA reads its
peer partitions' `(max, sum)` (M3) and partial-O (M4) out of their shared memory
after a `cluster_sync()`. That cross-CTA read goes through the `mapa.shared::
cluster` PTX instruction wrapped by `load_cluster_smem` /`store_cluster_smem`
(in `sm100/attention_utils.mojo`). Before wiring those into the reduction math,
this test confirms the transport primitive in isolation:

  * each CTA writes a **rank-distinct** pattern into its own shared memory,
  * a `cluster_sync()` makes those writes visible cluster-wide,
  * every CTA reads **every** peer's shared memory (including off-diagonal
    `reader != peer`) via `load_cluster_smem` and dumps the values to gmem,
  * the host checks `out[s][r][w] == BASE + r*RANK_STRIDE + w` for all `(s,r,w)`.

The pattern is rank-distinct (`RANK_STRIDE >> W`) and the output buffer is
sentinel-filled before launch, so a no-write, a self-read on an off-diagonal
entry, or a wrong-peer mapping all produce a detectable mismatch. Passing
confirms `block_rank_in_cluster()` indexes the cluster exactly as the launch
maps it, and that `mapa` + `ld.shared::cluster` move the right bytes.

Swept over cluster sizes `P in {2, 4, 8}` (the portable split-K range).
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext, Dim
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils.static_tuple import StaticTuple

from max.gpu.primitives.cluster import load_cluster_smem


# Rank-distinct value written by CTA `me` at word `i`: distinct across both rank
# and word because RANK_STRIDE far exceeds the word count W.
comptime BASE = 1000
comptime RANK_STRIDE = 100
comptime W = 8  # words per CTA (also the SIMD load width)
comptime SENTINEL = UInt32(0xFFFFFFFF)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
def dsmem_smoke_kernel[
    P: Int, cluster_shape: StaticTuple[Int32, 3]
](output: MutPointer[UInt32, MutAnyOrigin]):
    # Static shared scratch, identically offset in every CTA — `mapa` rebases it
    # onto a peer's window.
    var smem = unsafe_stack_allocation[
        W, DType.uint32, address_space=.SHARED, alignment=16
    ]()

    var me = block_rank_in_cluster()

    # Phase 1: each CTA writes its own rank-distinct pattern.
    if thread_idx.x == 0:
        comptime for i in range(W):
            smem[i] = UInt32(BASE) + me * UInt32(RANK_STRIDE) + UInt32(i)

    # Publish the writes to the whole cluster (release + acquire).
    cluster_sync()

    # Phase 2: read every peer's shared memory and dump to gmem out[me, r, :].
    if thread_idx.x == 0:
        var me_i = Int(me)
        comptime for r in range(P):
            var v = load_cluster_smem[.uint32, W](smem, UInt32(r))
            comptime for w in range(W):
                output[(me_i * P + r) * W + w] = v[w]

    # All peers must finish reading before any CTA exits and reclaims its smem.
    cluster_sync()


def run_dsmem_test[P: Int](ctx: DeviceContext) raises:
    var n = P * P * W
    var out_dev = ctx.enqueue_create_buffer[.uint32](n)

    # Sentinel-fill so a missing write is caught (not mistaken for a valid 0).
    with out_dev.map_to_host() as h:
        for i in range(n):
            h[i] = SENTINEL

    comptime cluster_shape = StaticTuple[Int32, 3](Int32(P), Int32(1), Int32(1))
    comptime kernel = dsmem_smoke_kernel[P, cluster_shape]
    ctx.enqueue_function[kernel](
        out_dev,
        grid_dim=(P, 1, 1),
        block_dim=(32, 1, 1),
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
                            "DSMEM peer-read mismatch P=",
                            P,
                            " reader=",
                            s,
                            " peer=",
                            r,
                            " word=",
                            w,
                        ),
                    )

    print("== dsmem cluster smoke P=", P, " OK")


def main() raises:
    with DeviceContext() as ctx:
        run_dsmem_test[2](ctx)
        run_dsmem_test[4](ctx)
        run_dsmem_test[8](ctx)
