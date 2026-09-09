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

"""Two-level 8-way intra-CTA LSE combine for the `.ws` MMA_M=32 datapath.

Purpose (Mc / §5c-5e of the SM100 FA4 `.ws` MMA_M=32 two-pipeline design; see
docs/plans/sm100-fa4-ws-mma32-two-pipeline-splitk.md)
=============================================================================
Isolate and verify the genuinely-new compute of the two-pipeline design: the
hierarchical terminal combine that merges the EIGHT `(pipeline, quarter)`
partitions -- two softmax warpgroups (WG0/WG1), each holding four packed-TMEM
datapath quarters over the SAME 32 query rows -- into one normalized output.

The 8 partitions live in 8 DIFFERENT warps: warp-group `w = tid // 128` is the
pipeline, warp `g = (tid % 128) >> 5` is the datapath quarter, lane `r = tid & 31`
is the query row. Partition `p = w*4 + g` supplies its own `(m_p, l_p, O_p)`. The
merge is two levels (identical LSE algebra to the flat 8-way reduction, only the
transport differs -- so a flat 8-way host reference verifies the composition):

    Level 1 (per WG w, over its 4 quarters):
        m_w       = max_g m_{w,g}
        l_w       = Σ_g exp2((m_{w,g} - m_w) * SCALE_LOG2E) * l_{w,g}
        O_w[band] = Σ_g exp2((m_{w,g} - m_w) * SCALE_LOG2E) * O_{w,g}[band]  (unnorm)
    Level 2 (across w in {0,1}):
        M         = max(m_0, m_1)
        s_w       = exp2((m_w - M) * SCALE_LOG2E)
        L         = s_1.fma(l_1, s_0 * l_0)
        O_final   = (s_1.fma(O_1, s_0 * O_0)) * recip(L)

This exercises the three new helpers end to end:
  * fa4_ws_level1_combine              (per-WG 4-way unnormalized partial)
  * fa4_ws_level2_reduce_scatter_write (cross-WG reduce-scatter + normalize)
  * fa4_tma_store_o_smem               (SWIZZLE_NONE TMA egress -> gmem)

Setup: 256 threads (two warpgroups) allocate TMEM; each warp injects its known
`O_p` into its WG's C-TMEM band (WG0 -> TMEM_O0 analog, WG1 -> TMEM_O1 analog)
via `tcgen05_st` (all 4 warps of a WG share the base; the HW subpartition routes
each warp its quarter); Level 1 + Level 2 run; then WG0 TMA-stores the normalized
output to a device buffer, which the host reads back and compares to the flat
8-way reference. There is NO MMA here -- the combine is pure f32.

Every case keeps >= 1 non-empty quarter PER warpgroup: Level 1's cross-quarter
`max` must stay finite (a fully-empty WG -> `exp2(-inf - -inf) = NaN`; the helper
has no all-empty guard, and a fully-empty WG cannot occur in production since at
T>=2 each WG streams >= 1 real key). Cases (each x depth in {64, 128}):
  * uniform        : m identical across all 8 partitions -> every scale = 1
  * divergent      : one dominant partition (m=200) per row, rotated through all
                     8 (others m=0, finite) -> exp2(-200) flushes to 0 at both
                     levels
  * per-wg-single  : one real quarter (q0) per WG; the other 3/WG neutral
                     (m=-inf,l=0,O=0) -> empty-quarter neutral + the >=1-per-WG
                     boundary, no NaN
  * wg1-negligible : all real, but m1 << m0 (150-unit gap) -> s1 = exp2(-150)
                     flushes to 0 (the Level-2 s1=0 path) without an all-empty WG
  * cross-wg-diverge: per-row m0 != m1 -> non-trivial Level-2 rescale (a bug that
                     equal-per-WG-max cases would mask)
"""

from std.math import exp2, recip, isnan
from std.random import randn, seed
from std.sys import size_of

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from std.utils.numerics import min_or_neg_inf
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.tma_async import RaggedTMA3DTile
from nn.attention.gpu.nvidia.sm100.softmax_warp import (
    fa4_ws_level1_combine,
    fa4_ws_level2_reduce_scatter_write,
    fa4_tma_store_o_smem,
)
from std.testing import assert_true

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------
comptime ACC_TYPE = DType.float32
comptime OUT_TYPE = DType.float32  # pure-f32 combine (no bf16 anywhere)

comptime MMA_M = 32
comptime M_PACK = 128 // MMA_M  # 4 datapath quarters per warpgroup
comptime NUM_WG = 2  # two softmax pipelines
comptime NUM_PARTITIONS = M_PACK * NUM_WG  # 8
comptime ROWS = MMA_M  # 32 query rows (shared by all partitions)
comptime DEPTH_TILE = 256 // M_PACK  # 64

comptime NUM_THREADS = 128 * NUM_WG  # 256 = two warpgroups
comptime MAX_TMEM_COLS: UInt32 = 512
comptime C_TMEM_OFFSET_0: UInt32 = 128  # WG0 O band (mirror TMEM_O0)
comptime META_BYTES = 128  # tmem_addr scratch, padded

# SCALE_LOG2E = 1.0 makes the helpers' `* scale_log2e` steps no-ops so the host
# reference is exactly `exp2(m - M)` at both levels (the combine still exercises
# the use_fma multiply path).
comptime SCALE_LOG2E: Float32 = 1.0

comptime ATOL: Float32 = 1e-3
comptime RTOL: Float32 = 1e-3


# The SWIZZLE_NONE per-block (tma_blocks_per_op == 0) O store op: gmem is 3D
# (rows, middle_dim=1, depth) row-major. `create` offsets the descriptor base
# back by `depth*ROWS`, so the device buffer needs a ROWS-row front pad and the
# store lands logical rows [0, ROWS) at physical rows [ROWS, 2*ROWS).
comptime OStoreT[depth: Int] = RaggedTMA3DTile[
    OUT_TYPE,
    TensorMapSwizzle.SWIZZLE_NONE,
    BM=ROWS,
    BN=depth,
    middle_dim=1,
    group=1,
    tma_blocks_per_op=0,
]


# ---------------------------------------------------------------------------
# GPU kernel: inject known O_p into TMEM, run the two-level combine + TMA egress.
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(o_store, `nvvm.grid_constant`)
def combine_kernel[
    depth: Int,
](
    O_in: LayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS * ROWS, depth), MutAnyOrigin
    ],
    m_in: LayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS, ROWS), MutAnyOrigin
    ],
    l_in: LayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS, ROWS), MutAnyOrigin
    ],
    o_store: OStoreT[depth],
):
    comptime num_d_tiles = depth // DEPTH_TILE
    comptime own_cols = depth // M_PACK
    comptime STAGE = M_PACK * ROWS * depth  # per-WG Level-1 raw-O staging
    comptime ML = M_PACK * ROWS * 2  # per-WG Level-1 (m, l)
    comptime L2STAGE = depth * ROWS  # WG1 -> WG0 O_1 staging
    comptime L2MS = ROWS * 2  # WG1's (m_1, l_1)
    comptime OSMEM = ROWS * depth

    # ---- Dynamic SMEM carve ----
    # [stage0 | stage1 | maxsum0 | maxsum1 | l2_stage | l2_maxsum | o_smem | meta]
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var f32_base = smem_base.bitcast[Scalar[ACC_TYPE]]()
    var stage0 = f32_base
    var stage1 = stage0 + STAGE
    var maxsum0 = stage1 + STAGE
    var maxsum1 = maxsum0 + ML
    var l2_stage = maxsum1 + ML
    var l2_maxsum = l2_stage + L2STAGE
    var o_smem = l2_maxsum + L2MS
    var ptr_tmem_addr = (o_smem + OSMEM).bitcast[UInt32]()

    var tid = thread_idx.x
    var wid = Int(tid >> 5)  # 0..7 : global warp
    var wg = Int(tid // 128)  # 0/1 : pipeline (warpgroup)
    var g = Int(
        (tid % 128) >> 5
    )  # 0..3 : datapath quarter / band / WG-local warp
    var row = Int(tid & 31)  # 0..31 : query row
    var p = wg * M_PACK + g  # 0..7 : partition

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]
    # WG0 O band at [128, 128+depth); WG1 at [128+depth, 128+2*depth) (mirror
    # TMEM_O0 / TMEM_O1 = TMEM_O0 + padded_ov_depth).
    var c_tmem = tmem_addr + C_TMEM_OFFSET_0 + UInt32(wg * depth)

    # ---- (A) inject known O_p into this WG's C-TMEM band (shared base per WG) ----
    for t in range(num_d_tiles):
        var o_frag = Array[_, DEPTH_TILE](
            fill_with=lambda (j: Int) -> Scalar[ACC_TYPE]: O_in[
                p * ROWS + row, t * DEPTH_TILE + j
            ][0]
        )
        tcgen05_st[datapaths=32, bits=32, repeat=DEPTH_TILE, pack=False](
            c_tmem + UInt32(t) * UInt32(DEPTH_TILE), o_frag
        )
    tcgen05_store_wait()
    tcgen05_fence_before()
    barrier()
    tcgen05_fence_after()

    # ---- (B) Level 1: per-WG 4-way unnormalized partial (into o_band regs) ----
    var stage_wg = stage0 if wg == 0 else stage1
    var maxsum_wg = maxsum0 if wg == 0 else maxsum1
    var o_band = Array[Scalar[ACC_TYPE], own_cols](fill={})
    var m_wg, l_wg = fa4_ws_level1_combine[M_PACK, ROWS, depth, use_fma=True](
        UInt32(row),
        UInt32(g),
        UInt32(wg),
        m_in[p, row][0],
        l_in[p, row][0],
        SCALE_LOG2E,
        c_tmem,
        stage_wg.as_unsafe_any_origin(),
        maxsum_wg.as_unsafe_any_origin(),
        o_band,
    )

    # ---- (C) Level 2: cross-WG reduce-scatter + normalize -> o_smem (WG0) ----
    _ = fa4_ws_level2_reduce_scatter_write[M_PACK, ROWS, depth, use_fma=True](
        UInt32(row),
        UInt32(g),
        UInt32(wg),
        m_wg,
        l_wg,
        SCALE_LOG2E,
        o_band,
        l2_stage.as_unsafe_any_origin(),
        l2_maxsum.as_unsafe_any_origin(),
        o_smem.as_unsafe_any_origin(),
    )

    # ---- (D) TMA egress: WG0 stores the normalized o_smem to gmem ----
    if wg == 0:
        fa4_tma_store_o_smem[depth, TensorMapSwizzle.SWIZZLE_NONE, ROWS, 1, 1](
            UInt32(g),  # local_warp_idx (0..3); warp 0 issues
            UInt32(wg),  # warp_group_idx (0)
            o_smem.as_unsafe_any_origin(),
            o_store,
            Int32(ROWS),  # num_output_rows
            UInt32(0),  # out_head_idx
            UInt32(0),  # out_row_idx
        )

    # Terminal rendezvous on id 4 (NOT block `barrier()`: that reuses barrier id
    # 0, which WG0's egress/Level-1 named_barrier[128](0) also use -- a 256- vs
    # 128-count clash. Matches the real kernel's terminal named_barrier(4).)
    named_barrier[Int32(NUM_THREADS)](Int32(4))
    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def test_combine[depth: Int, mode: Int](ctx: DeviceContext) raises:
    comptime STAGE = M_PACK * ROWS * depth
    comptime ML = M_PACK * ROWS * 2
    comptime L2STAGE = depth * ROWS
    comptime L2MS = ROWS * 2
    comptime OSMEM = ROWS * depth
    comptime total_smem = (
        2 * STAGE + 2 * ML + L2STAGE + L2MS + OSMEM
    ) * size_of[ACC_TYPE]() + META_BYTES

    var case_name = "uniform" if mode == 0 else (
        "divergent" if mode
        == 1 else (
            "per-wg-single" if mode
            == 2 else ("wg1-negligible" if mode == 3 else "cross-wg-diverge")
        )
    )
    print("=" * 70)
    print(
        "test_ws_two_level_combine: 8-way depth="
        + String(depth)
        + " case="
        + case_name
    )
    print("=" * 70)

    seed(42)

    var O_in = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS * ROWS, depth)
    ](ctx)
    var m_in = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS, ROWS)
    ](ctx)
    var l_in = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NUM_PARTITIONS, ROWS)
    ](ctx)
    # Output buffer with a ROWS-row front pad (see OStoreT): logical rows
    # [0, ROWS) land at physical rows [ROWS, 2*ROWS).
    var o_dev = ManagedLayoutTensor[
        OUT_TYPE, Layout.row_major(2 * ROWS, depth)
    ](ctx)

    var O_host = O_in.tensor[update=False]()
    var m_host = m_in.tensor[update=False]()
    var l_host = l_in.tensor[update=False]()

    # O ~ N(0,1); l = |N(0,1)| + 1 (strictly positive, it is a sum of exp).
    randn[ACC_TYPE](O_host.ptr, NUM_PARTITIONS * ROWS * depth)
    randn[ACC_TYPE](l_host.ptr, NUM_PARTITIONS * ROWS)
    for i in range(NUM_PARTITIONS * ROWS):
        l_host.ptr[i] = abs(l_host.ptr[i]) + 1.0

    # Per-case m (and neutral l/O for empty quarters). INVARIANT: every case
    # keeps >= 1 non-empty quarter PER warpgroup -- Level 1's cross-quarter
    # `m_wg = max_g m_g` must stay finite (an all-empty WG gives m_wg = -inf and
    # `exp2(m_g - m_wg) = exp2(-inf - -inf) = NaN`; the helper has no all-empty
    # guard, and an all-empty WG cannot occur in production: at T>=2 each WG
    # streams >= 1 real key). Empty *quarters* alongside a real one are fine
    # (`exp2(-inf - finite) = 0`).
    for p in range(NUM_PARTITIONS):
        var wg_ = p // M_PACK  # 0 / 1
        var q = p % M_PACK  # quarter within WG
        for r in range(ROWS):
            if mode == 0:  # uniform: identical across all partitions
                m_host.ptr[p * ROWS + r] = 0.5 * Float32(r)
            elif mode == 1:  # divergent: one dominant partition per row
                var dom = r % NUM_PARTITIONS
                m_host.ptr[p * ROWS + r] = 200.0 if p == dom else 0.0
            elif mode == 2:  # one real quarter PER WG (q0); other 3/WG neutral
                if q == 0:
                    m_host.ptr[p * ROWS + r] = 1.0
                else:
                    m_host.ptr[p * ROWS + r] = min_or_neg_inf[ACC_TYPE]()
                    l_host.ptr[p * ROWS + r] = 0.0
                    for d in range(depth):
                        O_host.ptr[(p * ROWS + r) * depth + d] = 0.0
            elif mode == 3:  # WG1 negligible: real, but m1 << m0 -> s1 -> 0
                # 150-unit cross-WG gap flushes s1 = exp2(-150) to 0 (the
                # Level-2 s1=0 path) WITHOUT an all-empty WG.
                var boost: Float32 = 150.0 if wg_ == 0 else 0.0
                m_host.ptr[p * ROWS + r] = 0.5 * Float32(r) + boost
            else:  # mode 4: cross-WG divergence (per-row m0 != m1)
                # WG1 dominant on even rows, WG0 on odd -> both s0<1 and s1<1.
                var offset: Float32 = 2.0 if (r % 2 == 0) else -2.0
                var extra: Float32 = 0.0 if wg_ == 0 else offset
                m_host.ptr[p * ROWS + r] = 0.5 * Float32(r) + extra

    # Initialize the output buffer to a sentinel and push it to device (this
    # also yields the persistent device pointer for the TMA descriptor). If the
    # store lands in the wrong region the sentinel survives and the test fails.
    var SENTINEL: Float32 = -1.0e30
    var o_dev_host = o_dev.tensor[update=False]()
    for i in range(2 * ROWS * depth):
        o_dev_host.ptr[i] = SENTINEL
    var dev_ptr = o_dev.device_tensor().ptr  # update=True: sentinel -> device
    var o_store = OStoreT[depth].create(ctx, dev_ptr + ROWS * depth, rows=ROWS)

    comptime kernel = combine_kernel[depth]
    ctx.enqueue_function[kernel](
        O_in.device_tensor(),
        m_in.device_tensor(),
        l_in.device_tensor(),
        o_store,
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=total_smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem)
        ),
    )
    ctx.synchronize()

    # ---- Host reference: FLAT 8-way LSE (must equal the hierarchical result) --
    var o_res = o_dev.tensor()  # update=True: device -> host
    var o_ptr = o_res.ptr

    var max_abs_err: Float32 = 0.0
    var max_rel_err: Float32 = 0.0
    var num_failures: Int = 0
    var nan_count: Int = 0

    for r in range(ROWS):
        var mmax = min_or_neg_inf[ACC_TYPE]()
        for p in range(NUM_PARTITIONS):
            mmax = max(mmax, m_host.ptr[p * ROWS + r])
        var l_ref: Float32 = 0.0
        for p in range(NUM_PARTITIONS):
            l_ref += (
                exp2(m_host.ptr[p * ROWS + r] - mmax) * l_host.ptr[p * ROWS + r]
            )
        var inv = recip(l_ref)
        for dd in range(depth):
            var o_ref: Float32 = 0.0
            for p in range(NUM_PARTITIONS):
                o_ref += (
                    exp2(m_host.ptr[p * ROWS + r] - mmax)
                    * O_host.ptr[(p * ROWS + r) * depth + dd]
                )
            var ref_val = o_ref * inv
            # logical row r -> physical row ROWS + r (front pad).
            var got = o_ptr[(ROWS + r) * depth + dd]
            if isnan(got):
                nan_count += 1
                continue
            var abs_err = abs(got - ref_val)
            var rel_err = abs_err / max(abs(ref_val), Float32(1.0))
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            if abs_err > ATOL and rel_err > RTOL:
                num_failures += 1

    print("  max abs err: " + String(max_abs_err))
    print("  max rel err: " + String(max_rel_err))
    print("  NaNs: " + String(nan_count))
    print(
        "  failures (atol="
        + String(ATOL)
        + " rtol="
        + String(RTOL)
        + "): "
        + String(num_failures)
        + " / "
        + String(ROWS * depth)
    )

    assert_true(
        num_failures == 0 and nan_count == 0,
        msg=String(
            "two-level combine FAILED (depth=",
            depth,
            ", case=",
            case_name,
            "): ",
            num_failures,
            " elements exceed tolerance, ",
            nan_count,
            " NaNs (max abs=",
            max_abs_err,
            ", max rel=",
            max_rel_err,
            ")",
        ),
    )
    print("  PASSED")

    _ = O_in^
    _ = m_in^
    _ = l_in^
    _ = o_dev^


def main() raises:
    with DeviceContext() as ctx:
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            print("Skipping: this test requires B200 (SM100)")
            return
        test_combine[64, 0](ctx)
        test_combine[64, 1](ctx)
        test_combine[64, 2](ctx)
        test_combine[64, 3](ctx)
        test_combine[64, 4](ctx)
        test_combine[128, 0](ctx)
        test_combine[128, 1](ctx)
        test_combine[128, 2](ctx)
        test_combine[128, 3](ctx)
        test_combine[128, 4](ctx)
        print("\ntwo-level 8-way LSE combine test PASSED.")
