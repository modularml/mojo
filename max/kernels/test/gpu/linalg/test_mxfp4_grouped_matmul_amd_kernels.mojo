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
"""Exhaustive kernel-level tests for the preshuffled-B grouped MXFP4 kernels.

Bypasses the public `block_scaled_grouped_matmul_amd_preb` dispatcher and exercises
each preb kernel directly against a per-expert ungrouped GPU reference
(`block_scaled_matmul_amd`):

  - `PreShuffledBGroupedGEMM.launch[persistent=True]`   — persistent 1D grid
                                                          + XCD swizzle
  - `PreShuffledBGroupedGEMM.launch[persistent=False]`  — direct 3D grid

Dense kernel coverage lives in `test_mxfp4_grouped_matmul_amd.mojo`.

Coverage matrix per kernel:
  - Single expert tiny  (most WGs idle for persistent)
  - Multi-expert mixed sizes
  - Inactive slot: M=0 in the middle
  - Inactive slot: expert_id=-1
  - More tiles than total_wg (persistent multi-wave per WG)
  - Kimi-decode / Kimi-prefill scale at L2 dimensions
  - BK_ELEMS=512 only — the preshuffled-scales preb path requires
    num_k_mmas % 2 == 0 (i.e. BK_ELEMS >= 256)

Currently MI355X-only.

Usage:
  br test_mxfp4_grouped_matmul_amd_kernels.mojo.test
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from max.gpu.memory import CacheOperation
from std.math import align_up, ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import (
    PreShuffledBGroupedGEMM,
    Shuffler,
    block_scaled_matmul_amd,
)


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_bytes(buf: HostBuffer[.uint8], n: Int):
    for i in range(n):
        buf[i] = UInt8(random_ui64(0, 255))


def _fill_random_e8m0(buf: HostBuffer[.float8_e8m0fnu], n: Int):
    """Scales clamped to E8M0 byte range [125..129] = magnitudes [0.25..4],
    keeping f32 accumulators in range while still exercising scale-dequant."""
    for i in range(n):
        buf[i] = bitcast[.float8_e8m0fnu](UInt8(random_ui64(125, 129)))


def _build_routing(
    a_offsets_host: HostBuffer[.uint32],
    expert_ids_host: HostBuffer[.int32],
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
):
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])


# ===----------------------------------------------------------------------=== #
# GPU reference — per-expert ungrouped `block_scaled_matmul_amd`.
# Shares the AMD MFMA code path, so a bug common to all MFMA paths would not
# be caught; but it's fast enough to support large shapes.
# ===----------------------------------------------------------------------=== #


def _gpu_per_expert_reference[
    num_experts: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    num_active: Int,
    a_offsets_host: HostBuffer[.uint32],
    expert_ids_host: HostBuffer[.int32],
    a_dev: DeviceBuffer[.uint8],
    b_dev: DeviceBuffer[.uint8],
    a_scales_dev: DeviceBuffer[.float8_e8m0fnu],
    b_scales_dev: DeviceBuffer[.float8_e8m0fnu],
    mut c_ref_dev: DeviceBuffer[.float32],
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // MXFP4_SF_VECTOR_SIZE

    for i in range(num_active):
        var token_start = Int(a_offsets_host[i])
        var token_end = Int(a_offsets_host[i + 1])
        var num_tokens = token_end - token_start
        var expert_id = Int(expert_ids_host[i])

        if num_tokens <= 0 or expert_id < 0:
            continue

        var a_expert = TileTensor(
            a_dev.unsafe_ptr() + token_start * packed_K,
            row_major(Coord(num_tokens, Idx[packed_K])),
        )
        var b_expert = TileTensor(
            b_dev.unsafe_ptr() + expert_id * N * packed_K,
            row_major[N, packed_K](),
        ).as_immut()
        var sfa_expert = TileTensor(
            a_scales_dev.unsafe_ptr() + token_start * scale_K,
            row_major(Coord(num_tokens, Idx[scale_K])),
        ).as_immut()
        var sfb_expert = TileTensor(
            b_scales_dev.unsafe_ptr() + expert_id * N * scale_K,
            row_major[N, scale_K](),
        ).as_immut()
        var c_expert = TileTensor(
            c_ref_dev.unsafe_ptr() + token_start * N,
            row_major(Coord(num_tokens, Idx[N])),
        )

        block_scaled_matmul_amd(
            c_expert, a_expert, b_expert, sfa_expert, sfb_expert, ctx
        )


# ===----------------------------------------------------------------------=== #
# Shared test body — only the `persistent` and `cu_count` comptime values
# differ between `test_persistent` and `test_direct`.
# ===----------------------------------------------------------------------=== #


def _run_preb[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int,
    persistent: Bool,
    cu_count: Int,  # struct param — `total_wg = cu_count * wg_per_cu`
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,  # struct param — sizes the persistent grid
    static_grid_z: Bool = False,  # comptime grid.z = n_local_experts (direct)
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
    grid_m_cap: Int = -1,  # direct grid.y cap (-1 => full stride)
    ascale_stride_toks: Int = -1,  # A-scale slot stride source (-1 => max)
) raises:
    comptime assert K % 128 == 0, "K must be a multiple of 128"
    comptime packed_K = K // 2
    comptime scale_K = K // MXFP4_SF_VECTOR_SIZE

    var total_tokens = 0
    var max_tokens = 0
    var num_active = len(num_tokens_by_expert)
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)

    comptime label = "[persistent]" if persistent else "[direct]    "
    print(
        "  ",
        label,
        name,
        " E=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " BM=",
        BM,
        " BN=",
        BN,
        " BK_ELEMS=",
        BK_ELEMS,
        " WN=",
        WN,
    )

    # Host buffers + random init.
    var a_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * packed_K)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[.int32](num_active)
    ctx.synchronize()

    _fill_random_bytes(a_h, total_tokens * packed_K)
    _fill_random_bytes(b_h, num_experts * N * packed_K)
    _fill_random_e8m0(a_sc_h, total_tokens * scale_K)
    _fill_random_e8m0(b_sc_h, num_experts * N * scale_K)
    _build_routing(a_off_h, eid_h, num_tokens_by_expert, expert_ids_list)

    # Per-expert preshuffled-A-scale slot stride: align_up(stride, 32).
    # ascale_stride_toks > max_tokens overshoots the grid.y cap to test decoupling.
    var ascale_toks = (
        ascale_stride_toks if ascale_stride_toks > 0 else max_tokens
    )
    var max_padded_M = align_up(ascale_toks, 32)

    # Device buffers + upload.
    var a_d = ctx.enqueue_create_buffer[.uint8](total_tokens * packed_K)
    var b_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var b_pre_d = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var a_sc_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    # Preshuffled scale buffers (uint8 — opaque bytes). The dispatcher
    # expects scale tensors in scale-4d byte order; we produce them
    # below via the GPU launcher (A) and CPU helper (B, static weights).
    var a_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * max_padded_M * scale_K
    )
    var b_sc_pre_d = ctx.enqueue_create_buffer[.uint8](
        num_experts * N * scale_K
    )
    var a_off_d = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[.int32](num_active)
    var c_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_ref_d = ctx.enqueue_create_buffer[.float32](total_tokens * N)

    # Zero c_d and c_ref_d. Inactive slots (M=0 or expert_id=-1) leave their
    # output range unwritten by both the kernel and the reference, so the
    # comparison against fresh-allocated garbage would otherwise mismatch.
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(a_sc_d, a_sc_h)
    ctx.enqueue_copy(b_sc_d, b_sc_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    # GPU preshuffle b_d → b_pre_d.
    var b_raw_tt = TileTensor(
        b_d, row_major[num_experts, N, packed_K]()
    ).as_immut()
    var b_pre_dst_tt = TileTensor(
        b_pre_d,
        Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=packed_K],
    )
    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=packed_K](
        b_raw_tt, b_pre_dst_tt, ctx
    )

    # GPU preshuffle of A-scales into per-expert fixed-stride slots.
    var a_sc_raw_u8_tt = TileTensor(
        a_sc_d.unsafe_ptr().bitcast[Scalar[.uint8]](),
        row_major(Coord(total_tokens, Idx[scale_K])),
    ).as_immut()
    var a_sc_pre_tt = TileTensor(
        a_sc_pre_d,
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    )
    var a_off_tt_for_pre = TileTensor(
        a_off_d, row_major(Coord(num_active + 1))
    ).as_immut()
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        a_sc_raw_u8_tt,
        a_sc_pre_tt,
        a_off_tt_for_pre,
        num_active,
        ascale_toks,
        ctx.default_device_info.sm_count * 2,
        ctx,
    )

    # CPU preshuffle of B-scales. B-scales are static weights — in
    # production this runs once at session.load. Done here on host since
    # the existing helper takes comptime MN; for B that's the static N.
    var b_sc_pre_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * scale_K
    )
    ctx.synchronize()
    var b_sc_raw_u8_tt = TileTensor(
        b_sc_h.unsafe_ptr().bitcast[UInt8](),
        row_major(Coord(Idx[num_experts], Idx[N], Idx[scale_K])),
    )
    Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=scale_K](
        b_sc_raw_u8_tt, b_sc_pre_h
    )
    ctx.enqueue_copy(b_sc_pre_d, b_sc_pre_h)

    # Reference: per-expert ungrouped matmul against raw B (raw scales).
    _gpu_per_expert_reference[num_experts, N, K](
        ctx,
        num_active,
        a_off_h,
        eid_h,
        a_d,
        b_d,
        a_sc_d,
        b_sc_d,
        c_ref_d,
    )

    # Run the preb kernel under test. Scales are the preshuffled
    # buffers; bitcast uint8 ptr → float8_e8m0fnu to match the dispatcher
    # signature (the kernel internally bitcasts back to uint8 for V#
    # construction; the dtype here is a wrapping convention).
    var a_tt = TileTensor(
        a_d, row_major(Coord(total_tokens, Idx[packed_K]))
    ).as_immut()
    var b_pre_tt = TileTensor(
        b_pre_d, row_major[num_experts, N * packed_K]()
    ).as_immut()
    var a_sc_tt = TileTensor(
        a_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
    ).as_immut()
    var b_sc_tt = TileTensor(
        b_sc_pre_d.unsafe_ptr().bitcast[Float8_e8m0fnu](),
        row_major[num_experts, N, scale_K](),
    ).as_immut()
    var a_off_tt = TileTensor(a_off_d, row_major(Coord(num_active + 1)))
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active)))
    var c_tt = TileTensor(c_d, row_major(Coord(total_tokens, Idx[N])))

    PreShuffledBGroupedGEMM[cu_count=cu_count, wg_per_cu=wg_per_cu].launch[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WN=WN,
        persistent=persistent,
        b_cache_policy=b_cache_policy,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
        pipeline_depth=pipeline_depth,
        waves_per_eu=waves_per_eu,
        static_grid_z=static_grid_z,
    ](
        c_tt,
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        ascale_toks,  # max_num_tokens_per_expert = A-scale slot stride
        num_active,
        ctx,
        grid_m_cap,
    )
    ctx.synchronize()

    # Compare.
    var c_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    assert_almost_equal(
        c_h.unsafe_ptr(),
        c_ref_h.unsafe_ptr(),
        total_tokens * N,
        atol=0.05,
        rtol=0.05,
    )
    print("    PASS")

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = a_sc_d^
    _ = b_sc_d^
    _ = a_sc_pre_d^
    _ = b_sc_pre_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_d^
    _ = c_ref_d^


def test_persistent[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,  # default = MI355X
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """`PreShuffledBGroupedGEMM.launch[persistent=True]` — 1D `total_wg` grid
    walking a global tile counter; XCD swizzle on `block_idx.x`.

    `cu_count` overrides the launch grid size; default 256 = MI355X. Lower
    values shrink `total_wg = cu_count * wg_per_cu`, which lets the multi-wave
    test trigger the per-WG `while target_tile < expert_end: target_tile +=
    total_wg` path with a CPU-tractable workload. `wg_per_cu` is a launch
    comptime sizing param; results must be unchanged across values.
    """
    _run_preb[
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=True,
        cu_count=cu_count,
        BM=BM,
        BN=BN,
        WN=WN,
        b_cache_policy=b_cache_policy,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
        pipeline_depth=pipeline_depth,
        waves_per_eu=waves_per_eu,
        wg_per_cu=wg_per_cu,
    ](name, num_tokens_by_expert, expert_ids_list, ctx)


def test_direct[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,
    BM: Int = 64,
    BN: Int = 128,
    WN: Int = 64,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    waves_per_eu: Int = 0,
    wg_per_cu: Int = 2,
    static_grid_z: Bool = False,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
    grid_m_cap: Int = -1,
    ascale_stride_toks: Int = -1,
) raises:
    """`PreShuffledBGroupedGEMM.launch[persistent=False]` — 3D workload-sized
    grid: one WG per (n_tile, m_tile, expert). `wg_per_cu` is accepted for
    signature parity with `test_persistent`; the direct grid ignores it.

    `grid_m_cap` sizes grid.y to a decode cap; `ascale_stride_toks` overrides
    the A-scale slot stride; `static_grid_z` uses the comptime local-expert
    count for grid.z (requires routing arrays sized to `num_experts`)."""
    _run_preb[
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=False,
        cu_count=cu_count,
        BM=BM,
        BN=BN,
        WN=WN,
        b_cache_policy=b_cache_policy,
        cluster_drain_sched=cluster_drain_sched,
        mfma_cluster=mfma_cluster,
        pipeline_depth=pipeline_depth,
        waves_per_eu=waves_per_eu,
        wg_per_cu=wg_per_cu,
        static_grid_z=static_grid_z,
    ](
        name,
        num_tokens_by_expert,
        expert_ids_list,
        ctx,
        grid_m_cap,
        ascale_stride_toks,
    )


# ===----------------------------------------------------------------------=== #
# Test matrix
# ===----------------------------------------------------------------------=== #


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_grouped_matmul_amd_kernels currently requires MI355X"

    print("===> preshuffled-B grouped MXFP4 — exhaustive kernel-level tests")

    # ----------------------------------------------------------------- #
    # Shape conventions (matching test_mxfp4_moe_matmul_amd_routed.mojo):
    #   L2 decode/prefill shape:  N=512,   K=2048
    #   L2 gate+up aspect:        N=1024,  K=512
    #   L2 down aspect:           N=512,   K=1024
    #   L3 kimi gate+up (real):   N=14336, K=4096
    #   L3 kimi down (real):      N=4096,  K=7168
    # ----------------------------------------------------------------- #

    # ----------------------------------------------------------------- #
    # Preb persistent kernel — launch[persistent=True]
    # ----------------------------------------------------------------- #
    print("---- preb persistent kernel ----")

    # Structural edge cases (L2 decode shape: N=512, K=2048).
    test_persistent[1, 512, 2048, cluster_drain_sched=True](
        "single-tiny", [16], [0], ctx
    )
    test_persistent[1, 512, 2048, cluster_drain_sched=True](
        "single-mid", [128], [0], ctx
    )
    test_persistent[4, 512, 2048, cluster_drain_sched=True](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    test_persistent[4, 512, 2048, cluster_drain_sched=True](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    test_persistent[4, 512, 2048, cluster_drain_sched=True](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )

    # More tiles than total_wg (= 512 on MI355X) — exercise multi-wave per WG.
    # 9 experts × m_count=4 × gx_n(N=1024)=8 = 288 tiles? need >512.
    # 9 × m_count=8 × gx_n=8 = 576 → M=512 per expert.
    test_persistent[9, 1024, 512, cluster_drain_sched=True](
        "multi-wave",
        [512, 512, 512, 512, 512, 512, 512, 512, 512],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ctx,
    )

    # Kimi-decode-like: 49 active experts, very few tokens each, mixed
    # inactive slots (slot 40: M=0; slot 47: expert_id=-1) — matches the
    # L2.2 / L2.3 pattern from test_mxfp4_moe_matmul_amd_routed.
    test_persistent[49, 512, 2048, cluster_drain_sched=True](
        "kimi-decode-49experts",
        [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            -1,
            48,
        ],
        ctx,
    )

    # Kimi-prefill scaled (L2.4): 49 active experts, ~40 tokens each.
    test_persistent[49, 512, 2048, cluster_drain_sched=True](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )

    # Tile-size variants on the persistent path. Cover the kernel's
    # supported BM/WN matrix (BM ∈ {16} ∪ multiples of 32, same for WN).
    print("---- preb persistent kernel — tile-size variants ----")

    # BM=16 path: 1-row-per-sub-MMA M, shrui scale rotation on odd-parity CTAs.
    test_persistent[
        4, 512, 2048, BM=16, BN=128, WN=64, cluster_drain_sched=True
    ]("bm16-default-wn", [16, 32, 8, 48], [0, 1, 2, 3], ctx)

    # WN=16 path: N-side shrui rotation; BM unchanged.
    test_persistent[
        4, 512, 2048, BM=64, BN=64, WN=16, cluster_drain_sched=True
    ]("wn16-default-bm", [64, 32, 16, 48], [0, 1, 2, 3], ctx)

    # Both BM=16 and WN=16 — exercises both shrui paths simultaneously.
    test_persistent[
        4, 256, 2048, BM=16, BN=64, WN=16, cluster_drain_sched=True
    ]("bm16-wn16", [16, 32, 8, 24], [0, 1, 2, 3], ctx)

    # Smaller BN=64 with default BM/WN — exercises the N-tile dispatcher
    # at a non-default BN.
    test_persistent[
        4, 256, 2048, BM=64, BN=64, WN=64, cluster_drain_sched=True
    ]("bn64", [64, 128, 32, 96], [0, 1, 2, 3], ctx)

    # ----------------------------------------------------------------- #
    # Preb direct kernel — launch[persistent=False]
    # ----------------------------------------------------------------- #
    print("---- preb direct kernel ----")
    test_direct[1, 512, 2048, cluster_drain_sched=True](
        "single-tiny", [16], [0], ctx
    )
    test_direct[1, 512, 2048, cluster_drain_sched=True](
        "single-mid", [128], [0], ctx
    )
    test_direct[4, 512, 2048, cluster_drain_sched=True](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    test_direct[4, 512, 2048, cluster_drain_sched=True](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    test_direct[4, 512, 2048, cluster_drain_sched=True](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )

    # Large single-expert prefill (where direct typically wins over persistent).
    test_direct[1, 1024, 512, cluster_drain_sched=True](
        "single-large", [8192], [0], ctx
    )

    # Tile-size variants on the direct path.
    print("---- preb direct kernel — tile-size variants ----")
    test_direct[4, 256, 2048, BM=16, BN=64, WN=16, cluster_drain_sched=True](
        "bm16-wn16", [3, 7, 1, 5], [0, 1, 2, 3], ctx
    )
    test_direct[4, 512, 2048, BM=64, BN=64, WN=16, cluster_drain_sched=True](
        "wn16-default-bm", [128, 64, 32, 48], [0, 1, 2, 3], ctx
    )

    # Kimi-prefill scaled.
    test_direct[49, 512, 2048, cluster_drain_sched=True](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )

    # ----------------------------------------------------------------- #
    # Production dispatch-band coverage — real kimi N/K with the EXACT
    # (BM, BN, BK_ELEMS, WN, persistent, b_cache_policy) each band in
    # block_scaled_grouped_matmul_amd.mojo launches. One representative token
    # distribution per band (a few active experts + an inactive slot
    # where useful). STREAMING vs ALWAYS is result-identical (cache hint);
    # these cases assert the exact instantiation compiles, runs, and is
    # correct. M values are illustrative of each band's regime, not the
    # dispatcher's estimated_total_m selector (this path bypasses it).
    # ----------------------------------------------------------------- #
    print("---- production band coverage: KIMI up-proj (N=4096, K=7168) ----")
    # band M==1 (decode)
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    # band 2<=M<=4
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 2..4", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # band 17<=M<=400
    test_persistent[
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("up 17..400", [200, 64, 128, 32], [0, 1, 2, 3], ctx)
    # default persistent fallback (M in the 5..16 / 401..4095 gaps)
    test_persistent[
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up default-fallback", [512, 128, 256, 0], [0, 1, 2, 3], ctx)
    # use_direct path (persistent=False). Token count kept modest for memory;
    # the dispatcher is bypassed so M doesn't select the band — the config does.
    test_direct[
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [1024], [0], ctx)

    print("---- production band coverage: KIMI down-proj (N=7168, K=2048) ----")
    # band M==1 (decode)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        cluster_drain_sched=True,
    ]("down M==1", [1, 1, 1, 1], [0, 1, 2, 3], ctx)
    # band 2<=M<=7 (B cached)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 2..7 cached", [4, 2, 6, 3], [0, 1, 2, 3], ctx)
    # band 8<=M<=16 (B STREAMING)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 8..16 STREAMING", [12, 8, 16, 10], [0, 1, 2, 3], ctx)
    # band 17<=M<=37 / 385<=M<=400 (same config, B cached)
    test_persistent[
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down 17..37 / 385..400 cached", [32, 24, 37, 0], [0, 1, 2, 3], ctx)
    # band 38<=M<=384 (B STREAMING)
    test_persistent[
        4,
        7168,
        2048,
        BM=32,
        BN=256,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=CacheOperation.STREAMING,
        cluster_drain_sched=True,
    ]("down 38..384 STREAMING", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    # band 401<=M<=1200 (BK_ELEMS=256, B cached)
    test_persistent[
        4,
        7168,
        2048,
        BM=64,
        BN=256,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down 401..1200 cached", [256, 256, 128, 64], [0, 1, 2, 3], ctx)
    # default persistent fallback (M in the 1201..4095 gap)
    test_persistent[
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down default-fallback", [384, 128, 128, 0], [0, 1, 2, 3], ctx)
    # use_direct path (persistent=False); token count kept modest for memory.
    test_direct[
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct", [512], [0], ctx)

    print("---- preb waves_per_eu EU-bounding cap ----")
    test_persistent[4, 512, 2048, waves_per_eu=1, cluster_drain_sched=True](
        "wpe=1 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx
    )
    test_persistent[4, 512, 2048, waves_per_eu=2, cluster_drain_sched=True](
        "wpe=2 persistent", [128, 96, 128, 64], [0, 1, 2, 3], ctx
    )
    test_direct[
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        waves_per_eu=2,
        cluster_drain_sched=True,
    ]("wpe=2 direct", [512], [0], ctx)

    # Each band config the retuned dispatcher selects.
    print("---- retuned dispatcher band configs ----")
    comptime SX = CacheOperation.STREAMING
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        4096,
        7168,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down BN64 tiny", [2, 1, 0, 1], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM16/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=32,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM32/BN128 STREAM", [256, 256, 256, 256], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down BM64/BN128 STREAM", [512, 512, 256, 256], [0, 1, 2, 3], ctx)
    test_direct[
        1,
        7168,
        2048,
        BM=64,
        BN=128,
        BK_ELEMS=256,
        WN=64,
        cluster_drain_sched=True,
    ]("down direct BK256", [512], [0], ctx)

    # Remaining bands the retuned dispatcher can select, plus the real
    # EP=4 batch-1 decode point and a skewed-routing stress case.
    # up etm<=20 ALWAYS band (16,64,512,16).
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up BM16/BN64 tiny ALWAYS", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # up etm<=4095 STREAM band (64,128,512,64). Token count kept modest to
    # stay under the harness HBM ceiling; config is what's under test.
    test_persistent[
        4,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up BM64/BN128 STREAM", [192, 192, 128, 128], [0, 1, 2, 3], ctx)
    # up else direct band (64,128,512,64).
    test_direct[
        1,
        4096,
        7168,
        BM=64,
        BN=128,
        BK_ELEMS=512,
        WN=64,
        cluster_drain_sched=True,
    ]("up direct", [512], [0], ctx)

    # etm==1 wg_per_cu=1 variant for both shapes (16,64,512,16). wg_per_cu is
    # a grid-sizing comptime; results must match the wg_per_cu=2 default.
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("up etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        cluster_drain_sched=True,
    ]("down etm1 wg_per_cu=1", [1, 0, 0, 0], [0, 1, 2, 3], ctx)

    # True EP=4 batch-1 decode point: 2 experts, 1 row each (M=2), tile
    # (16,64,512,16) for both up and down.
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("up EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        cluster_drain_sched=True,
    ]("down EP4 decode M=2", [1, 1, 0, 0], [0, 1, 2, 3], ctx)

    # Skewed routing (one hot expert, etm~203 -> (16,128,512,32) STREAM band).
    # Stresses persistent work-stealing under imbalance.
    test_persistent[
        4,
        4096,
        7168,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("up skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)
    test_persistent[
        4,
        7168,
        2048,
        BM=16,
        BN=128,
        BK_ELEMS=512,
        WN=32,
        b_cache_policy=SX,
        cluster_drain_sched=True,
    ]("down skewed hot expert", [200, 1, 1, 1], [0, 1, 2, 3], ctx)

    # MiniMax-M3 dispatcher band configs (N=6144; up K=6144, down K=3072).
    # Exactly the tiles block_scaled_grouped_matmul_amd_preb selects for M3 (defaults
    # for the scheduler knobs, matching run_kernel). Certifies each M3 band
    # instantiation against the per-expert reference.
    print("---- MiniMax-M3 dispatcher band configs ----")
    # up: etm<=128 BN64 STREAM
    test_persistent[
        4, 6144, 6144, BM=16, BN=64, BK_ELEMS=512, WN=16, b_cache_policy=SX
    ]("M3 up etm<=128 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # up: etm 7..14 dip micro-band BM16/BN128 STREAM
    test_persistent[
        4, 6144, 6144, BM=16, BN=128, BK_ELEMS=512, WN=32, b_cache_policy=SX
    ]("M3 up etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # up: etm<=512 BM32/BN128 STREAM
    test_persistent[
        4, 6144, 6144, BM=32, BN=128, BK_ELEMS=512, WN=32, b_cache_policy=SX
    ]("M3 up etm<=512 BM32/BN128 STREAM", [128, 96, 128, 64], [0, 1, 2, 3], ctx)
    # up: etm<=2047 BM64/BN128
    test_persistent[4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=32](
        "M3 up etm<=1023 BM64/BN128/WN32",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    test_persistent[4, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64](
        "M3 up etm<=2047 BM64/BN128/WN64",
        [256, 256, 256, 256],
        [0, 1, 2, 3],
        ctx,
    )
    # up: etm<=4095 BM128/BN128
    test_persistent[4, 6144, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64](
        "M3 up etm<=4095 BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx
    )
    # up: else direct BM64/BN128
    test_direct[1, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64](
        "M3 up direct", [512], [0], ctx
    )
    # down: etm<=96 BN64 STREAM
    test_persistent[
        4, 6144, 3072, BM=16, BN=64, BK_ELEMS=512, WN=16, b_cache_policy=SX
    ]("M3 down etm<=96 BN64 STREAM", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # down: etm 7..14 dip micro-band BM16/BN128 STREAM
    test_persistent[
        4, 6144, 3072, BM=16, BN=128, BK_ELEMS=512, WN=32, b_cache_policy=SX
    ]("M3 down etm7..14 BM16/BN128 STREAM", [4, 2, 3, 4], [0, 1, 2, 3], ctx)
    # down: etm<=384 BM32/BN256
    test_persistent[4, 6144, 3072, BM=32, BN=256, BK_ELEMS=512, WN=64](
        "M3 down etm<=384 BM32/BN256", [128, 96, 128, 64], [0, 1, 2, 3], ctx
    )
    # down: etm<=1536 BM64/BN128
    test_persistent[4, 6144, 3072, BM=64, BN=128, BK_ELEMS=512, WN=32](
        "M3 down etm<=1536 BM64/BN128", [256, 256, 256, 256], [0, 1, 2, 3], ctx
    )
    # down: else BM128/BN128 persistent
    test_persistent[4, 6144, 3072, BM=128, BN=128, BK_ELEMS=512, WN=64](
        "M3 down else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx
    )
    # down decode band: direct capped grid.y + static grid.z vs ref, across
    # balanced / skew / BM32 / static-z empties / A-scale decoupling.
    test_direct[
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    test_direct[
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    test_direct[
        4,
        6144,
        3072,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    # static grid.z over-launch: 8 slots, 2 active; 6 M==0 slots early-return.
    test_direct[
        8,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    # Decoupling: A-scale stride (512) >> grid.y cap (32).
    test_direct[
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 down decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    # gate_up decode band (deep K=6144): same coverage as the down cases.
    test_direct[
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap bal",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    test_direct[
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap skew",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    test_direct[
        4,
        6144,
        6144,
        BM=32,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap BM32",
        [8, 8, 8, 8],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
    )
    # static grid.z over-launch: 8 slots, 2 active; 6 M==0 slots early-return.
    test_direct[
        8,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct+cap static-z empties",
        [32, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        ctx,
        grid_m_cap=32,
    )
    # Decoupling under deep K: A-scale stride (512) >> grid.y cap (32).
    test_direct[
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        b_cache_policy=SX,
        static_grid_z=True,
    ](
        "M3 gate_up decode direct decouple stride=512",
        [32, 4, 4, 4],
        [0, 1, 2, 3],
        ctx,
        grid_m_cap=32,
        ascale_stride_toks=512,
    )
    # up/gate (unfused, N=3072, K=6144): etm<=4096 BM64/BN128 persistent
    test_persistent[4, 3072, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64](
        "M3 up/gate etm<=4096 BM64/BN128",
        [256, 128, 256, 128],
        [0, 1, 2, 3],
        ctx,
    )
    # up/gate (unfused): else BM128/BN128 persistent
    test_persistent[4, 3072, 6144, BM=128, BN=128, BK_ELEMS=512, WN=64](
        "M3 up/gate else BM128/BN128", [256, 128, 256, 128], [0, 1, 2, 3], ctx
    )
    # M3 etm<=2 low-batch-decode band
    test_persistent[
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    test_persistent[
        4,
        6144,
        6144,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 up etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)
    test_persistent[
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (1 expert)", [1], [0], ctx)
    test_persistent[
        4,
        6144,
        3072,
        BM=16,
        BN=64,
        BK_ELEMS=512,
        WN=16,
        wg_per_cu=1,
        b_cache_policy=SX,
    ]("M3 down etm<=2 decode BN64/wg1 (2 experts)", [1, 1], [0, 1], ctx)

    # ----------------------------------------------------------------- #
    # Pipeline-depth parameterization (deeper-prefetch lever).
    # `pipeline_depth > 2` sizes the B-fragment ring to `pipeline_depth`
    # slots and swaps the b_prefetch steady loop's end-of-iter draining
    # barrier() for the non-draining `s_waitcnt[lgkmcnt=0]` + bare s_barrier.
    # This is the CORRECTNESS gate for the depth plumbing + barrier seam: at
    # depth 3/4 the deeper ring is allocated and the non-draining barrier
    # engages, and the result must still match the per-expert reference.
    #
    # FALSE-NEGATIVE GUARD: this is numerics-only. A build that (regressively)
    # left the draining barrier in place would ALSO pass here — the perf
    # effect (B loads actually staying outstanding) is NOT observable from
    # numerics and MUST be rocprof-verified on GPU when depth is raised in a
    # dispatch band. Passing this test alone does not prove the lever works.
    #
    # M3 decode-band tiles (N=6144; up K=6144, down K=3072), a few active
    # experts + an inactive slot, on both the persistent and direct grids.
    print("---- pipeline_depth > 2 (deeper prefetch) correctness ----")
    # M3 up decode tile, depth 3 and 4 (persistent).
    test_persistent[
        4, 6144, 6144, BM=16, BN=64, BK_ELEMS=512, WN=16, pipeline_depth=3
    ]("M3 up decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    test_persistent[
        4, 6144, 6144, BM=16, BN=64, BK_ELEMS=512, WN=16, pipeline_depth=4
    ]("M3 up decode depth=4", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # M3 up decode tile, BN128/WN32 variant, depth 3 (persistent).
    test_persistent[
        4, 6144, 6144, BM=16, BN=128, BK_ELEMS=512, WN=32, pipeline_depth=3
    ]("M3 up decode BN128 depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # M3 down decode tile, depth 3 (persistent).
    test_persistent[
        4, 6144, 3072, BM=16, BN=64, BK_ELEMS=512, WN=16, pipeline_depth=3
    ]("M3 down decode depth=3", [8, 4, 0, 4], [0, 1, 2, 3], ctx)
    # Direct grid at depth 3 and 4 — a larger single expert so the steady
    # loop (where the non-draining barrier lives) runs many iterations.
    test_direct[
        1, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64, pipeline_depth=3
    ]("M3 up direct depth=3", [512], [0], ctx)
    test_direct[
        1, 6144, 6144, BM=64, BN=128, BK_ELEMS=512, WN=64, pipeline_depth=4
    ]("M3 up direct depth=4", [512], [0], ctx)

    print("==== all preb grouped MXFP4 kernel tests passed ====")
