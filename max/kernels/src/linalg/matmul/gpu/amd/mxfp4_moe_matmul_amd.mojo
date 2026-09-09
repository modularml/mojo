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
"""MXFP4 x MXFP4 routed MoE matmul kernel for AMD CDNA4.

`MXFP4MoERoutedMatmul` / `mxfp4_moe_matmul_amd_routed` is the full
routed MoE matmul. `block_idx.y` walks per-expert sort blocks, decodes
`sorted_token_ids` per row to gather A from original token order, and
scatters output to `c[t*topk + s, :]`. It's a drop-in replacement for
the gather + grouped-matmul + scatter pipeline.

Data layouts:
    A: `[num_tokens, K_BYTES]` uint8, FP4 packed two-per-byte, row-major.
    B: 5D-preshuffled (see `block_scaled_preshuffle_layouts.b_5d_grouped_layout`).
    sfa, sfb: 4D-preshuffled E8M0 scale bytes (`scale_4d_grouped_layout`).
    C: `[num_tokens * topk, N]` fp32, row-major.

For the MFMA scale convention (per-lane scale i32, OPSEL-selected byte
applied to the lane's `(M=lane%16, K-group=lane/16)` slot) see the AMD
CDNA4 ISA section 7.2.1.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.math import align_up, ceildiv
from std.memory import AddressSpace
from std.sys import simd_width_of
from std.utils import StaticTuple

from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
)
from layout._utils import make_amd_buffer_resource
from layout.tile_tensor import stack_allocation
from layout.tile_layout import col_major, row_major

from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import RegTileLoader

from .block_scaled_preshuffle_loaders import (
    PreshuffledBLoader,
    PreshuffledScaleLoader,
)


# ===----------------------------------------------------------------------=== #
# MXFP4MoERoutedMatmul — sorted-token MoE routing
# ===----------------------------------------------------------------------=== #
#
# 2D grid: `block_idx.x` walks N-tiles, `block_idx.y` walks sort blocks.
# Each sort block has `sort_block_m` rows of work for one expert.
#
#   - `expert_id = expert_ids[block_idx.y]` (per-block; -1 skips block).
#   - SFA is gathered + preshuffled per-call into a `[size_expert_ids *
#     sort_block_m, K_SCALES]` buffer; per-block offset = `block_idx.y *
#     sort_block_m * K_SCALES`.
#   - A is `[num_tokens, K_BYTES]` row-major. Each thread decodes its
#     rows' `(t, s)` from `sorted_token_ids` once, then uses `t` as the
#     DRAM row index for the cooperative A load. Buffer-resource
#     clamping handles `t >= num_tokens` (sentinel rows return zero).
#   - C-store scatters each lane's 4 outputs per `(m_mma, n_mma)` to
#     rows `t * topk + s`; padded rows skip the store.
#
# Two `INPUT_ROW_MODE`s share the kernel:
#   - TOKEN_ID    (stage 1 / up):   A_row = t            C_row = t*topk+s
#   - TOKEN_SLOT  (stage 2 / down): A_row = t*topk+s     C_row = t*topk+s


@fieldwise_init
struct InputRowMode(TrivialRegisterPassable):
    """Selects how the kernel decodes A's row index from `sorted_token_ids`."""

    var _value: Int8
    comptime TOKEN_ID = Self(0)  # A_row = t (stage1 / up)
    comptime TOKEN_SLOT = Self(1)  # A_row = t*topk + s (stage2 / down)


@always_inline
def _xcd_wgm_swizzle(
    wgid_raw: Int, num_pid_m: Int, num_pid_n: Int
) -> Tuple[Int, Int]:
    """HipKittens chiplet + L2 swizzle for MI355X (CDNA4).

    Mirrors `amd_4wave_matmul._xcd_wgm_swizzle`. Stage 1 spreads adjacent
    raw wgids across 8 XCDs (each with its own L2 slice); stage 2 walks
    WGM=4 row-blocks together to improve L2 reuse on the shared operand.

    Skipped when the WG count is below `4 * num_CUs`: the swizzle math
    only pays off when each CU runs multiple WGs in one launch.
    """
    comptime NUM_XCDS = 8
    comptime WGM = 4
    comptime NUM_CUS = 304  # MI355X
    comptime SWIZZLE_THRESHOLD = 4 * NUM_CUS

    var num_wgs = num_pid_m * num_pid_n

    if num_wgs <= SWIZZLE_THRESHOLD or num_wgs % NUM_XCDS != 0:
        var pid_m, pid_n = divmod(wgid_raw, num_pid_n)
        return (pid_m, pid_n)

    var intra_xcd, xcd = divmod(wgid_raw, NUM_XCDS)
    var wgid = xcd * (num_wgs // NUM_XCDS) + intra_xcd
    var num_wgid_in_group = WGM * num_pid_n
    var group_id, intra_group = divmod(wgid, num_wgid_in_group)
    var first_pid_m = group_id * WGM
    var group_size_m = min(num_pid_m - first_pid_m, WGM)
    var pid_n, intra_group_m = divmod(intra_group, group_size_m)
    var pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


struct MXFP4MoERoutedMatmul[
    BM: Int = 64,
    BN: Int = 64,
    BK_ELEMS: Int = 256,
    num_warps_m: Int = 2,
    num_warps_n: Int = 2,
    topk: Int = 1,
    INPUT_ROW_MODE: InputRowMode = InputRowMode.TOKEN_ID,
    enable_swizzle: Bool = True,
]:
    """Implements the routed MXFP4-by-MXFP4 MoE matmul for AMD CDNA4.

    The kernel walks a 2D grid where `block_idx.x` covers N-tiles and
    `block_idx.y` covers per-expert sort blocks. Each block decodes
    `sorted_token_ids` to gather A rows from the original token order,
    accumulates the block-scaled MFMA products against the preshuffled B
    and E8M0 scale buffers, and scatters results to `c[t*topk + s, :]`.

    Parameters:
        BM: M-tile size in rows, also the per-block sort block height.
        BN: N-tile size in columns.
        BK_ELEMS: K-tile size in MXFP4 elements (two per byte).
        num_warps_m: Number of warps assigned to the M dimension.
        num_warps_n: Number of warps assigned to the N dimension.
        topk: Number of experts each token routes to.
        INPUT_ROW_MODE: Selects how the kernel decodes A's row index from `sorted_token_ids`.
        enable_swizzle: Enables the XCD/WGM block-id swizzle for MI355X L2 reuse.
    """

    comptime BK_BYTES: Int = Self.BK_ELEMS // 2
    comptime BK_SCALES: Int = Self.BK_ELEMS // 32
    comptime WM: Int = Self.BM // Self.num_warps_m
    comptime WN: Int = Self.BN // Self.num_warps_n
    comptime num_warps: Int = Self.num_warps_m * Self.num_warps_n
    comptime num_threads: Int = Self.num_warps * WARP_SIZE

    comptime MMA_M: Int = 16
    comptime MMA_N: Int = 16
    comptime MMA_K_BYTES: Int = 64

    comptime num_m_mmas: Int = Self.WM // Self.MMA_M
    comptime num_n_mmas: Int = Self.WN // Self.MMA_N
    comptime num_k_tiles_per_BK: Int = Self.BK_BYTES // Self.MMA_K_BYTES
    comptime pack_K: Int = 2
    comptime num_scale_packs_per_BK: Int = (
        Self.num_k_tiles_per_BK // Self.pack_K
    )

    comptime FRAG_W_BYTES: Int = 16
    comptime C_FRAG_SIZE: Int = (Self.MMA_M * Self.MMA_N) // WARP_SIZE
    # sort_block_m == BM for now (simplification; flydsl uses max(32, tile_m)).
    comptime sort_block_m: Int = Self.BM

    @staticmethod
    def run[
        K_BYTES: Int,
        K_SCALES: Int,
        N: Int,
        N_padded_scale: Int,
    ](
        c: TileTensor[mut=True, ...],
        a_tt: TileTensor[.uint8, ...],
        b_pre_tt: TileTensor[.uint8, ...],
        sfa_pre_tt: TileTensor[.uint8, ...],
        sfb_pre_tt: TileTensor[.uint8, ...],
        sorted_token_ids: TileTensor[.uint32, ...],
        expert_ids: TileTensor[.int32, ...],
        num_tokens: Int32,
        size_expert_ids: Int32,
    ):
        var _num_tokens = Int(num_tokens)
        var _size_expert_ids = Int(size_expert_ids)
        comptime out_dtype = type_of(c).dtype
        comptime assert (
            Self.num_m_mmas * Self.pack_K <= 4
        ), "num_m_mmas * pack_K must be <= 4"
        comptime assert (
            Self.num_n_mmas * Self.pack_K <= 4
        ), "num_n_mmas * pack_K must be <= 4"
        comptime assert (
            K_BYTES % Self.BK_BYTES == 0
        ), "K_BYTES must be a multiple of BK_BYTES"

        # XCD/WGM block-id swizzle (HK-style). Logical (bx, by_n) come from
        # the swizzle when enabled, else they're the raw block_idx.{y,x}.
        # The swizzle no-ops below the per-arch threshold (`4 * num_CUs`).
        var bx: Int
        var by_n: Int
        comptime if Self.enable_swizzle:
            comptime num_pid_n = ceildiv(N, Self.BN)
            var num_pid_m = _size_expert_ids
            var wgid_raw = Int(block_idx.y) * num_pid_n + Int(block_idx.x)
            var pid = _xcd_wgm_swizzle(wgid_raw, num_pid_m, num_pid_n)
            bx = pid[0]
            by_n = pid[1]
        else:
            bx = Int(block_idx.y)
            by_n = Int(block_idx.x)
        var bx_m = bx * Self.sort_block_m

        # NOTE: flydsl-style routing producers can elide empty experts
        # (`skip_experts_with_zero_tokens`) and pair that with a persistent
        # kernel that monotonic-short-circuits past `num_valid_ids`. We
        # currently always allocate per-expert sort blocks (no elision)
        # and run non-persistent, so the predicate would be vacuous. If
        # we add persistent + elide support, restore a `num_valid_ids`
        # arg here and reintroduce both: `if bx_m >= num_valid_ids:
        # return` and `(sorted_row_idx < num_valid_ids)` in row_valids.

        var lane = Int(lane_id())
        var lane_nlane = lane % Self.MMA_M
        var lane_klane = lane // Self.MMA_M
        var wid = warp_id()
        var warp_m, warp_n = divmod(wid, Self.num_warps_n)

        # ---- Decode expert_id, build per-expert / per-block TileTensors ----
        var expert_id_i32 = expert_ids[Coord(bx)]

        # TODO: confirm no production routing producer emits -1; if not,
        # drop the predicate and update the test fixtures to express
        # "inactive" via empty n_e instead.
        if expert_id_i32 < 0:
            return
        var expert_id = Int(expert_id_i32)

        comptime b_per_expert_bytes = N * K_BYTES
        comptime sfb_per_expert_bytes = N_padded_scale * K_SCALES
        comptime sfa_per_block_bytes = Self.sort_block_m * K_SCALES

        var b_pre_expert = b_pre_tt.tile[1, b_per_expert_bytes](
            Coord(0, expert_id)
        )
        var sfb_pre_expert = sfb_pre_tt.tile[1, sfb_per_expert_bytes](
            Coord(0, expert_id)
        )
        var sfa_pre_block = sfa_pre_tt.tile[1, sfa_per_block_bytes](
            Coord(0, bx)
        )

        # ---- Loaders ----
        var b_loader = PreshuffledBLoader[N=N, K_BYTES=K_BYTES](
            TileTensor(b_pre_expert.ptr, b_pre_expert.layout)
        )
        var sfa_loader = PreshuffledScaleLoader[
            MN_padded=Self.sort_block_m, K_SCALES=K_SCALES
        ](TileTensor(sfa_pre_block.ptr, sfa_pre_block.layout))
        var sfb_loader = PreshuffledScaleLoader[
            MN_padded=N_padded_scale, K_SCALES=K_SCALES
        ](TileTensor(sfb_pre_expert.ptr, sfb_pre_expert.layout))
        var a_bc = make_amd_buffer_resource(a_tt)

        # ---- SMEM for A ----
        var a_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[Self.BM, Self.BK_BYTES]()
        )

        # ---- Cooperative A load setup (per-row token-ID indirection) ----
        comptime simd_width = simd_width_of[DType.uint8]()
        comptime load_thread_cols = Self.BK_BYTES // simd_width
        comptime load_thread_rows = Self.num_threads // load_thread_cols
        comptime a_loads_per_tile = Self.BM // load_thread_rows
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()

        var thread_idx_x = Int(thread_idx.x)
        var row_thread = thread_idx_x // load_thread_cols
        var col_thread = thread_idx_x % load_thread_cols

        # Decode (t, s) per row this thread will load. Cached across K-iters.
        # TOKEN_ID:   A_row = t        (stage1; multiple slots share the same A row)
        # TOKEN_SLOT: A_row = t*topk+s (stage2; each slot has its own A row)
        var a_rows = StaticTuple[Int, a_loads_per_tile]()
        comptime for i in range(a_loads_per_tile):
            var local_row = row_thread + i * load_thread_rows
            var sorted_row_idx = bx_m + local_row
            var fused = sorted_token_ids[Coord(sorted_row_idx)]
            var t = Int(fused & UInt32(0xFFFFFF))
            comptime if (
                Self.INPUT_ROW_MODE._value == InputRowMode.TOKEN_ID._value
            ):
                a_rows[i] = t
            else:
                var s = Int(fused >> UInt32(24))
                a_rows[i] = t * Self.topk + s

        # ---- Per-warp accumulator ----
        var c_acc = StaticTuple[
            SIMD[.float32, Self.C_FRAG_SIZE],
            Self.num_m_mmas * Self.num_n_mmas,
        ]()
        comptime for i in range(Self.num_m_mmas * Self.num_n_mmas):
            c_acc[i] = SIMD[.float32, Self.C_FRAG_SIZE](0.0)

        # ---- K-loop ----
        comptime num_k_iters = K_BYTES // Self.BK_BYTES

        for k_iter in range(num_k_iters):
            # 1. Manual cooperative A load with per-row t indirection.
            #    Each thread issues a_loads_per_tile buffer_loads from its own
            #    rows of A in DRAM (row index = t for that local_row), writes
            #    to LDS at the local_row position. Buffer-resource clamping
            #    handles invalid t (>= num_tokens) by returning zeros.
            comptime for i in range(a_loads_per_tile):
                var a_row = a_rows[i]
                var a_byte_off = (
                    a_row * K_BYTES
                    + k_iter * Self.BK_BYTES
                    + col_thread * simd_width
                )
                var data = a_bc.load[.uint8, simd_width](Int32(a_byte_off))
                var local_row = row_thread + i * load_thread_rows
                var smem_byte_off = (
                    local_row * Self.BK_BYTES + col_thread * simd_width
                )
                a_smem.raw_store[width=simd_width](smem_byte_off, data)
            barrier()

            # 2. Per-warp inner: scale-pack loop, ds_read A, load B + scales, MFMA.
            var a_smem_warp = a_smem.tile[Self.WM, Self.BK_BYTES](warp_m, 0)
            var warp_n_off = warp_n * Self.WN
            var warp_k_byte_base = k_iter * Self.BK_BYTES
            var warp_k_scale_base = k_iter * Self.BK_SCALES

            comptime lane_layout = col_major[
                Self.MMA_M, WARP_SIZE // Self.MMA_M
            ]()

            comptime for sp in range(Self.num_scale_packs_per_BK):
                var a_frags = StaticTuple[
                    SIMD[.uint8, Self.FRAG_W_BYTES],
                    Self.pack_K * Self.num_m_mmas,
                ]()
                comptime for ikxdl in range(Self.pack_K):
                    comptime k_tile = sp * Self.pack_K + ikxdl
                    comptime for m in range(Self.num_m_mmas):
                        var a_frag = (
                            a_smem_warp.tile[Self.MMA_M, Self.MMA_K_BYTES](
                                m, k_tile
                            )
                            .vectorize[1, Self.FRAG_W_BYTES]()
                            .distribute[lane_layout](lane_id())
                        )
                        a_frags[ikxdl * Self.num_m_mmas + m] = a_frag.raw_load[
                            width=Self.FRAG_W_BYTES
                        ](0)

                var b_frags = StaticTuple[
                    SIMD[.uint8, Self.FRAG_W_BYTES],
                    Self.pack_K * Self.num_n_mmas,
                ]()
                comptime for ikxdl in range(Self.pack_K):
                    comptime k_tile = sp * Self.pack_K + ikxdl
                    comptime for n in range(Self.num_n_mmas):
                        var n_log = (
                            by_n * Self.BN
                            + warp_n_off
                            + n * Self.MMA_N
                            + lane_nlane
                        )
                        var k_byte_log = (
                            warp_k_byte_base
                            + k_tile * Self.MMA_K_BYTES
                            + lane_klane * Self.FRAG_W_BYTES
                        )
                        b_frags[
                            ikxdl * Self.num_n_mmas + n
                        ] = b_loader.load_fragment(n_log, k_byte_log)

                var mn_lane_off_a = warp_m * Self.WM + lane_nlane
                var mn_lane_off_b = by_n * Self.BN + warp_n_off + lane_nlane
                var k_scale_lane_off = (
                    warp_k_scale_base + sp * Self.pack_K * 4 + lane_klane
                )
                var a_scale = sfa_loader.load_packed(
                    mn_lane_off_a, k_scale_lane_off
                )
                var b_scale = sfb_loader.load_packed(
                    mn_lane_off_b, k_scale_lane_off
                )

                comptime for ikxdl in range(Self.pack_K):
                    comptime for m in range(Self.num_m_mmas):
                        comptime for n in range(Self.num_n_mmas):
                            var a_frag = a_frags[ikxdl * Self.num_m_mmas + m]
                            var b_frag = b_frags[ikxdl * Self.num_n_mmas + n]
                            comptime acc_idx = m * Self.num_n_mmas + n
                            cdna4_block_scaled_mfma[
                                Int32(ikxdl * Self.num_m_mmas + m),
                                Int32(ikxdl * Self.num_n_mmas + n),
                                CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                                CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
                            ](c_acc[acc_idx], a_frag, b_frag, a_scale, b_scale)

            barrier()

        # ---- C store: scatter to c[t*topk+s, n] per row ----
        var c_block_n_off = by_n * Self.BN

        comptime for m_mma in range(Self.num_m_mmas):
            # Per-(m_mma, lane) row decode: 4 outputs per lane, 4 different rows.
            var c_dst_rows = StaticTuple[Int, Self.C_FRAG_SIZE]()
            var row_valids = StaticTuple[Bool, Self.C_FRAG_SIZE]()
            comptime for i in range(Self.C_FRAG_SIZE):
                var local_row = (
                    warp_m * Self.WM
                    + m_mma * Self.MMA_M
                    + lane_klane * Self.C_FRAG_SIZE
                    + i
                )
                var sorted_row_idx = bx_m + local_row
                var fused = sorted_token_ids[Coord(sorted_row_idx)]
                var t = Int(fused & UInt32(0xFFFFFF))
                var s = Int(fused >> UInt32(24))
                c_dst_rows[i] = t * Self.topk + s
                row_valids[i] = (t < _num_tokens) and (s < Self.topk)

            comptime for n_mma in range(Self.num_n_mmas):
                var c_col = (
                    c_block_n_off
                    + warp_n * Self.WN
                    + n_mma * Self.MMA_N
                    + lane_nlane
                )
                var acc = c_acc[m_mma * Self.num_n_mmas + n_mma]
                comptime for i in range(Self.C_FRAG_SIZE):
                    if row_valids[i]:
                        c[Coord(c_dst_rows[i], c_col)] = acc[i].cast[
                            out_dtype
                        ]()


# Thin host-launchable wrapper around `MXFP4MoERoutedMatmul.run`.
# `ctx.enqueue_function` requires a free function carrying
# `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)` so the compiler
# can budget VGPR/LDS for the launch — struct methods can't carry that
# decoration. This forwarder exists only to be the GPU entry point;
# callers should use `mxfp4_moe_matmul_amd_routed` (below), which
# enqueues this symbol.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(MXFP4MoERoutedMatmul[].num_threads)
    )
)
@__name(
    t"mxfp4_moe_routed_{out_dtype}_BM{BM}_BN{BN}_BK{BK_ELEMS}_N{N}_KS{K_SCALES}_topk{topk}"
)
def _mxfp4_moe_matmul_routed_kernel[
    out_dtype: DType,
    CLayout: TensorLayout,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    SFALayout: TensorLayout,
    SFBLayout: TensorLayout,
    STILayout: TensorLayout,
    EILayout: TensorLayout,
    K_BYTES: Int,
    K_SCALES: Int,
    N: Int,
    N_padded_scale: Int,
    topk: Int,
    INPUT_ROW_MODE: InputRowMode,
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    num_warps_m: Int,
    num_warps_n: Int,
    CEngine: TensorEngine,
    AEngine: TensorEngine,
    BEngine: TensorEngine,
    SFAEngine: TensorEngine,
    SFBEngine: TensorEngine,
    STIEngine: TensorEngine,
    EIEngine: TensorEngine,
](
    c: TileTensor[mut=True, out_dtype, CLayout, MutAnyOrigin, Engine=CEngine],
    a: TileTensor[.uint8, ALayout, ImmutAnyOrigin, Engine=AEngine],
    b_pre: TileTensor[.uint8, BLayout, ImmutAnyOrigin, Engine=BEngine],
    sfa_pre: TileTensor[.uint8, SFALayout, ImmutAnyOrigin, Engine=SFAEngine],
    sfb_pre: TileTensor[.uint8, SFBLayout, ImmutAnyOrigin, Engine=SFBEngine],
    sorted_token_ids: TileTensor[
        .uint32, STILayout, ImmutAnyOrigin, Engine=STIEngine
    ],
    expert_ids: TileTensor[.int32, EILayout, ImmutAnyOrigin, Engine=EIEngine],
    num_tokens: Int32,
    size_expert_ids: Int32,
):
    MXFP4MoERoutedMatmul[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        num_warps_m=num_warps_m,
        num_warps_n=num_warps_n,
        topk=topk,
        INPUT_ROW_MODE=INPUT_ROW_MODE,
    ].run[
        K_BYTES=K_BYTES, K_SCALES=K_SCALES, N=N, N_padded_scale=N_padded_scale
    ](
        c,
        a,
        b_pre,
        sfa_pre,
        sfb_pre,
        sorted_token_ids,
        expert_ids,
        num_tokens,
        size_expert_ids,
    )


def mxfp4_moe_matmul_amd_routed[
    topk: Int = 1,
    INPUT_ROW_MODE: InputRowMode = InputRowMode.TOKEN_ID,
    BM: Int = 64,
    BN: Int = 64,
    BK_ELEMS: Int = 256,
    num_warps_m: Int = 2,
    num_warps_n: Int = 2,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[.uint8, ...],
    b_pre: TileTensor[.uint8, ...],
    sfa_pre: TileTensor[.uint8, ...],
    sfb_pre: TileTensor[.uint8, ...],
    sorted_token_ids: TileTensor[.uint32, ...],
    expert_ids: TileTensor[.int32, ...],
    num_tokens: Int,
    size_expert_ids: Int,
    ctx: DeviceContext,
) raises:
    """Launches the routed MXFP4xMXFP4 matmul on AMD CDNA4.

    Each block_idx.y processes one `sort_block_m` of sorted rows for the
    expert in `expert_ids[block_idx.y]`. block_idx.x walks N-tiles within
    the per-expert N range.

    Parameters:
        topk: Number of experts each token routes to (defaults to 1).
        INPUT_ROW_MODE: Selects how the kernel decodes A's row index from
            `sorted_token_ids` (defaults to `InputRowMode.TOKEN_ID`).
        BM: M-tile size in rows, also the per-block sort block height
            (defaults to 64).
        BN: N-tile size in columns (defaults to 64).
        BK_ELEMS: K-tile size in MXFP4 elements, two per byte (defaults to
            256).
        num_warps_m: Number of warps assigned to the M dimension (defaults
            to 2).
        num_warps_n: Number of warps assigned to the N dimension (defaults
            to 2).

    Args:
        c: Output matrix of shape `[num_tokens * topk, N]`, row-major.
        a: Input matrix of shape `[num_tokens, K_BYTES]` uint8, FP4 packed
            two per byte, row-major.
        b_pre: Preshuffled B weights in the 5D grouped layout (see
            `block_scaled_preshuffle_layouts.b_5d_grouped_layout`).
        sfa_pre: Preshuffled A E8M0 scale bytes in the 4D grouped layout.
        sfb_pre: Preshuffled B E8M0 scale bytes in the 4D grouped layout.
        sorted_token_ids: Fused token/slot IDs packing token index `t` in
            the low 24 bits and slot `s` in the high 8 bits.
        expert_ids: Per-block expert ID; `expert_ids[block_idx.y]` selects
            the expert, and -1 skips the block.
        num_tokens: Number of input token rows in `a`.
        size_expert_ids: Number of per-expert sort blocks; sets `grid_dim.y`.
        ctx: Device context used to enqueue the kernel.
    """
    comptime Kernel = MXFP4MoERoutedMatmul[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        num_warps_m=num_warps_m,
        num_warps_n=num_warps_n,
        topk=topk,
        INPUT_ROW_MODE=INPUT_ROW_MODE,
    ]
    comptime out_dtype = type_of(c).dtype
    comptime K_BYTES = type_of(a).static_shape[1]
    comptime K_SCALES = K_BYTES // 16
    comptime N = type_of(c).static_shape[1]
    # SFB_pre is preshuffled with MN_padded(N) rows per expert. Pass through
    # the comptime padding factor so the loader's layout matches the host.
    comptime N_padded_scale = align_up(N, 32)

    comptime kernel = _mxfp4_moe_matmul_routed_kernel[
        out_dtype,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b_pre).LayoutType,
        type_of(sfa_pre).LayoutType,
        type_of(sfb_pre).LayoutType,
        type_of(sorted_token_ids).LayoutType,
        type_of(expert_ids).LayoutType,
        K_BYTES,
        K_SCALES,
        N,
        N_padded_scale,
        topk,
        INPUT_ROW_MODE,
        BM,
        BN,
        BK_ELEMS,
        num_warps_m,
        num_warps_n,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b_pre).Engine,
        type_of(sfa_pre).Engine,
        type_of(sfb_pre).Engine,
        type_of(sorted_token_ids).Engine,
        type_of(expert_ids).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        b_pre,
        sfa_pre,
        sfb_pre,
        sorted_token_ids,
        expert_ids,
        Int32(num_tokens),
        Int32(size_expert_ids),
        grid_dim=(ceildiv(N, Kernel.BN), size_expert_ids),
        block_dim=Kernel.num_threads,
    )


def mxfp4_moe_matmul_amd_routed_dispatch[
    topk: Int = 1,
    INPUT_ROW_MODE: InputRowMode = InputRowMode.TOKEN_ID,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[.uint8, ...],
    b_pre: TileTensor[.uint8, ...],
    sfa_pre: TileTensor[.uint8, ...],
    sfb_pre: TileTensor[.uint8, ...],
    sorted_token_ids: TileTensor[.uint32, ...],
    expert_ids: TileTensor[.int32, ...],
    num_tokens: Int,
    size_expert_ids: Int,
    max_tokens_per_expert: Int,
    ctx: DeviceContext,
) raises:
    """Dispatches the routed kernel to a tile shape based on `max_tokens_per_expert`.

    Keeps `BM=64` (= sort_block_m) fixed so callers' host-side preshuffle
    stays valid across all dispatch buckets. Varies `BN` and warp count.
    First-cut heuristic: perf-tune once we have flydsl-comparable numbers.

    Parameters:
        topk: Number of experts each token routes to (defaults to 1).
        INPUT_ROW_MODE: Selects how the kernel decodes A's row index from
            `sorted_token_ids` (defaults to `InputRowMode.TOKEN_ID`).

    Args:
        c: Output matrix of shape `[num_tokens * topk, N]`, row-major.
        a: Input matrix of shape `[num_tokens, K_BYTES]` uint8, FP4 packed
            two per byte, row-major.
        b_pre: Preshuffled B weights in the 5D grouped layout (see
            `block_scaled_preshuffle_layouts.b_5d_grouped_layout`).
        sfa_pre: Preshuffled A E8M0 scale bytes in the 4D grouped layout.
        sfb_pre: Preshuffled B E8M0 scale bytes in the 4D grouped layout.
        sorted_token_ids: Fused token/slot IDs packing token index `t` in
            the low 24 bits and slot `s` in the high 8 bits.
        expert_ids: Per-block expert ID; `expert_ids[block_idx.y]` selects
            the expert, and -1 skips the block.
        num_tokens: Number of input token rows in `a`.
        size_expert_ids: Number of per-expert sort blocks; sets `grid_dim.y`.
        max_tokens_per_expert: Upper bound on tokens routed to any single
            expert, used to pick the tile shape.
        ctx: Device context used to enqueue the kernel.
    """
    if max_tokens_per_expert <= 32:
        # Decode-class: smaller BN, fewer warps. 4 warps in 2x2.
        mxfp4_moe_matmul_amd_routed[
            topk=topk,
            INPUT_ROW_MODE=INPUT_ROW_MODE,
            BM=64,
            BN=64,
            BK_ELEMS=256,
            num_warps_m=2,
            num_warps_n=2,
        ](
            c,
            a,
            b_pre,
            sfa_pre,
            sfb_pre,
            sorted_token_ids,
            expert_ids,
            num_tokens,
            size_expert_ids,
            ctx,
        )
    else:
        # Prefill-class: wider BN for more N-parallelism. 8 warps in 2x4.
        mxfp4_moe_matmul_amd_routed[
            topk=topk,
            INPUT_ROW_MODE=INPUT_ROW_MODE,
            BM=64,
            BN=128,
            BK_ELEMS=256,
            num_warps_m=2,
            num_warps_n=4,
        ](
            c,
            a,
            b_pre,
            sfa_pre,
            sfb_pre,
            sorted_token_ids,
            expert_ids,
            num_tokens,
            size_expert_ids,
            ctx,
        )
