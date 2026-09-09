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

"""Grouped MXFP4 matmul kernels for AMD CDNA4 GPUs.

Provides MoE expert-dispatched grouped matmul in two variants: the native
path (`block_scaled_grouped_matmul_amd`) with on-the-fly B layout handling, and the
pre-shuffled-B path (`block_scaled_grouped_matmul_amd_preb` /
`PreShuffledBGroupedGEMM`) where weights are pre-arranged into a layout that
enables coalesced shared-memory reads and direct MFMA consumption.
"""

from std.math import align_up, ceildiv
from std.sys import get_defined_bool, get_defined_int
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from max.gpu.memory import CacheOperation

from layout import Coord, Idx, TensorLayout, TensorEngine, TileTensor
from layout.tile_layout import row_major

from std.utils import StaticTuple

from ....arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from .block_scaled_matmul_amd import BlockScaledMatmulAMD
from .block_scaled_matmul_amd_preb import BlockScaledMatmulAMD_PreB


@always_inline
def _waves_per_eu_attr[waves_per_eu: Int]() -> __mlir_type.`!kgen.string`:
    # `amdgpu-waves-per-eu` "1,MAX" cap (avoids EU over-subscription); 0 => "1,8"
    # (CDNA4 max waves/SIMD) = non-binding default. Literal per branch: the LLVM
    # passthrough attr needs a StringLiteral, not a computed string.
    comptime if waves_per_eu == 1:
        return "1,1".value
    elif waves_per_eu == 2:
        return "1,2".value
    elif waves_per_eu == 3:
        return "1,3".value
    elif waves_per_eu == 4:
        return "1,4".value
    elif waves_per_eu == 5:
        return "1,5".value
    elif waves_per_eu == 6:
        return "1,6".value
    elif waves_per_eu == 7:
        return "1,7".value
    else:
        return "1,8".value


struct PreShuffledBGroupedGEMM[
    cu_count: Int,
    wg_per_cu: Int = 2,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    xcd_stripe: Int = 8,
]:
    """Grouped GEMM for MXFP4 on AMD CDNA4 with pre-shuffled weights.

    This grouped GEMM operates on weights B that have been pre-shuffled into
    a layout enabling coalesced reads from shared memory and direct MFMA
    usage. It offers a persistent kernel (grid-stride over work tiles with
    XCD-aware work-group swizzling) and a direct kernel (one block per
    output tile, expert dispatched via `block_idx.z`), selected at launch
    time by the `persistent` comptime flag.

    Parameters:
        cu_count: Number of compute units on the target device.
        wg_per_cu: Work groups per compute unit (default 2).
        matrix_format: `f8f6f4` operand encoding for A and B (FP4 E2M1 by
            default). Both kernels below derive their fragment widths from
            it, so it must reach every `BlockScaledMatmulAMD_PreB` instantiation
            -- including the ones inside `MAX_THREADS_PER_BLOCK_METADATA`,
            or the launch bounds disagree with the body about num_threads.
        xcd_stripe: Size of the contiguous per-XCD run in `to_swizzled_idx`'s
            logical-index space (default 8). `0` is the escape hatch: it
            resolves to `wg_per_xcd`, recovering the fully-contiguous
            pre-fix mapping. See `to_swizzled_idx` for why this bounds the
            persistent grid's remainder-tile imbalance across XCDs.
    """

    comptime a_bits = Self.matrix_format.bits_per_element()
    comptime b_bits = Self.matrix_format.bits_per_element()
    comptime bits_per_element = Self.a_bits
    comptime lane_bytes = (32 * Self.a_bits) // 8
    comptime fmt_suffix: StaticString = (
        "e2m3" if (
            Self.matrix_format == CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3
        ) else (
            "e3m2" if (
                Self.matrix_format == CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
            ) else ""
        )
    )

    comptime num_xcd = 8
    comptime total_wg = Self.cu_count * Self.wg_per_cu
    comptime wg_per_xcd = Self.total_wg // Self.num_xcd
    comptime effective_xcd_stripe = (
        Self.wg_per_xcd if Self.xcd_stripe == 0 else Self.xcd_stripe
    )

    @always_inline
    @staticmethod
    def to_swizzled_idx(linear_idx: Int) -> Int:
        comptime assert (
            Self.wg_per_xcd % Self.effective_xcd_stripe == 0
        ), "wg_per_xcd must be a multiple of xcd_stripe"

        # If we have 10 blocks and 8 xcd's the block scheduler assigns
        # a block to the xcd in this round robin fashion

        # XCD:         0 1 2 3 4 5 6 7

        # block_idx.x: 0 1 2 3 4 5 6 7
        # continued:   8 9

        # blocks get assigned in a round robin fashion to xcd's
        # to make sure that blocks in the same xcd work on cache
        # local blocks we remap the id's in this fashion

        # XCD:         0 1 2 3 4 5 6 7

        # block_idx.x: 0 2 4 5 6 7 8 9
        # continued:   1 3

        # The persistent grid-stride loop hands the *low* logical-index band
        # `[0, total_tiles % total_wg)` one extra tile (each WG claims
        # `logical_wg, logical_wg + total_wg, ...`). A fully contiguous
        # per-XCD chunk (`xcd_linear_idx` ordered last, i.e. `xcd_stripe ==
        # wg_per_xcd`) can put that entire remainder band inside a handful of
        # XCDs' chunks, doubling half the chip's traffic while the other half
        # idles (see PreShuffledBGroupedGEMM callers' attribution). Striping
        # XCD assignment in blocks of `xcd_stripe` instead bounds that
        # imbalance to `xcd_stripe` tiles regardless of `wg_per_xcd`, at a
        # measured locality cost that grows as `xcd_stripe` shrinks (small
        # but real from `wg_per_xcd` down to `xcd_stripe=2`; much larger at
        # `xcd_stripe=1`). Which one wins net depends on the remainder as a
        # *fraction* of the tile count for the shape actually launched, not
        # on `xcd_stripe` alone — see the dispatcher callers' per-band notes.
        var xcd_idx = linear_idx % Self.num_xcd
        var xcd_linear_idx = linear_idx // Self.num_xcd
        var superblock_id = xcd_linear_idx // Self.effective_xcd_stripe
        var within_block = xcd_linear_idx % Self.effective_xcd_stripe
        return (
            superblock_id * (Self.num_xcd * Self.effective_xcd_stripe)
            + xcd_idx * Self.effective_xcd_stripe
            + within_block
        )

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(
                BlockScaledMatmulAMD_PreB[
                    BM=BM,
                    matrix_format=Self.matrix_format,
                    BN=BN,
                    BK_ELEMS=BK_ELEMS,
                    WN=WN,
                    b_prefetch=True,
                ].num_threads
            )
        )
    )
    @__name(
        t"mx_preb_pers_lb{Self.lane_bytes}{Self.fmt_suffix}_BM{BM}_BN{BN}_WN{WN}_BK{BK_ELEMS}_N{N}_KB{K_BYTES}{"_cds" if cluster_drain_sched else ""}{"_mc2" if mfma_cluster == 2 else ""}{"_pd3" if pipeline_depth == 3 else ""}{"_sg4" if scale_group == 4 else ""}{"_bas" if b_addr_split else ""}"
    )
    @__llvm_metadata(
        `llvm.amdgpu-waves-per-eu`=_waves_per_eu_attr[waves_per_eu]()
    )
    def persistent_kernel[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        out_dtype: DType,
        LayoutC: TensorLayout,
        LayoutA: TensorLayout,
        LayoutBPre: TensorLayout,
        LayoutSFA: TensorLayout,
        LayoutSFB: TensorLayout,
        AOffsetsLayout: TensorLayout,
        ExpertIdsLayout: TensorLayout,
        CEngine: TensorEngine,
        AEngine: TensorEngine,
        BPreEngine: TensorEngine,
        SFAEngine: TensorEngine,
        SFBEngine: TensorEngine,
        AOffsetsEngine: TensorEngine,
        ExpertIdsEngine: TensorEngine,
        N: Int,
        K_BYTES: Int,
        b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
        dram_to_lds: Bool = False,
        cluster_drain_sched: Bool = False,
        mfma_cluster: Int = 4,
        pipeline_depth: Int = 2,
        scale_group: Int = 1,
        b_addr_split: Bool = False,
        waves_per_eu: Int = 0,
    ](
        c_tensor: TileTensor[
            mut=True, out_dtype, LayoutC, MutAnyOrigin, Engine=CEngine
        ],
        a_tensor: TileTensor[.uint8, LayoutA, ImmutAnyOrigin, Engine=AEngine],
        b_pre_tensor: TileTensor[
            .uint8, LayoutBPre, ImmutAnyOrigin, Engine=BPreEngine
        ],
        sfa_tensor: TileTensor[
            .float8_e8m0fnu, LayoutSFA, ImmutAnyOrigin, Engine=SFAEngine
        ],
        sfb_tensor: TileTensor[
            .float8_e8m0fnu, LayoutSFB, ImmutAnyOrigin, Engine=SFBEngine
        ],
        a_offsets: TileTensor[
            mut=False,
            .uint32,
            AOffsetsLayout,
            ImmutAnyOrigin,
            Engine=AOffsetsEngine,
        ],
        expert_ids: TileTensor[
            mut=False,
            .int32,
            ExpertIdsLayout,
            ImmutAnyOrigin,
            Engine=ExpertIdsEngine,
        ],
        num_active_experts: Int32,
        max_padded_M: Int32,
    ):
        var _num_active_experts = Int(num_active_experts)
        var _max_padded_M = Int(max_padded_M)
        comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
        comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

        comptime Kernel = BlockScaledMatmulAMD_PreB[
            BM=BM,
            matrix_format=Self.matrix_format,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            b_prefetch=True,
            b_cache_policy=b_cache_policy,
            dram_to_lds=dram_to_lds,
            cluster_drain_sched=cluster_drain_sched,
            mfma_cluster=mfma_cluster,
            pipeline_depth=pipeline_depth,
            scale_group=scale_group,
            b_addr_split=b_addr_split,
        ]
        # K_SCALES (= K / 32) derived from A's K byte extent. The
        # preshuffled sfa_tensor's static shape is layout-dependent (i32-cell
        # vs uint8-byte views differ); A is canonically 2D so this is stable.
        comptime K_SCALES = a_tensor.static_shape[1] // (
            4 * Self.bits_per_element
        )
        comptime gx_n = ceildiv(N, BN)

        if N == 0 or _num_active_experts == 0:
            return

        var linear_wg = Int(block_idx.x)
        var logical_wg = Self.to_swizzled_idx(linear_wg)

        # grid stride loop over available work tiles using the logical WG ID
        # essentially each WG processes one tile per grid stride iteration
        # grid stride is the total number of WGs (Self.total_wg)

        # the next tile we want to work on for this WG
        var target_tile = logical_wg

        # the current tile in the grid stride loop, all WGs start at tile 0
        var current_tile = 0

        for expert_slot in range(_num_active_experts):
            var M = a_offsets[expert_slot + 1] - a_offsets[expert_slot]

            # No real work — skip this expert and don't update current_tile.
            # Skipped experts contribute zero tiles to the global counter,
            # so the invariant holds.
            if M == 0:
                continue

            var expert_id = expert_ids[expert_slot]

            # No real work — skip this expert and don't update current_tile.
            if expert_id == -1:
                continue

            # This expert has real work. Check if our target tile falls
            # within this expert's slice of the global counter.

            var m_count = ceildiv(Int(M), BM)
            var expert_work = m_count * gx_n
            var expert_end = current_tile + expert_work

            # Our target tile is beyond this expert's range — advance the
            # current_tile to the start of the next expert and continue.
            if target_tile >= expert_end:
                current_tile = expert_end
                continue

            # This expert has tile(s) for this WG. Hoist per-expert tile
            # descriptors out of the inner while loop.

            var a_start_row = a_offsets[expert_slot]
            # Preshuffled A-scales: fixed-stride slots. Expert e's chunk
            # starts at `e * max_padded_M` rows in `sfa_tensor`; each slot
            # is `max_padded_M * K_SCALES` bytes. The V# bound uses the
            # per-expert padded M (align_up(num_tokens, 32)) so scale reads
            # never cross past this expert's real-plus-pad-to-32 rows.
            # Producers (standalone preshuffle OR the fused ep_wait/fused_silu
            # stores) write only real-token scales; the pad-row matmul outputs
            # are discarded after the gather, so the slot tail is not
            # zero-filled.
            var sfa_start_row = UInt32(expert_slot * _max_padded_M)
            var sfa_padded_M = align_up(Int(M), 32)

            var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
            comptime A_K_BYTES = a_tensor.static_shape[1]
            var a_ptr = a_tensor.ptr + a_start_row * UInt32(A_K_BYTES)
            var b_pre_ptr = b_pre_tensor.ptr + expert_id * Int32(N) * Int32(
                K_BYTES
            )
            var sfa_ptr = sfa_tensor.ptr + sfa_start_row * UInt32(K_SCALES)
            var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(
                K_SCALES
            )

            # C's V# bound is the real `M` (not `align_up(M, 32)`): the matmul
            # stores only real-token rows, so the pad rows past `M` are never
            # written. That is what makes the uninitialized pad scale cells in
            # `sfa` (whose V# DOES extend to `sfa_padded_M`) safe — the C rows
            # they would feed are OOB-clamped and discarded.
            var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
            var a_tile = TileTensor(
                a_ptr, row_major(Coord(Int(M), Idx[A_K_BYTES]))
            )
            var b_pre_tile = TileTensor(
                b_pre_ptr, row_major(Coord(Idx[1], Idx[N * K_BYTES]))
            )
            # NOTE: the 2D `[MN_padded, K_SCALES]` shape is a fiction —
            # the buffer is in scale-4d byte order, not row-major. Only
            # the byte count (= product) is consulted, by V# construction
            # inside `PreshuffledScaleLoader`. The kernel uses
            # `Shuffler.scale_4d_byte_off` for actual addressing.
            # TODO: switch to a flat 1D uint8 tile so the layout stops
            # mis-suggesting row-major bytes.
            var sfa_tile = TileTensor(
                sfa_ptr, row_major(Coord(Int(sfa_padded_M), Idx[K_SCALES]))
            )
            var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

            # An expert may span multiple grid strides — iterate by stride
            # until we exit this expert's range.

            while target_tile < expert_end:
                var local = target_tile - current_tile
                # M-fast within (expert, n_tile): n is outer, m is inner.
                var n_tile = local // m_count
                var m_tile = local - n_tile * m_count

                Kernel.run[
                    out_dtype,
                    type_of(c_tile).LayoutType,
                    type_of(a_tile).LayoutType,
                    type_of(b_pre_tile).LayoutType,
                    type_of(sfa_tile).LayoutType,
                    type_of(sfb_tile).LayoutType,
                    type_of(c_tile).Engine,
                    type_of(a_tile).Engine,
                    type_of(b_pre_tile).Engine,
                    type_of(sfa_tile).Engine,
                    type_of(sfb_tile).Engine,
                    N,
                    K_BYTES,
                ](
                    c_tile,
                    a_tile,
                    b_pre_tile,
                    sfa_tile,
                    sfb_tile,
                    Int(n_tile),
                    Int(m_tile),
                )

                # advance to this WG's next claim in the global counter
                target_tile += Self.total_wg

            current_tile = expert_end

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(
                BlockScaledMatmulAMD_PreB[
                    BM=BM,
                    matrix_format=Self.matrix_format,
                    BN=BN,
                    BK_ELEMS=BK_ELEMS,
                    WN=WN,
                    b_prefetch=True,
                ].num_threads
            )
        )
    )
    @__name(
        t"mx_preb_lb{Self.lane_bytes}{Self.fmt_suffix}_BM{BM}_BN{BN}_WN{WN}_BK{BK_ELEMS}_N{N}_KB{K_BYTES}{"_cds" if cluster_drain_sched else ""}{"_mc2" if mfma_cluster == 2 else ""}{"_pd3" if pipeline_depth == 3 else ""}{"_sg4" if scale_group == 4 else ""}{"_bas" if b_addr_split else ""}"
    )
    @__llvm_metadata(
        `llvm.amdgpu-waves-per-eu`=_waves_per_eu_attr[waves_per_eu]()
    )
    def kernel[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        out_dtype: DType,
        LayoutC: TensorLayout,
        LayoutA: TensorLayout,
        LayoutBPre: TensorLayout,
        LayoutSFA: TensorLayout,
        LayoutSFB: TensorLayout,
        AOffsetsLayout: TensorLayout,
        ExpertIdsLayout: TensorLayout,
        CEngine: TensorEngine,
        AEngine: TensorEngine,
        BPreEngine: TensorEngine,
        SFAEngine: TensorEngine,
        SFBEngine: TensorEngine,
        AOffsetsEngine: TensorEngine,
        ExpertIdsEngine: TensorEngine,
        N: Int,
        K_BYTES: Int,
        b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
        dram_to_lds: Bool = False,
        cluster_drain_sched: Bool = False,
        mfma_cluster: Int = 4,
        pipeline_depth: Int = 2,
        scale_group: Int = 1,
        b_addr_split: Bool = False,
        waves_per_eu: Int = 0,
    ](
        c_tensor: TileTensor[
            mut=True, out_dtype, LayoutC, MutAnyOrigin, Engine=CEngine
        ],
        a_tensor: TileTensor[.uint8, LayoutA, ImmutAnyOrigin, Engine=AEngine],
        b_pre_tensor: TileTensor[
            .uint8, LayoutBPre, ImmutAnyOrigin, Engine=BPreEngine
        ],
        sfa_tensor: TileTensor[
            .float8_e8m0fnu, LayoutSFA, ImmutAnyOrigin, Engine=SFAEngine
        ],
        sfb_tensor: TileTensor[
            .float8_e8m0fnu, LayoutSFB, ImmutAnyOrigin, Engine=SFBEngine
        ],
        a_offsets: TileTensor[
            mut=False,
            .uint32,
            AOffsetsLayout,
            ImmutAnyOrigin,
            Engine=AOffsetsEngine,
        ],
        expert_ids: TileTensor[
            mut=False,
            .int32,
            ExpertIdsLayout,
            ImmutAnyOrigin,
            Engine=ExpertIdsEngine,
        ],
        num_active_experts: Int32,
        max_padded_M: Int32,
    ):
        var _max_padded_M = Int(max_padded_M)
        comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
        comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

        comptime Kernel = BlockScaledMatmulAMD_PreB[
            BM=BM,
            matrix_format=Self.matrix_format,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            b_prefetch=True,
            b_cache_policy=b_cache_policy,
            dram_to_lds=dram_to_lds,
            cluster_drain_sched=cluster_drain_sched,
            mfma_cluster=mfma_cluster,
            pipeline_depth=pipeline_depth,
            scale_group=scale_group,
            b_addr_split=b_addr_split,
        ]
        # K_SCALES (= K / 32) derived from A's K byte extent. The
        # preshuffled sfa_tensor's static shape is layout-dependent (i32-cell
        # vs uint8-byte views differ); A is canonically 2D so this is stable.
        comptime K_SCALES = a_tensor.static_shape[1] // (
            4 * Self.bits_per_element
        )

        var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
        if M == 0 or N == 0:
            return
        var expert_id = expert_ids[block_idx.z]
        if expert_id == -1:
            return
        if block_idx.y >= ceildiv(Int(M), BM):
            return

        var a_start_row = a_offsets[block_idx.z]
        # Preshuffled A-scales: fixed-stride slot at e * max_padded_M.
        # Per-expert tight V# bound = align_up(num_tokens, 32).
        var sfa_start_row = UInt32(Int(block_idx.z) * _max_padded_M)
        var sfa_padded_M = align_up(Int(M), 32)

        var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
        comptime A_K_BYTES = a_tensor.static_shape[1]
        var a_ptr = a_tensor.ptr + a_start_row * UInt32(A_K_BYTES)
        var b_pre_ptr = b_pre_tensor.ptr + expert_id * Int32(N) * Int32(K_BYTES)
        var sfa_ptr = sfa_tensor.ptr + sfa_start_row * UInt32(K_SCALES)
        var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(K_SCALES)

        var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
        var a_tile = TileTensor(a_ptr, row_major(Coord(Int(M), Idx[A_K_BYTES])))
        var b_pre_tile = TileTensor(
            b_pre_ptr, row_major(Coord(Idx[1], Idx[N * K_BYTES]))
        )
        # See persistent_kernel for why this 2D shape is a fiction.
        # TODO: switch to a flat 1D uint8 tile.
        var sfa_tile = TileTensor(
            sfa_ptr, row_major(Coord(Int(sfa_padded_M), Idx[K_SCALES]))
        )
        var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

        Kernel.run[
            out_dtype,
            type_of(c_tile).LayoutType,
            type_of(a_tile).LayoutType,
            type_of(b_pre_tile).LayoutType,
            type_of(sfa_tile).LayoutType,
            type_of(sfb_tile).LayoutType,
            type_of(c_tile).Engine,
            type_of(a_tile).Engine,
            type_of(b_pre_tile).Engine,
            type_of(sfa_tile).Engine,
            type_of(sfb_tile).Engine,
            N,
            K_BYTES,
        ](
            c_tile,
            a_tile,
            b_pre_tile,
            sfa_tile,
            sfb_tile,
            Int(block_idx.x),
            Int(block_idx.y),
        )

    # --------------------------------------------------------------------- #
    # Launch helper — picks persistent vs direct dispatch via comptime flag.
    # --------------------------------------------------------------------- #

    @staticmethod
    def launch[
        BM: Int,
        BN: Int,
        BK_ELEMS: Int,
        WN: Int,
        persistent: Bool,
        b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
        dram_to_lds: Bool = False,
        cluster_drain_sched: Bool = False,
        mfma_cluster: Int = 4,
        pipeline_depth: Int = 2,
        scale_group: Int = 1,
        b_addr_split: Bool = False,
        waves_per_eu: Int = 0,
        static_grid_z: Bool = False,
    ](
        c: TileTensor[mut=True, ...],
        a: TileTensor[.uint8, ...],
        b_pre: TileTensor[.uint8, ...],
        a_scales: TileTensor[.float8_e8m0fnu, ...],
        b_scales: TileTensor[.float8_e8m0fnu, ...],
        a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
        expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
        max_num_tokens_per_expert: Int,
        num_active_experts: Int,
        ctx: DeviceContext,
        grid_m_cap: Int = -1,
    ) raises:
        comptime MatmulDeviceFunctionType = BlockScaledMatmulAMD_PreB[
            BM=BM,
            matrix_format=Self.matrix_format,
            BN=BN,
            BK_ELEMS=BK_ELEMS,
            WN=WN,
            b_prefetch=True,
            b_cache_policy=b_cache_policy,
            dram_to_lds=dram_to_lds,
            cluster_drain_sched=cluster_drain_sched,
            mfma_cluster=mfma_cluster,
            pipeline_depth=pipeline_depth,
            scale_group=scale_group,
            b_addr_split=b_addr_split,
        ]

        comptime N = c.static_shape[1]
        comptime K_BYTES = (
            b_pre.static_shape[1] // N if b_pre.flat_rank
            == 2 else b_pre.static_shape[2]
        )

        var a_i = TileTensor(
            a.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            a.layout,
        )
        var b_pre_i = TileTensor(
            b_pre.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            b_pre.layout,
        )
        var a_scales_i = TileTensor(
            a_scales.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            a_scales.layout,
        )
        var b_scales_i = TileTensor(
            b_scales.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            b_scales.layout,
        )
        var a_off_i = TileTensor(
            a_offsets.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            a_offsets.layout,
        )
        var expert_ids_i = TileTensor(
            expert_ids.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
            expert_ids.layout,
        )

        if max_num_tokens_per_expert == 0:
            return

        # max_padded_M is the per-expert slot stride for the preshuffled
        # A-scale buffer (set by the upstream preshuffle launch). The
        # caller's max_num_tokens_per_expert must match what the
        # preshuffle was sized for.
        var max_padded_M = align_up(max_num_tokens_per_expert, 32)

        comptime out_dtype = type_of(c).dtype

        comptime if persistent:
            comptime kernel = Self.persistent_kernel[
                BM,
                BN,
                BK_ELEMS,
                WN,
                out_dtype,
                type_of(c).LayoutType,
                type_of(a_i).LayoutType,
                type_of(b_pre_i).LayoutType,
                type_of(a_scales_i).LayoutType,
                type_of(b_scales_i).LayoutType,
                type_of(a_off_i).LayoutType,
                type_of(expert_ids_i).LayoutType,
                type_of(c).Engine,
                type_of(a_i).Engine,
                type_of(b_pre_i).Engine,
                type_of(a_scales_i).Engine,
                type_of(b_scales_i).Engine,
                type_of(a_off_i).Engine,
                type_of(expert_ids_i).Engine,
                N,
                K_BYTES,
                b_cache_policy,
                dram_to_lds,
                cluster_drain_sched,
                mfma_cluster,
                pipeline_depth,
                scale_group,
                b_addr_split,
                waves_per_eu,
            ]
            ctx.enqueue_function[kernel](
                c,
                a_i,
                b_pre_i,
                a_scales_i,
                b_scales_i,
                a_off_i,
                expert_ids_i,
                Int32(num_active_experts),
                Int32(max_padded_M),
                grid_dim=(Self.total_wg, 1, 1),
                block_dim=MatmulDeviceFunctionType.num_threads,
            )
        else:
            comptime kernel = Self.kernel[
                BM,
                BN,
                BK_ELEMS,
                WN,
                out_dtype,
                type_of(c).LayoutType,
                type_of(a_i).LayoutType,
                type_of(b_pre_i).LayoutType,
                type_of(a_scales_i).LayoutType,
                type_of(b_scales_i).LayoutType,
                type_of(a_off_i).LayoutType,
                type_of(expert_ids_i).LayoutType,
                type_of(c).Engine,
                type_of(a_i).Engine,
                type_of(b_pre_i).Engine,
                type_of(a_scales_i).Engine,
                type_of(b_scales_i).Engine,
                type_of(a_off_i).Engine,
                type_of(expert_ids_i).Engine,
                N,
                K_BYTES,
                b_cache_policy,
                dram_to_lds,
                cluster_drain_sched,
                mfma_cluster,
                pipeline_depth,
                scale_group,
                b_addr_split,
                waves_per_eu,
            ]
            # grid.y cap: decode cap when supplied, else full A-scale stride.
            var m_cap = (
                grid_m_cap if grid_m_cap > 0 else max_num_tokens_per_expert
            )
            # grid.z: comptime local-expert count (capture-time constant) when
            # static_grid_z, else runtime num_active_experts.
            comptime n_local_experts = b_pre.static_shape[0]
            var grid_z = (
                n_local_experts if static_grid_z else num_active_experts
            )
            ctx.enqueue_function[kernel](
                c,
                a_i,
                b_pre_i,
                a_scales_i,
                b_scales_i,
                a_off_i,
                expert_ids_i,
                Int32(num_active_experts),
                Int32(max_padded_M),
                grid_dim=(
                    ceildiv(N, BN),
                    ceildiv(m_cap, BM),
                    grid_z,
                ),
                block_dim=MatmulDeviceFunctionType.num_threads,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            BlockScaledMatmulAMD[
                BM=BM,
                BN=BN,
                BK_ELEMS=BK_ELEMS,
                WM=WM,
                WN=WN,
                matrix_format=matrix_format,
            ].num_threads
        )
    )
)
@__name(t"mxfp4_grouped_{out_dtype}_BM{BM}_BN{BN}_WM{WM}_WN{WN}_BK{BK_ELEMS}")
def block_scaled_grouped_matmul_amd_kernel[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WM: Int,
    WN: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat,
    out_dtype: DType,
    LayoutC: TensorLayout,
    LayoutA: TensorLayout,
    LayoutB: TensorLayout,
    LayoutSFA: TensorLayout,
    LayoutSFB: TensorLayout,
    AOffsetsLayout: TensorLayout,
    ExpertIdsLayout: TensorLayout,
    CEngine: TensorEngine,
    AEngine: TensorEngine,
    BEngine: TensorEngine,
    SFAEngine: TensorEngine,
    SFBEngine: TensorEngine,
    AOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
](
    c_tensor: TileTensor[
        mut=True, out_dtype, LayoutC, MutAnyOrigin, Engine=CEngine
    ],
    a_tensor: TileTensor[.uint8, LayoutA, ImmutAnyOrigin, Engine=AEngine],
    b_tensor: TileTensor[.uint8, LayoutB, ImmutAnyOrigin, Engine=BEngine],
    sfa_tensor: TileTensor[
        .float8_e8m0fnu, LayoutSFA, ImmutAnyOrigin, Engine=SFAEngine
    ],
    sfb_tensor: TileTensor[
        .float8_e8m0fnu, LayoutSFB, ImmutAnyOrigin, Engine=SFBEngine
    ],
    a_offsets: TileTensor[
        mut=False,
        .uint32,
        AOffsetsLayout,
        ImmutAnyOrigin,
        Engine=AOffsetsEngine,
    ],
    expert_ids: TileTensor[
        mut=False,
        .int32,
        ExpertIdsLayout,
        ImmutAnyOrigin,
        Engine=ExpertIdsEngine,
    ],
    num_active_experts: Int32,
):
    """MXFP4 grouped matmul kernel with expert dispatch via block_idx.z.

    b_tensor and sfb_tensor are flattened from 3D to 2D:
      b: [num_experts*N, K//2], sfb: [num_experts*N, K//32]

    Parameters:
        BM: Block tile rows (output M per block).
        BN: Block tile cols (output N per block).
        BK_ELEMS: Block tile K in logical FP4 elements.
        WM: Warp tile rows; `BM` must be divisible by `WM`.
        WN: Warp tile cols; `BN` must be divisible by `WN`.
        matrix_format: `f8f6f4` operand encoding for A and B; the tile byte
            widths are derived from it.
        out_dtype: Element type of the output tensor `c_tensor`.
        LayoutC: Compile-time layout of the output tensor `c_tensor`.
        LayoutA: Compile-time layout of the A operand `a_tensor`.
        LayoutB: Compile-time layout of the B operand `b_tensor`.
        LayoutSFA: Compile-time layout of the A scales tensor `sfa_tensor`.
        LayoutSFB: Compile-time layout of the B scales tensor `sfb_tensor`.
        AOffsetsLayout: Compile-time layout of the token offsets tensor
            `a_offsets`.
        ExpertIdsLayout: Compile-time layout of the expert indices tensor
            `expert_ids`.
        CEngine: Engine of the output tensor `c_tensor`.
        AEngine: Engine of the A operand `a_tensor`.
        BEngine: Engine of the B operand `b_tensor`.
        SFAEngine: Engine of the A scales tensor `sfa_tensor`.
        SFBEngine: Engine of the B scales tensor `sfb_tensor`.
        AOffsetsEngine: Engine of the token offsets tensor `a_offsets`.
        ExpertIdsEngine: Engine of the expert indices tensor
            `expert_ids`.

    Args:
        c_tensor: Output matrix `[total_tokens, N]` of dtype `out_dtype`,
            indexed by per-expert token offsets.
        a_tensor: Packed activations `[total_tokens, K//2]` uint8, two
            MXFP4 nibbles per byte.
        b_tensor: Expert weights `[num_experts*N, K//2]` uint8, flattened
            from 3D `[num_experts, N, K//2]`, two MXFP4 nibbles per byte.
        sfa_tensor: A block scales `[total_tokens, K//32]` as
            `float8_e8m0fnu`, one scale per 32 MXFP4 elements.
        sfb_tensor: B block scales `[num_experts*N, K//32]` as
            `float8_e8m0fnu`, flattened from 3D `[num_experts, N, K//32]`,
            one scale per 32 MXFP4 elements.
        a_offsets: Token offsets `[num_active_experts+1]` uint32; expert
            slot `e` spans rows `a_offsets[e]` to `a_offsets[e+1]`.
        expert_ids: Expert indices `[num_active_experts]` int32;
            `expert_ids[slot]` selects the expert weight slice for that
            slot.
        num_active_experts: Number of active expert slots dispatched via
            `block_idx.z`.
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        matrix_format=matrix_format,
    ]
    comptime N = c_tensor.static_shape[1]
    comptime K_BYTES = b_tensor.static_shape[1]  # K//2
    comptime K_SCALES = sfa_tensor.static_shape[1]  # K//32

    var M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    if M == 0 or N == 0:
        return
    var expert_id = expert_ids[block_idx.z]
    var a_start_row = a_offsets[block_idx.z]

    if expert_id == -1:
        return

    # Grid Y is ceildiv(max_tokens_per_expert, BM); skip blocks past this
    # expert's M rows (other experts may be shorter than max_tokens).
    if block_idx.y >= ceildiv(Int(M), BM):
        return

    var c_ptr = c_tensor.ptr + a_start_row * UInt32(N)
    comptime A_K_BYTES = a_tensor.static_shape[1]
    var a_ptr = a_tensor.ptr + a_start_row * UInt32(A_K_BYTES)
    var b_ptr = b_tensor.ptr + expert_id * Int32(N) * Int32(K_BYTES)
    var sfa_ptr = sfa_tensor.ptr + a_start_row * UInt32(K_SCALES)
    var sfb_ptr = sfb_tensor.ptr + expert_id * Int32(N) * Int32(K_SCALES)

    var c_tile = TileTensor(c_ptr, row_major(Coord(Int(M), Idx[N])))
    var a_tile = TileTensor(a_ptr, row_major(Coord(Int(M), Idx[A_K_BYTES])))
    var b_tile = TileTensor(b_ptr, row_major[N, K_BYTES]())
    var sfa_tile = TileTensor(sfa_ptr, row_major(Coord(Int(M), Idx[K_SCALES])))
    var sfb_tile = TileTensor(sfb_ptr, row_major[N, K_SCALES]())

    Kernel.run[
        out_dtype,
        type_of(c_tile).LayoutType,
        type_of(a_tile).LayoutType,
        type_of(b_tile).LayoutType,
        type_of(sfa_tile).LayoutType,
        type_of(sfb_tile).LayoutType,
        type_of(c_tile).Engine,
        type_of(a_tile).Engine,
        type_of(b_tile).Engine,
        type_of(sfa_tile).Engine,
        type_of(sfb_tile).Engine,
    ](c_tile, a_tile, b_tile, sfa_tile, sfb_tile)


# ===----------------------------------------------------------------------=== #
# Public entry point
# ===----------------------------------------------------------------------=== #


def block_scaled_grouped_matmul_amd[
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[.uint8, ...],
    b: TileTensor[.uint8, ...],
    a_scales: TileTensor[.float8_e8m0fnu, ...],
    b_scales: TileTensor[.float8_e8m0fnu, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch native MXFP4 grouped matmul on AMD CDNA4.

    Grouped matmul for MoE: dispatches one expert per block_idx.z,
    using BlockScaledMatmulAMD.run per expert slice.

    Args:
        c: Output [total_tokens, N].
        a: Packed activations [total_tokens, K//2] uint8.
        b: Expert weights [num_experts, N, K//2] uint8.
        a_scales: Activation scales [total_tokens, K//32] float8_e8m0fnu.
        b_scales: Weight scales [num_experts, N, K//32] float8_e8m0fnu.
        a_offsets: Token offsets [num_active_experts+1] uint32.
        expert_ids: Expert indices [num_active_experts] int32.
        max_num_tokens_per_expert: Maximum token count for any active expert.
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    comptime assert b.flat_rank == 3, "b must be rank 3"
    comptime assert b_scales.flat_rank == 3, "b_scales must be rank 3"
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    comptime K_BYTES = b.static_shape[2]
    comptime bk_512_bytes = (512 * matrix_format.bits_per_element()) // 8
    comptime can_use_bk_512 = (
        K_BYTES >= bk_512_bytes and K_BYTES % bk_512_bytes == 0
    )

    comptime is_fp6 = matrix_format.bits_per_element() == 6
    comptime BM_wide = 96 if is_fp6 else 64
    comptime BN_wide = 64 if is_fp6 else 128
    comptime WM_wide = 32 if is_fp6 else 64
    comptime BM_deep = 96 if is_fp6 else 128
    comptime BN_deep = 64 if is_fp6 else 128
    comptime WM_deep = 32 if is_fp6 else 64

    if max_num_tokens_per_expert <= 64:
        comptime if can_use_bk_512:
            _launch_block_scaled_grouped[
                BM=BM_wide,
                BN=BN_wide,
                BK_ELEMS=512,
                WM=WM_wide,
                WN=64,
                matrix_format=matrix_format,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                expert_ids,
                max_num_tokens_per_expert,
                num_active_experts,
                ctx,
            )
            return

    _launch_block_scaled_grouped[
        BM=BM_deep,
        BN=BN_deep,
        BK_ELEMS=128,
        WM=WM_deep,
        WN=64,
        matrix_format=matrix_format,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_per_expert,
        num_active_experts,
        ctx,
    )


def _launch_block_scaled_grouped[
    BM: Int,
    BN: Int,
    BK_ELEMS: Int,
    WM: Int,
    WN: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[.uint8, ...],
    b: TileTensor[.uint8, ...],
    a_scales: TileTensor[.float8_e8m0fnu, ...],
    b_scales: TileTensor[.float8_e8m0fnu, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Instantiates and launches the grouped MXFP4 kernel."""
    comptime Kernel = BlockScaledMatmulAMD[
        BM=BM,
        BN=BN,
        BK_ELEMS=BK_ELEMS,
        WM=WM,
        WN=WN,
        matrix_format=matrix_format,
    ]
    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K_BYTES = b.static_shape[2]  # K//2

    comptime num_experts_sf = b_scales.static_shape[0]
    comptime N_sf = b_scales.static_shape[1]
    comptime K_SCALES = b_scales.static_shape[2]  # K//32

    var a_i = TileTensor(
        a.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        a.layout,
    )
    var b_2d = TileTensor(
        b.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        row_major[num_experts * N, K_BYTES](),
    )
    var a_scales_i = TileTensor(
        a_scales.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        a_scales.layout,
    )
    var sfb_2d = TileTensor(
        b_scales.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        row_major[num_experts_sf * N_sf, K_SCALES](),
    )
    var a_off_i = TileTensor(
        a_offsets.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        a_offsets.layout,
    )
    var expert_ids_i = TileTensor(
        expert_ids.ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
        expert_ids.layout,
    )

    if max_num_tokens_per_expert == 0:
        return

    comptime out_dtype = type_of(c).dtype
    comptime kernel = block_scaled_grouped_matmul_amd_kernel[
        BM,
        BN,
        BK_ELEMS,
        WM,
        WN,
        matrix_format,
        out_dtype,
        type_of(c).LayoutType,
        type_of(a_i).LayoutType,
        type_of(b_2d).LayoutType,
        type_of(a_scales_i).LayoutType,
        type_of(sfb_2d).LayoutType,
        type_of(a_off_i).LayoutType,
        type_of(expert_ids_i).LayoutType,
        type_of(c).Engine,
        type_of(a_i).Engine,
        type_of(b_2d).Engine,
        type_of(a_scales_i).Engine,
        type_of(sfb_2d).Engine,
        type_of(a_off_i).Engine,
        type_of(expert_ids_i).Engine,
    ]

    ctx.enqueue_function[kernel](
        c,
        a_i,
        b_2d,
        a_scales_i,
        sfb_2d,
        a_off_i,
        expert_ids_i,
        Int32(num_active_experts),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=Kernel.num_threads,
    )


def block_scaled_grouped_matmul_amd_preb[
    lane_bytes: Int = 0, fp6_format: Int = 0
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[.uint8, ...],
    b_pre: TileTensor[.uint8, ...],
    a_scales: TileTensor[.float8_e8m0fnu, ...],
    b_scales: TileTensor[.float8_e8m0fnu, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
    estimated_total_m: Int = 0,
    decode_grid_m_cap: Int = -1,
    decode_grid_m_rows: Int = 0,
) raises:
    """Launches grouped MXFP4 matmul on AMD CDNA4 with pre-shuffled weights.

    Dispatches to `PreShuffledBGroupedGEMM` with per-shape and per-M-band
    tuned tile configurations, choosing between persistent and direct kernel
    launch based on the estimated total token count. Currently restricted to
    MI355X and requires packed K (K // 2) to be at least 256 and divisible by
    256; smaller K should use `block_scaled_grouped_matmul_amd` instead.

    Parameters:
        lane_bytes: Operand bytes per lane per MFMA — 16 (MXFP4), 24 (MXFP6)
            or 32 (MXFP8). 0 (default) infers it from the tensors, since every
            format presents as `uint8` and a wrong value silently corrupts
            `K_SCALES`. Also part of the band dispatch key: `(N, packed_K)`
            alone does not identify a shape across formats.
        fp6_format: 0 selects E2M3, 1 selects E3M2. Ignored unless
            `lane_bytes` resolves to 24 — both FP6 encodings put 24 bytes in a
            lane, so the byte count cannot choose between them.

    Args:
        c: Output tensor [total_tokens, N].
        a: Packed activations [total_tokens, K//2] uint8 (MXFP4) or
            [total_tokens, K] uint8 (MXFP8).
        b_pre: Pre-shuffled expert weights, rank-2 flat or rank-3
            [num_experts, N, K//2] uint8.
        a_scales: Activation scales [total_tokens, K//32] float8_e8m0fnu.
        b_scales: Weight scales [num_experts, N, K//32] float8_e8m0fnu.
        a_offsets: Token offsets [num_active_experts+1] uint32.
        expert_ids: Expert indices [num_active_experts] int32.
        max_num_tokens_per_expert: Maximum token count for any active expert.
        num_active_experts: Number of active experts.
        ctx: Device context.
        estimated_total_m: Estimated total tokens across all experts, used
            to select the tuned kernel band (default 0).
        decode_grid_m_cap: Decode-band gate (the production max batch size)
            selecting the direct kernel over the persistent one; not a bound.
        decode_grid_m_rows: Rows grid.y must cover per expert on the decode
            bands. 0 falls back to the full A-scale stride.
    """

    comptime assert (
        b_pre.flat_rank == 2 or b_pre.flat_rank == 3
    ), "b_pre must be rank-2 (flat) or rank-3 ([E, N, K_BYTES])"
    comptime num_experts = b_pre.static_shape[0]
    comptime m_threshold = 4096

    comptime b_per_expert_bytes = (
        b_pre.static_shape[1] if b_pre.flat_rank
        == 2 else b_pre.static_shape[1] * b_pre.static_shape[2]
    )

    comptime N = c.static_shape[1]
    comptime packed_K = a.static_shape[1]

    comptime K_LOGICAL = a_scales.static_shape[1] * 32
    comptime _INFERRED_LB = (
        32 if packed_K
        == K_LOGICAL else (
            24 if packed_K * 4
            == K_LOGICAL * 3 else (16 if packed_K * 2 == K_LOGICAL else 0)
        )
    )
    comptime assert _INFERRED_LB != 0, (
        "a's K byte extent must be K (MXFP8), K*3/4 (MXFP6) or K/2 (MXFP4)"
        " where K is a_scales' K extent * 32; the pair is what identifies the"
        " format"
    )
    comptime LB = _INFERRED_LB if lane_bytes == 0 else lane_bytes
    comptime assert LB == 16 or LB == 24 or LB == 32, (
        "lane_bytes (inferred or explicit) must be 16 (MXFP4), 24 (MXFP6) or"
        " 32 (MXFP8)"
    )
    comptime assert (
        lane_bytes == 0 or lane_bytes == _INFERRED_LB
    ), "explicit lane_bytes disagrees with the shapes of a / a_scales"
    comptime assert (
        fp6_format == 0 or fp6_format == 1
    ), "fp6_format must be 0 (E2M3) or 1 (E3M2)"

    comptime assert (
        b_per_expert_bytes == N * packed_K
    ), "b_pre shape mismatch with c.N and a.K"

    # TODO: drop once we generalize the persistent grid + cu_count derivation
    # across other AMD CDNA parts.
    comptime assert (
        ctx.default_device_info == MI355X
    ), "preb path currently only supports MI355X"

    comptime _K_GRANULE = 256 if LB == 32 else 512
    comptime assert K_LOGICAL >= _K_GRANULE and K_LOGICAL % _K_GRANULE == 0, (
        "block_scaled_grouped_matmul_amd_preb requires a logical K that is a"
        " nonzero multiple of 512 (MXFP4/MXFP6) or 256 (MXFP8); smaller K"
        " should use the non-preb path (block_scaled_grouped_matmul_amd)"
        " instead."
    )

    # One launch per band; only the comptime config differs, so capture the
    # runtime args once and let each band be a single line.
    @__parameter
    def run_kernel[
        BM: Int,
        BN: Int,
        BK: Int,
        WN: Int,
        persistent: Bool,
        b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
        wg_per_cu: Int = 2,
        use_decode_cap: Bool = False,
        pipeline_depth: Int = 2,
        cluster_drain_sched: Bool = False,
        mfma_cluster: Int = 4,
        scale_group: Int = 1,
        b_addr_split: Bool = False,
        waves_per_eu: Int = 0,
    ]() raises:
        # The kernels' `@__name` spells only these values, and the schedule
        # knobs are not otherwise part of the symbol -- a new value here would
        # make two bands share a symbol and silently alias in an .amdgcn dump
        # or a rocprofv3 profile. Extend the suffix table alongside the value.
        comptime assert (
            mfma_cluster in (2, 4)
            and pipeline_depth in (2, 3)
            and scale_group in (1, 4)
        ), "add the new knob value to the kernels' @__name suffix table"

        # `decode_grid_m_cap` gates the band in pairs/rank; it does not bound
        # rows per expert, so grid.y takes the caller's row bound instead.
        var grid_m_cap = decode_grid_m_rows if use_decode_cap else -1
        PreShuffledBGroupedGEMM[
            cu_count=ctx.default_device_info.sm_count,
            wg_per_cu=wg_per_cu,
            matrix_format=(
                CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1 if LB
                == 16 else (
                    CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3 if LB
                    == 32 else (
                        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3 if fp6_format
                        == 0 else CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
                    )
                )
            ),
        ].launch[
            BM=BM,
            BN=BN,
            BK_ELEMS=BK,
            WN=WN,
            persistent=persistent,
            b_cache_policy=b_cache_policy,
            pipeline_depth=pipeline_depth,
            cluster_drain_sched=cluster_drain_sched,
            mfma_cluster=mfma_cluster,
            scale_group=scale_group,
            b_addr_split=b_addr_split,
            waves_per_eu=waves_per_eu,
            static_grid_z=use_decode_cap,
        ](
            c,
            a,
            b_pre,
            a_scales,
            b_scales,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
            grid_m_cap,
        )

    # Autotune entry point, mirroring the AMD dense-matmul hook in
    # `linalg/matmul/gpu/__init__.mojo`. Bypasses every per-shape band below
    # for whichever shape is being compiled -- shape-agnostic by
    # construction, since it only reads `TUNE_*` defines and forwards them to
    # `run_kernel`. Each knob defaults to that band table's own fallback
    # value; `AUTOTUNING_MODE` defaults off, so this never fires in
    # production.
    comptime if get_defined_bool["AUTOTUNING_MODE", False]():
        comptime _tune_bm = get_defined_int["TUNE_BM", 64]()
        comptime _tune_bn = get_defined_int["TUNE_BN", 128]()
        comptime _tune_wn = get_defined_int["TUNE_WN", 32]()
        comptime _tune_bk = get_defined_int["TUNE_BK", 256]()
        comptime _tune_persistent = get_defined_bool["TUNE_PERSISTENT", False]()
        comptime _tune_waves_per_eu = get_defined_int["TUNE_WAVES_PER_EU", 0]()
        comptime _tune_wg_per_cu = get_defined_int["TUNE_WG_PER_CU", 2]()
        comptime _tune_cds = get_defined_bool[
            "TUNE_CLUSTER_DRAIN_SCHED", False
        ]()
        comptime _tune_mfma_cluster = get_defined_int["TUNE_MFMA_CLUSTER", 4]()
        comptime _tune_pipeline_depth = get_defined_int[
            "TUNE_PIPELINE_DEPTH", 2
        ]()
        comptime _tune_scale_group = get_defined_int["TUNE_SCALE_GROUP", 1]()
        comptime _tune_b_addr_split = get_defined_bool[
            "TUNE_B_ADDR_SPLIT", False
        ]()
        return run_kernel[
            _tune_bm,
            _tune_bn,
            _tune_bk,
            _tune_wn,
            _tune_persistent,
            cluster_drain_sched=_tune_cds,
            mfma_cluster=_tune_mfma_cluster,
            pipeline_depth=_tune_pipeline_depth,
            scale_group=_tune_scale_group,
            b_addr_split=_tune_b_addr_split,
            wg_per_cu=_tune_wg_per_cu,
            waves_per_eu=_tune_waves_per_eu,
        ]()

    # Per-(shape, M-band) tuned picks: persistent decode -> direct prefill at
    # etm >= m_threshold; STREAMING on the BN128 mid/upper decode bands.
    comptime STREAM = CacheOperation.STREAMING
    var etm = estimated_total_m

    comptime if LB == 16 and N == 4096 and packed_K == (7168 // 2):  # gate+up
        if etm == 1:
            return run_kernel[16, 64, 512, 16, True, wg_per_cu=1]()
        elif etm <= 20:
            return run_kernel[16, 64, 512, 16, True]()
        elif etm <= 1023:
            return run_kernel[16, 128, 512, 32, True, STREAM]()
        elif etm <= 2047:
            return run_kernel[32, 128, 512, 32, True, STREAM]()
        elif etm <= 4095:
            return run_kernel[64, 128, 512, 64, True, STREAM]()
        else:
            return run_kernel[64, 128, 512, 64, False]()

    comptime if LB == 16 and N == 7168 and packed_K == (2048 // 2):  # down
        if etm == 1:
            return run_kernel[16, 64, 512, 16, True, wg_per_cu=1]()
        elif etm <= 3:
            return run_kernel[16, 64, 512, 16, True]()
        elif etm <= 1023:
            return run_kernel[16, 128, 512, 32, True, STREAM]()
        elif etm <= 2047:
            return run_kernel[32, 128, 512, 32, True]()
        elif etm <= 3072:
            return run_kernel[64, 128, 512, 64, True]()
        elif etm <= 6144:
            return run_kernel[128, 128, 512, 64, True]()
        elif etm <= 9216:
            return run_kernel[64, 128, 512, 64, True]()
        else:
            return run_kernel[64, 128, 512, 64, False]()

    comptime if LB == 16 and N == 6144 and packed_K == (
        6144 // 2
    ):  # MiniMax-M3 gate+up
        if etm <= 256:
            # Decode: direct capped grid (pipeline_depth=3) beats persistent, else persistent.
            if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
                return run_kernel[
                    16,
                    64,
                    512,
                    16,
                    False,
                    STREAM,
                    use_decode_cap=True,
                    pipeline_depth=3,
                ]()
            return run_kernel[16, 128, 512, 32, True, STREAM]()
        elif etm <= 512:
            return run_kernel[32, 128, 512, 32, True, STREAM]()
        elif etm <= 1023:
            return run_kernel[64, 128, 512, 32, True]()
        elif etm <= 2100:
            # Keep BM64 across the etm~2048 pothole (BM128 under-fills the grid there).
            return run_kernel[64, 128, 512, 64, True]()
        elif etm <= 4095:
            return run_kernel[128, 128, 512, 64, True]()
        else:
            return run_kernel[64, 128, 512, 64, False]()

    comptime if LB == 16 and N == 6144 and packed_K == (
        3072 // 2
    ):  # MiniMax-M3 down
        if etm <= 256:
            # Decode: direct capped grid (pipeline_depth=3) beats persistent, else persistent.
            if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
                return run_kernel[
                    16,
                    64,
                    512,
                    16,
                    False,
                    STREAM,
                    use_decode_cap=True,
                    pipeline_depth=3,
                ]()
            return run_kernel[16, 128, 512, 32, True, STREAM]()
        elif etm <= 512:
            return run_kernel[32, 128, 512, 32, True, STREAM]()
        elif etm <= 2100:
            # Keep BM64 across the etm~2048 pothole (BM128 under-fills the grid there).
            return run_kernel[64, 128, 512, 32, True]()
        else:
            return run_kernel[128, 128, 512, 64, True]()

    comptime if LB == 16 and N == 3072 and packed_K == (6144 // 2):
        if etm <= 4096:
            return run_kernel[64, 128, 512, 64, True]()
        else:
            return run_kernel[128, 128, 512, 64, True]()

    # MXFP8. `use_decode_cap` is required on the decode band, not preferred:
    # it makes grid.z a capture-time constant for device graph capture.
    comptime if LB == 32 and N == 6144 and packed_K == 6144:  # gate+up, K=6144
        if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
            return run_kernel[
                16,
                64,
                512,
                16,
                False,
                STREAM,
                use_decode_cap=True,
                pipeline_depth=3,
            ]()
        if etm <= 256:
            return run_kernel[
                32,
                128,
                256,
                32,
                False,
                STREAM,
                cluster_drain_sched=True,
                b_addr_split=True,
            ]()
        elif etm <= 512:
            return run_kernel[64, 128, 512, 32, False, b_addr_split=True]()
        elif etm <= 2048:
            return run_kernel[
                64,
                128,
                256,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
                b_addr_split=True,
            ]()
        else:
            return run_kernel[
                128,
                128,
                256,
                64,
                True,
                cluster_drain_sched=True,
                scale_group=4,
                b_addr_split=True,
            ]()

    comptime if LB == 32 and N == 6144 and packed_K == 3072:  # down, K=3072
        if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
            return run_kernel[
                16,
                64,
                512,
                16,
                False,
                STREAM,
                use_decode_cap=True,
                pipeline_depth=3,
            ]()
        if etm <= 256:
            return run_kernel[
                32, 128, 256, 32, False, STREAM, pipeline_depth=3
            ]()
        elif etm <= 512:
            return run_kernel[64, 128, 512, 32, False, STREAM]()
        elif etm <= 2048:
            # Real ragged-M scenarios across this band (low/high-M x
            # low/high-skew, plus a held-out check) show a consistent
            # 6.9%-23.3% win over the prior tile here.
            return run_kernel[64, 64, 256, 16, False]()
        else:
            return run_kernel[128, 128, 256, 64, True, waves_per_eu=2]()

    comptime if LB == 24 and N == 4096 and K_LOGICAL == 7168:  # gate+up
        if etm <= 20:
            return run_kernel[16, 64, 512, 16, False]()
        elif etm <= 127:
            return run_kernel[16, 128, 512, 32, False]()
        elif etm <= 255:
            return run_kernel[32, 128, 512, 64, False]()
        else:
            return run_kernel[32, 64, 256, 64, False]()

    comptime if LB == 24 and N == 6144 and K_LOGICAL == 6144:  # M3 gate+up
        if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
            return run_kernel[
                16,
                64,
                512,
                16,
                False,
                STREAM,
                use_decode_cap=True,
                pipeline_depth=3,
            ]()
        if etm <= 3:
            return run_kernel[
                16, 64, 512, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 7:
            return run_kernel[
                16,
                128,
                512,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        elif etm <= 31:
            return run_kernel[
                16, 64, 256, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 255:
            return run_kernel[
                16,
                128,
                256,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        else:
            return run_kernel[
                32,
                128,
                256,
                64,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()

    comptime if LB == 24 and N == 6144 and K_LOGICAL == 3072:  # M3 down
        if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
            return run_kernel[
                16,
                64,
                512,
                16,
                False,
                STREAM,
                use_decode_cap=True,
                pipeline_depth=3,
            ]()
        if etm <= 3:
            return run_kernel[
                16, 64, 512, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 7:
            return run_kernel[
                16,
                128,
                512,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        elif etm <= 31:
            return run_kernel[
                16, 64, 256, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 255:
            return run_kernel[
                16,
                128,
                256,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        else:
            return run_kernel[
                32,
                128,
                256,
                64,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()

    comptime if LB == 24:
        if decode_grid_m_cap > 0 and etm <= decode_grid_m_cap:
            return run_kernel[
                16,
                64,
                512,
                16,
                False,
                STREAM,
                use_decode_cap=True,
                pipeline_depth=3,
            ]()
        if etm <= 3:
            return run_kernel[
                16, 64, 512, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 7:
            return run_kernel[
                16,
                128,
                512,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        elif etm <= 31:
            return run_kernel[
                16, 64, 256, 16, False, cluster_drain_sched=True, mfma_cluster=2
            ]()
        elif etm <= 255:
            return run_kernel[
                16,
                128,
                256,
                32,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()
        else:
            return run_kernel[
                32,
                128,
                256,
                64,
                False,
                cluster_drain_sched=True,
                mfma_cluster=2,
            ]()

    # Other shapes: persistent below the threshold, direct at/above it.
    # Not a typo: BK counts ELEMENTS, so 256 at MXFP8 is the same LDS/register
    # footprint as 512 at MXFP4.
    comptime FALLBACK_BK = 256 if LB == 32 else 512
    if etm >= m_threshold:
        return run_kernel[64, 128, FALLBACK_BK, 64, False]()
    return run_kernel[64, 128, FALLBACK_BK, 64, True]()
