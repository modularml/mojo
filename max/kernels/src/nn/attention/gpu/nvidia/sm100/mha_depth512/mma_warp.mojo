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
"""MMA warp logic for depth=256/512 pair-CTA SM100 attention.

Orchestrates Q@K' and P@V matrix multiplications using pair-CTA SS MMA
(cta_group=2). Both MMAs read both operands from SMEM, unlike FA4 which
uses TS MMA for P@V.

Q@K' produces S in TMEM (double-buffered S_even/S_odd across iterations).

Depth-dependent P@V strategy:
  split_o=True (depth=512): P@V split into P@V_lo and P@V_hi, each with
    MMA_N=ov_depth/2. Produces O_lo and O_hi in separate TMEM regions.
  split_o=False (depth=256): Single P@V with MMA_N=ov_depth. Produces
    single O in TMEM.

K is sub-staged into num_qk_stages depth chunks (BK0 each). V is
sub-staged into num_pv_stages BN chunks (BK1 each). When split_o, V_lo
and V_hi are in separate slots (4 total V slots per iteration); otherwise
only 2 V slots per iteration.

CTA role split (cta_group=2):
    Leader CTA (even rank): Owns all pipeline interactions: waits on KV
        producer barriers, issues MMA, releases KV consumer barriers with
        cta_group=2 commit (fences MMA read of both CTAs' SMEM, then
        signals both CTAs' consumer barriers), and commits S/O barriers.
    Peer CTA (odd rank): Returns immediately.
"""

from std.sys import size_of
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAKind,
    mma_arrive_multicast,
)
from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    StagedPipeline,
    elect,
)
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem
from std.utils.index import Index


@always_inline
def depth512_mma[
    MaskType: MHAMask,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    page_size: Int,
](
    smem: Depth512AttentionSMem[config=config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    """Runs the pair-CTA Q@K' and P@V MMAs for depth=256/512 SM100 attention.

    Drives the leader CTA of a cta_group=2 pair through the full attention
    iteration loop: a peeled first iteration seeds S_even and O, then the main
    loop alternates S_even/S_odd while accumulating into O (or O_lo/O_hi when
    split_o). Peer CTAs return immediately.

    Parameters:
        MaskType: The MHAMask specialization governing tile masking behavior.
        qkv_dtype: Element dtype of the Q, K, V operands.
        config: Depth-256/512 SM100 attention configuration (tile sizes,
            pipeline depths, split_o strategy).
        page_size: Paged KV cache page size in keys.

    Args:
        smem: Shared-memory allocator holding Q, K, V, P buffers and mbarriers.
        tmem_addr: Base TMEM address for this CTA pair's S/O accumulators.
        seq_id: Sequence index used by the mask.
        score_row: Starting query row index for this CTA pair.
        num_keys: Number of valid keys in the KV cache for this sequence.
        mask: Mask instance used to skip fully-masked tiles.
    """
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime BK0 = config.BK0
    comptime BK1 = config.BK1
    comptime MMA_M = config.MMA_M
    comptime ov_depth = config.ov_depth
    comptime ov_half = ov_depth // 2
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime num_kv_stages = config.num_kv_stages
    comptime cta_group = config.cta_group
    comptime qkv_size = size_of[qkv_dtype]()

    # Full pair-CTA M dimension for mask computations.
    comptime PairBM = BM * 2
    comptime PairBM_mask: Int = config.BM_eff() * 2

    comptime assert BK0 % config.MMA_K == 0, "BK0 must be a multiple of MMA_K"
    comptime assert BK1 % config.MMA_K == 0, "BK1 must be a multiple of MMA_K"

    # ---- MMA types -----------------------------------------------------------
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if qkv_dtype.is_float8() else UMMAKind.KIND_F16
    )

    # Q@K' → S: SS MMA, cta_group=2
    comptime UMMA_QK = SM100TensorAccumulator[
        qkv_dtype,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=BN,
        BK=BK0,
        a_tmem=False,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        cta_group=cta_group,
        mma_kind=mma_kind,
    ]

    # P@V MMA types are defined inside pv_mma (depth-dependent).

    # ---- TMEM addresses ------------------------------------------------------
    # `tmem_addr` is read ONCE in the kernel prologue (post-`cluster_sync`) and
    # passed in by register; do NOT re-read `smem.tmem_addr_ptr()` here (see the
    # publish-handshake note in `kernel.mojo`).
    var o_tmem = tmem_addr + UInt32(config.TMEM_O)
    var o_hi_tmem = tmem_addr + UInt32(config.TMEM_O_hi)
    var s_even_tmem = tmem_addr + UInt32(config.TMEM_S_even)
    var s_odd_tmem = tmem_addr + UInt32(config.TMEM_S_odd)

    # ---- SMEM descriptors ----------------------------------------------------

    # Q: [BM, BK0] per depth sub-stage, k_major.
    var q_desc = smem_descriptor[
        BMN=BM,
        BK=BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.q_smem())

    # K: [BN//2, BK0] per CTA per pipeline slot, k_major.
    var kv_desc_k = smem_descriptor[
        BMN=BN // 2,
        BK=BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.kv_smem_base())

    # P: descriptor with [BM, BN] strides (matching the full P buffer layout).
    var p_desc = smem_descriptor[
        BMN=BM,
        BK=BN,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.p_smem())

    # V: [BK1, v_cols_per_cta] per CTA per pipeline slot, mn_major.
    # Each slot holds one [BK1, v_cols_per_cta] tile. The MMA with
    # cta_group=2 reads v_cols_per_cta cols from each CTA.
    var kv_desc_v = smem_descriptor[
        BMN=config.v_cols_per_cta,
        BK=BK1,
        swizzle_mode=config.swizzle_mode,
        is_k_major=False,
    ](smem.kv_smem_base())

    # ---- Byte offsets for sub-staging ----------------------------------------

    comptime q_sub_bytes = BM * BK0 * qkv_size
    comptime kv_stage_bytes = (BN // 2) * BK0 * qkv_size
    # P is [BM, BN] k_major with swizzle.  Advancing by BK1 columns skips
    # BK1/sw_K swizzle blocks of BM*sw_K elements each = BM*BK1 elements.
    comptime p_sub_bytes = BM * BK1 * qkv_size

    # ---- Barrier / pipeline setup --------------------------------------------

    var mbars = Depth512MBars[num_kv_stages, config.split_o](smem.mbar_base())

    # S double-buffered pipelines (MMA is the producer of S in TMEM).
    var pipeline_s_even = mbars.producer_s_even()
    pipeline_s_even.state._phase = 1
    var pipeline_s_odd = mbars.producer_s_odd()
    pipeline_s_odd.state._phase = 1

    # O pipeline: acquire waits PO_lo, commit arrives O_mma_lo.
    # Override phase to 0: first acquire must block until PO_lo fires.
    var pipeline_o_lo = mbars.producer_o_lo()
    pipeline_o_lo.state._phase = 0

    # O_hi pipeline (used only when split_o, but defined unconditionally).
    var pipeline_o_hi = mbars.producer_o_hi()
    pipeline_o_hi.state._phase = 0

    # KV consumer pipeline (load warp is the producer).
    comptime KVPipeType = StagedPipeline[num_kv_stages, 1]
    var kv_pipeline: KVPipeType = {mbars.get_kv_mbars()}

    # ---- Iteration bounds (must match load_warp exactly) ---------------------

    var kv_row: UInt32 = mask.start_column[PairBM_mask, BN, page_size](
        seq_id, score_row
    )
    var iter_count: UInt32 = (
        mask.last_masked_set_end[PairBM_mask, BN, page_size](
            seq_id, score_row, num_keys
        )
        - 1
    )

    comptime check_mask = (
        mask.nonfull_sets[PairBM_mask, BN]()[0] == TileMaskStatus.UNKNOWN_MASK
    )

    # ---- CTA identity --------------------------------------------------------
    var is_leader = block_rank_in_cluster() % 2 == 0

    if not is_leader:
        return

    var e = elect()

    # CTA mask for multicast arrive: signal both CTAs in the pair.
    # 0b11 = (1 << cta_group) - 1 for cta_group=2.
    comptime cta_mask = UInt16(0x3)

    # ---- Helper: P@V with depth-dependent commit strategy ---------------------

    @__parameter
    @always_inline
    def pv_mma(*, is_first: Bool):
        """Execute P@V multiplication(s) and commit O barriers.

        split_o: P@V_lo → commit O_mma_lo, then P@V_hi → commit O_mma_hi.
        !split_o: single P@V → commit O_mma_lo.

        Args:
            is_first: True for the peeled iteration (c_scale=0 on stage 0).
        """
        comptime if config.split_o:
            # P@V_lo/hi MMA types: MMA_N=ov_depth/2
            comptime UMMA_PV_lo = SM100TensorAccumulator[
                qkv_dtype,
                accum_type,
                MMA_M=MMA_M,
                MMA_N=ov_half,
                BK=BK1,
                a_tmem=False,
                swizzle_a=config.swizzle_mode,
                swizzle_b=config.swizzle_mode,
                transpose_b=False,
                cta_group=cta_group,
                mma_kind=mma_kind,
            ]
            comptime UMMA_PV_hi = SM100TensorAccumulator[
                qkv_dtype,
                accum_type,
                MMA_M=MMA_M,
                MMA_N=ov_half,
                BK=BK1,
                a_tmem=False,
                swizzle_a=config.swizzle_mode,
                swizzle_b=config.swizzle_mode,
                transpose_b=False,
                cta_group=cta_group,
                mma_kind=mma_kind,
            ]

            # -- P@V_lo → O_lo (own pipeline slots) --
            pipeline_o_lo.acquire()
            comptime for pv_stage in range(num_pv_stages):
                kv_pipeline.consumer_wait()
                UMMA_PV_lo.mma(
                    p_desc + UInt32(pv_stage * p_sub_bytes),
                    kv_desc_v
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                    o_tmem,
                    c_scale=UInt32(0) if is_first
                    and pv_stage == 0 else UInt32(1),
                    elect=e,
                )
                if e != 0:
                    mma_arrive_multicast[cta_group](
                        kv_pipeline.consumer_mbar(), cta_mask
                    )
                kv_pipeline.state.step()
            if e != 0:
                mma_arrive_multicast[cta_group](
                    pipeline_o_lo.producer_mbar(), cta_mask
                )
            pipeline_o_lo.step()

            # -- P@V_hi → O_hi (own pipeline slots) --
            pipeline_o_hi.acquire()
            comptime for pv_stage in range(num_pv_stages):
                kv_pipeline.consumer_wait()
                UMMA_PV_hi.mma(
                    p_desc + UInt32(pv_stage * p_sub_bytes),
                    kv_desc_v
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                    o_hi_tmem,
                    c_scale=UInt32(0) if is_first
                    and pv_stage == 0 else UInt32(1),
                    elect=e,
                )
                if e != 0:
                    mma_arrive_multicast[cta_group](
                        kv_pipeline.consumer_mbar(), cta_mask
                    )
                kv_pipeline.state.step()
            if e != 0:
                mma_arrive_multicast[cta_group](
                    pipeline_o_hi.producer_mbar(), cta_mask
                )
            pipeline_o_hi.step()
        else:
            # Single P@V MMA type: MMA_N=ov_depth
            comptime UMMA_PV = SM100TensorAccumulator[
                qkv_dtype,
                accum_type,
                MMA_M=MMA_M,
                MMA_N=ov_depth,
                BK=BK1,
                a_tmem=False,
                swizzle_a=config.swizzle_mode,
                swizzle_b=config.swizzle_mode,
                transpose_b=False,
                cta_group=cta_group,
                mma_kind=mma_kind,
            ]

            # Single P@V → O (no split)
            pipeline_o_lo.acquire()
            comptime for pv_stage in range(num_pv_stages):
                kv_pipeline.consumer_wait()
                UMMA_PV.mma(
                    p_desc + UInt32(pv_stage * p_sub_bytes),
                    kv_desc_v
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                    o_tmem,
                    c_scale=UInt32(0) if is_first
                    and pv_stage == 0 else UInt32(1),
                    elect=e,
                )
                if e != 0:
                    mma_arrive_multicast[cta_group](
                        kv_pipeline.consumer_mbar(), cta_mask
                    )
                kv_pipeline.state.step()
            if e != 0:
                mma_arrive_multicast[cta_group](
                    pipeline_o_lo.producer_mbar(), cta_mask
                )
            pipeline_o_lo.step()

    # ---- Peeled first iteration ----------------------------------------------

    # Q@K' → S_even (4 K sub-stages)
    comptime for qk_stage in range(num_qk_stages):
        kv_pipeline.consumer_wait()
        UMMA_QK.mma(
            q_desc + UInt32(qk_stage * q_sub_bytes),
            kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
            s_even_tmem,
            c_scale=UInt32(0) if qk_stage == 0 else UInt32(1),
            elect=e,
        )
        if e != 0:
            mma_arrive_multicast[cta_group](
                kv_pipeline.consumer_mbar(), cta_mask
            )
        kv_pipeline.state.step()
    if e != 0:
        mma_arrive_multicast[cta_group](
            pipeline_s_even.producer_mbar(), cta_mask
        )
    pipeline_s_even.step()

    # P@V_lo → O_lo, then P@V_hi → O_hi
    pv_mma(is_first=True)

    # ---- Main loop (alternating S_even / S_odd) ------------------------------

    var s_cur_pipeline = pipeline_s_odd
    var s_cur_tmem = s_odd_tmem
    var s_nxt_pipeline = pipeline_s_even
    var s_nxt_tmem = s_even_tmem

    while iter_count != 0:
        iter_count -= 1
        kv_row += UInt32(BN)

        # Mask check: skip fully-masked tiles.
        comptime if check_mask:
            if (
                mask.status(
                    seq_id,
                    Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                    Index[dtype=DType.int32](PairBM_mask, BN),
                )
                == TileMaskStatus.FULL_MASK
            ):
                continue

        # Q@K' → S (current buffer)
        s_cur_pipeline.acquire()
        comptime for qk_stage in range(num_qk_stages):
            kv_pipeline.consumer_wait()
            UMMA_QK.mma(
                q_desc + UInt32(qk_stage * q_sub_bytes),
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                s_cur_tmem,
                c_scale=UInt32(0) if qk_stage == 0 else UInt32(1),
                elect=e,
            )
            if e != 0:
                mma_arrive_multicast[cta_group](
                    kv_pipeline.consumer_mbar(), cta_mask
                )
            kv_pipeline.state.step()
        if e != 0:
            mma_arrive_multicast[cta_group](
                s_cur_pipeline.producer_mbar(), cta_mask
            )
        s_cur_pipeline.step()

        # P@V_lo → O_lo, then P@V_hi → O_hi
        pv_mma(is_first=False)

        # Swap S buffers for next iteration.
        var tmp_pipeline = s_cur_pipeline
        s_cur_pipeline = s_nxt_pipeline
        s_nxt_pipeline = tmp_pipeline
        var tmp_tmem = s_cur_tmem
        s_cur_tmem = s_nxt_tmem
        s_nxt_tmem = tmp_tmem
