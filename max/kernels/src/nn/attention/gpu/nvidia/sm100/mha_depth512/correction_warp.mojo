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
"""Correction warp group logic for depth=256/512 pair-CTA SM100 attention.

Rescales the O accumulator in TMEM when the per-row maximum changes during
online softmax.

Depth-dependent O layout:
  split_o=True (depth=512): O split into O_lo/O_hi (MMA_N=ov_depth/2 each,
    ov_depth/4 physical cols each). Two-phase rescale: O_lo then O_hi.
  split_o=False (depth=256): Single O (MMA_N=ov_depth, MMA_M*ov_depth/256
    physical cols). Single-phase rescale.

All 128 threads participate, processing o_cols physical columns per phase.
"""

from std.sys import size_of
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_fence_before,
    tcgen05_fence_after,
)
from max.gpu.primitives.warp import _vote_nvidia_helper
from max.gpu.sync import umma_arrive_leader_cta
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from max.gpu import thread_idx
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    mul_ftz,
)
from nn.attention.mha_mask import MHAMask
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem


@always_inline
def depth512_correction[
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
    """Rescales the O accumulator in TMEM when the per-row softmax maximum changes.

    Parameters:
        MaskType: Attention mask type used to compute the per-iteration loop count.
        qkv_dtype: The data type of the query, key, and value tensors.
        config: The depth-512 SM100 attention configuration holding tile
            sizes, TMEM layout, and pipeline stage counts.
        page_size: KV cache page size in tokens passed to the mask iteration
            counter.

    Args:
        smem: Shared-memory scratch holding the correction factor and mbarriers.
        tmem_addr: Base TMEM address for this CTA's accumulators.
        seq_id: Sequence identifier forwarded to the mask iteration counter.
        score_row: Starting row offset forwarded to the mask iteration counter.
        num_keys: Number of valid keys forwarded to the mask iteration counter.
        mask: Attention mask used to compute the per-iteration loop count.
    """
    comptime accum_type = DType.float32
    comptime assert size_of[accum_type]() == 4
    comptime BM = config.BM
    comptime BN = config.BN
    comptime PairBM = BM * 2
    comptime PairBM_mask: Int = config.BM_eff() * 2

    # Physical TMEM columns per O phase.
    # split_o: each MMA has MMA_N=ov_depth/2, physical cols = ov_depth/4
    # !split_o: single MMA has MMA_N=ov_depth, physical cols = MMA_M*ov_depth/256
    comptime o_cols = config.ov_depth // 4 if config.split_o else config.MMA_M * config.ov_depth // 256
    var mbars = Depth512MBars[config.num_kv_stages, config.split_o](
        smem.mbar_base()
    )

    # Dummy arrives for the prologue iteration (no previous O to protect).
    # This satisfies the correction half of PO_lo (and PO_hi when split_o)
    # for the first P@V. Cluster-scope arrive so both CTAs' signals reach leader.
    umma_arrive_leader_cta(mbars.po_lo_mbar())
    comptime if config.split_o:
        umma_arrive_leader_cta(mbars.po_hi_mbar())

    # ---- Thread identity -----------------------------------------------------
    var tid: UInt32 = UInt32(thread_idx.x)
    var row: UInt32 = tid % 128
    var m_row = row % UInt32(BM)

    # `tmem_addr` passed in by register (read once post-`cluster_sync` in the
    # kernel prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var o_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
    var o_hi_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O_hi))

    var correction_smem = smem.correction_smem() + m_row

    var pipeline_c = mbars.consumer_c()
    var pipeline_o_lo = mbars.consumer_o_lo()
    var pipeline_o_hi = mbars.consumer_o_hi()

    var iter_count: UInt32 = (
        mask.total_iters[PairBM_mask, BN, page_size](
            seq_id, score_row, num_keys
        )
        - 1
    )

    # ---- Double-buffer constants for the TMEM rescale loop -------------------
    # Each phase processes o_cols columns. Follows the FA4 pattern:
    # alternate between o_b0 and o_b1 loads so masking overlaps TMEM access.
    comptime batch_size = 16 if o_cols % 16 == 0 else 8
    comptime assert o_cols % batch_size == 0
    comptime load_iters = o_cols // (2 * batch_size)
    comptime load_remainder = o_cols % (2 * batch_size)
    comptime assert load_iters > 1
    comptime assert (load_remainder == batch_size) or (load_remainder == 0)

    # ---- Rescale helper (inlined for O_lo and O_hi) --------------------------

    @__parameter
    @always_inline
    def rescale_o(o_tmem: TmemAddress, c_pair: SIMD[.float32, 2]):
        """Double-buffered TMEM load/scale/store over o_cols columns."""
        var o_b0: Array[Scalar[accum_type], batch_size]
        var o_b1: Array[Scalar[accum_type], batch_size]
        o_b0 = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=batch_size,
            dtype=accum_type,
            pack=False,
            width=batch_size,
        ](o_tmem.addr)

        comptime for b in range(load_iters):
            comptime b0_offset0 = 2 * b * batch_size
            comptime b1_offset = b0_offset0 + batch_size
            comptime b0_offset1 = b1_offset + batch_size
            o_b1 = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                dtype=accum_type,
                pack=False,
                width=batch_size,
            ]((o_tmem + b1_offset).addr)
            var o_b0_scaled = Array[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](
                        o_b0[_i],
                        o_b0[_i + 1],
                    ),
                    c_pair,
                )
                o_b0_scaled[_i] = pair[0]
                o_b0_scaled[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                pack=False,
            ]((o_tmem + b0_offset0).addr, o_b0_scaled)

            comptime if b0_offset1 + batch_size <= o_cols:
                o_b0 = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=batch_size,
                    dtype=accum_type,
                    pack=False,
                    width=batch_size,
                ]((o_tmem + b0_offset1).addr)
            var o_b1_scaled = Array[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](
                        o_b1[_i],
                        o_b1[_i + 1],
                    ),
                    c_pair,
                )
                o_b1_scaled[_i] = pair[0]
                o_b1_scaled[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                pack=False,
            ]((o_tmem + b1_offset).addr, o_b1_scaled)

        comptime if load_remainder > 0:
            comptime offset = 2 * batch_size * load_iters
            var o_b0_scaled_rem = Array[Scalar[accum_type], load_remainder](
                uninitialized=True
            )

            comptime for _i in range(0, load_remainder, 2):
                var pair = mul_ftz(
                    SIMD[.float32, 2](
                        o_b0[_i],
                        o_b0[_i + 1],
                    ),
                    c_pair,
                )
                o_b0_scaled_rem[_i] = pair[0]
                o_b0_scaled_rem[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=load_remainder,
                pack=False,
            ]((o_tmem + offset).addr, o_b0_scaled_rem)
        tcgen05_store_wait()
        tcgen05_fence_before()

    # ---- Main loop -----------------------------------------------------------

    while iter_count != 0:
        iter_count -= 1

        # Read correction factor from softmax.
        pipeline_c.wait()
        var c_scalar: Scalar[accum_type] = correction_smem[0]

        var change = _vote_nvidia_helper(c_scalar < 1.0) != 0

        comptime if config.split_o:
            # Phase 1: rescale O_lo (all 128 threads, o_cols cols each).
            pipeline_o_lo.wait()
            if change:
                tcgen05_fence_after()
                var c_pair = SIMD[.float32, 2](c_scalar, c_scalar)
                rescale_o(o_tmem, c_pair)
                umma_arrive_leader_cta(pipeline_o_lo.consumer_mbar())
                pipeline_o_lo.step()

                # Phase 2: rescale O_hi (all 128 threads, o_cols cols each).
                pipeline_o_hi.wait()
                tcgen05_fence_after()
                rescale_o(o_hi_tmem, c_pair)
            else:
                umma_arrive_leader_cta(pipeline_o_lo.consumer_mbar())
                pipeline_o_lo.step()
                pipeline_o_hi.wait()

            umma_arrive_leader_cta(pipeline_o_hi.consumer_mbar())
            pipeline_o_hi.step()
        else:
            # Single phase: rescale O (all 128 threads, o_cols cols each).
            pipeline_o_lo.wait()
            if change:
                tcgen05_fence_after()
                var c_pair = SIMD[.float32, 2](c_scalar, c_scalar)
                rescale_o(o_tmem, c_pair)

            umma_arrive_leader_cta(pipeline_o_lo.consumer_mbar())
            pipeline_o_lo.step()

        pipeline_c.release()
