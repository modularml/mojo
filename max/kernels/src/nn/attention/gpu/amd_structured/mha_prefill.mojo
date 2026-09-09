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
"""Unified gfx950 MHA prefill kernel.

Handles BF16+FP8, any mask, depth∈{64,80,128,256,512}, with and without sink.
Consolidates the two older kernels (pipelined `mha_prefill` and
non-pipelined `mha_prefill_gfx950`) into one, matching `amd/mha.mojo`
from main but built on `KVBuffer` (our version of KVBuffer).

Recipe:
  * Double-buffered LDS slots (slot ping-pong) to overlap DRAM→LDS with MMA.
  * V registers loaded *after* QK (V-load-later) to reduce peak VGPR use.
  * Per-iteration: wait K → mma_qk → prefetch next tile → split softmax →
    wait V → mma_pv.
  * Softmax picks between prescaled / deferred-scale (_fma) / plain based on
    prescale_q + depth + mask.apply_log2e_after_mask.
  * Non-causal masks check for FULL_MASK tiles per-iteration and skip.

Sink is handled entirely in `Attention.__init__` (rowmax/rowsum are
pre-filled from sink_weights); the kernel body needs no sink-specific code.
"""

from std.math import align_up, ceildiv
from std.sys import llvm_intrinsic, get_defined_bool
from std.sys.intrinsics import readfirstlane
from max.gpu import warp_id as get_warp_id
from max.gpu.sync import s_waitcnt
from layout.swizzle import Swizzle
from nn.attention.mha_mask import CausalMask, TileMaskStatus
from std.memory import bitcast
from std.utils.numerics import get_accum_type

from .attention import Attention
from .kv_buffer import KVBuffer
from .mma import TiledMmaOp


# --- Synchronization primitives (re-exported for mla_prefill.mojo) ---


@always_inline
def barrier():
    """Emits the AMD GPU `s.barrier` instruction to synchronize all waves in the workgroup.
    """
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


@always_inline
def block_sync_lds_direct_load[*, vmcnt: UInt32 = 0]():
    """Waits for outstanding vector-memory instructions to complete before a direct LDS load.

    Parameters:
        vmcnt: The count of vector-memory instructions to wait on; zero waits for all outstanding VM operations.
    """
    s_waitcnt[vmcnt=vmcnt]()


# =============================================================
# Unified gfx950 MHA prefill
# =============================================================


__extension Attention:
    def mha_prefill(mut self):
        """Unified gfx950 MHA prefill: BF16+FP8, any mask, depth∈{64,80,128,256,512}.

        See module docstring for the recipe.
        """

        # --- Assertions ---
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        comptime assert (
            Self.depth == 64
            or Self.depth == 80
            or Self.depth == 128
            or Self.depth == 256
            or Self.depth == 512
        ), "depth must be 64, 80, 128, 256, or 512"
        comptime assert not Self.q_type.is_float8() or (
            Self.depth == 128 or Self.depth == 256
        ), "fp8 only supports depth=128 or 256"
        comptime assert (
            Self.depth % Self.BK == 0
            or Self.depth % Self.BK % Self.mma_shape[2] == 0
        ), "depth % BK must be a multiple of the MFMA K dimension"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"
        # `mha_prefill` hard-codes `full_kv=True` for both K and V buffers, so
        # trap any caller that instantiates it with the windowed-KV config.
        comptime assert (
            Self.amd_structured_config.full_kv
        ), "mha_prefill requires amd_structured_config.full_kv=True"

        # Pre-scale Q by (scale * log2e) for BF16 depth<=128 so QK matmul
        # produces already-scaled scores.  Disabled for depth>128 (LLVM MI
        # Scheduler crash, isReg assertion in RewriteMFMAFormStage) and for
        # FP8 (quantization back to fp8 loses too much precision).
        comptime prescale_q = get_defined_bool[
            "PRESCALE_Q", True
        ]() and Self.depth <= 128 and not Self.q_type.is_float8()
        comptime num_qk_strips = ceildiv(Self.depth, Self.BK)

        # --- Buffer init ---

        var warp_id = UInt32(
            readfirstlane(bitcast[.int32](UInt32(get_warp_id())))
        )
        comptime _smem_depth = align_up(Self.depth, Self.BK)
        comptime KBufT = KVBuffer[
            kv_t=Self.k_t,
            mma_shape=Self.mma_shape,
            swizzle=Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None),
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
            full_kv=True,
            smem_depth=_smem_depth,
        ]
        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=None,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            full_kv=True,
            smem_depth=_smem_depth,
        ]
        var k_buffer = KBufT(
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            KBufT.SmemParentType(
                self.k_smem_ptr.as_unsafe_any_origin(),
                KBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )
        var v_buffer = VBufT(
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            VBufT.SmemParentType(
                self.v_smem_ptr.as_unsafe_any_origin(),
                VBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )

        comptime accum_type = get_accum_type[Self.k_t.dtype]()

        comptime QKMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=True,
        ]
        comptime PVMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=True,
        ]

        # =============================================================
        # MMA helpers
        # =============================================================

        @always_inline
        @__parameter
        def mma_qk():
            """All QK MMAs for the tile, reading K regs from SMEM."""
            self.zero_p_buffer[0]()
            comptime for strip in range(num_qk_strips):
                comptime for k_mma in range(Self.num_k_mmas2):
                    comptime k_mma_start = k_mma * Self.mma_shape[2]
                    comptime partial_valid = (
                        Self.depth % Self.BK
                    ) if Self.depth % Self.BK != 0 else Self.BK
                    comptime is_partial_strip = (
                        strip == num_qk_strips - 1 and Self.depth % Self.BK != 0
                    )
                    comptime skip = (
                        is_partial_strip and k_mma_start >= partial_valid
                    )
                    comptime if not skip:
                        QKMmaOp.mma[swap_a_b=Self.swap_a_b](
                            self.q_buffer.mma_tile[strip, Int(k_mma)](),
                            k_buffer.get_mma_tile[Int(k_mma), strip](),
                            self.p_reg_buffer.stage_tile[0](),
                        )

        @always_inline
        @__parameter
        def mma_pv():
            """All PV MMAs for the tile, reading V regs from SMEM."""
            comptime for strip in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[strip, k_mma, 0](),
                        v_buffer.get_mma_tile[k_mma, strip](),
                        self.out_reg_buffer.reg_tile,
                    )

        # =============================================================
        # Iteration bounds
        # =============================================================

        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](
            UInt32(self.batch_idx), score_row
        )
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                UInt32(self.batch_idx), score_row, UInt32(self.num_keys)
            )
        )

        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        # Epilogue split: the LAST process_tile call must use has_next=False
        # so we never DMA past num_tiles. With paged KV, an OOB prefetch
        # dereferences a non-resident block-table entry -> hipErrorIllegalAddress.
        var num_with_next = num_tiles - 1
        var num_full_pairs = num_with_next // 2
        var has_extra_true = (num_with_next & 1) != 0

        comptime if prescale_q:
            self.scale_q_buffer()

        # Pre-load first tile K+V into LDS slot 0.
        _ = k_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        # For CausalMask, start_column/last_masked_set_end already
        # guarantee no FULL_MASK tiles in the iteration range.  Non-causal
        # masks (e.g. ChunkedCausalMask) can have interior FULL_MASK tiles,
        # so we need a per-tile status check.
        comptime has_interior_full_mask = Self.mask_t != CausalMask

        # =============================================================
        # process_tile — one KV tile: wait K, mma_qk, prefetch next,
        # softmax (split), wait V, mma_pv.
        # =============================================================

        @always_inline
        @__parameter
        def process_tile[slot: Int, has_next: Bool]():
            comptime next_slot = 1 - slot

            # Wait for K+V of this slot.  When has_next, only the previous
            # iteration's next-slot V prefetch is still in-flight (K was
            # already awaited); otherwise both K+V are freshly queued.
            comptime if has_next:
                block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            else:
                block_sync_lds_direct_load[vmcnt=0]()
            barrier()

            # For masks that can have FULL_MASK in the middle, skip now.
            # We still must prefetch the next tile (if any) so the pipeline
            # stays fed.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        _ = k_buffer.load_from_dram[next_slot]()
                        _ = v_buffer.load_from_dram[next_slot]()
                        barrier()
                    return

            k_buffer.load_from_shared(slot)
            mma_qk()

            # Prefetch next tile before softmax to hide DRAM latency under
            # the softmax VALU.
            comptime if has_next:
                _ = k_buffer.load_from_dram[next_slot]()
                _ = v_buffer.load_from_dram[next_slot]()

            # Split softmax — variant chosen at comptime from
            # prescale_q / depth / mask.apply_log2e_after_mask.
            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            elif Self.depth > 128 or Self.mask_t.apply_log2e_after_mask:
                self.online_softmax_step_0[0]()
                self.online_softmax_step_1[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Wait for V (this slot). When has_next, next slot's K+V
            # prefetch (2 DMAs) is still outstanding.
            comptime if has_next:
                block_sync_lds_direct_load[
                    vmcnt=k_buffer.vm_instrs_per_load
                    + v_buffer.vm_instrs_per_load
                ]()
            # Barrier ensures all waves' V DMA is visible before any
            # wave reads V from LDS (cross-wave coherence).
            barrier()

            v_buffer.load_from_shared(slot)

            self.online_softmax_update_output()

            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        for _ in range(num_full_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Tail: close out so the final call is has_next=False (no OOB prefetch).
        if has_extra_true:
            process_tile[0, True]()
            process_tile[1, False]()
        else:
            process_tile[0, False]()

        # Final softmax denominator and output store.
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()
