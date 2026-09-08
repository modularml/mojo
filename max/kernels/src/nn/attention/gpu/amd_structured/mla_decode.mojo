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
"""Gfx950 MLA (Multi-Latent Attention) decode kernel, built on KVBuffer.

Forked from `mha_decode.mojo` so the MLA-specific pipeline (K==V SMEM
aliasing, `output_depth < depth` rope tail, BN×64 SMEM block split for
depth=576) can evolve independently from the MHA path.

Key invariants (enforced via `mla_decode` callsite in `mla.mojo`):
  * `mla_kv_alias = True`: V reads from K's swizzled SMEM, so no V DMA.
  * `shared_kv = True` (required by `mla_kv_alias`): V SMEM is a bitcast
    view of K SMEM (see `attention.mojo` constructor).
  * `BN = 128` ⇒ `double_buffer_k_only = False`: single-buffer K loop.
  * `output_depth ≤ depth`: per-warp V buffer slices the V-nope portion
    out of K's full-depth SMEM.

Recipe (see `process_tile`):
  * Wait K → optional FULL_MASK skip → load K from SMEM → mma_qk →
    softmax → write P to SMEM → barrier → load V (= aliased K) from
    SMEM → prefetch next K → update output → mma_pv → iter-end drain.

NOTE: `process_tile`, `prefetch_next`, the empty-partition handling,
the softmax block, and the epilogue are largely line-for-line shared
with `mha_decode.mojo`; the two files only diverge in the K==V
aliasing path, the single-buffer-K loop, and the `bk_smem<BK` K SMEM
split.  Bug fixes to the shared logic (softmax numerics, barrier
placement, etc.) must be mirrored to both files until the kernels are
re-converged behind a shared helper.
"""

from std.sys import get_defined_bool
from std.math import align_up, ceildiv
from std.math.uutils import umod, ufloordiv
from max.gpu import block_idx
from max.gpu import warp_id as get_warp_id
from max.gpu.sync import s_waitcnt
from std.utils.numerics import get_accum_type, min_or_neg_inf

from nn.attention.mha_mask import TileMaskStatus
from nn.attention.mha_utils import get_start_and_end_for_partitions

from .attention import Attention
from .mha_prefill import barrier
from .mma import TiledMmaOp

from .kv_buffer import KVBuffer, _get_k_swizzle


__extension Attention:
    @always_inline
    def mla_decode(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """gfx950 MLA decode on KVBuffer.  See module docstring."""
        comptime assert Self.token_gen, "mla_decode requires token_gen=True"
        comptime assert (
            Self.mla_kv_alias
        ), "mla_decode requires mla_kv_alias=True"
        comptime assert (
            Self.amd_structured_config.shared_kv
        ), "mla_decode requires shared_kv=True (implied by mla_kv_alias)"
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        # K-tail zero-pad path: at BK=128 we allow depth%128 != 0 as
        # long as depth is still a multiple of 64.  The kernel's inner-K
        # loop zero-pads the partial 128-element tile so the 16x16x128
        # MFMA sees zeros in the out-of-bounds half.
        comptime assert Self.depth % Self.BK == 0 or (
            Self.BK == 128 and Self.depth % 64 == 0
        ), "depth must be a multiple of BK (or of 64 when BK==128)"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"

        # Use the SMEM physical block width (`_bk_smem`) chosen by the
        # `Attention` struct.  Diverges from `BK` only for the FP8 MLA
        # decode case (BK=128, depth%128 != 0) where it becomes 64 to
        # avoid an 8 KB padded block; otherwise `_bk_smem == BK`.
        comptime _bk_smem = Self._bk_smem
        comptime k_swizzle = _get_k_swizzle[Self.mma_shape[0], _bk_smem]()

        var warp_id = UInt32(get_warp_id[broadcast=True]())

        # Split-K: compute this partition's key range.
        var start, end = get_start_and_end_for_partitions[Self.BN](
            self.num_keys, num_partitions, block_idx.x
        )

        # Empty partitions (from power-of-two bucketing): reset to
        # rowsum=0/rowmax=-inf so the reduce masks them via `scale > 0`.
        # In sink mode, __init__ primes `rowsum=1/rowmax=sink_weight` —
        # we must override that here or the reduce reads uninitialized
        # partition outputs with a nonzero scale.
        if start >= end:
            _ = self.softmax.rowmax_tensor.fill(
                min_or_neg_inf[Self.accum_type]()
            )
            _ = self.softmax.rowsum_tensor.fill(0)
            self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
            return

        # K: full BN × depth SMEM, per-warp DMA cooperation, transpose read.
        # smem_depth is padded to a multiple of `_bk_smem` so every block
        # is exactly `_bk_smem`-wide.  When `_bk_smem == BK` the formula
        # simplifies to `align_up(depth, BK)`; when `_bk_smem < BK`
        # (MLA decode rope tail) it equals `depth`, since the depth%64==0
        # gate ensures depth is a multiple of `_bk_smem`.
        comptime _k_smem_depth = align_up(Self.depth, _bk_smem)
        comptime KBufT = KVBuffer[
            kv_t=Self.k_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
            smem_depth=_k_smem_depth,
            bk_smem=_bk_smem,
        ]
        var k_buffer = KBufT(
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            KBufT.SmemParentType(
                self.k_smem_ptr.as_unsafe_any_origin(),
                KBufT._SmemParentLayout(),
            ),
            end,
            warp_id,
        )

        # V is aliased onto K's SMEM (mla_kv_alias).  We need only the
        # per-warp V reader; no V DMA buffer.  MHA: output_depth == depth.
        # MLA: output_depth < depth (V-nope only; rope tail consumed in QK).
        comptime depth_per_warp = Self.output_depth // Self.num_warps_n
        # `num_warps_n` must evenly split `output_depth`, and each slice must be
        # `_bk_smem`-aligned (or smaller than one block) so `v_warp_smem_offset`
        # lands on the K writer's block boundaries rather than aliasing wrong K
        # bytes. Holds for num_warps_n in {1,2,4} (depth_per_warp {512,256,128},
        # all 64-aligned); S=1 is num_warps_n=4.
        comptime assert Self.output_depth % Self.num_warps_n == 0
        comptime assert (depth_per_warp % _bk_smem == 0) or (
            depth_per_warp < _bk_smem
        )
        # SMEM block width seen by each warp.  Must be >= _bk_smem so the
        # blocked product layout has at least one repeat.
        comptime smem_depth_per_warp = max(depth_per_warp, _bk_smem)

        # Per-warp V SMEM offset in the blocks-stacked physical layout
        # (row-major (BN, _bk_smem) blocks laid out contiguously): for
        # (row=0, col) → (col // _bk_smem) * BN * _bk_smem + col % _bk_smem.
        # Uses `_bk_smem` (not `BK`) because the K writer lays out SMEM in
        # `_bk_smem`-wide blocks.
        var v_warp_col = umod(Int(warp_id), Self.num_warps_n)
        var v_col_start = v_warp_col * depth_per_warp
        var v_warp_smem_offset = ufloordiv(
            v_col_start, _bk_smem
        ) * Self.BN * _bk_smem + umod(v_col_start, _bk_smem)

        # V reads from K's swizzled SMEM (mla_kv_alias), so V's
        # `ds_read_tr8_b64` addressing applies the same swizzle K uses;
        # `load_v_fp8_strip` consumes that swizzle and XORs the byte
        # offset at vec granularity to land on the writer's permuted slot.
        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=depth_per_warp,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
            smem_depth=smem_depth_per_warp,
            bk_smem=_bk_smem,
        ]
        var v_buffer = VBufT(
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            VBufT.SmemParentType(
                (self.v_smem_ptr + v_warp_smem_offset).as_unsafe_any_origin(),
                VBufT._SmemParentLayout(),
            ),
            end,
            warp_id,
        )

        # Advance K iterator and mask tracking to partition start.  V is
        # aliased onto K's SMEM so V has no iterator of its own.
        k_buffer.kv_cache_iter.tile_start_row = start
        self.kv_start_row = UInt32(start)

        # Pre-scale Q by (scale * log2e).  Default on for bf16: the deferred
        # `_fma` scaling path can overflow before softmax for large decode
        # scores (for example Gemma 4's depth=512, scale=1.0 global layers).
        # Disabled for fp8 because quantizing scaled Q back to fp8 loses too
        # much precision, so fp8 stays on `_fma`.
        # Sink forces pre-scaling (rowmax in scaled-log2 space).
        comptime prescale_q = (
            get_defined_bool["PRESCALE_Q", True]()
            and not Self.q_type.is_float8()
        ) or Self.sink
        comptime if prescale_q:
            self.scale_q_buffer()

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

        @always_inline
        @__parameter
        def mma_qk():
            self.zero_p_buffer[0]()
            # ceildiv (not floor): when depth % BK != 0 (Kimi MLA: 576 %
            # 128 == 64) we must still process the partial K-tile.  Q's
            # OOB lanes are zero'd in QRegisterBuffer (buffers.mojo) and
            # K's OOB cols come back as zero via buffer_load's
            # resource-bound clamp, so the partial tile MFMA is correct
            # without any explicit masking inside this loop.
            comptime for i in range(ceildiv(Self.depth, Self.BK)):
                comptime for k_mma in range(Self.num_k_mmas2):
                    QKMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @always_inline
        @__parameter
        def mma_pv():
            # Each warp's v_buffer holds only depth_per_warp tiles
            # (loaded from the warp's LDS depth offset), so mma_subtile
            # directly gives the warp's depth range.
            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.mma_subtile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Masks with interior FULL_MASK tiles (e.g. ChunkedCausalMask) need
        # a per-tile status check: when a tile is fully masked, skip QK,
        # softmax and PV but keep the DMA prefetch pipeline moving.
        # CausalMask during decode never produces FULL_MASK tiles.
        comptime has_interior_full_mask = (
            Self.mask_t.check_mask_during_decoding
        )

        @always_inline
        @__parameter
        def prefetch_next():
            """Prefetch next K after current LDS reads have drained.

            MLA: V is aliased onto K's SMEM, so there's no separate V DMA;
            single-buffer K (no `double_buffer_k_only` since BN=128 > 64),
            overwrites slot 0.
            """
            s_waitcnt[lgkmcnt=0]()
            barrier()
            _ = k_buffer.load_from_dram[0]()

        @always_inline
        @__parameter
        def process_tile[has_next: Bool]():
            """Process one KV tile.

            K is already in LDS when called.  V is loaded from the same
            SMEM (mla_kv_alias) mid-tile after K is consumed by QK.
            """
            # Wait for K DMA.
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier()

            # Skip fully-masked tiles (ChunkedCausalMask interior).
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        prefetch_next()
                    return

            k_buffer.load_from_shared(0)
            # AITER-style register-side zero pad for the partial K-tile's
            # OOB tail.  No-op when depth % BK == 0.  Mirrors
            # `QRegisterBuffer.__init__`'s partial-tile zero on the Q side;
            # both zero the upper half of each lane's partial-tile fragment
            # (the pad K-positions in MFMA-K), preserving the lower half
            # which holds valid K-rope / Q-rope data.
            k_buffer.zero_partial_tile_pad()
            mma_qk()

            # Online softmax on P (cross-warp reduction via SMEM).
            # Prescaled path is numerically safer (Qwen-class models need
            # it — `_fma` accumulates NaN after a few decode steps on real
            # weights).  `_fma` variants fold the score scale into exp2 and
            # save VALU; opt in via `-D PRESCALE_Q=False` if tolerable.
            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Write P to shared memory so all warps can read it for PV.
            self.p_reg_buffer.copy_to_shared()
            s_waitcnt[lgkmcnt=0]()
            barrier()

            v_buffer.load_from_shared(0)

            # Prefetch next tile after V→REG.  Barrier ensures all warps
            # finished V LDS reads before DMA.
            comptime if has_next:
                prefetch_next()

            self.online_softmax_update_output()
            mma_pv()

            # Iter-end drain + barrier.  The entry barrier at the next
            # iteration is not sufficient on its own: without this
            # explicit full drain, mma_pv's pipelined P SMEM ds_reads (and
            # prefetch_next's in-flight DMA) can overlap with iter N+1's
            # copy_to_shared writes and K LDS reads.  Waitcnt-only and
            # barrier-only are both empirically insufficient — both are
            # required.
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier()

        var num_tiles = ceildiv(end - start, Self.BN)

        # Prologue: load first tile's K (slot 0).  V is aliased onto K's
        # SMEM and is consumed mid-tile after K's reads finish.
        _ = k_buffer.load_from_dram[0]()

        # Single-buffer K: simple linear loop.  MLA never enables
        # `double_buffer_k_only` because BN=128 > 64.
        for _ in range(num_tiles - 1):
            process_tile[True]()

        if num_tiles > 0:
            process_tile[False]()

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
