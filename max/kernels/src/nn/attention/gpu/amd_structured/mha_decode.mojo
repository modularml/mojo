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
"""Gfx950 MHA decode kernel, built on KVBuffer.

Ported from `amd/mha.mojo::mha_decoding`.  Uses a full-depth KVBuffer
for K (optionally double-buffered via `double_buffer_k_only`) and a
two-view V setup sharing one SMEM region: `v_dma_buffer` (full
BN x depth) for cooperative DRAM→LDS, `v_buffer` (per-warp
BN x depth_per_warp) for LDS→REG and MMA.

MLA decode (K==V aliasing, `output_depth < depth`, K-tail rope handling)
lives in `mla_decode.mojo`; this file is MHA-only.

Recipe (see `process_tile`):
  * Wait K (and V unless shared_kv) → optional FULL_MASK skip → load K
    from SMEM → mma_qk → softmax → write P to SMEM → barrier →
    (DMA V now if shared_kv) → load V from SMEM → prefetch next K+V →
    update output → mma_pv → iter-end drain.

`shared_kv` (depth>256) overlays V onto K's SMEM and defers V DMA until
K is consumed.  `double_buffer_k_only` pings K between slots 0/1 while
V always lives in slot 0.
"""

from std.sys import get_defined_bool
from std.sys.intrinsics import readfirstlane
from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from max.gpu import block_idx
from max.gpu import warp_id as get_warp_id
from max.gpu.sync import s_waitcnt
from std.memory import bitcast
from std.utils.numerics import get_accum_type, min_or_neg_inf

from nn.attention.mha_mask import TileMaskStatus
from nn.attention.mha_utils import get_start_and_end_for_partitions

from .attention import Attention
from .mha_prefill import barrier
from .mma import TiledMmaOp

from .kv_buffer import KVBuffer, _get_k_swizzle


__extension Attention:
    @always_inline
    def mha_decode(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """gfx950 MHA decode on KVBuffer.  See module docstring."""
        comptime assert Self.token_gen, "mha_decode requires token_gen=True"
        comptime assert (
            not Self.mla_kv_alias
        ), "mha_decode does not support mla_kv_alias=True; use mla_decode"
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"

        comptime k_swizzle = _get_k_swizzle[Self.mma_shape[0], Self.BK]()

        var warp_id = UInt32(
            readfirstlane(bitcast[.int32](UInt32(get_warp_id())))
        )

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

        # V uses two buffers to avoid loading full depth into registers:
        # - v_dma_buffer: full depth, for cooperative DRAM→LDS.
        # - v_buffer:     per-warp depth_per_warp slice, for LDS→REG and MMA.
        # The per-warp buffer's SMEM pointer is offset into the shared V SMEM
        # region.  When depth_per_warp < BK, multiple warps share one
        # BK-wide block; the offset encodes within-block position.
        comptime depth_per_warp = Self.depth // Self.num_warps_n
        # SMEM block width seen by each warp.  Must be >= BK so the
        # blocked product layout has at least one repeat.
        comptime smem_depth_per_warp = max(depth_per_warp, Self.BK)

        comptime VDmaBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=None,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
        ]
        var v_dma_buffer = VDmaBufT(
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            VDmaBufT.SmemParentType(
                self.v_smem_ptr.as_unsafe_any_origin(),
                VDmaBufT._SmemParentLayout(),
            ),
            end,
            warp_id,
        )

        # Per-warp V SMEM offset in the blocks-stacked physical layout
        # (row-major (BN, BK) blocks laid out contiguously): for
        # (row=0, col) → (col // BK) * BN * BK + col % BK.
        var v_warp_col = umod(Int(warp_id), Self.num_warps_n)
        var v_col_start = v_warp_col * depth_per_warp
        var v_warp_smem_offset = ufloordiv(
            v_col_start, Self.BK
        ) * Self.BN * Self.BK + umod(v_col_start, Self.BK)

        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=None,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=depth_per_warp,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
            smem_depth=smem_depth_per_warp,
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

        # Advance iterators and mask tracking to partition start.
        k_buffer.kv_cache_iter.tile_start_row = start
        v_dma_buffer.kv_cache_iter.tile_start_row = start
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
            comptime for i in range(Self.depth // Self.BK):
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

        comptime shared_kv = Self.amd_structured_config.shared_kv
        comptime double_buffer_k_only = (
            Self.amd_structured_config.double_buffer_k_only
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
        def prefetch_next[slot: Int]():
            """Prefetch next K (and V if not shared_kv) after current LDS
            reads have drained."""
            s_waitcnt[lgkmcnt=0]()
            barrier()
            # Double-buffer K: prefetch to OTHER slot.
            # Single-buffer K: overwrite slot 0.
            comptime if double_buffer_k_only:
                _ = k_buffer.load_from_dram[1 - slot]()
            else:
                _ = k_buffer.load_from_dram[0]()
            comptime if not shared_kv:
                _ = v_dma_buffer.load_from_dram[0]()

        @always_inline
        @__parameter
        def process_tile[slot: Int, has_next: Bool]():
            """Process one KV tile.

            K[slot] is already in LDS when called.  If not shared_kv, V
            is also in LDS (parallel DMA).  If shared_kv, V is loaded
            mid-tile after K is consumed.
            """
            # Wait for K DMA (and V DMA unless shared_kv).
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier()

            # Skip fully-masked tiles (ChunkedCausalMask interior).
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    # When `shared_kv`, V DMA is deferred to mid-tile (inside
                    # the non-skipped branch below) and `prefetch_next` only
                    # advances K's iterator.  Skipped FULL_MASK tiles must
                    # still advance V's iterator manually, or the first
                    # PARTIAL_MASK tile (e.g.  SlidingWindow visible window
                    # after many FULL_MASK tiles) would load V from the
                    # original start row instead of the current row.
                    comptime if shared_kv and not Self.mla_kv_alias:
                        v_dma_buffer.kv_cache_iter.increment()
                    comptime if has_next:
                        prefetch_next[slot]()
                    return

            k_buffer.load_from_shared(slot)
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

            # Write P to shared memory so all warps can read it for PV. The
            # wide-MMA fold keeps P in registers, so it skips this and the
            # barrier that ordered it; `shared_kv` re-syncs on its own below.
            comptime if Self._p_shared_memory_backed:
                self.p_reg_buffer.copy_to_shared()
                s_waitcnt[lgkmcnt=0]()
                barrier()

            comptime if shared_kv:
                # K consumed + P synced.  Load V into shared SMEM
                # (overwrites K).  All warps finished K LDS reads above.
                _ = v_dma_buffer.load_from_dram[0]()
                s_waitcnt[vmcnt=0, lgkmcnt=0]()
                barrier()

            v_buffer.load_from_shared(0)

            # Prefetch next tile after V→REG.  Barrier ensures all warps
            # finished V LDS reads before DMA.
            comptime if has_next:
                prefetch_next[slot]()

            self.online_softmax_update_output()
            mma_pv()

            # Iter-end drain + barrier.  The entry barrier at the next
            # iteration is not sufficient on its own: without this
            # explicit full drain, prefetch_next's in-flight DMA (and, on
            # the SMEM-backed P path, mma_pv's pipelined P ds_reads) can
            # overlap with iter N+1's K LDS reads and copy_to_shared
            # writes.  Waitcnt-only and barrier-only are both empirically
            # insufficient — both are required.
            s_waitcnt[vmcnt=0, lgkmcnt=0]()
            barrier()

        var num_tiles = ceildiv(end - start, Self.BN)

        # Prologue: load first tile's K (slot 0).  V loaded in parallel
        # when separate SMEM, or mid-tile when shared.
        _ = k_buffer.load_from_dram[0]()
        comptime if not shared_kv:
            _ = v_dma_buffer.load_from_dram[0]()

        comptime if double_buffer_k_only:
            # Double-buffer K: process tiles in pairs (slot 0, slot 1).
            # Epilogue split: the LAST process_tile call must use
            # has_next=False so we never DMA past num_tiles.  With paged
            # KV, an OOB prefetch dereferences a non-resident block-table
            # entry and corrupts/faults the pipeline -> decode mismatches.
            var num_with_next = num_tiles - 1
            var num_full_pairs = num_with_next // 2
            var has_extra_true = (num_with_next & 1) != 0

            for _ in range(num_full_pairs):
                process_tile[0, True]()
                process_tile[1, True]()

            if has_extra_true:
                process_tile[0, True]()
                process_tile[1, False]()
            else:
                process_tile[0, False]()
        else:
            # Single-buffer K: simple linear loop.
            for _ in range(num_tiles - 1):
                process_tile[0, True]()

            if num_tiles > 0:
                process_tile[0, False]()

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
