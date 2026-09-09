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
"""MLA (Multi-Latent Attention) prefill kernel for gfx950.

Double-buffered MLA prefill with K_rope support. Uses TileTensor
throughout; no LayoutTensor in the public or internal API.

Two-phase QK matmul per tile:
  Phase 1 (nope): Q[:,:depth] @ K^T
  Phase 2 (rope): Q[:,depth:q_depth] @ K_rope^T
"""

from std.math.uutils import ufloordiv
from std.sys import align_of, simd_width_of
from std.sys.intrinsics import readfirstlane
from max.gpu import warp_id as get_warp_id
from std.memory import bitcast, unsafe_stack_allocation
from layout.swizzle import Swizzle
from nn.attention.mha_mask import CausalMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type

from .attention import Attention
from .iglp import _iglp_opt, AMDIGLPStrategy
from .kv_buffer import KVBuffer
from .mha_prefill import barrier, block_sync_lds_direct_load
from .mma import TiledMmaOp


__extension Attention:
    @always_inline
    def mla_prefill[
        k_rope_t: MHAOperand,
        //,
    ](mut self, k_rope: k_rope_t):
        """Double-buffered gfx950 MLA prefill with K_rope support.

        Uses gfx950 double-buffered KVBuffer for K, V, and K_rope
        (DRAM->SMEM direct). K_rope has its own SMEM double-buffer.
        Two-phase QK matmul:
        Phase 1 (nope): Q[:,:depth] @ K^T  (KVBuffer path)
        Phase 2 (rope): Q[:,depth:q_depth] @ K_rope^T  (KVBuffer path)
        """
        comptime cache_num_heads = 1
        comptime cache_depth = 576
        comptime rope_depth = q_depth - Self.depth
        comptime cache_group = Self.num_heads // cache_num_heads

        comptime assert Self.BK == 32 or Self.BK == 64, "BK must be 32 or 64"
        comptime assert Self.depth == 128, "MLA depth must be 128"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"
        comptime assert (
            rope_depth % Self.BK == 0
        ), "rope_depth must be a multiple of BK"

        comptime k_swizzle = (
            Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None)
        )

        var warp_id = UInt32(
            readfirstlane(bitcast[.int32](UInt32(get_warp_id())))
        )

        # K buffer (nope): depth=128, double-buffered gfx950 style.
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
            self.num_keys,
            warp_id,
        )

        # V buffer: depth=128, double-buffered gfx950 style.
        # FP8 V uses Swizzle(3, 0, 4) (same as K) to eliminate the
        # ds_read_tr8_b64 bank conflicts inherent to BK=64 row stride
        # (4 lane-quads per 32-bank LDS row → 4-way conflict). Both the
        # SubTileLoaderLDS writer and the load_v_fp8_strip reader honor
        # the same XOR so the data is read back correctly.
        comptime v_swizzle = (
            Swizzle(3, 0, 4) if Self.v_t.dtype.is_float8()
            and Self.mma_shape[0] == 32 else Optional[Swizzle](None)
        )
        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=v_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
        ]
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

        # K_rope buffer: depth=rope_depth, double-buffered gfx950 style.
        # Separate SMEM allocation since K_rope DMA overlaps with K_nope.
        comptime alignment = align_of[
            SIMD[k_rope_t.dtype, simd_width_of[k_rope_t.dtype]()]
        ]()
        comptime k_rope_smem_elems = 2 * Self.BN * rope_depth
        var k_rope_smem_ptr = unsafe_stack_allocation[
            k_rope_smem_elems,
            k_rope_t.dtype,
            address_space=.SHARED,
            alignment=alignment,
        ]()
        comptime KRopeBufT = KVBuffer[
            kv_t=k_rope_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=rope_depth,
            kv_num_heads=cache_num_heads,
            transpose=True,
            cache_depth=cache_depth,
            head_dim_offset=cache_depth - rope_depth,
        ]
        var k_rope_buffer = KRopeBufT(
            k_rope,
            self.batch_idx,
            ufloordiv(Int(self.kv_head_idx()), cache_group),
            KRopeBufT.SmemParentType(
                k_rope_smem_ptr.as_unsafe_any_origin(),
                KRopeBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )

        comptime accum_type = get_accum_type[Self.k_t.dtype]()

        # Phase 1: Q_nope @ K^T (depth // BK iterations)
        @always_inline
        @__parameter
        def mma_qk_nope():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.get_mma_tile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        # Phase 2: Q_rope @ K_rope^T (rope_depth // BK iterations)
        @always_inline
        @__parameter
        def mma_qk_rope():
            comptime MmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]
            comptime nope_tiles = Self.depth // Self.BK

            comptime for i in range(rope_depth // Self.BK):
                comptime for k_mma in range(k_rope_buffer.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[nope_tiles + i, k_mma](),
                        k_rope_buffer.get_mma_tile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @always_inline
        @__parameter
        def mma_pv():
            comptime PVMmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.get_mma_tile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Calculate iteration bounds using mask helpers.
        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](
            UInt32(self.batch_idx), score_row
        )
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                UInt32(self.batch_idx), score_row, UInt32(self.num_keys)
            )
        )

        # Advance KV iterators and mask tracking to start_col.
        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        k_rope_buffer.kv_cache_iter.tile_start_row = (
            Int(start_col) + self.cache_start_pos
        )
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        var num_pairs = num_tiles // 2

        # Pre-load first tile K+K_rope+V into LDS slot 0.
        # Issue order: K, K_rope, V — so waiting for vmcnt=V.vm
        # ensures K and K_rope are done while V may still be in-flight.
        _ = k_buffer.load_from_dram[0]()
        _ = k_rope_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        comptime has_interior_full_mask = Self.mask_t != CausalMask

        @always_inline
        @__parameter
        def process_tile[slot: Int, has_next: Bool]():
            comptime next_slot = 1 - slot

            # Wait for K + K_rope loads to complete for this slot.
            # V was issued last, so vmcnt=V.vm means K and K_rope are done.
            comptime if has_next:
                block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            else:
                block_sync_lds_direct_load[vmcnt=0]()
            barrier()

            # IGroupLP: co-issue the softmax exp with the next tile's QK MMA to
            # hide the online-softmax latency on this overlap-bound kernel.
            _iglp_opt[AMDIGLPStrategy.MFMA_EXP_INTERLEAVE]()

            # Skip fully masked tiles for non-causal masks.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        _ = k_buffer.load_from_dram[next_slot]()
                        _ = k_rope_buffer.load_from_dram[next_slot]()
                        _ = v_buffer.load_from_dram[next_slot]()
                        barrier()
                    return

            # Load K (nope) from shared -> registers and compute Q_nope @ K^T
            k_buffer.load_from_shared(slot)
            mma_qk_nope()

            # Load K_rope from shared -> registers and compute Q_rope @ K_rope^T
            k_rope_buffer.load_from_shared(slot)
            mma_qk_rope()

            # Prefetch next K+K_rope+V tile before softmax to hide latency.
            comptime if has_next:
                _ = k_buffer.load_from_dram[next_slot]()
                _ = k_rope_buffer.load_from_dram[next_slot]()
                _ = v_buffer.load_from_dram[next_slot]()

            # Online softmax: deferred-scale variant that emits packed FMAs
            # (`v_pk_fma_f32 score, scale, neg_scaled_max`) instead of the
            # separate `v_pk_add (score - max)` + `v_pk_mul (* scale)` pair.
            # Saves ~16 VALU ops per warp per tile and matches aiter's
            # softmax inner loop.
            self.online_softmax_step_0_pkfma[0]()
            self.online_softmax_step_1_pkfma[0]()

            # Wait for V loads from current slot before reading.
            comptime if has_next:
                block_sync_lds_direct_load[
                    vmcnt=k_buffer.vm_instrs_per_load
                    + k_rope_buffer.vm_instrs_per_load
                    + v_buffer.vm_instrs_per_load
                ]()
            barrier()

            v_buffer.load_from_shared(slot)

            self.online_softmax_update_output()

            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        for _ in range(num_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Remainder: last odd tile from slot 0.
        if num_tiles % 2 != 0:
            process_tile[0, False]()

        # Apply final softmax denominator and store.
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()
