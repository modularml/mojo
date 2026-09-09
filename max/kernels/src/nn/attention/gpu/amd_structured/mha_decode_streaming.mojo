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
"""MHA streaming decode kernel for gfx950.

Per-tile loop: K strips from DRAM→LDS→REG for QK MMA,
P scores through SMEM for PV MMA, split-K partitioning.

Uses DecodeStreamingKVBuffer for single-buffer, per-strip DRAM→SMEM staging
(no KVCacheIterator; strips are sub-tiled from an external DRAM tile).
"""

from std.math import ceildiv
from std.sys.intrinsics import readfirstlane
from max.gpu import block_idx
from max.gpu.sync import barrier
from max.gpu import warp_id as get_warp_id
from std.memory import bitcast
from std.utils import IndexList
from std.utils.numerics import get_accum_type

from layout.swizzle import Swizzle
from nn.attention.mha_utils import get_start_and_end_for_partitions

from .attention import Attention
from .kv_buffer import DecodeStreamingKVBuffer
from .mma import TiledMmaOp


__extension Attention:
    @always_inline
    def mha_decode_streaming(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """MHA decode: streams K strips DRAM→LDS→REG per QK MMA.

        Trades higher per-launch overhead for better throughput at
        batch ≥ 128 with long keys. Selected via `MHA_STREAMING_DECODE=True`.
        """
        comptime assert (
            Self.token_gen
        ), "mha_decode_streaming requires token_gen=True"
        comptime assert Self.BK == 32 or Self.BK == 64, "BK must be 32 or 64"
        comptime if not Self.mla_mode:
            comptime assert (
                Self.depth == 64
                or Self.depth == 128
                or Self.depth == 256
                or Self.depth == 512
            ), "mha_decode_streaming supports depth=64, 128, 256, 512"

        comptime accum_type = get_accum_type[Self.k_t.dtype]()
        comptime num_qk_strips = Self.depth // Self.BK
        comptime num_pv_strips = Self.BN // Self.BK

        # Swizzle for decode K — match legacy mha_gfx942 decode path.
        comptime k_swizzle = Swizzle(2, 0, 2)

        var warp_id = UInt32(
            readfirstlane(bitcast[.int32](UInt32(get_warp_id())))
        )

        # QK MMA op.
        comptime QKMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=True,
        ]

        # PV MMA op (matches legacy: transpose_b=False for V).
        comptime PVMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=False,
        ]

        # Split-K range for this block.
        var start, end = get_start_and_end_for_partitions[Self.BN](
            self.num_keys, num_partitions, block_idx.x
        )

        for i in range(start, end, Self.BN):
            var kv_tile_start_row = UInt32(i)
            var kv_tile_num_rows = min(
                UInt32(Self.BN), UInt32(end) - kv_tile_start_row
            )

            if self.mask_skip_and_advance(kv_tile_start_row):
                continue

            var k_tile = Self.make_kv_tile(
                self.k,
                UInt32(self.batch_idx),
                kv_tile_start_row,
                UInt32(Self.kv_head_idx()),
                kv_tile_num_rows,
            )

            var v_tile = Self.make_kv_tile(
                self.v,
                UInt32(self.batch_idx),
                kv_tile_start_row,
                UInt32(Self.kv_head_idx()),
                kv_tile_num_rows,
            )

            # K buffer — single-buffer, per-BK-strip DMA.
            var k_buffer = DecodeStreamingKVBuffer[
                mma_shape=Self.mma_shape,
                swizzle=k_swizzle,
                BN=Self.BN,
                WN=Self.WN,
                BK=Self.BK,
                num_threads=Self.num_threads,
                depth=Self.depth,
                kv_num_heads=Self.num_heads // Self.group,
                transpose=True,
            ](
                self.k,
                self.batch_idx,
                Self.kv_head_idx(),
                self.k_smem_ptr.as_unsafe_any_origin(),
                self.num_keys,
                warp_id,
            )

            # V buffer — reuses same SMEM (shared_kv=True).
            var v_buffer = DecodeStreamingKVBuffer[
                mma_shape=Self.mma_shape,
                swizzle=None,
                BN=Self.BN,
                WN=Self.WN,
                BK=Self.BK,
                num_threads=Self.num_threads,
                depth=Self.output_depth,
                kv_num_heads=Self.num_heads // Self.group,
                transpose=False,
            ](
                self.v,
                self.batch_idx,
                Self.kv_head_idx(),
                self.v_smem_ptr.as_unsafe_any_origin(),
                self.num_keys,
                warp_id,
            )

            self.zero_p_buffer[0]()

            # --- QK phase: K DRAM→LDS→REG per strip, QK MMA ---
            comptime for strip in range(num_qk_strips):
                k_buffer.load_from_dram[strip](k_tile)
                barrier()
                k_buffer.load_from_shared()
                comptime for k_mma in range(k_buffer.num_k_mmas2):
                    QKMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[strip, Int(k_mma)](),
                        k_buffer.get_mma_tile[Int(k_mma)](),
                        self.p_reg_buffer.stage_tile[0](),
                    )
                barrier()

            self.scale_p_reg[0]()

            var not_last_iter = UInt32(end) - kv_tile_start_row > UInt32(
                Self.BN
            )
            self.mask_apply[0](
                kv_tile_start_row,
                kv_tile_num_rows,
                Bool(not_last_iter),
            )

            barrier()
            self.online_softmax[0]()
            barrier()

            # --- P→SMEM ---
            self.copy_fragment_to_smem()
            barrier()

            # --- PV phase: V DRAM→LDS→REG per strip, PV MMA ---
            comptime for strip in range(num_pv_strips):
                v_buffer.load_from_dram[strip](v_tile)
                barrier()
                v_buffer.load_from_shared()
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[strip, Int(k_mma), 0](),
                        v_buffer.get_mma_tile[Int(k_mma)](),
                        self.out_reg_buffer.reg_tile,
                    )
                barrier()

        # Epilogue: softmax denominator, partition info, output store.
        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
