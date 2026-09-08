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
"""RDNA Wave32 MHA prefill kernel.

Recipe (per KV tile):
  * K loaded to LDS strip-by-strip; QK MMA fragments emitted per strip.
  * V is prefetched as a side DMA during the second-to-last K strip so
    it overlaps the QK compute.
  * Mask + online softmax + barriers between QK and PV.
  * P (post-softmax scores) cast and staged in SMEM, then PV MMA reads
    P from SMEM as the A operand and V from LDS as B.
"""

from max.gpu import lane_id
from max.gpu.sync import barrier

from .attention import AttentionRDNA
from .buffers import KBufferRDNA, VBufferRDNA
from .mma import rdna_mma


__extension AttentionRDNA:
    @always_inline
    def mha_prefill(mut self):
        comptime assert Self.BK == 32, "BK must be 32 for RDNA"

        @always_inline
        @__parameter
        def loop_over_kvcache[
            tile_size: Int
        ](kv_tile_start_row: Int, end: Int, not_last_iter: Bool):
            if self.mask_skip_and_advance(UInt32(kv_tile_start_row)):
                return

            var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

            var k_tile = Self.make_kv_tile(
                self.k,
                UInt32(self.get_batch_idx()),
                UInt32(kv_tile_start_row),
                UInt32(Self.kv_head_idx()),
                UInt32(kv_tile_num_rows),
            )
            var v_tile = Self.make_kv_tile(
                self.v,
                UInt32(self.get_batch_idx()),
                UInt32(kv_tile_start_row),
                UInt32(Self.kv_head_idx()),
                UInt32(kv_tile_num_rows),
            )

            self.zero_p_buffer()

            var k_buffer = KBufferRDNA[
                tensor_core_mma=Self.get_tensor_core_mma_qk(),
                BN=Self.BN,
                WN=Self.WN,
                BK=Self.BK,
                depth=Self.depth,
                num_threads=Self.num_threads,
                num_stages=Self.num_stages,
            ](k_tile, self.k_smem_ptr.as_unsafe_any_origin())

            var v_buffer = VBufferRDNA[
                tensor_core_mma=Self.get_tensor_core_mma_pv(),
                BN=Self.BN,
                BK=Self.BK,
                depth=Self.depth,
                num_threads=Self.num_threads,
                num_stages=Self.num_stages,
                num_warps_n=Self.num_warps_n,
            ](
                v_tile,
                self.v_smem_ptr.as_unsafe_any_origin(),
                total_rows=kv_tile_num_rows,
            )

            # ===== QK BK-strip loop =====
            k_buffer.load_from_dram()  # initial K strip 0

            comptime num_qk_strips = Self.depth // Self.BK
            comptime for strip in range(num_qk_strips):
                comptime if strip < num_qk_strips - 1:
                    k_buffer.load_from_dram()  # prefetch K strip strip+1
                    comptime if strip == num_qk_strips - 2:
                        # Prefetch V's first strip near the end of the QK
                        # loop so it overlaps QK compute.
                        v_buffer.load_from_dram()

                k_buffer.copy_to_shared[strip % 2]()
                barrier()

                # swap_a_b=True: K is supplied to hardware A, Q (b) to
                # hardware B; result is P^T.
                comptime for k_mma in range(Self.num_k_mmas2):
                    var a_reg = self.q_buffer.get_mma_tile[strip, k_mma]()
                    k_buffer.load_from_shared[k_mma]()
                    rdna_mma(
                        a_reg,
                        k_buffer.get_mma_tile(),
                        self.p_reg_buffer.get_reg_tile(),
                    )

                barrier()

            self.mask_apply(
                UInt32(kv_tile_start_row),
                UInt32(kv_tile_num_rows),
                not_last_iter,
            )

            barrier()
            self.online_softmax()
            barrier()

            # ===== PV BK-strip loop =====
            # V strip 0 was prefetched at the tail of the QK loop.
            comptime num_pv_strips = Self.BN // Self.BK
            comptime for strip in range(num_pv_strips):
                comptime if strip < num_pv_strips - 1:
                    v_buffer.load_from_dram()

                # Cast & write the strip-th P chunk to SMEM (PV's A
                # operand reads this back).
                self.p_reg_buffer.copy_to_shared[strip]()

                v_buffer.copy_to_shared[strip % 2]()
                barrier()

                comptime for k_mma in range(Self.num_k_mmas2):
                    var a_reg = self.p_reg_buffer.get_mma_tile[strip, k_mma]()
                    v_buffer.load_from_shared[k_mma]()
                    rdna_mma(
                        a_reg,
                        v_buffer.get_mma_tile(),
                        self.out_reg_buffer.get_reg_tile(),
                    )

                barrier()

        for i in range(0, self.num_keys, Self.BN):
            var end = min(i + Self.BN, self.num_keys)
            loop_over_kvcache[Self.BN](i, end, end != self.num_keys)

        self.apply_softmax_denominator()
        self.store_output()
