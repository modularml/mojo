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
"""RDNA Wave32 MHA decode kernel.

Same recipe as prefill, plus split-K partitioning of the KV span across
blocks for grid-level parallelism.
"""

from std.collections import OptionalReg
from max.gpu import block_idx
from max.gpu.sync import barrier
from std.utils.numerics import get_accum_type, min_or_neg_inf

from nn.attention.mha_utils import get_start_and_end_for_partitions

from .attention import AttentionRDNA
from .buffers import KBufferRDNA, VBufferRDNA
from .mma import rdna_mma


__extension AttentionRDNA:
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
        comptime assert Self.BK == 32, "BK must be 32 for RDNA"
        comptime assert (
            Self.output_depth == Self.depth
        ), "RDNA decode requires output_depth == depth (no MLA)"

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
                UInt32(self.kv_head_idx()),
                UInt32(kv_tile_num_rows),
            )
            var v_tile = Self.make_kv_tile(
                self.v,
                UInt32(self.get_batch_idx()),
                UInt32(kv_tile_start_row),
                UInt32(self.kv_head_idx()),
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
                depth=Self.output_depth,
                num_threads=Self.num_threads,
                num_stages=Self.num_stages,
                num_warps_n=Self.num_warps_n,
            ](
                v_tile,
                self.v_smem_ptr.as_unsafe_any_origin(),
                total_rows=kv_tile_num_rows,
            )

            # ===== QK BK-strip loop =====
            k_buffer.load_from_dram()

            comptime num_qk_strips = Self.depth // Self.BK
            comptime for strip in range(num_qk_strips):
                comptime if strip < num_qk_strips - 1:
                    k_buffer.load_from_dram()
                    comptime if strip == num_qk_strips - 2:
                        v_buffer.load_from_dram()

                k_buffer.copy_to_shared[strip % 2]()
                barrier()

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
            comptime num_pv_strips = Self.BN // Self.BK
            comptime for strip in range(num_pv_strips):
                comptime if strip < num_pv_strips - 1:
                    v_buffer.load_from_dram()

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
            barrier()

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

        for i in range(start, end, Self.BN):
            var end_ = min(i + Self.BN, end)
            loop_over_kvcache[Self.BN](i, end_, end_ != end)

        self.apply_softmax_denominator()
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
