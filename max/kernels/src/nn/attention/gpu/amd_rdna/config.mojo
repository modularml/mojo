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
"""RDNA Wave32 attention config.

Single fixed config for both prefill and decode: MMA shape 16x16x16
(the only RDNA WMMA shape), no shared K/V SMEM, BK-strip K SMEM,
single-buffered, depth-padded V SMEM.
"""

from max.gpu import block_idx, lane_id
from std.math.uutils import umod, ufloordiv
from std.utils import IndexList

from nn.attention.mha_utils import MHAConfig

comptime RDNA_MMA_M = 16
comptime RDNA_MMA_N = 16
comptime RDNA_MMA_K = 16


@fieldwise_init
struct MHAAttentionConfigRDNA[token_gen: Bool, config: MHAConfig, group: Int](
    ImplicitlyCopyable
):
    """Holds the fixed RDNA Wave32 multi-head attention configuration and index math.

    Parameterizes a single attention kernel shape (16x16x16 WMMA) for both
    prefill and decode passes, and exposes the head, tile, and offset helpers
    the kernel uses to map block and lane IDs to Q/K/V rows in global memory.

    Parameters:
        token_gen: Whether the kernel is running in decode (token-by-token)
            generation mode versus prefill.
        config: The base multi-head attention configuration supplying head
            counts and tile dimensions.
        group: Group-query attention group size (number of query heads
            sharing one key/value head).
    """

    comptime shared_kv = False
    comptime full_kv = False
    comptime depth_padded = True
    comptime double_buffer = False
    comptime double_buffer_k_only = False

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        comptime if Self.token_gen:
            var group_idx = umod(lane_id(), Self.group)
            return block_idx.y * Self.group + group_idx
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return block_idx.y if not Self.token_gen else 0

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        return block_idx.y if Self.token_gen else ufloordiv(
            Self.q_head_idx(), Self.group
        )

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        return IndexList[3](RDNA_MMA_M, RDNA_MMA_N, RDNA_MMA_K)

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: Int]() -> UInt32:
        return UInt32(
            q_depth
            * (
                (
                    Self.kv_head_idx()
                    * Self.group if Self.token_gen else Self.q_head_idx()
                )
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            )
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: Int]() -> UInt32:
        return Self.get_q_offset[output_depth]()
