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

"""Host-side pins for the MLA decode SMEM stage counts.

The unified-gather path stages KV twice and gives the MMA-operand ring one
stage more than the gather ring, but only when the leftover SMEM covers it. The
sizing fails safe: when it does not fit, both rings keep the shared depth and
the kernel still produces the right answer, just slower. No correctness test
can tell the two apart, so the depth is pinned here rather than assumed.
"""

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
)


def _config[
    page_size: Int, num_heads: Int, unified_gather: Bool
]() -> MLA_SM100_Decode_Config:
    return MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=num_heads,
        depth=512,
        q_depth=576,
        dtype_size=1,
        kv_type_size=1,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        kv_mma_swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        page_size=page_size,
        decoding_warp_split_k=False,
        split_page_size=128,
        scale_block_size=0,
        native_fp8=True,
        per_token_scale_rope_aware=False,
        native_fp8_unified_gather=unified_gather,
    )


def _check[page_size: Int, num_heads: Int]() raises:
    comptime cfg = _config[page_size, num_heads, True]()
    if cfg.num_kv_mma_stages != cfg.num_kv_stages + 1:
        raise Error(
            "unified gather lost its extra MMA-operand stage at page=",
            page_size,
            " heads=",
            num_heads,
            ": gather ring ",
            cfg.num_kv_stages,
            ", MMA ring ",
            cfg.num_kv_mma_stages,
        )
    # The extra stage and its barrier pair are charged to `smem_used` by hand.
    # Under-counting them overflows the carveout at launch, not here.
    if cfg.smem_used > MLA_SM100_Decode_Config.sm100_smem_carveout:
        raise Error(
            "SMEM over the carveout at page=",
            page_size,
            " heads=",
            num_heads,
            ": ",
            cfg.smem_used,
            " > ",
            MLA_SM100_Decode_Config.sm100_smem_carveout,
        )

    # Every other path shares one depth between the two rings.
    comptime shared = _config[page_size, num_heads, False]()
    if shared.num_kv_mma_stages != shared.num_kv_stages:
        raise Error(
            "non-unified path grew a second ring depth at page=",
            page_size,
            " heads=",
            num_heads,
            ": ",
            shared.num_kv_stages,
            " vs ",
            shared.num_kv_mma_stages,
        )


def main() raises:
    _check[64, 8]()
    _check[128, 8]()
    _check[64, 16]()
    _check[128, 128]()
    print("decode config stage pins OK")
