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

"""Gemma4 vision (SigLIP) attention for the ModuleV3 API."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged_gpu,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    apply_multidimensional_rope as _apply_multidimensional_rope,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)

from ..layers.rms_norm import Gemma4RMSNorm

# The 2-D RoPE helper is gemma4-specific, so it is wrapped here instead of
# being registered in common_layers/functional_kernels.py.
apply_multidimensional_rope = F.functional(_apply_multidimensional_rope)


class Gemma4VisionAttention(Module[..., Tensor]):
    """Ragged bidirectional attention: per-head q/k norms, weightless v norm,
    2-D RoPE, scale=1.0. Port of the graph arch's Gemma4VisionAttention.
    """

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, layer_idx: int
    ) -> None:
        super().__init__()
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        self.layer_idx = layer_idx
        self.head_dim = vision_cfg.head_dim
        self.num_attention_heads = vision_cfg.num_attention_heads
        self.num_key_value_heads = vision_cfg.num_key_value_heads

        bias = vision_cfg.attention_bias
        self.q_proj = Linear(
            vision_cfg.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            vision_cfg.hidden_size,
            bias=bias,
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=vision_cfg.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=vision_cfg.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=vision_cfg.rms_norm_eps, with_weight=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        xq = self.q_norm(
            self.q_proj(hidden_states).reshape(
                (-1, self.num_attention_heads, self.head_dim)
            )
        )
        xk = self.k_norm(
            self.k_proj(hidden_states).reshape(
                (-1, self.num_key_value_heads, self.head_dim)
            )
        )
        xv = self.v_norm(
            self.v_proj(hidden_states).reshape(
                (-1, self.num_key_value_heads, self.head_dim)
            )
        )

        # 2-D RoPE on Q/K; broadcast a heads dim onto freqs_cis first.
        freqs_bcast = freqs_cis.reshape((-1, 1, self.head_dim // 2, 2))
        xq = apply_multidimensional_rope(xq, freqs_bcast, ndim=2).reshape(
            (-1, self.num_attention_heads, self.head_dim)
        )
        xk = apply_multidimensional_rope(xk, freqs_bcast, ndim=2).reshape(
            (-1, self.num_key_value_heads, self.head_dim)
        )

        output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0,
        ).reshape((-1, self.num_attention_heads * self.head_dim))
        return self.o_proj(output)
