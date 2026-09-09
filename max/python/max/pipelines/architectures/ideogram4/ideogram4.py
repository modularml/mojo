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
"""Ideogram 4 flow-matching Diffusion Transformer (single-stream DiT).

Graph-API (ModuleV3) reimplementation of
``ideogram4.modeling_ideogram4.Ideogram4Transformer``. Module names match the
diffusers ``transformer/`` checkpoint exactly so weight loading is identity
(plus FP8 dequant; see ``weight_adapters.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module
from max.experimental.nn.norm import LayerNorm, RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType

from .layers.attention import Ideogram4Attention
from .layers.embeddings import (
    Ideogram4EmbedScalar,
    Ideogram4MRoPE,
)
from .layers.fp8_linear import (
    Ideogram4FP8Linear,
    fp8_matmul_2d,
    fp8_quantize_2d,
)
from .model_config import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Config,
)


class Ideogram4MLP(Module[[Tensor], Tensor]):
    """SwiGLU feed-forward (``w2(silu(w1 x) * w3 x)``) as native FP8 GEMMs."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        self.w1 = Ideogram4FP8Linear(dim, hidden_dim)
        self.w2 = Ideogram4FP8Linear(hidden_dim, dim)
        self.w3 = Ideogram4FP8Linear(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Fuse the gate (w1) and up (w3) projections into one FP8 GEMM: both
        # read the same input and share the (dim -> hidden_dim) shape, so the
        # activation is quantized once and the row-stacked FP8 weight (with its
        # stacked rowwise scales) runs as a single wider matmul, halving the
        # projection launch + activation-quant count. The split halves are
        # numerically identical to two separate FP8 matmuls.
        b, seq, dim = x.shape
        hidden_dim = int(self.w1.weight.shape[0])
        x_2d = F.reshape(x, [b * seq, dim])
        x_fp8, x_scales = fp8_quantize_2d(x_2d)

        gate_up_weight = F.concat((self.w1.weight, self.w3.weight))
        gate_up_scale = F.concat((self.w1.weight_scale, self.w3.weight_scale))
        gate_up = fp8_matmul_2d(x_fp8, x_scales, gate_up_weight, gate_up_scale)
        gate, up = gate_up.split([hidden_dim, hidden_dim], axis=-1)
        hidden = F.silu(gate) * up

        hidden_fp8, hidden_scales = fp8_quantize_2d(hidden)
        out_2d = fp8_matmul_2d(
            hidden_fp8, hidden_scales, self.w2.weight, self.w2.weight_scale
        )
        return F.reshape(out_2d, [b, seq, int(self.w2.weight.shape[0])])


class Ideogram4TransformerBlock(Module[..., Tensor]):
    """Single-stream DiT block with tanh-gated AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adaln_dim: int,
    ) -> None:
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = Linear(adaln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        adaln_input: Tensor,
    ) -> Tensor:
        mod = self.adaln_modulation(adaln_input)  # (B, 1, 4*hidden)
        scale_msa, gate_msa, scale_mlp, gate_mlp = F.chunk(mod, 4, axis=-1)
        gate_msa = F.tanh(gate_msa)
        gate_mlp = F.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa, cos=cos, sin=sin
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x


class Ideogram4FinalLayer(Module[..., Tensor]):
    """Final AdaLN + projection to ``out_channels``."""

    def __init__(
        self, hidden_size: int, out_channels: int, adaln_dim: int
    ) -> None:
        self.norm_final = LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False, use_bias=False
        )
        self.linear = Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = Linear(adaln_dim, hidden_size, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer2DModel(Module[..., Sequence[Tensor]]):
    """Ideogram 4 flow-matching transformer producing velocity predictions."""

    def __init__(self, config: Ideogram4Config) -> None:
        self.config = config
        self.max_device = config.device
        self.max_dtype = config.dtype

        dim = config.emb_dim
        self.in_channels = config.in_channels
        self.llm_features_dim = config.llm_features_dim

        self.input_proj = Linear(config.in_channels, dim, bias=True)
        self.llm_cond_norm = RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = Linear(config.llm_features_dim, dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(dim, device=config.device)
        self.adaln_proj = Linear(dim, config.adaln_dim, bias=True)
        self.embed_image_indicator = Embedding(2, dim=dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=config.head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
            device=config.device,
        )

        self.layers: ModuleList[Ideogram4TransformerBlock] = ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=dim,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    norm_eps=config.norm_eps,
                    adaln_dim=config.adaln_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=dim,
            out_channels=config.in_channels,
            adaln_dim=config.adaln_dim,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.max_dtype,
                shape=["batch", "seq", self.in_channels],
                device=self.max_device,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch", "seq", self.llm_features_dim],
                device=self.max_device,
            ),
            TensorType(DType.float32, shape=["batch"], device=self.max_device),
            TensorType(
                DType.int64, shape=["batch", "seq", 3], device=self.max_device
            ),
            TensorType(
                DType.int64, shape=["batch", "seq"], device=self.max_device
            ),
        )

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        x, llm_features, t, position_ids, indicator = args[:5]

        dtype = self.max_dtype
        x = F.cast(x, dtype)
        llm_features = F.cast(llm_features, dtype)

        llm_token_mask = F.unsqueeze(
            F.cast(indicator == LLM_TOKEN_INDICATOR, dtype), -1
        )
        output_image_mask = F.unsqueeze(
            F.cast(indicator == OUTPUT_IMAGE_INDICATOR, dtype), -1
        )

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)  # (B, dim)
        t_cond = F.unsqueeze(t_cond, 1)  # (B, 1, dim)
        adaln_input = F.silu(self.adaln_proj(t_cond))  # (B, 1, adaln_dim)

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        img_idx = F.cast(indicator == OUTPUT_IMAGE_INDICATOR, DType.int64)
        h = h + self.embed_image_indicator(img_idx)

        cos, sin = self.rotary_emb(position_ids)
        cos = F.cast(cos, h.dtype)
        sin = F.cast(sin, h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin, adaln_input)

        out = self.final_layer(h, adaln_input)
        return (F.cast(out, DType.float32),)
