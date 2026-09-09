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
"""FLUX.2 ModuleV3 dual-stream and single-stream transformer blocks.

Single-device only: the legacy ``Allreduce`` / ``Signals`` plumbing and
sharded attention paths are deferred to a multi-device follow-up.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.norm import LayerNorm
from max.experimental.tensor import Tensor

from .layers import Flux2Attention, Flux2FeedForward, Flux2ParallelSelfAttention


class Flux2TransformerBlock(Module[..., tuple[Tensor, Tensor]]):
    """Dual-stream image+text transformer block.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.flux2.Flux2TransformerBlock`
    single-device path: pre-norm + modulation feeds joint attention,
    then per-stream gate + residual + post-norm + FF + gate + residual.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        self.norm1 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.norm1_context = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.norm2 = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.norm2_context = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
        )
        self.ff = Flux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )
        self.ff_context = Flux2FeedForward(
            dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb_mod_params_img: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ],
        temb_mod_params_txt: tuple[
            tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
        ],
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the dual-stream block.

        Args:
            hidden_states: Image tokens ``[B, S_img, D]``.
            encoder_hidden_states: Text tokens ``[B, S_txt, D]``.
            temb_mod_params_img: Two ``(shift, scale, gate)`` triples
                for the image stream (msa, mlp).
            temb_mod_params_txt: Two ``(shift, scale, gate)`` triples
                for the text stream.
            image_rotary_emb: Optional ``(cos, sin)`` RoPE.

        Returns:
            Tuple ``(encoder_hidden_states, hidden_states)``, matching
            the legacy block's return order so the top-level transformer
            can keep the same outer-loop wiring.
        """
        (
            (shift_msa, scale_msa, gate_msa),
            (
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ),
        ) = temb_mod_params_img
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = temb_mod_params_txt

        norm_hidden_states = (1 + scale_msa) * self.norm1(
            hidden_states
        ) + shift_msa
        norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(
            encoder_hidden_states
        ) + c_shift_msa

        result = self.attn(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if not isinstance(result, tuple):
            raise ValueError("Expected tuple from dual-stream attention")
        attn_output, context_attn_output = result

        # Image stream: gate + residual, then norm2 + modulation + FF + gate.
        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = (
            self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        )
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # Text stream: same shape.
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_msa * context_attn_output
        )
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp)
            + c_shift_mlp
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        # Float16 saturation, matching legacy block.
        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = F.min(
                F.max(encoder_hidden_states, -65504), 65504
            )

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(Module[..., Tensor]):
    """Single-stream parallel-attention+MLP transformer block.

    The block expects the caller to have concatenated text and image
    tokens along the sequence axis ahead of time (matching the legacy
    top-level transformer's ``hidden_states_d`` construction); the
    split-back-into-streams handling stays at the transformer level.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        self.norm = LayerNorm(
            dim, eps=eps, elementwise_affine=False, use_bias=False
        )
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb_mod_params: tuple[Tensor, Tensor, Tensor],
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Run the single-stream block: pre-norm + mod + parallel attn+MLP."""
        mod_shift, mod_scale, mod_gate = temb_mod_params
        norm_hidden_states = (1 + mod_scale) * self.norm(
            hidden_states
        ) + mod_shift
        attn_output = self.attn(
            norm_hidden_states, image_rotary_emb=image_rotary_emb
        )
        hidden_states = hidden_states + mod_gate * attn_output

        if hidden_states.dtype == DType.float16:
            hidden_states = F.min(F.max(hidden_states, -65504), 65504)

        return hidden_states
