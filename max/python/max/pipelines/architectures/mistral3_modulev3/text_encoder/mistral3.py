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

"""Mistral3 text encoder transformer without KV cache dependency.

This is a standalone transformer implementation for text encoding that does not
require KV cache. Suitable for single-pass encoding in diffusion pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType

from .attention import EncoderAttention

if TYPE_CHECKING:
    from .model_config import Mistral3TextEncoderConfig


class Mistral3MLP(Module[[Tensor], Tensor]):
    """Mistral3 MLP with SiLU gate activation."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class EncoderTransformerBlock(Module[..., Tensor]):
    """Transformer block for encoder-only models without KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        scale: float,
    ) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            scale=scale,
        )
        self.mlp = Mistral3MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        """Forward pass without KV cache.

        Args:
            x: Input hidden states [seq_len, hidden_dim]
            rope: RoPE embedding module

        Returns:
            Output hidden states [seq_len, hidden_dim]
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Mistral3TextEncoderTransformer(Module[[Tensor], Tensor]):
    """Mistral3 text encoder transformer without KV cache dependency.

    Encodes tokens and returns fused prompt embeddings by stacking hidden
    states from the configured layers and merging the layer/hidden dimensions.
    """

    def __init__(self, config: Mistral3TextEncoderConfig) -> None:
        super().__init__()

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.device = config.device
        self._hidden_state_layers = set(config.hidden_state_layers)
        self._sorted_hidden_state_layers = sorted(config.hidden_state_layers)
        self._output_seq_len: int | None = config.output_seq_len

        # Compute the rotary-embedding cos/sin lookup table on CPU to
        # match the legacy graph API's ``max.nn.RotaryEmbedding`` (see
        # ``max/python/max/nn/rotary_embedding.py``).  GPU and CPU fp32
        # transcendentals can disagree by ~1 ULP, and even tiny rope
        # offsets shift the entire denoising trajectory once they feed
        # text conditioning into 50 FLUX.2 diffusion steps -- enough to
        # drop V3 vs V2 SSIM well below the verify_pipelines threshold.
        # The attention code transfers the resulting freqs to ``x.device``
        # at use time.
        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=CPU(),
            head_dim=config.head_dim,
            interleaved=False,
        )

        self.layers = ModuleList(
            [
                EncoderTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    scale=config.attention_multiplier,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for compilation."""
        return (
            TensorType(
                DType.int64,
                shape=["total_seq_len"],
                device=self.device,
            ),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Forward pass returning fused prompt embeddings.

        Runs the transformer up to the last configured layer, collects hidden
        states from the configured layers, then stacks and reshapes them into
        a single prompt-embedding tensor.

        Args:
            tokens: Input token IDs [total_seq_len]

        Returns:
            Tensor of shape [1, seq_len, num_layers * hidden_dim] with the
            selected hidden states stacked and the layer/hidden dimensions
            merged, ready for the diffusion transformer.
        """
        h = self.embed_tokens(tokens)

        selected: dict[int, Tensor] = {}
        # Use 1-indexed layer numbering to match diffusers convention
        # where hidden_states[0] = embedding output and hidden_states[k]
        # = output of transformer layer k-1. So hidden_states[10] means
        # the output after the 10th transformer block (0-indexed layer 9).
        max_layer = self._sorted_hidden_state_layers[-1] - 1
        for i, layer in enumerate(self.layers):
            h = layer(h, self.rope)
            if (i + 1) in self._hidden_state_layers:
                selected[i + 1] = h
            if i == max_layer:
                break

        hidden_states = [selected[i] for i in self._sorted_hidden_state_layers]

        # Stack [L tensors of (S, D)] -> [L, S, D]
        # then fuse into [1, S, L*D] for the diffusion transformer.
        stacked = F.stack(hidden_states, axis=0)  # [L, S, D]
        stacked = F.unsqueeze(stacked, axis=0)  # [1, L, S, D]
        stacked = F.permute(stacked, [0, 2, 1, 3])  # [1, S, L, D]
        # Read L and D directly from the tensor dims to avoid any Python-side
        # constant that could force a device sync at eager execution time.
        seq_len = stacked.shape[1]
        embeds = F.reshape(
            stacked, [1, seq_len, stacked.shape[2] * stacked.shape[3]]
        )

        # When ``output_seq_len`` is configured, left-pad with zeros so the
        # downstream transformer (FLUX.2 joint attention) sees the static
        # text sequence length it was trained with.  Without this, the
        # transformer's rope position table covers tokens 0..(pad-1) for
        # text but the actual encoder output only fills the first
        # ``seq_len`` slots; image tokens then index the rope table at
        # offsets that mistake them for text positions, scrambling the
        # spatial layout of the generated image.  Matches V2's
        # ``Mistral3TextEncoderTransformer`` output-padding path.
        if self._output_seq_len is not None:
            pad_count = self._output_seq_len - seq_len
            zero = F.constant(0, dtype=embeds.dtype, device=self.device)
            zero = F.reshape(zero, [1, 1, 1])
            zeros = F.broadcast_to(zero, shape=[1, pad_count, embeds.shape[2]])
            embeds = F.concat([zeros, embeds], axis=1)

        return embeds
