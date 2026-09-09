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
"""Simplified Transformer for Qwen3 embedding models without KV caching."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, TensorValueLike, ops
from max.nn.embedding import Embedding
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding


class Qwen3EmbeddingTransformerBlock(Module):
    """Transformer block for embedding models without KV caching.

    Similar to the standard TransformerBlock but simplified:
    - No kv_collection parameter
    - No freqs_cis parameter (RoPE handled in attention)
    - Attention layer handles its own position embeddings
    """

    def __init__(
        self,
        attention: Module,
        mlp: Module,
        attention_norm: Module,
        mlp_norm: Module,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.residual_multiplier = residual_multiplier

    def __call__(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass without KV caching.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_size]
            input_row_offsets: Row offsets for ragged tensor

        Returns:
            Output tensor of shape [total_seq_len, hidden_size]
        """
        residual_multiplier = ops.constant(
            self.residual_multiplier, x.dtype, device=x.device
        )

        # Attention with pre-normalization
        attn_out = self.self_attn(
            self.input_layernorm(x),
            input_row_offsets,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out

        # MLP with pre-normalization
        mlp_out = self.mlp(self.post_attention_layernorm(h))

        if self.residual_multiplier != 1.0:
            mlp_out = mlp_out * residual_multiplier

        return h + mlp_out


class Qwen3EmbeddingTransformer(Module):
    """Transformer model for embedding generation without KV caching.

    This is a simplified transformer that:
    - Does not use KVCache (no PagedCacheValues)
    - Designed for single-pass forward (no autoregressive generation)
    - Returns hidden states for pooling, not logits
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[Qwen3EmbeddingTransformerBlock],
        norm: Module,
        output: Linear,  # Still keep for weight sharing, but won't use
        embedding: Embedding,
        rope: RotaryEmbedding,
        embedding_multiplier: float = 1.0,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        """Initialize the embedding transformer.

        Args:
            dim: Hidden dimension size
            n_heads: Number of attention heads
            layers: List of transformer blocks
            norm: Final normalization layer
            output: Output projection (for weight sharing with embedding)
            embedding: Token embedding layer
            rope: Rotary position embedding
            embedding_multiplier: Multiplier for embeddings (if applicable)
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output  # Keep for weight sharing
        self.embed_tokens = embedding
        self.embedding_multiplier = embedding_multiplier
        self.rope = rope
        self.device = device

    def input_types(self) -> tuple[TensorType, ...]:
        """Get the input types for the graph.

        Returns:
            Tuple of (tokens, input_row_offsets, return_n_logits)
        """
        return (
            TensorType(
                DType.uint32, shape=("total_seq_len",), device=self.device
            ),
            TensorType(
                DType.uint32,
                shape=("batch_size_plus_1",),
                device=DeviceRef.CPU(),
            ),
            TensorType(DType.uint32, shape=(1,), device=DeviceRef.CPU()),
        )

    def __call__(
        self,
        tokens: TensorValueLike,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,  # Kept for interface compatibility
    ) -> TensorValue:
        """Forward pass for embedding generation.

        Args:
            tokens: Input token IDs [total_seq_len]
            input_row_offsets: Sequence boundaries [batch_size + 1]
            return_n_logits: Number of logits to return (unused for embeddings)

        Returns:
            Hidden states tensor [total_seq_len, hidden_size]
        """
        # Embed tokens
        h = self.embed_tokens(tokens)
        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Process through transformer layers
        input_row_offsets_device = input_row_offsets.to(self.device)
        for layer in self.layers:
            h = layer(h, input_row_offsets_device)
        h = self.norm(h)

        return h
