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
"""The global model that decides what happens in the song, frame by frame.

A stock Qwen3 -- 36 layers, grouped-query attention with per-head QK norm, SwiGLU
-- so the layers are the shared ones: ``AttentionWithRope`` with ``use_qk_norm``
is the Qwen3 attention, and the MLP and norms are the common ones. Three things
about how this one is *driven* differ from text generation, and they are why it
gets its own module rather than reusing an existing Qwen3 port.

It never sees a token id. The prompt arrives as embeddings the host looked up
once, and every subsequent step's input is a frame of eight audio codes summed
into a single vector, so the 200000-row embedding table has no reason to be on the
device at all -- 1.6 GiB saved on a 22 GiB card. What replaces it is a slice: the
16385 rows the model may actually emit.

Prefill and decode are one graph. Attention is ragged, so a batch of two
sequences of length T and a batch of two single tokens differ only in the row
offsets, and 36 layers are worth compiling once.

Its output is used twice over. The sampled code chooses the frame's semantic
content, and the hidden state that produced it is both the depth decoder's
conditioning and the first eighth of what the diffusion stage denoises against.
"""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module, ModuleList, RMSNorm
from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType
from max.nn.kv_cache import KVCacheParams, PagedCacheValues

from ..model_config import LanguageModelConfig


class TransformerBlock(Module[..., Tensor]):
    """One Qwen3 layer: pre-norm attention, pre-norm SwiGLU, both residual."""

    def __init__(
        self,
        config: LanguageModelConfig,
        layer_index: int,
        rope: RotaryEmbedding,
        kv_params: KVCacheParams,
    ) -> None:
        self.self_attn = AttentionWithRope(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            layer_idx=layer_index,
            # Per-head norm on Q and K before RoPE is what makes this Qwen3
            # rather than Llama.
            use_qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        attended = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
        )
        hidden = x + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LanguageModel(Module[..., tuple[Tensor, Tensor]]):
    """Qwen3 over input embeddings, with a head cut down to the audio rows."""

    def __init__(
        self,
        config: LanguageModelConfig,
        kv_params: KVCacheParams,
        device: Device,
        max_seq_len: int,
    ) -> None:
        """Builds the body and the head.

        Args:
            config: The model's shape.
            kv_params: The cache the attention layers write through.
            device: Where the weights live. Also where the rotary table is
                built, which is the one thing this module cannot defer to
                :meth:`Module.to`.
            max_seq_len: Longest sequence this graph will see. It only sizes the
                rotary table, whose rows do not depend on it -- but the table is a
                baked constant, so asking for the config's 10240 rather than the
                request's few hundred puts 21 MB in the compiled graph.
        """
        self.config = config
        self.kv_params = kv_params
        self.device = device
        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            head_dim=config.head_dim,
            # Safetensors keeps rotary halves split; only GGUF interleaves them.
            interleaved=False,
        )
        self.layers = ModuleList(
            [
                TransformerBlock(config, index, self.rope, kv_params)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = Linear(
            config.hidden_size, config.head_vocab_size, bias=False
        )

    @property
    def dtype(self) -> DType:
        """The dtype the model computes in, taken from its parameters."""
        return self.head.weight.dtype

    def input_types(self, batch: int) -> tuple[TensorType | BufferType, ...]:
        """The graph inputs: one ragged batch of embeddings, and the cache.

        Args:
            batch: Sequences per step, which the row offsets are sized by. The
                positions per sequence are not fixed: prefill and decode differ
                only in the offsets, which is what lets them share a graph.
        """
        return (
            TensorType(
                self.dtype,
                ["total_seq_len", self.config.hidden_size],
                self.device,
            ),
            TensorType(DType.uint32, [batch + 1], self.device),
            *self.kv_params.flattened_kv_inputs(),
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the body and reads the last position of every sequence.

        Args:
            inputs_embeds: ``(total_seq_len, hidden_size)``, the sequences of the
                batch laid end to end as ragged attention wants them.
            kv_collection: The paged cache for this step.
            input_row_offsets: ``(batch + 1)`` start of each sequence, so that
                prefill and decode are the same graph.

        Returns:
            The normalized hidden state of each sequence's last position, and its
            logits over the emittable rows. Both are read on the host: the hidden
            state conditions the frame, the logits choose its semantic code.
        """
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, kv_collection, input_row_offsets)
        last = self.norm(F.gather(hidden, input_row_offsets[1:] - 1, axis=0))
        return last, self.head(last)
