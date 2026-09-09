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

# ===----------------------------------------------------------------------=== #
# GritLM ModuleV3 graph implementation.
#
# CausalLM path only (no pooling head — embedding mode not supported in MAX).
# All 32 layers use sliding window attention (window=4096).
# ===----------------------------------------------------------------------=== #
"""Implements the GritLM causal LM model using the ModuleV3 API."""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)

from .layers.attention import GritLMAttention
from .layers.transformer_block import GritLMTransformerBlock
from .model_config import GritLMConfig


class GritLMTextModel(
    Module[
        [Tensor, PagedCacheValues, Tensor, Tensor],
        tuple[Tensor | TensorValue, ...],
    ]
):
    """GritLM decoder-only transformer (CausalLM path).

    32 layers, GQA (32 Q / 8 KV heads), SwiGLU, RMSNorm.
    Sliding window attention (window=4096) on every layer.
    Separate lm_head (tie_word_embeddings=false).
    """

    def __init__(self, config: GritLMConfig) -> None:
        super().__init__()

        if config.rms_norm_eps is None:
            raise ValueError("rms_norm_eps cannot be None for GritLM.")

        create_norm: Callable[..., Module[[Tensor], Tensor]] = (
            functools.partial(
                RMSNorm, config.hidden_size, eps=config.rms_norm_eps
            )
        )

        # GritLM uses standard Mistral RoPE: theta=10000, no scaling.
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=GritLMConfig.get_head_dim_from_config(config),
            interleaved=config.interleaved_rope_weights,
            scaling_params=None,
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = create_norm()

        # GritLM: tie_word_embeddings=false — always has a separate lm_head
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        layers = []
        for i in range(config.num_hidden_layers):
            attention = GritLMAttention(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=i,
                sliding_window=config.sliding_window,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                clip_qkv=config.clip_qkv,
            )
            mlp = MLP(
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
            )
            layers.append(
                GritLMTransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                )
            )

        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = config.logits_scaling

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits cast to float32.

        Args:
            h: Hidden-state tensor of shape ``(seq_len, hidden_size)``.

        Returns:
            Float32 logit tensor of shape ``(seq_len, vocab_size)``.
        """

        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor | TensorValue, ...]:
        """Run the full GritLM decoder stack and return logits/hidden states.

        Args:
            tokens: Flat token IDs, shape ``(total_seq_len,)``.
            kv_collection: Paged KV cache for all layers.
            return_n_logits: Controls variable-logit return mode.
            input_row_offsets: Ragged-batch row delimiters,
                shape ``(batch_size + 1,)``.

        Returns:
            Tuple whose layout depends on ``return_logits`` /
            ``return_hidden_states``:
                ``(last_logits,)``
                ``(last_logits, logits, offsets)``
                ``(last_logits, hidden_states)``
                ``(last_logits, logits, offsets, hidden_states)``
        """

        h = self.embed_tokens(tokens)

        for idx, layer in enumerate(self.layers):
            layer_idx_t = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_t,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits: Tensor | None = None
        offsets: Tensor | TensorValue | None = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offset_tensor = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offset_tensor, shape=(-1,))
            logits = self._compute_logits(
                self.norm(F.gather(h, last_indices, axis=0))
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[Tensor | TensorValue, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)
        if self.return_hidden_states == ReturnHiddenStates.LAST:
            ret_val += (last_h,)
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            ret_val += (self.norm(h),)
        return ret_val


class GritLM(Module[..., tuple[Tensor | TensorValue, ...]]):
    """Top-level GritLM model."""

    def __init__(
        self, config: GritLMConfig, kv_params: KVCacheParamInterface
    ) -> None:
        super().__init__()
        self.language_model = GritLMTextModel(config)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor | TensorValue, ...]:
        """Unpack KV cache tensors and delegate to ``GritLMTextModel``.

        Args:
            tokens: Flat int64 token IDs.
            return_n_logits: Single-element int64 logit-count control tensor.
            input_row_offsets: uint32 ragged-batch row delimiters.
            *variadic_args: Flattened paged KV cache tensors (all layers,
                all devices) produced by ``KVCacheParamInterface.get_symbolic_inputs``.

        Returns:
            Output tuple forwarded from ``GritLMTextModel.forward``.
        """

        kv_inputs = iter(x._graph_value for x in variadic_args)
        unflattened = self.kv_params.get_symbolic_inputs().unflatten(kv_inputs)
        assert isinstance(unflattened, KVCacheInputs), (
            f"Expected KVCacheInputs, got {type(unflattened)}"
        )
        kv_collections = unflattened.inputs
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
