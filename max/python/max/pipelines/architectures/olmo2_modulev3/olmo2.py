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
"""Implements the Olmo2 model."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import ops
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.transformer import ReturnLogits

from .layers.attention import Olmo2Attention
from .layers.rms_norm import Olmo2RMSNorm
from .layers.transformer import Olmo2TransformerBlock
from .model_config import Olmo2Config


class Olmo2TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The Olmo2 language model.

    Decoder-only Transformer with standard MLP feed-forward,
    rotary embeddings, and Q/K normalization after projections.
    """

    def __init__(self, config: Olmo2Config) -> None:
        super().__init__()
        self.devices = config.devices

        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=config.interleaved_rope_weights,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = Olmo2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        create_norm = functools.partial(
            Olmo2RMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        kv_params = config.kv_params
        assert isinstance(kv_params, KVCacheParams)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(
                Olmo2TransformerBlock(
                    attention=Olmo2Attention(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params,
                        layer_idx=i,
                        scale=config.attention_multiplier,
                        has_bias=config.attention_bias,
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                    ),
                    post_attention_layernorm=create_norm(),
                    post_feedforward_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.tie_word_embeddings = config.tie_word_embeddings
        self.embedding_multiplier = config.embedding_multiplier

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Compute logits from hidden states, handling weight tying."""
        if self.tie_word_embeddings:
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
            )

        # Compute logits based on return mode.
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        return ret_val


class Olmo2(Module[..., tuple[Tensor, ...]]):
    """The Olmo2 model."""

    def __init__(
        self,
        config: Olmo2Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Olmo2TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        symbolic_inputs = self.kv_params.unflatten_kv_inputs(
            iter(variadic_args)
        )
        assert isinstance(symbolic_inputs, KVCacheInputs)
        kv_collections = symbolic_inputs.inputs
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
