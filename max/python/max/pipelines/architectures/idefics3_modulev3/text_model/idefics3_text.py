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
"""Idefics3 language model (ModuleV3).

Extends the Llama3 architecture with multimodal embedding merging for
processing interleaved text and image tokens.
"""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import ops
from max.nn.kernels import scatter_nd_skip_oob_indices as _scatter_nd
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnLogits

from ...llama3_modulev3.layers.mlp import LlamaStackedMLP
from ...llama3_modulev3.layers.rotary_embedding import Llama3RotaryEmbedding
from ...llama3_modulev3.layers.transformer_block import LlamaTransformerBlock
from ...llama3_modulev3.model_config import Llama3Config

# Wrap scatter_nd_skip_oob_indices for use with V3 Tensors.
scatter_nd_skip_oob_indices = F.functional(_scatter_nd)


def merge_multimodal_embeddings(
    inputs_embeds: Tensor,
    multimodal_embeddings: Tensor,
    image_token_indices: Tensor,
) -> Tensor:
    """Merge multimodal embeddings into text embeddings at pre-computed indices.

    Uses scatter_nd_skip_oob_indices to place vision embeddings at positions
    specified by image_token_indices.

    Args:
        inputs_embeds: [num_tokens, hidden_size]
        multimodal_embeddings: [num_multimodal_tokens, hidden_size]
        image_token_indices: [num_multimodal_tokens]

    Returns:
        Copy of inputs_embeds with multimodal embeddings merged in.
    """
    indices_2d = F.unsqueeze(image_token_indices, -1)
    if multimodal_embeddings.dtype != inputs_embeds.dtype:
        multimodal_embeddings = F.cast(
            multimodal_embeddings, dtype=inputs_embeds.dtype
        )
    return scatter_nd_skip_oob_indices(
        input=inputs_embeds,
        updates=multimodal_embeddings,
        indices=indices_2d,
    )


class Idefics3TextModel(
    Module[
        [Tensor, PagedCacheValues, Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, ...],
    ]
):
    """Idefics3 text model (ModuleV3).

    A Llama3-based decoder with multimodal embedding merging: image embeddings
    are scattered into text embeddings before running through transformer layers.
    """

    def __init__(
        self,
        config: Llama3Config,
        image_token_id: int,
    ) -> None:
        super().__init__()
        self.image_token_id = image_token_id

        # RoPE embedding (same construction as Llama3TextModel).
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=Llama3Config.get_head_dim_from_config(config),
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )

        # Norm creation.
        if config.rms_norm_eps is None:
            raise ValueError(
                "rms_norm_eps cannot be None for model that uses RMSNorm."
            )
        create_norm = functools.partial(
            RMSNorm, config.hidden_size, eps=config.rms_norm_eps
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = create_norm()

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        # Build transformer layers.
        layers = []
        for i in range(config.num_hidden_layers):
            mlp: MLP | LlamaStackedMLP
            if config.stacked_mlp:
                mlp = LlamaStackedMLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                )
            else:
                mlp = MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                )

            layers.append(
                LlamaTransformerBlock(
                    attention=AttentionWithRope(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        scale=config.attention_multiplier,
                        has_bias=config.attention_bias,
                        stacked_qkv=config.stacked_qkv,
                        clip_qkv=config.clip_qkv,
                    ),
                    mlp=mlp,
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = ModuleList(layers)
        self.return_logits = config.return_logits
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling

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
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        # Get text embeddings and merge in image embeddings.
        h = self.embed_tokens(tokens)
        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )

        if self.embedding_multiplier != 1.0:
            h = h * F.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Run through transformer layers.
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Compute logits.
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            # Descending range [n, n-1, ..., 1]: subtracting from each
            # row-end offset gives the indices of the last n tokens per row.
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

        if self.logits_scaling != 1.0:
            last_logits = last_logits / self.logits_scaling
            if logits is not None:
                logits = logits / self.logits_scaling

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        return ret_val


class Idefics3Language(Module[..., tuple[Tensor, ...]]):
    """Top-level language model wrapper (ModuleV3).

    Unflattens the variadic KV cache arguments and delegates to
    :class:`Idefics3TextModel`.
    """

    def __init__(
        self,
        config: Llama3Config,
        image_token_id: int,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Idefics3TextModel(config, image_token_id)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        symbolic_inputs = self.kv_params.unflatten_kv_inputs(
            iter(variadic_args)
        )
        assert isinstance(symbolic_inputs, KVCacheInputs)
        kv_collections = symbolic_inputs.inputs

        return self.language_model(
            tokens,
            kv_collections[0],
            return_n_logits,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
        )
