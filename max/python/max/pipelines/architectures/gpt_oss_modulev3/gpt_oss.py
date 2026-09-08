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

"""Implements the GPT OSS model."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)

from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.rms_norm import GptOssRMSNorm
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(
    Module[
        [Tensor, PagedCacheValues, PagedCacheValues, Tensor, Tensor],
        tuple[Tensor, ...],
    ]
):
    """The GPT OSS language model.

    Decoder-only Transformer with MoE feed-forward, rotary embeddings (YARN),
    and mixed attention (full + sliding window).
    """

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Create YARN scaling params if configured
        assert config.rope_scaling is not None, (
            "RoPE scaling is required for GPT-OSS models"
        )
        assert isinstance(config.rope_scaling, YarnScalingParams), (
            "Only YARN scaling is supported for GPT-OSS models"
        )
        yarn_scaling_params: YarnScalingParams = config.rope_scaling

        # RoPE with YARN scaling for full and window attention layers
        rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = GptOssRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        create_norm = functools.partial(
            GptOssRMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(config.num_hidden_layers):
            if i < len(config.layer_types):
                layer_type = config.layer_types[i]
            else:
                layer_type = "full_attention"
            mask_variant = (
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if layer_type == "sliding_attention"
                else MHAMaskVariant.CAUSAL_MASK
            )
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                GptOssTransformerBlock(
                    attention=GptOssAttention(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx=layer_idx_in_cache,
                        local_window_size=config.sliding_window,
                        has_bias=config.attention_bias,
                        mask_variant=mask_variant,
                    ),
                    mlp=GptOssMoE(config),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                )
            )

        self._layer_kv_key = [
            config.layer_types[i]
            if i < len(config.layer_types)
            else "full_attention"
            for i in range(config.num_hidden_layers)
        ]

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def forward(
        self,
        tokens: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)
        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }
        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

        # Get last token logits only (no variable logits support).
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            # Take only the device 0 logits to device-to-host transfer.
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )

        # For now, simplified to return last token only
        # TODO: Handle VARIABLE and ALL logits cases for distributed processing
        return (last_logits,)


class GptOss(Module[..., tuple[Tensor, ...]]):
    """The GPT OSS model."""

    def __init__(
        self,
        config: GptOssConfig,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = GptOssTextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        assert isinstance(self.kv_params, MultiKVCacheParams)
        sliding_inputs, global_inputs = self.kv_params.unflatten_basic_kv_tree(
            kv_inputs
        )
        sliding_kv = PagedCacheValues.from_upstream(
            sliding_inputs, tokens.mapping
        )
        global_kv = PagedCacheValues.from_upstream(
            global_inputs, tokens.mapping
        )
        return self.language_model(
            tokens, sliding_kv, global_kv, return_n_logits, input_row_offsets
        )
