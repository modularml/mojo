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

"""Implements the Olmo3 model."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
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
from max.pipelines.architectures.olmo2_modulev3.layers.rms_norm import (
    Olmo2RMSNorm,
)
from max.pipelines.architectures.olmo3.layers.transformer import (
    Olmo3TransformerBlock,
)

from .layers.attention import Olmo3Attention
from .model_config import Olmo3Config


class Olmo3TextModel(
    Module[
        [Tensor, PagedCacheValues, PagedCacheValues, Tensor, Tensor],
        tuple[Tensor],
    ]
):
    """The Olmo3 language model.

    Decoder-only Transformer with standard MLP feed-forward,
    rotary embeddings (YARN), and mixed attention (full + sliding window).

    Olmo3 includes Q and K normalization after Q/K projections.
    """

    def __init__(self, config: Olmo3Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rope_scaling is not None:
            if not isinstance(config.rope_scaling, YarnScalingParams):
                raise ValueError(
                    "Only YARN scaling is supported for Olmo3 models"
                )
            yarn_scaling_params: YarnScalingParams = config.rope_scaling
        else:
            yarn_scaling_params = YarnScalingParams(
                factor=32.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_max_position_embeddings=4096,
                truncate=False,
            )

        # Create YARN RoPE for full attention layers
        yarn_rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )

        # Create basic RoPE for sliding attention layers
        basic_rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = Olmo2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
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
            # Use basic RoPE for sliding attention, YARN RoPE for full attention
            layer_rope = (
                basic_rope if layer_type == "sliding_attention" else yarn_rope
            )
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                Olmo3TransformerBlock(
                    attention=Olmo3Attention(
                        rope=layer_rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx=layer_idx_in_cache,
                        local_window_size=config.sliding_window,
                        has_bias=config.attention_bias,
                        mask_variant=mask_variant,
                        use_qk_norm=config.use_qk_norm,
                        qk_norm_eps=config.qk_norm_eps,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        bias=config.attention_bias,
                        activation_function=config.hidden_activation,
                    ),
                    post_attention_layernorm=create_norm(),
                    post_feedforward_layernorm=create_norm(),
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
        self.tie_word_embeddings = config.tie_word_embeddings

    def forward(
        self,
        tokens: Tensor,
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor]:
        h = self.embed_tokens(tokens)
        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets,
            )

        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_token_h = self.norm(last_token_h)

        if self.tie_word_embeddings:
            last_logits = F.cast(
                last_token_h @ self.embed_tokens.weight.T,
                DType.float32,
            )
        else:
            assert self.lm_head is not None
            last_logits = F.cast(
                self.lm_head(last_token_h),
                DType.float32,
            )

        return (last_logits,)


class Olmo3(Module[[Tensor, Tensor, Tensor], tuple[Tensor]]):
    """The Olmo3 model."""

    def __init__(
        self,
        config: Olmo3Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Olmo3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor]:
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
