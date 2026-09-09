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
"""Implements the DeepseekV2 model using the ModuleV3 API."""

from __future__ import annotations

import functools
import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.moe import MoE
from max.experimental.nn.common_layers.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams

from .layers.moe_gate import DeepSeekV2MoEGate
from .layers.rotary_embedding import DeepseekYarnRotaryEmbedding
from .layers.transformer_block import DeepseekV2TransformerBlock
from .model_config import DeepseekV2Config


def _get_mlp(
    config: DeepseekV2Config, layer_idx: int
) -> Module[[Tensor], Tensor]:
    """Returns either an MoE or MLP module for the given layer index."""
    use_moe = (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )
    if use_moe:
        return MoE(
            hidden_dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                DeepSeekV2MoEGate,
                topk_method=config.topk_method,
                n_group=config.n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
            ),
            has_shared_experts=True,
            shared_experts_dim=config.n_shared_experts
            * config.moe_intermediate_size,
        )
    return MLP(
        hidden_dim=config.hidden_size,
        feed_forward_length=config.intermediate_size,
        bias=False,
    )


class DeepseekV2TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The DeepseekV2 language model.

    Decoder-only Transformer with Multi-Latent Attention, MoE feed-forward, and
    DeepSeek YaRN rotary embeddings.
    """

    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        assert config.rope_scaling is not None

        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        self.rope = DeepseekYarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            scaling_params=scaling_params,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        scale = self.rope.compute_scale(math.sqrt(1.0 / qk_head_dim))

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(
                DeepseekV2TransformerBlock(
                    attention=LatentAttentionWithRope(
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        scale=scale,
                        q_lora_rank=config.q_lora_rank,
                        kv_lora_rank=config.kv_lora_rank,
                        qk_nope_head_dim=config.qk_nope_head_dim,
                        qk_rope_head_dim=config.qk_rope_head_dim,
                        v_head_dim=config.v_head_dim,
                        graph_mode=config.graph_mode,
                        buffer_size=config.max_batch_context_length,
                    ),
                    mlp=_get_mlp(config, i),
                    attention_norm=RMSNorm(
                        dim=config.hidden_size, eps=config.rms_norm_eps
                    ),
                    mlp_norm=RMSNorm(
                        dim=config.hidden_size, eps=config.rms_norm_eps
                    ),
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.config = config

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        freqs_cis = F.cast(self.rope.freqs_cis, h.dtype).to(h.device)

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
            )

        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )
        return (last_logits,)


class DeepseekV2(Module[..., tuple[Tensor, ...]]):
    """Top-level DeepseekV2 wrapper that unflattens variadic KV cache args."""

    def __init__(
        self,
        config: DeepseekV2Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV2TextModel(config)
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
        symbolic_inputs = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(symbolic_inputs, KVCacheInputs)
        kv_collections = symbolic_inputs.inputs
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
