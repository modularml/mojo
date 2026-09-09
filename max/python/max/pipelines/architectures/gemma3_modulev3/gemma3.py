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

"""Implements the Gemma3 model using the ModuleV3 API."""

from __future__ import annotations

import functools

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.nn.transformer import ReturnLogits

from .layers.attention import Gemma3Attention
from .layers.rms_norm import Gemma3RMSNorm
from .layers.rotary_embedding import Llama3RotaryEmbedding
from .layers.scaled_word_embedding import ScaledEmbedding
from .layers.transformer_block import Gemma3TransformerBlock
from .model_config import Gemma3Config


class Gemma3TextModel(
    Module[
        [Tensor, PagedCacheValues, PagedCacheValues, Tensor, Tensor],
        tuple[Tensor, ...],
    ]
):
    """The Gemma3 language model."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.mesh = config.mesh
        self.dtype = config.dtype

        # Use scaling_params for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=config.rope_scaling.factor,
                low_freq_factor=1e38,  # degenerates to linear scaling
                high_freq_factor=1e38,
                orig_max_position=config.max_position_embeddings,
            )
            if config.rope_scaling is not None
            else None
        )

        device = config.mesh.devices[0]

        self.rope_global = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )

        self.rope_local = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_local_base_freq,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            embed_scale=config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = ColumnParallelLinear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        create_norm = functools.partial(
            Gemma3RMSNorm,
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
            is_sliding = bool((i + 1) % config.sliding_window_pattern)
            layer_type = "sliding_attention" if is_sliding else "full_attention"
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                Gemma3TransformerBlock(
                    attention=Gemma3Attention(
                        rope_global=self.rope_global,
                        rope_local=self.rope_local,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        sliding_window_pattern=config.sliding_window_pattern,
                        has_bias=config.attention_bias,
                        qk_norm_eps=config.rms_norm_eps,
                        local_window_size=config.sliding_window,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        activation_function=config.hidden_activation,
                    ),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    pre_feedforward_layernorm=create_norm(),
                    post_feedforward_layernorm=create_norm(),
                )
            )

        self._layer_kv_key = [
            "sliding_attention"
            if bool((i + 1) % config.sliding_window_pattern)
            else "full_attention"
            for i in range(config.num_hidden_layers)
        ]

        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Compute logits from hidden states, handling weight tying."""
        if self.tie_word_embeddings:
            outputs = F.matmul(h, F.transpose(self.embed_tokens.weight, -1, -2))
            outputs = F.allgather(outputs, tensor_axis=-1)
            return F.cast(outputs, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def prepare_freq_cis(self, mesh: DeviceMesh) -> None:
        self.rope_global.freqs_cis = self.rope_global.freqs_cis.cast(
            self.dtype
        ).to(mesh)
        self.rope_local.freqs_cis = self.rope_local.freqs_cis.cast(
            self.dtype
        ).to(mesh)

    def forward(
        self,
        tokens: Tensor,
        sliding_kv_collection: PagedCacheValues,
        global_kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        tokens = tokens.to(self.mesh)
        input_row_offsets = input_row_offsets.to(self.mesh)
        h = self.embed_tokens(tokens)
        self.prepare_freq_cis(self.mesh)

        kv_by_type = {
            "sliding_attention": sliding_kv_collection,
            "full_attention": global_kv_collection,
        }
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

        # Gather last tokens and compute last-token logits.
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))

        logits: Tensor | None = None
        offsets: Tensor | None = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = F.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=CPU(),
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = F.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=CPU(),
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


class Gemma3(Module[..., tuple[Tensor, ...]]):
    """The Gemma3 model (ModuleV3 wrapper).

    Top-level wrapper that unflattens the variadic KV cache arguments
    and delegates to :class:`Gemma3TextModel`.
    """

    def __init__(
        self,
        config: Gemma3Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Gemma3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_cache_local, kv_cache_global = (
            self.kv_params.unflatten_basic_kv_tree(kv_inputs)
        )
        sliding_kv = PagedCacheValues.from_upstream(
            kv_cache_local, tokens.mapping
        )
        global_kv = PagedCacheValues.from_upstream(
            kv_cache_global, tokens.mapping
        )
        return self.language_model(
            tokens, sliding_kv, global_kv, return_n_logits, input_row_offsets
        )
