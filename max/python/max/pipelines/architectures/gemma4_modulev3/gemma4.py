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

"""Implements the Gemma4 text model using the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma3_modulev3.layers.scaled_word_embedding import (
    ScaledEmbedding,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings

from .layers.attention import Gemma4Attention
from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import ProportionalRotaryEmbedding
from .layers.transformer_block import Gemma4TransformerBlock


class Gemma4TextModel(Module[..., tuple[Tensor, ...]]):
    """The Gemma4 language model (text-only)."""

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        text_config = config.text_config
        self.mesh = mesh
        self.dtype = config.dtype
        device = mesh.devices[0]

        self.rope_local = RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            device=device,
            head_dim=text_config.head_dim,
            interleaved=False,
        )
        self.rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            device=device,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )

        self.embed_tokens = ScaledEmbedding(
            text_config.vocab_size,
            dim=text_config.hidden_size,
            embed_scale=text_config.hidden_size**0.5,
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = ColumnParallelLinear(
                in_dim=text_config.hidden_size,
                out_dim=text_config.vocab_size,
                bias=False,
            )
        self.logit_softcapping = text_config.final_logit_softcapping

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[i]
            is_sliding = layer_type == "sliding_attention"
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1

            layers.append(
                Gemma4TransformerBlock(
                    attention=Gemma4Attention(
                        rope_global=self.rope_global,
                        rope_local=self.rope_local,
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        num_global_key_value_heads=text_config.num_global_key_value_heads,
                        attention_k_eq_v=text_config.attention_k_eq_v,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx_in_cache=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        qk_norm_eps=text_config.rms_norm_eps,
                        local_window_size=text_config.sliding_window,
                    ),
                    mlp=MLP(
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        activation_function=text_config.hidden_activation,
                    ),
                    hidden_size=text_config.hidden_size,
                    rms_norm_eps=text_config.rms_norm_eps,
                )
            )
        self.layers = ModuleList(layers)
        self._layer_kv_key = [
            text_config.layer_types[i]
            for i in range(text_config.num_hidden_layers)
        ]
        self.return_logits = text_config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            outputs = h @ self.embed_tokens.weight.T
            outputs = F.allgather(outputs, tensor_axis=-1)
        else:
            assert self.lm_head is not None
            outputs = self.lm_head(h)
        outputs = outputs.cast(DType.float32)
        if self.logit_softcapping:
            cap = self.logit_softcapping
            outputs = F.tanh(outputs / cap) * cap
        return outputs

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
        sliding_kv: PagedCacheValues,
        global_kv: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
    ) -> tuple[Tensor, ...]:
        tokens = tokens.to(self.mesh)
        input_row_offsets = input_row_offsets.to(self.mesh)
        image_embeddings = image_embeddings.to(self.mesh)
        image_token_indices = image_token_indices.to(self.mesh)
        h = self.embed_tokens(tokens)
        self.prepare_freq_cis(self.mesh)

        # Scatter the vision tower's soft tokens over their placeholder
        # positions, after the sqrt(hidden) embedding scale: image
        # embeddings arrive pre-scaled.
        # Out-of-bounds indices are skipped, so an empty (text-only or decode)
        # batch is a no-op.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings.cast(h.dtype), image_token_indices
        )

        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }
        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

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
                input_row_offsets[1:].unsqueeze(-1) - return_n_logits_range
            )
            last_indices = offsets.reshape((-1,))
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


class Gemma4(Module[..., tuple[Tensor, ...]]):
    """Top-level wrapper: unflattens the two-cache KV tree, delegates."""

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        kv_params: KVCacheParamInterface,
        mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.language_model = Gemma4TextModel(config, mesh)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        assert isinstance(self.kv_params, MultiKVCacheParams)
        kv_inputs = iter(x._graph_value for x in variadic_args)
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
            tokens,
            sliding_kv,
            global_kv,
            return_n_logits,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
        )
