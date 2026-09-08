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

"""Gemma3 multimodal model components using the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm.layer_norm import LayerNorm
from max.experimental.tensor import Tensor
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma3_modulev3.gemma3 import Gemma3TextModel
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings

from ..model_config import Gemma3ForConditionalGenerationConfig
from .embedding import Gemma3VisionEmbeddings
from .encoding import Gemma3VisionEncoder
from .projection import Gemma3MultiModalProjector


class Gemma3LanguageModel(Module[..., tuple[Tensor, ...]]):
    """The Gemma3 multimodal language model.

    Wraps :class:`Gemma3TextModel` to add multimodal embedding merge and
    KV cache unflattening from variadic args.
    """

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Gemma3TextModel(config.text_config)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        tokens = tokens.to(self.language_model.mesh)
        return_n_logits = return_n_logits.to(self.language_model.mesh)
        input_row_offsets = input_row_offsets.to(self.language_model.mesh)
        image_embeddings = image_embeddings.to(self.language_model.mesh)
        image_token_indices = image_token_indices.to(self.language_model.mesh)

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

        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(tokens)
        self.language_model.prepare_freq_cis(tokens.mesh)

        # Merge image embeddings at pre-computed token positions.
        merged = F_merge_multimodal_embeddings(
            inputs_embeds, image_embeddings, image_token_indices
        )

        kv_by_type = {
            "sliding_attention": sliding_kv,
            "full_attention": global_kv,
        }

        # Run through transformer layers
        h = merged
        for idx, layer in enumerate(self.language_model.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_by_type[self.language_model._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
            )

        # Gather last tokens and compute logits
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self.language_model._compute_logits(
            self.language_model.norm(last_h)
        )

        logits: Tensor | None = None
        offsets: Tensor | None = None

        if self.language_model.return_logits == ReturnLogits.VARIABLE:
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
            last_indices = offsets.reshape(shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self.language_model._compute_logits(
                self.language_model.norm(last_tokens)
            )
            offsets = F.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=CPU(),
                dtype=DType.int64,
            )
        elif self.language_model.return_logits == ReturnLogits.ALL:
            logits = self.language_model._compute_logits(
                self.language_model.norm(h)
            )
            offsets = input_row_offsets

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        return ret_val


class Gemma3VisionModel(Module[[Tensor], tuple[Tensor, ...]]):
    """The Gemma3 multimodal vision model (single-device, ModuleV3).

    Processes pixel values through vision embeddings, encoder, layer norm,
    and multimodal projector to produce image embeddings.
    """

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
    ) -> None:
        super().__init__()
        vision_config = config.vision_config

        self.embeddings = Gemma3VisionEmbeddings(config)
        self.encoder = Gemma3VisionEncoder(config)
        self.post_layernorm = LayerNorm(
            dim=vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
        )
        self.projector = Gemma3MultiModalProjector(config)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        image_embeddings = self.projector(hidden_states)
        return (image_embeddings,)
