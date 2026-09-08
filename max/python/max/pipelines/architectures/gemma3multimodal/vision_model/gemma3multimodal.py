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

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.norm import LayerNorm
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.architectures.gemma3.layers.transformer_block import (
    Gemma3TransformerBlock,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from ..model_config import Gemma3ForConditionalGenerationConfig
from .embedding import Gemma3VisionEmbeddings
from .encoding import Gemma3VisionEncoder
from .projection import Gemma3MultiModalProjector

logger = logging.getLogger("max.pipelines")


class Gemma3LanguageModel(DistributedLogitsPostprocessMixin, Module):
    """The Gemma3 Multi-Modal model's text component, shared with Gemma3"""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.devices = config.devices
        # Use scaling_params for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=text_config.rope_scaling.factor,
                low_freq_factor=1e38,  # This degenerates to linear scaling
                high_freq_factor=1e38,
                orig_max_position=text_config.max_position_embeddings,
            )
            if text_config.rope_scaling is not None
            else None
        )

        rope_global = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )

        # rope_local doesn't use scaling
        rope_local = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.rope_local_base_freq,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,  # No scaling
        )

        embedding_output_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype

        self.embed_tokens = ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=text_config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(
            text_config.hidden_size,
            DType.bfloat16,
            text_config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            Gemma3RMSNorm,
            text_config.hidden_size,
            DType.bfloat16,
            eps=text_config.rms_norm_eps,
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(text_config.num_hidden_layers):
            is_sliding = bool((i + 1) % text_config.sliding_window_pattern)
            layer_type = "sliding_attention" if is_sliding else "full_attention"
            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                Gemma3TransformerBlock(
                    attention=Gemma3Attention(
                        rope_global=rope_global,
                        rope_local=rope_local,
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params_by_type[layer_type],
                        layer_idx=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        dtype=config.dtype,
                        devices=config.devices,
                        qk_norm_eps=text_config.rms_norm_eps,
                        local_window_size=text_config.sliding_window,
                        quant_config=config.quant_config,
                    ),
                    mlp=MLP(
                        dtype=config.dtype,
                        quantization_encoding=None,
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        devices=config.devices,
                        activation_function=text_config.hidden_activation,
                        quant_config=config.quant_config,
                    ),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                    pre_feedforward_layernorm=create_norm(),
                    post_feedforward_layernorm=create_norm(),
                    devices=config.devices,
                )
            )

        self._layer_kv_key = [
            "sliding_attention"
            if bool((i + 1) % text_config.sliding_window_pattern)
            else "full_attention"
            for i in range(text_config.num_hidden_layers)
        ]

        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        # Replace image placeholder tokens with vision embeddings
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=img_embed,
                image_token_indices=img_tok_indices,
            )
            for h_device, img_embed, img_tok_indices in zip(
                h, image_embeddings, image_token_indices, strict=True
            )
        ]

        kv_collections_by_type = {
            "sliding_attention": sliding_kv_collections,
            "full_attention": global_kv_collections,
        }

        # Run through transformer layers, passing image_token_indices so
        # that global attention layers can apply bidirectional masking for
        # image tokens
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections_by_type[self._layer_kv_key[idx]],
                input_row_offsets=input_row_offsets,
                image_token_indices=image_token_indices[0],
            )

        return self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )


class Gemma3VisionModel(Module):
    """The Gemma3 Multi-Modal model's vision component"""

    def __init__(
        self, config: Gemma3ForConditionalGenerationConfig, device: DeviceRef
    ) -> None:
        """Initializes the necessary components for processing vision inputs and
        projecting into language space, with multi-device functionality."""
        super().__init__()
        self.config = config
        self.devices = config.devices
        vision_config = config.vision_config
        vision_dtype = DType.bfloat16

        # Vision embeddings, sharded for multi-device setups
        self.embeddings = Gemma3VisionEmbeddings(
            config, device=config.devices[0]
        )
        self.embeddings.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.embeddings_list = self.embeddings.shard(config.devices)

        # Vision encoder (27 transformer layers)
        self.encoder = Gemma3VisionEncoder(config)

        # Post-encoder layer norm
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
            devices=[device],
            dtype=vision_dtype,
        )
        self.post_layernorm.weight.sharding_strategy = (
            ShardingStrategy.replicate(len(config.devices))
        )
        if self.post_layernorm.bias is not None:
            self.post_layernorm.bias.sharding_strategy = (
                ShardingStrategy.replicate(len(config.devices))
            )

        # Shard post_layernorm across devices
        post_layernorm_weight_shards = self.post_layernorm.weight.shard(
            config.devices
        )
        post_layernorm_bias_shards: list[Weight | None] = (
            list(self.post_layernorm.bias.shard(config.devices))
            if self.post_layernorm.bias is not None
            else [None] * len(config.devices)
        )

        self.post_layernorm_list = []
        for device, weight_shard, bias_shard in zip(  # noqa: PLR1704 (FIXME)
            config.devices,
            post_layernorm_weight_shards,
            post_layernorm_bias_shards,
            strict=True,
        ):
            ln = LayerNorm(
                vision_config.hidden_size,
                eps=vision_config.layer_norm_eps,
                devices=[device],
                dtype=vision_dtype,
            )
            ln.weight = weight_shard
            if bias_shard is not None:
                ln.bias = bias_shard
            self.post_layernorm_list.append(ln)

        # Multimodal projector to project vision embeddings to language space
        self.projector = Gemma3MultiModalProjector(
            config, device=config.devices[0]
        )
        self.projector.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.projector_list = self.projector.shard(config.devices)

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Processes vision inputs through the Gemma3 vision tower and produces
        a sequence of image embeddings"""
        hidden_states: TensorValue | Sequence[TensorValue] = [
            embed(pixels)
            for embed, pixels in zip(
                self.embeddings_list, pixel_values, strict=True
            )
        ]

        # Pass through encoder layers
        hidden_states = self.encoder(hidden_states, signal_buffers)

        # Apply post-encoder layer norm
        if isinstance(hidden_states, Sequence):
            hidden_states = [
                layer(states)
                for layer, states in zip(
                    self.post_layernorm_list, hidden_states, strict=True
                )
            ]
        else:
            hidden_states = self.post_layernorm_list[0](hidden_states)

        # Project to language model hidden size
        if isinstance(hidden_states, Sequence):
            image_embeddings_list = [
                projector(states)
                for projector, states in zip(
                    self.projector_list, hidden_states, strict=True
                )
            ]
        else:
            image_embeddings_list = [self.projector_list[0](hidden_states)]

        return image_embeddings_list
