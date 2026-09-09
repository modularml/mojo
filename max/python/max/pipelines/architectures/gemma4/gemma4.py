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

"""Implements the Gemma4 model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, FusedMLP
from max.nn.moe import MoE, MoEQuantized, make_concatenated_gated_activation_fn
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .layers.attention import Gemma4Attention
from .layers.decoder_layer import Gemma4TextDecoderLayer
from .layers.moe import Gemma4MoEGate
from .layers.rms_norm import Gemma4RMSNorm
from .layers.rotary_embedding import ProportionalRotaryEmbedding
from .model_config import Gemma4ForConditionalGenerationConfig
from .weight_adapters import gemma4_uses_fused_projections


class Gemma4TextModel(DistributedLogitsPostprocessMixin, Module):
    """The Gemma 4 language model."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.devices = config.devices

        # Build per-layer-type rotary embeddings.
        # sliding_attention uses default rope, full_attention uses proportional.
        rope_sliding = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )

        unquantized_dtype = config.unquantized_dtype

        embedding_output_dtype = config.dtype
        quant_config = text_config.quant_config
        if quant_config and quant_config.embedding_output_dtype:
            embedding_output_dtype = quant_config.embedding_output_dtype

        # DISTINF-194: load projections pre-fused (single-device, bf16 only).
        # The weight adapter concatenates the checkpoint projections; the graph
        # consumes one constant, avoiding the in-graph concat that init would
        # otherwise materialize as a second on-device weight copy. The shared
        # predicate keeps layer selection and adapter key fusion in lockstep.
        use_fused_projections = gemma4_uses_fused_projections(config)

        self.embed_tokens = ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=text_config.hidden_size**0.5,
        )

        self.norm = Gemma4RMSNorm(
            text_config.hidden_size,
            unquantized_dtype,
            text_config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            dtype=unquantized_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        # Resolve per-layer KVCacheParams from MultiKVCacheParams. The tree is
        # keyed by layer-type name ("sliding_attention" / "full_attention").
        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_layer_type: dict[str, KVCacheParams] = {}
        for _k, _p in config.kv_params.children.items():
            assert isinstance(_p, KVCacheParams)
            kv_params_by_layer_type[_k] = _p

        layer_type_counts: dict[str, int] = {
            "sliding_attention": 0,
            "full_attention": 0,
        }
        layers = []
        for i in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[i]
            kv_params = kv_params_by_layer_type[layer_type]

            layer_idx_in_cache = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            is_sliding = layer_type == "sliding_attention"

            is_nvfp4 = quant_config is not None and quant_config.is_nvfp4
            moe_nvfp4 = is_nvfp4 and text_config.enable_moe_block
            # The first-party NVFP4 checkpoints (nvidia/Gemma-4-*) keep
            # attention in BF16, but other modelopt quants (e.g. the
            # community 12B NVFP4 ones) quantize it too -- honor the
            # per-layer classification instead of assuming BF16.
            # is_nvfp4 already implies quant_config is not None.
            attn_quantized = (
                is_nvfp4
                and quant_config is not None
                and i in quant_config.attn_quantized_layers
            )

            moe_block: MoE | None = None
            if text_config.enable_moe_block:
                moe_gate_cls = functools.partial(
                    Gemma4MoEGate,
                    eps=text_config.rms_norm_eps,
                )
                moe_norm_cls = functools.partial(
                    Gemma4RMSNorm,
                    text_config.hidden_size,
                    unquantized_dtype,
                    eps=text_config.rms_norm_eps,
                )
                if is_nvfp4:
                    moe_block = MoEQuantized(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        gated_activation_fn=make_concatenated_gated_activation_fn(
                            functools.partial(ops.gelu, approximate="tanh")
                        ),
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                        quant_config=quant_config,
                    )
                else:
                    moe_block = MoE(
                        devices=config.devices,
                        hidden_dim=text_config.hidden_size,
                        num_experts=text_config.num_experts,
                        num_experts_per_token=text_config.top_k_experts,
                        moe_dim=text_config.moe_intermediate_size,
                        gate_cls=moe_gate_cls,
                        gated_activation_fn=make_concatenated_gated_activation_fn(
                            functools.partial(ops.gelu, approximate="tanh")
                        ),
                        pre_expert_norm_cls=moe_norm_cls,
                        dtype=config.dtype,
                    )

            layers.append(
                Gemma4TextDecoderLayer(
                    attention=Gemma4Attention(
                        rope_global=rope_global,
                        rope_local=rope_sliding,
                        num_attention_heads=text_config.num_attention_heads,
                        num_key_value_heads=text_config.num_key_value_heads,
                        num_global_key_value_heads=text_config.num_global_key_value_heads,
                        attention_k_eq_v=text_config.attention_k_eq_v,
                        hidden_size=text_config.hidden_size,
                        kv_params=kv_params,
                        global_head_dim=text_config.global_head_dim,
                        layer_idx=i,
                        layer_idx_in_cache=layer_idx_in_cache,
                        is_sliding=is_sliding,
                        dtype=unquantized_dtype
                        if (is_nvfp4 and not attn_quantized)
                        else config.dtype,
                        devices=config.devices,
                        qk_norm_eps=text_config.rms_norm_eps,
                        local_window_size=text_config.sliding_window,
                        quant_config=None
                        if (is_nvfp4 and not attn_quantized)
                        else quant_config,
                        fused_qkv=use_fused_projections,
                    ),
                    mlp=FusedMLP(
                        dtype=config.dtype,
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        devices=config.devices,
                        activation_function=text_config.hidden_activation,
                    )
                    if use_fused_projections
                    else MLP(
                        dtype=unquantized_dtype if moe_nvfp4 else config.dtype,
                        quantization_encoding=None,
                        hidden_dim=text_config.hidden_size,
                        feed_forward_length=text_config.intermediate_size,
                        devices=config.devices,
                        activation_function=text_config.hidden_activation,
                        quant_config=None if moe_nvfp4 else quant_config,
                    ),
                    hidden_size=text_config.hidden_size,
                    rms_norm_eps=text_config.rms_norm_eps,
                    devices=config.devices,
                    unquantized_dtype=unquantized_dtype,
                    enable_moe_block=text_config.enable_moe_block,
                    moe_block=moe_block,
                )
            )

        # Store the per-layer cache-tree key ("sliding_attention" /
        # "full_attention") so __call__ can route the correct cache to each
        # layer.
        self._layer_kv_key = [
            text_config.layer_types[i]
            for i in range(text_config.num_hidden_layers)
        ]

        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = text_config.return_logits
        self.return_hidden_states = text_config.return_hidden_states
        self.target_layer_ids: list[int] | None = (
            list(text_config.target_layer_ids)
            if text_config.target_layer_ids
            else None
        )
        if self.target_layer_ids is not None:
            n_layers = text_config.num_hidden_layers
            for pos, layer_id in enumerate(self.target_layer_ids):
                if not 0 <= layer_id < n_layers:
                    raise ValueError(
                        f"target_layer_ids[{pos}]={layer_id} is out of range "
                        f"[0, {n_layers}) for a model with {n_layers} layers."
                    )
        # Final logit softcapping: matches the reference and bounds logits to
        # (-cap, cap), keeping them finite under float16.
        self.logit_softcapping = text_config.final_logit_softcapping

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        **kwargs: object,
    ) -> tuple[TensorValue, ...]:
        kv_collections_by_type = {
            "sliding_attention": sliding_kv_collections,
            "full_attention": global_kv_collections,
        }

        h = self.embed_tokens(tokens, signal_buffers)

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

        capture_layer_set: set[int] | None = None
        capture_hidden_states: list[list[TensorValue]] | None = None
        if (
            self.target_layer_ids is not None
            and self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        ):
            capture_layer_set = set(self.target_layer_ids)
            capture_hidden_states = []

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            kv_collections = kv_collections_by_type[self._layer_kv_key[idx]]
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )
            if (
                capture_layer_set is not None
                and capture_hidden_states is not None
                and idx in capture_layer_set
            ):
                capture_hidden_states.append(list(h))

        return self._postprocess_logits(
            h,
            input_row_offsets,
            return_n_logits,
            signal_buffers,
            capture_hidden_states=capture_hidden_states,
        )
