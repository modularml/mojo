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
"""Implements the DeepseekV3.2 model."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from typing import Any, cast

from max._core.driver import is_virtual_device_mode
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.attention.multi_latent_attention import MLAPrefillMetadata
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import ColumnParallelLinear
from max.nn.moe import MoE
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    forward_sharded_layers,
)

from ..deepseekV3.deepseekV3 import deepseek_logits_postprocess
from .layers import DeepseekV3_2MLP, DeepseekV3_2MoE, DeepseekV3_2TopKRouter
from .layers.sparse_mla import (
    DataParallelSparseLatentAttentionWithRope,
    DataParallelSparseLatentAttentionWithRopeFp8,
    TensorParallelSparseLatentAttentionWithRope,
    TensorParallelSparseLatentAttentionWithRopeFp8,
)
from .model_config import DeepseekV3_2Config

# Opt-in dual-carry + fused AG+RMSNorm. Default (unset / any other value) is
# the baseline path: consumer-side ``input_layernorm``, plain post-MLP
# all-gather. Unfused dual-carry still regresses large-CE TTFT/util.
_FUSE_AG_RMS_NORM = (
    os.environ.get("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM") == "0"
)


def _validate_parallelism_config(config: DeepseekV3_2Config) -> None:
    """Validate parallelism configuration for DeepseekV3.2.

    Supported multi-GPU modes:
      - DP attention + EP MoE: ``data_parallel_degree == num_devices``
      - TP attention + EP MoE: ``data_parallel_degree == 1``
    ``DeepseekV3_2Config.__post_init__`` already enforces
    ``data_parallel_degree in (1, num_devices)``.
    """
    num_devices = len(config.devices)
    # Skip EP validation in virtual device mode (compilation-only) since EP
    # will be disabled later due to NVSHMEM linking requirements
    if (
        num_devices > 1
        and config.ep_config is None
        and not is_virtual_device_mode()
    ):
        raise ValueError(
            "Expert-parallel (ep_config) must be enabled for multi-GPU DeepseekV3.2."
        )


def _validate_indexer_types(config: DeepseekV3_2Config) -> None:
    """Validate the cross-layer index-sharing schedule.

    A ``"shared"`` layer reuses the top-k selection from the most recent
    ``"full"`` layer, so the first layer must be ``"full"`` — otherwise there is
    no prior selection to reuse. An empty schedule means every layer is full.
    """
    if config.indexer_types and config.indexer_types[0] != "full":
        raise ValueError(
            "indexer_types[0] must be 'full': the first layer cannot be "
            "'shared' because no preceding full indexer layer exists to "
            f"reuse a top-k selection from (got {config.indexer_types[0]!r})."
        )


def apply_initial_input_layernorm(
    layers: Sequence[DeepseekV3_2DecoderLayer],
    hidden_states: list[TensorValue],
) -> list[TensorValue]:
    """Apply layer 0 ``input_layernorm`` after token embedding.

    Dual-carry Pre-LN: the first decoder block expects a separate normalized
    stream for attention while residuals keep the raw embed output.
    """
    if not layers:
        return hidden_states
    return forward_sharded_layers(
        layers[0].input_layernorm_shards, hidden_states
    )


def apply_input_layernorm_at_layer_entry(
    layer: DeepseekV3_2DecoderLayer,
    hidden_states: list[TensorValue],
) -> list[TensorValue]:
    """Apply ``input_layernorm`` outside a decoder subgraph call.

    Used on the non-carry path (and as the shared primitive behind
    :func:`apply_initial_input_layernorm`). Attention consumes the normalized
    tensor; residuals use the raw hidden-state stream.
    """
    return forward_sharded_layers(layer.input_layernorm_shards, hidden_states)


def next_input_layernorm_gammas(
    next_layer: DeepseekV3_2DecoderLayer,
    hidden_states: list[TensorValue],
) -> list[TensorValue]:
    """Cast/place ``next_layer.input_layernorm`` weights as graph inputs.

    Threaded into the producer block so subgraph weight-prefix rebinding cannot
    mis-bind the *next* layer's gamma to the current layer's prefix.
    """
    return [
        shard.weight.cast(h.dtype).to(h.device)
        for shard, h in zip(
            next_layer.input_layernorm_shards, hidden_states, strict=True
        )
    ]


class DeepseekV3_2DecoderLayer(Module):
    """Decoder layer for DeepseekV3.2."""

    def __init__(
        self,
        rope: DeepseekYarnRotaryEmbedding | RotaryEmbedding,
        config: DeepseekV3_2Config,
        layer_idx: int,
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ep_manager = ep_manager
        num_devices = len(config.devices)

        self.mlp: DeepseekV3_2MLP | DeepseekV3_2MoE | MoE
        self.mlp_shards: list[DeepseekV3_2MLP | DeepseekV3_2MoE | Module]

        if config.quant_config is None:
            raise ValueError(
                "DeepSeekV3.2 sparse attention requires a quantization config."
            )

        attn_quantized = layer_idx in config.quant_config.attn_quantized_layers

        assert isinstance(config.kv_params, MultiKVCacheParams)
        mla_kv_params = config.kv_params.children["mla"]
        _indexer_kv_params = config.kv_params.children["indexer"]

        skip_topk = (
            bool(config.indexer_types)
            and config.indexer_types[layer_idx] == "shared"
        )

        sparse_attn_kwargs: dict[str, Any] = {
            "rope": rope,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "hidden_size": config.hidden_size,
            "kv_params": mla_kv_params,
            "q_lora_rank": config.q_lora_rank,
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
            "devices": config.devices,
            "graph_mode": config.graph_mode,
            "buffer_size": config.max_batch_context_length,
            "index_n_heads": config.index_n_heads,
            "index_head_dim": config.index_head_dim,
            "index_topk": config.index_topk,
            "skip_topk": skip_topk,
            "indexer_rope_interleave": config.indexer_rope_interleave,
        }

        self.tp_attention = num_devices > 1 and config.data_parallel_degree == 1
        self.self_attn: (
            DataParallelSparseLatentAttentionWithRope
            | DataParallelSparseLatentAttentionWithRopeFp8
            | TensorParallelSparseLatentAttentionWithRope
            | TensorParallelSparseLatentAttentionWithRopeFp8
        )
        if self.tp_attention:
            if attn_quantized:
                self.self_attn = TensorParallelSparseLatentAttentionWithRopeFp8(
                    skip_allreduce=True,
                    norm_dtype=config.norm_dtype,
                    quant_config=config.quant_config,
                    **sparse_attn_kwargs,
                )
            else:
                self.self_attn = TensorParallelSparseLatentAttentionWithRope(
                    skip_allreduce=True,
                    norm_dtype=config.norm_dtype,
                    dtype=DType.bfloat16,
                    indexer_quant_config=config.quant_config,
                    **sparse_attn_kwargs,
                )
        elif attn_quantized:
            self.self_attn = DataParallelSparseLatentAttentionWithRopeFp8(
                norm_dtype=config.norm_dtype,
                quant_config=config.quant_config,
                **sparse_attn_kwargs,
            )
        else:
            self.self_attn = DataParallelSparseLatentAttentionWithRope(
                norm_dtype=config.norm_dtype,
                dtype=DType.bfloat16,
                indexer_quant_config=config.quant_config,
                **sparse_attn_kwargs,
            )

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx)
        if self.mlp.sharding_strategy is not None:
            self.mlp_shards = list(self.mlp.shard(config.devices))
        else:
            self.mlp_shards = [self.mlp]

        # Fused AG+RMSNorm requires mbc=True. Baseline (no fuse flag) keeps
        # Llama-style mbc=False. post-attn stays mbc=False either way.
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            dtype=config.norm_dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=_FUSE_AG_RMS_NORM,
        )
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            dtype=config.norm_dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

        # Dual-carry + fused AG+RMSNorm only when opted in via
        # ``MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=0``. Otherwise baseline
        # consumer-side input norm (see ``apply_input_layernorm_at_layer_entry``).
        self.carry_next_input_norm = (
            _FUSE_AG_RMS_NORM
            and self.tp_attention
            and config.ep_config is not None
            and not config.ep_config.use_allreduce
        )

    def _get_mlp(
        self, config: DeepseekV3_2Config, layer_idx: int
    ) -> DeepseekV3_2MLP | DeepseekV3_2MoE | MoE:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            layer_idx: Layer index

        Returns:
            MLP or MoE module depending on the layer index and config
        """
        quant_cfg = config.quant_config
        mlp_quantized = (
            quant_cfg is not None
            and layer_idx in quant_cfg.mlp_quantized_layers
        )
        mlp_dtype = config.dtype if mlp_quantized else DType.bfloat16
        layer_quant_config = quant_cfg if mlp_quantized else None

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            if config.ep_config is not None:
                ep_size = (
                    config.ep_config.n_gpus_per_node * config.ep_config.n_nodes
                )
            else:
                ep_size = 1

            moe_kwargs: dict[str, Any] = {
                "devices": config.devices,
                "hidden_dim": config.hidden_size,
                "num_experts": config.n_routed_experts,
                "num_experts_per_token": config.num_experts_per_tok,
                "moe_dim": config.moe_intermediate_size,
                "gate_cls": functools.partial(
                    DeepseekV3_2TopKRouter,
                    routed_scaling_factor=config.routed_scaling_factor,
                    scoring_func=config.scoring_func,
                    topk_method=config.topk_method,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    norm_topk_prob=config.norm_topk_prob,
                    # Use the same dtype for the gate as the norm
                    gate_dtype=DType.bfloat16,
                    correction_bias_dtype=config.correction_bias_dtype,
                ),
                "mlp_cls": DeepseekV3_2MLP,
                "has_shared_experts": True,
                "shared_experts_dim": config.n_shared_experts
                * config.moe_intermediate_size,
                "dtype": mlp_dtype,
                "ep_size": ep_size,
                "apply_router_weight_first": False,
                "ep_batch_manager": self.ep_manager,
                "quant_config": layer_quant_config,
                "shared_experts_dtype": (
                    quant_cfg.shared_experts_dtype(mlp_dtype)
                    if quant_cfg is not None
                    else DType.bfloat16
                ),
            }

            moe: DeepseekV3_2MoE | MoE
            if mlp_quantized:
                moe = DeepseekV3_2MoE(**moe_kwargs)
            else:
                moe = MoE(**moe_kwargs)

            num_devices = len(config.devices)
            if num_devices > 1:
                moe.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            return moe
        else:
            mlp = DeepseekV3_2MLP(
                dtype=mlp_dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
                quant_config=layer_quant_config,
            )
            if config.ep_config is not None and config.ep_config.use_allreduce:
                mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                    len(config.devices)
                )
            else:
                mlp.sharding_strategy = ShardingStrategy.replicate(
                    len(config.devices)
                )
            return mlp

    def __call__(
        self,
        layer_idx: TensorValue,
        xs_raw: list[TensorValue],
        xs_norm: list[TensorValue],
        signal_buffers: list[BufferValue],
        mla_kv_collections: list[PagedCacheValues],
        indexer_kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        mla_prefill_metadata_flat: list[TensorValue],
        input_row_offsets: list[TensorValue],
        prev_topk_indices: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        # Next block's input-norm gamma (graph input). Empty on the last block
        # and when ``carry_next_input_norm`` is off.
        next_input_gamma: list[TensorValue] | None = None,
        # Note: this is only used for MTP iterations after step 0.
        reuse_prev_topk: bool = False,
    ) -> list[TensorValue]:
        # Dual-carry Pre-LN: attention uses ``xs_norm``; residuals use ``xs_raw``.
        apply_next = bool(next_input_gamma)

        # Re-pack flat MLA inputs into MLAPrefillMetadata dataclasses
        num_devices = len(mla_kv_collections)
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        for i in range(num_devices):
            mla_prefill_metadata.append(
                MLAPrefillMetadata(
                    buffer_row_offsets=mla_prefill_metadata_flat[3 * i],
                    cache_offsets=mla_prefill_metadata_flat[3 * i + 1],
                    buffer_lengths=mla_prefill_metadata_flat[3 * i + 2],
                )
            )

        # ``prev_topk_indices`` may arrive empty (first layer, before any full
        # layer has produced a selection); ``full`` layers ignore it.
        prev_topk = prev_topk_indices if prev_topk_indices is not None else None
        attn_outs, topk_indices = self.self_attn(
            layer_idx,
            xs_norm,
            signal_buffers,
            mla_kv_collections,
            indexer_kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
            mla_prefill_metadata=mla_prefill_metadata,
            prev_topk_indices=prev_topk,
            reuse_prev_topk=reuse_prev_topk,
        )

        ep_config = self.config.ep_config
        if (
            self.tp_attention
            and ep_config is not None
            and not ep_config.use_allreduce
        ):
            # Fused reduce-scatter + post-attention norm: eliminates the
            # separate reduce-scatter collective and a global-memory norm
            # round-trip by keeping the partial sums in float32 registers.
            hs_partial = [xs_raw[0] + attn_outs[0], *attn_outs[1:]]
            gammas = [
                shard.weight.cast(DType.bfloat16).to(x.device)
                for shard, x in zip(
                    self.post_attention_layernorm_shards, xs_raw, strict=True
                )
            ]
            norm_outs, hs = ops.reduce_scatter_rms_norm(
                hs_partial,
                signal_buffers,
                gammas,
                self.config.rms_norm_eps,
            )
        else:
            hs = self._post_attention(xs_raw, attn_outs, signal_buffers)
            # Post-attention norm (per-device)
            norm_outs = forward_sharded_layers(
                self.post_attention_layernorm_shards, hs
            )

        if self.config.ep_config is not None:
            assert ep_inputs is not None
            if self.ep_manager is not None:
                self.ep_manager.fetch_buffers(ep_inputs)

        mlp_outs = forward_moe_sharded_layers(self.mlp_shards, norm_outs)

        if apply_next:
            assert next_input_gamma is not None
            # Producer-side next input norm after all-gather (fused op).
            hs_raw, hs_norm = self._post_mlp_with_next_input_norm(
                hs, mlp_outs, signal_buffers, next_input_gamma
            )
            hs_raw = [
                ops.rebind(h, x.shape)
                for h, x in zip(hs_raw, xs_raw, strict=True)
            ]
            hs_norm = [
                ops.rebind(n, x.shape)
                for n, x in zip(hs_norm, xs_raw, strict=True)
            ]
            return hs_raw + hs_norm + topk_indices

        hs = self._post_mlp(hs, mlp_outs, signal_buffers)

        # In TP mode the reduce-scatter/all-gather round trip can lose the
        # static shape; rebind to the original per-device input shape.
        if self.tp_attention:
            hs = [
                ops.rebind(h, x.shape) for h, x in zip(hs, xs_raw, strict=True)
            ]

        # Subgraphs require the outputs to be a single list of TensorValue,
        # which is why the returned lists are concatenated.
        return hs + topk_indices

    def _post_attention(
        self,
        xs: list[TensorValue],
        attn_outs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Residual connection and collective after attention."""
        if not self.tp_attention:
            return [x + a for x, a in zip(xs, attn_outs, strict=True)]

        assert self.config.ep_config is not None
        if self.config.ep_config.use_allreduce:
            attn_outs = ops.allreduce.sum(attn_outs, signal_buffers)
            return [x + a for x, a in zip(xs, attn_outs, strict=True)]

        # The residual is replicated across devices, so add it only on device 0
        # to avoid counting it once per device after the reduce-scatter sum.
        hs = [xs[0] + attn_outs[0], *attn_outs[1:]]
        return ops.reducescatter.sum(hs, signal_buffers, axis=0)

    def _post_mlp(
        self,
        hs: list[TensorValue],
        mlp_outs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Residual connection and collective after the MoE/MLP."""
        if not self.tp_attention:
            return [h + m for h, m in zip(hs, mlp_outs, strict=True)]

        assert self.config.ep_config is not None
        if self.config.ep_config.use_allreduce:
            mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
            return [h + m for h, m in zip(hs, mlp_outs, strict=True)]

        hs = [h + m for h, m in zip(hs, mlp_outs, strict=True)]
        return ops.allgather(hs, signal_buffers, axis=0)

    def _post_mlp_with_next_input_norm(
        self,
        hs: list[TensorValue],
        mlp_outs: list[TensorValue],
        signal_buffers: list[BufferValue],
        next_input_gamma: list[TensorValue],
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Residual add + fused all-gather + next layer's ``input_layernorm``.

        Only used when ``carry_next_input_norm`` is set (requires
        ``MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=0``). Returns
        ``(residual_full, norm_full)`` for dual-carry into the next block.
        """
        hs_in = [h + m for h, m in zip(hs, mlp_outs, strict=True)]
        norm_full, residual_full = ops.allgather_rms_norm(
            hs_in,
            signal_buffers,
            next_input_gamma,
            epsilon=self.input_layernorm.eps,
            weight_offset=self.input_layernorm.weight_offset,
        )
        return residual_full, norm_full


class DeepseekV3_2(Module):
    """Defines the DeepseekV3.2 transformer model.

    This is a combination of the DeepseekV3.2Model and the DeepseekV3.2ForCausalLM
    classes from the HuggingFace Transformers implementation.

    DeepseekV3.2 extends DeepseekV3 with sparse attention using an indexer mechanism.
    """

    def __init__(self, config: DeepseekV3_2Config) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        _validate_parallelism_config(config)
        _validate_indexer_types(config)

        embedding_output_dtype = config.dtype
        if embedding_output_dtype == DType.uint8:
            embedding_output_dtype = DType.bfloat16
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        if config.rope_scaling is not None:
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
            self.rope: DeepseekYarnRotaryEmbedding | RotaryEmbedding = (
                DeepseekYarnRotaryEmbedding(
                    config.qk_rope_head_dim,
                    n_heads=config.num_attention_heads,
                    theta=config.rope_theta,
                    max_seq_len=config.max_seq_len,
                    scaling_params=scaling_params,
                )
            )
        else:
            self.rope = RotaryEmbedding(
                dim=config.qk_rope_head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                head_dim=config.qk_rope_head_dim,
                interleaved=config.rope_interleave,
            )

        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        self.layers = LayerList(
            [
                DeepseekV3_2DecoderLayer(
                    self.rope,
                    config,
                    i,
                    None
                    if i < config.first_k_dense_replace
                    else self.ep_manager,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.norm_shards = self.norm.shard(devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        # Carry next-layer input norm after post-MLP all-gather on the TP+EP
        # path (see DecoderLayer.carry_next_input_norm). Uniform across layers.
        first_layer = (
            cast(DeepseekV3_2DecoderLayer, self.layers[0])
            if self.layers
            else None
        )
        self.carry_next_input_norm = bool(
            first_layer and first_layer.carry_next_input_norm
        )

        if config.use_subgraphs:
            # ``full`` and ``shared`` layers differ structurally (shared layers
            # have no indexer weights), so they cannot share a subgraph. Split
            # the MoE layers into one subgraph group per indexer type. An empty
            # schedule (DeepSeek V3.2, GLM-5.1) collapses to a single full group.
            moe_layers = range(
                config.first_k_dense_replace, config.num_hidden_layers
            )

            def _is_shared(i: int) -> bool:
                return (
                    bool(config.indexer_types)
                    and config.indexer_types[i] == "shared"
                )

            full_group = [i for i in moe_layers if not _is_shared(i)]
            shared_group = [i for i in moe_layers if _is_shared(i)]
            self.subgraph_layer_groups = [
                g for g in (full_group, shared_group) if g
            ]
            # Last block returns residual(+topk) only; interior carry blocks
            # also emit the next input-norm. Peel so subgraph arity stays
            # uniform.
            if self.carry_next_input_norm:
                last = config.num_hidden_layers - 1
                self.subgraph_layer_groups = [
                    [i for i in g if i != last]
                    for g in self.subgraph_layer_groups
                ]
                self.subgraph_layer_groups = [
                    g for g in self.subgraph_layer_groups if g
                ]
        else:
            self.subgraph_layer_groups = []
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.emit_last_token_logits = True
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        mla_kv_collections: list[PagedCacheValues],
        indexer_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
    ) -> tuple[TensorValue, ...]:
        if not host_input_row_offsets.device == DeviceRef.CPU():
            raise ValueError("input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices
        h = self.embed_tokens(tokens, signal_buffers)

        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        input_row_offsets_ = ops.distributed_broadcast(
            input_row_offsets.to(devices[0]), signal_buffers
        )

        if self.config.data_parallel_degree > 1:
            # Split batch across devices for data-parallel attention.
            h, input_row_offsets_ = split_batch_replicated(
                devices,
                h,
                input_row_offsets_,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        layers_list = [
            cast(DeepseekV3_2DecoderLayer, layer) for layer in self.layers
        ]
        num_devices = len(devices)
        last_layer = self.config.num_hidden_layers - 1

        # Layer 0 input norm after embeddings (dual-carry: raw residual kept).
        initial_input_norm: list[TensorValue] | None = None
        if self.carry_next_input_norm:
            initial_input_norm = apply_initial_input_layernorm(layers_list, h)

        # Create MLA prefill metadata if not in decode mode
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = self.layers[
                0
            ].self_attn.create_mla_prefill_metadata(  # type: ignore
                input_row_offsets_, mla_kv_collections
            )

            # replace each device's buffer_lengths with the batch context length
            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]

        # Flatten MLAPrefillMetadata to list of TensorValues for subgraph calls
        mla_prefill_metadata_flat: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_prefill_metadata_flat.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        def inputs_for_layer(
            idx: int, h: list[TensorValue]
        ) -> list[SubgraphInput]:
            # Layout of ``h``:
            # - carry_next_input_norm, idx==0: residual only (embeddings);
            #   norm comes from ``apply_initial_input_layernorm`` above
            # - carry_next_input_norm, idx>0: residual + norm + topk
            # - else: residual [+ topk]
            next_input_gamma: list[TensorValue] | None = None
            if self.carry_next_input_norm:
                if idx == 0:
                    hidden_raw = h
                    prev_topk: list[TensorValue] = []
                    assert initial_input_norm is not None
                    hidden_norm = initial_input_norm
                else:
                    assert len(h) >= 2 * num_devices, (
                        f"carry block {idx - 1} returned {len(h)} tensors, "
                        f"expected at least {2 * num_devices} (residual+norm)"
                    )
                    hidden_raw = h[:num_devices]
                    hidden_norm = h[num_devices : 2 * num_devices]
                    prev_topk = h[2 * num_devices :]
                if idx < last_layer:
                    next_layer = layers_list[idx + 1]
                    cur_norm = layers_list[idx].input_layernorm
                    assert (
                        next_layer.input_layernorm.eps == cur_norm.eps
                        and next_layer.input_layernorm.weight_offset
                        == cur_norm.weight_offset
                        and next_layer.input_layernorm.multiply_before_cast
                        == cur_norm.multiply_before_cast
                    ), (
                        "carry_next_input_norm assumes uniform input-norm "
                        "eps/weight_offset/mbc across layers"
                    )
                    next_input_gamma = next_input_layernorm_gammas(
                        next_layer, hidden_raw
                    )
            else:
                if len(h) > num_devices:
                    hidden_raw = h[:num_devices]
                    prev_topk = h[num_devices:]
                else:
                    hidden_raw = h
                    prev_topk = []
                hidden_norm = apply_input_layernorm_at_layer_entry(
                    layers_list[idx], hidden_raw
                )
            values: list[SubgraphInput] = [
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                hidden_raw,
                hidden_norm,
                signal_buffers,
                mla_kv_collections,
                indexer_kv_collections,
                freqs_cis,
                mla_prefill_metadata_flat,
                input_row_offsets_,
                prev_topk,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            # Trailing gamma list when carrying end-norm (empty on last block).
            if self.carry_next_input_norm:
                values.append(next_input_gamma or [])
            return values

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=self.subgraph_layer_groups,
            name_for_subgraph=lambda g: f"dist_transformer_block_{g}",
            initial_hidden_states=h,
        )

        # Residual is always the leading per-device group.
        h = h[:num_devices]

        return deepseek_logits_postprocess(
            h=h,
            input_row_offsets=input_row_offsets_,
            all_logits_input_row_offsets=None,
            return_n_logits=return_n_logits,
            norm_shards=self.norm_shards,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            devices=devices,
            is_data_parallel_attention=self.config.data_parallel_degree > 1,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.logits_scaling,
            emit_last_token_logits=self.emit_last_token_logits,
            unpadded_vocab_size=self.config.unpadded_vocab_size,
            vocab_size=self.config.vocab_size,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        # It should be possible to compute the input symbols from the model
        # config.
        device_ref = self.config.devices[0]

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )

        # Add host input row offsets type, this is used to split the
        # concatenated DP inputs.
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=[self.config.data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        )

        signals = Signals(devices=self.config.devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.flattened_kv_inputs())

        # Add batch context lengths
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(self.config.devices))]
        )

        if self.ep_manager is not None:
            all_input_types.extend(self.ep_manager.input_types())
        return tuple(all_input_types)
