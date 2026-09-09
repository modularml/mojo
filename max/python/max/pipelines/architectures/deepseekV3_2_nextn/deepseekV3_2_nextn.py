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
"""Implements the DeepseekV3.2 NextN (Next-N token prediction) draft model.

Similar to DeepseekV3NextN, but uses sparse attention.
"""

from __future__ import annotations

from typing import Any

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
from max.nn.kernels import mtp_eh_norm
from max.nn.kv_cache import (
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.transformer.distributed_transformer import forward_sharded_layers

from ..deepseekV3.deepseekV3 import deepseek_logits_postprocess
from ..deepseekV3_2.deepseekV3_2 import DeepseekV3_2DecoderLayer
from .model_config import DeepseekV3_2NextNConfig


class DeepseekV3_2NextN(Module):
    """DeepSeek-V3.2 NextN draft: embed + enorm/hnorm + eh_proj + sparse decoder."""

    def __init__(self, config: DeepseekV3_2NextNConfig) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        self.use_data_parallel_attention = (
            num_devices > 1 and config.data_parallel_degree == num_devices
        )

        embedding_output_dtype = config.dtype
        if embedding_output_dtype == DType.uint8:
            embedding_output_dtype = DType.bfloat16
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        self.embedding_output_dtype = embedding_output_dtype
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        self.enorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.enorm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.enorm_shards = self.enorm.shard(devices)

        self.hnorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.hnorm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.hnorm_shards = self.hnorm.shard(devices)

        # `mtp_eh_norm` implements only the Llama-style arm of `ops.rms_norm`,
        # and applies one epsilon to both halves.
        for _norm in (self.enorm, self.hnorm):
            assert _norm.weight_offset == 0.0, (
                "mtp_eh_norm assumes weight_offset == 0"
            )
            assert not _norm.multiply_before_cast, (
                "mtp_eh_norm assumes multiply_before_cast is False"
            )
        assert self.enorm.eps == self.hnorm.eps, (
            "mtp_eh_norm applies one epsilon to both normalizations"
        )

        self.eh_proj = Linear(
            config.hidden_size * 2,
            config.hidden_size,
            embedding_output_dtype,
            devices[0],
            quantization_encoding=None,
            has_bias=False,
        )
        self.eh_proj.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.eh_proj_shards = self.eh_proj.shard(devices)

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

        # Ensure MoE layer creation by using layer_idx >= first_k_dense_replace.
        # ``indexer_types`` is set empty on the draft config so this single
        # layer is a ``full`` indexer layer (skip_topk=False).
        nextn_layer_idx = max(
            config.num_hidden_layers, config.first_k_dense_replace
        )
        self.decoder_layer = DeepseekV3_2DecoderLayer(
            self.rope,
            config,
            layer_idx=nextn_layer_idx,
            ep_manager=self.ep_manager,
        )

        self.shared_head_norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.shared_head_norm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.shared_head_norm_shards = self.shared_head_norm.shard(devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        # When False, ``deepseek_logits_postprocess`` skips the last-token
        # lm_head projection and omits ``last_logits`` from the output tuple.
        # Default True keeps the standalone ragged contract (``last_logits``
        # at index 0) unchanged.
        self.emit_last_token_logits = True
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        hidden_state: list[TensorValue],
        signal_buffers: list[BufferValue],
        mla_kv_collections: list[PagedCacheValues],
        indexer_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        prev_topk_indices: list[TensorValue] | None = None,
        reuse_prev_topk: bool = False,
        split_prefix: str = "draft",
    ) -> tuple[TensorValue, ...]:
        if not host_input_row_offsets.device == DeviceRef.CPU():
            raise ValueError("host_input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices
        n_devs = len(devices)

        h_embed = self.embed_tokens(tokens, signal_buffers)

        hidden_states = list(hidden_state)
        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        input_row_offsets_ = list(input_row_offsets)
        all_logits_input_row_offsets = input_row_offsets_[0]
        if self.use_data_parallel_attention:
            host_offsets_i64 = host_input_row_offsets.cast(DType.int64)
            # Splitting before normalizing is equivalent: the split slices
            # the token axis, and RMSNorm reduces within a row.
            h_embed, input_row_offsets_ = split_batch_replicated(
                devices,
                h_embed,
                input_row_offsets_,
                host_offsets_i64,
                data_parallel_splits,
                prefix=split_prefix,
            )

            h_embed = [
                ops.rebind(
                    h_embed[i],
                    [
                        f"{split_prefix}_seq_len_device_{i}",
                        self.config.hidden_size,
                    ],
                )
                for i in range(n_devs)
            ]
            hidden_states = [
                ops.rebind(
                    hidden_states[i],
                    [
                        f"{split_prefix}_seq_len_device_{i}",
                        self.config.hidden_size,
                    ],
                )
                for i in range(n_devs)
            ]
        else:
            # TP or single-device case: use a COMMON dim name so collectives
            # that require matching shapes work in TP mode.
            common_dim = f"{split_prefix}_seq_len"
            h_embed = [
                ops.rebind(
                    h_embed[i],
                    [common_dim, self.config.hidden_size],
                )
                for i in range(n_devs)
            ]
            hidden_states = [
                ops.rebind(
                    hidden_states[i],
                    [common_dim, self.config.hidden_size],
                )
                for i in range(n_devs)
            ]

        concat_inputs = [
            mtp_eh_norm(
                h_embed[i],
                hidden_states[i],
                self.enorm_shards[i]
                .weight.cast(h_embed[i].dtype)
                .to(h_embed[i].device),
                self.hnorm_shards[i]
                .weight.cast(h_embed[i].dtype)
                .to(h_embed[i].device),
                self.enorm.eps,
            )
            for i in range(n_devs)
        ]
        h = forward_sharded_layers(self.eh_proj_shards, concat_inputs)
        h_norm = forward_sharded_layers(
            self.decoder_layer.input_layernorm_shards, h
        )

        # Create MLA prefill metadata if not in decode mode.
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = (
                self.decoder_layer.self_attn.create_mla_prefill_metadata(
                    input_row_offsets_, mla_kv_collections
                )
            )
            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]

        mla_prefill_metadata_flat: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_prefill_metadata_flat.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        layer_outs = self.decoder_layer(
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            h,
            h_norm,
            signal_buffers,
            mla_kv_collections,
            indexer_kv_collections,
            freqs_cis,
            mla_prefill_metadata_flat,
            input_row_offsets_,
            prev_topk_indices if prev_topk_indices is not None else [],
            ep_inputs=ep_inputs,
            reuse_prev_topk=reuse_prev_topk,
        )

        # The decoder layer returns ``hidden_states + topk_indices`` (both
        # per-device lists concatenated).
        h = layer_outs[:n_devs]
        topk_indices = layer_outs[n_devs:]

        postprocess = deepseek_logits_postprocess(
            h=h,
            input_row_offsets=input_row_offsets_,
            all_logits_input_row_offsets=all_logits_input_row_offsets,
            return_n_logits=return_n_logits,
            norm_shards=self.shared_head_norm_shards,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            devices=devices,
            is_data_parallel_attention=self.use_data_parallel_attention,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.logits_scaling,
            emit_last_token_logits=self.emit_last_token_logits,
            unpadded_vocab_size=self.config.unpadded_vocab_size,
            vocab_size=self.config.vocab_size,
        )

        # Append the per-device top-k selection so the unified MTP module can
        # gather and reuse it across subsequent draft steps.
        return (*postprocess, *topk_indices)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        devices = self.config.devices
        device_ref = devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        hidden_state_type = TensorType(
            self.embedding_output_dtype,
            shape=["total_seq_len", self.config.hidden_size],
            device=device_ref,
        )
        device_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )
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

        signals = Signals(devices=devices)
        signal_buffer_types: list[BufferType] = signals.input_types()

        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            hidden_state_type,
            device_input_row_offsets_type,
            host_input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
        ]
        all_input_types.extend(signal_buffer_types)
        all_input_types.extend(kv_params.flattened_kv_inputs())

        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            [batch_context_length_type for _ in range(len(devices))]
        )

        if self.ep_manager is not None:
            all_input_types.extend(self.ep_manager.input_types())

        return tuple(all_input_types)
