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
"""Shared Eagle3 draft model for DeepseekV3-shaped MLA targets.

Eagle3 fuses 3 target hidden states (early, middle, last layer) via a linear
projection (``fc``: ``3*H -> H``), concatenates the result with the token
embedding (``H + H = 2*H``), and feeds that through a single MLA decoder layer
configured to ``hidden_size = 2*H``, followed by a dense MLP and the output
projection. Shared by the Kimi K2.5 and DeepseekV3 Eagle3 draft paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.attention.multi_latent_attention import MLAPrefillMetadata
from max.nn.comm import Signals
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.transformer import ReturnLogits
from max.nn.transformer.distributed_transformer import (
    extract_hs,
    forward_sharded_layers,
)
from max.nn.transformer.transformer import fuse_captured_hidden_states

from ..deepseekV3.deepseekV3 import DeepseekV3DecoderLayer
from ..deepseekV3.model_config import DeepseekV3Config


class Eagle3MLADraft(Module):
    """Eagle3 draft model for DeepseekV3-shaped MLA targets.

    Paired with a DeepseekV3-architecture target (DeepseekV3 or Kimi K2.5);
    target-specific wrappers subclass this without overrides.
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        embedding_output_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        self.embedding_output_dtype = embedding_output_dtype

        # Shared with target (aliased before weight loading)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=embedding_output_dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        # fc: fuses 3 hidden states [seq, 3*H] -> [seq, H]
        self.fc = Linear(
            config.hidden_size * 3,
            config.hidden_size,
            embedding_output_dtype,
            devices[0],
            quantization_encoding=None,
            has_bias=False,
        )
        self.fc.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.fc_shards = self.fc.shard(devices)

        if config.rope_scaling is None:
            raise ValueError(
                "Eagle3MLADraft requires DeepseekYarn-style rope_scaling on"
                " the model config; got None."
            )
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
            config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
        )

        self.use_tp_ep = config.data_parallel_degree == 1 and num_devices > 1

        wide_config = replace(config, hidden_size=config.hidden_size * 2)
        self.decoder_layer = DeepseekV3DecoderLayer(
            self.rope,
            wide_config,
            layer_idx=0,  # Dense MLP (idx < first_k_dense_replace)
            ep_manager=None,  # No EP for dense MLP
        )

        # The draft uses a dense MLP (not EP-MoE) so we don't need
        # sequence-parallel form.  Disable skip_allreduce so the TP MLA
        # allreduces internally — the residual `fused + attn_out` requires
        # attn_out to be replicated since fused_hs is replicated.
        if self.use_tp_ep:
            self.decoder_layer.self_attn.skip_allreduce = False

        self.decoder_layer.input_layernorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.decoder_layer.input_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.decoder_layer.input_layernorm_shards = (
            self.decoder_layer.input_layernorm.shard(devices)
        )

        self.hidden_norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.hidden_norm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.hidden_norm_shards = self.hidden_norm.shard(devices)

        self.decoder_layer.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.decoder_layer.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.decoder_layer.post_attention_layernorm_shards = (
            self.decoder_layer.post_attention_layernorm.shard(devices)
        )

        replacement_o_proj = Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            DType.bfloat16,
            devices[0],
            quantization_encoding=None,
        )
        if self.use_tp_ep:
            replacement_o_proj.sharding_strategy = ShardingStrategy.columnwise(
                num_devices
            )
        else:
            replacement_o_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        self.decoder_layer.self_attn.o_proj = replacement_o_proj
        # The MLA eagerly shards into list_of_attentions during __init__,
        # so we must propagate the replacement to every per-device shard.
        o_proj_shards = replacement_o_proj.shard(devices)
        for shard_idx, attn_shard in enumerate(
            self.decoder_layer.self_attn.list_of_attentions
        ):
            attn_shard.o_proj = o_proj_shards[shard_idx]

        dense_mlp = MLP(
            dtype=config.dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            devices=config.devices,
            quant_config=config.quant_config,
        )
        if self.use_tp_ep:
            dense_mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            dense_mlp.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        self.decoder_layer.mlp = dense_mlp
        self.decoder_layer.mlp_shards = list(dense_mlp.shard(devices))

        self.norm = RMSNorm(
            config.hidden_size,
            config.norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
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

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = 1.0
        self.use_data_parallel_attention = (
            num_devices > 1 and config.data_parallel_degree == num_devices
        )

    def __call__(
        self,
        tokens: TensorValue,
        fused_target_hs: Sequence[TensorValue]
        | Sequence[Sequence[TensorValue]],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        split_prefix: str = "eagle3_draft",
    ) -> tuple[TensorValue, ...]:
        """Forward pass of the Eagle3 draft model.

        Args:
            tokens: Input token IDs.
            fused_target_hs: Per-device hidden states — either the target's 3
                captured layers (a list per device) at step 0, or the previous
                draft step's output (one ``[local_seq, hidden_size]`` tensor
                per device). The caller must produce per-device values already
                sharded to each device's local DP batch; ``fc`` runs
                per-device on the captures.
            split_prefix: Prefix for symbolic dim names. Must be unique per
                graph invocation to avoid dim conflicts between prefill
                (step 0) and decode (step 1+).
        """
        devices = self.config.devices

        captures_per_dev = [
            [hs] if isinstance(hs, TensorValue) else list(hs)
            for hs in fused_target_hs
        ]
        fused_hs = fuse_captured_hidden_states(captures_per_dev)
        if fused_hs[0].shape[-1] != self.config.hidden_size:
            fused_width = self.config.hidden_size * 3
            assert fused_hs[0].shape[-1] == fused_width, (
                f"Eagle3MLADraft expects 3 captures or one already-fused "
                f"[seq, {fused_width}] hidden state per device, got width "
                f"{fused_hs[0].shape[-1]}"
            )
            fused_hs = forward_sharded_layers(self.fc_shards, fused_hs)

        h_embed = self.embed_tokens(tokens, signal_buffers)

        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        input_row_offsets_ = list(input_row_offsets)

        if self.use_data_parallel_attention:
            host_offsets_i64 = host_input_row_offsets.cast(DType.int64)
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
                    [f"{split_prefix}_seq_dev_{i}", self.config.hidden_size],
                )
                for i in range(len(devices))
            ]
            fused_hs = [
                ops.rebind(
                    fused_hs[i],
                    [f"{split_prefix}_seq_dev_{i}", self.config.hidden_size],
                )
                for i in range(len(devices))
            ]
        else:
            # TP or single-device case: rebind both h_embed and fused_hs.
            # Use a COMMON dim name so reduce-scatter/allgather (which
            # require matching shapes) work in TP mode.
            common_dim = f"{split_prefix}_seq_len"
            h_embed = [
                ops.rebind(
                    h_embed[i],
                    [common_dim, self.config.hidden_size],
                )
                for i in range(len(devices))
            ]
            fused_hs = [
                ops.rebind(
                    fused_hs[i],
                    [common_dim, self.config.hidden_size],
                )
                for i in range(len(devices))
            ]

        norm_embed = forward_sharded_layers(
            self.decoder_layer.input_layernorm_shards, h_embed
        )
        norm_fused = forward_sharded_layers(self.hidden_norm_shards, fused_hs)

        concat_inputs = [
            ops.concat([norm_embed[i], norm_fused[i]], axis=-1)
            for i in range(len(devices))
        ]
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = (
                self.decoder_layer.self_attn.create_mla_prefill_metadata(
                    input_row_offsets_, kv_collections
                )
            )
            assert len(mla_prefill_metadata) == len(batch_context_lengths)
            for i in range(len(batch_context_lengths)):
                mla_prefill_metadata[i].buffer_lengths = batch_context_lengths[
                    i
                ]
        mla_inputs: list[TensorValue] = []
        for metadata in mla_prefill_metadata:
            mla_inputs.extend(
                [
                    metadata.buffer_row_offsets,
                    metadata.cache_offsets,
                    metadata.buffer_lengths,
                ]
            )

        attn_outs = self.decoder_layer.self_attn(
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            concat_inputs,
            signal_buffers,
            kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets_,
            mla_prefill_metadata=mla_prefill_metadata,
        )
        hs = [
            fused + attn_out
            for fused, attn_out in zip(fused_hs, attn_outs, strict=True)
        ]

        norm_outs = forward_sharded_layers(
            self.decoder_layer.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_moe_sharded_layers(
            self.decoder_layer.mlp_shards, norm_outs
        )
        if self.use_tp_ep:
            mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        if self.config.data_parallel_degree > 1:
            last_token_per_dev: list[TensorValue] = []
            for dev_idx in range(len(devices)):
                h0 = hs[dev_idx]
                last_token_indices = input_row_offsets_[dev_idx][1:] - 1
                last_token_h = ops.gather(h0, last_token_indices, axis=0)
                last_token_per_dev.append(last_token_h)
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            last_token_distributed = [
                ops.gather(h_i, offsets_i[1:] - 1, axis=0)
                for h_i, offsets_i in zip(hs, input_row_offsets_, strict=True)
            ]

        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        ret_val: tuple[TensorValue, ...] = (last_logits,)

        if self.return_logits == ReturnLogits.VARIABLE:
            draft_return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="draft_return_n_logits_range",
                dtype=DType.int64,
                device=devices[0],
            )
            draft_return_n_logits_range_per_dev = ops.distributed_broadcast(
                draft_return_n_logits_range, signal_buffers
            )
            if self.use_data_parallel_attention:
                # DP: each device has a batch shard; gather per-device then
                # allgather to reconstruct the full batch.
                variable_per_dev: list[TensorValue] = []
                for dev_idx in range(len(devices)):
                    dev_offsets = (
                        ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                        - draft_return_n_logits_range_per_dev[dev_idx]
                    )
                    variable_per_dev.append(
                        ops.gather(
                            hs[dev_idx],
                            ops.reshape(dev_offsets, shape=(-1,)),
                            axis=0,
                        )
                    )
                variable_distributed = ops.allgather(
                    variable_per_dev, signal_buffers
                )
                norm_variable = forward_sharded_layers(
                    self.norm_shards, variable_distributed
                )
                variable_logits = ops.cast(
                    self.lm_head(norm_variable, signal_buffers)[0],
                    DType.float32,
                )
            else:
                # TP: tokens replicated; gather from the same offsets on each
                # device, then norm + lm_head across all devices.
                # (input_row_offsets_ is same for all devices)
                last_indices_per_dev = [
                    ops.reshape(
                        ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                        - draft_return_n_logits_range_per_dev[dev_idx],
                        shape=(-1,),
                    )
                    for dev_idx in range(len(devices))
                ]
                gathered = [
                    ops.gather(h_dev, last_indices, axis=0)
                    for h_dev, last_indices in zip(
                        hs, last_indices_per_dev, strict=True
                    )
                ]  # [batch*num_steps, hidden] per shard
                norm_variable = forward_sharded_layers(
                    self.norm_shards, gathered
                )
                variable_logits = ops.cast(
                    self.lm_head(norm_variable, signal_buffers)[0],
                    DType.float32,
                )
            logit_offsets = ops.range(
                0,
                TensorValue(variable_logits.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="draft_logit_offsets",
                dtype=DType.int64,
                device=devices[0],
            )
            ret_val += (variable_logits, logit_offsets)

        ret_val += extract_hs(
            return_hidden_states=self.return_hidden_states,
            last_token_hs_distributed=last_token_distributed,
            all_hs_distributed=hs,
            normalizer=self.norm_shards,
        )

        return ret_val

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        devices = self.config.devices
        device_ref = devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        fused_hs_type = TensorType(
            DType.bfloat16,
            shape=["total_seq_len", self.config.hidden_size * 3],
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
            fused_hs_type,
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

        return tuple(all_input_types)
