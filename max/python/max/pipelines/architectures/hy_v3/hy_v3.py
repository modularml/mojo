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
"""Hy3-preview graph.

Uniform-GQA decoder-only MoE. The first ``first_k_dense_replace``
layers are dense SwiGLU MLPs; the rest are sparse MoE with sigmoid +
correction-bias top-k routing, a scalar ``router_scaling_factor``,
and one always-on shared expert.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.comm import Allreduce, Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import MLP as DenseMLP
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.moe import MoE, forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)

from .layers.attention import HYV3Attention
from .layers.moe_gate import HYV3TopKRouter
from .model_config import HYV3Config


def _is_dense_layer(config: HYV3Config, layer_idx: int) -> bool:
    return layer_idx < config.first_k_dense_replace


class HYV3TransformerBlock(Module):
    """Hy3 decoder block."""

    def __init__(
        self,
        config: HYV3Config,
        layer_idx: int,
        rope: RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)
        self.layer_idx = layer_idx
        self.is_sparse_mlp = not _is_dense_layer(config, layer_idx)

        attn_dtype = config.dtype
        self.self_attn = HYV3Attention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=attn_dtype,
            rope=rope,
            linear_cls=Linear,
            devices=config.devices,
            scale=config.attention_multiplier,
            qk_norm_eps=config.rms_norm_eps or 1e-6,
            norm_dtype=config.norm_dtype or config.dtype,
        )
        # Tensor-parallel attention when DP=1 (the per-device Q/KV head
        # count then matches the TP-sharded paged KV cache, and the partial
        # o_proj outputs are allreduced below). Under data-parallel the cache
        # is replicated, so attention replicates too. MAX forbids DP+TP, so
        # DP>1 implies replicate.
        tp_degree = num_devices if config.data_parallel_degree == 1 else 1
        if tp_degree > 1:
            self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                tp_degree
            )
            self.attn_allreduce: Allreduce | None = Allreduce(tp_degree)
        else:
            self.self_attn.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.attn_allreduce = None
        self.self_attn_shards = self.self_attn.shard(config.devices)

        self.ep_manager = ep_manager
        self.mlp: DenseMLP | MoE = self._get_mlp(config, linear_cls)
        if self.is_sparse_mlp:
            if ep_manager is not None:
                self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            else:
                self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                    num_devices
                )
        else:
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.mlp_shards: list[DenseMLP | MoE] = list(
            self.mlp.shard(config.devices)
        )

        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

    def _get_mlp(
        self,
        config: HYV3Config,
        linear_cls: Callable[..., Linear],
    ) -> DenseMLP | MoE:
        if self.is_sparse_mlp:
            return self._get_moe(config, linear_cls)
        return self._get_dense_mlp(config, linear_cls)

    def _get_moe(
        self,
        config: HYV3Config,
        linear_cls: Callable[..., Linear],
    ) -> MoE:
        shared_dim = config.moe_intermediate_size * config.num_shared_experts
        return MoE(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                HYV3TopKRouter,
                norm_topk_prob=config.route_norm,
                gate_dtype=config.gate_dtype or config.dtype,
                correction_bias_dtype=(
                    config.correction_bias_dtype or DType.float32
                ),
                routed_scaling_factor=config.router_scaling_factor,
            ),
            dtype=config.dtype,
            quant_config=None,
            has_shared_experts=True,
            shared_experts_dim=shared_dim,
            ep_batch_manager=self.ep_manager,
            ep_size=(
                self.ep_manager.config.n_gpus_per_node
                * self.ep_manager.config.n_nodes
                if self.ep_manager is not None
                else 1
            ),
        )

    def _get_dense_mlp(
        self,
        config: HYV3Config,
        linear_cls: Callable[..., Linear],
    ) -> DenseMLP:
        return DenseMLP(
            dtype=config.dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size_dense,
            devices=config.devices,
            linear_cls=Linear,
            has_bias=False,
            activation_function="silu",
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        attn_outs = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        # Under TP each shard returned a PARTIAL o_proj output (its local
        # head slice). Sum them across devices before the residual add.
        # Replicate (single-GPU) needs no reduction.
        if self.attn_allreduce is not None:
            attn_outs = self.attn_allreduce(
                inputs=attn_outs, signal_buffers=signal_buffers
            )
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.is_sparse_mlp:
            if self.ep_manager is not None:
                if ep_inputs is not None:
                    self.ep_manager.fetch_buffers(ep_inputs)
                mlp_outs = forward_moe_sharded_layers(
                    self.mlp_shards, norm_outs
                )
            else:
                mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
        else:
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]
        return hs


class HYV3(DistributedLogitsPostprocessMixin, Module):
    """Hy3-preview transformer."""

    def __init__(self, config: HYV3Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        if config.model_quantization_encoding is not None:
            raise NotImplementedError(
                "Quantized Hy3-preview is not implemented yet"
            )

        # interleaved=False -> NeoX split-half rotate (matches HF).
        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=False,
        )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps or 1e-6,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=None)

        self.layers = LayerList(
            [
                HYV3TransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=self.rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    ep_manager=self.ep_manager,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        if config.use_subgraphs:
            # Sparse layers share one subgraph body; the dense layer(s)
            # share another. This is essential for tractable compile time:
            # the 79 sparse MoE layers are structurally identical, as are
            # the dense layer(s).
            groups: dict[str, list[int]] = {}
            for i in range(config.num_hidden_layers):
                key = "dense" if _is_dense_layer(config, i) else "sparse"
                groups.setdefault(key, []).append(i)
            self.subgraph_layer_groups: list[list[int]] = list(groups.values())
        else:
            self.subgraph_layer_groups = []

        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        embedding_dtype = config.dtype
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_dtype,
            config.devices,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
        data_parallel_splits: TensorValue | None = None,
        host_input_row_offsets: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        freqs_cis_list = [
            self.rope.freqs_cis.to(device) for device in self.devices
        ]

        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        if (
            data_parallel_splits is not None
            and host_input_row_offsets is not None
            and self.config.data_parallel_degree > 1
        ):
            h, input_row_offsets_list = split_batch_replicated(
                self.devices,
                h,
                input_row_offsets_list,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        def inputs_for_layer(
            idx: int, h: list[TensorValue]
        ) -> list[SubgraphInput]:
            values: list[SubgraphInput] = [
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                signal_buffers,
                kv_collections,
                freqs_cis_list,
                input_row_offsets_list,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            return values

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=(self.subgraph_layer_groups or None),
            initial_hidden_states=h,
        )

        last_token_per_dev = [
            ops.gather(h[i], input_row_offsets_list[i][1:] - 1, axis=0)
            for i in range(len(self.devices))
        ]
        if self.config.data_parallel_degree > 1:
            # DP-N: each device holds a different batch shard;
            # allgather concatenates them along batch.
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            # DP=1: TP attention allreduces its o_proj output and MoE
            # allreduces too, so every device already holds the same
            # hidden state. Allgather would concatenate identical replicas
            # and break the sampler batch dim.
            last_token_distributed = last_token_per_dev
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        # Upcast lm_head output to FP32 for sampler stability (HF hy_v3
        # returns BF16 logits; values are equivalent within BF16 rounding).
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=["data_parallel_splits_len"],
            device=DeviceRef.CPU(),
        )
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        if self.config.data_parallel_degree > 1:
            base_inputs += [
                data_parallel_splits_type,
                host_input_row_offsets_type,
            ]

        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()
        flattened_kv_types = kv_inputs.flatten()

        ep_input_types: list[TensorType | BufferType] = []
        if self.ep_manager is not None:
            ep_input_types = list(self.ep_manager.input_types())

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + ep_input_types
        )
