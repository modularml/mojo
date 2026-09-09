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

"""Implements the Laguna model.

Laguna is a Mixture-of-Experts decoder-only transformer with:

- GQA attention with per-head QK norm and rotary embeddings
- A per-element softplus gate applied to the attention output
- Sigmoid MoE routing with per-expert score-correction bias
- Gated MLP (SwiGLU) experts

Attention is uniform GQA (64 query / 8 KV heads, full rotary, single RoPE
table); only the MLP type (dense prefix vs sparse MoE) varies per layer, read
from the HF config's ``mlp_layer_types``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

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
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import MLP as DenseMLP
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.moe import MoE, MoEQuantized, forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)

from .layers.attention import LagunaAttention
from .layers.moe_gate import LagunaTopKRouter
from .layers.rotary_embedding import LagunaRotaryEmbedding
from .model_config import LagunaConfig


class LagunaTransformerBlock(Module):
    """Laguna decoder block.

    Attention is uniform GQA with per-head QK norm and a single (YaRN) RoPE
    table. The only per-layer difference is the MLP type: a dense SwiGLU MLP
    for the dense-prefix layers (``mlp_layer_types[i] == "dense"``) and the
    EP-sharded sparse MoE block for the rest.
    """

    def __init__(
        self,
        config: LagunaConfig,
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

        mlp_type = (
            config.mlp_layer_types[layer_idx]
            if config.mlp_layer_types is not None
            else "sparse"
        )
        self.is_sparse_mlp = mlp_type == "sparse"

        # M.1-NVFP4 keeps attention in bf16 (it's in the quant ignore list);
        # only MLP/MoE are FP4. Quantize attention only when its weights are
        # actually a packed/low-precision dtype (auto-detected from q_proj),
        # NOT merely because attn_dtype matches the bf16 compute dtype.
        attn_dtype = config.attn_dtype or config.dtype
        attn_is_quantized = (
            config.quant_config is not None
            and attn_dtype
            not in (
                DType.bfloat16,
                DType.float16,
                DType.float32,
            )
        )
        attn_quant_config = config.quant_config if attn_is_quantized else None
        attn_linear_cls = linear_cls if attn_is_quantized else Linear

        self.self_attn = LagunaAttention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=attn_dtype,
            rope=rope,
            linear_cls=attn_linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            norm_dtype=config.norm_dtype or config.dtype,
            quant_config=attn_quant_config,
        )
        self.self_attn.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        self.ep_manager = ep_manager
        if self.is_sparse_mlp:
            # Sparse and dense layers assign different (both Shardable) types
            # to the same attribute, so annotate the union.
            self.mlp: MoE | DenseMLP = self._get_moe(config, linear_cls)
            if ep_manager is not None:
                # Multi-GPU: expert-parallel sharding via the EP batch
                # manager. Each device owns a slice of the 256 experts;
                # the EP all-to-all dispatch routes tokens to experts.
                self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            else:
                # Single-GPU: no EP. MoE class doesn't accept
                # ``replicate`` — only ``tensor_parallel`` or
                # ``expert_parallel``. With num_devices=1 TP collapses
                # to no-op (each "tensor parallel rank" gets all
                # experts), giving us the local-single-device path.
                self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                    num_devices
                )
            # MoE and dense shards both satisfy this callable-layer type; the
            # annotation keeps the dense-branch assignment below compatible.
            self.mlp_shards: Sequence[Callable[[TensorValue], TensorValue]] = (
                self.mlp.shard(config.devices)
            )
        else:
            self.mlp = self._get_dense_mlp(config, linear_cls)
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
            self.mlp_shards = self.mlp.shard(config.devices)

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

    def _get_moe(
        self,
        config: LagunaConfig,
        linear_cls: Callable[..., Linear],
    ) -> MoE:
        """Creates the sparse MoE block (sigmoid routing + shared experts + scaling).

        ``moe_dim`` here is ``moe_intermediate_size`` (the routed-expert
        intermediate dim from the HF config — NOT the dense-layer
        intermediate, ``intermediate_size_dense``).
        """
        moe_cls = MoEQuantized if config.quant_config is not None else MoE
        return moe_cls(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                LagunaTopKRouter,
                norm_topk_prob=config.norm_topk_prob,
                gate_dtype=config.gate_dtype or config.dtype,
                correction_bias_dtype=(
                    config.correction_bias_dtype or DType.float32
                ),
                routed_scaling_factor=config.moe_routed_scaling_factor,
                router_logit_softcapping=config.moe_router_logit_softcapping,
            ),
            dtype=config.dtype,
            quant_config=config.quant_config,
            # Laguna has an always-on shared expert added to the routed
            # output. MAX MoE handles this when ``has_shared_experts=True``
            # — it materialises a ``self.shared_experts`` MLP and adds its
            # output to the routed sum.
            has_shared_experts=True,
            shared_experts_dim=config.shared_expert_intermediate_size,
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
        config: LagunaConfig,
        linear_cls: Callable[..., Linear],
    ) -> DenseMLP:
        """Builds the dense SwiGLU MLP used for layer 0 (``mlp_layer_types[0] == "dense"``).

        Distinct from the routed-expert MLPs (which use
        ``moe_intermediate_size``). The dense MLP uses
        ``intermediate_size_dense`` from the HF config.
        """
        return DenseMLP(
            dtype=config.dtype,
            quantization_encoding=None,
            # MAX's MLP passes its own ``quant_config`` to the projection
            # Linears (overriding any partial in ``linear_cls``), so set it here
            # to make dense layer 0 NVFP4-packed when the model is quantized.
            # ``None`` for a bf16 config leaves the projections unquantized.
            quant_config=config.quant_config,
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
        """Runs the forward pass through the decoder block."""
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Replicated — no allreduce needed.
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

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.is_sparse_mlp:
            if self.ep_manager is not None:
                # Multi-GPU EP path: dispatch via the EP batch manager.
                if ep_inputs is not None:
                    self.ep_manager.fetch_buffers(ep_inputs)
                mlp_outs = forward_moe_sharded_layers(
                    self.mlp_shards, norm_outs
                )
            else:
                # Single-GPU: each shard is a complete local MoE; run
                # it directly like a replicate-sharded module.
                mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
        else:
            # Dense MLP path (layer 0): replicate shards run per-device, no
            # allreduce.
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Laguna(DistributedLogitsPostprocessMixin, Module):
    """Laguna model supporting single-GPU and multi-GPU TP inference."""

    def __init__(self, config: LagunaConfig) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Laguna is not implemented yet")
        if (
            config.model_quantization_encoding is not None
            and config.model_quantization_encoding != QuantizationEncoding.GPTQ
        ):
            raise NotImplementedError("GGUFQ Laguna is not implemented yet")

        # Single (YaRN) RoPE table. ``partial_rotary_factor`` (1.0 for M.1 =
        # full rotary) and the YaRN scaling come from the HF config.
        rope = LagunaRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(
                partial_rotary_factor=config.partial_rotary_factor,
            ),
        )
        self.rope = rope

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps or 1e-6,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        self.layers = LayerList(
            [
                LagunaTransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    ep_manager=self.ep_manager,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        if config.use_subgraphs:
            # Only the MLP type varies across layers (dense vs sparse), so a
            # single subgraph can't cover both shapes. Group layers by MLP
            # type so each subgraph is reused across structurally identical
            # layers (two groups: the dense prefix and the sparse MoE rest).
            num_layers = config.num_hidden_layers
            mlp_types = config.mlp_layer_types or (["sparse"] * num_layers)
            groups: dict[str, list[int]] = {}
            for i in range(num_layers):
                groups.setdefault(mlp_types[i], []).append(i)
            self.subgraph_layer_groups: list[list[int]] = list(groups.values())
        else:
            self.subgraph_layer_groups = []

        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        embedding_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_dtype = config.quant_config.embedding_output_dtype

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
        """Runs the forward pass through the model.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            ep_inputs: Expert parallelism communication buffers.
            data_parallel_splits: Per-rank token counts for DP splitting.
            host_input_row_offsets: Host-side row offsets for DP splitting.
        """
        # Allreduce produces identical embeddings on all GPUs.
        h = self.embed_tokens(tokens, signal_buffers)

        # Single RoPE table, replicated per device.
        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]

        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # Split replicated batch across GPUs for DP. With DP=1 (single-
        # replica, the default), there is nothing to split — every
        # device sees the same single batch, and we leave ``h`` /
        # ``input_row_offsets_list`` as the broadcast form. The donor
        # required DP > 1 by construction;
        # ``LagunaConfig.construct_kv_params`` now decouples DP degree
        # from device count, so this guard fires for both single-GPU
        # and multi-GPU runs.
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
                freqs_cis,
                input_row_offsets_list,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            return values

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=self.subgraph_layer_groups or None,
            initial_hidden_states=h,
        )

        # DP logit postprocessing: allgather last tokens, then lm_head.
        last_token_per_dev = [
            ops.gather(h[i], input_row_offsets_list[i][1:] - 1, axis=0)
            for i in range(len(self.devices))
        ]
        last_token_distributed = ops.allgather(
            last_token_per_dev, signal_buffers
        )
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Returns the input types for graph construction."""
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
        # DP-N path (multi-replica): graph also takes DP splits + host row
        # offsets. With DP=1 (the only validated configuration; see arch.py)
        # the model's forward doesn't call ``split_batch_replicated`` and the
        # batch processor omits these, so the graph input list must match.
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
