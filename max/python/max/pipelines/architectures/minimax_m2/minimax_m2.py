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

"""Implements the MiniMax-M2 model.

MiniMax-M2 is a Mixture-of-Experts decoder-only transformer with:
- GQA attention with QK norm and partial RoPE
- Sigmoid MoE routing with expert score correction bias
- Gated MLP (SwiGLU) experts
"""

from __future__ import annotations

import enum
import functools
from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module, SubgraphInput
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

from .layers.attention import MiniMaxM2Attention
from .layers.moe_gate import MiniMaxM2TopKRouter
from .layers.rotary_embedding import MiniMaxM2RotaryEmbedding
from .model_config import MiniMaxM2Config


class ParallelismMode(enum.Enum):
    """Parallelism strategy for a MiniMax-M2 transformer block.

    Each mode determines which attention/MoE sharding is used and which
    collective communication ops run after attention and after the MoE.
    """

    TP_TP = "tp_tp"
    """TP attention (allreduce after o_proj) + TP MoE (allreduce after).
    Selected when ``data_parallel_degree == 1`` and EP is disabled."""

    TP_EP = "tp_ep"
    """TP attention + EP MoE.  With ``ep_config.use_allreduce`` false (default),
    reduce-scatter after attention puts hidden states in sequence-parallel
    ``[S/P, H]`` form so EP dispatch sees each token once, and allgather after
    MoE restores ``[S, H]``.  With ``use_allreduce`` true the collectives match
    TP_TP (allreduce after attention and after MoE)."""

    DP_EP = "dp_ep"
    """DP attention (batch split per device) + EP MoE.  No inter-device
    collectives in the residual path.  Also used for single-GPU."""


def _select_parallelism_mode(
    num_devices: int,
    data_parallel_degree: int,
    ep_config: EPConfig | None,
) -> ParallelismMode:
    """Select the parallelism strategy from the device/DP/EP configuration.

    Shared by :class:`MiniMaxM2` and every :class:`MiniMaxM2TransformerBlock`
    so the model-level and per-block views of the mode cannot drift. The
    mapping is:

    - single GPU                                -> ``DP_EP`` (no collectives)
    - ``data_parallel_degree == 1``, with EP    -> ``TP_EP``
    - ``data_parallel_degree == 1``, without EP -> ``TP_TP``
    - ``data_parallel_degree > 1``              -> ``DP_EP``

    Note that ``DP_EP`` covers *both* multi-GPU data-parallel and the
    single-GPU fallback. Callers that specifically mean "the batch is split
    across devices" must additionally require ``num_devices > 1`` -- see
    :attr:`MiniMaxM2.dp_attention`.
    """
    if num_devices <= 1:
        return ParallelismMode.DP_EP
    if data_parallel_degree > 1:
        return ParallelismMode.DP_EP
    # data_parallel_degree == 1: TP attention, with EP MoE if configured.
    if ep_config is not None:
        return ParallelismMode.TP_EP
    return ParallelismMode.TP_TP


class MiniMaxM2TransformerBlock(Module):
    """MiniMax-M2 transformer block with GQA attention and MoE feed-forward."""

    def __init__(
        self,
        config: MiniMaxM2Config,
        layer_idx: int,
        rope: RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        assert config.quant_config is not None, (
            "MiniMax-M2 requires quantized weights (FP8, NVFP4, or MXFP4)"
        )
        self.config = config
        self.devices = config.devices
        num_devices = len(config.devices)

        # Select the parallelism strategy for this block (see
        # _select_parallelism_mode for the device/DP/EP -> mode mapping).
        self.mode = _select_parallelism_mode(
            num_devices, config.data_parallel_degree, config.ep_config
        )

        # TP attention is used by both TP_TP and TP_EP.
        self.tp_attention = self.mode in (
            ParallelismMode.TP_TP,
            ParallelismMode.TP_EP,
        )
        self.use_allreduce = (
            config.ep_config.use_allreduce
            if config.ep_config is not None
            else False
        )

        attn_dtype = config.attn_dtype or config.dtype
        attn_is_quantized = (
            attn_dtype == config.dtype and config.quant_config is not None
        )
        attn_quant_config = config.quant_config if attn_is_quantized else None
        attn_linear_cls = linear_cls if attn_is_quantized else Linear

        # Attention layer with per-layer QK norm
        self.self_attn = MiniMaxM2Attention(
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
        if self.tp_attention:
            # TP: heads sharded across devices; reduced after o_proj
            self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            # DP+EP: attention replicated, batch split across devices
            self.self_attn.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # MoE layer (all layers are MoE in MiniMax-M2)
        self.ep_manager = ep_manager
        self.mlp = self._get_mlp(config, linear_cls)
        if self.mode == ParallelismMode.TP_TP:
            # TP: expert intermediate dim split across devices; allreduce after
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            # TP_EP / DP_EP: experts distributed across devices via EP dispatch
            self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                num_devices
            )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Layer norms (replicated)
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
        config: MiniMaxM2Config,
        linear_cls: Callable[..., Linear],
    ) -> MoE:
        """Create the MoE layer with sigmoid routing."""
        moe_cls = MoEQuantized if config.quant_config is not None else MoE
        return moe_cls(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=functools.partial(
                MiniMaxM2TopKRouter,
                norm_topk_prob=config.norm_topk_prob,
                gate_dtype=config.gate_dtype or config.dtype,
                correction_bias_dtype=(
                    config.correction_bias_dtype or DType.float32
                ),
            ),
            dtype=config.dtype,
            quant_config=config.quant_config,
            ep_batch_manager=self.ep_manager,
            ep_size=(
                self.ep_manager.config.n_gpus_per_node
                * self.ep_manager.config.n_nodes
                if self.ep_manager is not None
                else 1
            ),
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
    ) -> list[TensorValue]:
        """Forward pass through the block."""
        # Input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention. TP modes shard heads and produce partial o_proj sums
        # that the post-attention collective reduces; DP replicates attention.
        if self.tp_attention:
            # The per-layer QK RMSNorm is a cross-head norm over the full Q/K
            # projection, but each device holds only a head slice. Project
            # locally, all-reduce the per-token norm statistics so every rank
            # gets the global RMS, then finish attention with the correct norm.
            num_devices = len(self.self_attn_shards)
            projected = [
                shard.tp_project(norm_xs[i], input_row_offsets[i])
                for i, shard in enumerate(self.self_attn_shards)
            ]
            qk_var_local = [p[3] for p in projected]
            qk_var_global = ops.allreduce.sum(qk_var_local, signal_buffers)
            attn_outs = [
                shard.tp_finish(
                    layer_idx,
                    projected[i][0],
                    projected[i][1],
                    projected[i][2],
                    qk_var_global[i] * (1.0 / num_devices),
                    kv_collections[i],
                    freqs_cis[i],
                    input_row_offsets[i],
                )
                for i, shard in enumerate(self.self_attn_shards)
            ]
        else:
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

        # Residual + post-attention collective (allreduce / reduce-scatter).
        hs = self._post_attention(xs, attn_outs, signal_buffers)

        # Post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # Fetch EP buffers before MoE (if using EP)
        if self.ep_manager is not None and ep_inputs is not None:
            self.ep_manager.fetch_buffers(ep_inputs)

        # MoE forward (TP_TP: partial intermediate outputs; EP: dispatch)
        mlp_outs = forward_moe_sharded_layers(self.mlp_shards, norm_outs)

        # Residual + post-MoE collective (allreduce / allgather).
        hs = self._post_mlp(hs, mlp_outs, signal_buffers)

        # Re-link the sequence dim to the block input shape. Under TP+EP the
        # reduce-scatter/allgather round trip yields a fresh symbolic seq dim
        # that the per-layer subgraph cannot prove equals its declared output
        # dim; rebind restores the static link. No-op for TP_TP / DP_EP.
        hs = [ops.rebind(h, x.shape) for h, x in zip(hs, xs, strict=True)]

        return hs

    def _post_attention(
        self,
        xs: list[TensorValue],
        attn_outs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Residual connection and collective after attention."""
        match self.mode:
            case ParallelismMode.TP_TP:
                attn_outs = ops.allreduce.sum(attn_outs, signal_buffers)
                return [
                    x + attn_out
                    for x, attn_out in zip(xs, attn_outs, strict=True)
                ]
            case ParallelismMode.TP_EP:
                if self.use_allreduce:
                    attn_outs = ops.allreduce.sum(attn_outs, signal_buffers)
                    return [
                        x + attn_out
                        for x, attn_out in zip(xs, attn_outs, strict=True)
                    ]
                # attn_outs[i] is device i's partial o_proj sum (no allreduce).
                # Add the residual only on device 0 so it isn't counted P times
                # after the reduce-scatter, which leaves sequence-parallel
                # [S/P, H] hidden states for the EP MoE.
                hs = [xs[0] + attn_outs[0], *attn_outs[1:]]
                return ops.reducescatter.sum(hs, signal_buffers, axis=0)
            case ParallelismMode.DP_EP:
                return [
                    x + attn_out
                    for x, attn_out in zip(xs, attn_outs, strict=True)
                ]
            case _:
                raise ValueError(f"Unsupported parallelism mode: {self.mode}")

    def _post_mlp(
        self,
        hs: list[TensorValue],
        mlp_outs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Residual connection and collective after the MoE."""
        match self.mode:
            case ParallelismMode.TP_TP:
                mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
                return [
                    h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)
                ]
            case ParallelismMode.TP_EP:
                if self.use_allreduce:
                    mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
                    return [
                        h + mlp_out
                        for h, mlp_out in zip(hs, mlp_outs, strict=True)
                    ]
                # Hidden states are sequence-parallel [S/P, H] here; add the
                # residual then allgather to restore the full [S, H] layout.
                hs = [
                    h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)
                ]
                return ops.allgather(hs, signal_buffers, axis=0)
            case ParallelismMode.DP_EP:
                return [
                    h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)
                ]
            case _:
                raise ValueError(f"Unsupported parallelism mode: {self.mode}")


class MiniMaxM2(DistributedLogitsPostprocessMixin, Module):
    """MiniMax-M2 model supporting single-GPU, DP+EP, TP+TP, and TP+EP."""

    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        # Parallelism strategy for the model, derived identically to each
        # transformer block via the shared helper. ``mode`` selects which
        # collectives run in the residual path; ``dp_attention`` is the
        # narrower "the batch is split across devices" predicate.
        #
        # ``mode is DP_EP`` alone is NOT "batch split": DP_EP also covers the
        # single-GPU fallback, where the full batch stays on one device.
        # ``dp_attention`` adds ``num_devices > 1`` to exclude that case, making
        # it provably equal to ``data_parallel_degree > 1`` (the DP degree can
        # never exceed the device count).
        self.mode = _select_parallelism_mode(
            self.num_devices, config.data_parallel_degree, config.ep_config
        )
        self.dp_attention = (
            self.mode is ParallelismMode.DP_EP and self.num_devices > 1
        )
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ MiniMax-M2 is not implemented yet")
        if (
            config.model_quantization_encoding is not None
            and config.model_quantization_encoding != QuantizationEncoding.GPTQ
        ):
            raise NotImplementedError("GGUFQ MiniMax-M2 is not implemented yet")

        # RoPE with proportional scaling for partial rotation
        # MiniMax-M2 has rotary_dim=64, head_dim=128 -> partial_rotary_factor=0.5
        # Uses MiniMaxM2RotaryEmbedding which zeros out non-rotated dims
        rope = MiniMaxM2RotaryEmbedding(
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

        # Norm factory
        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps or 1e-6,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        # Transformer layers
        self.layers = LayerList(
            [
                MiniMaxM2TransformerBlock(
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
            self.subgraph_layer_groups: list[list[int]] = [
                list(range(config.num_hidden_layers))
            ]
        else:
            self.subgraph_layer_groups = []

        # Final norm (replicated)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers
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
        ep_inputs: list[Value[Any]] | None = None,
        data_parallel_splits: TensorValue | None = None,
        host_input_row_offsets: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the model.

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
        # Embeddings (allreduce produces identical states on all GPUs)
        h = self.embed_tokens(tokens, signal_buffers)

        # Distribute RoPE frequencies
        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]

        # Broadcast input_row_offsets to all devices
        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        if self.dp_attention:
            # DP attention: split the replicated batch across GPUs.
            assert data_parallel_splits is not None
            assert host_input_row_offsets is not None
            h, input_row_offsets_list = split_batch_replicated(
                self.devices,
                h,
                input_row_offsets_list,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )
        else:
            # TP / single-GPU: every device processes the full batch. h and
            # input_row_offsets_list are already replicated on all devices by
            # VocabParallelEmbedding + distributed_broadcast above. (Under
            # TP+EP the per-layer reduce-scatter/allgather keeps inter-layer
            # hidden states in the full [S, H] layout, so this path is shared.)
            pass

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

        # Logit postprocessing: gather last token per sequence, norm, lm_head
        last_token_per_dev = [
            ops.gather(h[i], input_row_offsets_list[i][1:] - 1, axis=0)
            for i in range(len(self.devices))
        ]
        if self.dp_attention:
            # DP attention: allgather last tokens from each DP rank so every
            # device sees the full batch before lm_head.
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
            norm_last_token = forward_sharded_layers(
                self.norm_shards, last_token_distributed
            )
        else:
            # TP / single-GPU: every device already holds the full batch, so
            # each holds all last tokens — no allgather needed.
            norm_last_token = forward_sharded_layers(
                self.norm_shards, last_token_per_dev
            )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Get input types for graph construction."""
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

        kv_inputs = kv_params.get_symbolic_inputs()
        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()
        flattened_kv_types = kv_inputs.flatten()

        # Input layout (must match MiniMaxM2Inputs.buffers and the graph-input
        # unpacking in model.py):
        #   tokens, input_row_offsets, return_n_logits,
        #   [data_parallel_splits, host_input_row_offsets]  (DP attention only),
        #   *signals, *kv,
        #   *ep_inputs                                       (EP MoE only)
        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]

        if self.dp_attention:
            # DP attention needs the batch-split tensors.
            base_inputs.append(
                TensorType(
                    DType.int64,
                    shape=["data_parallel_splits_len"],
                    device=DeviceRef.CPU(),
                )
            )
            base_inputs.append(
                TensorType(
                    DType.uint32,
                    shape=["input_row_offsets_len"],
                    device=DeviceRef.CPU(),
                )
            )

        all_inputs = base_inputs + signal_buffer_types + flattened_kv_types

        if self.ep_manager is not None:
            all_inputs += list(self.ep_manager.input_types())

        return tuple(all_inputs)
