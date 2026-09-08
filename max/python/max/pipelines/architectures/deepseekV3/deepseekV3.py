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
"""Implements the DeepseekV3 model."""

from __future__ import annotations

import enum
import functools
from collections.abc import Callable, Sequence
from typing import Any

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
from max.nn.attention.multi_latent_attention import (
    DataParallelLatentAttentionWithRope,
    MLAPrefillMetadata,
    TensorParallelLatentAttentionWithRope,
)
from max.nn.attention.multi_latent_attention_fp8 import (
    DataParallelLatentAttentionWithRopeFp8,
    TensorParallelLatentAttentionWithRopeFp8,
)
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.moe import MoE, MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.transformer import (
    ReturnHiddenStates,
    ReturnLogits,
    forward_sequential_layers,
)
from max.nn.transformer.distributed_transformer import (
    extract_hs,
    forward_sharded_layers,
)

from .layers.moe_gate import DeepseekV3TopKRouter
from .model_config import DeepseekV3Config


class ParallelismMode(enum.Enum):
    """Parallelism strategy for a DeepseekV3 decoder layer.

    Each mode determines which attention/MoE implementations are used and which
    collective communication ops run after attention and after the MoE/MLP.
    """

    DP_EP = "dp_ep"
    """DP attention + EP MoE.  No inter-device collectives in the residual path."""

    TP_EP = "tp_ep"
    """TP attention (skip allreduce) + EP MoE.  Reduce-scatter after attention
    puts hidden states in sequence-parallel ``[S/P, H]`` form; allgather after
    MoE restores ``[S, H]``."""

    TP_TP = "tp_tp"
    """TP attention (with allreduce) + TP MoE.  Standard allreduce after MoE."""


def _validate_parallelism_config(config: DeepseekV3Config) -> None:
    """Validate parallelism configuration for DeepseekV3.

    Supported multi-GPU modes:
      - DP attention + EP MoE: ``data_parallel_degree == num_devices``
      - TP attention + EP MoE: ``data_parallel_degree == 1``
      - TP attention + TP MoE: ``data_parallel_degree == 1``, no EP
    ``DeepseekV3Config.__post_init__`` already enforces
    ``data_parallel_degree in (1, num_devices)``.
    """
    num_devices = len(config.devices)
    # TP+TP (data_parallel_degree == 1, no ep_config) is valid.
    # Only require EP when using data-parallel attention on multiple GPUs.
    # Skip EP validation in virtual device mode (compilation-only) since EP
    # will be disabled later due to NVSHMEM linking requirements.
    if (
        num_devices > 1
        and config.ep_config is None
        and config.data_parallel_degree != 1
        and not is_virtual_device_mode()
    ):
        raise ValueError(
            "Expert-parallel (ep_config) must be enabled for multi-GPU"
            " DeepseekV3 with data-parallel attention."
        )


def mask_padded_tail(
    logits: TensorValue, vocab_size: int, unpadded_vocab_size: int | None
) -> TensorValue:
    """Sends the dummy/padding rows of the vocabulary to negative infinity."""
    if unpadded_vocab_size is None or unpadded_vocab_size >= vocab_size:
        return logits
    device = logits.device
    # Two broadcast scalars rather than a materialized row: keeps a
    # vocab-sized fp32 constant out of the graph.
    keep = ops.broadcast_to(
        ops.constant(0.0, DType.float32, device=device), [unpadded_vocab_size]
    )
    drop = ops.broadcast_to(
        ops.constant(float("-inf"), DType.float32, device=device),
        [vocab_size - unpadded_vocab_size],
    )
    return logits + ops.cast(ops.concat([keep, drop]), logits.dtype)


def deepseek_logits_postprocess(
    h: list[TensorValue],
    input_row_offsets: list[TensorValue],
    all_logits_input_row_offsets: TensorValue | None,
    return_n_logits: TensorValue,
    norm_shards: Sequence[Callable[[TensorValue], TensorValue]],
    lm_head: Callable[
        [list[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
    ],
    signal_buffers: list[BufferValue],
    devices: list[DeviceRef],
    is_data_parallel_attention: bool,
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates,
    logits_scaling: float = 1.0,
    capture_hidden_states: list[list[TensorValue]] | None = None,
    emit_last_token_logits: bool = True,
    unpadded_vocab_size: int | None = None,
    vocab_size: int | None = None,
) -> tuple[TensorValue, ...]:
    """Logits postprocessing for DeepseekV3 and DeepseekV3NextN.

    Handles last-token gathering, DP-attention-specific allgather (needed
    because ``ColumnParallelLinear`` expects the full batch on each device),
    variable / all logits computation, logits scaling, and hidden-states
    extraction.

    When ``emit_last_token_logits`` is False, the last-token norm + full-vocab
    lm_head projection (and, under DP attention, its last-token allgather) is
    skipped and ``last_logits`` is omitted from the output tuple. Callers that
    consume only the VARIABLE logits (e.g. the unified MTP graph) set this to
    avoid an unused vocab-sized GEMM + collective per step.

    Returns:
        ``([last_logits], [logits, offsets], [hidden_states])`` — the optional
        segments are present only when the corresponding mode is active.
    """
    if is_data_parallel_attention:
        last_token_per_dev: list[TensorValue] = []
        for dev_idx in range(len(devices)):
            h0 = h[dev_idx]
            last_token_indices = input_row_offsets[dev_idx][1:] - 1
            last_token_h = ops.gather(h0, last_token_indices, axis=0)
            last_token_per_dev.append(last_token_h)
        # ``last_token_distributed`` is only consumed by the last-token lm_head
        # below and by ``extract_hs`` for the LAST / LAST_PER_DEVICE modes;
        # callers that suppress ``last_logits`` pair it with ALL /
        # ALL_NORMALIZED hidden states, so the allgather is skipped too.
        if emit_last_token_logits:
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            last_token_distributed = last_token_per_dev
    else:
        last_token_distributed = [
            ops.gather(h_i, offsets_i[1:] - 1, axis=0)
            for h_i, offsets_i in zip(h, input_row_offsets, strict=True)
        ]

    last_logits: TensorValue | None = None
    if emit_last_token_logits:
        norm_last_token = forward_sharded_layers(
            norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

    logits = None
    offsets = None

    if return_logits == ReturnLogits.VARIABLE:
        # Compute the range on device 0 and broadcast to all devices.
        # Using distributed_broadcast instead of per-device .to() copies
        # avoids cross-stream D2D event sync that breaks CUDA graph
        # capture. Per-device ops.range with a shared out_dim was also
        # attempted and hit "input device gpu:0 must match result device
        # gpu:1 in rebind()" — the shared symbolic dim triggers a cross-
        # device rebind downstream.
        return_n_logits_range = ops.range(
            start=return_n_logits[0],
            stop=0,
            step=-1,
            out_dim="return_n_logits_range",
            dtype=DType.int64,
            device=devices[0],
        )
        return_n_logits_range_per_dev = ops.distributed_broadcast(
            return_n_logits_range, signal_buffers
        )
        variable_per_dev: list[TensorValue] = []
        for dev_idx in range(len(devices)):
            dev_offsets = (
                ops.unsqueeze(input_row_offsets[dev_idx][1:], -1)
                - return_n_logits_range_per_dev[dev_idx]
            )
            dev_indices = ops.reshape(dev_offsets, shape=(-1,))
            variable_per_dev.append(ops.gather(h[dev_idx], dev_indices, axis=0))
        if is_data_parallel_attention:
            variable_per_dev = ops.allgather(variable_per_dev, signal_buffers)

        logits = ops.cast(
            lm_head(
                forward_sharded_layers(norm_shards, variable_per_dev),
                signal_buffers,
            )[0],
            DType.float32,
        )
        offsets = ops.range(
            0,
            TensorValue(variable_per_dev[0].shape[0]) + return_n_logits[0],
            return_n_logits[0],
            out_dim="logit_offsets",
            dtype=DType.int64,
            device=devices[0],
        )
    elif return_logits == ReturnLogits.ALL:
        if is_data_parallel_attention:
            h = ops.allgather(h, signal_buffers)
        logits = ops.cast(
            lm_head(
                forward_sharded_layers(norm_shards, h),
                signal_buffers,
            )[0],
            DType.float32,
        )
        offsets = (
            all_logits_input_row_offsets
            if all_logits_input_row_offsets is not None
            else input_row_offsets[0]
        )

    if unpadded_vocab_size is not None:
        assert vocab_size is not None, (
            "vocab_size must accompany unpadded_vocab_size"
        )
        if last_logits is not None:
            last_logits = mask_padded_tail(
                last_logits, vocab_size, unpadded_vocab_size
            )
        if logits is not None:
            logits = mask_padded_tail(logits, vocab_size, unpadded_vocab_size)

    if logits_scaling != 1.0:
        if last_logits is not None:
            last_logits = last_logits / logits_scaling
        if logits is not None:
            logits = logits / logits_scaling

    assert last_logits is not None or logits is not None, (
        "skipping last_logits requires VARIABLE/ALL logits to be emitted"
    )
    ret_val: tuple[TensorValue, ...] = (
        () if last_logits is None else (last_logits,)
    )
    if logits is not None and offsets is not None:
        ret_val += (logits, offsets)

    ret_val += extract_hs(
        return_hidden_states=return_hidden_states,
        last_token_hs_distributed=last_token_distributed,
        all_hs_distributed=h,
        normalizer=norm_shards,
        capture_hidden_states=capture_hidden_states,
    )

    return ret_val


class DeepseekV3DecoderLayer(Module):
    def __init__(
        self,
        rope: RotaryEmbedding,
        config: DeepseekV3Config,
        layer_idx: int,
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ep_manager = ep_manager
        num_devices = len(config.devices)

        if num_devices <= 1:
            self.mode = ParallelismMode.DP_EP
        elif config.ep_config is not None:
            if config.data_parallel_degree == 1:
                self.mode = ParallelismMode.TP_EP
            else:
                self.mode = ParallelismMode.DP_EP
        else:
            self.mode = ParallelismMode.TP_TP

        # Create Multi-head Latent Attention layer.
        mla_kwargs: dict[str, Any] = dict(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            devices=config.devices,
            graph_mode=config.graph_mode,
            buffer_size=config.max_batch_context_length,
            norm_dtype=config.norm_dtype,
        )

        nvfp4_enabled = (
            config.quant_config is not None and config.quant_config.is_nvfp4
        )
        use_fp8_mla = (
            config.quant_config is not None and not config.quant_config.is_fp4
        )

        if (
            nvfp4_enabled
            and config.n_routed_experts
            != 384  # nvidia/KimiK2.5-NVFP4 out projections are not quantized
        ):
            mla_kwargs["o_proj_quant_config"] = config.quant_config
            mla_kwargs["o_proj_dtype"] = config.dtype

        mla_cls: type[
            DataParallelLatentAttentionWithRope
            | DataParallelLatentAttentionWithRopeFp8
            | TensorParallelLatentAttentionWithRope
            | TensorParallelLatentAttentionWithRopeFp8
        ]
        match self.mode:
            case ParallelismMode.TP_EP:
                # TP attention + EP MoE: the cross-device communication is
                # handled by the EP MoE, so skip the attention all-reduce.
                mla_kwargs["skip_allreduce"] = True
                if use_fp8_mla:
                    mla_kwargs["quant_config"] = config.quant_config
                    mla_cls = TensorParallelLatentAttentionWithRopeFp8
                else:
                    mla_kwargs["dtype"] = DType.bfloat16
                    mla_cls = TensorParallelLatentAttentionWithRope
            case ParallelismMode.TP_TP:
                mla_kwargs["skip_allreduce"] = False
                if use_fp8_mla:
                    mla_kwargs["quant_config"] = config.quant_config
                    mla_cls = TensorParallelLatentAttentionWithRopeFp8
                else:
                    mla_kwargs["dtype"] = DType.bfloat16
                    mla_cls = TensorParallelLatentAttentionWithRope
            case ParallelismMode.DP_EP:
                if use_fp8_mla:
                    mla_kwargs["quant_config"] = config.quant_config
                    mla_cls = DataParallelLatentAttentionWithRopeFp8
                else:
                    mla_kwargs["dtype"] = DType.bfloat16
                    mla_cls = DataParallelLatentAttentionWithRope

        self.self_attn = mla_cls(**mla_kwargs)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx)

        self.mlp_shards: list[MLP | MoE]
        if self.mlp.sharding_strategy is not None:
            self.mlp_shards = list(self.mlp.shard(config.devices))
        else:
            self.mlp_shards = [self.mlp]

        # Create normalization layers
        create_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.norm_dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
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

    def _get_mlp(self, config: DeepseekV3Config, layer_idx: int) -> MLP | MoE:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            layer_idx: Layer index

        Returns:
            List of MLP shards or MoE modules depending on the layer index and config
        """
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

            num_phy = (
                config.ep_config.n_experts
                if config.ep_config is not None
                and config.ep_config.eplb_enabled
                else config.n_routed_experts
            )

            moe_kwargs: dict[str, Any] = dict(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=num_phy,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                num_logical_experts=config.n_routed_experts,
                gate_cls=functools.partial(
                    DeepseekV3TopKRouter,
                    routed_scaling_factor=config.routed_scaling_factor,
                    scoring_func=config.scoring_func,
                    topk_method=config.topk_method,
                    n_group=config.n_group,
                    topk_group=config.topk_group,
                    norm_topk_prob=config.norm_topk_prob,
                    gate_dtype=config.gate_dtype or config.norm_dtype,
                    correction_bias_dtype=config.correction_bias_dtype,
                ),
                has_shared_experts=True,
                shared_experts_dim=config.n_shared_experts
                * config.moe_intermediate_size,
                dtype=config.dtype,
                ep_size=ep_size,
                apply_router_weight_first=False,
                ep_batch_manager=self.ep_manager,
                quant_config=config.quant_config,
                shared_experts_dtype=(
                    config.quant_config.shared_experts_dtype(config.dtype)
                    if config.quant_config is not None
                    else DType.bfloat16
                ),
            )

            moe: MoE
            if config.quant_config is not None:
                moe = MoEQuantized(**moe_kwargs)
            else:
                moe = MoE(**moe_kwargs)

            moe.layer_idx = layer_idx

            num_devices = len(config.devices)
            if self.mode == ParallelismMode.TP_TP:
                moe.sharding_strategy = ShardingStrategy.tensor_parallel(
                    num_devices
                )
            elif num_devices > 1:
                moe.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            return moe
        else:
            dense_quant = (
                config.quant_config
                if layer_idx not in config.dense_mlp_layers_without_quant
                else None
            )
            # ``config.dtype`` is the packed-weight / graph encoding dtype
            # (e.g. uint8 for ``float4_e2m1fnx2``). Unquantized dense MLPs use
            # BF16 tensors; :class:`~max.nn.Linear` only switches to uint8 when
            # ``quant_config.is_fp4`` is true.
            mlp_weight_dtype = (
                config.dtype if dense_quant is not None else DType.bfloat16
            )
            if (
                dense_quant is None
                and config.quant_config
                and config.quant_config.embedding_output_dtype
            ):
                mlp_weight_dtype = config.quant_config.embedding_output_dtype
            mlp = MLP(
                dtype=mlp_weight_dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
                quant_config=dense_quant,
            )
            if self.mode == ParallelismMode.TP_TP or (
                self.config.ep_config is not None
                and self.config.ep_config.use_allreduce
            ):
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
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        mla_prefill_metadata_flat: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        eplb_counter_buffers: list[BufferValue] | None = None,
        layer_idx_per_device: list[TensorValue] | None = None,
    ) -> list[TensorValue]:
        num_devices = len(kv_collections)

        # Re-pack flat MLA inputs into MLAPrefillMetadata dataclasses
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        if self.config.graph_mode != "decode":
            assert len(mla_prefill_metadata_flat) == 3 * num_devices
            for i in range(num_devices):
                mla_prefill_metadata.append(
                    MLAPrefillMetadata(
                        buffer_row_offsets=mla_prefill_metadata_flat[3 * i],
                        cache_offsets=mla_prefill_metadata_flat[3 * i + 1],
                        buffer_lengths=mla_prefill_metadata_flat[3 * i + 2],
                    )
                )

        # Apply input layer norm to each shard
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        attn_outs = self.self_attn(
            layer_idx,
            norm_xs,
            signal_buffers,
            kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
            mla_prefill_metadata=mla_prefill_metadata,
        )

        hs = self._post_attention(xs, attn_outs, signal_buffers)

        # Post-attention norm (per-device)
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.config.ep_config is not None:
            if self.ep_manager is not None and ep_inputs is not None:
                self.ep_manager.fetch_buffers(ep_inputs)

        mlp_outs = forward_moe_sharded_layers(
            self.mlp_shards,
            norm_outs,
            eplb_counter_buffers,
            layer_idx_per_device,
        )

        hs = self._post_mlp(hs, mlp_outs, signal_buffers)
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
            case ParallelismMode.TP_EP:
                assert self.config.ep_config is not None
                if self.config.ep_config.use_allreduce:
                    attn_outs = ops.allreduce.sum(attn_outs, signal_buffers)
                    return [
                        x + attn_out
                        for x, attn_out in zip(xs, attn_outs, strict=True)
                    ]
                else:
                    # attn_outs[i] is device i's partial sum (allreduce was
                    # skipped).  Add the residual only on device 0 so it isn't
                    # counted P times after the reduce-scatter.
                    hs = [xs[0] + attn_outs[0], *attn_outs[1:]]
                    return ops.reducescatter.sum(hs, signal_buffers, axis=0)
            case ParallelismMode.DP_EP | ParallelismMode.TP_TP:
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
        """Collective after MoE/MLP to restore the expected hidden-state layout."""
        match self.mode:
            case ParallelismMode.TP_EP:
                assert self.config.ep_config is not None
                if self.config.ep_config.use_allreduce:
                    mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
                    return [
                        h + mlp_out
                        for h, mlp_out in zip(hs, mlp_outs, strict=True)
                    ]
                else:
                    hs = [
                        h + mlp_out
                        for h, mlp_out in zip(hs, mlp_outs, strict=True)
                    ]
                    return ops.allgather(hs, signal_buffers, axis=0)
            case ParallelismMode.TP_TP:
                if len(self.config.devices) > 1:
                    mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
                    hs = [
                        h + mlp_out
                        for h, mlp_out in zip(hs, mlp_outs, strict=True)
                    ]
                    return hs
                return hs
            case ParallelismMode.DP_EP:
                return [
                    h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)
                ]
            case _:
                raise ValueError(f"Unsupported parallelism mode: {self.mode}")


class DeepseekV3(Module):
    """Defines the DeepseekV3 transformer model.

    This is a combination of the DeepseekV3Model and the DeepseekV3ForCausalLM
    classes from the HuggingFace Transformers implementation.
    """

    subgraph_layer_prefix: str = "layers"

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        self.config = config
        num_devices = len(config.devices)
        devices = config.devices

        _validate_parallelism_config(config)

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
            self.rope: RotaryEmbedding = DeepseekYarnRotaryEmbedding(
                config.qk_rope_head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_position_embeddings,
                scaling_params=scaling_params,
            )
        else:
            self.rope = RotaryEmbedding(
                dim=config.qk_rope_head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_position_embeddings,
                head_dim=config.qk_rope_head_dim,
                interleaved=config.rope_interleave,
            )

        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        self.layers = LayerList(
            [
                DeepseekV3DecoderLayer(
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

        if config.use_subgraphs:
            self.subgraph_layer_groups = [
                [
                    i
                    for i in range(
                        config.first_k_dense_replace, config.num_hidden_layers
                    )
                ]
            ]
        else:
            self.subgraph_layer_groups = []
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        eplb_counter_buffers_per_layer: list[list[BufferValue]] | None = None,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        return self._process_hidden_states(
            h,
            signal_buffers,
            kv_collections,
            return_n_logits,
            input_row_offsets,
            host_input_row_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
            eplb_counter_buffers_per_layer,
        )

    def _process_hidden_states(
        self,
        h: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        eplb_counter_buffers_per_layer: list[list[BufferValue]]
        | None = None,  # (num_layers, num_devices)
    ) -> tuple[TensorValue, ...]:
        if not host_input_row_offsets.device == DeviceRef.CPU():
            raise ValueError("input_row_offsets must be located on CPU")
        if not data_parallel_splits.device == DeviceRef.CPU():
            raise ValueError("data_parallel_splits must be located on CPU")

        devices = self.config.devices
        mla_prefill_metadata: list[MLAPrefillMetadata] = []
        # Keep this as explicit per-device `.to()` copies.
        # Broadcasting graph-time constants can hang when chained after
        # runtime-dependent collectives (GEX-3200).
        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        # ``input_row_offsets`` arrives pre-broadcast (per-device list) from
        # the caller. The caller is responsible for producing one copy per
        # device so we do not need a local distributed_broadcast here.
        input_row_offsets_ = list(input_row_offsets)
        all_logits_input_row_offsets = input_row_offsets_[0]

        if self.config.data_parallel_degree > 1:
            # Split batch across devices for data-parallel attention.
            h, input_row_offsets_ = split_batch_replicated(
                devices,
                h,
                input_row_offsets_,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        # Create MLA prefill metadata if not in decode mode
        if self.config.graph_mode != "decode":
            mla_prefill_metadata = self.layers[
                0
            ].self_attn.create_mla_prefill_metadata(  # type: ignore
                input_row_offsets_, kv_collections
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

        # For EAGLE3 mode, capture hidden states
        eagle3_captured: list[list[TensorValue]] = []
        eagle3_capture_ids: set[int] = set()
        if self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS:
            assert self.config.eagle_aux_hidden_state_layer_ids is not None, (
                "EAGLE3 hidden-state capture requires "
                "eagle_aux_hidden_state_layer_ids on the target config. "
                "Ensure the draft HF config's eagle_config is propagated."
            )
            eagle3_capture_ids = set(
                self.config.eagle_aux_hidden_state_layer_ids
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
                mla_prefill_metadata_flat,
                input_row_offsets_,
            ]

            values.append(ep_inputs if ep_inputs is not None else [])
            values.append(
                eplb_counter_buffers_per_layer[idx]
                if eplb_counter_buffers_per_layer is not None
                else []
            )
            if self.config.ep_config is not None and getattr(
                self.config.ep_config, "eplb_enabled", False
            ):
                values.append(
                    [
                        ops.constant(idx, DType.int32, device=d)
                        for d in self.config.devices
                    ]
                )
            else:
                values.append([])

            return values

        def capture_for_eagle3(idx: int, h_out: list[TensorValue]) -> None:
            if idx in eagle3_capture_ids:
                eagle3_captured.append(list(h_out))

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: (
                f"{self.subgraph_layer_prefix}.{i}."
            ),
            subgraph_layer_groups=self.subgraph_layer_groups,
            name_for_subgraph=lambda g: f"dist_transformer_block_{g}",
            on_layer_output=capture_for_eagle3 if eagle3_capture_ids else None,
            initial_hidden_states=h,
        )

        return deepseek_logits_postprocess(
            h=h,
            input_row_offsets=input_row_offsets_,
            all_logits_input_row_offsets=all_logits_input_row_offsets,
            return_n_logits=return_n_logits,
            norm_shards=self.norm_shards,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            devices=devices,
            is_data_parallel_attention=self.config.data_parallel_degree > 1,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.logits_scaling,
            capture_hidden_states=eagle3_captured or None,
        )

    def input_types(
        self,
        kv_params: KVCacheParamInterface,
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

        if self.config.eplb_profile_enabled:
            for _layer_idx in range(self.config.num_hidden_layers):
                for device in self.config.devices:
                    all_input_types.append(
                        BufferType(
                            DType.int64,
                            shape=[self.config.n_routed_experts],
                            device=device,
                        )
                    )
        return tuple(all_input_types)
