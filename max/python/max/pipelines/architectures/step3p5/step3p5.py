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
"""Step-3.5-Flash model implementation.

Features:
- Mixed attention: full + sliding window (per-layer)
- Per-layer RoPE theta and partial rotary factors
- MoE with shared expert, sigmoid router, router bias
- Zero-centered RMSNorm (weight_offset=1)
- Head-wise attention gate (g_proj with sigmoid)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence
from enum import Enum, auto
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
from max.nn.comm.allreduce import Allreduce
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe import MoE, make_concatenated_gated_activation_fn
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)

from .layers.attention import Step3p5Attention
from .layers.moe_gate import Step3p5MoEGate
from .model_config import Step3p5Config


class ParallelismMode(Enum):
    """Parallelism strategies supported by Step-3.5-Flash.

    - ``TP_TP``: tensor-parallel attention + tensor-parallel MoE (no EP).
      Allreduce after attention and after MoE.
    - ``TP_EP``: tensor-parallel attention + expert-parallel MoE.
      Allreduce after attention; EP combine handles post-MoE reduction.
    - ``DP_EP``: data-parallel (replicated) attention + expert-parallel MoE.
      No attention allreduce; EP combine handles post-MoE reduction.
    """

    TP_TP = auto()
    TP_EP = auto()
    DP_EP = auto()


def _select_parallelism_mode(
    config: Step3p5Config, ep_manager: EPBatchManager | None
) -> ParallelismMode:
    """Pick a parallelism mode from ``data_parallel_degree`` and EP state.

    Single-GPU degenerates to TP_TP regardless of flags.
    """
    num_devices = len(config.devices)
    if num_devices <= 1:
        return ParallelismMode.TP_TP
    if ep_manager is not None:
        if config.data_parallel_degree == 1:
            return ParallelismMode.TP_EP
        if config.data_parallel_degree == num_devices:
            return ParallelismMode.DP_EP
        raise ValueError(
            "Step-3.5: data_parallel_degree must be 1 (TP+EP) or "
            f"{num_devices} (DP+EP); got {config.data_parallel_degree}"
        )
    if config.data_parallel_degree != 1:
        raise ValueError(
            "Step-3.5: DP-attention requires --ep-size > 1; got "
            f"data_parallel_degree={config.data_parallel_degree} with no EP."
        )
    return ParallelismMode.TP_TP


class _PartialRotaryEmbedding(Llama3RotaryEmbedding):
    """RoPE that only rotates the first ``rotary_dim`` dimensions.

    Produces a full ``head_dim``-sized ``freqs_cis`` tensor where the
    first ``rotary_dim`` dimensions carry real rotation frequencies and
    the remaining dimensions are identity (cos=1, sin=0) via zero
    frequencies.  This avoids the kernel's interleaved-only partial-RoPE
    constraint.
    """

    rotary_dim: int

    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        head_dim: int,
        rotary_dim: int,
        interleaved: bool,
        scaling_params: Llama3RopeScalingParams | None,
    ) -> None:
        # Construct at full head_dim so freqs_cis has the right width.
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=interleaved,
            scaling_params=scaling_params,
        )
        self.rotary_dim = rotary_dim

    def _compute_inv_freqs(self) -> TensorValue:
        # Compute frequencies for only rotary_dim/2 blocks.
        n = self.rotary_dim
        iota = ops.range(
            0, n, step=2, dtype=DType.float64, device=DeviceRef.CPU()
        )
        active = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        # Apply Llama3 scaling to active freqs BEFORE zero-padding,
        # since the scaling divides by inv_freqs (would be div-by-zero
        # on the padded zeros).
        if self.scaling_params is not None:
            active = self._apply_scaling(active)
        # Zero-pad to full head_dim // 2
        pad_count = (self.head_dim - self.rotary_dim) // 2
        if pad_count > 0:
            zeros = ops.constant(
                [0.0] * pad_count, DType.float32, device=DeviceRef.CPU()
            )
            active = ops.concat((active, zeros))
        return active

    def _apply_scaling(self, inv_freqs: TensorValue) -> TensorValue:
        # Override required: scaling must happen BEFORE zero-padding (the parent
        # applies it inside _compute_inv_freqs which we also override).  The
        # parent's _apply_scaling is not directly reusable because it operates
        # on the full inv_freqs tensor, but here we only have the active
        # (non-padded) portion.
        sp = self.scaling_params
        assert sp is not None
        low_freq_wavelen = sp.orig_max_position / sp.low_freq_factor
        high_freq_wavelen = sp.orig_max_position / sp.high_freq_factor
        wave_len = 2 * math.pi / inv_freqs
        if sp.low_freq_factor != sp.high_freq_factor:
            smooth = (sp.orig_max_position / wave_len - sp.low_freq_factor) / (
                sp.high_freq_factor - sp.low_freq_factor
            )
        else:
            smooth = ops.constant(0, DType.float32, device=DeviceRef.CPU())
        return ops.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            ops.where(
                wave_len > low_freq_wavelen,
                inv_freqs / sp.factor,
                (1 - smooth) * inv_freqs / sp.factor + smooth * inv_freqs,
            ),
        )


class Step3p5TransformerBlock(Module):
    """Transformer block for Step-3.5 with mixed attention and MoE/MLP.

    Dispatches sharding and collectives based on a :class:`ParallelismMode`:

    - ``TP_TP``: TP attention + TP MoE; allreduce after both.
    - ``TP_EP``: TP attention + EP MoE; allreduce after attention; EP
      combine handles the post-MoE reduction.
    - ``DP_EP``: replicated attention (each rank owns its DP batch shard)
      + EP MoE; no attention allreduce.
    """

    def __init__(
        self,
        config: Step3p5Config,
        layer_idx: int,
        cache_layer_idx: int,
        kv_params: KVCacheParams,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        mode: ParallelismMode,
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)
        self.mode = mode
        self.ep_manager = ep_manager

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.is_sliding = is_sliding
        self.is_moe = layer_idx in config.moe_layers

        # Select num heads based on layer type
        if is_sliding:
            num_heads = config.sliding_num_attention_heads
            num_kv_heads = config.sliding_num_attention_groups
        else:
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_attention_groups

        # Create attention layer
        self.self_attn = Step3p5Attention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            layer_idx=cache_layer_idx,
            is_sliding=is_sliding,
            sliding_window=config.sliding_window,
            use_head_wise_attn_gate=config.use_head_wise_attn_gate,
            dtype=config.dtype,
            rope=rope,
            linear_cls=linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            norm_dtype=config.norm_dtype or config.dtype,
            qk_norm_eps=config.rms_norm_eps or 1e-5,
        )
        if mode == ParallelismMode.DP_EP:
            self.self_attn.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx, linear_cls, ep_manager)

        # MoE layers go EP under TP_EP / DP_EP; dense MLPs replicate under
        # DP_EP and TP-shard otherwise. Wrapper auto-replicates share_expert
        # when the routed strategy is ``expert_parallel``.
        self._is_ep_moe_layer = (
            isinstance(self.mlp, Step3p5MoEWithSharedExpert)
            and ep_manager is not None
            and mode in (ParallelismMode.TP_EP, ParallelismMode.DP_EP)
        )
        if self._is_ep_moe_layer:
            self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                num_devices
            )
        elif mode == ParallelismMode.DP_EP:
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
        else:
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Layer norms (zero-centered: weight_offset=1.0)
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

        self.allreduce = Allreduce(num_accelerators=num_devices)

    def _get_mlp(
        self,
        config: Step3p5Config,
        layer_idx: int,
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> MLP | Step3p5MoEWithSharedExpert:
        """Get MLP or MoE layer based on config and layer index."""
        if layer_idx in config.moe_layers:
            swiglu_limit = (
                config.swiglu_limits[layer_idx]
                if layer_idx < len(config.swiglu_limits)
                else 0.0
            )
            swiglu_limit_shared = (
                config.swiglu_limits_shared[layer_idx]
                if layer_idx < len(config.swiglu_limits_shared)
                else 0.0
            )
            return Step3p5MoEWithSharedExpert(
                devices=config.devices,
                config=config,
                linear_cls=linear_cls,
                swiglu_limit=swiglu_limit,
                swiglu_limit_shared=swiglu_limit_shared,
                ep_batch_manager=ep_manager,
            )
        else:
            return MLP(
                config.dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                linear_cls,
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
        # Input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention
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

        # Allreduce attention outputs only for TP attention; DP-replicated
        # attention is independent per rank.
        attn_is_tp = self.mode in (
            ParallelismMode.TP_TP,
            ParallelismMode.TP_EP,
        )
        if attn_is_tp and len(self.devices) > 1:
            attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        if self.ep_manager is not None and ep_inputs is not None:
            self.ep_manager.fetch_buffers(ep_inputs)

        # MLP/MoE dispatch by per-layer sharding (set in __init__):
        # - EP MoE: EP combine handles cross-rank reduction, no allreduce.
        # - Replicated (DP_EP dense MLP): independent per rank, no allreduce.
        # - TP (TP_TP, or TP_EP dense MLP): partial outputs, allreduce.
        if self._is_ep_moe_layer:
            mlp_outs = forward_moe_sharded_layers(self.mlp_shards, norm_outs)
        else:
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
            mlp_is_tp = self.mode != ParallelismMode.DP_EP
            if mlp_is_tp and len(self.devices) > 1:
                mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Step3p5MoEWithSharedExpert(Module):
    """MoE layer with a shared (always-on) expert, using sigmoid routing.

    output = routed_moe(x) + shared_expert(x)

    Supports optional SwiGLU activation clipping (swiglu_limits from the
    paper) to prevent numerical blow-up at long contexts.

    Also supports expert parallelism: when an :class:`EPBatchManager` is
    passed, the wrapper exposes the attributes required by
    :func:`forward_moe_sharded_layers` (``gate``, ``num_experts_per_token``,
    ``ep_batch_manager``, ``_local_ep_compute``, ``has_shared_experts``,
    ``shared_experts``) by forwarding to the underlying :class:`MoE` while
    keeping the shared expert at the wrapper level so it sees the
    layer-specific ``swiglu_limit_shared``.
    """

    def __init__(
        self,
        *,
        devices: list[DeviceRef],
        swiglu_limit: float = 0.0,
        swiglu_limit_shared: float = 0.0,
        config: Step3p5Config | None = None,
        linear_cls: Callable[..., Linear] | None = None,
        is_sharding: bool = False,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.swiglu_limit = swiglu_limit
        self.swiglu_limit_shared = swiglu_limit_shared
        self._ep_batch_manager = ep_batch_manager

        if is_sharding:
            return

        if config is None or linear_cls is None:
            raise ValueError(
                "config and linear_cls are required when is_sharding=False"
            )

        ep_size = 1
        if ep_batch_manager is not None:
            ep_size = (
                ep_batch_manager.config.n_gpus_per_node
                * ep_batch_manager.config.n_nodes
            )

        # Routed MoE
        self.moe = MoE(
            devices=self.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.moe_num_experts,
            num_experts_per_token=config.moe_top_k,
            moe_dim=config.moe_intermediate_size,
            gate_cls=functools.partial(
                Step3p5MoEGate,
                routed_scaling_factor=config.moe_router_scaling_factor,
                norm_topk_prob=config.norm_expert_weight,
            ),
            dtype=config.dtype,
            gated_activation_fn=make_concatenated_gated_activation_fn(
                ops.silu, swiglu_limit
            )
            if swiglu_limit > 0
            else None,
            ep_size=ep_size,
            ep_batch_manager=ep_batch_manager,
        )

        # Shared expert (always-on MLP)
        self.share_expert = MLP(
            config.dtype,
            config.model_quantization_encoding,
            config.hidden_size,
            config.share_expert_dim,
            self.devices,
            linear_cls,
            swiglu_limit=swiglu_limit_shared,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        if self._ep_batch_manager is not None:
            # In EP mode, ``forward_moe_sharded_layers`` drives this layer
            # via the per-shard helpers (``gate``, ``_local_ep_compute``,
            # ``shared_experts``); the wrapper's own ``__call__`` is unused.
            raise ValueError(
                "Use forward_moe_sharded_layers for expert-parallel "
                "inference instead of calling Step3p5MoEWithSharedExpert "
                "directly."
            )
        routed = self.moe(x)
        shared = self.share_expert(x)
        return routed + shared

    # --- EP forwarding to the inner MoE -------------------------------- #
    # ``forward_moe_sharded_layers`` treats each shard as a ``MoE``-shaped
    # object.  The forwarders below let the wrapper stand in for that
    # interface without changing the weight-loading hierarchy
    # (``mlp.moe.*`` / ``mlp.share_expert.*``).

    @property
    def gate(self) -> Callable[[TensorValue], tuple[TensorValue, TensorValue]]:
        return self.moe.gate

    @property
    def num_experts_per_token(self) -> int:
        return self.moe.num_experts_per_token

    @property
    def ep_batch_manager(self) -> EPBatchManager:
        assert self._ep_batch_manager is not None, (
            "EPBatchManager must be provided to use expert-parallel forward"
        )
        return self._ep_batch_manager

    @property
    def has_shared_experts(self) -> bool:
        # The shared expert is always present for Step-3.5; advertise that
        # to the EP forward path so it adds the shared-expert output after
        # the combine step.
        return True

    def shared_experts(self, x: TensorValue) -> TensorValue:
        return self.share_expert(x)

    def _ep_dispatch_input_scales(self) -> TensorValue | None:
        return self.moe._ep_dispatch_input_scales()

    def _local_ep_compute(
        self,
        expert_inputs: tuple[TensorValue, ...],
        x: TensorValue,
        estimated_total_m: TensorValue,
    ) -> TensorValue:
        return self.moe._local_ep_compute(expert_inputs, x, estimated_total_m)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.moe.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.moe.sharding_strategy = strategy
        if strategy.is_expert_parallel:
            # Shared expert is replicated across EP ranks (every rank holds
            # the full share_expert and runs it on its DP-attention shard).
            self.share_expert.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            self.share_expert.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Step3p5MoEWithSharedExpert]:
        devices_list = list(devices)
        moe_shards = self.moe.shard(devices_list)
        share_expert_shards = self.share_expert.shard(devices_list)

        shards: list[Step3p5MoEWithSharedExpert] = []
        for i, device in enumerate(devices_list):
            # Bypass __init__ to avoid creating sub-modules that would
            # be immediately replaced by the already-sharded versions.
            sharded = Step3p5MoEWithSharedExpert(
                devices=[device],
                swiglu_limit=self.swiglu_limit,
                swiglu_limit_shared=self.swiglu_limit_shared,
                is_sharding=True,
                ep_batch_manager=self._ep_batch_manager,
            )
            sharded.moe = moe_shards[i]
            sharded.share_expert = share_expert_shards[i]
            shards.append(sharded)
        return shards


class Step3p5(DistributedLogitsPostprocessMixin, Module):
    """Step-3.5-Flash model.

    Supports single-GPU, multi-GPU TP, and (when ``data_parallel_degree
    > 1`` together with an :class:`EPBatchManager`) DP-attention + EP-MoE
    inference.
    """

    layers: LayerList
    embed_tokens: VocabParallelEmbedding

    def __init__(
        self,
        config: Step3p5Config,
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.mode = _select_parallelism_mode(config, ep_manager)
        self.ep_manager = ep_manager

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Step-3.5 is not implemented yet")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ Step-3.5 is not implemented yet")

        # Per-layer RoPE configuration.
        # Step-3.5 uses per-layer rope_theta and partial rotary factors:
        #   - Full attention: partial_rotary_factor=0.5 (64/128 dims), with
        #     rope_scaling
        #   - SWA: partial_rotary_factor=1.0 (128/128 dims), no rope_scaling
        # We store RoPE objects keyed by (theta, rotary_dim, use_scaling) and
        # a per-layer mapping.  freqs_cis access is deferred to __call__
        # (inside the Graph context) since RoPE computation emits graph ops.
        full_head_dim = config.head_dim
        num_layers = config.num_hidden_layers
        yarn_only = set(config.yarn_only_types)

        per_layer_thetas = (
            config.per_layer_rope_theta or [config.rope_theta] * num_layers
        )
        per_layer_prf = config.partial_rotary_factors or [1.0] * num_layers

        rope_cache: dict[tuple[float, int, bool], Llama3RotaryEmbedding] = {}
        layer_rope_keys: list[tuple[float, int, bool]] = []
        for i in range(num_layers):
            theta = (
                per_layer_thetas[i]
                if i < len(per_layer_thetas)
                else config.rope_theta
            )
            prf = per_layer_prf[i] if i < len(per_layer_prf) else 1.0
            rotary_dim = int(full_head_dim * prf)
            layer_type = config.layer_types[i]
            use_scaling = layer_type in yarn_only
            key = (theta, rotary_dim, use_scaling)
            if key not in rope_cache:
                if rotary_dim < full_head_dim:
                    rope_cache[key] = _PartialRotaryEmbedding(
                        dim=config.hidden_size,
                        n_heads=config.num_attention_heads,
                        theta=theta,
                        max_seq_len=config.max_seq_len,
                        head_dim=full_head_dim,
                        rotary_dim=rotary_dim,
                        interleaved=config.interleaved_rope_weights,
                        scaling_params=(
                            config.rope_scaling_params if use_scaling else None
                        ),
                    )
                else:
                    rope_cache[key] = Llama3RotaryEmbedding(
                        dim=config.hidden_size,
                        n_heads=config.num_attention_heads,
                        theta=theta,
                        max_seq_len=config.max_seq_len,
                        head_dim=full_head_dim,
                        interleaved=config.interleaved_rope_weights,
                        scaling_params=(
                            config.rope_scaling_params if use_scaling else None
                        ),
                    )
            layer_rope_keys.append(key)

        self._rope_cache = rope_cache
        self._layer_rope_keys = layer_rope_keys
        # Keep a rope reference for the interleaved flag used by attention.
        first_key = layer_rope_keys[0]
        self.rope = rope_cache[first_key]

        # Zero-centered RMSNorm factory (weight_offset=1.0)
        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "Step-3.5 requires RMSNorm. Set norm_method='rms_norm' and "
                "provide rms_norm_eps."
            )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        # Transformer layers (all share self.rope for the interleaved flag;
        # actual per-layer freqs_cis is passed at call time via _layer_freqs).
        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_type: dict[str, KVCacheParams] = {}
        for layer_type_key, kv_params_leaf in config.kv_params.children.items():
            assert isinstance(kv_params_leaf, KVCacheParams)
            kv_params_by_type[layer_type_key] = kv_params_leaf

        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            cache_layer_idx = layer_type_counts[layer_type]
            layer_type_counts[layer_type] += 1
            layers.append(
                Step3p5TransformerBlock(
                    config=config,
                    layer_idx=i,
                    cache_layer_idx=cache_layer_idx,
                    kv_params=kv_params_by_type[layer_type],
                    rope=self.rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    mode=self.mode,
                    ep_manager=ep_manager,
                )
            )
        self.layers = LayerList(layers)

        self.subgraph_layer_groups: list[list[int]] = []
        if config.use_subgraphs:
            groups: dict[tuple[bool, bool], list[int]] = {}
            for i, layer in enumerate(self.layers):
                assert isinstance(layer, Step3p5TransformerBlock)
                group_key = (layer.is_sliding, layer.is_moe)
                groups.setdefault(group_key, []).append(i)
            for indices in groups.values():
                if len(indices) > 1:
                    self.subgraph_layer_groups.append(indices)

        # Final norm (zero-centered)
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
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        sliding_kv_collections: list[PagedCacheValues],
        global_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        host_input_row_offsets: TensorValue | None = None,
        data_parallel_splits: TensorValue | None = None,
        ep_inputs: list[Value[Any]] | None = None,
    ) -> tuple[TensorValue, ...]:
        # Embeddings
        h = self.embed_tokens(tokens, signal_buffers)

        # Build per-layer freqs_cis inside the graph context (deferred
        # from __init__ because RoPE emits graph ops).
        _freqs_by_key: dict[tuple[float, int, bool], list[TensorValue]] = {}
        per_layer_freqs: list[list[TensorValue]] = []
        for key in self._layer_rope_keys:
            if key not in _freqs_by_key:
                fc = self._rope_cache[key].freqs_cis
                _freqs_by_key[key] = [fc.to(device) for device in self.devices]
            per_layer_freqs.append(_freqs_by_key[key])

        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # In DP mode, split the (replicated) batch across devices so each
        # GPU only processes its own shard.
        is_dp_ep = self.mode == ParallelismMode.DP_EP
        if is_dp_ep and data_parallel_splits is not None:
            assert host_input_row_offsets is not None
            h, input_row_offsets_list = split_batch_replicated(
                self.devices,
                h,
                input_row_offsets_list,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
            )

        def inputs_for_layer(
            idx: int, hs: list[TensorValue]
        ) -> list[SubgraphInput]:
            layer = self.layers[idx]
            assert isinstance(layer, Step3p5TransformerBlock)
            kv_collections = (
                sliding_kv_collections
                if layer.is_sliding
                else global_kv_collections
            )
            values: list[SubgraphInput] = [
                ops.constant(
                    layer.self_attn.layer_idx,
                    DType.uint32,
                    device=DeviceRef.CPU(),
                ),
                hs,
                signal_buffers,
                kv_collections,
                per_layer_freqs[idx],
                input_row_offsets_list,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            return values

        def name_for_subgraph(group_idx: int) -> str:
            group = self.subgraph_layer_groups[group_idx]
            first = self.layers[group[0]]
            assert isinstance(first, Step3p5TransformerBlock)
            attn = "sliding" if first.is_sliding else "full"
            mlp = "moe" if first.is_moe else "mlp"
            return f"step3p5_{attn}_{mlp}_block"

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=self.subgraph_layer_groups or None,
            name_for_subgraph=name_for_subgraph,
            initial_hidden_states=h,
        )

        if is_dp_ep:
            return self._dp_logits_postprocess(
                h,
                input_row_offsets_list,
                norm_shards=self.norm_shards,
                lm_head=self.lm_head,
                signal_buffers=signal_buffers,
                devices=self.devices,
            )

        return self._postprocess_logits(
            h, input_row_offsets_list, return_n_logits, signal_buffers
        )

    @staticmethod
    def _dp_logits_postprocess(
        h: list[TensorValue],
        input_row_offsets: list[TensorValue],
        norm_shards: Sequence[Callable[[TensorValue], TensorValue]],
        lm_head: Callable[
            [list[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
        ],
        signal_buffers: list[BufferValue],
        devices: list[DeviceRef],
    ) -> tuple[TensorValue, ...]:
        """Logits post-processing for DP mode.

        Each device holds the hidden states for its own batch shard. We
        gather the last-token hidden state from every device, normalize on
        each device, and run the vocab-parallel LM head on the gathered
        tensor.
        """
        last_token_per_dev: list[TensorValue] = []
        for dev_idx in range(len(devices)):
            last_token_indices = input_row_offsets[dev_idx][1:] - 1
            last_token_h = ops.gather(h[dev_idx], last_token_indices, axis=0)
            last_token_per_dev.append(last_token_h)

        last_token_distributed = ops.allgather(
            last_token_per_dev, signal_buffers
        )

        norm_last_token = forward_sharded_layers(
            norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            lm_head(norm_last_token, signal_buffers)[0],
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

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]

        # DP mode needs host-side row offsets and per-device batch splits
        # to do the data-parallel batch slice.  TP+EP keeps the TP_TP layout.
        if self.mode == ParallelismMode.DP_EP:
            host_input_row_offsets_type = TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            )
            data_parallel_splits_type = TensorType(
                DType.int64,
                shape=[self.num_devices + 1],
                device=DeviceRef.CPU(),
            )
            base_inputs.extend(
                [host_input_row_offsets_type, data_parallel_splits_type]
            )

        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()

        flattened_kv_types = kv_inputs.flatten()

        # EP communication buffers are appended at the very end so the
        # graph-builder can split off ``len(ep_manager.input_types())``
        # tail inputs.
        ep_input_types: list[TensorType | BufferType] = []
        if self.ep_manager is not None:
            ep_input_types = list(self.ep_manager.input_types())

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + ep_input_types
        )
