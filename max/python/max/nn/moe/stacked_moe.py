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
"""Provides a consolidated Mixture of Experts layer with stacked weights.

This module provides a unified MoE implementation that consolidates patterns
from multiple architectures (Llama4, Qwen3VL, GptOss) into a single base layer
with configurable components.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.graph.weight import _compute_shard_range
from max.support.math import ceildiv
from typing_extensions import Self

from ..kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
)
from ..layer import Module, Shardable
from ..linear import MLP
from ..quant_config import QuantConfig, QuantFormat
from ..quant_ops import quantized_grouped_matmul
from .moe import MoEGate


@dataclass
class RoutingInfo:
    """Intermediate routing tensors for MoE computation."""

    token_expert_order: TensorValue
    """The indices that sort tokens into expert-processing order."""

    expert_start_indices: TensorValue
    """The starting index of each expert's token group."""

    restore_token_order: TensorValue
    """The indices that restore tokens to their original order."""

    expert_ids: TensorValue
    """The expert identifier for each token group."""

    expert_usage_stats: TensorValue
    """The usage statistics for each expert."""

    router_idx_flat: TensorValue
    """The flattened router indices for each token."""


class GateUpFormat(Enum):
    """Specifies the format of the combined gate/up projection weights."""

    CONCATENATED = "concatenated"
    """Gate and up projections concatenated as ``[gate | up]``.

    Stored as ``[num_experts, hidden_dim, 2 * moe_dim]``.
    Split at ``moe_dim``: ``gate = output[:, :moe_dim]``,
    ``up = output[:, moe_dim:]``. Used by Llama4 and Qwen3VL.
    """

    INTERLEAVED = "interleaved"
    """Gate and up projections interleaved as ``[g0, u0, g1, u1, ...]``.

    Stored as ``[num_experts, hidden_dim, 2 * moe_dim]``.
    Split with stride: ``gate = output[:, 0::2]``,
    ``up = output[:, 1::2]``. Used by GptOss.
    """


def make_stacked_gated_activation_fn(
    activation_fn: Callable[[TensorValue], TensorValue],
) -> Callable[[TensorValue, TensorValue], TensorValue]:
    """Builds a gated activation for split ``(gate, up)`` projections.

    The returned callable applies ``activation_fn`` to the gate tensor
    and multiplies with the up tensor:
    ``activation_fn(gate) * up``.

    Args:
        activation_fn: Pointwise activation to apply to the gate tensor.

    Returns:
        A callable ``(gate, up) -> activated``.
    """

    def _stacked_gated_activation_fn(
        gate: TensorValue, up: TensorValue
    ) -> TensorValue:
        return activation_fn(gate) * up

    return _stacked_gated_activation_fn


def _gate_up_scale_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    moe_dim: int,
    block_size: int,
    axis: int = 2,
) -> TensorValue:
    """Shards a combined gate/up projection scale tensor.

    This strategy properly maps weight shard indices to scale indices,
    accounting for the block size used in FP8 quantization.

    Args:
        weight: The scale weight tensor to shard.
        i: The shard index.
        num_devices: The total number of devices.
        moe_dim: The intermediate dimension of each expert.
        block_size: The block size used for FP8 quantization scaling.
        axis: The axis along which to shard. Defaults to ``2``.

    Returns:
        The concatenated gate and up scale shards for this device.
    """
    weight_start, weight_end = _compute_shard_range(moe_dim, i, num_devices)

    scale_gate_start = weight_start // block_size
    scale_gate_end = ceildiv(weight_end, block_size)

    scale_up_start = (moe_dim + weight_start) // block_size
    scale_up_end = ceildiv(moe_dim + weight_end, block_size)

    rank = len(weight.shape)
    if axis < 0:
        axis += rank

    gate_slices = [slice(None)] * rank
    gate_slices[axis] = slice(scale_gate_start, scale_gate_end)

    up_slices = [slice(None)] * rank
    up_slices[axis] = slice(scale_up_start, scale_up_end)

    sharded_gate_scale = weight[tuple(gate_slices)]
    sharded_up_scale = weight[tuple(up_slices)]

    return ops.concat((sharded_gate_scale, sharded_up_scale), axis=axis)


def _down_proj_scale_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    moe_dim: int,
    block_size: int,
    axis: int = 1,
) -> TensorValue:
    """Shards a down projection scale tensor along the given axis.

    This strategy properly maps weight shard indices to scale indices,
    accounting for the block size used in FP8 quantization.

    Args:
        weight: The scale weight tensor to shard.
        i: The shard index.
        num_devices: The total number of devices.
        moe_dim: The intermediate dimension of each expert.
        block_size: The block size used for FP8 quantization scaling.
        axis: The axis along which to shard. Defaults to ``1``.

    Returns:
        The down projection scale shard for this device.
    """
    weight_start, weight_end = _compute_shard_range(moe_dim, i, num_devices)

    scale_start = weight_start // block_size
    scale_end = ceildiv(weight_end, block_size)

    rank = len(weight.shape)
    if axis < 0:
        axis += rank

    slices = [slice(None)] * rank
    slices[axis] = slice(scale_start, scale_end)

    return weight[tuple(slices)]


class StackedMoE(Module, Shardable):
    """Stacked Mixture of Experts layer with configurable components.

    This class consolidates MoE implementations from multiple architectures
    (Llama4, Qwen3VL, GptOss) into a single base layer. All expert weights
    are stored in stacked format rather than as individual MLP experts.

    Weight tensor shapes:

    - ``gate_up_proj``: ``[num_experts, hidden_dim, 2 * moe_dim]``
    - ``down_proj``: ``[num_experts, moe_dim, hidden_dim]``
    - Optional FP8 scales: ``[num_experts, scaled_rows, scaled_cols]``

    Supported configurations:

    - Gate/up formats: concatenated or interleaved.
    - Activation functions: configurable (default: SiLU).
    - Optional bias support for projections.
    - Optional FP8 quantization with block scaling.
    - Optional shared experts.

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.graph import DeviceRef
        from max.nn.moe import MoEGate, StackedMoE

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        # Basic usage (Llama4/Qwen3VL style)
        moe = StackedMoE(
            devices=[device_ref],
            hidden_dim=4096,
            num_experts=8,
            num_experts_per_token=2,
            moe_dim=14336,
            gate_cls=MoEGate,
        )

    Pass ``gate_up_format=GateUpFormat.INTERLEAVED``, a custom
    ``gated_activation_fn``, ``has_bias=True``, or a ``quant_config`` to match
    other architectures (for example, GptOss style with interleaved format and
    bias, or FP8 quantization).

    Args:
        devices: A list of devices to use for the MoE.
        hidden_dim: The dimension of the hidden state.
        num_experts: The total number of experts.
        num_experts_per_token: The number of experts per token (top-k).
        moe_dim: The intermediate dimension of each expert.
        gate_cls: The model-specific gate implementation class.
        dtype: The data type of the MoE weights. Defaults to
            ``DType.bfloat16``.
        gate_up_format: The format of the combined gate/up weights. Defaults
            to ``GateUpFormat.CONCATENATED``.
        gated_activation_fn: Activation applied to the split
            ``(gate, up)`` projections. ``None`` (default) uses SiLU
            gating; use :func:`make_stacked_gated_activation_fn` for
            custom activations.
        has_bias: Whether to include bias for projections. Defaults to
            ``False``.
        has_shared_experts: Whether to use shared experts. Defaults to
            ``False``.
        shared_experts_dim: The dimension of the shared experts. Defaults to
            ``0``.
        quant_config: The configuration for scaled quantization. Defaults to
            ``None``.
        apply_router_weight_first: Whether to apply router weights before
            expert computation. Defaults to ``False``.
        is_sharding: Whether this instance is being created for sharding.
            Set by :meth:`shard()` to skip weight initialization for sharded
            instances. Defaults to ``False``.
    """

    _sharding_strategy: ShardingStrategy | None = None

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate],
        dtype: DType = DType.bfloat16,
        gate_up_format: GateUpFormat = GateUpFormat.CONCATENATED,
        gated_activation_fn: Callable[[TensorValue, TensorValue], TensorValue]
        | None = None,
        has_bias: bool = False,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        quant_config: QuantConfig | None = None,
        apply_router_weight_first: bool = False,
        is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self.gate_cls = gate_cls
        self.dtype = dtype
        self.gate_up_format = gate_up_format
        self.gated_activation_fn = (
            gated_activation_fn or make_stacked_gated_activation_fn(ops.silu)
        )
        self.has_bias = has_bias
        self.has_shared_experts = has_shared_experts
        self.shared_experts_dim = shared_experts_dim
        self.quant_config = quant_config
        self.apply_router_weight_first = apply_router_weight_first
        self.tp_size = 1

        self.gate = gate_cls(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=DType.bfloat16,
        )

        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            self.shared_experts = MLP(
                dtype=dtype,
                quantization_encoding=None,
                hidden_dim=hidden_dim,
                feed_forward_length=shared_experts_dim,
                devices=devices,
                quant_config=quant_config,
            )

        if not is_sharding:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initializes stacked weight tensors for all experts."""
        if self.quant_config and self.quant_config.format == QuantFormat.MXFP4:
            self._init_mxfp4_weights()
        else:
            self._gate_up_weight = Weight(
                name="experts.gate_up_proj",
                shape=[self.num_experts, self.hidden_dim, 2 * self.moe_dim],
                dtype=self.dtype,
                device=self.devices[0],
            )
            self._down_weight = Weight(
                name="experts.down_proj",
                shape=[self.num_experts, self.moe_dim, self.hidden_dim],
                dtype=self.dtype,
                device=self.devices[0],
            )

        if self.has_bias:
            self._gate_up_bias = Weight(
                name="experts.gate_up_proj_bias",
                shape=[self.num_experts, 2 * self.moe_dim],
                dtype=self.dtype,
                device=self.devices[0],
            )
            self._down_bias = Weight(
                name="experts.down_proj_bias",
                shape=[self.num_experts, self.hidden_dim],
                dtype=self.dtype,
                device=self.devices[0],
            )

        # FP8 scales (only for non-MXFP4 float8)
        if self.quant_config and self.quant_config.format != QuantFormat.MXFP4:
            block_size = self.quant_config.weight_scale.block_size
            if self.quant_config.weight_scale.is_rowwise:
                # Per-output-channel (rowwise) scales, e.g. compressed-tensors
                # FP8-dynamic: one scale per expert output channel. Stored in
                # the same [E, out_features, 1] layout the matmul transposes to
                # a per-N scale. ``gate_up`` output is the concatenated
                # 2 * moe_dim; ``down`` output is hidden_dim.
                gate_up_scale_shape = [self.num_experts, 2 * self.moe_dim, 1]
                down_scale_shape = [self.num_experts, self.hidden_dim, 1]
            else:
                assert block_size is not None, "FP8 MoE requires block scaling"

                gate_up_scale_shape = [
                    self.num_experts,
                    ceildiv(self.hidden_dim, block_size[0]),
                    ceildiv(2 * self.moe_dim, block_size[1]),
                ]
                down_scale_shape = [
                    self.num_experts,
                    ceildiv(self.moe_dim, block_size[0]),
                    ceildiv(self.hidden_dim, block_size[1]),
                ]

            self._gate_up_scale = Weight(
                name="experts.gate_up_proj_scale",
                shape=gate_up_scale_shape,
                dtype=self.quant_config.weight_scale.dtype,
                device=self.devices[0],
            )
            self._down_scale = Weight(
                name="experts.down_proj_scale",
                shape=down_scale_shape,
                dtype=self.quant_config.weight_scale.dtype,
                device=self.devices[0],
            )

    def _init_mxfp4_weights(self) -> None:
        """Initializes MXFP4 packed weight tensors for all experts.

        MXFP4 weights are stored as [E, out_features, in_features//2] uint8
        with scales [E, out_features, in_features//32] float8_e8m0fnu.
        """
        assert self.quant_config is not None
        # gate_up: maps hidden_dim -> 2*moe_dim
        self._gate_up_weight = Weight(
            name="experts.gate_up_proj",
            shape=[
                self.num_experts,
                2 * self.moe_dim,
                self.hidden_dim // 2,
            ],
            dtype=DType.uint8,
            device=self.devices[0],
        )
        # down: maps moe_dim -> hidden_dim
        self._down_weight = Weight(
            name="experts.down_proj",
            shape=[
                self.num_experts,
                self.hidden_dim,
                self.moe_dim // 2,
            ],
            dtype=DType.uint8,
            device=self.devices[0],
        )

        # MXFP4 scales: [E, out_features, in_features//32]
        scale_dtype = self.quant_config.weight_scale.dtype
        self._gate_up_scale = Weight(
            name="experts.gate_up_proj_scale",
            shape=[
                self.num_experts,
                2 * self.moe_dim,
                ceildiv(self.hidden_dim, 32),
            ],
            dtype=scale_dtype,
            device=self.devices[0],
        )
        self._down_scale = Weight(
            name="experts.down_proj_scale",
            shape=[
                self.num_experts,
                self.hidden_dim,
                ceildiv(self.moe_dim, 32),
            ],
            dtype=scale_dtype,
            device=self.devices[0],
        )

    @property
    def gate_up_proj_transposed(self) -> TensorValue:
        """The gate/up weights transposed to ``[num_experts, out_features, in_features]`` layout."""
        return self._gate_up_weight.transpose(1, 2)

    @property
    def down_proj_transposed(self) -> TensorValue:
        """The down weights transposed to ``[num_experts, out_features, in_features]`` layout."""
        return self._down_weight.transpose(1, 2)

    def _split_gate_up(
        self, gate_up_output: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Splits the combined gate/up output based on the configured format.

        Args:
            gate_up_output: The combined output of shape
                ``[tokens, 2 * moe_dim]``.

        Returns:
            A tuple of ``(gate, up)`` tensors, each of shape
            ``[tokens, moe_dim]``.
        """
        if self.gate_up_format == GateUpFormat.CONCATENATED:
            gate = gate_up_output[:, : self.moe_dim]
            up = gate_up_output[:, self.moe_dim :]
        else:
            gate = gate_up_output[:, 0::2]
            up = gate_up_output[:, 1::2]
        return gate, up

    def _apply_bias(
        self,
        output: TensorValue,
        bias_weight: TensorValue,
        expert_assignments: TensorValue,
    ) -> TensorValue:
        """Applies expert-specific bias to the output.

        Args:
            output: The matmul output tensor.
            bias_weight: The stacked bias tensor of shape
                ``[num_experts, out_dim]``.
            expert_assignments: The expert indices for each token.

        Returns:
            The output with expert-specific bias added.
        """
        bias_per_token = ops.gather(bias_weight, expert_assignments, axis=0)
        return output + bias_per_token

    def _apply_gated_activation(
        self,
        gate_up_output: TensorValue,
        routing: RoutingInfo,
    ) -> TensorValue:
        """Applies bias (if present), splits gate/up, and computes gated activation.

        Args:
            gate_up_output: The combined gate/up projection output.
            routing: The routing information for expert assignments.

        Returns:
            The activated output tensor.

        Raises:
            ValueError: If ``gate_up_output`` is not BF16.
        """
        if self.has_bias:
            expert_assignments = ops.gather(
                routing.router_idx_flat, routing.token_expert_order, axis=0
            )
            gate_up_output = self._apply_bias(
                gate_up_output, self._gate_up_bias, expert_assignments
            )

        if gate_up_output.dtype != DType.bfloat16:
            raise ValueError("Gate+Up output must be BF16 for activation")

        gate, up = self._split_gate_up(gate_up_output)
        return self.gated_activation_fn(gate, up)

    def _prepare_routing(self, router_idx: TensorValue) -> RoutingInfo:
        """Computes token-to-expert routing indices.

        Args:
            router_idx: The router index tensor from the gate.

        Returns:
            A ``RoutingInfo`` containing all routing tensors.
        """
        router_idx_flat = ops.reshape(router_idx, [-1])
        router_idx_int32 = ops.cast(router_idx_flat, DType.int32)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_int32, self.num_experts)

        return RoutingInfo(
            token_expert_order=token_expert_order,
            expert_start_indices=expert_start_indices,
            restore_token_order=restore_token_order,
            expert_ids=expert_ids,
            expert_usage_stats=expert_usage_stats,
            router_idx_flat=router_idx_flat,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies the stacked MoE layer.

        Args:
            x: The input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            The output tensor of shape ``(seq_len, hidden_dim)``.
        """
        seq_len = x.shape[0]

        # Route tokens to experts
        router_idx, router_weight = self.gate(x)
        routing = self._prepare_routing(router_idx)

        # Gather tokens in expert-processing order
        token_indices = ops.cast(
            routing.token_expert_order // self.num_experts_per_token,
            DType.int32,
        )
        permuted_states = ops.gather(x, token_indices, axis=0)

        # Optionally apply router weights before expert computation
        if self.apply_router_weight_first:
            permuted_states = permuted_states * ops.gather(
                router_weight.reshape([-1, 1]),
                routing.token_expert_order,
                axis=0,
            ).cast(x.dtype)

        # Run expert computation (quantized or BF16 path)
        if self.quant_config:
            down_projs = self._forward_quantized(permuted_states, routing)
        else:
            down_projs = self._forward_bf16(permuted_states, routing)

        # Restore original token order and combine expert outputs
        down_projs = ops.gather(
            down_projs, routing.restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, self.hidden_dim])

        if not self.apply_router_weight_first:
            routed_expert_out = (
                ops.unsqueeze(router_weight, axis=1) @ down_projs
            )
            routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = ops.squeeze(
                ops.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out

    def _forward_bf16(
        self,
        permuted_states: TensorValue,
        routing: RoutingInfo,
    ) -> TensorValue:
        """Runs the BF16 forward pass through the expert projections.

        Args:
            permuted_states: The input states reordered by expert assignment.
            routing: The routing information for expert assignments.

        Returns:
            The down-projected output tensor.
        """
        gate_up_output = grouped_matmul_ragged(
            permuted_states,
            self.gate_up_proj_transposed,
            routing.expert_start_indices,
            routing.expert_ids,
            routing.expert_usage_stats,
        )

        gated_output = self._apply_gated_activation(gate_up_output, routing)

        down_output = grouped_matmul_ragged(
            gated_output,
            self.down_proj_transposed,
            routing.expert_start_indices,
            routing.expert_ids,
            routing.expert_usage_stats,
        )

        if self.has_bias:
            expert_assignments = ops.gather(
                routing.router_idx_flat, routing.token_expert_order, axis=0
            )
            down_bias: TensorValue = self._down_bias
            if self.tp_size > 1:
                down_bias = down_bias / self.tp_size
            down_output = self._apply_bias(
                down_output, down_bias, expert_assignments
            )

        return down_output

    def _forward_quantized(
        self,
        permuted_states: TensorValue,
        routing: RoutingInfo,
    ) -> TensorValue:
        """Runs the quantized forward pass (MXFP4 or FP8).

        Delegates to ``quantized_grouped_matmul`` which dispatches to the
        appropriate kernel based on ``quant_config.format``.

        Args:
            permuted_states: The input states reordered by expert assignment.
            routing: The routing information for expert assignments.

        Returns:
            The down-projected output tensor.
        """
        assert self.quant_config is not None

        gate_up_output = quantized_grouped_matmul(
            x=permuted_states,
            weight=self._gate_up_weight,
            weight_scale=self._gate_up_scale,
            expert_start_indices=routing.expert_start_indices,
            expert_ids=routing.expert_ids,
            usage_stats=routing.expert_usage_stats,
            quant_config=self.quant_config,
        )

        gated_output = self._apply_gated_activation(gate_up_output, routing)

        down_output = quantized_grouped_matmul(
            x=gated_output,
            weight=self._down_weight,
            weight_scale=self._down_scale,
            expert_start_indices=routing.expert_start_indices,
            expert_ids=routing.expert_ids,
            usage_stats=routing.expert_usage_stats,
            quant_config=self.quant_config,
        )

        if self.has_bias:
            expert_assignments = ops.gather(
                routing.router_idx_flat, routing.token_expert_order, axis=0
            )
            down_bias: TensorValue = self._down_bias
            if self.tp_size > 1:
                down_bias = down_bias / self.tp_size
            down_output = self._apply_bias(
                down_output, down_bias, expert_assignments
            )

        return down_output

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """The sharding strategy for this module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy and configures sharding for all sub-components.

        Args:
            strategy: The tensor-parallel sharding strategy to apply.

        Raises:
            ValueError: If ``strategy`` is not tensor-parallel.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for StackedMoE"
            )

        self._sharding_strategy = strategy
        self._set_gate_sharding(strategy)
        self._set_weight_sharding(strategy)

        if self.has_shared_experts:
            self.shared_experts.sharding_strategy = strategy
        if self.has_bias:
            self._set_bias_sharding(strategy)
        if self.quant_config:
            self._set_scale_sharding(strategy)

    def _set_gate_sharding(self, strategy: ShardingStrategy) -> None:
        """Configures sharding for the gate module."""
        self.gate.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )

    def _set_weight_sharding(self, strategy: ShardingStrategy) -> None:
        """Configures sharding for weight tensors."""
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for StackedMoE"
            )

        if self.quant_config and self.quant_config.format == QuantFormat.MXFP4:
            # MXFP4 weights are [E, out_features, in_features//2] (transposed
            # vs BF16's [E, in_features, out_features]).  TP splits moe_dim:
            # gate_up shards output dim axis=1 (2*moe_dim), down shards
            # input dim axis=2 (moe_dim//2 packed).
            #
            # Plain axiswise sharding works even for INTERLEAVED gate_up
            # format because the rows already alternate gate/up in the
            # checkpoint, so splitting axis 1 into N equal parts naturally
            # keeps each shard balanced.
            self._gate_up_weight.sharding_strategy = ShardingStrategy.axiswise(
                axis=1, num_devices=strategy.num_devices
            )
            self._down_weight.sharding_strategy = ShardingStrategy.axiswise(
                axis=2, num_devices=strategy.num_devices
            )
            return

        if self.gate_up_format == GateUpFormat.CONCATENATED:
            gate_up_strategy = ShardingStrategy.gate_up(strategy.num_devices)
        else:
            gate_up_strategy = ShardingStrategy.axiswise(
                axis=2, num_devices=strategy.num_devices
            )

        self._gate_up_weight.sharding_strategy = gate_up_strategy
        self._down_weight.sharding_strategy = ShardingStrategy.axiswise(
            axis=1, num_devices=strategy.num_devices
        )

    def _set_bias_sharding(self, strategy: ShardingStrategy) -> None:
        """Configures sharding for bias tensors."""
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for StackedMoE"
            )

        if self.gate_up_format == GateUpFormat.CONCATENATED:
            gate_up_bias_strategy = ShardingStrategy.gate_up(
                strategy.num_devices
            )
        else:
            gate_up_bias_strategy = ShardingStrategy.axiswise(
                axis=1, num_devices=strategy.num_devices
            )

        self._gate_up_bias.sharding_strategy = gate_up_bias_strategy
        self._down_bias.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )

    def _set_scale_sharding(self, strategy: ShardingStrategy) -> None:
        """Configures sharding for FP8 scale tensors."""
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for StackedMoE"
            )

        assert self.quant_config is not None
        block_size = self.quant_config.weight_scale.block_size
        assert block_size is not None

        if self.quant_config.format == QuantFormat.MXFP4:
            # MXFP4 scale sharding mirrors the weight sharding axes.
            # Weights are [E, out_features, in_features//2] so scales are
            # [E, out_features, ceildiv(in_features, 32)]:
            # - gate_up_scale [E, 2*moe_dim, ceildiv(hidden, 32)]: axis 1
            #   (output dim, matching gate_up_weight axis 1)
            # - down_scale [E, hidden, ceildiv(moe_dim, 32)]: axis 2
            #   (input dim, matching down_weight axis 2)
            self._gate_up_scale.sharding_strategy = ShardingStrategy.axiswise(
                axis=1, num_devices=strategy.num_devices
            )
            self._down_scale.sharding_strategy = ShardingStrategy.axiswise(
                axis=2, num_devices=strategy.num_devices
            )
        else:
            gate_up_scale_shard_fn = partial(
                _gate_up_scale_sharding_strategy,
                moe_dim=self.moe_dim,
                block_size=block_size[1],
                axis=2,
            )
            self._gate_up_scale.sharding_strategy = ShardingStrategy(
                num_devices=strategy.num_devices,
                shard=gate_up_scale_shard_fn,
            )

            down_proj_scale_shard_fn = partial(
                _down_proj_scale_sharding_strategy,
                moe_dim=self.moe_dim,
                block_size=block_size[0],
                axis=1,
            )
            self._down_scale.sharding_strategy = ShardingStrategy(
                num_devices=strategy.num_devices,
                shard=down_proj_scale_shard_fn,
            )

    def _create_sharded_instance(
        self, device: DeviceRef, sharded_moe_dim: int, sharded_shared_dim: int
    ) -> Self:
        """Creates a sharded instance of this module.

        Subclasses can override this to use config-based initialization.

        Args:
            device: The device to place the shard on.
            sharded_moe_dim: The sharded ``moe_dim`` for this instance.
            sharded_shared_dim: The sharded ``shared_experts_dim`` for this
                instance.

        Returns:
            A new instance configured for sharding, without weights assigned.
        """
        return self.__class__(
            devices=[device],
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token,
            moe_dim=sharded_moe_dim,
            gate_cls=self.gate_cls,
            dtype=self.dtype,
            gate_up_format=self.gate_up_format,
            gated_activation_fn=self.gated_activation_fn,
            has_bias=self.has_bias,
            has_shared_experts=self.has_shared_experts,
            shared_experts_dim=sharded_shared_dim,
            quant_config=self.quant_config,
            apply_router_weight_first=self.apply_router_weight_first,
            is_sharding=True,
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Self]:
        """Creates sharded views of this MoE module across multiple devices.

        Args:
            devices: The devices to place the shards on.

        Returns:
            A sequence of sharded instances, one for each device.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "StackedMoE cannot be sharded without a sharding strategy."
            )

        gate_score_shards = self.gate.gate_score.shard(devices)
        gate_up_shards = self._gate_up_weight.shard(devices)
        down_shards = self._down_weight.shard(devices)

        if self.has_shared_experts:
            shared_experts_shards = self.shared_experts.shard(devices)

        if self.has_bias:
            gate_up_bias_shards = self._gate_up_bias.shard(devices)
            down_bias_shards = self._down_bias.shard(devices)

        if self.quant_config:
            gate_up_scale_shards = self._gate_up_scale.shard(devices)
            down_scale_shards = self._down_scale.shard(devices)

        shards = []
        num_devices = self._sharding_strategy.num_devices
        sharded_moe_dim = self.moe_dim // num_devices
        sharded_shared_dim = self.shared_experts_dim // num_devices

        for shard_idx, device in enumerate(devices):
            sharded = self._create_sharded_instance(
                device, sharded_moe_dim, sharded_shared_dim
            )

            sharded.tp_size = num_devices
            sharded.gate.gate_score = gate_score_shards[shard_idx]
            sharded._gate_up_weight = gate_up_shards[shard_idx]
            sharded._down_weight = down_shards[shard_idx]

            if self.has_shared_experts:
                sharded.shared_experts = shared_experts_shards[shard_idx]

            if self.has_bias:
                sharded._gate_up_bias = gate_up_bias_shards[shard_idx]
                sharded._down_bias = down_bias_shards[shard_idx]

            if self.quant_config:
                sharded._gate_up_scale = gate_up_scale_shards[shard_idx]
                sharded._down_scale = down_scale_shards[shard_idx]

            shards.append(sharded)

        return shards
