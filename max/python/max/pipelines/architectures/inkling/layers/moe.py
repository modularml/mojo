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
"""Inkling's sink-expert MoE, verified against the reference
``_inkling_gate_select_kernel`` (``nvidia/moe.py`` at vLLM v0.26.0)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kernels import moe_sink_gate_router
from max.nn.layer import LayerList
from max.nn.moe import MoEGate, MoEQuantized
from max.nn.quant_config import fp4_packed_k
from typing_extensions import Self

_ROUTER_DTYPE = DType.float32


class InklingRouting(NamedTuple):
    """One token batch's routing decision, including the sink weights.

    Sinks are shared experts the router weights, not attention sinks.
    """

    expert_ids: TensorValue
    expert_weights: TensorValue
    sink_weights: TensorValue


class InklingGate(MoEGate):
    """Sigmoid gate with a selection bias, sink lanes, and a global scale.

    ``num_experts`` counts routed experts only; ``weight`` covers
    ``num_experts + n_shared_experts`` rows, ``bias`` the routed range alone.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        n_shared_experts: int,
        route_scale: float,
        is_sharding: bool = False,
    ) -> None:
        # Skip MoEGate's gate_score Linear so the checkpoint's ``weight``
        # name lands on this module directly.
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts + n_shared_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=True,
        )
        self.n_routed_experts = num_experts
        self.n_shared_experts = n_shared_experts
        self.route_scale = route_scale

        if not is_sharding:
            self.weight = Weight(
                "weight",
                dtype,
                [num_experts + n_shared_experts, hidden_dim],
                device=devices[0],
            )
            self.bias = Weight(
                "bias",
                _ROUTER_DTYPE,
                [num_experts],
                device=devices[0],
            )
            self.global_scale = Weight(
                "global_scale",
                _ROUTER_DTYPE,
                [1],
                device=devices[0],
            )

    def route(self, hidden_states: TensorValue) -> InklingRouting:
        """Routes one ragged batch of tokens."""
        device = hidden_states.device
        # Only the score math needs float32; the GEMM runs in the activation
        # dtype, as in the minimax/deepseek gate pattern.
        weight = ops.cast(self.weight, hidden_states.dtype).to(device)
        logits = ops.cast(hidden_states @ weight.T, _ROUTER_DTYPE)

        return InklingRouting(
            *moe_sink_gate_router(
                logits,
                self.bias.to(device),
                self.global_scale.to(device),
                n_routed_experts=self.n_routed_experts,
                n_experts_per_tok=self.num_experts_per_token,
                n_shared_experts=self.n_shared_experts,
                route_scale=self.route_scale,
            )
        )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # Deliberately no super() call: this gate skips MoEGate's gate_score
        # Linear, which the base implementation would try to shard.
        if not strategy.is_replicate:
            raise ValueError(
                "Only replicate sharding strategy is supported for InklingGate."
            )
        self._sharding_strategy = strategy
        replicate = ShardingStrategy.replicate(strategy.num_devices)
        self.weight.sharding_strategy = replicate
        self.bias.sharding_strategy = replicate
        self.global_scale.sharding_strategy = replicate

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[InklingGate]:
        """Creates one replicated view of this gate per device."""
        if not self._sharding_strategy:
            raise ValueError(
                "InklingGate cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        weight_shards = self.weight.shard(devices)
        bias_shards = self.bias.shard(devices)
        global_scale_shards = self.global_scale.shard(devices)

        shards: list[InklingGate] = []
        for shard_idx, device in enumerate(devices):
            sharded = InklingGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.n_routed_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                n_shared_experts=self.n_shared_experts,
                route_scale=self.route_scale,
                is_sharding=True,
            )
            sharded.weight = weight_shards[shard_idx]
            sharded.bias = bias_shards[shard_idx]
            sharded.global_scale = global_scale_shards[shard_idx]
            shards.append(sharded)
        return shards


class InklingMoE(MoEQuantized):
    """Routed experts plus weighted sink experts."""

    def _init_experts(self) -> None:
        """Declares the routed experts as the checkpoint's stacked tensors."""
        # The expert list stays empty: MoE only reads it to stack per-expert
        # weights, which the properties below replace.
        self._all_experts = []
        self.experts = LayerList([])

        experts, hidden = self.num_experts, self.hidden_dim
        device = self.devices[0]
        self.w13_weight = Weight(
            "experts.w13_weight",
            self.dtype,
            [
                experts,
                2 * self.moe_dim,
                fp4_packed_k(hidden, self.quant_config),
            ],
            device=device,
        )
        self.w2_weight = Weight(
            "experts.w2_weight",
            self.dtype,
            [experts, hidden, fp4_packed_k(self.moe_dim, self.quant_config)],
            device=device,
        )
        if self.quant_config is None:
            return

        # The scale shapes below assume one scale row per weight row, which
        # holds for NVFP4's (1, 16) block but not for a block-FP8 config.
        block = self.quant_config.weight_scale.block_size
        assert block is not None and block[0] == 1
        scale_dtype = self.quant_config.weight_scale.dtype
        global_dtype = self.quant_config.input_scale.dtype
        self.w13_weight_scale = Weight(
            "experts.w13_weight.weight_scale",
            scale_dtype,
            [experts, 2 * self.moe_dim, hidden // block[1]],
            device=device,
        )
        self.w2_weight_scale = Weight(
            "experts.w2_weight.weight_scale",
            scale_dtype,
            [experts, hidden, self.moe_dim // block[1]],
            device=device,
        )
        self.w13_weight_scale_2 = Weight(
            "experts.w13_weight.weight_scale_2",
            global_dtype,
            [experts],
            device=device,
        )
        self.w2_weight_scale_2 = Weight(
            "experts.w2_weight.weight_scale_2",
            global_dtype,
            [experts],
            device=device,
        )
        # One activation scale per stacked tensor: the checkpoint records a
        # single absmax for all experts.
        self.w13_input_scale = Weight(
            "experts.w13_weight.input_scale", global_dtype, [1], device=device
        )
        self.w2_input_scale = Weight(
            "experts.w2_weight.input_scale", global_dtype, [1], device=device
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        return self.w13_weight

    @property
    def down_proj(self) -> TensorValue:
        return self.w2_weight

    @property
    def gate_up_proj_scales(self) -> TensorValue:
        return self.w13_weight_scale

    @property
    def down_proj_scales(self) -> TensorValue:
        return self.w2_weight_scale

    def _collect_scale_2(self, proj_name: str) -> TensorValue:
        return (
            self.w13_weight_scale_2
            if proj_name == "gate_proj"
            else self.w2_weight_scale_2
        )

    def _collect_input_scale(
        self, proj_name: str, collect_all: bool = False
    ) -> TensorValue:
        """One absmax per stacked tensor; the kernels index by expert id."""
        del collect_all
        scale = (
            self.w13_input_scale
            if proj_name == "gate_proj"
            else self.w2_input_scale
        )
        return ops.broadcast_to(scale, [self.num_experts])

    def _routed_weight_axes(self) -> dict[str, int | None]:
        """Routed tensor -> sharded axis (``None`` = replicated)."""
        axes: dict[str, int | None] = {"w13_weight": 1, "w2_weight": 2}
        if self.quant_config is not None:
            axes |= {
                "w13_weight_scale": 1,
                "w2_weight_scale": 2,
                "w13_weight_scale_2": None,
                "w2_weight_scale_2": None,
                "w13_input_scale": None,
                "w2_input_scale": None,
            }
        return axes

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the expert intermediate: an equal split of ``w13``'s
        interleaved output axis keeps each rank's gate/up rows paired."""
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor parallel sharding is supported for InklingMoE."
            )
        super()._set_sharding_strategy(strategy)
        for name, axis in self._routed_weight_axes().items():
            weight: Weight = getattr(self, name)
            weight.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
                if axis is None
                else ShardingStrategy.axiswise(
                    axis=axis, num_devices=strategy.num_devices
                )
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Self]:
        """Creates one per-device view of this layer."""
        devices = list(devices)
        routed_shards = {
            name: getattr(self, name).shard(devices)
            for name in self._routed_weight_axes()
        }
        shards = super().shard(devices)
        for shard_idx, shard in enumerate(shards):
            for name, weight_shards in routed_shards.items():
                setattr(shard, name, weight_shards[shard_idx])
        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        assert isinstance(self.gate, InklingGate)
        routing = self.gate.route(x)
        return self._routed_experts(x, routing) + self._sink_experts(
            x, routing.sink_weights
        )

    def _routed_experts(
        self, x: TensorValue, routing: InklingRouting
    ) -> TensorValue:
        down_projs = self._expert_matmuls(
            x, ops.reshape(routing.expert_ids, [-1])
        )

        weights = ops.cast(routing.expert_weights, down_projs.dtype)
        return ops.squeeze(
            ops.sum(ops.unsqueeze(weights, axis=-1) * down_projs, axis=1),
            axis=1,
        )

    def _sink_experts(
        self, x: TensorValue, sink_weights: TensorValue
    ) -> TensorValue:
        """Runs every sink expert on every token, weighted by its gate value,
        applied to the intermediate as the reference fuses it."""
        mlp = self.shared_experts
        ffl = mlp.gate_proj.weight.shape[0]
        gate_up = ops.concat((mlp.gate_proj.weight, mlp.up_proj.weight))
        gate, up = ops.split(x @ gate_up.T, [ffl, ffl], axis=-1)
        hidden = ops.silu(gate) * up
        sinks_here = self._sink_ids_on_this_shard()
        if len(sinks_here) != int(sink_weights.shape[1]):
            # A TP shard sees only the sinks its column range falls in.
            sink_weights = sink_weights[:, sinks_here.start : sinks_here.stop]
        weights = ops.cast(sink_weights, hidden.dtype)
        if len(sinks_here) == 1:
            # Rank-2 saves a kernel launch here: the rank-3 reshape stops
            # folding once the intermediate comes from split slices.
            weighted = hidden * weights
        else:
            weighted = ops.reshape(
                ops.reshape(hidden, [hidden.shape[0], len(sinks_here), -1])
                * ops.unsqueeze(weights, axis=-1),
                hidden.shape,
            )
        return mlp.down_proj(weighted)

    def _sink_ids_on_this_shard(self) -> range:
        """Which sink experts this shard's intermediate columns belong to."""
        assert isinstance(self.gate, InklingGate)
        n_sinks = self.gate.n_shared_experts
        width = self.shared_experts_dim
        num_shards = max(len(self.shard_devices), 1)
        per_sink = width * num_shards // n_sinks
        start = self.shard_index * width
        if per_sink % width == 0:
            return range(start // per_sink, start // per_sink + 1)
        # A shard covers whole sink experts when it does not sit inside one.
        assert width % per_sink == 0 and start % per_sink == 0
        first = start // per_sink
        return range(first, first + width // per_sink)
