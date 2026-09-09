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
"""A generalized Mixture of Experts (MoE) module.

These classes need to be cleaned up before moving to `max.nn`.
"""

from __future__ import annotations

from collections.abc import Callable

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear
from max.experimental.nn.common_layers.functional_kernels import (
    fused_silu,
    grouped_matmul_ragged,
    moe_create_indices,
    shard_and_stack,
    stack_device_shards,
)
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.module import Module
from max.experimental.nn.sequential import ModuleList
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    PlacementMapping,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from typing_extensions import Self


class MoEGate(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Gate module for MoE."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        """Initialize MoE gate.

        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.gate_score = Linear(
            in_dim=hidden_dim, out_dim=num_experts, bias=False
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for MoE gate.

        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        return topk_indices, topk_scores


class MoE(Module[[Tensor], Tensor]):
    """Implementation of Mixture of Experts (MoE)."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        apply_router_weight_first: bool = False,
    ):
        """Initialize MoE layer.

        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            gate_cls: The model specific gate implementation.
            has_shared_experts: Whether to use shared experts.
            shared_experts_dim: The dimension of the shared experts.
            apply_router_weight_first: Whether to apply the router weight first.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.apply_router_weight_first = apply_router_weight_first
        self.gate = gate_cls(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.moe_dim = moe_dim

        self.shared_experts: MLP | None = None
        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            self.shared_experts = MLP(
                hidden_dim=self.hidden_dim,
                feed_forward_length=shared_experts_dim,
                bias=False,
            )

        self._init_experts()

    def _init_experts(self) -> None:
        self.experts: ModuleList[MLP] = ModuleList(
            [
                MLP(
                    hidden_dim=self.hidden_dim,
                    feed_forward_length=self.moe_dim,
                    bias=False,
                )
                for _ in range(self.num_experts)
            ]
        )

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer layers with fully replicated placements."""
        mesh = _mesh(target)
        if mesh.num_devices > 1:
            raise ValueError(
                "Cannot transfer MoE Layer to multi-device mesh. Use TensorParallelMoE or ExpertParallelMoE instead."
            )
        super().to(target)
        return self

    @property
    def gate_up_proj(self) -> list[Tensor]:
        """Per-device ``[gate, up]`` expert-weight bundle (one entry here)."""
        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[Tensor] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        return [
            F.stack(gate_up_list, axis=0).reshape(
                [self.num_experts, -1, self.hidden_dim]
            )
        ]

    @property
    def down_proj(self) -> list[Tensor]:
        """Per-device down-projection weight bundle (one entry here)."""
        return [
            F.stack(
                [expert.down_proj.weight for expert in self.experts], axis=0
            )
        ]

    def _expert_matmuls_local(
        self,
        tokens: Tensor,
        gate_up: Tensor,
        down: Tensor,
        expert_start: Tensor,
        expert_ids: Tensor,
        usage_stats: Tensor,
    ) -> Tensor:
        """Gate/up matmul -> SiLU -> down matmul."""
        gate_up_out = grouped_matmul_ragged(
            tokens, gate_up, expert_start, expert_ids, usage_stats
        )
        silu_out = fused_silu(gate_up_out, expert_start)
        return grouped_matmul_ragged(
            silu_out, down, expert_start, expert_ids, usage_stats
        )

    def _grouped_expert_compute(
        self,
        permuted_states: Tensor,
        expert_start_indices: Tensor,
        expert_ids: Tensor,
        expert_usage_stats: Tensor,
    ) -> Tensor:
        """Runs :meth:`_expert_matmuls_local` on the (possibly TP) mesh directly."""
        mesh = permuted_states.mesh
        gate_up = stack_device_shards(self.gate_up_proj, axis=1, mesh=mesh)
        down = stack_device_shards(self.down_proj, axis=2, mesh=mesh)
        out = self._expert_matmuls_local(
            permuted_states,
            gate_up,
            down,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        return Tensor.from_shard_values(
            [TensorValue(s) for s in out.local_shards],
            mapping=permuted_states.mapping,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for MoE layer.

        Args:
            x: (seq_len, hidden_dim)

        Returns:
            Tensor with shape (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)
        router_idx = F.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            F.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = F.gather(
            x,
            F.cast(
                F.floor_div(token_expert_order, self.num_experts_per_token),
                DType.int32,
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        down_projs = self._grouped_expert_compute(
            permutated_states,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        down_projs = F.gather(down_projs, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, -1]
        )

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_projs
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.shared_experts is not None:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out


class TensorParallelMoE(MoE):
    """MoE layer with tensor parallelism."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initial_moe_dim = self.moe_dim
        self.mesh = DeviceMesh.single(self.device)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Transfer the MoE layer to the target device."""
        self.mesh = _mesh(target)
        if self.mesh.ndim != 1:
            raise ValueError(
                f"Mesh used with TensorParallelMoE must have exactly one device axis, but got {self.mesh}"
            )
        self.moe_dim = self._initial_moe_dim // self.mesh.num_devices

        self.gate.to(target)
        if self.shared_experts is not None:
            self.shared_experts.to(target)

        # If there are multiple devices, keep expert weights on CPU because
        # the weights will be transferred via the `shard_and_stack` operation.
        device = CPU() if self.mesh.num_devices > 1 else self.mesh.devices[0]
        for expert in self.experts:
            expert.to(device)
        return self

    @property
    def gate_up_proj(self) -> list[Tensor]:
        """Per-device ``[gate, up]`` weight bundle, sharded along ``moe_dim``."""
        if self.mesh.num_devices == 1:
            return super().gate_up_proj

        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        gate_up_list: list[Tensor] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        shards: list[Tensor] = []
        for shard in shard_and_stack(gate_up_list, devices=self.mesh.devices):
            assert isinstance(shard, Tensor)
            shards.append(
                shard.reshape([self.num_experts, -1, self.hidden_dim])
            )
        return shards

    @property
    def down_proj(self) -> list[Tensor]:
        """Per-device down-projection weight bundle, sharded along ``moe_dim``."""
        if self.mesh.num_devices == 1:
            return super().down_proj
        down_proj_list = [expert.down_proj.weight for expert in self.experts]
        shards: list[Tensor] = []
        for shard in shard_and_stack(
            down_proj_list, devices=self.mesh.devices, axis=-1
        ):
            assert isinstance(shard, Tensor)
            shards.append(shard)
        return shards

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for tensor parallel MoE layer.

        This method assumes that `x` is already replicated along the mesh.
        """
        with ensure_context():
            output = super().forward(x)

            # `_grouped_expert_compute` reassembles the per-device expert
            # outputs under the replicated activation placement, but each
            # device only summed its own ``moe_dim`` slice, so the values are
            # partial. Re-tag as Partial; the caller resolves with an
            # all-reduce.
            return Tensor.from_shard_values(
                [TensorValue(s) for s in output.local_shards],
                mapping=PlacementMapping(self.mesh, (Partial(),)),
            )


def _mesh(target: Device | DeviceMesh | DeviceMapping) -> DeviceMesh:
    """Compute the mesh for the target device."""
    if isinstance(target, DeviceMesh):
        return target
    elif isinstance(target, Device):
        return DeviceMesh.single(target)
    elif isinstance(target, DeviceMapping):
        return target.mesh
    else:
        raise ValueError(f"Invalid target device: {target}")
