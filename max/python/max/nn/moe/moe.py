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
"""A generalized Mixture of Experts (MoE) module."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from typing_extensions import Self

from ..comm.ep import EPBatchManager
from ..comm.ep.ep_kernels import fused_silu
from ..kernels import grouped_matmul_ragged, moe_create_indices
from ..layer import Layer, LayerList, Module, Shardable
from ..linear import MLP, Linear
from ..quant_config import QuantConfig


def make_concatenated_gated_activation_fn(
    activation_fn: Callable[[TensorValue], TensorValue],
    limit: float | None = None,
) -> Callable[[TensorValue, int], TensorValue]:
    """Builds a gated activation for concatenated ``[gate | up]`` projections.

    The returned callable splits ``gate_up`` at ``moe_dim``, applies
    ``activation_fn`` to the gate half, and multiplies with the up half:
    ``activation_fn(gate_up[:, :moe_dim]) * gate_up[:, moe_dim:]``.

    When ``limit`` is provided, both halves are clamped to
    ``[-limit, limit]`` before multiplying.
    """
    assert limit is None or limit > 0, (
        f"limit must be None or positive, got {limit}"
    )

    if limit is not None:

        def _clamped_concatenated_gated_activation_fn(
            gate_up: TensorValue, moe_dim: int
        ) -> TensorValue:
            gate = activation_fn(gate_up[:, :moe_dim])
            up = gate_up[:, moe_dim:]
            lim = ops.constant(limit, gate.dtype, device=gate.device)
            neg_lim = ops.constant(-limit, up.dtype, device=up.device)
            gate = ops.min(gate, lim)
            up = ops.min(ops.max(up, neg_lim), lim)
            return gate * up

        return _clamped_concatenated_gated_activation_fn

    def _concatenated_gated_activation_fn(
        gate_up: TensorValue, moe_dim: int
    ) -> TensorValue:
        gate = activation_fn(gate_up[:, :moe_dim])
        up = gate_up[:, moe_dim:]
        return gate * up

    return _concatenated_gated_activation_fn


@dataclass
class _InterleavedGatedActivation:
    """Gated activation for interleaved ``[g0, u0, g1, u1, ...]`` projections.

    Reads the gate and up halves with a stride rather than a split point,
    which lets a checkpoint with interleaved gate/up rows reach the matmul
    unrearranged. A distinct type carrying ``activation_fn`` so
    :class:`MoEQuantized` can recognize both the layout and the activation
    when selecting the fused SwiGLU kernel.
    """

    activation_fn: Callable[[TensorValue], TensorValue]

    def __call__(self, gate_up: TensorValue, moe_dim: int) -> TensorValue:
        del moe_dim  # The halves are separated by a stride, not a split point.
        return self.activation_fn(gate_up[:, 0::2]) * gate_up[:, 1::2]


def make_interleaved_gated_activation_fn(
    activation_fn: Callable[[TensorValue], TensorValue],
) -> Callable[[TensorValue, int], TensorValue]:
    """Builds a gated activation for interleaved ``[gate | up]`` projections."""
    return _InterleavedGatedActivation(activation_fn)


def _swigluoai_activation(
    gate_up: TensorValue,
    moe_dim: int,
    alpha: float,
    limit: float,
) -> TensorValue:
    """Applies the OAI-style clamped SwiGLU activation."""
    gate = gate_up[:, :moe_dim]
    up = gate_up[:, moe_dim:]

    lim = ops.constant(limit, gate.dtype, device=gate.device)
    neg_lim = ops.constant(-limit, up.dtype, device=up.device)
    alpha_value = ops.constant(alpha, gate.dtype, device=gate.device)

    gate = ops.min(gate, lim)
    up = ops.min(ops.max(up, neg_lim), lim)
    return (up + 1.0) * gate * ops.sigmoid(gate * alpha_value)


class MoEGate(Module):
    """Gate module for MoE."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        is_sharding: bool = False,
        linear_cls: Callable[..., Linear] = Linear,
    ) -> None:
        """Args:
        devices: List of devices to use for the MoEGate.
        hidden_dim: The dimension of the hidden state.
        num_experts: The number of experts.
        num_experts_per_token: The number of experts per token.
        dtype: The data type of the MoEGate.
        """
        super().__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.dtype = dtype

        if not is_sharding:
            self.gate_score = linear_cls(
                in_dim=hidden_dim,
                out_dim=num_experts,
                dtype=dtype,
                device=devices[0],
            )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        return topk_indices, topk_scores

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        self._set_sharding_strategy(strategy)

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Applies ``strategy`` to this gate's weights.

        Subclasses that add per-weight strategies override this and call
        ``super()._set_sharding_strategy(strategy)``; a property setter cannot
        be extended through ``super()``.
        """
        if strategy.is_replicate:
            self._sharding_strategy = strategy
            self.gate_score.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            raise ValueError(
                "Only replicate sharding strategy is supported for MoEGate."
            )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MoEGate instances, one for each device.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = MoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            shards.append(sharded)
        return shards


class MoE(Module, Shardable):
    """Implementation of Mixture of Experts (MoE).

    Args:
        devices: The list of devices to use for the MoE.
        hidden_dim: The dimension of the hidden state.
        num_experts: The number of experts.
        num_experts_per_token: The number of experts per token.
        moe_dim: The intermediate dimension of each expert.
        gate_cls: The model-specific gate implementation. Defaults to
            :class:`~max.nn.moe.moe.MoEGate`.
        mlp_cls: The MLP class to use for experts. Defaults to
            :class:`~max.nn.linear.MLP`.
        has_shared_experts: Whether to use shared experts. Defaults to
            ``False``.
        shared_experts_dim: The dimension of the shared experts.
            Defaults to ``0``.
        ep_size: The expert parallelism size. Defaults to ``1``.
        dtype: The data type of the MoE. Defaults to
            ``DType.bfloat16``.
        apply_router_weight_first: Whether to apply the router weight
            first. Defaults to ``False``.
        ep_batch_manager: The expert parallel batch manager. Defaults to
            ``None``.
        quant_config: The scaled quantization configuration. Defaults to
            ``None``.
        use_swigluoai: Whether to use the OAI-style clamped SwiGLU activation
            function. Defaults to ``False``.
        swiglu_alpha: The alpha value for the clamped SwiGLU activation function.
            Defaults to ``0.0``.
        swiglu_limit: The limit value for the clamped SwiGLU activation function.
            Defaults to ``0.0``.
        gated_activation_fn: Activation applied to the concatenated
            ``[gate | up]`` projection. ``None`` (default) uses a fused
            SiLU kernel; use
            :func:`make_concatenated_gated_activation_fn` for custom
            activations.
        shared_experts_dtype: Weight storage dtype for shared-expert MLPs. When
            equal to ``dtype`` (routed experts) and ``quant_config`` is set,
            shared experts use the same quantization as routed experts. When
            different (e.g. BF16 shared weights with packed NVFP4 routed experts),
            shared linears omit ``quant_config`` unless
            ``shared_experts_quant_config`` is set. Defaults to ``dtype``.
        shared_experts_quant_config: Optional separate :class:`QuantConfig` for
            shared-expert MLPs when their storage dtype differs from routed
            experts (e.g. MXFP8 shared with NVFP4 routed). Defaults to
            ``None``.
        pre_expert_norm_cls: A callable that returns a normalization
            module to apply before expert computation. Defaults to
            ``None``.
        is_sharding: Whether the constructor is being called during
            sharding. Defaults to ``False``.
    """

    _ep_batch_manager: EPBatchManager | None = None
    """The expert parallel batch manager."""

    _sharding_strategy: ShardingStrategy | None = None
    """The sharding strategy for the module."""

    experts: LayerList
    """The list of experts."""

    _all_experts: list[Layer]
    """The list of all experts when using expert parallel strategy."""

    shard_devices: list[DeviceRef] = []
    """The list of devices the MoE layer was sharded to."""

    shard_index: int = 0
    """The index of the current shard (if the MoE layer was sharded)."""

    layer_idx: int | None = None
    """The index of the MoE layer."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,  # phycial id
        num_experts_per_token: int,
        moe_dim: int,
        num_logical_experts: int | None = None,
        gate_cls: Callable[..., MoEGate] = MoEGate,
        mlp_cls: Callable[..., MLP] = MLP,
        shared_mlp_cls: Callable[..., MLP] | None = None,
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        ep_size: int = 1,
        dtype: DType = DType.bfloat16,
        apply_router_weight_first: bool = False,
        use_swigluoai: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
        gated_activation_fn: Callable[[TensorValue, int], TensorValue]
        | None = None,
        pre_expert_norm_cls: Callable[[], Module] | None = None,
        ep_batch_manager: EPBatchManager | None = None,
        quant_config: QuantConfig | None = None,
        shared_experts_dtype: DType | None = None,
        shared_experts_quant_config: QuantConfig | None = None,
        is_sharding: bool = False,
    ):
        super().__init__()
        self.devices = devices
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_logical_experts = num_logical_experts or num_experts
        self.moe_dim = moe_dim
        self.gate_cls = gate_cls
        self.mlp_cls = mlp_cls
        self.shared_mlp_cls = shared_mlp_cls
        self.has_shared_experts = has_shared_experts
        self.shared_experts_dim = shared_experts_dim
        self.ep_size = ep_size
        self.dtype = dtype
        self.apply_router_weight_first = apply_router_weight_first
        self.use_swigluoai = use_swigluoai
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.gated_activation_fn = gated_activation_fn
        self.pre_expert_norm_cls = pre_expert_norm_cls
        self.pre_expert_norm = (
            pre_expert_norm_cls() if pre_expert_norm_cls else None
        )
        self.gate = gate_cls(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=self.num_logical_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=DType.bfloat16,
        )
        self.num_local_experts = num_experts // ep_size
        self.quant_config = quant_config
        self.shared_experts_dtype = (
            shared_experts_dtype if shared_experts_dtype is not None else dtype
        )
        self.shared_experts_quant_config = shared_experts_quant_config

        if use_swigluoai:
            assert swiglu_alpha != 0.0 and swiglu_limit != 0.0, (
                "swiglu_alpha and swiglu_limit must be set when use_swigluoai is True"
            )

        assert not (use_swigluoai and gated_activation_fn is not None), (
            "use_swigluoai and gated_activation_fn cannot be set at the same time"
        )

        if has_shared_experts:
            assert shared_experts_dim > 0, (
                "shared_experts_dim must be greater than 0"
            )
            if shared_experts_quant_config is not None:
                shared_quant = shared_experts_quant_config
            elif (
                quant_config is not None and self.shared_experts_dtype == dtype
            ):
                shared_quant = quant_config
            else:
                shared_quant = None
            self.shared_experts = (
                self.shared_mlp_cls
                if self.shared_mlp_cls is not None
                else mlp_cls
            )(
                dtype=self.shared_experts_dtype,
                quantization_encoding=None,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.shared_experts_dim,
                devices=self.devices,
                quant_config=shared_quant,
            )

        if ep_batch_manager:
            assert not apply_router_weight_first, (
                "apply_router_weight_first is not supported for expert parallel strategy"
            )

            self._ep_batch_manager = ep_batch_manager

        if not is_sharding:
            self._init_experts()

    def _init_experts(self) -> None:
        self._all_experts = [
            self.mlp_cls(
                dtype=self.dtype,
                quantization_encoding=None,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.moe_dim,
                devices=self.devices,
                quant_config=self.quant_config,
            )
            for _ in range(self.num_logical_experts)
        ]

        self.experts = LayerList(self._all_experts)

    @property
    def ep_batch_manager(self) -> EPBatchManager:
        """Get the expert parallel batch manager."""
        assert self._ep_batch_manager is not None, (
            "EPBatchManager must be provided if using expert parallel strategy"
        )
        return self._ep_batch_manager

    def configure_ep_scale_fusion(self, dispatch_supports_fold: bool) -> None:
        """Configure any EP dispatch-scale fusion before the dispatch op.

        No-op on the base class; ``MoEQuantized`` overrides it to enable the
        MXFP4 up-proj A-scale preshuffle fold. Defined here (rather than
        duck-typed) so the EP forward driver can call it on any ``MoE`` shard:
        non-quantized subclasses inherit this no-op and consistently skip the
        fold (no fusion, no corruption).

        Args:
            dispatch_supports_fold: Whether the selected dispatch path wires the
                A-scale fold params. Ignored by this base no-op.
        """

    @property
    def _shared_experts_use_quant(self) -> bool:
        """Whether shared experts use quantized weights in the MoE path."""
        if self.shared_experts_quant_config is not None:
            return True
        return (
            self.quant_config is not None
            and self.shared_experts_dtype == self.dtype
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        self._set_sharding_strategy(strategy)

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Applies ``strategy`` to the gate, shared experts, and experts.

        Subclasses that add per-weight strategies override this and call
        ``super()._set_sharding_strategy(strategy)``; a property setter cannot
        be extended through ``super()``.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy
            self.gate.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            if self.has_shared_experts:
                self.shared_experts.sharding_strategy = strategy

            for expert in self.experts:
                expert.sharding_strategy = strategy
        elif strategy.is_expert_parallel:
            self._sharding_strategy = strategy
            self.gate.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            if self.has_shared_experts:
                self.shared_experts.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
            for expert in self.experts:
                # each expert will only present on one device
                expert.sharding_strategy = ShardingStrategy.replicate(1)
        else:
            raise ValueError(
                "Only tensor parallel or expert parallel sharding strategies are supported for MoE"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Self]:
        """Create sharded views of this MoE module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MoE instances, one for each device.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoE module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_shards = self.gate.shard(devices)

        if self.has_shared_experts:
            shared_experts_shards = self.shared_experts.shard(devices)

        # Replicate the pre-expert norm; the per-shard constructor would
        # otherwise register duplicate weights under one name.
        pre_expert_norm_shards = None
        if self.pre_expert_norm is not None:
            assert isinstance(self.pre_expert_norm, Shardable)
            self.pre_expert_norm.sharding_strategy = ShardingStrategy.replicate(
                self._sharding_strategy.num_devices
            )
            pre_expert_norm_shards = self.pre_expert_norm.shard(devices)

        shards = []
        num_devices = self._sharding_strategy.num_devices
        sharded_moe_dim = self.moe_dim // num_devices
        sharded_shared_experts_dim = self.shared_experts_dim // num_devices
        if self._sharding_strategy.is_expert_parallel:
            sharded_moe_dim = self.moe_dim
            sharded_shared_experts_dim = self.shared_experts_dim

        devices = list(devices)

        for shard_idx, device in enumerate(devices):
            sharded = self.__class__(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                num_logical_experts=self.num_logical_experts,
                moe_dim=sharded_moe_dim,
                gate_cls=self.gate_cls,
                mlp_cls=self.mlp_cls,
                shared_mlp_cls=self.shared_mlp_cls,
                has_shared_experts=self.has_shared_experts,
                shared_experts_dim=sharded_shared_experts_dim,
                ep_size=self.ep_size,
                dtype=self.dtype,
                apply_router_weight_first=self.apply_router_weight_first,
                use_swigluoai=self.use_swigluoai,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_limit=self.swiglu_limit,
                gated_activation_fn=self.gated_activation_fn,
                pre_expert_norm_cls=self.pre_expert_norm_cls,
                quant_config=self.quant_config,
                shared_experts_dtype=self.shared_experts_dtype,
                shared_experts_quant_config=self.shared_experts_quant_config,
                is_sharding=True,
            )

            # Keep a reference to the original experts for sharded instances.
            sharded._all_experts = self._all_experts

            # Replace layers and weights with sharded versions.
            sharded.gate = gate_shards[shard_idx]
            if self.has_shared_experts:
                sharded.shared_experts = shared_experts_shards[shard_idx]
            if pre_expert_norm_shards is not None:
                sharded.pre_expert_norm = pre_expert_norm_shards[shard_idx]

            if self._sharding_strategy.is_tensor_parallel:
                sharded.shard_index = shard_idx
                sharded.shard_devices = devices
                sharded.experts = LayerList(list(self.experts))

            elif self._sharding_strategy.is_expert_parallel:
                curr_node_idx = self.ep_batch_manager.config.node_id
                num_experts_per_node = (
                    self.num_experts // self.ep_batch_manager.config.n_nodes
                )
                expert_idx = (
                    curr_node_idx * num_experts_per_node
                    + shard_idx * self.num_local_experts
                )

                experts_list: list[MLP] = []
                for _ in range(self.num_local_experts):
                    plan = self.ep_batch_manager._eplb_phy2log
                    if plan is not None:
                        assert self.layer_idx is not None, (
                            "MoE.layer_idx must be set when EPLB is enabled"
                        )
                        log_id = int(plan[self.layer_idx, expert_idx])
                    else:
                        log_id = expert_idx
                    curr_expert = self.experts[log_id]
                    assert isinstance(curr_expert, MLP)
                    experts_list.append(curr_expert.shard([device])[0])
                    expert_idx += 1

                sharded.experts = LayerList(experts_list)
                sharded._ep_batch_manager = self.ep_batch_manager

            sharded.layer_idx = self.layer_idx
            shards.append(sharded)

        return shards

    def _uses_fused_swiglu_layout(self) -> bool:
        # True when gate_up weights and scales are sigma-permuted to the
        # (gate, up) interleaved N-axis layout that the fused
        # SwiGLU+NVFP4 grouped-matmul kernel consumes. The kernel runs a
        # fused SiLU so gated_activation_fn must be None, and the layout
        # only makes sense under expert parallelism.
        return (
            self.quant_config is not None
            and self.quant_config.can_use_fused_swiglu
            and self._ep_batch_manager is not None
            and self.gated_activation_fn is None
        )

    @property
    def gate_up_proj(self) -> TensorValue:
        gate_list = [expert.gate_proj.weight for expert in self.experts]
        up_list = [expert.up_proj.weight for expert in self.experts]

        if (
            self._ep_batch_manager
            and self.ep_batch_manager.config.fused_shared_expert
        ):
            assert self.has_shared_experts, (
                "Shared experts must present if fused shared expert is enabled"
            )
            gate_list = [
                self.shared_experts.gate_proj.weight,
            ] + gate_list
            up_list = [
                self.shared_experts.up_proj.weight,
            ] + up_list

        # Use the actual weight K dimension to support packed formats (e.g. NVFP4).
        k_dim = gate_list[0].shape[1]

        gate_up_list: list[TensorValue] = []
        for tensors in zip(gate_list, up_list, strict=True):
            gate_up_list.extend(tensors)

        if not self.shard_devices:
            shard = ops.stack(gate_up_list, axis=0)
        else:
            # Create target devices for each shard - each shard goes to a
            # different GPU. For tensor parallelism, total_shards == len(devices).
            shard = ops.shard_and_stack(
                gate_up_list,
                devices=self.shard_devices,
            )[self.shard_index]

        # The fused SwiGLU+NVFP4 grouped matmul kernel requires the per-expert
        # N axis to be sigma-permuted: rows 2i = gate row i, rows 2i+1 = up
        # row i. Reshape the stacked [2E, D, K] tensor to [E, 2, D, K], permute
        # axes 1 and 2 to [E, D, 2, K], then collapse to [E, 2D, K] — the
        # innermost rows are now interleaved (g_0, u_0, g_1, u_1, ...). One
        # bulk permute replaces E per-expert stacks to keep the graph small
        # and the constant-folding tractable.
        if self._uses_fused_swiglu_layout():
            shard = shard.reshape([len(gate_list), 2, -1, k_dim])
            shard = ops.permute(shard, [0, 2, 1, 3])
            return shard.reshape([len(gate_list), -1, k_dim])

        return shard.reshape([len(gate_list), -1, k_dim])

    @property
    def down_proj(self) -> TensorValue:
        down_list = [expert.down_proj.weight for expert in self.experts]

        if (
            self._ep_batch_manager
            and self.ep_batch_manager.config.fused_shared_expert
        ):
            assert self.has_shared_experts, (
                "Shared experts must present if fused shared expert is enabled"
            )
            down_list = [
                self.shared_experts.down_proj.weight,
            ] + down_list

        if not self.shard_devices:
            shard = ops.stack(down_list, axis=0)
        else:
            devices = [DeviceRef.CPU()] * len(self.shard_devices)
            shard = ops.shard_and_stack(
                down_list,
                devices=devices,
                axis=-1,
            )[self.shard_index].to(self.devices[0])

        return shard

    def _ep_dispatch_input_scales(self) -> TensorValue | None:
        """Returns quantized input scales for EP dispatch, or ``None``.

        Overridden in :class:`MoEQuantized` for NVFP4 support.
        """
        return None

    def _swigluoai_activation(self, gate_up: TensorValue) -> TensorValue:
        """Applies the configured OAI-style clamped SwiGLU activation."""
        return _swigluoai_activation(
            gate_up,
            self.moe_dim,
            self.swiglu_alpha,
            self.swiglu_limit,
        )

    def _local_ep_compute(
        self,
        expert_inputs: tuple[TensorValue, ...],
        x: TensorValue,
        estimated_total_m: TensorValue,
    ) -> TensorValue:
        """Runs local expert matmuls on dispatched tokens.

        This is the computation between ``ep_dispatch`` and
        ``ep_combine``.  ``x`` is unused by :class:`MoE` but accepted
        for interface consistency with :class:`MoEQuantized`.

        Args:
            expert_inputs: Dispatch outputs (tokens, row_offsets, ...).
            x: Original input (unused by base MoE, used by subclasses).
            estimated_total_m: Estimated total received tokens for the current
                device. Used as a matmul shape hint.
        """
        gate_up_projs = grouped_matmul_ragged(
            expert_inputs[0],
            self.gate_up_proj,
            *expert_inputs[1:],
        )
        if self.gated_activation_fn is not None:
            activated = self.gated_activation_fn(gate_up_projs, self.moe_dim)
        elif self.use_swigluoai:
            activated = self._swigluoai_activation(gate_up_projs)
        else:
            activated = fused_silu(gate_up_projs, expert_inputs[1])
        return grouped_matmul_ragged(
            activated,
            self.down_proj,
            *expert_inputs[1:],
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        if self._ep_batch_manager:
            raise ValueError(
                "Use forward_moe_sharded_layers for expert-parallel inference "
                f"instead of calling {type(self).__name__} directly."
            )

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)

        if self.pre_expert_norm is not None:
            x = self.pre_expert_norm(x)

        down_projs = self._expert_matmuls(
            x, ops.reshape(router_idx, [-1]), router_weight
        )

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
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

    def _expert_matmuls(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue | None = None,
    ) -> TensorValue:
        """Runs the unquantized expert matmuls for one flat expert assignment.

        Args:
            x: ``[seq_len, hidden_dim]`` expert input.
            router_idx: ``[seq_len * num_experts_per_token]`` selected expert
                ids, in token-major order.
            router_weight: ``[seq_len, num_experts_per_token]`` router weights,
                read only when ``apply_router_weight_first`` is set.

        Returns:
            ``[seq_len, num_experts_per_token, hidden_dim]``, each selected
            expert's output before the router weights are applied.

        A subclass whose router carries state beyond the ids and weights
        overrides ``__call__`` and calls this from there. ``MoEQuantized``
        overrides it with the quantized expert matmuls, so such a subclass
        works against either base.
        """
        seq_len = x.shape[0]

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            ops.cast(
                ops.floor_div(token_expert_order, self.num_experts_per_token),
                DType.int32,
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            assert router_weight is not None, (
                "router_weight is required when apply_router_weight_first is set"
            )
            permutated_states = permutated_states * ops.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        if self.gated_activation_fn is not None:
            gate_up_projs = self.gated_activation_fn(
                gate_up_projs, self.moe_dim
            )
        elif self.use_swigluoai:
            gate_up_projs = self._swigluoai_activation(gate_up_projs)
        else:
            gate_up_projs = fused_silu(gate_up_projs, expert_start_indices)

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        return ops.gather(down_projs, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, self.hidden_dim]
        )
