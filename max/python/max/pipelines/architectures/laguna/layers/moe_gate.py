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

"""Laguna MoE gate.

Sigmoid routing with per-expert score-correction bias and an optional
tanh softcap on the router logits. ``moe_router_group_limited`` with
``n_groups=1`` provides simple top-k routing on top of the sigmoid
scores. Routed-output scaling (``routed_scaling_factor`` from the HF
config) is folded into the routing weights here so the per-expert
output multiplication picks it up automatically — mathematically
equivalent to HF's post-experts ``hidden * routed_scaling_factor``
scaling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.linear import Linear
from max.nn.moe import MoEGate
from max.nn.moe.moe import ShardingStrategy


class LagunaTopKRouter(MoEGate):
    """MoE gate with sigmoid routing and expert score correction bias.

    Implements the Laguna routing strategy:

    1. Compute gate logits via linear projection.
    2. Apply sigmoid activation.
    3. Add the learnable ``e_score_correction_bias`` for expert selection.
    4. Select top-k experts.
    5. Normalize top-k weights to sum to 1.

    Args:
        num_experts_per_token: Top-k expert count per token.
        num_experts: Total number of routed experts.
        norm_topk_prob: Whether to L1-normalise the selected top-k
            routing weights so they sum to 1.
        hidden_dim: Hidden dimension of the model.
        dtype: Data type carried by the parent ``MoEGate``.
        gate_dtype: Linear-projection dtype for the gate logits.
        correction_bias_dtype: Dtype of the per-expert
            ``e_score_correction_bias`` weight.
        devices: Devices to place the gate on.
        linear_cls: Linear class for the gate projection.
        is_sharding: Whether this instance is being created during
            ``shard()`` (skip weight creation in that case).
        routed_scaling_factor: Scalar multiplied into routing
            weights — equivalent to scaling the routed-expert
            output. Read from the HF config; donor models use 1.0.
        router_logit_softcapping: If > 0, router logits are passed
            through ``softcap * tanh(logits / softcap)`` before
            sigmoid. Disabled (0.0) for
            ``poolside/Laguna-M.1-NVFP4``.
    """

    def __init__(
        self,
        num_experts_per_token: int,
        num_experts: int,
        norm_topk_prob: bool,
        hidden_dim: int,
        dtype: DType,
        gate_dtype: DType,
        correction_bias_dtype: DType,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        is_sharding: bool = False,
        routed_scaling_factor: float = 1.0,
        router_logit_softcapping: float = 0.0,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=gate_dtype,
            linear_cls=linear_cls,
            is_sharding=is_sharding,
        )

        self.norm_topk_prob = norm_topk_prob
        self.gate_dtype = gate_dtype
        self.correction_bias_dtype = correction_bias_dtype
        self.routed_scaling_factor = routed_scaling_factor
        self.router_logit_softcapping = router_logit_softcapping

        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Computes expert routing weights and indices.

        Args:
            hidden_states: The input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A tuple ``(topk_idx, topk_weight)``: the indices of the selected
            experts and their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        logits = self.gate_score(hidden_states)
        # Optional tanh softcap on router logits — HF reference:
        #   if router_logit_softcapping > 0.0:
        #       router_logits = tanh(router_logits / softcap) * softcap
        if self.router_logit_softcapping > 0.0:
            softcap = ops.constant(
                self.router_logit_softcapping,
                logits.dtype,
                device=logits.device,
            )
            logits = ops.tanh(logits / softcap) * softcap
        scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))

        # Plain top-k routing — Laguna-M.1 has NO expert groups (unlike the
        # MiniMax-M2/DeepSeek lineage this gate was adapted from). Select experts
        # by the bias-corrected score, but weight them by the *unbiased* sigmoid
        # score (HF reference). ``moe_router_group_limited(n_groups=1)`` forced
        # the single-group GPU router kernel, whose ``phase2_candidates <=
        # WARP_SIZE`` constraint rejects top-k > 32 — M.1 is top-16 of 256
        # (XS.2 was top-8, which is why the donor idiom slipped through).
        # ``ops.top_k`` has no such constraint.
        sel = scores + self.e_score_correction_bias
        topk_sel, topk_idx = ops.top_k(
            sel, k=self.num_experts_per_token, axis=-1
        )
        # Unbiased routing weight = (score + bias) - bias at the selected experts.
        bias_at_idx = ops.gather(self.e_score_correction_bias, topk_idx, axis=0)
        topk_weight = topk_sel - bias_at_idx
        if self.norm_topk_prob:
            topk_weight = topk_weight / ops.sum(topk_weight, axis=-1)
        # Fold routed_scaling_factor into the weights (HF applies it to the
        # summed expert output, which is mathematically equivalent).
        topk_weight = topk_weight * ops.constant(
            self.routed_scaling_factor,
            topk_weight.dtype,
            device=topk_weight.device,
        )
        return topk_idx, topk_weight

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Replicates the correction bias alongside the base gate weights."""
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[LagunaTopKRouter]:
        """Creates sharded views of this gate across multiple devices.

        Args:
            devices: The iterable of devices to place the shards on.

        Returns:
            The list of sharded ``LagunaTopKRouter`` instances.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding "
                "strategy was provided."
            )

        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards: list[LagunaTopKRouter] = []
        for shard_idx, device in enumerate(devices):
            sharded = LagunaTopKRouter(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                norm_topk_prob=self.norm_topk_prob,
                dtype=self.dtype,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                devices=[device],
                is_sharding=True,
                routed_scaling_factor=self.routed_scaling_factor,
                router_logit_softcapping=self.router_logit_softcapping,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
