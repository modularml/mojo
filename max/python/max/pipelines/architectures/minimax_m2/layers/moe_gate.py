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

"""Mixture of Experts Gate Layer for MiniMax-M2.

Uses sigmoid routing with expert score correction bias, similar to
DeepSeek V3 but without group-limited routing (n_groups=1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import moe_router_group_limited
from max.nn.linear import Linear
from max.nn.moe import MoEGate
from max.nn.moe.moe import ShardingStrategy


class MiniMaxM2TopKRouter(MoEGate):
    """MoE gate with sigmoid routing and expert score correction bias.

    Implements the MiniMax-M2 routing strategy:
    1. Compute gate logits via linear projection
    2. Apply sigmoid activation
    3. Add learnable e_score_correction_bias for expert selection
    4. Select top-k experts
    5. Normalize top-k weights to sum to 1
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
    ) -> None:
        """Initializes the MiniMax-M2 MoE gate.

        Args:
            num_experts_per_token: The number of experts per token.
            num_experts: The total number of experts.
            norm_topk_prob: Whether to normalize top-k probabilities.
            hidden_dim: The dimension of the hidden state.
            dtype: The data type of the MoEGate.
            gate_dtype: The data type for the gate linear layer.
            correction_bias_dtype: The data type of the correction bias.
            devices: The devices to use.
            linear_cls: Linear class for the gate projection.
            is_sharding: Whether this is being created during sharding.
        """
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

        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim).

        Returns:
            tuple containing:
                - topk_idx: Indices of selected experts (seq_len, num_experts_per_token).
                - topk_weight: Routing weights (seq_len, num_experts_per_token).
        """
        logits = self.gate_score(hidden_states)
        scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))

        # Use moe_router_group_limited with n_groups=1 for simple topk
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=1.0,
        )
        return topk_idx, topk_weight

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Replicates the correction bias alongside the base gate weights."""
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> Sequence[MiniMaxM2TopKRouter]:
        """Create sharded views of this gate across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MiniMaxM2TopKRouter instances.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding "
                "strategy was provided."
            )

        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards: list[MiniMaxM2TopKRouter] = []
        for shard_idx, device in enumerate(devices):
            sharded = MiniMaxM2TopKRouter(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                norm_topk_prob=self.norm_topk_prob,
                dtype=self.dtype,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                devices=[device],
                is_sharding=True,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
