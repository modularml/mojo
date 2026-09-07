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

"""Mixture of Experts Gate Layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorValue, Weight, ops
from max.nn.kernels import moe_router_group_limited
from max.nn.linear import Linear
from max.nn.moe import MoEGate
from max.nn.moe.moe import ShardingStrategy


def _fill(
    fill_value: bool, dtype: DType, shape: Shape, device: DeviceRef
) -> TensorValue:
    return ops.constant(fill_value, dtype=dtype, device=device).broadcast_to(
        shape
    )


class DeepseekV3TopKRouter(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V3."""

    def __init__(
        self,
        num_experts_per_token: int,
        num_experts: int,
        routed_scaling_factor: float,
        scoring_func: str,
        topk_method: str,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        hidden_dim: int,
        dtype: DType,
        gate_dtype: DType,
        correction_bias_dtype: DType | None,
        devices: list[DeviceRef],
        use_fused_kernel: bool = True,
        linear_cls: Callable[..., Linear] = Linear,
    ) -> None:
        """
        Args:
            num_experts_per_token: The number of experts per token.
            num_experts: The number of experts to route to.
            routed_scaling_factor: The scaling factor for the routed experts.
            scoring_func: The scoring function for the experts.
            topk_method: The method to select the top-k experts.
            n_group: The number of groups.
            topk_group: The number of top-k groups.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            hidden_dim: The dimension of the hidden state.
            dtype: The data type of the MoEGate.
            correction_bias_dtype: The data type of the correction bias.
            devices: The devices to use for the MoEGate.
            use_fused_kernel: Whether to use the fused kernel for the MoEGate.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=gate_dtype,
            linear_cls=linear_cls,
        )

        if topk_method != "noaux_tc":
            raise ValueError(f"Invalid topk_method: {topk_method}")
        assert correction_bias_dtype

        assert scoring_func == "sigmoid"

        # This value is renamed to top_k in the original implementation, keep it
        # here for consistency.
        self.top_k = num_experts_per_token

        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.gate_dtype = gate_dtype
        self.correction_bias_dtype = correction_bias_dtype
        self.use_fused_kernel = use_fused_kernel

        if self.num_experts % self.n_group != 0:
            raise ValueError(
                f"num_experts must be divisible by n_group: {self.num_experts} % {self.n_group} != 0"
            )

        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def compute_scores(self, hidden_states: TensorValue) -> TensorValue:
        """Sigmoid scores in correction-bias dtype, ready for the router kernel."""
        logits = self.gate_score(hidden_states)
        return ops.sigmoid(logits.cast(self.correction_bias_dtype))

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        # compute gate score
        # logits = self.gate_score(hidden_states)
        scores = self.compute_scores(hidden_states)
        # scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return topk_idx, topk_weight

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Replicates the correction bias alongside the base gate weights."""
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded DeepseekV3TopKRouter instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = DeepseekV3TopKRouter(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.scoring_func,
                topk_method=self.topk_method,
                n_group=self.n_group,
                topk_group=self.topk_group,
                norm_topk_prob=self.norm_topk_prob,
                dtype=self.dtype,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                devices=[device],
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
