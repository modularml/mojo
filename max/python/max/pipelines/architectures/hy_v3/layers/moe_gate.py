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
"""Hy3 MoE gate: sigmoid + per-expert correction bias + scaled top-k."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import moe_router_group_limited
from max.nn.linear import Linear
from max.nn.moe import MoEGate
from max.nn.moe.moe import ShardingStrategy


class HYV3TopKRouter(MoEGate):
    """MoE gate with sigmoid + correction-bias top-k routing."""

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

        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        # Compute the router matmul in FP32 to match HF
        # (`F.linear(hidden.float(), weight.float())`); the top-8-of-192
        # decision is sensitive to BF16 matmul rounding.
        hs_fp32 = ops.cast(hidden_states, DType.float32)
        w_fp32 = ops.cast(self.gate_score.weight, DType.float32).to(
            hs_fp32.device
        )
        logits = hs_fp32 @ ops.transpose(w_fp32, -1, -2)
        scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        return topk_idx, topk_weight

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[HYV3TopKRouter]:
        if not self._sharding_strategy:
            raise ValueError("MoEGate cannot be sharded: no sharding strategy.")
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)
        shards: list[HYV3TopKRouter] = []
        for shard_idx, device in enumerate(devices):
            sharded = HYV3TopKRouter(
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
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
