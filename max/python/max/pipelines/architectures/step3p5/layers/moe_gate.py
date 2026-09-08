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
"""Step-3.5 MoE gate with sigmoid routing, router bias, and weight normalization."""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.graph.weight import ShardingStrategy
from max.nn.moe import MoEGate


class Step3p5MoEGate(MoEGate):
    """Step-3.5 gate with sigmoid scoring, additive router bias, and top-k normalization.

    Routing:
        1. scores = sigmoid(gate(x))
        2. corrected = scores + router_bias
        3. top-k selection on corrected scores
        4. weights taken from original (uncorrected) scores
        5. optionally normalize top-k weights to sum to 1
        6. scale by routed_scaling_factor
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        routed_scaling_factor: float = 3.0,
        norm_topk_prob: bool = True,
        is_sharding: bool = False,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=is_sharding,
        )
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        if not is_sharding:
            self.router_bias = Weight(
                name="router_bias",
                dtype=DType.float32,
                shape=[num_experts],
                device=devices[0],
            )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Replicates ``router_bias`` alongside the base gate weights."""
        if not strategy.is_replicate:
            raise ValueError(
                "Only replicate sharding strategy is supported for Step3p5MoEGate."
            )
        super()._set_sharding_strategy(strategy)
        self.router_bias.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing with sigmoid + bias + normalization.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim).

        Returns:
            Tuple of (topk_indices, topk_weights).
        """
        # FP32 gate computation: cast both input and weight to float32
        # before the matmul to match the HF reference (need_fp32_gate=True).
        gate_weight = self.gate_score.weight.cast(DType.float32).to(
            hidden_states.device
        )
        logits = hidden_states.cast(DType.float32) @ gate_weight.T
        scores = ops.sigmoid(logits)

        # Add router bias for expert selection
        corrected_scores = scores + self.router_bias.to(scores.device)

        # Select top-k based on corrected scores
        _, topk_indices = ops.top_k(
            corrected_scores, k=self.num_experts_per_token, axis=-1
        )

        # Safety clamp: NaN/Inf in scores (e.g. from fp16 overflow) can make
        # top_k return out-of-range indices on some backends.  Clamping here
        # prevents index-out-of-bounds crashes at the cost of silently routing
        # to a wrong expert — acceptable as a last-resort guard since the root
        # cause (NaN scores) should be fixed upstream.
        topk_indices = ops.min(
            ops.max(
                topk_indices,
                ops.constant(0, topk_indices.dtype, device=topk_indices.device),
            ),
            ops.constant(
                self.num_experts - 1,
                topk_indices.dtype,
                device=topk_indices.device,
            ),
        )

        # Gather weights from original (uncorrected) scores
        # topk_indices: [seq_len, k], scores: [seq_len, num_experts]
        # Use gather_nd with batch_dims=1 to index per-row
        topk_weights = ops.gather_nd(
            scores,
            ops.unsqueeze(topk_indices, axis=-1),
            batch_dims=1,
        )

        # Normalize weights
        # Note: ops.sum keeps the reduced dim (returns [..., 1]), so no
        # unsqueeze is needed — the division broadcasts correctly.
        if self.norm_topk_prob:
            denominator = ops.sum(topk_weights, axis=-1) + ops.constant(
                1e-20, DType.float32, device=topk_weights.device
            )
            topk_weights = topk_weights / denominator

        # Scale weights
        topk_weights = topk_weights * ops.constant(
            self.routed_scaling_factor,
            DType.float32,
            device=topk_weights.device,
        )
        topk_weights = topk_weights.cast(hidden_states.dtype)
        return topk_indices, topk_weights

    def shard(self, devices: Iterable[DeviceRef]) -> list[Step3p5MoEGate]:
        """Create sharded views of this gate."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding "
                "strategy was provided."
            )
        gate_score_shards = self.gate_score.shard(devices)
        router_bias_shards = self.router_bias.shard(devices)
        shards: list[Step3p5MoEGate] = []
        for shard_idx, device in enumerate(devices):
            sharded = Step3p5MoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                routed_scaling_factor=self.routed_scaling_factor,
                norm_topk_prob=self.norm_topk_prob,
                is_sharding=True,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.router_bias = router_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
