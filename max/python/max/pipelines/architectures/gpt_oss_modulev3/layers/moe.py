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

"""GPT OSS Mixture of Experts Layer."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear
from max.experimental.nn.common_layers.functional_kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
)
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.nn.module import Module
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from ..model_config import GptOssConfig


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        """
        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
        """
        # Initialize parent class
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        # Override gate_score with bias-enabled Linear layer
        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            bias=True,
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores with softmax applied.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        # Apply softmax to top-k scores (matching GptOss behavior)
        topk_scores = F.softmax(topk_scores)

        return topk_indices, topk_scores


class GptOssMoE(Module[[Tensor], Tensor]):
    """GptOss-style MoE implementation with custom activation and biases."""

    def __init__(self, config: GptOssConfig):
        """
        Args:
            config: The configuration for the GPT OSS Model.
        """
        self.config = config

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.apply_router_weight_first = False
        self.gate = GptOssMoEGate(
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
        )
        self.moe_dim = config.intermediate_size

        self.alpha = 1.702
        self.limit = 7.0

        self._init_experts()

    def _init_experts(self) -> None:
        # Instead of creating individual MLP experts, we'll use combined weight tensors
        # This matches how the weights are stored in the checkpoint
        self.experts = ModuleList([])  # Empty list to maintain compatibility

        # Create combined weight tensors for all experts
        # Gate and up projections are combined into one tensor
        self._experts_gate_up_proj_weight = Tensor.zeros(
            shape=[self.num_experts, self.hidden_dim, 2 * self.moe_dim],
        )

        # Down projection weights
        self._experts_down_proj_weight = Tensor.zeros(
            shape=[self.num_experts, self.moe_dim, self.hidden_dim],
        )

        # Bias terms for gate_up projection (combined)
        self._experts_gate_up_proj_bias = Tensor.zeros(
            shape=[self.num_experts, 2 * self.moe_dim],
        )

        # Bias terms for down projection
        self._experts_down_proj_bias = Tensor.zeros(
            shape=[self.num_experts, self.hidden_dim],
        )

    @property
    def gate_up_proj(self) -> Tensor:
        # Return the combined gate_up projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_gate_up_proj_weight.transpose(1, 2)

    @property
    def down_proj(self) -> Tensor:
        # Return the combined down projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_down_proj_weight.transpose(1, 2)

    @property
    def gate_up_proj_bias_stacked(self) -> Tensor:
        # Return the combined gate_up projection biases
        return self._experts_gate_up_proj_bias

    @property
    def down_proj_bias_stacked(self) -> Tensor:
        # Return the combined down projection biases
        return self._experts_down_proj_bias

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
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
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        # Apply gate_up projection with bias
        gate_up_output = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        # Apply bias based on expert assignment
        # We need to gather the bias for each token based on which expert it was routed to
        # router_idx contains the expert assignment for each token
        expert_assignments = F.gather(router_idx, token_expert_order, axis=0)
        bias_per_token = F.gather(
            self.gate_up_proj_bias_stacked, expert_assignments, axis=0
        )
        gate_up_output += bias_per_token

        # Split gate and up projections
        gate = gate_up_output[:, 0::2]
        up = gate_up_output[:, 1::2]

        # Apply clamping (NOTE: This is specific to GptOss)
        gate = F.min(gate, self.limit)
        up = up.clip(min=-self.limit, max=self.limit)

        # GptOss-style activation: gate * sigmoid(gate * alpha) * (up + 1)
        glu = gate * F.sigmoid(gate * self.alpha)
        gated_output = (up + 1.0) * glu

        # Apply down projection
        down_output = grouped_matmul_ragged(
            gated_output,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        # Apply bias based on expert assignment
        # Use the same expert assignments we calculated earlier
        down_bias_per_token = F.gather(
            self.down_proj_bias_stacked, expert_assignments, axis=0
        )

        down_output = down_output + down_bias_per_token

        # Reshape and apply routing weights
        down_output = F.gather(
            down_output, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, -1])

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_output
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_output.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        return routed_expert_out
