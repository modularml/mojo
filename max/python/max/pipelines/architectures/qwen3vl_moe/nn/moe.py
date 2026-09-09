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
"""Mixture of Experts Layer for Qwen3VL MoE."""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.moe import MoEGate, StackedMoE


class Qwen3VLMoEGate(MoEGate):
    """Qwen3VL MoE Gate with simple top-k routing and softmax normalization."""

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
        # Compute router logits
        router_logits = self.gate_score(hidden_states).cast(DType.float32)
        # Apply softmax to get routing weights
        routing_weights = ops.softmax(router_logits, axis=-1)
        # Select top-k experts
        topk_weights, topk_indices = ops.top_k(
            routing_weights, k=self.num_experts_per_token, axis=-1
        )

        # Normalize top-k weights to sum to 1
        denominator = ops.sum(topk_weights, axis=-1) + 1e-20

        topk_weights = topk_weights / denominator
        # Cast back to original dtype
        topk_weights = topk_weights.cast(hidden_states.dtype)
        return topk_indices, topk_weights


class Qwen3VLMoE(StackedMoE):
    """Qwen3VL MoE implementation with standard gated activation.

    Uses StackedMoE defaults: concatenated gate/up format, silu activation,
    no bias, no shared experts.
    """
