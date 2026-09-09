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

"""DeepSeek-V3 Mixture of Experts gate (ModuleV3)."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    moe_router_group_limited,
)
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.tensor import Tensor


class DeepseekV3TopKRouter(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V3 (ModuleV3).

    Uses sigmoid scoring with a group-limited top-k router and a learned
    expert correction bias (``noaux_tc``).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        routed_scaling_factor: float,
        scoring_func: str,
        topk_method: str,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        correction_bias_dtype: DType | None,
    ) -> None:
        """
        Args:
            hidden_dim: Dimension of the hidden state.
            num_experts: Number of routed experts.
            num_experts_per_token: Top-k experts to select per token.
            routed_scaling_factor: Scaling factor applied to expert weights.
            scoring_func: Score function for the router. Only "sigmoid" is
                supported.
            topk_method: Top-k routing method. Only "noaux_tc" is supported.
            n_group: Number of expert groups.
            topk_group: Number of expert groups selected per token.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            correction_bias_dtype: Data type of the correction bias.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        if topk_method != "noaux_tc":
            raise ValueError(
                f"Invalid topk_method: {topk_method}. Only 'noaux_tc' is "
                "supported for DeepSeek-V3."
            )
        if scoring_func != "sigmoid":
            raise ValueError(
                f"Invalid scoring_func: {scoring_func}. Only 'sigmoid' is "
                "supported for DeepSeek-V3."
            )

        if num_experts % n_group != 0:
            raise ValueError(
                f"num_experts must be divisible by n_group: {num_experts} % "
                f"{n_group} != 0"
            )

        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        if correction_bias_dtype is None:
            raise ValueError(
                "correction_bias_dtype must be set for noaux_tc router"
            )
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=correction_bias_dtype
        )

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)`` of selected expert indices and
            their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        logits = self.gate_score(hidden_states)
        scores = F.sigmoid(logits.cast(self.e_score_correction_bias.dtype))
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
