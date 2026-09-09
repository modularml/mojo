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

"""DeepSeek-V2 Mixture of Experts gate (ModuleV3)."""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.moe import MoEGate
from max.experimental.tensor import Tensor
from max.graph import Shape


def _fill(
    fill_value: bool, dtype: DType, shape: Shape, device: Device
) -> Tensor:
    return F.broadcast_to(
        F.constant(fill_value, dtype=dtype, device=device), shape
    )


class DeepSeekV2MoEGate(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V2."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
    ) -> None:
        """
        Args:
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            topk_method: The method to select the top-k experts. Supported
                methods: "greedy", "group_limited_greedy"
            n_group: The number of groups (with group_limited_greedy).
            topk_group: The number of top k groups (with group_limited_greedy).
            routed_scaling_factor: Scaling factor for the routed experts
                when using group_limited_greedy top-k.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )

        if topk_method not in ("greedy", "group_limited_greedy"):
            raise ValueError(f"Invalid topk_method: {topk_method}")

        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)`` of selected expert indices and
            their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        logits = self.gate_score(hidden_states.cast(DType.float32))
        scores = F.softmax(logits.cast(DType.float32))

        if self.topk_method == "greedy":
            topk_weight, topk_idx = F.top_k(
                scores, self.num_experts_per_token, -1
            )
            return topk_idx, topk_weight

        # group_limited_greedy
        bsz_seq_len, _ = hidden_states.shape
        group_scores = F.max(
            scores.reshape((bsz_seq_len, self.n_group, -1)), axis=-1
        )
        group_scores = F.squeeze(group_scores, -1)  # [n, n_group]

        # Shape of group_idx: [n, top_k_group]
        group_idx = F.top_k(group_scores, k=self.topk_group, axis=-1)[1]

        group_mask = _fill(
            False, DType.bool, group_scores.shape, group_scores.device
        )  # [n, n_group]
        update = _fill(True, DType.bool, group_idx.shape, group_scores.device)
        group_mask = F.scatter(group_mask, update, group_idx, 1)

        score_mask = F.broadcast_to(
            F.unsqueeze(group_mask, -1),
            (
                bsz_seq_len,
                self.n_group,
                self.num_experts // self.n_group,
            ),
        ).reshape((bsz_seq_len, -1))  # [n, e]

        tmp_scores = F.where(
            score_mask.cast(DType.bool),
            scores,
            F.constant(0, dtype=scores.dtype, device=scores.device),
        )  # [n, e]

        topk_weight, topk_idx = F.top_k(
            tmp_scores, k=self.num_experts_per_token, axis=-1
        )
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight
