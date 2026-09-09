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

"""Gemma4 proportional rotary embedding for the ModuleV3 API."""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)


class ProportionalRotaryEmbedding(RotaryEmbedding):
    """RoPE where only ``partial_rotary_factor`` of head dims rotate.

    The first ``partial_rotary_factor * head_dim // 2`` inverse frequencies
    come from theta; the rest are zero (identity rotation). Used by Gemma4
    full-attention layers (theta 1e6, factor 0.25, head_dim 512).
    """

    scaling_params: ProportionalScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = False,
        scaling_params: ProportionalScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim, n_heads, theta, max_seq_len, device, head_dim, interleaved
        )
        self.scaling_params = scaling_params

    def _compute_inv_freqs(self) -> Tensor:
        n = self.head_dim
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )
        rope_angles = int(partial_rotary_factor * n // 2)
        nope_angles = n // 2 - rope_angles

        iota = F.arange(
            0, 2 * rope_angles, step=2, dtype=DType.float64, device=self.device
        )
        inv_freqs = F.cast(1.0 / (self.theta ** (iota / n)), DType.float32)

        if nope_angles > 0:
            zeros = Tensor.zeros(
                [nope_angles], dtype=DType.float32, device=self.device
            )
            inv_freqs = F.concat([inv_freqs, zeros], axis=0)
        return inv_freqs
