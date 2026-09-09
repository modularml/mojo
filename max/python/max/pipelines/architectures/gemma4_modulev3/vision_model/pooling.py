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

"""Gemma4 vision pooling for the ModuleV3 API."""

from __future__ import annotations

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor


class Gemma4VisionPooler(Module[..., Tensor]):
    """Gather-based sparse average pooling; no learnable parameters.

    Port of the graph arch's Gemma4VisionPooler.__call__. The float16
    early-return branch is dropped: this arch is bf16-only.
    """

    def __init__(self, hidden_size: int, pooling_kernel_size: int) -> None:
        super().__init__()
        k2 = pooling_kernel_size * pooling_kernel_size
        self._post_scale: float = math.sqrt(hidden_size) / k2

    def forward(
        self, hidden_states: Tensor, pool_gather_index: Tensor
    ) -> Tensor:
        original_dtype = hidden_states.dtype
        h = hidden_states.cast(DType.float32)

        # Sentinel index (== total_patches) selects an appended zero row so
        # under-filled bins contribute nothing.
        zero_row = Tensor.zeros(
            [1, h.shape[1]], dtype=DType.float32, device=h.device
        )
        padded = F.concat([h, zero_row], axis=0)

        gathered = F.gather(padded, pool_gather_index, axis=0)
        pooled = gathered.sum(axis=1).squeeze(1)
        return (pooled * self._post_scale).cast(original_dtype)
