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

"""Gemma4 RMSNorm for the ModuleV3 API."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.norm.rms_norm import rms_norm
from max.experimental.tensor import Tensor


class Gemma4RMSNorm(Module[[Tensor], Tensor]):
    """RMSNorm computing ``weight * rms_norm(x)`` (Gemma4 convention).

    Gemma4 checkpoints store the full scale, so ``weight_offset`` is 0.0 —
    unlike Gemma3's ``(1 + weight)`` GemmaRMSNorm. ``with_weight=False``
    applies bare normalization (the v-norm has no learned scale).
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, with_weight: bool = True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.with_weight = with_weight
        if with_weight:
            self.weight = Tensor.ones([dim])

    def forward(self, x: Tensor) -> Tensor:
        if self.with_weight:
            w = self.weight
        else:
            w = Tensor.ones([self.dim], dtype=x.dtype, device=x.device)
        return rms_norm(
            x, w, self.eps, weight_offset=0.0, multiply_before_cast=True
        )
