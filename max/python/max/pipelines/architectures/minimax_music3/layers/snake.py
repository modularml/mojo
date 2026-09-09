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
"""Snake activation used by the MiniMax Music 3 vocoder."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor


class Snake1d(Module[[Tensor], Tensor]):
    """Periodic activation ``x + sin(alpha * x)^2 / alpha`` with per-channel alpha.

    Unlike a ReLU or SiLU, this activation has a learned period per channel,
    which is what lets a DAC-style decoder produce harmonic structure. Operates
    on channels-last activations of shape ``(batch, length, channels)``.
    """

    alpha: Tensor
    """The learned per-channel period, shaped to broadcast over the input."""

    def __init__(self, channels: int) -> None:
        """Initializes the activation.

        Args:
            channels: Number of channels, matching the last input axis.
        """
        # Stored as (1, 1, channels) so it broadcasts over a channels-last
        # activation; the checkpoint's (1, channels, 1) is reshaped by the
        # weight adapter.
        self.alpha = Tensor.zeros([1, 1, channels])

    def forward(self, x: Tensor) -> Tensor:
        """Applies the activation to a channels-last tensor."""
        alpha = self.alpha.cast(x.dtype)
        # The reference adds 1e-9 before the reciprocal rather than after the
        # division, which matters for channels whose alpha underflows.
        return x + F.sin(alpha * x) ** 2 / (alpha + 1e-9)
