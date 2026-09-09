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

"""Laguna rotary embedding.

Overrides ``ProportionalRotaryEmbedding`` to normalize inv_freq by rotary_dim
instead of head_dim. When ``partial_rotary_factor < 1`` the rotary
dimension is smaller than head_dim; Laguna's HF reference computes
``1/(theta^(i/rotary_dim))`` not ``1/(theta^(i/head_dim))``. For
``poolside/Laguna-M.1-NVFP4`` the factor is 1.0, so rotary_dim equals
head_dim (full rotary).
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)


class LagunaRotaryEmbedding(ProportionalRotaryEmbedding):
    """Partial RoPE where ``inv_freq`` is normalized by ``rotary_dim``, not ``head_dim``.

    Gemma4's ``ProportionalRotaryEmbedding`` computes:

    .. code-block:: text

        inv_freq = 1/(theta^(iota / head_dim))

    Laguna's HF reference computes:

    .. code-block:: text

        inv_freq = 1/(theta^(iota / rotary_dim))

    When ``partial_rotary_factor < 1`` (so ``rotary_dim`` < ``head_dim``),
    Gemma4's frequencies would be off in the exponent and weaken
    positional encoding; normalizing by ``rotary_dim`` corrects this. With
    ``partial_rotary_factor=1.0`` (M.1) the two definitions coincide.
    """

    def _compute_inv_freqs(self) -> TensorValue:
        n = self.head_dim
        partial_rotary_factor = (
            self.scaling_params.partial_rotary_factor
            if self.scaling_params is not None
            else 1.0
        )

        rotary_dim = int(partial_rotary_factor * n)
        rope_angles = rotary_dim // 2

        iota = ops.range(
            0,
            2 * rope_angles,
            step=2,
            dtype=DType.float64,
            device=DeviceRef.CPU(),
        )
        # Normalize by rotary_dim (not head_dim) to match HF reference
        inv_freq_rotated = ops.cast(
            1.0 / (self.theta ** (iota / rotary_dim)), DType.float32
        )

        return inv_freq_rotated
