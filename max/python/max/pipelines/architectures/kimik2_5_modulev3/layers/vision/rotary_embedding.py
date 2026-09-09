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

from __future__ import annotations

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.sharding import DeviceMapping, DeviceMesh, Replicated
from max.experimental.tensor import Tensor
from typing_extensions import Self


class Rope2DPosEmbRepeated(Module[[Tensor], Tensor]):
    """2D rotary positional embedding for vision tokens.

    Builds a (max_height * max_width, dim//2, 2) table of [cos, sin] values.

    The interleaving matches the Kimi K2.5 torch reference: for each
    frequency index *i* the table stores [x_cos_i, x_sin_i, y_cos_i, y_sin_i]
    when viewed as a flat dim//2 * 2 vector.

    Args:
        dim: Embedding dimension, must be divisible by 4.
        max_height: Maximum grid height for precomputed frequencies.
        max_width: Maximum grid width for precomputed frequencies.
        theta_base: Base for the inverse-frequency exponent.
        device: Device on which to build the frequency table.
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.mapping = _replicated_mapping(self.device)

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        self.mapping = _replicated_mapping(target)
        return super().to(target)

    def _freqs_cis(self) -> Tensor:
        """Flattened frequency table of shape (max_height * max_width, dim//2, 2).

        The last dimension is [cos, sin].
        """
        N = self.max_height * self.max_width
        quarter = self.dim // 4  # dim//4 frequency components

        # Inverse exponentials. Use float64 for the exponent to avoid overflow,
        # then cast down.
        dim_range = F.range(
            0,
            self.dim,
            4,
            out_dim=quarter,
            dtype=DType.float64,
            device=CPU(),
        )
        freqs = F.cast(
            1.0 / (self.theta_base ** (dim_range / self.dim)),
            DType.float32,
        )
        freqs = freqs.to(self.mapping)

        # Spatial positions: flat_index -> (col, row)
        flat = F.range(
            0, N, 1, out_dim=N, dtype=DType.float32, device=self.mapping
        )
        mw = F.constant(self.max_width, DType.float32, device=self.mapping)
        x_pos = flat % mw
        y_pos = F.floor(flat / mw)

        # Outer products -> (N, quarter)
        x_freqs = F.outer(x_pos, freqs)
        y_freqs = F.outer(y_pos, freqs)

        # cos/sin -> (N, quarter, 2)
        x_embed = F.stack([F.cos(x_freqs), F.sin(x_freqs)], axis=-1)
        y_embed = F.stack([F.cos(y_freqs), F.sin(y_freqs)], axis=-1)

        # Interleave x and y: (N, quarter, 2, 2)
        combined = F.stack([x_embed, y_embed], axis=2)

        # Flatten to (N, dim//2, 2)
        return combined.reshape((N, self.dim // 2, 2))

    def forward(self, position_ids: Tensor) -> Tensor:
        """Gathers precomputed [cos, sin] pairs for the given position ids.

        Args:
            position_ids: 1-D int tensor of flat grid indices
                (row * max_width + col).

        Returns:
            Tensor of shape (len(position_ids), dim//2, 2).
        """
        return F.gather(self._freqs_cis(), position_ids, axis=0)


def _replicated_mapping(
    target: Device | DeviceMesh | DeviceMapping,
) -> DeviceMapping:
    """Fully-replicated mapping over the target's mesh.

    The frequency table is a whole lookup table gathered by absolute position,
    so it is replicated regardless of how ``target`` places anything else.
    """
    if isinstance(target, DeviceMesh):
        mesh = target
    elif isinstance(target, Device):
        mesh = DeviceMesh.single(target)
    elif isinstance(target, DeviceMapping):
        mesh = target.mesh
    else:
        raise ValueError(f"Invalid target device: {target}")
    return DeviceMapping(mesh, (Replicated(),) * mesh.ndim)
