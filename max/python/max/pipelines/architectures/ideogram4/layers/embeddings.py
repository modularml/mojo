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
"""Positional and timestep embeddings for Ideogram 4.

Reproduces ``ideogram4.modeling_ideogram4``:
  - ``Ideogram4MRoPE``: interleaved 3D multimodal rotary embeddings.
  - ``Ideogram4EmbedScalar``: sinusoidal scalar (timestep) embedding + MLP.
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef


def _mrope_axis_masks(half: int, mrope_section: tuple[int, ...]) -> np.ndarray:
    """Static per-frequency axis selection for interleaved MRoPE.

    Returns a ``(3, half)`` float32 array where row ``a`` is 1.0 at the
    frequency positions that take their value from axis ``a`` (0=temporal,
    1=height, 2=width). Mirrors the reference loop that scatters H/W freqs
    into indices ``arange(offset, mrope_section[axis]*3, 3)`` and leaves the
    rest temporal.
    """
    masks = np.zeros((3, half), dtype=np.float32)
    masks[0, :] = 1.0
    for axis, offset in ((1, 1), (2, 2)):
        length = mrope_section[axis] * 3
        idx = np.arange(offset, min(length, half), 3)
        masks[0, idx] = 0.0
        masks[axis, idx] = 1.0
    return masks


class Ideogram4MRoPE(Module[[Tensor], "tuple[Tensor, Tensor]"]):
    """Interleaved 3D multimodal rotary position embeddings.

    Produces ``(cos, sin)`` of shape ``(B, L, head_dim)`` from integer
    position ids of shape ``(B, L, 3)`` holding ``(t, h, w)`` coordinates.
    """

    def __init__(
        self,
        head_dim: int,
        base: float,
        mrope_section: tuple[int, ...],
        device: DeviceRef,
    ) -> None:
        self.head_dim = head_dim
        self.base = base
        self.mrope_section = mrope_section
        self.device = device
        self._half = head_dim // 2
        self._masks = _mrope_axis_masks(self._half, mrope_section)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        exponent = (
            F.arange(
                0, self.head_dim, 2, dtype=DType.float32, device=self.device
            )
            / self.head_dim
        )
        inv_freq = F.exp(exponent * -math.log(self.base))  # (half,)

        pos = F.cast(position_ids, DType.float32)  # (B, L, 3)

        freqs_t = None
        for axis in range(3):
            pos_a = pos[:, :, axis]  # (B, L)
            freqs_a = F.unsqueeze(pos_a, -1) * inv_freq  # (B, L, half)
            mask_a = Tensor(
                self._masks[axis], dtype=DType.float32, device=self.device
            )
            term = freqs_a * mask_a
            freqs_t = term if freqs_t is None else freqs_t + term

        emb = F.concat([freqs_t, freqs_t], axis=-1)  # (B, L, head_dim)
        return F.cos(emb), F.sin(emb)


def apply_mrope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings via rotate_half.

    Args:
        x: ``(B, L, H, D)`` query or key tensor.
        cos: ``(B, L, D)`` cosine table.
        sin: ``(B, L, D)`` sine table.

    Returns:
        Rotated tensor of shape ``(B, L, H, D)`` in ``x``'s dtype.
    """
    input_dtype = x.dtype
    d = int(x.shape[-1])
    half = d // 2

    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = F.concat([-x2, x1], axis=-1)  # rotate_half

    cos_e = F.unsqueeze(cos, 2)  # (B, L, 1, D)
    sin_e = F.unsqueeze(sin, 2)

    x_f = F.cast(x, DType.float32)
    rot_f = F.cast(rot, DType.float32)
    out = x_f * F.cast(cos_e, DType.float32) + rot_f * F.cast(
        sin_e, DType.float32
    )
    return F.cast(out, input_dtype)


class Ideogram4EmbedScalar(Module[[Tensor], Tensor]):
    """Sinusoidal scalar embedding followed by a 2-layer MLP (SiLU).

    Matches ``Ideogram4EmbedScalar`` with ``input_range=(0, 1)`` and
    ``scale=1e4``.
    """

    def __init__(
        self,
        dim: int,
        device: DeviceRef,
        input_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.dim = dim
        self.device = device
        self.range_min, self.range_max = input_range
        self.mlp_in = Linear(dim, dim, bias=True)
        self.mlp_out = Linear(dim, dim, bias=True)

    def _sinusoidal(self, t: Tensor, scale: float = 1e4) -> Tensor:
        half = self.dim // 2
        freq = math.log(scale) / (half - 1)
        exponent = F.arange(0, half, dtype=DType.float32, device=self.device)
        freq_band = F.exp(exponent * -freq)  # (half,)
        emb = F.unsqueeze(t, -1) * freq_band  # (..., half)
        emb = F.concat([F.sin(emb), F.cos(emb)], axis=-1)  # (..., dim)
        if self.dim % 2 == 1:
            zero = F.broadcast_to(
                F.reshape(
                    F.constant(0.0, emb.dtype, device=self.device),
                    (1,) * emb.rank,
                ),
                (*emb.shape[:-1], 1),
            )
            emb = F.concat([emb, zero], axis=-1)
        return emb

    def forward(self, x: Tensor) -> Tensor:
        x = F.cast(x, DType.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = self._sinusoidal(scaled)
        emb = F.cast(emb, self.mlp_in.weight.dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)
