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

"""Patch embedding layer equivalent to MoonVision3dPatchEmbed (Kimi K2.5).

Reference: nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
- MoonVision3dPatchEmbed: proj (Conv2d) + pos_emb (Learnable2DInterpPosEmbDivided_fixed).

Checkpoint weight names and shapes (vision_tower.patch_embed.*, BF16):
  - vision_tower.patch_embed.pos_emb.weight   [64, 64, 1152]
  - vision_tower.patch_embed.proj.bias        [1152]
  - vision_tower.patch_embed.proj.weight      [1152, 3, 14, 14]  (out_ch, in_ch, kH, kW)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.conv import Conv2d
from max.experimental.sharding.action import ActionSet
from max.experimental.sharding.cost import force_replicated_action_set
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.nn.kernels import (
    learnable_2d_interp_pos_emb as _learnable_2d_interp_pos_emb,
)


def _learnable_2d_interp_pos_emb_rule(
    x: TensorLayout,
    weight: TensorLayout,
    grid_thws: TensorLayout,
    time_weight: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Replicated on every operand: the vision tower is not sharded."""
    return force_replicated_action_set(x, weight, grid_thws, time_weight)


learnable_2d_interp_pos_emb = F.functional(
    _learnable_2d_interp_pos_emb, rule=_learnable_2d_interp_pos_emb_rule
)


class Learnable2DInterpPosEmbDividedFixed(Module[[Tensor, Tensor], Tensor]):
    """Learnable 2D spatial position grid for MoonVision3dPatchEmbed.

    Holds the weight so checkpoint key ``patch_embed.pos_emb.weight`` loads.
    For each ``(t, h, w)`` in ``grid_thws``, bicubic-interpolates the 2D grid
    to ``(h, w)`` if needed, adds a 1D sincos temporal embedding when ``t > 1``,
    and adds the result element-wise to the patch embeddings ``x``.

    ``time_weight`` is pre-computed once on the host and materialized as a
    constant on the input device inside :meth:`forward`.
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        dim: int = 1152,
        num_frames: int = 4,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """Initializes the 2D position embedding grid.

        Args:
            height: Height of the learnable 2D grid (init_pos_emb_height).
            width: Width of the learnable 2D grid (init_pos_emb_width).
            dim: Embedding dimension (vt_hidden_size).
            num_frames: Maximum temporal frames for sincos embedding.
            dtype: Data type for the weight.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.num_frames = num_frames
        self.weight = Tensor.zeros((height, width, dim), dtype=dtype)
        self._time_weight_np = self._compute_time_weight()

    def _compute_time_weight(self) -> np.ndarray:
        """1D sincos temporal positional embedding of shape (num_frames, dim).

        Matches the reference ``get_1d_sincos_pos_embed``.
        """
        half = self.dim // 2
        omega = np.arange(half, dtype=np.float32)
        omega /= self.dim / 2.0
        omega = 1.0 / (10000.0**omega)
        grid_t = np.arange(self.num_frames, dtype=np.float32)
        out = np.einsum("m,d->md", grid_t, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    def forward(self, x: Tensor, grid_thws: Tensor) -> Tensor:
        """Adds interpolated 2D position embeddings to x via GPU kernel.

        Args:
            x: (L, dim) patch embeddings.
            grid_thws: (N, 3) temporal, height, width per video, int64.

        Returns:
            (L, dim) tensor with position embeddings added.
        """
        time_weight = F.constant(
            self._time_weight_np, dtype=DType.float32, device=x.device
        )
        return learnable_2d_interp_pos_emb(
            x=x,
            weight=self.weight,
            grid_thws=grid_thws,
            time_weight=time_weight,
        )


class PatchEmbedding(Module[[Tensor, Tensor], Tensor]):
    """Equivalent to MoonVision3dPatchEmbed from Kimi K2.5 (MoonViT3d).

    Implements:
    1. Projection: Conv2d(in_channels, hidden_size, kernel_size=patch_size,
       stride=patch_size). Input (L, 3, patch_size, patch_size) ->
       output (L, hidden_size).
    2. Position embedding: Learnable2DInterpPosEmbDividedFixed (pos_emb)
       applies bicubic-interpolated 2D spatial + 1D sincos temporal position
       embeddings via a GPU kernel.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1152,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        dtype: DType = DType.bfloat16,
        has_bias: bool = True,
    ) -> None:
        """Initializes the patch embedding layer to match MoonVision3dPatchEmbed.

        Args:
            patch_size: Spatial size of each patch (height and width).
            in_channels: Number of input channels (3 for RGB).
            hidden_size: Output embedding dimension (vt_hidden_size).
            init_pos_emb_height: Height of learnable 2D position grid (vision_config).
            init_pos_emb_width: Width of learnable 2D position grid.
            init_pos_emb_time: Number of temporal steps for 1D sincos time embedding.
            dtype: Data type for weights and computation.
            has_bias: Whether the Conv2d projection includes a bias term.
        """
        super().__init__()
        self.hidden_size = hidden_size
        # Conv2d with permute=True consumes NCHW input -> NHWC internally.
        self.proj = Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            has_bias=has_bias,
            dtype=dtype,
            permute=True,
        )
        self.pos_emb = Learnable2DInterpPosEmbDividedFixed(
            height=init_pos_emb_height,
            width=init_pos_emb_width,
            dim=hidden_size,
            num_frames=init_pos_emb_time,
            dtype=dtype,
        )

    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> Tensor:
        """Patch projection followed by 2D interpolated position embedding.

        Matches reference: ``x = self.proj(x).view(x.size(0), -1)`` then
        ``x = self.pos_emb(x, grid_thws)``.

        Args:
            pixel_values: (n_patches, in_channels, patch_size, patch_size) in NCHW format.
            grid_thws: (n_videos, 3) temporal, height, width per video, int64.

        Returns:
            Tensor of shape (n_patches, hidden_size).
        """
        # Conv2d output (n_patches, hidden_size, 1, 1) -> (n_patches, hidden_size).
        patch_embeds = self.proj(pixel_values)
        x = patch_embeds.reshape([patch_embeds.shape[0], self.hidden_size])
        x = self.pos_emb(x, grid_thws)
        return x
