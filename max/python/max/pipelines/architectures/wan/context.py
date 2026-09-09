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
"""Wan-specific pixel generation context."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.context import PixelContext


@dataclass(kw_only=True)
class WanContext(PixelContext):
    """Pixel generation context with Wan-specific video/MoE fields."""

    num_frames: int | None = field(default=None)
    """Number of frames for video generation."""

    guidance_scale_2: float | None = field(default=None)
    """Secondary guidance scale for low-noise expert (MoE models)."""

    step_coefficients: npt.NDArray[np.float32] | None = field(default=None)
    """Pre-computed scheduler step coefficients."""

    boundary_timestep: float | None = field(default=None)
    """Timestep threshold for switching between high/low noise experts."""
