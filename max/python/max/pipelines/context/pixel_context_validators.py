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

"""Context validators for pixel-generation pipelines (FLUX2, WAN)."""

from __future__ import annotations

from .context import PixelContext, TextAndVisionContext, TextContext
from .exceptions import InputError

# Matches `Flux2Pipeline.max_area = 1024 * 1024` in upstream diffusers, and
# mirrors `_max_pixel_size` already used by the FLUX2 input-image scale-down
# path in `pixel_tokenizer.py`.
_FLUX2_MAX_PIXEL_AREA: int = 1024 * 1024

# The resolutions the ticket explicitly baselines in
# `max/examples/diffusion/wan_comparison.py` go up to 2048x2048. Above the
# 720p ceiling that Wan-AI model cards advertise, but matches what the team
# validates against in practice.
_WAN_MAX_PIXEL_AREA: int = 2048 * 2048


def _check_pixel_area(
    context: TextContext | TextAndVisionContext | PixelContext,
    *,
    arch_name: str,
    max_pixel_area: int,
) -> None:
    if not isinstance(context, PixelContext):
        return
    area = context.height * context.width
    if area > max_pixel_area:
        raise InputError(
            f"{arch_name}: requested {context.width}x{context.height} "
            f"({area} pixels) exceeds the maximum allowed pixel area of "
            f"{max_pixel_area} for this architecture."
        )


def validate_flux2_max_pixel_area(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Rejects FLUX2 requests whose ``width * height`` exceeds the per-arch cap."""
    _check_pixel_area(
        context, arch_name="FLUX2", max_pixel_area=_FLUX2_MAX_PIXEL_AREA
    )


def validate_wan_max_pixel_area(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Rejects WAN requests whose ``width * height`` exceeds the per-arch cap."""
    _check_pixel_area(
        context, arch_name="WAN", max_pixel_area=_WAN_MAX_PIXEL_AREA
    )
