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
"""Pixel generation input types for Modular's MAX API."""

from __future__ import annotations

__all__ = [
    "PixelGenerationInputs",
]

from dataclasses import dataclass, field
from typing import Generic

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    PixelGenerationContextType,
)
from max.pipelines.modeling.types.pipeline import PipelineInputs
from max.pipelines.request import RequestID


@dataclass(frozen=True)
class _PixelGenerationRequest:
    """An immutable request for pixel (image) generation from a pipeline."""

    request_id: RequestID = field()
    model_name: str = field()
    prompt: str
    secondary_prompt: str | None = None
    negative_prompt: str | None = None
    secondary_negative_prompt: str | None = None
    guidance_scale: float = 3.5
    true_cfg_scale: float = 1.0
    height: int | None = None
    width: int | None = None
    num_inference_steps: int = 50
    num_images_per_prompt: int = 1
    seed: int | None = None
    input_image: npt.NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        if self.prompt == "":
            raise ValueError("Prompt must be provided.")
        if (self.height is not None and self.height <= 0) or (
            self.width is not None and self.width <= 0
        ):
            raise ValueError("Height and width must be positive.")
        if self.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be positive.")
        if self.num_images_per_prompt <= 0:
            raise ValueError("Number of images per prompt must be positive.")


@dataclass(frozen=True)
class PixelGenerationInputs(
    PipelineInputs, Generic[PixelGenerationContextType]
):
    """Input data structure for pixel generation pipelines."""

    batch: dict[RequestID, PixelGenerationContextType]
    """A dictionary mapping RequestID to PixelGenerationContextType instances."""
