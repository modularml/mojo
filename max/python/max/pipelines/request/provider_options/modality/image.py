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
"""Image generation modality provider options."""

from pydantic import Field, model_validator

from .common import PixelProviderOptionsBase


class ImageProviderOptions(PixelProviderOptionsBase):
    """Options specific to image generation pipelines.

    Inherits all generation options shared with other pixel modalities from
    :class:`PixelProviderOptionsBase`; adds image-specific fields (secondary
    text-encoder prompts, ``num_images``, ``strength``, ``output_format``) and
    tightens the dimension constraints to image-specific limits.
    """

    # Override base width/height to add image-specific minimum and the
    # multiple-of-8 constraint enforced by ``_validate_dimensions``. The
    # multiple-of-8 floor matches the VAE spatial scale factor used by every
    # currently supported pixel architecture; per-model tokenizers are
    # responsible for any further rounding to a patchification-compatible
    # size (e.g., Wan rounds each dimension down to a multiple of
    # ``vae_scale_factor * 2`` = 16).
    # Using 8 here lets callers request standard resolutions such as 1080p
    # (1920x1080) without the schema layer rejecting heights that are not
    # exact multiples of 16.
    width: int | None = Field(
        None,
        description="The width of the generated image in pixels. Must be at least 128 and a multiple of 8.",
        ge=128,
    )

    height: int | None = Field(
        None,
        description="The height of the generated image in pixels. Must be at least 128 and a multiple of 8.",
        ge=128,
    )

    secondary_prompt: str | None = Field(
        None,
        description="The second text prompt to generate images for.",
    )

    secondary_negative_prompt: str | None = Field(
        None,
        description="The second negative prompt to guide what NOT to generate.",
    )

    num_images: int = Field(
        1,
        description="The number of images to generate. Defaults to 1.",
        ge=1,
    )

    strength: float = Field(
        0.6,
        description=(
            "Image-to-image strength. Must be in (0, 1]. "
            "Higher values add more noise and preserve less from input image. "
            "Ignored for text-to-image requests."
        ),
        gt=0.0,
        le=1.0,
    )

    output_format: str = Field(
        "jpeg",
        description=(
            "The image format to use for encoding the output (e.g., 'jpeg', "
            "'png', 'webp'). Defaults to 'jpeg'."
        ),
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "ImageProviderOptions":
        for name, value in [("width", self.width), ("height", self.height)]:
            if value is not None and value % 8 != 0:
                raise ValueError(f"{name} must be a multiple of 8, got {value}")

        return self
