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
"""Provider-specific options for MAX platform and modalities."""

from pydantic import BaseModel, ConfigDict, Field

from .max import MaxProviderOptions
from .modality import (
    AudioProviderOptions,
    GeneratedMediaResponseFormat,
    ImageProviderOptions,
    PixelProviderOptionsBase,
    VideoProviderOptions,
)


class ProviderOptions(BaseModel):
    """Container for all provider-specific options.

    Includes both universal MAX options and modality-specific options.
    All options are validated at the API layer.

    Example:

    .. code-block:: json

        {
            "max": {"target_endpoint": "instance-123"},
            "image": {"width": 1024, "height": 768}
        }
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max: MaxProviderOptions | None = Field(
        None,
        description="Universal MAX platform options.",
    )

    image: ImageProviderOptions | None = Field(
        None,
        description="Image generation modality options.",
    )

    video: VideoProviderOptions | None = Field(
        None,
        description="Video generation modality options.",
    )

    audio: AudioProviderOptions | None = Field(
        None,
        description="Audio generation modality options.",
    )

    # Add more modality fields here as needed:
    # text: TextModalityOptions | None = None


__all__ = [
    "AudioProviderOptions",
    "GeneratedMediaResponseFormat",
    "ImageProviderOptions",
    "MaxProviderOptions",
    "PixelProviderOptionsBase",
    "ProviderOptions",
    "VideoProviderOptions",
]
