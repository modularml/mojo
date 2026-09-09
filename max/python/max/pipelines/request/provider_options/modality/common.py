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
"""Shared provider option types for generated media modalities."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class GeneratedMediaResponseFormat(str, Enum):
    """Response transport format for generated image/video outputs."""

    url = "url"
    b64_json = "b64_json"


class PixelProviderOptionsBase(BaseModel):
    """Generation options shared by all pixel-output modalities (image, video).

    Fields here apply to any text-to-pixels or pixels-to-pixels pipeline and
    are read by the pixel tokenizer regardless of which modality block carries
    them on the request. Modality-specific fields (e.g., ``num_frames`` for
    video, ``secondary_prompt`` for image) live on the per-modality subclasses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    negative_prompt: str | None = Field(
        None,
        description=(
            "A text description of what to exclude from the generated output. "
            "Used to guide the generation away from unwanted elements."
        ),
    )

    guidance_scale: float = Field(
        3.5,
        description=(
            "Guidance scale for classifier-free guidance. "
            "Higher values make the generation follow the prompt more closely. "
            "Set to 1.0 to disable CFG. Some distilled or guidance-light models "
            "may prefer lower values. Defaults to 3.5."
        ),
        ge=0.0,
    )

    true_cfg_scale: float = Field(
        1.0,
        description=(
            "True classifier-free guidance scale. "
            "True CFG is enabled when true_cfg_scale > 1.0 and negative_prompt is provided. "
            "Defaults to 1.0."
        ),
        gt=0.0,
    )

    width: int | None = Field(
        None,
        description="The width of the generated output in pixels.",
        gt=0,
    )

    height: int | None = Field(
        None,
        description="The height of the generated output in pixels.",
        gt=0,
    )

    steps: int | None = Field(
        None,
        description=(
            "The number of denoising steps. More steps generally produce "
            "higher quality results but take longer to generate. When unset, "
            "the model's pipeline-specific default is used."
        ),
        gt=0,
    )

    cfg_normalization: bool = Field(
        False,
        description=(
            "Enable CFG output renormalization when supported by the selected model. "
            "When enabled, the guided prediction norm is clipped to the positive "
            "prediction norm."
        ),
    )

    cfg_truncation: float = Field(
        1.0,
        description=(
            "CFG truncation threshold in normalized time when supported by the selected model. "
            "CFG is disabled for steps where t_norm > cfg_truncation."
        ),
        gt=0.0,
    )

    flow_shift: float | None = Field(
        None,
        description=(
            "Flow-matching scheduler shift parameter for the UniPC "
            "flow-sigma scheduler. Higher values push the noise schedule "
            "toward later timesteps and typically produce smoother "
            "long-horizon dynamics; lower values weight early timesteps "
            "more. When unset, the model's per-pipeline default is used. "
            "Ignored by schedulers that do not support it."
        ),
        gt=0.0,
    )

    response_format: GeneratedMediaResponseFormat = Field(
        GeneratedMediaResponseFormat.b64_json,
        description=(
            "How generated media is returned. Use 'url' for file-backed "
            "downloads or 'b64_json' for inline base64-encoded data."
        ),
    )
