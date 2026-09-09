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
"""Audio generation modality provider options."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SUPPORTED_AUDIO_FORMATS = frozenset({"wav"})
"""Containers the server can encode a generated waveform into."""


class AudioProviderOptions(BaseModel):
    """Options specific to audio generation pipelines.

    Audio does not share the pixel modalities' base: it has no width,
    height, or negative prompt, and the text it takes beyond the prompt is
    lyrics rather than a second caption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lyrics: str | None = Field(
        None,
        description=(
            "Lyrics for models that sing. Structure tags such as '[verse]' "
            "are model-specific and passed through as written."
        ),
    )

    audio_duration: float | None = Field(
        None,
        description=(
            "Upper bound on the generated audio in seconds. A model may stop "
            "earlier. When unset, the model's default duration is used."
        ),
        gt=0.0,
    )

    steps: int | None = Field(
        None,
        description=(
            "The number of denoising steps, for models whose audio comes from "
            "a diffusion or flow-matching stage. When unset, the model's "
            "default is used."
        ),
        gt=0,
    )

    guidance_scale: float | None = Field(
        None,
        description=(
            "Classifier-free guidance scale. When unset, the model's own "
            "scales apply -- audio models commonly guide their autoregressive "
            "and denoising stages at different strengths."
        ),
        gt=0.0,
    )

    audio_format: str = Field(
        "wav",
        description=(
            "Container for the returned audio. Only 'wav' is supported; the "
            "server has no encoder for anything else."
        ),
    )

    @field_validator("audio_format")
    @classmethod
    def _validate_audio_format(cls, value: str) -> str:
        """Rejects a container the server cannot write, and lowercases the rest.

        Checked when the request is admitted rather than when the waveform is
        persisted, because generating the audio comes first: a request naming
        an unsupported container would otherwise pay for a whole render and
        then fail on the way out, as a server error rather than as the bad
        request it was.

        Raises:
            ValueError: If the format is not one this server can encode.
        """
        audio_format = value.strip().lower()
        if audio_format not in _SUPPORTED_AUDIO_FORMATS:
            supported = ", ".join(
                f"'{name}'" for name in sorted(_SUPPORTED_AUDIO_FORMATS)
            )
            raise ValueError(
                f"audio_format '{value}' is not supported; this server "
                f"returns {supported}."
            )
        return audio_format
