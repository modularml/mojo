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
"""Tests for provider options schemas."""

import json

import pytest
from max.pipelines.request.provider_options import (
    AudioProviderOptions,
    ImageProviderOptions,
    MaxProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from pydantic import ValidationError


def test_import_provider_options() -> None:
    """Test that all provider option types can be imported."""
    # If we get here, all imports succeeded
    assert MaxProviderOptions is not None
    assert ImageProviderOptions is not None
    assert ProviderOptions is not None


def test_max_provider_options_minimal() -> None:
    """Test creating MaxProviderOptions with no fields."""
    opts = MaxProviderOptions()
    assert opts.target_endpoint is None


def test_max_provider_options_with_target_endpoint() -> None:
    """Test creating MaxProviderOptions with target_endpoint."""
    opts = MaxProviderOptions(target_endpoint="instance-123")
    assert opts.target_endpoint == "instance-123"


def test_max_provider_options_frozen() -> None:
    """Test that MaxProviderOptions is frozen (immutable)."""
    opts = MaxProviderOptions(target_endpoint="instance-123")

    with pytest.raises(ValidationError):
        opts.target_endpoint = "instance-456"  # type: ignore[misc]


def test_image_provider_options_minimal() -> None:
    """Test creating ImageProviderOptions with no fields."""
    opts = ImageProviderOptions()
    assert opts.negative_prompt is None
    assert opts.width is None
    assert opts.height is None


def test_image_provider_options_with_param() -> None:
    """Test creating ImageProviderOptions with parameters."""
    opts = ImageProviderOptions(width=1024, height=768, num_images=2)
    assert opts.width == 1024
    assert opts.height == 768
    assert opts.num_images == 2


def test_image_provider_options_frozen() -> None:
    """Test that ImageProviderOptions is frozen (immutable)."""
    opts = ImageProviderOptions(width=512, height=512)

    with pytest.raises(ValidationError):
        opts.width = 1024  # type: ignore[misc]


def test_video_provider_options_guidance_scale() -> None:
    """VideoProviderOptions exposes guidance_scale inherited from the base."""
    # Inherits the modality-shared default (3.5) from PixelProviderOptionsBase.
    opts = VideoProviderOptions()
    assert opts.guidance_scale == 3.5

    # Accepts explicit values.
    opts = VideoProviderOptions(guidance_scale=5.0)
    assert opts.guidance_scale == 5.0

    # Negative values are rejected (ge=0.0).
    with pytest.raises(ValidationError):
        VideoProviderOptions(guidance_scale=-1.0)


def test_video_provider_options_flow_shift() -> None:
    """VideoProviderOptions exposes flow_shift inherited from the base."""
    # Default is None (defer to the pipeline's per-model default).
    opts = VideoProviderOptions()
    assert opts.flow_shift is None

    # Accepts explicit positive values.
    opts = VideoProviderOptions(flow_shift=5.0)
    assert opts.flow_shift == 5.0

    # gt=0.0: zero and negative values are rejected.
    with pytest.raises(ValidationError):
        VideoProviderOptions(flow_shift=0.0)
    with pytest.raises(ValidationError):
        VideoProviderOptions(flow_shift=-1.0)


def test_image_provider_options_flow_shift_inherited() -> None:
    """flow_shift is exposed on ImageProviderOptions via the shared base."""
    opts = ImageProviderOptions()
    assert opts.flow_shift is None

    opts = ImageProviderOptions(flow_shift=3.0)
    assert opts.flow_shift == 3.0


def test_provider_options_empty() -> None:
    """Test creating ProviderOptions with no fields."""
    opts = ProviderOptions()
    assert opts.max is None
    assert opts.image is None
    assert opts.video is None


def test_provider_options_with_max_only() -> None:
    """Test creating ProviderOptions with only MAX options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is None
    assert opts.video is None


def test_provider_options_with_image_only() -> None:
    """Test creating ProviderOptions with only image modality options."""
    opts = ProviderOptions(image=ImageProviderOptions(width=512, height=512))
    assert opts.max is None
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512
    assert opts.video is None


def test_provider_options_with_all_fields() -> None:
    """Test creating ProviderOptions with both MAX and modality options."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123"),
        image=ImageProviderOptions(width=512, height=512),
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512


def test_provider_options_frozen() -> None:
    """Test that ProviderOptions is frozen (immutable)."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )

    with pytest.raises(ValidationError):
        opts.max = MaxProviderOptions(target_endpoint="instance-456")  # type: ignore[misc]


def test_provider_options_json_serialization() -> None:
    """Test that ProviderOptions can be serialized to JSON."""
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123"),
        image=ImageProviderOptions(width=1024, height=768),
    )

    json_str = opts.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["max"]["target_endpoint"] == "instance-123"
    assert json_data["image"]["width"] == 1024
    assert json_data["image"]["height"] == 768


def test_provider_options_json_deserialization() -> None:
    """Test that ProviderOptions can be deserialized from JSON."""
    json_data = {
        "max": {"target_endpoint": "instance-123"},
        "image": {"width": 512, "height": 512},
    }

    opts = ProviderOptions.model_validate(json_data)

    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512


def test_provider_options_json_deserialization_partial() -> None:
    """Test deserializing ProviderOptions with only some fields."""
    # Only MAX options
    max_json_data = {"max": {"target_endpoint": "instance-123"}}
    opts = ProviderOptions.model_validate(max_json_data)
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"
    assert opts.image is None
    assert opts.video is None

    # Only image options
    image_json_data = {"image": {"width": 512, "height": 512}}
    opts = ProviderOptions.model_validate(image_json_data)
    assert opts.max is None
    assert opts.image is not None
    assert opts.image.width == 512
    assert opts.image.height == 512
    assert opts.video is None


def test_provider_options_nested_validation() -> None:
    """Test that nested validation works correctly."""
    # Valid nested structure with explicit type
    opts = ProviderOptions(
        max=MaxProviderOptions(target_endpoint="instance-123")
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-123"

    # Valid nested structure from dict (Pydantic auto-converts)
    opts = ProviderOptions.model_validate(
        {"max": {"target_endpoint": "instance-456"}}
    )
    assert opts.max is not None
    assert opts.max.target_endpoint == "instance-456"

    # Invalid nested structure should fail at creation
    with pytest.raises(ValidationError):
        ProviderOptions.model_validate({"max": {"invalid_field": "value"}})


class TestImageDimensionValidation:
    """Tests for height/width validation on ImageProviderOptions.

    Pixel-area limits are enforced per-architecture via context validators
    (see ``max.pipelines.context.pixel_context_validators``); the schema only
    enforces a minimum size and multiple-of-8 (the VAE spatial scale factor).
    Per-model tokenizers handle any further patchification-related rounding.
    """

    def test_dimensions_none_is_valid(self) -> None:
        opts = ImageProviderOptions()
        assert opts.width is None
        assert opts.height is None

    def test_valid_dimensions(self) -> None:
        opts = ImageProviderOptions(width=512, height=512)
        assert opts.width == 512
        assert opts.height == 512

    def test_minimum_dimensions(self) -> None:
        opts = ImageProviderOptions(width=128, height=128)
        assert opts.width == 128
        assert opts.height == 128

    def test_width_too_small(self) -> None:
        with pytest.raises(
            ValidationError, match="greater than or equal to 128"
        ):
            ImageProviderOptions(width=64, height=512)

    def test_height_too_small(self) -> None:
        with pytest.raises(
            ValidationError, match="greater than or equal to 128"
        ):
            ImageProviderOptions(width=512, height=64)

    def test_width_not_multiple_of_8(self) -> None:
        with pytest.raises(ValidationError, match="must be a multiple of 8"):
            ImageProviderOptions(width=130, height=512)

    def test_height_not_multiple_of_8(self) -> None:
        with pytest.raises(ValidationError, match="must be a multiple of 8"):
            ImageProviderOptions(width=512, height=130)

    def test_height_1080_accepted(self) -> None:
        # 1080 is a multiple of 8 but not 16; the API must accept it so that
        # callers can request 1080p workloads (e.g., for Wan video benchmarks).
        # Per-model tokenizers round to the nearest patchification-compatible
        # size as needed.
        opts = ImageProviderOptions(width=1920, height=1080)
        assert opts.width == 1920
        assert opts.height == 1080

    def test_multiple_of_8_but_not_16_accepted(self) -> None:
        # 1080 = 8 * 135 is allowed; 1088 = 16 * 68 is also allowed.
        opts = ImageProviderOptions(width=1080, height=1080)
        assert opts.width == 1080
        assert opts.height == 1080

    def test_large_dimensions_accepted_at_schema_layer(self) -> None:
        opts = ImageProviderOptions(width=4096, height=4096)
        assert opts.width == 4096
        assert opts.height == 4096


class TestAudioFormatValidation:
    """Tests for the container AudioProviderOptions accepts.

    Only WAV can be encoded, and generating the audio comes before writing
    it, so an unsupported container has to be caught here rather than after a
    render has already been paid for.
    """

    def test_default_is_wav(self) -> None:
        assert AudioProviderOptions().audio_format == "wav"

    def test_wav_is_accepted(self) -> None:
        assert AudioProviderOptions(audio_format="wav").audio_format == "wav"

    def test_case_and_padding_are_normalized(self) -> None:
        opts = AudioProviderOptions(audio_format=" WAV ")
        assert opts.audio_format == "wav"

    @pytest.mark.parametrize("audio_format", ["mp3", "flac", "opus", ""])
    def test_an_unsupported_container_is_rejected(
        self, audio_format: str
    ) -> None:
        with pytest.raises(ValidationError, match="is not supported"):
            AudioProviderOptions(audio_format=audio_format)

    def test_the_error_names_what_is_supported(self) -> None:
        with pytest.raises(ValidationError, match="'wav'"):
            AudioProviderOptions(audio_format="mp3")
