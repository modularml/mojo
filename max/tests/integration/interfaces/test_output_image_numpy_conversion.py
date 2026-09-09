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
"""Tests for numpy to OutputImageContent conversion."""

import base64
from io import BytesIO

import numpy as np
import pytest
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.context.status import GenerationStatus
from max.pipelines.request import RequestID
from max.pipelines.request.open_responses import (
    ImageDetail,
    OutputImageContent,
)


def test_output_image_from_numpy_rgb_uint8() -> None:
    """Test converting RGB uint8 numpy array to OutputImageContent."""
    # Create a simple RGB image with values in [0, 255]
    rgb_array = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    output = OutputImageContent.from_numpy(rgb_array, format="png")

    assert output.type == "output_image"
    assert output.format == "png"
    assert output.image_data is not None
    assert output.image_url is None
    assert len(output.image_data) > 0

    # Verify it's valid base64
    base64.b64decode(output.image_data)


def test_output_image_from_numpy_grayscale() -> None:
    """Test converting grayscale numpy array to OutputImageContent."""
    # Create a grayscale image
    gray_array = (np.random.rand(100, 100) * 255).astype(np.uint8)

    output = OutputImageContent.from_numpy(gray_array, format="png")

    assert output.type == "output_image"
    assert output.format == "png"
    assert output.image_data is not None
    assert len(output.image_data) > 0


def test_output_image_from_numpy_rgba() -> None:
    """Test converting RGBA numpy array to OutputImageContent."""
    # Create an RGBA image with alpha channel
    rgba_array = (np.random.rand(100, 100, 4) * 255).astype(np.uint8)

    output = OutputImageContent.from_numpy(rgba_array, format="png")

    assert output.type == "output_image"
    assert output.format == "png"
    assert output.image_data is not None


def test_output_image_from_numpy_with_detail() -> None:
    """Test creating OutputImageContent with detail level."""
    rgb_array = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)

    output = OutputImageContent.from_numpy(
        rgb_array, format="png", detail=ImageDetail.high
    )

    assert output.detail == ImageDetail.high


def test_output_image_from_numpy_different_formats() -> None:
    """Test creating OutputImageContent with different image formats."""
    rgb_array = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)

    for format_str in ["png", "jpeg", "webp"]:
        output = OutputImageContent.from_numpy(rgb_array, format=format_str)
        assert output.format == format_str.lower()


def test_output_image_from_numpy_invalid_shape() -> None:
    """Test that invalid array shapes raise ValueError."""
    # 1D array should fail
    invalid_array = np.random.randint(0, 255, size=100, dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        OutputImageContent.from_numpy(invalid_array)

    # 4D array should fail
    invalid_array_4d = np.random.randint(
        0, 255, size=(10, 10, 10, 3), dtype=np.uint8
    )

    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        OutputImageContent.from_numpy(invalid_array_4d)


def test_output_image_from_numpy_invalid_channels() -> None:
    """Test that invalid number of channels raises ValueError."""
    # 5 channels should fail
    invalid_channels = np.random.randint(
        0, 255, size=(50, 50, 5), dtype=np.uint8
    )

    with pytest.raises(ValueError, match="Unsupported number of channels"):
        OutputImageContent.from_numpy(invalid_channels)


def test_generation_output_with_numpy_images() -> None:
    """Test creating GenerationOutput with numpy-converted images."""
    # Create multiple images
    img1 = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    img2 = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

    generation_output = GenerationOutput(
        request_id=RequestID(value="test-request-123"),
        final_status=GenerationStatus.END_OF_SEQUENCE,
        output=[
            OutputImageContent.from_numpy(img1, format="png"),
            OutputImageContent.from_numpy(img2, format="jpeg"),
        ],
    )

    assert generation_output.request_id.value == "test-request-123"
    assert generation_output.final_status == GenerationStatus.END_OF_SEQUENCE
    assert len(generation_output.output) == 2

    # Narrow types for mypy
    output_0 = generation_output.output[0]
    output_1 = generation_output.output[1]
    assert isinstance(output_0, OutputImageContent)
    assert isinstance(output_1, OutputImageContent)

    assert output_0.format == "png"
    assert output_1.format == "jpeg"
    assert generation_output.is_done is True


def test_generation_output_not_done() -> None:
    """Test GenerationOutput is_done property with ACTIVE status."""
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)

    generation_output = GenerationOutput(
        request_id=RequestID(value="active-request"),
        final_status=GenerationStatus.ACTIVE,
        output=[OutputImageContent.from_numpy(img)],
    )

    assert generation_output.is_done is False


def test_output_image_base64_decode() -> None:
    """Test that generated base64 can be decoded back to an image."""
    pytest.importorskip("PIL")
    from PIL import Image

    # Create a known pattern
    rgb_array = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb_array[:25, :, 0] = 255  # Red top half

    output = OutputImageContent.from_numpy(rgb_array, format="png")

    # Decode and verify
    assert output.image_data is not None
    image_bytes = base64.b64decode(output.image_data)
    image = Image.open(BytesIO(image_bytes))

    assert image.size == (50, 50)
    assert image.mode == "RGB"
