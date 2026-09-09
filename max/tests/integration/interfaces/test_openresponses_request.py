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
"""Tests for OpenResponsesRequest factory methods."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from max.pipelines.request import (
    InputImageContent,
    OpenResponsesRequest,
    OpenResponsesRequestBody,
    RequestID,
)
from pydantic import ValidationError


@pytest.fixture
def mock_fastapi_request() -> Any:
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.state = MagicMock()
    request.state.request_id = "test-request-id-123"
    return request


@pytest.fixture
def minimal_request_body() -> str:
    """Create a minimal OpenResponsesRequestBody JSON."""
    return json.dumps(
        {
            "model": "test-model",
            "input": "Generate an image of a cat",
        }
    )


@pytest.fixture
def full_request_body() -> str:
    """Create a full OpenResponsesRequestBody JSON with provider options."""
    return json.dumps(
        {
            "model": "test-model",
            "input": "Generate an image of a cat",
            "seed": 42,
            "provider_options": {
                "image": {
                    "width": 1024,
                    "height": 768,
                    "negative_prompt": "blurry, low quality",
                    "guidance_scale": 7.5,
                    "steps": 50,
                }
            },
        }
    )


@pytest.mark.asyncio
async def test_from_fastapi_request_minimal(
    mock_fastapi_request: Any, minimal_request_body: str
) -> None:
    """Test creating OpenResponsesRequest from minimal FastAPI request."""
    mock_fastapi_request.body = AsyncMock(
        return_value=minimal_request_body.encode()
    )

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    assert request.request_id.value == "test-request-id-123"
    assert request.body.model == "test-model"
    assert request.body.input == "Generate an image of a cat"
    assert request.body.seed is None
    # provider_options defaults to ProviderOptions with default ImageProviderOptions
    assert request.body.provider_options is not None
    assert request.body.provider_options.image is not None
    assert request.body.provider_options.image.guidance_scale == 3.5


@pytest.mark.asyncio
async def test_from_fastapi_request_with_provider_options(
    mock_fastapi_request: Any, full_request_body: str
) -> None:
    """Test creating OpenResponsesRequest with full provider options."""
    mock_fastapi_request.body = AsyncMock(
        return_value=full_request_body.encode()
    )

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    assert request.request_id.value == "test-request-id-123"
    assert request.body.model == "test-model"
    assert request.body.input == "Generate an image of a cat"
    assert request.body.seed == 42

    # Check provider options
    assert request.body.provider_options is not None
    assert request.body.provider_options.image is not None
    assert request.body.provider_options.image.width == 1024
    assert request.body.provider_options.image.height == 768
    assert (
        request.body.provider_options.image.negative_prompt
        == "blurry, low quality"
    )
    assert request.body.provider_options.image.guidance_scale == 7.5
    assert request.body.provider_options.image.steps == 50


@pytest.mark.asyncio
async def test_from_fastapi_request_missing_request_id(
    minimal_request_body: str,
) -> None:
    """Test that missing request_id raises ValueError."""
    mock_request = MagicMock()
    mock_request.state = MagicMock(spec=[])  # state exists but no request_id
    mock_request.body = AsyncMock(return_value=minimal_request_body.encode())

    with pytest.raises(
        ValueError, match=r"request\.state\.request_id not found"
    ):
        await OpenResponsesRequest.from_fastapi_request(mock_request)


@pytest.mark.asyncio
async def test_from_fastapi_request_invalid_body(
    mock_fastapi_request: Any,
) -> None:
    """Test that invalid JSON body raises ValidationError."""
    invalid_body = json.dumps({"invalid": "missing required fields"})
    mock_fastapi_request.body = AsyncMock(return_value=invalid_body.encode())

    with pytest.raises(ValidationError):
        await OpenResponsesRequest.from_fastapi_request(mock_fastapi_request)


@pytest.mark.asyncio
async def test_from_fastapi_request_with_message_input(
    mock_fastapi_request: Any,
) -> None:
    """Test creating OpenResponsesRequest with message list input."""
    message_body = json.dumps(
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Generate an image of a cat"},
            ],
        }
    )
    mock_fastapi_request.body = AsyncMock(return_value=message_body.encode())

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    assert request.request_id.value == "test-request-id-123"
    assert request.body.model == "test-model"
    assert isinstance(request.body.input, list)
    assert len(request.body.input) == 1
    assert request.body.input[0].role == "user"
    assert request.body.input[0].content == "Generate an image of a cat"


@pytest.mark.asyncio
async def test_from_fastapi_request_preserves_all_fields(
    mock_fastapi_request: Any,
) -> None:
    """Test that all OpenResponsesRequestBody fields are preserved."""
    complex_body = json.dumps(
        {
            "model": "test-model",
            "input": "Test prompt",
            "seed": 42,
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "stop": ["END"],
            "provider_options": {
                "image": {
                    "width": 512,
                    "height": 512,
                    "secondary_prompt": "detailed",
                    "true_cfg_scale": 2.0,
                }
            },
        }
    )
    mock_fastapi_request.body = AsyncMock(return_value=complex_body.encode())

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    # Verify all fields are accessible through body
    assert request.body.seed == 42
    assert request.body.temperature == 0.8
    assert request.body.max_output_tokens == 1024
    assert request.body.stop == ["END"]
    assert request.body.provider_options is not None
    assert request.body.provider_options.image is not None
    assert request.body.provider_options.image.secondary_prompt == "detailed"
    assert request.body.provider_options.image.true_cfg_scale == 2.0


def test_openresponses_request_is_frozen() -> None:
    """Test that OpenResponsesRequest is immutable."""
    body = OpenResponsesRequestBody(model="test", input="test")
    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    # Frozen dataclass raises AttributeError or FrozenInstanceError on assignment
    with pytest.raises((AttributeError, Exception)):
        request.body = OpenResponsesRequestBody(  # type: ignore[misc]
            model="changed", input="changed"
        )


def test_openresponses_request_inherits_from_request() -> None:
    """Test that OpenResponsesRequest has request_id field."""
    body = OpenResponsesRequestBody(model="test", input="test")
    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    # Verify it has a request_id field
    assert hasattr(request, "request_id")
    assert isinstance(request.request_id, RequestID)
    # Verify it has a __str__ method that returns the request_id
    assert str(request) == str(request.request_id)


# ---------------------------------------------------------------------------
# Web image URLs are fetched through the server-supplied fetcher.
#
# InputImageContent accepts data: URIs only, so an http(s) image must be inlined
# before the body validates. This library owns no downloader of its own -- the
# byte caps and host validation live in the injected fetcher -- so a web URL with
# no fetcher installed is rejected rather than fetched unguarded.
# ---------------------------------------------------------------------------

_TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _image_request_body(url: str) -> str:
    return json.dumps(
        {
            "model": "test-model",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": url},
                    ],
                }
            ],
        }
    )


def _image_part(request: OpenResponsesRequest) -> InputImageContent:
    """Narrow the parsed body down to its one image content part."""
    assert isinstance(request.body.input, list)
    content = request.body.input[0].content
    assert isinstance(content, list)
    part = content[1]
    assert isinstance(part, InputImageContent)
    return part


@pytest.mark.asyncio
async def test_web_image_url_fetched_through_injected_fetcher(
    mock_fastapi_request: Any,
) -> None:
    """A web URL is handed to the server's fetcher and replaced by its result."""
    fetcher = AsyncMock(return_value=_TINY_PNG_DATA_URI)
    mock_fastapi_request.app.state.media_data_uri_fetcher = fetcher
    mock_fastapi_request.body = AsyncMock(
        return_value=_image_request_body("https://example.com/cat.png").encode()
    )

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    fetcher.assert_awaited_once_with("https://example.com/cat.png")
    assert _image_part(request).image_url == _TINY_PNG_DATA_URI


@pytest.mark.asyncio
async def test_web_image_url_rejected_when_no_fetcher_installed(
    mock_fastapi_request: Any,
) -> None:
    """Without a fetcher a web URL is a clean error, never an unguarded fetch."""
    mock_fastapi_request.app.state = MagicMock(spec=[])
    mock_fastapi_request.body = AsyncMock(
        return_value=_image_request_body(
            "http://169.254.169.254/latest"
        ).encode()
    )

    with pytest.raises(ValueError, match="data: URI"):
        await OpenResponsesRequest.from_fastapi_request(mock_fastapi_request)


@pytest.mark.asyncio
async def test_data_uri_image_is_not_fetched(
    mock_fastapi_request: Any,
) -> None:
    """A data URI passes through untouched, with no fetch attempted."""
    fetcher = AsyncMock()
    mock_fastapi_request.app.state.media_data_uri_fetcher = fetcher
    mock_fastapi_request.body = AsyncMock(
        return_value=_image_request_body(_TINY_PNG_DATA_URI).encode()
    )

    request = await OpenResponsesRequest.from_fastapi_request(
        mock_fastapi_request
    )

    fetcher.assert_not_awaited()
    assert _image_part(request).image_url == _TINY_PNG_DATA_URI


@pytest.mark.asyncio
async def test_fetcher_error_propagates_unchanged(
    mock_fastapi_request: Any,
) -> None:
    """The fetcher's own message reaches the client, not a re-wrapped one.

    The previous implementation caught the network error and echoed ``str(e)``,
    which distinguished refused from reachable-but-not-an-image and turned the
    endpoint into a host scanner.
    """
    mock_fastapi_request.app.state.media_data_uri_fetcher = AsyncMock(
        side_effect=ValueError("image exceeds the maximum allowed size of 10MB")
    )
    mock_fastapi_request.body = AsyncMock(
        return_value=_image_request_body("https://example.com/big.png").encode()
    )

    with pytest.raises(ValueError, match="exceeds the maximum") as excinfo:
        await OpenResponsesRequest.from_fastapi_request(mock_fastapi_request)
    assert "example.com" not in str(excinfo.value)
