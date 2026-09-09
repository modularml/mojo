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
"""Tests for OpenResponses API routes."""

import io
import logging
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.parse import urlparse

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.pipelines.context import (
    BaseContext,
    GenerationStatus,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import (
    PipelineTokenizer,
    RequestID,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    OutputAudioContent,
    OutputImageContent,
    OutputVideoContent,
)
from max.serve.media import GeneratedMediaStore
from max.serve.pipelines.general_handler import GeneralPipelineHandler
from max.serve.request import register_request
from max.serve.router import openresponses_routes

logger = logging.getLogger(__name__)


@dataclass
class MockOpenResponsesTokenizer(
    PipelineTokenizer[BaseContext, Any, OpenResponsesRequest]
):
    """Mock tokenizer for OpenResponses requests."""

    @property
    def eos_token_ids(self) -> set[int]:
        """Mock tokenizer has no EOS tokens."""
        return set()

    @property
    def expects_content_wrapping(self) -> bool:
        """Mock tokenizer doesn't require content wrapping."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> Any:
        """Mock encode method."""
        return np.array([ord(char) for char in prompt], dtype=np.int64)

    async def decode(self, encoded: Any, **kwargs) -> str:
        """Mock decode method."""
        return "mock decoded text"

    async def new_context(self, request: OpenResponsesRequest) -> BaseContext:
        """Creates a mock BaseContext for OpenResponses requests."""

        # Create a minimal mock BaseContext
        @dataclass
        class MockContext(BaseContext):
            """Minimal mock context for testing."""

            request_id: RequestID
            status: GenerationStatus = GenerationStatus.ACTIVE

        return MockContext(request_id=request.request_id)


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to return BaseContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
    ) -> type[BaseContext]:
        return BaseContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


class MockGeneralPipelineHandler(GeneralPipelineHandler):
    """Mock implementation of GeneralPipelineHandler for testing."""

    def __init__(self) -> None:
        # Skip the parent constructor that requires real dependencies
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        """Mock implementation that yields a simple text response."""
        # Create a simple mock response
        pixel_data = np.array([[1, 2, 3]], dtype=np.uint8)
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputImageContent.from_numpy(pixel_data, format="png")],
        )


class MockVideoPipelineHandler(GeneralPipelineHandler):
    """Mock implementation that yields multiple frames for a video response."""

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_a[:, :, 0] = 255
        frame_b = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_b[:, :, 1] = 255
        frames = np.stack([frame_a, frame_b], axis=0)
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputVideoContent.from_numpy_frames(frames)],
        )


@pytest.fixture(scope="function")
def app(
    fixture_tokenizer: Any,
    mock_pipeline_config: PipelineConfig,
    tmp_path: Path,
) -> FastAPI:
    """Create a test app with OpenResponses API enabled."""
    _ = fixture_tokenizer
    _ = mock_pipeline_config

    # Create a minimal FastAPI app without the full lifespan
    app = FastAPI(title="MAX Serve Test")

    # Register request middleware to add request_id to request.state
    register_request(app)

    # Register the OpenResponses routes
    app.include_router(openresponses_routes.router)

    # Inject a mock handler into app.state
    app.state.handler = MockGeneralPipelineHandler()
    app.state.media_store = GeneratedMediaStore(tmp_path / "media")
    app.state.pipeline_config = Mock(model=Mock(model_name="test-model"))

    return app


def test_openresponses_simple_request(app) -> None:  # noqa: ANN001
    """Test a simple OpenResponses request with string input."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Generate an image of a cat",
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        response_json = response.json()

        # Check response structure
        assert "id" in response_json
        assert response_json["id"].startswith("resp_")
        assert response_json["object"] == "response"
        assert response_json["status"] == "completed"
        assert response_json["model"] == "test-model"
        assert "output" in response_json
        assert len(response_json["output"]) > 0

        # Check message structure
        message = response_json["output"][0]
        assert "id" in message
        assert message["id"].startswith("msg_")
        assert message["role"] == "assistant"
        assert "content" in message


def test_openresponses_message_list_input(app) -> None:  # noqa: ANN001
    """Test OpenResponses request with message list input."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "What is 2+2?"},
            ],
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "completed"


def test_openresponses_streaming_not_supported(app) -> None:  # noqa: ANN001
    """Test that streaming requests are rejected during validation."""
    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Test",
            "stream": True,
        }
        response = client.post("/v1/responses", json=request_data)

        # Streaming validation happens during Pydantic validation, so returns 422
        assert response.status_code == 422  # UNPROCESSABLE_ENTITY
        assert "not currently supported" in response.json()["detail"].lower()


def test_openresponses_invalid_request(app) -> None:  # noqa: ANN001
    """Test that invalid requests return appropriate errors."""
    with TestClient(app) as client:
        # Missing required 'model' field
        request_data = {
            "input": "Test",
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 422  # UNPROCESSABLE_ENTITY


def test_openresponses_rejects_unknown_model(app: FastAPI) -> None:
    """Requests should fail when the requested model differs from the served model."""
    with TestClient(app) as client:
        request_data = {
            "model": "other-model",
            "input": "Generate an image of a cat",
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 404
        assert "currently serving" in response.json()["detail"]


def test_openresponses_video_request_returns_download_url(
    app: FastAPI,
) -> None:
    """Video requests should return a downloadable mp4 URL."""
    app.state.handler = MockVideoPipelineHandler()

    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Animate a red square becoming green.",
            "provider_options": {
                "video": {
                    "width": 64,
                    "height": 64,
                    "num_frames": 2,
                    "frames_per_second": 8,
                    "steps": 2,
                }
            },
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        content = response.json()["output"][0]["content"][0]
        assert content["type"] == "output_video"
        assert content["format"] == "mp4"
        assert content["frames_per_second"] == 8
        assert content["num_frames"] == 2

        video_path = urlparse(content["video_url"]).path
        video_response = client.get(video_path)
        assert video_response.status_code == 200
        assert video_response.headers["content-type"] == "video/mp4"


def test_openresponses_video_request_returns_inline_base64_when_requested(
    app: FastAPI,
) -> None:
    """Video requests should return inline base64 mp4 when b64_json is requested."""
    app.state.handler = MockVideoPipelineHandler()

    with TestClient(app) as client:
        request_data = {
            "model": "test-model",
            "input": "Animate a red square becoming green.",
            "provider_options": {
                "video": {
                    "width": 64,
                    "height": 64,
                    "num_frames": 2,
                    "frames_per_second": 8,
                    "steps": 2,
                    "response_format": "b64_json",
                }
            },
        }
        response = client.post("/v1/responses", json=request_data)

        assert response.status_code == 200
        content = response.json()["output"][0]["content"][0]
        assert content["type"] == "output_video"
        assert content["format"] == "mp4"
        assert content["frames_per_second"] == 8
        assert content["num_frames"] == 2
        assert content["video_data"]
        assert "video_url" not in content


class MockAudioPipelineHandler(GeneralPipelineHandler):
    """Mock implementation that yields a raw waveform, as an audio model does."""

    SAMPLE_RATE = 44100
    FRAMES = 256

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        ramp = np.linspace(-1.0, 1.0, self.FRAMES, dtype=np.float32)
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputAudioContent.from_numpy_samples(
                    np.stack([ramp, -ramp]),
                    sample_rate=self.SAMPLE_RATE,
                    format="wav",
                )
            ],
        )


def test_openresponses_audio_request_returns_download_url(
    app: FastAPI,
) -> None:
    """Audio requests should return a downloadable WAV URL, not raw samples."""
    app.state.handler = MockAudioPipelineHandler()

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "a slow jazz ballad, upright bass",
                "provider_options": {"audio": {"lyrics": "[verse] hello"}},
            },
        )

        assert response.status_code == 200
        content = response.json()["output"][0]["content"][0]
        assert content["type"] == "output_audio"
        assert content["format"] == "wav"
        assert content["sample_rate"] == MockAudioPipelineHandler.SAMPLE_RATE
        assert content["num_samples"] == MockAudioPipelineHandler.FRAMES
        assert "samples" not in content

        audio_path = urlparse(content["audio_url"]).path
        audio_response = client.get(audio_path)
        assert audio_response.status_code == 200
        assert audio_response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(audio_response.content), "rb") as container:
            assert container.getnchannels() == 2
            assert container.getframerate() == (
                MockAudioPipelineHandler.SAMPLE_RATE
            )
            assert container.getnframes() == MockAudioPipelineHandler.FRAMES


class CountingAudioPipelineHandler(MockAudioPipelineHandler):
    """Records whether the model was ever asked to render."""

    def __init__(self) -> None:
        super().__init__()
        self.renders = 0

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        self.renders += 1
        async for output in super().next(request):
            yield output


def test_openresponses_audio_request_rejects_an_unwritable_format(
    app: FastAPI,
) -> None:
    """An unsupported container is refused before anything is generated.

    Only WAV can be written, and the waveform is persisted after the model
    has run, so leaving this to the media store would spend a full render and
    then report a server error for what was a bad request.
    """
    handler = CountingAudioPipelineHandler()
    app.state.handler = handler

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "a slow jazz ballad, upright bass",
                "provider_options": {"audio": {"audio_format": "mp3"}},
            },
        )

    assert response.status_code == 422
    assert "audio_format" in response.text
    assert handler.renders == 0


def test_openresponses_audio_request_accepts_wav_in_any_case(
    app: FastAPI,
) -> None:
    app.state.handler = MockAudioPipelineHandler()

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "a slow jazz ballad, upright bass",
                "provider_options": {"audio": {"audio_format": "WAV"}},
            },
        )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["format"] == "wav"


class _PromptTooLongPipelineHandler(GeneralPipelineHandler):
    """Mock handler whose generator raises ``PromptTooLongError`` on first
    iteration — simulating a tokenizer that rejected an over-length prompt."""

    def __init__(self) -> None:
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        raise PromptTooLongError(
            num_tokens=4033,
            max_length=512,
            limit_description="text encoder's maximum sequence length",
        )
        yield  # pragma: no cover -- unreachable, marks this as a generator


def test_openresponses_prompt_too_long_returns_400(app: FastAPI) -> None:
    """A ``PromptTooLongError`` from the handler must surface as 400 with
    the exception's message intact, not propagate to a generic 500."""
    app.state.handler = _PromptTooLongPipelineHandler()

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "irrelevant"},
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Prompt is too long" in detail
    assert "4033 tokens" in detail
    assert "text encoder's maximum sequence length" in detail
    assert "512 tokens" in detail
