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
"""Tests for the OpenAI-compatible /v1/audio/speech route."""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.pipelines.context import GenerationStatus
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    OutputAudioContent,
    OutputTextContent,
)
from max.serve.pipelines.general_handler import GeneralPipelineHandler
from max.serve.request import register_request
from max.serve.router import openai_routes

SAMPLE_RATE = 44100
CHANNELS = 2
FRAMES = 512

MODEL = "test-audio-model"


def _waveform() -> npt.NDArray[np.float32]:
    """A deterministic stereo ramp whose two channels differ.

    Channels that differ catch an interleave that transposes the wrong way,
    which a mono or silent waveform would let through.
    """
    ramp = np.linspace(-1.0, 1.0, FRAMES, dtype=np.float32)
    return np.stack([ramp, -ramp])


class MockAudioHandler(GeneralPipelineHandler):
    """Stands in for the worker, recording the request it was handed."""

    def __init__(self) -> None:
        self.model_name = MODEL
        self.logger = Mock()
        self.debug_logging = False
        self.seen: OpenResponsesRequest | None = None

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        self.seen = request
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputAudioContent.from_numpy_samples(
                    _waveform(), sample_rate=SAMPLE_RATE, format="wav"
                )
            ],
        )


class MockTextHandler(GeneralPipelineHandler):
    """A handler whose output carries no audio, as a text model's would."""

    def __init__(self) -> None:
        self.model_name = MODEL
        self.logger = Mock()
        self.debug_logging = False

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        yield GenerationOutput(
            request_id=request.request_id,
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputTextContent(text="not audio")],
        )


@pytest.fixture
def app() -> FastAPI:
    """An app serving the speech route against a mock audio worker."""
    app = FastAPI(title="MAX Serve Test")
    register_request(app)
    app.include_router(openai_routes.router)

    app.state.handler = MockAudioHandler()
    app.state.task = PipelineTask.AUDIO_GENERATION
    return app


def _speech_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "model": MODEL,
        "input": "[verse] the falsifiable line",
        "instructions": "a slow jazz ballad, upright bass",
    }
    request.update(overrides)
    return request


def test_speech_returns_wav(app: FastAPI) -> None:
    """A speech request returns a WAV body the stdlib can parse."""
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=_speech_request())

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    with wave.open(io.BytesIO(response.content), "rb") as container:
        assert container.getnchannels() == CHANNELS
        assert container.getframerate() == SAMPLE_RATE
        assert container.getsampwidth() == 2
        assert container.getnframes() == FRAMES
        decoded = np.frombuffer(
            container.readframes(FRAMES), dtype="<i2"
        ).reshape(FRAMES, CHANNELS)

    # 16-bit quantization is the only loss the round trip is allowed.
    expected = _waveform()
    np.testing.assert_allclose(
        decoded.T.astype(np.float32) / 32767, expected, atol=1 / 32767
    )


def test_speech_maps_input_to_lyrics_and_instructions_to_prompt(
    app: FastAPI,
) -> None:
    """``input`` reaches the model as lyrics, ``instructions`` as the prompt.

    Reversing the two would still return audio, so the mapping is checked on
    the request the handler received rather than on the response.
    """
    handler: MockAudioHandler = app.state.handler
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            json=_speech_request(
                audio_duration=12.0, steps=30, guidance_scale=3.0, seed=1234
            ),
        )

    assert response.status_code == 200
    assert handler.seen is not None
    body = handler.seen.body
    assert body.input == "a slow jazz ballad, upright bass"
    assert body.seed == 1234

    audio = body.provider_options.audio
    assert audio is not None
    assert audio.lyrics == "[verse] the falsifiable line"
    assert audio.audio_duration == 12.0
    assert audio.steps == 30
    assert audio.guidance_scale == 3.0


def test_speech_unset_options_stay_unset(app: FastAPI) -> None:
    """Options the request omits arrive as None, not as route defaults.

    The model's own defaults are the ones that should apply, so the route
    must not invent a duration or a step count on the way through.
    """
    handler: MockAudioHandler = app.state.handler
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=_speech_request())
    assert response.status_code == 200

    assert handler.seen is not None
    audio = handler.seen.body.provider_options.audio
    assert audio is not None
    assert audio.audio_duration is None
    assert audio.steps is None
    assert audio.guidance_scale is None
    assert handler.seen.body.seed is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"response_format": "mp3"}, "response_format"),
        ({"speed": 2.0}, "speed"),
        ({"stream_format": "sse"}, "Streaming"),
        ({"instructions": None}, "instructions"),
    ],
)
def test_speech_refuses_unsupported_options(
    app: FastAPI, overrides: dict[str, object], expected: str
) -> None:
    """An option this server cannot honor is refused, not ignored."""
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json=_speech_request(**overrides)
        )

    assert response.status_code == 400
    assert expected in response.json()["detail"]


def test_speech_accepts_explicit_wav_format(app: FastAPI) -> None:
    """``response_format: wav`` names what the server already returns."""
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json=_speech_request(response_format="wav")
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"


def test_speech_rejects_missing_input(app: FastAPI) -> None:
    """A request without the required fields fails validation.

    The route validates the raw body itself, as the other OpenAI routes do,
    so a schema violation is a 400 rather than FastAPI's own 422.
    """
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json={"model": MODEL, "instructions": "jazz"}
        )

    assert response.status_code == 400
    assert "input" in response.json()["detail"]


def test_speech_rejects_unknown_model(app: FastAPI) -> None:
    """Asking for a model this server does not serve is a client error."""
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json=_speech_request(model="other-model")
        )

    assert response.status_code == 400
    assert "currently serving" in response.json()["detail"]


def test_speech_unavailable_on_a_text_server(app: FastAPI) -> None:
    """The route is mounted everywhere, so a text server must refuse it."""
    app.state.task = PipelineTask.TEXT_GENERATION

    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=_speech_request())

    assert response.status_code == 404
    assert "not serving an audio generation model" in response.json()["detail"]


def test_speech_fails_when_the_model_returns_no_audio(app: FastAPI) -> None:
    """Output with no waveform is a server fault, not an empty WAV."""
    app.state.handler = MockTextHandler()

    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=_speech_request())

    assert response.status_code == 500
