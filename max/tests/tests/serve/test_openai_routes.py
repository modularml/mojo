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


import asyncio
import base64
import io
import json
import logging
import sys
from collections.abc import Generator
from threading import Thread
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient as AsyncTestClient
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient as SyncTestClient
from max.pipelines.architectures.kimik2_5.tokenizer import (
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
)
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.architectures.qwen3_5.tool_parser import Qwen3_5ToolParser
from max.pipelines.context import (
    BaseContext,
    GenerationStatus,
    TextContext,
    TextGenerationResponseFormat,
)
from max.pipelines.context.exceptions import InputError, PromptTooLongError
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.lib.tokenizer import open_image
from max.pipelines.modeling.types import (
    ParsedToolCallDelta,
    ParsedToolResponse,
    PipelineTask,
    RequestID,
    TextGenerationRequestTool,
)
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.parser import LlamaToolParser
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorOutput, TokenGeneratorPipeline
from max.serve.router._image_resolution import (
    _decode_data_uri_base64,
    decode_and_validate_images,
    resolve_image_from_url,
)
from max.serve.router.openai_routes import (
    CompletionStreamResponse,
    OpenAIChatResponseGenerator,
    OpenAICompletionResponseGenerator,
    _batch_id,
    _coerce_positive_float,
    _coerce_positive_int,
    _create_response_format,
    _get_cache_salt,
    _process_chat_log_probabilities,
    _resolve_grammar_constraints,
    _set_batch_id_attributes,
    get_tool_parser,
    openai_create_chat_completion,
    openai_parse_chat_completion_request,
)
from max.serve.schemas.openai import (
    ChatCompletionLogprobs,
    ChatCompletionMessageToolCall,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    ChatCompletionTokenLogprob,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from PIL import Image
from pydantic import AnyUrl, ValidationError

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    from taskgroup import TaskGroup

logger = logging.getLogger(__name__)

# FIXME: SERVSYS-1275 — this suite runs for many minutes on contended macOS CI
# workers and has timed out at 1200s, pushing the macOS job past its 2h cap.
# Skip on macOS until the serve-test slowness is addressed.
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="SERVSYS-1275: too slow on macOS CI; exceeds the 2h job timeout",
)


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@pytest_asyncio.fixture(scope="function")
def app(fixture_tokenizer, mock_pipeline_config: PipelineConfig):  # noqa: ANN001, ANN201
    settings = Settings(api_types=[APIType.OPENAI], use_heartbeat=False)

    model_factory = EchoTokenGenerator
    tokenizer = EchoPipelineTokenizer()

    serving_settings = ServingTokenGeneratorSettings(
        model_factory=model_factory,
        pipeline_config=mock_pipeline_config,
        tokenizer=tokenizer,
    )
    return fastapi_app(settings, serving_settings)


@pytest.mark.asyncio
async def test_openai_chat_completion_single(app) -> None:  # noqa: ANN001
    async with AsyncTestClient(app) as client:
        request_content = "test data"
        response_json = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content
            ),
        )
        # This is not a streamed completion - There is no [DONE] at the end.
        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.content == request_content
        assert choice.finish_reason == "stop"


def _force_request_queue_full(app: FastAPI) -> None:
    """Make the worker request queue report itself as full.

    Mimics a bounded ZMQ PUSH socket at its high-water mark whose consumer has
    stopped draining: ``writable`` returns False, so admission rejects
    immediately with RequestQueueFull instead of attempting a push. (Startup
    already established connectivity, so an unwritable queue means "full".)
    """
    worker = app.state.pipeline.model_worker

    async def _not_writable(timeout_s: float | None = 0.0) -> bool:
        return False

    worker.request_queue.writable = _not_writable


@pytest.mark.asyncio
async def test_chat_completion_rejects_when_queue_full(app) -> None:  # noqa: ANN001
    """A full model-worker queue rejects new (non-streaming) requests with 429."""
    async with AsyncTestClient(app) as client:
        _force_request_queue_full(app)

        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name="echo", content="test data"),
        )

        assert response.status_code == 429
        body = response.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "429"
        # The rejected request must not leak an output-queue registration.
        assert len(app.state.pipeline.model_worker.pending_out_queues) == 0


@pytest.mark.asyncio
async def test_chat_completion_streaming_rejects_with_429_not_200(app) -> None:  # noqa: ANN001
    """Streaming admission failures surface as a 429 status, never a 200.

    Regression: the push to the worker is awaited before the SSE response is
    constructed, so a full queue (or crashed worker) fails with a real status
    code instead of a truncated 200 stream.
    """
    async with AsyncTestClient(app) as client:
        _force_request_queue_full(app)

        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content="test data", stream=True
            ),
        )

        assert response.status_code == 429
        assert len(app.state.pipeline.model_worker.pending_out_queues) == 0


@pytest.mark.asyncio
async def test_completion_streaming_rejects_with_429_not_200(app) -> None:  # noqa: ANN001
    """The legacy /v1/completions streaming path also fails fast with 429."""
    async with AsyncTestClient(app) as client:
        _force_request_queue_full(app)

        response = await client.post(
            "/v1/completions",
            json={"model": "echo", "prompt": "test data", "stream": True},
        )

        assert response.status_code == 429
        assert len(app.state.pipeline.model_worker.pending_out_queues) == 0


def test_openai_chat_completion_concurrent(app) -> None:  # noqa: ANN001
    request_contents: dict[int, str] = {}
    responses: dict[int, CreateChatCompletionResponse] = {}

    def execute_request(client: SyncTestClient, idx: int) -> None:
        # Ensure we always have at least one token in the request
        request_content = ",".join(f"_{i}_" for i in range(idx + 1))
        request_contents[idx] = request_content
        response_json = client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=request_content
            ),
        )
        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        responses[idx] = response

    num_threads = 10
    with SyncTestClient(app) as client:
        threads = []
        for i in range(num_threads):
            threads.append(Thread(target=execute_request, args=(client, i)))
            threads[i].start()
        for t in threads:
            t.join()

    assert len(responses) == num_threads
    for id, response in responses.items():
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
        received_response = response.choices[0].message.content
        expected_response = request_contents[id]
        assert received_response == expected_response


@pytest.mark.parametrize(
    "runtime_overrides", [{"tool_parser": "kimik2_5"}], indirect=True
)
def test_get_tool_parser_uses_runtime_override(
    mock_pipeline_config: PipelineConfig,
) -> None:
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    parser = get_tool_parser(app)

    assert isinstance(parser, KimiToolParser)


@pytest.mark.parametrize(
    "runtime_overrides", [{"tool_parser": None}], indirect=True
)
def test_get_tool_parser_returns_none_when_unset(
    mock_pipeline_config: PipelineConfig,
) -> None:
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    assert get_tool_parser(app) is None


@pytest.mark.parametrize(
    "runtime_overrides", [{"tool_parser": "does_not_exist"}], indirect=True
)
def test_get_tool_parser_unknown_parser_raises(
    mock_pipeline_config: PipelineConfig,
) -> None:
    app = FastAPI()
    app.state.pipeline_config = mock_pipeline_config

    with pytest.raises(ValueError, match="Unknown tool parser"):
        get_tool_parser(app)


_KIMI_TOOL_CALL_MARKUP = (
    f"{TOOL_CALLS_SECTION_BEGIN}{TOOL_CALL_BEGIN}functions.list_skills:0"
    f"{TOOL_CALL_ARGUMENT_BEGIN}{{}}{TOOL_CALL_END}{TOOL_CALLS_SECTION_END}"
)


@pytest.mark.parametrize(
    "runtime_overrides", [{"tool_parser": "kimik2_5"}], indirect=True
)
@pytest.mark.asyncio
async def test_chat_completion_leaves_tool_markup_alone_without_declared_tools(
    app,  # noqa: ANN001
) -> None:
    """Parsing a request that declares no tools is MiniMax-M3-only.

    Every other model keeps the pre-existing behavior: with no declared
    inventory the parser does not run, so tool-call markup stays in
    ``content``. `<tool_call>` is plain text for several tag-based parsers,
    which would otherwise let an ordinary chat answer be swallowed.
    """
    async with AsyncTestClient(app) as client:
        response_json = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(
                model_name="echo", content=_KIMI_TOOL_CALL_MARKUP
            ),
        )

    response = CreateChatCompletionResponse.model_validate(response_json.json())
    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.tool_calls is None
    assert choice.message.content == _KIMI_TOOL_CALL_MARKUP


@pytest.mark.asyncio
async def test_openai_chat_completion_empty_model_name(app) -> None:  # noqa: ANN001
    async with AsyncTestClient(app) as client:
        request_content = "test with empty model"

        # Create request with empty model name
        request_data = simple_openai_request(
            model_name="", content=request_content
        )

        response_json = await client.post(
            "/v1/chat/completions",
            json=request_data,
        )

        response = CreateChatCompletionResponse.model_validate(
            response_json.json()
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.content == request_content
        assert choice.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_chat_completion_prompt_too_long_returns_400(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``PromptTooLongError`` from the tokenizer must surface as 400."""

    async def _raise(self, request) -> None:  # noqa: ANN001
        raise PromptTooLongError(num_tokens=4096, max_length=2048)

    monkeypatch.setattr(EchoPipelineTokenizer, "new_context", _raise)

    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name="echo", content="anything"),
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["message"].startswith("Prompt is too long")
    assert "4096 tokens" in body["error"]["message"]
    assert "2048 tokens" in body["error"]["message"]
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_chat_completion_input_error_returns_400(app) -> None:  # noqa: ANN001
    async with AsyncTestClient(app) as client:
        app.state.pipeline.all_tokens = AsyncMock(
            side_effect=InputError("invalid image input")
        )

        response_json = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name="echo", content="test data"),
        )

        assert response_json.status_code == 400
        assert response_json.json()["error"]["message"] == "invalid image input"
        assert response_json.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_error_envelope_shape(app) -> None:  # noqa: ANN001
    """HTTP errors are returned in the OpenAI ``{"error": {...}}`` envelope."""
    async with AsyncTestClient(app) as client:
        app.state.pipeline.all_tokens = AsyncMock(
            side_effect=InputError("bad request")
        )

        response = await client.post(
            "/v1/chat/completions",
            json=simple_openai_request(model_name="echo", content="test data"),
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error"]["message"] == "bad request"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "400"
        assert "detail" not in body


@pytest.mark.asyncio
async def test_chat_completion_schema_validation_error_uses_openai_envelope(
    app,  # noqa: ANN001
) -> None:
    """Pydantic ``ValidationError`` from the chat schema surfaces as the OpenAI envelope.

    Regression that composes SERVSYS-1257 (strongly-typed ``messages`` field,
    so unknown roles raise ``pydantic.ValidationError`` at
    ``CreateChatCompletionRequest.model_validate_json`` time) with the
    ``HTTPException`` handler from #87521. The chat route catches
    ``ValidationError`` and re-raises as ``HTTPException(status_code=400)``,
    which the registered ``_openai_http_exception_handler`` turns into the
    ``{"error": {"message", "type", "code", "param"}}`` body that
    OpenAI/OpenRouter clients expect - not the raw FastAPI ``{"detail": ...}``.
    """
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [{"role": "wizard", "content": "abracadabra"}],
            },
        )

        assert response.status_code == 400
        body = response.json()
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "400"
        # ``str(ValidationError)`` includes the offending input, so the
        # rejected role makes it into the user-facing message.
        assert "wizard" in body["error"]["message"]


def test_decode_and_validate_images_rejects_bad_bytes() -> None:
    # Empty / non-image bytes must raise (the request handler maps this to a
    # 400), not reach the worker and crash it later with an unhandled
    # PIL.UnidentifiedImageError (HTTP 500).
    for bad in (b"", b"tiny", b"\x00\x01\x02\x03"):
        with pytest.raises(InputError):
            decode_and_validate_images([bad])


def test_decode_and_validate_images_returns_decoded_images() -> None:
    # The validator decodes each image once and returns the decoded PIL images
    # so the tokenizer can reuse them (decode-once); it must not discard them.
    buf = io.BytesIO()
    Image.new("RGB", (7, 11)).save(buf, format="PNG")
    decoded = decode_and_validate_images([buf.getvalue()])
    assert len(decoded) == 1
    assert decoded[0].size == (7, 11)
    # Must be fully decoded (load() already called), usable without the source.
    assert decoded[0].convert("RGB").size == (7, 11)


def test_decode_and_validate_images_rejects_truncated_image() -> None:
    # A header-valid but truncated image passes the lazy ``Image.open`` header
    # parse, but its pixel decode fails. Before the fix this slipped through
    # validation and crashed the worker with an unhandled OSError (HTTP 500)
    # in the tokenizer's decode; the validator must now force the decode and
    # turn it into a clean 400 (InputError). (MXSERV-162, bug 2.)
    buf = io.BytesIO()
    Image.effect_noise((256, 256), 80).convert("RGB").save(
        buf, format="JPEG", quality=90
    )
    full = buf.getvalue()
    # Sanity-check the precondition: a header parse alone does not raise.
    with Image.open(io.BytesIO(full[: int(len(full) * 0.88)])):
        pass
    with pytest.raises(InputError):
        decode_and_validate_images([full[: int(len(full) * 0.88)]])


def test_decode_and_validate_images_rejects_decompression_bomb(
    monkeypatch,  # noqa: ANN001
) -> None:
    # An image whose pixel count blows past PIL's decompression-bomb guard must
    # become a clean 400 (InputError), not an unhandled DecompressionBombError
    # (which is not an OSError/ValueError, so it would otherwise escape as 500).
    # (MXSERV-162.)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    data = buf.getvalue()
    # Lower the limit *after* building the bytes so 64*64 px trips the guard
    # (DecompressionBombError fires above 2x MAX_IMAGE_PIXELS).
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    with pytest.raises(InputError):
        decode_and_validate_images([data])


def test_open_image_carry_path_matches_bytes_path() -> None:
    # Decode-once invariant: a pre-decoded image (carry path,
    # request.decoded_images) and the raw bytes (fallback path) must yield
    # byte-identical pixels, so reusing the validator's decode in the tokenizer
    # cannot change model inputs. (MXSERV-162 follow-up: decode-once.)
    buf = io.BytesIO()
    Image.effect_noise((48, 32), 64).convert("RGBA").save(buf, format="PNG")
    data = buf.getvalue()

    # The validator decodes once and hands the image to the tokenizer.
    pre_decoded = decode_and_validate_images([data])[0]
    # open_image passes an already-decoded image through untouched (no re-decode)
    # and decodes raw bytes on the fallback path.
    assert open_image(pre_decoded) is pre_decoded
    carry = np.asarray(open_image(pre_decoded).convert("RGB"))
    fallback = np.asarray(open_image(data).convert("RGB"))
    assert np.array_equal(carry, fallback)


def test_decode_data_uri_base64_padded_unpadded_and_urlsafe() -> None:
    # Real clients and the OpenRouter relay send unpadded and/or url-safe
    # base64; the decoder must accept all three and yield identical bytes.
    # (MXSERV-162, bug 1.)
    raw = bytes(range(256))  # contains bytes that map to +/ and -_
    std = base64.b64encode(raw).decode()
    assert _decode_data_uri_base64(f"data:image/png;base64,{std}") == raw
    assert (
        _decode_data_uri_base64(f"data:image/png;base64,{std.rstrip('=')}")
        == raw
    )
    urlsafe = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert _decode_data_uri_base64(f"data:image/png;base64,{urlsafe}") == raw


def test_decode_data_uri_base64_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="no base64 payload"):
        _decode_data_uri_base64("data:image/png;base64,")


def test_coerce_positive_int() -> None:
    # Positive ints (incl. numeric strings) pass through; everything else,
    # including bool and non-positive values, becomes None.
    assert _coerce_positive_int(1008) == 1008
    assert _coerce_positive_int("512") == 512
    assert _coerce_positive_int(None) is None
    assert _coerce_positive_int(0) is None
    assert _coerce_positive_int(-4) is None
    assert _coerce_positive_int(True) is None
    assert _coerce_positive_int("not-a-number") is None


def test_coerce_positive_float() -> None:
    # Positive floats (incl. ints and numeric strings) pass through; bool,
    # None, non-positive, and garbage become None.
    assert _coerce_positive_float(1.0) == 1.0
    assert _coerce_positive_float(2) == 2.0
    assert _coerce_positive_float("0.5") == 0.5
    assert _coerce_positive_float(None) is None
    assert _coerce_positive_float(0) is None
    assert _coerce_positive_float(-1.0) is None
    assert _coerce_positive_float(True) is None
    assert _coerce_positive_float("nope") is None


@pytest.mark.asyncio
async def test_resolve_image_from_url_data_uri_unpadded() -> None:
    # End-to-end through resolve_image_from_url: an unpadded data URI used to
    # raise binascii.Error (surfaced as a 400); it must now round-trip.
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    png = buf.getvalue()
    b64 = base64.b64encode(png).decode().rstrip("=")
    out = await resolve_image_from_url(
        AnyUrl(f"data:image/png;base64,{b64}"),
        settings=None,  # type: ignore[arg-type]
    )
    assert out == png


def test_vllm_response_deserialization() -> None:
    vllm_response = """{"id":"chat-f33946bf8faf42849b11a4f948fc23f9","object":"chat.completion","created":1730306055,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Arrrr, listen close me hearty! Here be another one:\\n\\nWhy did the parrot go to the doctor?\\n\\nBecause it had a fowl temper! (get it? fowl, like a bird, but also a play on \\"foul\\" temper! ahh, shiver me timbers, I be laughin' me hook off!)","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":20,"total_tokens":92,"completion_tokens":72},"prompt_logprobs":null}"""

    CreateChatCompletionResponse.model_validate_json(vllm_response)


def test_max_server_response() -> None:
    response = """{"id":"7a0d00d-8f85-4a69-aa07-f51724787e3f","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Arrrr, here be another one:nnWhy did the pirate quit his job?nnBecause he was sick o' all the arrrr-guments with his boss! (get it? arrrr-guments? ahh, never mind, matey, I'll just be walkin' the plank if I don't get a laugh out o' ye!)","refusal":"","tool_calls":null,"role":"assistant","function_call":null},"logprobs":{"content":[],"refusal":[]}}],"created":1730310250,"model":"","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":null}"""
    CreateChatCompletionResponse.model_validate_json(response)


def test_create_chat_completion_request_with_target_endpoint() -> None:
    """Test that CreateChatCompletionRequest correctly parses target_endpoint field."""
    # Test with target_endpoint provided
    request_with_target = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "target_endpoint": "endpoint-instance-123",
    }

    parsed_request = CreateChatCompletionRequest.model_validate(
        request_with_target
    )
    assert parsed_request.target_endpoint == "endpoint-instance-123"
    assert parsed_request.model == "gpt-3.5-turbo"
    assert len(parsed_request.messages) == 1
    assert parsed_request.messages[0]["content"] == "Hello, world!"

    # Test without target_endpoint (should default to None)
    request_without_target = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    parsed_request_default = CreateChatCompletionRequest.model_validate(
        request_without_target
    )
    assert parsed_request_default.target_endpoint is None
    assert parsed_request_default.model == "gpt-3.5-turbo"


def test_create_chat_completion_request_with_cache_salt() -> None:
    """Test that CreateChatCompletionRequest correctly parses cache_salt field
    and enforces the 512-char length cap."""
    request_with_salt = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "cache_salt": "tenant-abc",
    }

    parsed_request = CreateChatCompletionRequest.model_validate(
        request_with_salt
    )
    assert parsed_request.cache_salt == "tenant-abc"

    request_without_salt = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    parsed_default = CreateChatCompletionRequest.model_validate(
        request_without_salt
    )
    assert parsed_default.cache_salt is None

    request_oversized = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "cache_salt": "x" * 600,
    }

    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(request_oversized)


def _make_request(headers: dict[str, str]) -> Request:
    """Minimal fastapi.Request carrying only headers, for testing
    header-extraction helpers without a full app/TestClient."""
    scope = {
        "type": "http",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in headers.items()
        ],
    }
    return Request(scope)


def test_get_cache_salt_header_takes_precedence_over_body() -> None:
    request = _make_request({"X-Cache-Salt": "from-header"})
    assert (
        _get_cache_salt(request, "from-body", use_client_cache_salt=True)
        == "from-header"
    )


def test_get_cache_salt_falls_back_to_body_without_header() -> None:
    request = _make_request({})
    assert (
        _get_cache_salt(request, "from-body", use_client_cache_salt=True)
        == "from-body"
    )


def test_get_cache_salt_returns_none_when_neither_present() -> None:
    request = _make_request({})
    assert _get_cache_salt(request, None, use_client_cache_salt=True) is None


def test_get_cache_salt_ignored_when_not_trusted() -> None:
    request = _make_request({"X-Cache-Salt": "from-header"})
    assert (
        _get_cache_salt(request, "from-body", use_client_cache_salt=False)
        is None
    )


def test_create_chat_completion_request_with_chat_template_kwargs() -> None:
    """Test that CreateChatCompletionRequest correctly parses chat_template_kwargs field."""
    # Test with chat_template_kwargs provided
    request_with_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
    }

    parsed_request = CreateChatCompletionRequest.model_validate(
        request_with_kwargs
    )
    assert parsed_request.chat_template_kwargs == {
        "enable_thinking": True,
        "thinking": True,
    }

    # Test without chat_template_kwargs (should default to None)
    request_without_kwargs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    parsed_request_default = CreateChatCompletionRequest.model_validate(
        request_without_kwargs
    )
    assert parsed_request_default.chat_template_kwargs is None


# ============================================================================
# Tests for batch id span attributes
# ============================================================================


def _chunk(batch_id: int | None) -> TokenGeneratorOutput:
    return TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE, batch_id=batch_id
    )


@pytest.mark.parametrize(
    ("batch_ids", "expected_first", "expected_last"),
    [
        ([], None, None),
        ([None], None, None),
        ([None, None], None, None),
        ([5], 5, 5),
        ([10, 11, 12], 10, 12),
        # A request that is admitted late, or released early, has no batch id
        # on its leading/trailing chunks; those must be skipped rather than
        # reported as the bounds.
        ([None, 10, 11, None], 10, 11),
        ([10, None, None, 14], 10, 14),
        ([None, None, 7, None], 7, 7),
        # batch_id is a scheduler-wide counter, so a preempted request can see
        # it move backwards. first/last are positional, never min/max.
        ([12, 11, 10], 12, 10),
    ],
)
def test_batch_id_returns_first_and_last_non_none(
    batch_ids: list[int | None],
    expected_first: int | None,
    expected_last: int | None,
) -> None:
    chunks = [_chunk(b) for b in batch_ids]
    assert _batch_id(chunks) == expected_first
    assert _batch_id(chunks, last=True) == expected_last


def test_batch_id_spans_a_multi_prompt_batch() -> None:
    """The completions route flattens per-prompt chunk lists before scanning.

    The bounds must cover the whole flattened batch -- reading them off any
    single prompt's chunks would under-report the span.
    """
    prompt_a = [_chunk(10), _chunk(11), _chunk(12)]
    prompt_b = [_chunk(11), _chunk(12), _chunk(13), _chunk(14)]
    all_outputs = [c for req in (prompt_a, prompt_b) for c in req]

    assert _batch_id(all_outputs) == 10
    assert _batch_id(all_outputs, last=True) == 14


def test_batch_id_does_not_consume_the_sequence() -> None:
    """Both directions are scanned from the same list, so it must be reusable."""
    chunks = [_chunk(10), _chunk(11)]

    assert _batch_id(chunks) == 10
    assert _batch_id(chunks, last=True) == 11
    # Re-reading yields the same answers.
    assert _batch_id(chunks) == 10
    assert _batch_id(chunks, last=True) == 11


def test_set_batch_id_attributes_tags_both_bounds() -> None:
    span = Mock()

    _set_batch_id_attributes(span, [_chunk(10), _chunk(11), _chunk(12)])

    span.set_attribute.assert_any_call("max.first_batch_id", 10)
    span.set_attribute.assert_any_call("max.last_batch_id", 12)


def test_set_batch_id_attributes_skips_unknown_bounds() -> None:
    """No batch id was ever observed, so neither attribute is worth emitting."""
    span = Mock()

    _set_batch_id_attributes(span, [_chunk(None)])

    span.set_attribute.assert_not_called()


# ============================================================================
# Tests for log probabilities functionality
# ============================================================================


def test_process_chat_log_probabilities_empty_outputs() -> None:
    """Test that _process_chat_log_probabilities handles empty outputs."""
    outputs: list[TokenGeneratorOutput] = []
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, ChatCompletionLogprobs)
    assert result.content == []
    assert result.refusal == []


def test_process_chat_log_probabilities_no_logprobs() -> None:
    """Test that _process_chat_log_probabilities handles outputs without log probs."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="hello",
            token_count=1,
            token_log_probabilities=None,
            top_log_probabilities=None,
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, ChatCompletionLogprobs)
    assert result.content == []
    assert result.refusal == []


def test_process_chat_log_probabilities_with_logprobs() -> None:
    """Test that _process_chat_log_probabilities correctly converts log probs."""
    # Simulate a token with log probabilities
    token_log_probs = [-0.5, -1.2]  # Log probs for 2 tokens
    top_log_probs = [
        {"hello": -0.5, "world": -1.0, "foo": -2.0},  # Top 3 for token 1
        {"bar": -1.2, "baz": -1.5, "qux": -2.5},  # Top 3 for token 2
    ]

    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_tokens="hello bar",
            token_count=2,
            token_log_probabilities=token_log_probs,
            top_log_probabilities=top_log_probs,
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, ChatCompletionLogprobs)
    content = result.content
    assert content is not None
    assert len(content) == 2
    assert result.refusal == []

    # Check first token
    first_token = content[0]
    assert isinstance(first_token, ChatCompletionTokenLogprob)
    assert first_token.logprob == -0.5
    assert first_token.token == "hello"  # Should match the sampled token
    assert len(first_token.top_logprobs) == 3

    # Check second token
    second_token = content[1]
    assert isinstance(second_token, ChatCompletionTokenLogprob)
    assert second_token.logprob == -1.2
    assert second_token.token == "bar"  # Should match the sampled token
    assert len(second_token.top_logprobs) == 3


def test_process_chat_log_probabilities_multiple_outputs() -> None:
    """Test that _process_chat_log_probabilities handles multiple output chunks."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="a",
            token_count=1,
            token_log_probabilities=[-0.1],
            top_log_probabilities=[{"a": -0.1, "b": -0.5}],
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_tokens="b",
            token_count=1,
            token_log_probabilities=[-0.2],
            top_log_probabilities=[{"b": -0.2, "c": -0.8}],
        ),
    ]
    result = _process_chat_log_probabilities(outputs)

    assert isinstance(result, ChatCompletionLogprobs)
    content = result.content
    assert content is not None
    assert len(content) == 2

    # First chunk's token
    assert content[0].logprob == -0.1
    assert content[0].token == "a"

    # Second chunk's token
    assert content[1].logprob == -0.2
    assert content[1].token == "b"


def test_process_chat_log_probabilities_top_logprobs_sorted() -> None:
    """Test that top_logprobs are sorted by logprob descending."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="x",
            token_count=1,
            token_log_probabilities=[-1.0],
            top_log_probabilities=[{"x": -1.0, "y": -0.5, "z": -2.0}],
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    content = result.content
    assert content is not None
    assert len(content) == 1
    top_logprobs = content[0].top_logprobs

    # Should be sorted by logprob descending: y (-0.5), x (-1.0), z (-2.0)
    assert len(top_logprobs) == 3
    assert top_logprobs[0].token == "y"
    assert top_logprobs[0].logprob == -0.5
    assert top_logprobs[1].token == "x"
    assert top_logprobs[1].logprob == -1.0
    assert top_logprobs[2].token == "z"
    assert top_logprobs[2].logprob == -2.0


def test_process_chat_log_probabilities_bytes_encoding() -> None:
    """Test that token bytes are correctly encoded as UTF-8."""
    outputs = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_tokens="é",
            token_count=1,
            token_log_probabilities=[-0.3],
            top_log_probabilities=[{"é": -0.3}],
        )
    ]
    result = _process_chat_log_probabilities(outputs)

    content = result.content
    assert content is not None
    assert len(content) == 1
    token_info = content[0]
    assert token_info.token == "é"
    # "é" in UTF-8 is [195, 169]
    assert token_info.bytes == [195, 169]


def test_create_chat_completion_request_with_logprobs() -> None:
    """Test that CreateChatCompletionRequest correctly parses logprobs fields."""
    # Test with logprobs enabled
    request_with_logprobs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "logprobs": True,
        "top_logprobs": 5,
    }

    parsed = CreateChatCompletionRequest.model_validate(request_with_logprobs)
    assert parsed.logprobs is True
    assert parsed.top_logprobs == 5

    # Test with logprobs disabled (default)
    request_without_logprobs = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    parsed_default = CreateChatCompletionRequest.model_validate(
        request_without_logprobs
    )
    # OpenAI defaults ``logprobs`` to ``None`` (omitted), not ``False``.
    assert parsed_default.logprobs is None
    assert parsed_default.top_logprobs is None

    # Test with logprobs=True but no top_logprobs specified
    request_logprobs_no_top = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "logprobs": True,
    }

    parsed_no_top = CreateChatCompletionRequest.model_validate(
        request_logprobs_no_top
    )
    assert parsed_no_top.logprobs is True
    assert parsed_no_top.top_logprobs is None


def test_max_server_response_with_logprobs() -> None:
    """Test deserialization of a response with populated logprobs."""
    response_with_logprobs = """{
        "id": "test-id",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello",
                "refusal": "",
                "tool_calls": null,
                "role": "assistant",
                "function_call": null
            },
            "logprobs": {
                "content": [{
                    "token": "Hello",
                    "logprob": -0.5,
                    "bytes": [72, 101, 108, 108, 111],
                    "top_logprobs": [{
                        "token": "Hello",
                        "logprob": -0.5,
                        "bytes": [72, 101, 108, 108, 111]
                    }, {
                        "token": "Hi",
                        "logprob": -1.2,
                        "bytes": [72, 105]
                    }]
                }],
                "refusal": []
            }
        }],
        "created": 1730310250,
        "model": "test-model",
        "service_tier": null,
        "system_fingerprint": null,
        "object": "chat.completion",
        "usage": null
    }"""

    response = CreateChatCompletionResponse.model_validate_json(
        response_with_logprobs
    )
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.logprobs is not None
    content = choice.logprobs.content
    assert content is not None
    assert len(content) == 1
    assert content[0].token == "Hello"
    assert content[0].logprob == -0.5
    assert len(content[0].top_logprobs) == 2


# ============================================================================
# Tests for reasoning functionality
# ============================================================================


def _make_mock_request() -> Mock:
    """Create a mock request for reasoning tests."""
    mock_request = Mock()
    mock_request.request_id = RequestID("test")
    mock_request.model_name = "test-model"
    mock_request.tools = None
    mock_request.response_format = None
    mock_request.timestamp_ns = 1
    mock_request.request_path = "/v1/chat/completions"
    mock_request.sampling_params = Mock()
    mock_request.sampling_params.stop = []
    return mock_request


@pytest.fixture
def patch_openai_metrics() -> Generator[None, None, None]:
    """Patch metrics so unit tests can exercise route helpers in isolation."""
    with (
        patch("max.serve.router.openai_routes.METRICS", MagicMock()),
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end"),
    ):
        yield


def _make_disconnect_request(
    *,
    pipeline: TokenGeneratorPipeline,
    pipeline_config: PipelineConfig,
    request_started: asyncio.Event,
    body: bytes,
) -> Mock:
    """Creates a request that disconnects once generation begins."""

    async def mock_receive() -> dict[str, str]:
        await request_started.wait()
        return {"type": "http.disconnect"}

    request = Mock()
    request._is_disconnected = False
    request.app = SimpleNamespace(
        state=SimpleNamespace(
            pipeline=pipeline,
            pipeline_config=pipeline_config,
            settings=Settings(api_types=[APIType.OPENAI], use_heartbeat=False),
        )
    )
    request.body = AsyncMock(return_value=body)
    request.headers = {}
    request.receive = mock_receive
    request.state = SimpleNamespace(
        request_id="disconnect-test",
        request_timer=Mock(start_ns=1, elapsed_ms=0.0),
    )
    request.url = SimpleNamespace(path="/v1/chat/completions")
    return request


_QueueItemT = TypeVar("_QueueItemT")


class _WritableQueue(asyncio.Queue[_QueueItemT]):
    """``asyncio.Queue`` that also satisfies the ``MAXAsyncPushQueue`` protocol.

    Admission probes ``writable()`` before pushing; a plain ``asyncio.Queue``
    lacks it, so this stand-in reports itself as always writable.
    """

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        return True


@pytest.mark.asyncio
async def test_openai_chat_completion_cancels_disconnected_request(
    mock_pipeline_config: PipelineConfig,
    patch_openai_metrics: None,
) -> None:
    """Regression test for the zombie-request bug in chat completions."""

    request_started = asyncio.Event()

    request_queue = _WritableQueue[BaseContext]()
    response_queue = asyncio.Queue[Any]()  # not used here
    cancel_queue = _WritableQueue[list[RequestID]]()
    model_worker = ZmqModelWorkerProxy(
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
    )

    pipeline = TokenGeneratorPipeline(
        model_name="echo",
        tokenizer=EchoPipelineTokenizer(),
        model_worker=model_worker,
    )

    request_body = json.dumps(
        simple_openai_request(model_name="echo", content="test data")
    ).encode("utf-8")
    mock_request = _make_disconnect_request(
        pipeline=pipeline,
        pipeline_config=mock_pipeline_config,
        request_started=request_started,
        body=request_body,
    )

    async with TaskGroup() as tg:
        session = tg.create_task(openai_create_chat_completion(mock_request))
        # wait for request to reach backend
        req = await asyncio.wait_for(request_queue.get(), timeout=1.0)
        assert req.request_id == RequestID("disconnect-test")
        request_started.set()

        # simulate disconnection
        session.cancel()

        # expect cancellation request to backend
        cancel = await asyncio.wait_for(cancel_queue.get(), timeout=1.0)
        assert cancel == [RequestID("disconnect-test")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks,expected_reasoning,expected_content,expected_completion_tokens",
    [
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="thinking...",
                    reasoning_token_count=3,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello world",
                    token_count=2,
                    prompt_token_count=5,
                ),
            ],
            "thinking...",
            "hello world",
            5,
            id="with_reasoning",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello",
                    token_count=1,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens=" world",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            None,
            "hello world",
            2,
            id="no_reasoning",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="thinking deeply",
                    reasoning_token_count=5,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=8,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="here is my answer",
                    token_count=10,
                    prompt_token_count=8,
                ),
            ],
            "thinking deeply",
            "here is my answer",
            15,
            id="usage_sums",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="A",
                    reasoning_token_count=1,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="B",
                    reasoning_token_count=1,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="C",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            "AB",
            "C",
            3,
            id="multiple_reasoning_chunks_joined",
        ),
        pytest.param(
            [
                TokenGeneratorOutput(
                    status=GenerationStatus.ACTIVE,
                    decoded_reasoning_tokens="",
                    reasoning_token_count=0,
                    decoded_tokens=None,
                    token_count=0,
                    prompt_token_count=5,
                ),
                TokenGeneratorOutput(
                    status=GenerationStatus.END_OF_SEQUENCE,
                    decoded_reasoning_tokens=None,
                    reasoning_token_count=0,
                    decoded_tokens="hello",
                    token_count=1,
                    prompt_token_count=5,
                ),
            ],
            None,
            "hello",
            1,
            id="empty_string_reasoning_is_none",
        ),
    ],
)
async def test_openai_chat_completion_reasoning(
    chunks: list[TokenGeneratorOutput],
    expected_reasoning: str | None,
    expected_content: str,
    expected_completion_tokens: int,
    patch_openai_metrics: None,
) -> None:
    """Test non-streaming response with various reasoning scenarios."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(mock_pipeline)
    response = await generator.complete([mock_request])

    message = response.choices[0].message
    assert message.reasoning == expected_reasoning
    assert message.content == expected_content
    assert response.usage is not None
    assert response.usage.completion_tokens == expected_completion_tokens


async def test_openai_chat_completion_usage_present_with_max_tokens_one(
    patch_openai_metrics: None,
) -> None:
    """CENG-920: max_tokens=1 must still report usage, not null.

    A single-token budget can be spent without incrementing either token
    counter (e.g. hitting the length cap before any content or reasoning
    token is produced), but ``prompt_tokens`` is always known once the
    request ran, so ``usage`` must never collapse to ``None``.
    """
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.MAXIMUM_LENGTH,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=12,
        ),
    ]
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(mock_pipeline)
    response = await generator.complete([mock_request])

    assert response.choices[0].finish_reason == "length"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 12


def _all_reasoning_chunks(
    status: GenerationStatus,
) -> list[TokenGeneratorOutput]:
    """A turn whose entire output is reasoning (no content tokens).

    Mirrors Kimi K2.5 answering inside the prefilled ``<think>`` block and
    stopping without ever emitting ``</think>`` — the parser routes everything
    to reasoning and content stays empty.
    """
    return [
        TokenGeneratorOutput(
            status=status,
            decoded_reasoning_tokens="The weather is 22F and Sunny.",
            reasoning_token_count=7,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
    ]


@pytest.mark.asyncio
async def test_reasoning_promoted_to_content_on_stop(
    patch_openai_metrics: None,
) -> None:
    """A1: all-reasoning + voluntary stop surfaces as content (not null)."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(
        return_value=_all_reasoning_chunks(GenerationStatus.END_OF_SEQUENCE)
    )

    response = await OpenAIChatResponseGenerator(mock_pipeline).complete(
        [_make_mock_request()]
    )
    message = response.choices[0].message
    assert message.content == "The weather is 22F and Sunny."
    assert message.reasoning is None


@pytest.mark.asyncio
async def test_reasoning_not_promoted_on_length(
    patch_openai_metrics: None,
) -> None:
    """A1 gate: a length-truncated thought stays reasoning, not content."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(
        return_value=_all_reasoning_chunks(GenerationStatus.MAXIMUM_LENGTH)
    )

    response = await OpenAIChatResponseGenerator(mock_pipeline).complete(
        [_make_mock_request()]
    )
    message = response.choices[0].message
    assert message.reasoning == "The weather is 22F and Sunny."
    assert not message.content


@pytest.mark.asyncio
async def test_stop_sequence_not_found_leaves_message_intact(
    patch_openai_metrics: None,
) -> None:
    """Regression: a stop_sequence recorded but not found in the joined
    message (idx == -1, e.g. already stripped by the streaming coalescer's
    stop truncation) must not slice off the last real character via a bare
    ``[:-1]``. The non-streaming trim is guarded with ``if idx >= 0``."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(
        return_value=[
            TokenGeneratorOutput(
                status=GenerationStatus.END_OF_SEQUENCE,
                decoded_tokens="hello world",
                stop_sequence="STOP",
                token_count=2,
                prompt_token_count=1,
            )
        ]
    )

    response = await OpenAIChatResponseGenerator(mock_pipeline).complete(
        [_make_mock_request()]
    )
    message = response.choices[0].message
    assert message.content == "hello world"
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_tool_parse_failure_does_not_leak_structural_marker(
    patch_openai_metrics: None,
) -> None:
    """Regression (fuzz-found): a ``max_tokens`` truncation landing mid
    ``<tool_call>`` block makes the qwen3_5 parser's ``parse_complete``
    raise (intentional — no complete block to parse). The non-streaming
    raw-text fallback must then surface only the content *before* the
    structural marker, never the raw response with the literal
    ``<tool_call>`` marker in ``message.content``."""
    truncated = (
        "I'll get the weather for Paris first.\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon"
    )
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(
        return_value=[
            TokenGeneratorOutput(
                status=GenerationStatus.MAXIMUM_LENGTH,
                decoded_tokens=truncated,
                token_count=24,
                prompt_token_count=5,
            )
        ]
    )

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=Qwen3_5ToolParser(),
        parse_tool_calls=True,
    )
    response = await generator.complete([_make_mock_request()])

    choice = response.choices[0]
    assert choice.finish_reason == "length"
    assert not choice.message.tool_calls
    assert "<tool_call>" not in (choice.message.content or "")
    assert choice.message.content == "I'll get the weather for Paris first."


async def _run_stream(
    chunks: list[TokenGeneratorOutput],
    *,
    stream_options: ChatCompletionStreamOptionsParam | None = None,
    fold_reasoning_into_content: bool = False,
    emit_reasoning_content: bool = False,
) -> list[CreateChatCompletionStreamResponse]:
    """Run streaming generator and return parsed responses."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        stream_options=stream_options,
        fold_reasoning_into_content=fold_reasoning_into_content,
        emit_reasoning_content=emit_reasoning_content,
    )
    return [
        CreateChatCompletionStreamResponse.model_validate_json(p)
        async for p in await generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


async def _run_completion_stream(
    chunks: list[TokenGeneratorOutput],
    *,
    stream_options: ChatCompletionStreamOptionsParam | None = None,
) -> list[CompletionStreamResponse]:
    """Run legacy text-completion streaming generator and parse chunks."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()
    mock_request.request_path = "/v1/completions"

    generator = OpenAICompletionResponseGenerator(
        mock_pipeline, stream_options=stream_options
    )
    return [
        CompletionStreamResponse.model_validate_json(p)
        async for p in await generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


async def _run_stream_with_kimi_tool_parser(
    chunks: list[TokenGeneratorOutput],
) -> list[CreateChatCompletionStreamResponse]:
    """Stream with parse_tool_calls + KimiToolParser (same path as OpenAI + tools)."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=KimiToolParser(),
        parse_tool_calls=True,
    )
    return [
        CreateChatCompletionStreamResponse.model_validate_json(p)
        async for p in await generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


_STREAM_REASONING_CHUNKS = [
    TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE,
        decoded_reasoning_tokens="thinking",
        reasoning_token_count=2,
        decoded_tokens=None,
        token_count=0,
        prompt_token_count=5,
    ),
    TokenGeneratorOutput(
        status=GenerationStatus.END_OF_SEQUENCE,
        decoded_reasoning_tokens=None,
        reasoning_token_count=0,
        decoded_tokens="answer",
        token_count=1,
        prompt_token_count=5,
    ),
]


@pytest.mark.asyncio
async def test_openai_chat_stream_reasoning_in_delta(
    patch_openai_metrics: None,
) -> None:
    """Test that streaming response includes reasoning in delta."""
    responses = await _run_stream(_STREAM_REASONING_CHUNKS)
    assert len(responses) == 2
    assert responses[0].choices[0].delta.reasoning == "thinking"
    assert responses[0].choices[0].delta.content is None
    assert responses[1].choices[0].delta.content == "answer"
    assert responses[1].choices[0].delta.reasoning is None


# ============================================================================
# Tests for MiniMax ``reasoning_split=False`` (fold reasoning into content).
#
# Reference behavior captured from the official MiniMax-M3 endpoint
# (api.minimax.io, model "MiniMax-M3"): with reasoning_split=False the response
# ``content`` is ``<think>\n{thinking}\n</think>\n\n{answer}`` and no separate
# reasoning field is returned. See the design doc for CENG-592.
# ============================================================================


@pytest.mark.asyncio
async def test_fold_reasoning_into_content_non_streaming(
    patch_openai_metrics: None,
) -> None:
    """Non-streaming fold wraps reasoning in <think> tags inside content."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="The answer is 408.",
            reasoning_token_count=5,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="17 x 24 = 408",
            token_count=4,
            prompt_token_count=5,
        ),
    ]
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    generator = OpenAIChatResponseGenerator(
        mock_pipeline, fold_reasoning_into_content=True
    )
    response = await generator.complete([_make_mock_request()])
    message = response.choices[0].message
    assert (
        message.content
        == "<think>\nThe answer is 408.\n</think>\n\n17 x 24 = 408"
    )
    assert message.reasoning is None


@pytest.mark.asyncio
async def test_fold_reasoning_disabled_keeps_reasoning_field(
    patch_openai_metrics: None,
) -> None:
    """Default (split=True) behavior is unchanged: reasoning stays separate."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="The answer is 408.",
            reasoning_token_count=5,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="17 x 24 = 408",
            token_count=4,
            prompt_token_count=5,
        ),
    ]
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    generator = OpenAIChatResponseGenerator(
        mock_pipeline, fold_reasoning_into_content=False
    )
    response = await generator.complete([_make_mock_request()])
    message = response.choices[0].message
    assert message.content == "17 x 24 = 408"
    assert message.reasoning == "The answer is 408."


@pytest.mark.asyncio
async def test_fold_reasoning_into_content_streaming(
    patch_openai_metrics: None,
) -> None:
    """Streaming fold emits <think> open/close in the content deltas only.

    Reconstructing the concatenated content must equal the official
    ``<think>\\n{reasoning}\\n</think>\\n\\n{answer}`` format, and no delta
    carries a separate reasoning field.
    """
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="The user wants 17 x 24.",
            reasoning_token_count=6,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=" It is 408.",
            reasoning_token_count=4,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="17 x 24 = 408",
            token_count=4,
            prompt_token_count=5,
        ),
    ]
    responses = await _run_stream(chunks, fold_reasoning_into_content=True)
    content = "".join(r.choices[0].delta.content or "" for r in responses)
    assert (
        content == "<think>\nThe user wants 17 x 24. It is 408.\n</think>\n\n"
        "17 x 24 = 408"
    )
    assert all(r.choices[0].delta.reasoning is None for r in responses)
    # The opening tag rides the first reasoning delta; the close + answer ride
    # the first content delta.
    assert responses[0].choices[0].delta.content == (
        "<think>\nThe user wants 17 x 24."
    )
    assert responses[-1].choices[0].delta.content == (
        "\n</think>\n\n17 x 24 = 408"
    )


@pytest.mark.asyncio
async def test_openai_chat_stream_usage_includes_reasoning_tokens(
    patch_openai_metrics: None,
) -> None:
    """Test streaming usage with stream_options.include_usage=True."""
    responses = await _run_stream(
        _STREAM_REASONING_CHUNKS,
        stream_options={"include_usage": True},
    )
    usage = responses[-1].usage
    assert usage is not None
    assert usage.completion_tokens == 3
    assert usage.prompt_tokens == 5
    assert usage.total_tokens == 8


@pytest.mark.asyncio
async def test_openai_chat_stream_reasoning_finish_reason(
    patch_openai_metrics: None,
) -> None:
    """Test that intermediate chunks have finish_reason=None and final has 'stop'."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="partial",
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=" answer",
            token_count=1,
            prompt_token_count=5,
        ),
    ]
    responses = await _run_stream(chunks)
    assert len(responses) == 3
    assert responses[0].choices[0].finish_reason is None
    assert responses[1].choices[0].finish_reason is None
    assert responses[2].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_completion_stream_skips_active_empty_chunks(
    patch_openai_metrics: None,
) -> None:
    """Regression: reasoning-only ACTIVE chunks do not crash /completions stream."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="partial",
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=" answer",
            token_count=1,
            prompt_token_count=5,
        ),
    ]

    responses = await _run_completion_stream(chunks)
    assert len(responses) == 2
    assert responses[0].choices[0].text == "partial"
    assert responses[0].choices[0].finish_reason is None
    assert responses[1].choices[0].text == " answer"
    assert responses[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_completion_stream_accounts_reasoning_tokens_for_metrics() -> (
    None
):
    """Billing/metrics counts include reasoning tokens even when chunk is skipped."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=3,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="done",
            token_count=2,
            prompt_token_count=5,
        ),
    ]

    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()
    mock_request.request_path = "/v1/completions"

    with (
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end") as end_mock,
    ):
        generator = OpenAICompletionResponseGenerator(mock_pipeline)
        _ = [p async for p in await generator.stream(mock_request)]

    assert end_mock.call_count == 1
    args = end_mock.call_args.args
    assert args[0] == "/v1/completions"
    assert args[2] == 5  # 3 reasoning + 2 completion tokens
    assert args[3] == 5


@pytest.mark.asyncio
async def test_openai_completion_non_stream_accounts_reasoning_tokens_for_metrics() -> (
    None
):
    """Billing/metrics counts include reasoning tokens in non-streaming mode."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=4,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="done",
            token_count=1,
            prompt_token_count=4,
        ),
    ]

    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)
    mock_request = _make_mock_request()
    mock_request.request_path = "/v1/completions"

    with (
        patch("max.serve.router.openai_routes.record_request_start"),
        patch("max.serve.router.openai_routes.record_request_end") as end_mock,
    ):
        generator = OpenAICompletionResponseGenerator(mock_pipeline)
        _ = await generator.complete([mock_request])

    assert end_mock.call_count == 1
    args = end_mock.call_args.args
    assert args[0] == "/v1/completions"
    assert args[2] == 3  # 2 reasoning + 1 completion tokens
    assert args[3] == 4


@pytest.mark.asyncio
async def test_openai_completion_non_stream_includes_usage(
    patch_openai_metrics: None,
) -> None:
    """Legacy /completions non-streaming response populates the usage block."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens="par",
            token_count=1,
            prompt_token_count=5,
            cached_token_count=3,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="tial",
            token_count=1,
            prompt_token_count=5,
            cached_token_count=3,
        ),
    ]

    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)
    mock_request = _make_mock_request()
    mock_request.request_path = "/v1/completions"

    generator = OpenAICompletionResponseGenerator(mock_pipeline)
    response = await generator.complete([mock_request])

    assert response.usage is not None
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 4  # 2 reasoning + 2 completion
    assert response.usage.total_tokens == 9
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.cached_tokens == 3


@pytest.mark.asyncio
async def test_openai_completion_stream_usage_includes_reasoning_tokens(
    patch_openai_metrics: None,
) -> None:
    """Streaming /completions with include_usage emits a final usage chunk."""
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="thinking",
            reasoning_token_count=2,
            decoded_tokens="partial",
            token_count=1,
            prompt_token_count=5,
            cached_token_count=3,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=" answer",
            token_count=1,
            prompt_token_count=5,
            cached_token_count=3,
        ),
    ]

    responses = await _run_completion_stream(
        chunks, stream_options={"include_usage": True}
    )

    # Final chunk carries usage with an empty choices list.
    final = responses[-1]
    assert final.choices == []
    assert final.usage is not None
    assert final.usage.prompt_tokens == 5
    assert final.usage.completion_tokens == 4  # 2 reasoning + 2 completion
    assert final.usage.total_tokens == 9
    assert final.usage.prompt_tokens_details is not None
    assert final.usage.prompt_tokens_details.cached_tokens == 3

    # Without include_usage, no usage chunk is appended.
    responses_no_usage = await _run_completion_stream(chunks)
    assert all(r.usage is None for r in responses_no_usage)


@pytest.mark.asyncio
async def test_openai_chat_stream_kimi_tool_prefix_maps_to_delta_content(
    patch_openai_metrics: None,
) -> None:
    """Integration: prose before tool markers maps to ``delta.content``, not arguments."""
    intro = "I'll check the weather for you.\n\n"
    section_begin = "<|tool_calls_section_begin|>"
    tool_body_end = (
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>"
        '{"location": "Boston"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=intro + section_begin,
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=tool_body_end,
            token_count=1,
            prompt_token_count=5,
        ),
    ]
    responses = await _run_stream_with_kimi_tool_parser(chunks)

    content_chunks = [
        r.choices[0].delta.content
        for r in responses
        if r.choices and r.choices[0].delta.content is not None
    ]
    assert "".join(content_chunks) == intro

    all_arguments_parts: list[str] = []
    for r in responses:
        assert r.choices
        for tc in r.choices[0].delta.tool_calls or []:
            if tc.function is not None and tc.function.arguments is not None:
                all_arguments_parts.append(tc.function.arguments)
                assert intro not in tc.function.arguments
    assert "".join(all_arguments_parts) == '{"location": "Boston"}'


# ============================================================================
# Regression tests: empty delta packets during tool-call generation
#
# While the parser captures/suppresses structural tool-call tokens it returns
# ``[]`` ("consumed this chunk, nothing to emit yet"). The stream must skip
# those chunks instead of pushing a delta with no content, no reasoning, and
# no tool-call fragment (an "empty packet").
# ============================================================================


class _ScriptedToolParser:
    """ToolParser stub that replays a pre-scripted list of ``parse_delta`` results.

    Each ``parse_delta`` call pops the next scripted result, letting a test
    drive the exact streaming shape — in particular the ``[]`` "consumed but
    nothing to emit" state that previously produced empty packets. Any calls
    beyond the script return ``None`` (plain passthrough).
    """

    def __init__(self, results: list[list[ParsedToolCallDelta] | None]) -> None:
        self._results = list(results)
        self.reset_calls = 0
        self.parse_delta_calls: list[str] = []

    def parse_complete(self, response: str) -> ParsedToolResponse:
        return ParsedToolResponse()

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        self.parse_delta_calls.append(delta)
        if self._results:
            return self._results.pop(0)
        return None

    def reset(self) -> None:
        self.reset_calls += 1


async def _run_stream_with_parser(
    chunks: list[TokenGeneratorOutput],
    parser: Any,
    *,
    stream_options: ChatCompletionStreamOptionsParam | None = None,
) -> list[CreateChatCompletionStreamResponse]:
    """Stream with a caller-supplied tool parser and parse the emitted chunks."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=parser,
        parse_tool_calls=True,
        stream_options=stream_options,
    )
    return [
        CreateChatCompletionStreamResponse.model_validate_json(p)
        async for p in await generator.stream(mock_request)
        if isinstance(p, str) and p != "[DONE]"
    ]


def _delta_is_empty(response: CreateChatCompletionStreamResponse) -> bool:
    """True when a streamed chunk carries a choice delta with nothing useful.

    A delta always pins ``role="assistant"``; "empty" means no content, no
    reasoning, no tool-call fragment, and no terminal ``finish_reason``.
    """
    if not response.choices:
        # A usage-only final chunk (choices == []) is not an empty packet.
        return False
    choice = response.choices[0]
    delta = choice.delta
    reasoning = getattr(delta, "reasoning", None) or getattr(
        delta, "reasoning_content", None
    )
    return (
        not delta.content
        and not delta.tool_calls
        and not reasoning
        and choice.finish_reason is None
    )


@pytest.mark.asyncio
async def test_openai_chat_stream_suppresses_empty_tool_call_packets(
    patch_openai_metrics: None,
) -> None:
    """Hidden tool-call tokens must not surface as empty delta packets.

    Chunk timeline (parser return in parens):
      1. real assistant prose        -> [content="Sure! "]
      2. structural token, hidden    -> []   (must be skipped)
      3. structural token, hidden    -> []   (must be skipped)
      4. first tool-call fragment     -> [id + name]
      5. argument bytes, terminal     -> [arguments]
    """
    parser = _ScriptedToolParser(
        [
            [ParsedToolCallDelta(index=0, content="Sure! ")],
            [],
            [],
            [ParsedToolCallDelta(index=0, id="call_abc", name="get_weather")],
            [ParsedToolCallDelta(index=0, arguments='{"location": "Boston"}')],
        ]
    )
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="Sure! ",
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_calls_section_begin|>",
            token_count=2,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_call_begin|>",
            token_count=3,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="functions.get_weather:0",
            token_count=4,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens='{"location": "Boston"}',
            token_count=5,
            prompt_token_count=5,
        ),
    ]

    responses = await _run_stream_with_parser(chunks, parser)

    # No empty packet ever reaches the client.
    assert not any(_delta_is_empty(r) for r in responses), (
        "empty delta packet leaked to the client: "
        f"{[r.model_dump(exclude_none=True) for r in responses]}"
    )

    # Exactly three deltas: prose, tool-call header, tool-call arguments.
    assert len(responses) == 3

    assert responses[0].choices[0].delta.content == "Sure! "
    assert responses[0].choices[0].delta.tool_calls is None

    header = responses[1].choices[0].delta.tool_calls
    assert header is not None and len(header) == 1
    assert header[0].id == "call_abc"
    assert header[0].function is not None
    assert header[0].function.name == "get_weather"
    assert responses[1].choices[0].delta.content is None
    assert responses[1].choices[0].finish_reason is None

    args = responses[2].choices[0].delta.tool_calls
    assert args is not None and len(args) == 1
    assert args[0].function is not None
    assert args[0].function.arguments == '{"location": "Boston"}'
    assert responses[2].choices[0].finish_reason == "tool_calls"

    # The suppressed chunks were still consumed by the parser.
    assert parser.parse_delta_calls == [
        "Sure! ",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "functions.get_weather:0",
        '{"location": "Boston"}',
    ]


@pytest.mark.asyncio
async def test_openai_chat_stream_suppressed_packets_still_counted(
    patch_openai_metrics: None,
) -> None:
    """Skipping empty packets must not drop their tokens from usage totals."""
    parser = _ScriptedToolParser(
        [
            [],  # hidden structural token
            [],  # hidden structural token
            [
                ParsedToolCallDelta(
                    index=0, id="call_1", name="f", arguments="{}"
                )
            ],
        ]
    )
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_calls_section_begin|>",
            token_count=2,
            prompt_token_count=7,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_call_begin|>",
            token_count=3,
            prompt_token_count=7,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="functions.f:0<|tool_call_argument_begin|>{}",
            token_count=5,
            prompt_token_count=7,
        ),
    ]

    responses = await _run_stream_with_parser(
        chunks, parser, stream_options={"include_usage": True}
    )

    assert not any(_delta_is_empty(r) for r in responses)

    # The two hidden chunks are suppressed; only the tool-call delta and the
    # final usage-only chunk remain.
    content_chunks = [r for r in responses if r.choices]
    usage_chunks = [r for r in responses if not r.choices]
    assert len(content_chunks) == 1
    assert len(usage_chunks) == 1

    usage = usage_chunks[0].usage
    assert usage is not None
    # completion_tokens must include the suppressed chunks (2 + 3 + 5 = 10).
    assert usage.completion_tokens == 10
    assert usage.prompt_tokens == 7
    assert usage.total_tokens == 17


@pytest.mark.asyncio
async def test_openai_chat_stream_no_empty_packet_when_only_content_suppressed(
    patch_openai_metrics: None,
) -> None:
    """A lone ``[]`` chunk (parser consumed everything) yields no packet at all."""
    parser = _ScriptedToolParser([[]])
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_calls_section_begin|>",
            token_count=1,
            prompt_token_count=3,
        ),
    ]

    responses = await _run_stream_with_parser(chunks, parser)

    # The single chunk is terminal, so it still needs to carry finish_reason —
    # but it is not an "empty" packet because finish_reason is set.
    assert len(responses) == 1
    assert not _delta_is_empty(responses[0])
    assert responses[0].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_chat_stream_kimi_section_marker_alone_no_empty_packet(
    patch_openai_metrics: None,
) -> None:
    """Integration: real KimiToolParser, section marker arriving alone.

    When ``<|tool_calls_section_begin|>`` lands in its own chunk the parser
    returns ``[]`` (in-section, nothing to emit). That chunk must be dropped,
    not forwarded as an empty delta.
    """
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="<|tool_calls_section_begin|>",
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=(
                "<|tool_call_begin|>functions.get_weather:0"
                "<|tool_call_argument_begin|>"
            ),
            token_count=1,
            prompt_token_count=5,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=(
                '{"location": "Boston"}'
                "<|tool_call_end|><|tool_calls_section_end|>"
            ),
            token_count=1,
            prompt_token_count=5,
        ),
    ]

    responses = await _run_stream_with_parser(chunks, KimiToolParser())

    assert not any(_delta_is_empty(r) for r in responses), (
        "empty delta packet leaked during Kimi tool-call streaming: "
        f"{[r.model_dump(exclude_none=True) for r in responses]}"
    )

    # The tool name and arguments still stream through intact.
    names = [
        tc.function.name
        for r in responses
        if r.choices
        for tc in (r.choices[0].delta.tool_calls or [])
        if tc.function is not None and tc.function.name is not None
    ]
    assert names == ["get_weather"]

    argument_parts = [
        tc.function.arguments
        for r in responses
        if r.choices
        for tc in (r.choices[0].delta.tool_calls or [])
        if tc.function is not None and tc.function.arguments is not None
    ]
    assert "".join(argument_parts) == '{"location": "Boston"}'

    # A terminal finish_reason of tool_calls is present exactly once.
    finish_reasons = [
        r.choices[0].finish_reason
        for r in responses
        if r.choices and r.choices[0].finish_reason is not None
    ]
    assert finish_reasons == ["tool_calls"]


# ============================================================================
# Tests for response format conversion
# ============================================================================


def test_create_response_format_json_object() -> None:
    """Test that json_object format is converted to json_schema with permissive schema."""
    result = _create_response_format(
        {"type": "json_object"}, enable_response_format_schema=True
    )

    assert result is not None
    # json_object should be normalized to json_schema internally
    assert result.type == "json_schema"
    # Should use a permissive schema that accepts any JSON object
    assert result.json_schema == {"type": "object"}


def test_create_response_format_json_schema() -> None:
    """Test that json_schema format preserves the provided schema."""
    person_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    result = _create_response_format(
        {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": person_schema},
        },
        enable_response_format_schema=True,
    )

    assert result is not None
    assert result.type == "json_schema"
    # Schema should contain the provided JSON schema
    assert result.json_schema is not None
    assert "properties" in result.json_schema
    assert "name" in result.json_schema["properties"]
    assert "age" in result.json_schema["properties"]


def test_create_response_format_boolean_schema_true() -> None:
    """A boolean schema ``true`` (any value) de-sugars to ``{}``."""
    result = _create_response_format(
        {"type": "json_schema", "json_schema": {"name": "t", "schema": True}},
        enable_response_format_schema=True,
    )
    assert result is not None
    assert result.json_schema == {}


def test_create_response_format_empty_schema_is_enforced() -> None:
    """An explicit empty schema (``{}`` / boolean ``true``) is "any valid JSON
    value" and is ENFORCED, not treated as "no schema". The empty schema flows
    through unchanged (it compiles to the backend's any-value grammar), and both
    enforcement flags are set so generation is constrained to exactly one
    well-formed JSON value (no trailing prose)."""
    schema: dict[str, Any] | bool
    for schema in ({}, True):
        result = _create_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "any", "schema": schema},
            },
            enable_response_format_schema=True,
        )
        assert result is not None, schema
        assert result.json_schema == {}, schema
        assert result.grammar_enforced is True, schema
        assert result.has_json_schema is True, schema
        assert result.requires_structured_output_flag is True, schema


def test_create_response_format_absent_is_unconstrained() -> None:
    """No ``response_format`` means no schema is provided: return ``None`` so the
    request is unconstrained (distinct from an explicit empty any-value schema)."""
    assert _create_response_format(
        None, enable_response_format_schema=True
    ) is (None)


def test_create_response_format_boolean_schema_false() -> None:
    """A boolean schema ``false`` (matches nothing) de-sugars to the
    unsatisfiable ``{"anyOf": [False]}``, which the worker rejects as a 400 --
    no output can satisfy it. (``{"anyOf": [False]}`` is used over
    ``{"not": {}}`` because llguidance lacks ``not`` and reports a misleading
    error.)"""
    result = _create_response_format(
        {
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": False},
        },
        enable_response_format_schema=True,
    )

    assert result is not None
    assert result.json_schema == {"anyOf": [False]}


def test_create_response_format_text() -> None:
    """Test that text format returns empty json_schema."""
    result = _create_response_format(
        {"type": "text"}, enable_response_format_schema=False
    )

    assert result is not None
    assert result.type == "text"
    assert result.json_schema == {}


def test_create_response_format_none() -> None:
    """Test that None input returns None."""
    result = _create_response_format(None, enable_response_format_schema=False)
    assert result is None


@pytest.mark.parametrize("response_type", ["json_schema", "json_object"])
def test_create_response_format_rejects_schema_without_flag(
    response_type: str,
) -> None:
    """Reject json_schema / json_object at the route boundary when the
    server was not started with --enable-structured-output.

    Without this guard the worker hits the same condition later in
    ``StructuredOutputHelper.update_context`` and the InputError escapes
    the scheduler loop, killing the worker (MXSERV-106).
    """
    response_format: dict[str, Any] = {"type": response_type}
    if response_type == "json_schema":
        response_format["json_schema"] = {
            "name": "person",
            "schema": {"type": "object"},
        }

    with pytest.raises(InputError, match=r"--enable-structured-output"):
        _create_response_format(
            response_format,  # type: ignore[arg-type]
            enable_response_format_schema=False,
        )


# ============================================================================
# Tests for _resolve_grammar_constraints
# ============================================================================


def _make_tools(names: list[str]) -> list[TextGenerationRequestTool]:
    """Helper to create tool definitions for testing."""
    return [
        TextGenerationRequestTool(
            type="function",
            function={"name": name, "description": None, "parameters": {}},
        )
        for name in names
    ]


def _make_response_format(
    json_schema: dict[str, Any],
) -> TextGenerationResponseFormat:
    """Helper to create response format for testing."""
    return TextGenerationResponseFormat(
        type="json_schema",
        json_schema=json_schema,
        grammar=None,
        grammar_enforced=True,
        tools_forced=False,
    )


def test_resolve_grammar_constraints_tools_required() -> None:
    """When tool_choice='required', constrain to all tools, no response schema."""
    tools = _make_tools(["get_weather", "search"])
    response_format = _make_response_format({"type": "object"})

    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=tools,
            tool_choice="required",
            response_format=response_format,
        )
    )

    assert grammar_tools == tools
    assert schema is None  # response_format ignored when tools forced
    assert tools_forced is True  # tool_choice=required forces tools
    assert (
        enforce_from_start is True
    )  # forced tools enforce from the first token


def test_resolve_grammar_constraints_named_function() -> None:
    """When tool_choice names a specific function, constrain to that tool only."""
    tools = _make_tools(["get_weather", "search"])
    response_format = _make_response_format({"type": "object"})

    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
            response_format=response_format,
        )
    )

    assert grammar_tools is not None
    assert len(grammar_tools) == 1
    assert grammar_tools[0]["function"]["name"] == "get_weather"
    assert schema is None  # response_format ignored when tools forced
    assert tools_forced is True  # specific function forces tools
    assert enforce_from_start is True


def test_resolve_grammar_constraints_auto_with_response_format() -> None:
    """Auto mode + response_format: include all tools and response schema."""
    tools = _make_tools(["get_weather", "search"])
    response_format = _make_response_format({"type": "object"})

    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=tools,
            tool_choice="auto",
            response_format=response_format,
        )
    )

    assert grammar_tools == tools
    assert schema == {"type": "object"}
    assert tools_forced is False  # auto mode doesn't force tools
    # auto + response_format: enforce from start since schema is in play
    assert enforce_from_start is True


def test_resolve_grammar_constraints_auto_no_response_format() -> None:
    """Auto mode + no response_format: grammar generated for conditional enforcement."""
    tools = _make_tools(["get_weather", "search"])

    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=tools,
            tool_choice="auto",
            response_format=None,
        )
    )

    # auto with tools now generates a grammar so the bitmask can engage
    # conditionally once a tool-call start token is detected.
    assert grammar_tools == tools
    assert schema is None
    assert tools_forced is False
    assert enforce_from_start is False  # conditional enforcement


def test_resolve_grammar_constraints_response_format_only() -> None:
    """Response format only (no tools): constrain to JSON schema."""
    response_format = _make_response_format({"type": "object"})

    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=None,
            tool_choice=None,
            response_format=response_format,
        )
    )

    assert grammar_tools is None
    assert schema == {"type": "object"}
    assert tools_forced is False
    assert enforce_from_start is False  # no tools, no grammar to enforce


def test_resolve_grammar_constraints_no_constraints() -> None:
    """No tools, no response_format: no grammar generated."""
    grammar_tools, schema, tools_forced, enforce_from_start = (
        _resolve_grammar_constraints(
            tools=None,
            tool_choice=None,
            response_format=None,
        )
    )

    assert grammar_tools is None
    assert schema is None
    assert tools_forced is False
    assert enforce_from_start is False


# ============================================================================
# Tests for OpenAIChatResponseGenerator with tool calling
# ============================================================================


@pytest.mark.asyncio
async def test_openai_chat_completion_tool_calling_with_reasoning(
    patch_openai_metrics: None,
) -> None:
    """Test non-streaming response with tool calls and reasoning tokens."""
    # The model outputs reasoning first, then a tool call JSON
    tool_call_json = (
        '{"name": "get_weather", "parameters": {"location": "Boston"}}'
    )
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens="Let me check the weather for Boston...",
            reasoning_token_count=7,
            decoded_tokens=None,
            token_count=0,
            prompt_token_count=10,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=tool_call_json,
            token_count=15,
            prompt_token_count=10,
        ),
    ]

    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=LlamaToolParser(),
        parse_tool_calls=True,
    )
    response = await generator.complete([mock_request])

    # Check that reasoning is present
    message = response.choices[0].message
    assert message.reasoning == "Let me check the weather for Boston..."

    # Check that tool calls were parsed
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageToolCall)
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "Boston"}'
    assert tool_call.type == "function"
    assert tool_call.id.startswith("call_")

    # Check finish reason is tool_calls
    assert response.choices[0].finish_reason == "tool_calls"

    # Check usage includes reasoning tokens
    assert response.usage is not None
    assert response.usage.completion_tokens == 22  # 7 reasoning + 15 content
    assert response.usage.prompt_tokens == 10
    assert response.usage.total_tokens == 32


@pytest.mark.asyncio
async def test_openai_chat_completion_tool_calling_with_content(
    patch_openai_metrics: None,
) -> None:
    """Test non-streaming response with tool calls and regular content (no reasoning)."""
    # The model outputs a tool call JSON without any reasoning
    tool_call_json = '{"name": "get_time", "parameters": {"timezone": "EST"}}'
    chunks = [
        TokenGeneratorOutput(
            status=GenerationStatus.ACTIVE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens="Here is the time: ",
            token_count=4,
            prompt_token_count=8,
        ),
        TokenGeneratorOutput(
            status=GenerationStatus.END_OF_SEQUENCE,
            decoded_reasoning_tokens=None,
            reasoning_token_count=0,
            decoded_tokens=tool_call_json,
            token_count=12,
            prompt_token_count=8,
        ),
    ]

    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(return_value=chunks)

    mock_request = _make_mock_request()

    generator = OpenAIChatResponseGenerator(
        mock_pipeline,
        parser=LlamaToolParser(),
        parse_tool_calls=True,
    )
    response = await generator.complete([mock_request])

    # Check that reasoning is NOT present (no reasoning tokens)
    message = response.choices[0].message
    assert message.reasoning is None

    # Check that tool calls were parsed
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageToolCall)
    assert tool_call.function.name == "get_time"
    assert tool_call.function.arguments == '{"timezone": "EST"}'
    assert tool_call.type == "function"

    # Check finish reason is tool_calls
    assert response.choices[0].finish_reason == "tool_calls"

    # Check usage (no reasoning tokens)
    assert response.usage is not None
    assert response.usage.completion_tokens == 16  # 4 + 12
    assert response.usage.prompt_tokens == 8
    assert response.usage.total_tokens == 24


@pytest.mark.asyncio
async def test_chat_stream_error_yields_json(
    patch_openai_metrics: None,
) -> None:
    """Regression test for MXSERV-95: errors raised mid-stream are serialized as JSON.

    Once the SSE response has begun (headers sent, first chunk yielded), an
    error can no longer change the HTTP status, so it must be serialized as a
    JSON error payload inside the stream rather than propagating.
    """
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        async def _gen() -> Any:
            yield TokenGeneratorOutput(
                status=GenerationStatus.ACTIVE,
                decoded_tokens="hi",
                token_count=1,
                prompt_token_count=5,
            )
            raise ValueError(
                "Input string is larger than tokenizer's max length "
                "(264823 > 262144)."
            )

        return _gen()

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    generator = OpenAIChatResponseGenerator(mock_pipeline)

    results = [p async for p in await generator.stream(_make_mock_request())]

    # The first payload is the streamed chunk; the last is the serialized
    # error. The error path does not emit [DONE].
    payload = results[-1]
    assert isinstance(payload, str), (
        f"Expected a JSON string, got {type(payload).__name__}: {payload!r}"
    )

    # Must parse as JSON — not as Python repr like ErrorResponse(error=Error(...))
    parsed = json.loads(payload)
    assert parsed["error"]["code"] == "500"
    assert "262144" in parsed["error"]["message"]


@pytest.mark.asyncio
async def test_chat_stream_submission_error_raises(
    patch_openai_metrics: None,
) -> None:
    """SERVSYS-1277: a failed submission raises from ``stream`` before the SSE
    response begins, so the route can map it to an HTTP error status instead of
    burying it in an already-200 stream."""
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    async def mock_next_token_chunk(request: Any) -> Any:
        raise ValueError(
            "Input string is larger than tokenizer's max length "
            "(264823 > 262144)."
        )

    mock_pipeline.next_token_chunk = mock_next_token_chunk
    generator = OpenAIChatResponseGenerator(mock_pipeline)

    # Awaiting the coroutine performs the submission, which raises here rather
    # than yielding an error payload inside the stream.
    with pytest.raises(ValueError, match="262144"):
        await generator.stream(_make_mock_request())


# ============================================================================
# Tests for relaxed-request runtime flags:
#   - allow_unsupported_logprobs: drop logprobs requests that the runtime
#     cannot honor (e.g. overlap scheduler) instead of returning 400.
#   - allow_extra_request_fields: silently drop unknown top-level body fields
#     instead of failing pydantic validation with 400.
# ============================================================================


def test_pipeline_runtime_config_allow_unsupported_logprobs_default_false() -> (
    None
):
    """``allow_unsupported_logprobs`` is opt-in; default preserves strictness."""
    runtime = PipelineRuntimeConfig()
    assert runtime.allow_unsupported_logprobs is False


def test_pipeline_runtime_config_allow_extra_request_fields_default_false() -> (
    None
):
    """``allow_extra_request_fields`` is opt-in; default preserves strictness."""
    runtime = PipelineRuntimeConfig()
    assert runtime.allow_extra_request_fields is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"enable_overlap_scheduler": True, "allow_unsupported_logprobs": False}],
    indirect=True,
)
async def test_chat_completion_logprobs_with_overlap_scheduler_rejected_by_default(
    app,  # noqa: ANN001
) -> None:
    """With the overlap scheduler on and the flag off, logprobs is a 400."""
    async with AsyncTestClient(app) as client:
        body = simple_openai_request(model_name="echo", content="hi")
        body["logprobs"] = True
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "overlap" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"enable_overlap_scheduler": True, "allow_unsupported_logprobs": True}],
    indirect=True,
)
async def test_chat_completion_logprobs_with_overlap_scheduler_dropped_when_flag_set(
    app,  # noqa: ANN001
) -> None:
    """With the flag on, logprobs requests succeed and return ``logprobs: null``."""
    async with AsyncTestClient(app) as client:
        body = simple_openai_request(
            model_name="echo", content="logprobs please"
        )
        body["logprobs"] = True
        body["top_logprobs"] = 5
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    parsed = CreateChatCompletionResponse.model_validate(response.json())
    assert len(parsed.choices) == 1
    choice = parsed.choices[0]
    assert choice.message.content == "logprobs please"
    # When logprobs is downgraded, the response carries no logprob content.
    assert choice.logprobs is None or not choice.logprobs.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"allow_extra_request_fields": False}],
    indirect=True,
)
async def test_chat_completion_extra_field_rejected_by_default(
    app,  # noqa: ANN001
) -> None:
    """With the flag off, an unknown top-level field returns a 400."""
    async with AsyncTestClient(app) as client:
        body = simple_openai_request(model_name="echo", content="hello")
        body["dynamic_temperature"] = {"</think>": 0}
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "dynamic_temperature" in response.json()["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"allow_extra_request_fields": True}],
    indirect=True,
)
async def test_chat_completion_extra_field_dropped_when_flag_set(
    app,  # noqa: ANN001
) -> None:
    """With the flag on, an unknown top-level field is dropped and the request succeeds."""
    async with AsyncTestClient(app) as client:
        body = simple_openai_request(model_name="echo", content="hello")
        body["dynamic_temperature"] = {"</think>": 0}
        body["some_other_vendor_field"] = "ignored"
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    parsed = CreateChatCompletionResponse.model_validate(response.json())
    assert parsed.choices[0].message.content == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"enable_overlap_scheduler": True, "allow_unsupported_logprobs": False}],
    indirect=True,
)
async def test_completion_logprobs_with_overlap_scheduler_rejected_by_default(
    app,  # noqa: ANN001
) -> None:
    """Legacy /v1/completions also rejects logprobs under the overlap scheduler."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "echo",
                "prompt": "hi",
                "logprobs": 3,
            },
        )

    assert response.status_code == 400
    assert "overlap" in response.json()["error"]["message"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"enable_overlap_scheduler": True, "allow_unsupported_logprobs": True}],
    indirect=True,
)
async def test_completion_logprobs_with_overlap_scheduler_dropped_when_flag_set(
    app,  # noqa: ANN001
) -> None:
    """Legacy /v1/completions silently drops logprobs when the flag is on."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "echo",
                "prompt": "echo this",
                "logprobs": 3,
            },
        )

    assert response.status_code == 200
    body = response.json()
    # The legacy endpoint returns the OpenAI logprobs container shape even
    # when downgraded; what matters is that no per-token logprobs were
    # actually emitted.
    logprobs_field = body["choices"][0]["logprobs"]
    assert logprobs_field is None or not logprobs_field.get("token_logprobs")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"allow_extra_request_fields": False}],
    indirect=True,
)
async def test_completion_extra_field_rejected_by_default(
    app,  # noqa: ANN001
) -> None:
    """Legacy /v1/completions rejects unknown fields by default."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "echo",
                "prompt": "hi",
                "dynamic_temperature": {"</think>": 0},
            },
        )

    assert response.status_code == 400
    assert "dynamic_temperature" in response.json()["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_overrides",
    [{"allow_extra_request_fields": True}],
    indirect=True,
)
async def test_completion_extra_field_dropped_when_flag_set(
    app,  # noqa: ANN001
) -> None:
    """Legacy /v1/completions drops unknown fields when the flag is on."""
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "model": "echo",
                "prompt": "echo this",
                "dynamic_temperature": {"</think>": 0},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "echo this"


def test_response_message_carries_reasoning_content() -> None:
    msg = ChatCompletionResponseMessage(
        role="assistant", reasoning_content="thinking"
    )
    assert msg.reasoning_content == "thinking"
    # Unselected field stays None and is dropped from the wire.
    assert '"reasoning":' not in msg.model_dump_json(exclude_none=True)


def test_stream_delta_carries_reasoning_content() -> None:
    delta = ChatCompletionStreamResponseDelta(reasoning_content="frag")
    assert delta.reasoning_content == "frag"


# ============================================================================
# Tests for emit_reasoning_content flag (CENG-651).
# ============================================================================


@pytest.mark.asyncio
async def test_non_stream_emits_reasoning_content_when_flag_on(
    patch_openai_metrics: None,
) -> None:
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"
    mock_pipeline.all_tokens = AsyncMock(
        return_value=[
            TokenGeneratorOutput(
                status=GenerationStatus.ACTIVE,
                decoded_reasoning_tokens="thinking",
                reasoning_token_count=1,
                decoded_tokens=None,
                token_count=0,
                prompt_token_count=5,
            ),
            TokenGeneratorOutput(
                status=GenerationStatus.END_OF_SEQUENCE,
                decoded_reasoning_tokens=None,
                reasoning_token_count=0,
                decoded_tokens="answer",
                token_count=1,
                prompt_token_count=5,
            ),
        ]
    )

    generator = OpenAIChatResponseGenerator(
        mock_pipeline, emit_reasoning_content=True
    )
    response = await generator.complete([_make_mock_request()])

    message = response.choices[0].message
    assert message.reasoning_content == "thinking"
    assert message.reasoning is None
    assert message.content == "answer"


@pytest.mark.asyncio
async def test_stream_emits_reasoning_content_when_flag_on(
    patch_openai_metrics: None,
) -> None:
    responses = await _run_stream(
        _STREAM_REASONING_CHUNKS, emit_reasoning_content=True
    )
    assert responses[0].choices[0].delta.reasoning_content == "thinking"
    assert responses[0].choices[0].delta.reasoning is None
    assert responses[1].choices[0].delta.content == "answer"


@pytest.mark.asyncio
async def test_stream_emits_reasoning_by_default(
    patch_openai_metrics: None,
) -> None:
    responses = await _run_stream(_STREAM_REASONING_CHUNKS)
    assert responses[0].choices[0].delta.reasoning == "thinking"
    assert responses[0].choices[0].delta.reasoning_content is None


# A single boundary chunk carrying both the reasoning tail and the first
# content tokens — the case CENG-892 must split apart.
_STREAM_REASONING_CONTENT_BOUNDARY_CHUNKS = [
    TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE,
        decoded_reasoning_tokens="thinking",
        reasoning_token_count=1,
        decoded_tokens="answer",
        token_count=1,
        prompt_token_count=5,
    ),
    TokenGeneratorOutput(
        status=GenerationStatus.END_OF_SEQUENCE,
        decoded_reasoning_tokens=None,
        reasoning_token_count=0,
        decoded_tokens=" more",
        token_count=1,
        prompt_token_count=5,
    ),
]


@pytest.mark.asyncio
async def test_stream_splits_reasoning_and_content_boundary_chunk(
    patch_openai_metrics: None,
) -> None:
    """CENG-892: a chunk carrying both reasoning and content is split.

    No emitted delta may set both ``reasoning_content`` and ``content``, the
    reasoning fragment must precede the content fragment, and a terminal
    finish_reason must ride the content delta, not the reasoning delta.
    """
    responses = await _run_stream(
        _STREAM_REASONING_CONTENT_BOUNDARY_CHUNKS,
        emit_reasoning_content=True,
    )
    deltas = [r.choices[0].delta for r in responses]

    # No delta carries both fields at once.
    for delta in deltas:
        assert not (delta.reasoning_content and delta.content)

    # Reasoning is emitted before content, and both fragments survive intact.
    reasoning_idx = next(i for i, d in enumerate(deltas) if d.reasoning_content)
    content_idx = next(i for i, d in enumerate(deltas) if d.content)
    assert reasoning_idx < content_idx
    assert deltas[reasoning_idx].reasoning_content == "thinking"
    assert "".join(d.content or "" for d in deltas) == "answer more"

    # The reasoning delta is non-terminal; finish_reason rides content only.
    assert responses[reasoning_idx].choices[0].finish_reason is None


_REASONING_CONTENT_CHUNKS = [
    TokenGeneratorOutput(
        status=GenerationStatus.ACTIVE,
        decoded_reasoning_tokens="thinking",
        reasoning_token_count=1,
        decoded_tokens=None,
        token_count=0,
        prompt_token_count=5,
    ),
    TokenGeneratorOutput(
        status=GenerationStatus.END_OF_SEQUENCE,
        decoded_reasoning_tokens=None,
        reasoning_token_count=0,
        decoded_tokens="answer",
        token_count=1,
        prompt_token_count=5,
    ),
]


@pytest.mark.asyncio
async def test_non_stream_reasoning_content_wire_serialization(
    patch_openai_metrics: None,
) -> None:
    """Verify non-streaming wire serialization of reasoning_content vs reasoning.

    The chat completion route is declared with ``response_model=None``, so FastAPI
    serializes the returned Pydantic model via ``jsonable_encoder`` WITHOUT
    ``exclude_none``. As a result, the unselected reasoning field appears in the
    wire body as ``null`` (null-not-absent) rather than being omitted — this is
    the documented, spec-accepted behavior. The streaming path serializes with
    ``model_dump_json(exclude_none=True)`` and is covered by separate tests.

    Flag ON:  reasoning_content == "thinking", reasoning is present but null.
    Flag OFF: reasoning == "thinking", reasoning_content is present but null.
    """
    mock_pipeline = Mock()
    mock_pipeline.model_name = "test-model"

    # --- Flag ON: emit_reasoning_content=True ---
    mock_pipeline.all_tokens = AsyncMock(return_value=_REASONING_CONTENT_CHUNKS)
    generator_on = OpenAIChatResponseGenerator(
        mock_pipeline, emit_reasoning_content=True
    )
    response_on = await generator_on.complete([_make_mock_request()])
    body_on = jsonable_encoder(response_on)
    message_on = body_on["choices"][0]["message"]
    assert message_on["reasoning_content"] == "thinking"
    # Non-streaming route uses response_model=None → no exclude_none → null on wire.
    assert message_on["reasoning"] is None

    # --- Flag OFF (default): emit_reasoning_content=False ---
    mock_pipeline.all_tokens = AsyncMock(return_value=_REASONING_CONTENT_CHUNKS)
    generator_off = OpenAIChatResponseGenerator(
        mock_pipeline, emit_reasoning_content=False
    )
    response_off = await generator_off.complete([_make_mock_request()])
    body_off = jsonable_encoder(response_off)
    message_off = body_off["choices"][0]["message"]
    assert message_off["reasoning"] == "thinking"
    # The unselected field is null-not-absent on this path (same behavior as above).
    assert message_off["reasoning_content"] is None


@pytest.mark.asyncio
async def test_parse_chat_completion_accepts_replayed_reasoning_key() -> None:
    """Assistant turns replaying MAX's own ``reasoning`` key carry CoT forward.

    By default (``emit_reasoning_content=False``) MAX emits prior-turn
    reasoning under the ``reasoning`` JSON key. A client that echoes MAX's
    assistant output back into a follow-up request therefore sends
    ``reasoning`` (not ``reasoning_content``). The parser must read both so
    the chain-of-thought is not silently dropped before the chat template
    runs. Mirrors the agentic replay: assistant reasoning + tool_calls
    followed by the tool reply.
    """
    settings = Settings(api_types=[APIType.OPENAI], use_heartbeat=False)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "The user wants weather; call the tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                },
            ],
        }
    )

    parsed = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )

    assistant = parsed.messages[1]
    assert assistant.role == "assistant"
    assert (
        assistant.reasoning_content == "The user wants weather; call the tool."
    )


@pytest.mark.asyncio
async def test_parse_chat_completion_reasoning_content_key_and_precedence() -> (
    None
):
    """``reasoning_content`` still works and wins when both keys are present."""
    settings = Settings(api_types=[APIType.OPENAI], use_heartbeat=False)

    # ``reasoning_content`` alone (no regression).
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "explicit content key",
                },
            ],
        }
    )
    parsed = await openai_parse_chat_completion_request(
        request, wrap_content=False, settings=settings
    )
    assert parsed.messages[0].reasoning_content == "explicit content key"

    # Both keys present: ``reasoning_content`` takes precedence (``or`` semantics).
    request_both = CreateChatCompletionRequest.model_validate(
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "wins",
                    "reasoning": "loses",
                },
            ],
        }
    )
    parsed_both = await openai_parse_chat_completion_request(
        request_both, wrap_content=False, settings=settings
    )
    assert parsed_both.messages[0].reasoning_content == "wins"
