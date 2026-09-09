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


"""Unit tests for serve/pipelines/llm.py."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    IdentityPipelineTokenizer,
    PipelineConfig,
)
from max.pipelines.modeling.types import (
    Pipeline,
    PipelinesFactory,
    PipelineTask,
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.mocks.mock_api_requests import simple_openai_request
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.llm import (
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.telemetry.stopwatch import StopWatch

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class MockContext(Mock):
    """Mock context that implements BaseContext protocol."""

    request_id: RequestID
    status: GenerationStatus = GenerationStatus.ACTIVE

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


class MockValueErrorTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    """A mock generator that throws a value error when used."""

    def execute(
        self,
        inputs: TextGenerationInputs[MockContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@dataclass(frozen=True)
class MockTokenizer(IdentityPipelineTokenizer[str]):
    async def new_context(self, request: TextGenerationRequest) -> str:
        return ""


@pytest.fixture
def model_factory(request: pytest.FixtureRequest) -> PipelinesFactory:  # type: ignore[type-arg]
    """Fixture for a pipeline's generator
    This is bound indirectly - hence the request.param pattern.
    See https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    return request.param


@pytest.fixture(scope="function")
def app(
    model_factory: PipelinesFactory,  # type: ignore[type-arg]
    mock_pipeline_config: PipelineConfig,
) -> Generator[FastAPI, None, None]:
    """Fixture for a FastAPI app using a given pipeline."""
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=model_factory,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], use_heartbeat=False),
        serving_settings,
    )
    yield app


@pytest.fixture
def reset_sse_starlette_appstatus_event() -> None:
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None


@pytest_asyncio.fixture
async def test_client(app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """Fixture for a asgi TestClient using a given FastAPI app."""
    async with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("model_factory", [EchoTokenGenerator], indirect=True)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.parametrize("request_json", [None, "{{}"])
@pytest.mark.asyncio
async def test_llm_json_missing(
    test_client: TestClient,
    request_url: str,
    request_json: dict[str, Any] | None,
) -> None:
    """Test the server's response to malformed JSON."""
    logger.info("Test: Running Client: %s", request_url)
    response = await test_client.post(request_url, json=request_json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize(
    "model_factory", [MockValueErrorTokenGenerator], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error(
    test_client: TestClient, request_url: str
) -> None:
    """Test the server's response to a value error when calling new context."""
    request_json = {
        "model": "test",
        "prompt": "test",
        "temperature": 0.7,
        "stream": True,
    }
    # request_json = simple_openai_request(model_name="test", content="test")
    response = await test_client.post(request_url, json=request_json)
    assert response.status_code == 400


@pytest.mark.skip("TODO(ylou): Restore!!")
@pytest.mark.parametrize(
    "model_factory", [MockValueErrorTokenGenerator], indirect=True
)
@pytest.mark.parametrize(
    "request_url", ["/v1/chat/completions", "/v1/completions"]
)
@pytest.mark.asyncio
async def test_llm_new_context_value_error_stream(
    test_client: TestClient,
    request_url: str,
) -> None:
    """Test the server's response to a value error when calling new context while streaming."""
    MAX_CHUNK_TO_READ_BYTES = 10 * 1024

    payload = simple_openai_request(model_name="test", content="test")
    payload["stream"] = True
    # Prompt is required for completions endpoint.
    payload["prompt"] = "test prompt"
    response = await test_client.post(request_url, json=payload, stream=True)
    assert response.status_code == 200

    async for chunk in response.iter_content(MAX_CHUNK_TO_READ_BYTES):
        chunk = chunk.decode("utf-8").strip()[len("data: ") :]
        chunk = json.loads(chunk)
        assert chunk["result"] == "error"
        break


@pytest.mark.asyncio
async def test_ttft_recorded_once_per_chunk() -> None:
    """Test that TTFT is recorded exactly once per request, with ITL per chunk."""
    mock_metrics = MagicMock()

    # Create 3 chunks with 2, 3, 2 tokens = 7 total
    # Expect: 1 TTFT (first chunk), 2 ITLs (remaining 2 chunks)
    test_request_id = RequestID(value="test-request")
    scheduler_responses = [
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[101, 102],
            final_status=GenerationStatus.ACTIVE,
        ),
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[103, 104, 105],
            final_status=GenerationStatus.ACTIVE,
        ),
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[106, 107],
            final_status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[TextGenerationOutput], int | None], None]:
        async def _gen() -> AsyncGenerator[
            tuple[list[TextGenerationOutput], int | None], None
        ]:
            for response in scheduler_responses:
                yield [response], None

        return _gen()

    # Mock context returned by tokenizer
    mock_tokens = Mock()
    mock_tokens.prompt_length = 10

    mock_context = Mock(request_id=test_request_id, tokens=mock_tokens)

    # Mock request
    mock_request = Mock(request_id=test_request_id, tools=None, timestamp_ns=0)
    mock_request.sampling_params.stop = []

    # Create pipeline mock - Mock() auto-generates nested attributes
    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(return_value=mock_context)
    # Mock decode to return combined tokens text
    pipeline.tokenizer.decode = AsyncMock(return_value="chunk_text")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    # Match the real ctor contract: coalescing is off by default (floor == 1).
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=None)

    # Patch METRICS and call the real next_token_chunk method.
    # Binding lets us test real method logic with our mock pipeline.
    with patch("max.serve.pipelines.llm.METRICS", mock_metrics):
        bound_method = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in await bound_method(mock_request)]

    # Verify TTFT called exactly once, ITL called for remaining 2 chunks
    assert mock_metrics.ttft.call_count == 1
    assert mock_metrics.itl.call_count == 2
    assert len(chunks) == 3

    # TPOT is emitted exactly once per request, after the stream completes:
    # decode_elapsed_ms / (num_generated_tokens - 1). With 7 generated tokens
    # the denominator is 6, and the decode span is a small positive duration.
    assert mock_metrics.time_per_output_token.call_count == 1
    (tpot_value,) = mock_metrics.time_per_output_token.call_args.args
    assert tpot_value >= 0.0

    # Verify token counts are preserved
    total_tokens = sum(chunk.token_count for chunk in chunks)
    assert total_tokens == 7

    # Verify each chunk has the expected token count
    assert chunks[0].token_count == 2
    assert chunks[1].token_count == 3
    assert chunks[2].token_count == 2

    # Verify decoded_tokens is set for each chunk
    for chunk in chunks:
        assert chunk.decoded_tokens == "chunk_text"

    # Verify prompt_token_count uses prompt_length
    for chunk in chunks:
        assert chunk.prompt_token_count == 10

    # next_token_chunk must not emit the input-token counter; that is owned
    # by record_request_end so the counter stays consistent with the
    # per-request histogram and the API usage field.
    mock_metrics.input_tokens.assert_not_called()


async def _recorded_ttft_ms(timestamp_ns: int) -> float:
    """Run a one-chunk pipeline for a request that arrived at ``timestamp_ns``."""
    mock_metrics = MagicMock()
    test_request_id = RequestID(value="test-request")

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[TextGenerationOutput], int | None], None]:
        async def _gen() -> AsyncGenerator[
            tuple[list[TextGenerationOutput], int | None], None
        ]:
            yield (
                [
                    TextGenerationOutput(
                        request_id=test_request_id,
                        tokens=[101],
                        final_status=GenerationStatus.END_OF_SEQUENCE,
                    )
                ],
                0,
            )

        return _gen()

    mock_tokens = Mock()
    mock_tokens.prompt_length = 10
    mock_request = Mock(
        request_id=test_request_id, tools=None, timestamp_ns=timestamp_ns
    )
    mock_request.sampling_params.stop = []

    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(
        return_value=Mock(request_id=test_request_id, tokens=mock_tokens)
    )
    pipeline.tokenizer.decode = AsyncMock(return_value="chunk_text")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=None)

    with patch("max.serve.pipelines.llm.METRICS", mock_metrics):
        bound = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        _ = [chunk async for chunk in await bound(mock_request)]

    (ttft_value,) = mock_metrics.ttft.call_args.args
    assert isinstance(ttft_value, float)
    return ttft_value


@pytest.mark.asyncio
async def test_ttft_measured_from_request_arrival() -> None:
    """TTFT covers the pre-pipeline work the HTTP layer did (MXSERV-336).

    ``timestamp_ns`` is stamped by the request middleware before the body is
    parsed, so time spent in parsing, validation and media resolution belongs
    in TTFT rather than being dropped by starting the clock at pipeline entry.
    """
    pre_pipeline_ms = 500.0
    ttft_ms = await _recorded_ttft_ms(
        StopWatch.time_ns() - int(pre_pipeline_ms * 1e6)
    )
    assert ttft_ms >= pre_pipeline_ms


@pytest.mark.asyncio
async def test_ttft_falls_back_to_pipeline_entry_without_arrival_time() -> None:
    """Offline callers leave ``timestamp_ns`` at 0, so TTFT starts at entry.

    ``StopWatch`` treats a zero start as "not started" and raises, so the
    unset default must not reach it.
    """
    ttft_ms = await _recorded_ttft_ms(0)
    assert 0.0 <= ttft_ms < 1000.0


@pytest.mark.asyncio
async def test_tpot_not_recorded_for_single_token() -> None:
    """TPOT is skipped when only one token is generated (no decode span).

    The guard ``num_generated_tokens > 1`` mirrors vLLM's
    ``num_generation_tokens - 1 > 0`` and avoids a divide-by-zero.
    """
    mock_metrics = MagicMock()

    test_request_id = RequestID(value="test-request")
    scheduler_responses = [
        TextGenerationOutput(
            request_id=test_request_id,
            tokens=[101],
            final_status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[TextGenerationOutput], int | None], None]:
        async def _gen() -> AsyncGenerator[
            tuple[list[TextGenerationOutput], int | None], None
        ]:
            for response in scheduler_responses:
                yield [response], None

        return _gen()

    mock_tokens = Mock()
    mock_tokens.prompt_length = 10
    mock_context = Mock(request_id=test_request_id, tokens=mock_tokens)
    mock_request = Mock(request_id=test_request_id, tools=None, timestamp_ns=0)
    mock_request.sampling_params.stop = []

    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(return_value=mock_context)
    pipeline.tokenizer.decode = AsyncMock(return_value="chunk_text")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    # Match the real ctor contract: coalescing is off by default (floor == 1).
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=None)

    with patch("max.serve.pipelines.llm.METRICS", mock_metrics):
        bound_method = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in await bound_method(mock_request)]

    assert len(chunks) == 1
    # One token generated -> no inter-token span, so TPOT is not emitted.
    assert mock_metrics.time_per_output_token.call_count == 0


THINK_START_TOKEN_ID = 1
THINK_END_TOKEN_ID = 2
TOOL_SECTION_START_TOKEN_ID = 3


async def _run_reasoning_pipeline(
    scheduler_responses: list[TextGenerationOutput],
    *,
    prompt_tokens: list[int] | None = None,
    decode: Any = None,
    stop: list[str] | None = None,
    top_log_probs: AsyncMock | None = None,
) -> list[TokenGeneratorOutput]:
    """Build a mock TokenGeneratorPipeline with reasoning and collect all chunks."""
    from max.pipelines.architectures.kimik2_5.reasoning import (
        KimiK2_5ReasoningParser,
    )

    test_request_id = RequestID(value="test-request")

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[TextGenerationOutput], int | None], None]:
        async def _gen() -> AsyncGenerator[
            tuple[list[TextGenerationOutput], int | None], None
        ]:
            for response in scheduler_responses:
                yield [response], None

        return _gen()

    mock_tokens = Mock()
    mock_tokens.prompt = (
        prompt_tokens if prompt_tokens is not None else [99, 98]
    )
    mock_tokens.prompt_length = 10

    mock_request = Mock(request_id=test_request_id, tools=None, timestamp_ns=0)
    mock_request.sampling_params.stop = stop or []

    pipeline = Mock()
    mock_context = Mock(request_id=test_request_id, tokens=mock_tokens)
    # Explicitly set grammar/json_schema to None so getattr doesn't return Mock
    mock_context.grammar = None
    mock_context.json_schema = None
    pipeline.tokenizer.new_context = AsyncMock(return_value=mock_context)
    pipeline.tokenizer.decode = decode or AsyncMock(return_value="decoded_text")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    # Match the real ctor contract: coalescing is off by default (floor == 1).
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(
        return_value=KimiK2_5ReasoningParser(
            think_start_token_id=THINK_START_TOKEN_ID,
            think_end_token_id=THINK_END_TOKEN_ID,
            tool_section_start_token_id=TOOL_SECTION_START_TOKEN_ID,
        )
    )
    if top_log_probs is not None:
        pipeline._top_log_probs = top_log_probs

    with patch("max.serve.pipelines.llm.METRICS", MagicMock()):
        bound = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        return [chunk async for chunk in await bound(mock_request)]


def _make_responses(
    token_lists: list[list[int]],
    **kwargs: Any,
) -> list[TextGenerationOutput]:
    """Build TextGenerationOutput list from token lists (last is EOS)."""
    request_id = RequestID(value="test-request")
    return [
        TextGenerationOutput(
            request_id=request_id,
            tokens=tokens,
            final_status=(
                GenerationStatus.END_OF_SEQUENCE
                if i == len(token_lists) - 1
                else GenerationStatus.ACTIVE
            ),
            **kwargs,
        )
        for i, tokens in enumerate(token_lists)
    ]


async def _input_sensitive_decode(token_array: Any, **kwargs: Any) -> str:
    tokens = token_array.tolist()
    if tokens == [10, 20]:
        return "reasoning_text"
    elif tokens == [30]:
        return "content_text"
    return "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_lists,prompt_tokens,decode,expected_chunks",
    [
        pytest.param(
            [[THINK_START_TOKEN_ID, 10, 20, THINK_END_TOKEN_ID], [30]],
            None,
            _input_sensitive_decode,
            [(2, 0, "reasoning_text", None), (0, 1, None, "content_text")],
            id="reasoning_then_content",
        ),
        pytest.param(
            [[10, 20]],
            [99, THINK_END_TOKEN_ID, 98],
            None,
            [(0, 2, None, "decoded_text")],
            id="reasoning_disabled_by_prompt",
        ),
        pytest.param(
            [
                [THINK_START_TOKEN_ID, 10],
                [20, THINK_END_TOKEN_ID],
                [30],
            ],
            None,
            None,
            [
                (1, 0, "decoded_text", None),
                (1, 0, "decoded_text", None),
                (0, 1, None, "decoded_text"),
            ],
            id="multi_chunk_reasoning",
        ),
        pytest.param(
            [[THINK_START_TOKEN_ID, 10, 20], [30]],
            None,
            None,
            [(2, 0, "decoded_text", None), (1, 0, "decoded_text", None)],
            id="all_reasoning_no_content",
        ),
    ],
)
async def test_next_token_chunk_reasoning(
    token_lists: list[list[int]],
    prompt_tokens: list[int] | None,
    decode: Any,
    expected_chunks: list[tuple[int, int, str | None, str | None]],
) -> None:
    chunks = await _run_reasoning_pipeline(
        _make_responses(token_lists),
        prompt_tokens=prompt_tokens,
        decode=decode,
    )
    assert len(chunks) == len(expected_chunks)
    for chunk, (r_count, t_count, r_text, t_text) in zip(
        chunks, expected_chunks, strict=True
    ):
        assert chunk.reasoning_token_count == r_count
        assert chunk.token_count == t_count
        assert chunk.decoded_reasoning_tokens == r_text
        assert chunk.decoded_tokens == t_text


@pytest.mark.asyncio
async def test_delimiter_only_generation_reports_prompt_and_cached_tokens() -> (
    None
):
    """A generation whose only token is a stripped reasoning delimiter must
    still carry the prompt and cached token counts on its terminal chunk, and
    must still bill for the delimiter token itself.

    With max_tokens=1 a reasoning model's single generated token is often the
    think-start delimiter; the parser strips it, so the terminal chunk is the
    only one the streaming route ever sees. Without prompt_token_count on it,
    the final usage chunk reports prompt_tokens=0 (CLIN-1523); without
    cached_token_count the prefix-cache hits never surface either. Because
    this terminal chunk is the first chunk, it carries num_cached_tokens the
    same way the regular first-chunk path does.

    The delimiter token was still generated and spent the max_tokens budget,
    so it must count toward completion_tokens as a reasoning token -- reporting
    token_count=0 with no reasoning_token_count under-bills the request
    (CENG-932).
    """
    chunks = await _run_reasoning_pipeline(
        _make_responses([[THINK_START_TOKEN_ID]], num_cached_tokens=7)
    )
    assert len(chunks) == 1
    assert chunks[0].token_count == 0
    assert chunks[0].reasoning_token_count == 1
    assert chunks[0].prompt_token_count == 10
    assert chunks[0].cached_token_count == 7


@pytest.mark.asyncio
async def test_next_token_chunk_reasoning_partitions_logprobs() -> None:
    """Test that logprobs are correctly partitioned between reasoning and content."""
    logprob_content = Mock()
    logprob_content.token_log_probabilities = [-0.4]

    responses = [
        TextGenerationOutput(
            request_id=RequestID(value="test-request"),
            tokens=[THINK_START_TOKEN_ID, 10, THINK_END_TOKEN_ID, 30],
            log_probabilities=[Mock(), Mock(), Mock(), logprob_content],
            final_status=GenerationStatus.END_OF_SEQUENCE,
        ),
    ]

    chunks = await _run_reasoning_pipeline(
        responses,
        decode=AsyncMock(return_value="text"),
        top_log_probs=AsyncMock(return_value=[{"tok": -0.5}]),
    )

    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].reasoning_token_count == 1
    assert chunks[0].token_log_probabilities == [-0.4]
    assert chunks[0].top_log_probabilities == [{"tok": -0.5}]


@pytest.mark.asyncio
async def test_next_token_chunk_stop_sequence_ignores_reasoning() -> None:
    """Stop sequences only match against content tokens, not reasoning tokens."""

    async def mock_decode(token_array: Any, **kwargs: Any) -> str:
        tokens = token_array.tolist()
        if tokens == [10, 20]:
            return "STOP"
        elif tokens == [30]:
            return "hello"
        return "unknown"

    chunks = await _run_reasoning_pipeline(
        _make_responses(
            [[THINK_START_TOKEN_ID, 10, 20, THINK_END_TOKEN_ID], [30]]
        ),
        decode=mock_decode,
        stop=["STOP"],
    )

    assert len(chunks) == 2
    assert chunks[0].decoded_reasoning_tokens == "STOP"
    assert chunks[0].decoded_tokens is None
    assert chunks[1].decoded_tokens == "hello"


@pytest.mark.asyncio
async def test_next_token_chunk_tool_section_without_think_end_to_content() -> (
    None
):
    """Kimi K2.5 can open a tool-call section from inside ``<think>`` with no
    closing ``</think>``. The tool section must route to *content* (where the
    tool parser runs), not leak into the reasoning channel.

    Regression for the intermittent OpenRouter ``tool-choice-auto`` failure:
    the tool-call payload landed in ``reasoning`` and ``content`` was empty,
    so no tool call was ever emitted. Sampling-dependent, hence flaky.
    """

    async def mock_decode(token_array: Any, **kwargs: Any) -> str:
        tokens = token_array.tolist()
        if tokens == [10]:
            return "thinking"
        if tokens == [TOOL_SECTION_START_TOKEN_ID, 40]:
            # The tool markers are non-special tokens, so they survive
            # detokenization and reach the tool parser verbatim.
            return "<|tool_calls_section_begin|>...args..."
        return "unknown"

    chunks = await _run_reasoning_pipeline(
        _make_responses(
            [[THINK_START_TOKEN_ID, 10], [TOOL_SECTION_START_TOKEN_ID, 40]]
        ),
        decode=mock_decode,
    )

    assert len(chunks) == 2
    # First chunk is pure reasoning (still inside <think>).
    assert chunks[0].decoded_reasoning_tokens == "thinking"
    assert chunks[0].decoded_tokens is None
    # Second chunk: the tool section ends reasoning and routes to content,
    # NOT reasoning — so the tool parser downstream actually sees it.
    assert chunks[1].decoded_reasoning_tokens is None
    assert chunks[1].reasoning_token_count == 0
    assert chunks[1].decoded_tokens == "<|tool_calls_section_begin|>...args..."
    assert chunks[1].token_count == 2


@pytest.mark.asyncio
async def test_next_token_chunk_stop_sequence_sets_eos_status() -> None:
    """Status is END_OF_SEQUENCE when a stop sequence matches, even if the
    model response itself is still ACTIVE."""
    test_request_id = RequestID(value="test-request")

    async def mock_stream(
        request_id: str, context: Any
    ) -> AsyncGenerator[tuple[list[TextGenerationOutput], int | None], None]:
        async def _gen() -> AsyncGenerator[
            tuple[list[TextGenerationOutput], int | None], None
        ]:
            yield (
                [
                    TextGenerationOutput(
                        request_id=test_request_id,
                        tokens=[10],
                        final_status=GenerationStatus.ACTIVE,
                    )
                ],
                None,
            )

        return _gen()

    mock_context = Mock(
        request_id=test_request_id,
        tokens=Mock(prompt_length=5, prompt=[99]),
    )
    mock_context.eos_tracker.eos_stop_strings = ["stop_word"]
    mock_context.eos_tracker.is_eos_from_string.return_value = "stop_word"

    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(return_value=mock_context)
    pipeline.tokenizer.decode = AsyncMock(return_value="stop_word")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    # Match the real ctor contract: coalescing is off by default (floor == 1).
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=None)

    mock_request = Mock(
        request_id=test_request_id,
        tools=None,
        timestamp_ns=0,
        sampling_params=Mock(stop=["stop_word"]),
    )

    with patch("max.serve.pipelines.llm.METRICS", MagicMock()):
        bound = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in await bound(mock_request)]

    assert len(chunks) == 1
    assert chunks[0].status == GenerationStatus.END_OF_SEQUENCE
