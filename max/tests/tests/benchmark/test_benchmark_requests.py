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

"""Unit tests for benchmark_shared.request module."""

import concurrent.futures
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from max.benchmark.benchmark_serving import validate_task_and_endpoint
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    PixelGenerationImageOptions,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.request import (
    OpenAIChatCompletionsRequestDriver,
    OpenAICompletionsRequestDriver,
    OpenResponsesRequestDriver,
    PixelGenerationRequestFuncInput,
    ProgressBarRequestDriver,
    RequestCounter,
    RequestFuncInput,
    RequestFuncOutput,
    SglangPixelGenerationRequestDriver,
    SglangVideoRequestDriver,
    TRTLLMRequestDriver,
    VllmOmniPixelGenerationRequestDriver,
    VllmOmniVideoRequestDriver,
    _build_final_payload,
    _build_sglang_pixel_generation_payload,
    _build_sglang_video_payload,
    _build_vllm_omni_pixel_generation_payload,
    _build_vllm_omni_video_payload,
    _count_generated_media,
    async_request_lora_load,
    async_request_lora_unload,
    get_request_driver_class,
    mark_cancelled_if_past_deadline,
)
from pytest_mock import MockerFixture
from tqdm.asyncio import tqdm


@pytest.fixture
def mock_aiohttp_session(mocker: MockerFixture) -> Any:
    """Fixture to create a properly mocked aiohttp ClientSession.

    Returns a mock client with a helper method to easily setup POST responses.
    """
    mock_session_class = mocker.patch(
        "max.benchmark.benchmark_shared.request.aiohttp.ClientSession"
    )
    mock_session = mock_session_class.return_value
    mock_client = mock_session.__aenter__.return_value

    def setup_post_response(response_mock: Any) -> None:
        """Helper to setup POST response mock."""
        # Create an async context manager mock for the post method
        mock_post_context = mocker.AsyncMock()
        mock_post_context.__aenter__ = mocker.AsyncMock(
            return_value=response_mock
        )
        mock_post_context.__aexit__ = mocker.AsyncMock(return_value=None)
        mock_client.post = mocker.Mock(return_value=mock_post_context)

    mock_client.setup_post_response = setup_post_response
    return mock_client


@pytest.fixture
def mock_openai_env(mocker: MockerFixture) -> None:
    """Fixture to mock OpenAI API key environment variable."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})


class TestRequestFuncInput:
    """Test cases for RequestFuncInput dataclass."""

    def test_request_func_input_creation(self) -> None:
        """Test creating a RequestFuncInput with required fields."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        assert request_input.prompt == "Test prompt"
        assert request_input.images == []
        assert request_input.api_url == "http://localhost:8000/completions"
        assert request_input.prompt_len == 10
        assert request_input.max_tokens == 100
        assert request_input.ignore_eos is False
        assert request_input.model == "test-model"
        assert request_input.session_id is None
        assert request_input.sampling.temperature is None
        assert request_input.sampling.top_p is None
        assert request_input.sampling.top_k is None

    def test_request_func_input_with_optional_fields(self) -> None:
        """Test creating a RequestFuncInput with optional fields."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id="session-123",
            sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        assert request_input.session_id == "session-123"
        assert request_input.sampling.temperature == 0.7
        assert request_input.sampling.top_p == 0.9
        assert request_input.sampling.top_k == 50

    def test_request_func_input_with_chat_messages(self) -> None:
        """Test creating a RequestFuncInput with chat messages."""
        messages = [
            ChatMessage(
                role="user",
                content=[TextContentBlock(type="text", text="Hello")],
            ),
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(type="text", text="Hi there!")],
            ),
        ]

        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt=messages,
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=20,
            max_tokens=50,
            ignore_eos=True,
        )

        assert request_input.prompt == messages
        assert isinstance(request_input.prompt, list)


class TestRequestFuncOutput:
    """Test cases for RequestFuncOutput dataclass."""

    def test_request_func_output_defaults(self) -> None:
        """Test RequestFuncOutput with default values."""
        output = RequestFuncOutput()

        assert output.cancelled is False
        assert output.generated_text == ""
        assert output.success is False
        assert output.latency == 0.0
        assert output.ttft == 0.0
        assert output.itl == []
        assert output.prompt_len == 0
        assert output.error == ""

    def test_request_func_output_with_values(self) -> None:
        """Test RequestFuncOutput with custom values."""
        output = RequestFuncOutput(
            cancelled=True,
            generated_text="Hello world",
            success=True,
            latency=1.5,
            ttft=0.1,
            itl=[0.05, 0.06, 0.07],
            prompt_len=10,
            error="",
        )

        assert output.cancelled is True
        assert output.generated_text == "Hello world"
        assert output.success is True
        assert output.latency == 1.5
        assert output.ttft == 0.1
        assert output.itl == [0.05, 0.06, 0.07]
        assert output.prompt_len == 10
        assert output.error == ""

    def test_request_func_output_itl_field_default(self) -> None:
        """Test that itl field has proper default factory."""
        output1 = RequestFuncOutput()
        output2 = RequestFuncOutput()

        # Both should have empty lists, not the same list object
        assert output1.itl == []
        assert output2.itl == []
        assert output1.itl is not output2.itl


class TestRequestCounter:
    """Test cases for RequestCounter class."""

    def test_request_counter_advance_success(self) -> None:
        """Test successful request counter advancement."""
        counter = RequestCounter(max_requests=5)

        # Should be able to advance 5 times
        for i in range(5):
            result = counter.advance_until_max()
            assert result is True
            assert counter.total_sent_requests == i + 1

    def test_request_counter_advance_max_reached(self) -> None:
        """Test request counter when max requests reached."""
        counter = RequestCounter(max_requests=3)

        # Advance 3 times successfully
        for _ in range(3):
            result = counter.advance_until_max()
            assert result is True

        # 4th attempt should fail
        result = counter.advance_until_max()
        assert result is False
        assert counter.total_sent_requests == 3

    def test_request_counter_concurrent_access(self) -> None:
        """Test request counter with concurrent access."""
        counter = RequestCounter(max_requests=10)

        def advance_counter() -> bool:
            return counter.advance_until_max()

        # Create multiple concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(advance_counter) for _ in range(10)]
            results = [future.result() for future in futures]

        # All should succeed (race conditions may cause slight overage)
        assert all(results)
        # Due to race conditions, we might exceed by 1-2, which is acceptable
        assert counter.total_sent_requests >= 10
        assert counter.total_sent_requests <= 12  # Reasonable upper bound

    def test_request_counter_with_initial_count(self) -> None:
        """Test request counter with initial sent requests count."""
        counter = RequestCounter(max_requests=5, total_sent_requests=3)

        # Should be able to advance 2 more times
        for _ in range(2):
            result = counter.advance_until_max()
            assert result is True

        # Next attempt should fail
        result = counter.advance_until_max()
        assert result is False
        assert counter.total_sent_requests == 5


class TestAsyncRequestFunctions:
    """Test cases for async request functions (LoRA operations)."""

    @pytest.mark.asyncio
    async def test_async_request_lora_load_success(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test successful async_request_lora_load call."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200

        mock_aiohttp_session.setup_post_response(mock_response)

        success, load_time = await async_request_lora_load(
            "http://localhost:8000", "test-lora", "/path/to/lora"
        )

        assert success is True
        assert load_time >= 0

    @pytest.mark.asyncio
    async def test_async_request_lora_load_failure(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test async_request_lora_load with failure."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 400
        mock_response.text = mocker.AsyncMock(return_value="Bad Request")

        mock_aiohttp_session.setup_post_response(mock_response)

        success, load_time = await async_request_lora_load(
            "http://localhost:8000", "test-lora", "/path/to/lora"
        )

        assert success is False
        assert load_time >= 0

    @pytest.mark.asyncio
    async def test_async_request_lora_unload_success(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test successful async_request_lora_unload call."""
        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200

        mock_aiohttp_session.setup_post_response(mock_response)

        success, unload_time = await async_request_lora_unload(
            "http://localhost:8000", "test-lora"
        )

        assert success is True
        assert unload_time >= 0


class TestRequestDriver:
    """Test cases for RequestDriver classes."""

    @pytest.mark.asyncio
    async def test_openai_completions_request_driver_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test successful OpenAICompletionsRequestDriver request."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"text": "Hello"}]}\n\n',
            b'data: {"choices": [{"text": " world"}]}\n\n',
            b'data: {"choices": [{"text": "!"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAICompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Hello world!"
        assert result.prompt_len == 10

    @pytest.mark.asyncio
    async def test_openai_chat_completions_request_driver_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test successful OpenAIChatCompletionsRequestDriver request."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Mock response data
        mock_response_data = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "!"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Hello world!"
        assert result.prompt_len == 10

    async def test_openai_chat_completions_choices_but_no_text_is_failure(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Choices present but no modeled text field must fail, not succeed with ttft=0.

        Regression test for PERF-2615: gpt-oss streamed chunks that had
        ``choices`` but carried text in a delta field this client does not
        model, so ``reasoning``/``reasoning_content``/``content`` were all
        empty. Those runs were marked success with ttft=0 and no tokens, which
        zeroed out TTFT/TPOT for the whole run and produced a misleading
        "0 valid requests" steady-state warning. Such a response must now be a
        failure.
        """
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Chunks have choices but the text is in an unmodeled delta field;
        # reasoning/reasoning_content/content are all absent, so no text is
        # captured.
        mock_response_data = [
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
            b'data: {"choices": [{"delta": {"unmodeled_channel": "hi"}}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert result.ttft == 0.0
        assert result.generated_text == ""
        assert result.error is not None
        assert "No text content" in result.error

    @pytest.mark.asyncio
    async def test_openai_chat_completions_tool_call_only_response_is_success(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """A pure tool-call turn is a successful, content-bearing response.

        Regression test for the nemotron-opencode "No text content captured"
        failures: an agentic response can consist entirely of ``tool_calls``
        deltas (no ``content``/``reasoning`` at all). The client must model
        the ``tool_calls`` field and treat those chunks as content-bearing
        rather than reporting the request as failed.
        """
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Write a file",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Mirrors the server's streaming shape for a tool-call-only turn:
        # an opener chunk with id + name, then argument fragments.
        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {"role": "assistant", '
                b'"tool_calls": [{"index": 0, "id": "write:fa3c5398", '
                b'"type": "function", '
                b'"function": {"name": "write", "arguments": "{\\""}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
                b'"function": {"arguments": "content\\": \\"hi\\"}"}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {}, '
                b'"finish_reason": "length"}]}\n\n'
            ),
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.error == ""
        assert result.ttft > 0.0
        # The tool name and argument bytes are the generated text.
        assert result.generated_text == 'write{"content": "hi"}'

    @pytest.mark.asyncio
    async def test_openai_chat_completions_content_before_tool_call_is_not_double_counted(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Content and tool_calls deltas are additive, never overlapping.

        Regression guard for double-counting: a turn that streams prose
        before deciding to call a tool must count that prose exactly once,
        even though ``_extract_chat_delta_text`` sums ``content`` and
        ``tool_calls`` text from the same delta object. The server only ever
        populates one of the two per ``ParsedToolCallDelta``, so summing them
        is additive by construction -- this test pins that with an exact
        length/text check rather than relying on that invariant by
        inspection alone.
        """
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="What's the weather, then check the time",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {"role": "assistant", '
                b'"content": "Let me check that for you."}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {'
                b'"tool_calls": [{"index": 0, "id": "get_weather:1", '
                b'"type": "function", '
                b'"function": {"name": "get_weather", "arguments": "{}"}}]}}]}\n\n'
            ),
            (
                b'data: {"choices": [{"delta": {}, '
                b'"finish_reason": "tool_calls"}]}\n\n'
            ),
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        # The prose and the tool call are disjoint deltas; the total is
        # their concatenation, not double the prose or double the call.
        assert (
            result.generated_text == "Let me check that for you.get_weather{}"
        )
        assert result.generated_text.count("Let me check that for you.") == 1

    @pytest.mark.asyncio
    async def test_openai_chat_completions_merges_reasoning_reasoning_content_and_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that reasoning, reasoning_content, and content are merged in order."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Think step by step",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=5,
            max_tokens=100,
            ignore_eos=False,
        )

        # Chunks: out-of-order reasoning, reasoning_content, and content
        mock_response_data = [
            (
                b'data: {"choices": [{"delta": {'
                b'"content": " The answer is 42.", '
                b'"reasoning": "Let me think.", '
                b'"reasoning_content": " chain-of-thought."'
                b"}}]}\n\n"
            ),
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert (
            result.generated_text
            == "Let me think. chain-of-thought. The answer is 42."
        ), "reasoning, reasoning_content, and content must be merged in order"
        assert result.prompt_len == 5

    @pytest.mark.asyncio
    async def test_openai_chat_completions_handles_missing_or_none_delta_keys(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing or None reasoning/reasoning_content/content are treated as empty."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test",
            images=[],
            api_url="http://localhost:8000/chat/completions",
            prompt_len=1,
            max_tokens=50,
            ignore_eos=False,
        )

        # Chunks with only some keys present or None; no "reasoning" or "reasoning_content"
        mock_response_data = [
            b'data: {"choices": [{"delta": {"reasoning": "Let me "}}]}\n\n',
            b'data: {"choices": [{"delta": {"reasoning": "think.", "content": " The"}}]}\n\n',
            b'data: {"choices": [{"delta": {"reasoning": null, "content": " answer"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " is 42."}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        assert result.generated_text == "Let me think. The answer is 42.", (
            "content-only and None keys must yield merged content"
        )
        assert result.prompt_len == 1

    @pytest.mark.asyncio
    async def test_openai_chat_completions_error_chunk_reports_server_message(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Regression test for MXTOOLS-203: server error message is surfaced in output.error."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="A very long prompt",
            images=[],
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=264408,
            max_tokens=100,
            ignore_eos=False,
        )

        error_msg = "Input string is larger than tokenizer's max length (264823 > 262144)."
        error_json = json.dumps(
            {
                "error": {
                    "code": "500",
                    "message": error_msg,
                    "param": "",
                    "type": "",
                }
            }
        )

        async def async_iter() -> AsyncIterator[bytes]:
            yield f"data: {error_json}\n\n".encode()

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert result.error == error_msg

    @pytest.mark.asyncio
    async def test_openai_chat_completions_forwards_tools(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """``tools`` on RequestFuncInput is forwarded as the chat ``tools`` field."""
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
        ]
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Run ls",
            images=[],
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=5,
            max_tokens=50,
            ignore_eos=False,
            tools=tool_schemas,
        )

        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        # Inspect the POST payload to confirm the tools made it through.
        _, post_kwargs = mock_aiohttp_session.post.call_args
        assert post_kwargs["json"]["tools"] == tool_schemas

    @pytest.mark.asyncio
    async def test_openai_chat_completions_omits_tools_when_none(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """No ``tools`` field is sent when SampledRequest.tools is None/empty."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="hi",
            images=[],
            api_url="http://localhost:8000/v1/chat/completions",
            prompt_len=1,
            max_tokens=8,
            ignore_eos=False,
            tools=None,
        )

        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True
        _, post_kwargs = mock_aiohttp_session.post.call_args
        assert "tools" not in post_kwargs["json"]

    @pytest.mark.asyncio
    async def test_openai_completions_request_driver_no_content(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing content returns an error."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/completions",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        mock_response_data = [
            b"data: [DONE]\n\n",
        ]

        async def async_iter() -> AsyncIterator[bytes]:
            for item in mock_response_data:
                yield item

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAICompletionsRequestDriver()
        result = await driver.request(request_input)

        assert result.success is False
        assert "No text content" in result.error

    @pytest.mark.asyncio
    async def test_request_driver_with_progress_bar(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test that RequestDriver works with progress bars."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Create a mock progress bar
        mock_pbar = mocker.Mock(spec=tqdm)

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data: {"text_output": "Hello"}\n\n'

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        base_driver = TRTLLMRequestDriver()
        driver = ProgressBarRequestDriver(base_driver, mock_pbar)
        result = await driver.request(request_input)

        # Verify progress bar was updated
        mock_pbar.update.assert_called_once_with(1)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_request_driver_without_progress_bar(
        self, mock_aiohttp_session: Any, mocker: MockerFixture
    ) -> None:
        """Test that RequestDriver works without progress bars."""
        request_input = RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="Test prompt",
            images=[],
            api_url="http://localhost:8000/generate_stream",
            prompt_len=10,
            max_tokens=100,
            ignore_eos=False,
        )

        # Create async iterator for response.content
        async def async_iter() -> AsyncIterator[bytes]:
            yield b'data: {"text_output": "Hello"}\n\n'

        # Setup the response object
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = async_iter()

        mock_aiohttp_session.setup_post_response(mock_response)

        driver = TRTLLMRequestDriver()
        result = await driver.request(request_input)

        assert result.success is True


class TestBuildFinalPayload:
    """Unit tests for the ``_build_final_payload`` payload helper."""

    def test_none_and_empty_return_copy_unchanged(self) -> None:
        payload = {"model": "m"}
        assert _build_final_payload(payload, None) == {"model": "m"}
        assert _build_final_payload(payload, {}) == {"model": "m"}
        # The helper returns a fresh dict; the caller's payload is not mutated.
        assert _build_final_payload(payload, {"x": 1}) is not payload
        assert "x" not in payload

    def test_adds_new_keys_with_nested_verbatim(self) -> None:
        payload: dict[str, Any] = {"model": "m"}
        extra = {
            "vendor_ext": {"flags": ["a", "b"]},
            "stop": ["}"],
        }
        body = _build_final_payload(payload, extra)
        assert body["vendor_ext"] == {"flags": ["a", "b"]}
        assert body["stop"] == ["}"]

    def test_collision_overwrites_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = {"max_tokens": 100, "stream": True}
        with caplog.at_level(
            logging.WARNING, logger="max.benchmark.benchmark_shared.request"
        ):
            body = _build_final_payload(payload, {"max_tokens": 15})
        # last-writer-wins
        assert body["max_tokens"] == 15
        assert body["stream"] is True
        # collision is surfaced
        assert any(
            "overwrites managed request field" in rec.message
            and "max_tokens" in rec.message
            for rec in caplog.records
        )


class TestDriverExtraBody:
    """End-to-end: extra_body reaches the POST payload of text-gen drivers."""

    _EXTRA = {
        "stop": ["}"],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        # Synthetic vendor extension: a nested object holding an array, to
        # confirm arbitrary nested structures pass through verbatim.
        "vendor_ext": {"flags": ["a", "b"]},
    }

    @staticmethod
    async def _single_chunk(chunk: bytes) -> AsyncIterator[bytes]:
        yield chunk
        yield b"data: [DONE]\n\n"

    def _make_input(self, api_url: str) -> RequestFuncInput:
        return RequestFuncInput(
            model="test-model",
            session_id=None,
            sampling=SamplingConfig(),
            prompt="hi",
            images=[],
            api_url=api_url,
            prompt_len=1,
            max_tokens=100,
            ignore_eos=False,
        )

    @pytest.mark.asyncio
    async def test_chat_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = self._single_chunk(
            b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        )
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAIChatCompletionsRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            self._make_input("http://localhost:8000/v1/chat/completions")
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_completions_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mock_openai_env: None,
        mocker: MockerFixture,
    ) -> None:
        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = self._single_chunk(
            b'data: {"choices": [{"text": "ok"}]}\n\n'
        )
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = OpenAICompletionsRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            self._make_input("http://localhost:8000/v1/completions")
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_trtllm_driver_merges_extra_body(
        self,
        mock_aiohttp_session: Any,
        mocker: MockerFixture,
    ) -> None:
        async def trtllm_chunk() -> AsyncIterator[bytes]:
            # The TRT-LLM driver validates every event as a chunk and has no
            # ``[DONE]`` sentinel, so emit only a single data line.
            yield b'data: {"text_output": "ok"}\n\n'

        mock_response = mocker.AsyncMock()
        mock_response.status = 200
        mock_response.content = trtllm_chunk()
        mock_aiohttp_session.setup_post_response(mock_response)

        driver = TRTLLMRequestDriver(extra_body=self._EXTRA)
        result = await driver.request(
            self._make_input(
                "http://localhost:8000/v2/models/ensemble/generate_stream"
            )
        )
        assert result.success is True
        sent = mock_aiohttp_session.post.call_args[1]["json"]
        assert sent["stop"] == ["}"]
        assert sent["vendor_ext"] == {"flags": ["a", "b"]}


class TestRequestDriverSelection:
    """Test cases for request driver selection."""

    def test_get_request_driver_class_completions(self) -> None:
        assert (
            get_request_driver_class("http://localhost:8000/v1/completions")
            is OpenAICompletionsRequestDriver
        )
        assert (
            get_request_driver_class("http://localhost:8000/v1/profile")
            is OpenAICompletionsRequestDriver
        )

    def test_get_request_driver_class_chat(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/chat/completions"
            )
            is OpenAIChatCompletionsRequestDriver
        )

    def test_get_request_driver_class_generate_stream(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v2/models/ensemble/generate_stream"
            )
            is TRTLLMRequestDriver
        )

    def test_get_request_driver_class_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class("http://localhost:8000/unsupported")

    # Pixel generation driver selection tests

    def test_get_request_driver_class_pixel_gen_responses(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/responses",
                task="text-to-image",
            )
            is OpenResponsesRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_sglang(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/images/generations",
                task="text-to-image",
            )
            is SglangPixelGenerationRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_vllm_omni(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/chat/completions",
                task="text-to-image",
            )
            is VllmOmniPixelGenerationRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_image_to_image(self) -> None:
        """image-to-image task should also dispatch to pixel gen drivers."""
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/images/generations",
                task="image-to-image",
            )
            is SglangPixelGenerationRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_vllm_omni_video(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/videos/sync",
                task="text-to-video",
            )
            is VllmOmniVideoRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_sglang_video(self) -> None:
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/videos",
                task="text-to-video",
            )
            is SglangVideoRequestDriver
        )

    def test_get_request_driver_class_pixel_gen_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="pixel-generation"):
            get_request_driver_class(
                "http://localhost:8000/v1/completions",
                task="text-to-image",
            )

    def test_get_request_driver_class_text_gen_videos_sync_rejected(
        self,
    ) -> None:
        """text-generation + /v1/videos/sync should raise, not return the
        video driver."""
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class(
                "http://localhost:8000/v1/videos/sync",
                task="text-generation",
            )

    def test_get_request_driver_class_text_gen_videos_rejected(self) -> None:
        """text-generation + /v1/videos should raise."""
        with pytest.raises(ValueError, match="Unsupported API URL"):
            get_request_driver_class(
                "http://localhost:8000/v1/videos",
                task="text-generation",
            )

    def test_get_request_driver_class_text_gen_chat_not_pixel(self) -> None:
        """text-generation + /v1/chat/completions should return chat driver,
        not VllmOmniPixelGenerationRequestDriver."""
        assert (
            get_request_driver_class(
                "http://localhost:8000/v1/chat/completions",
                task="text-generation",
            )
            is OpenAIChatCompletionsRequestDriver
        )


class TestPixelGenerationPayloadBuilders:
    """Tests for sglang and vllm omni pixel generation payload builders."""

    def _make_input(
        self,
        image_options: PixelGenerationImageOptions | None = None,
    ) -> PixelGenerationRequestFuncInput:
        return PixelGenerationRequestFuncInput(
            model="black-forest-labs/FLUX.2-dev",
            session_id=None,
            prompt="A beautiful sunset",
            input_image_paths=None,
            api_url="http://localhost:8000/v1/images/generations",
            image_options=image_options,
        )

    def test_sglang_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=1024,
            height=768,
            steps=28,
            guidance_scale=3.5,
            seed=42,
            negative_prompt="blurry",
        )
        payload = _build_sglang_pixel_generation_payload(self._make_input(opts))
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["size"] == "1024x768"
        assert payload["num_inference_steps"] == 28
        assert payload["guidance_scale"] == 3.5
        assert payload["seed"] == 42
        assert payload["n"] == 1
        assert payload["response_format"] == "b64_json"
        # negative_prompt is intentionally not supported
        assert "negative_prompt" not in payload

    def test_sglang_payload_no_options(self) -> None:
        payload = _build_sglang_pixel_generation_payload(self._make_input(None))
        assert payload["prompt"] == "A beautiful sunset"
        assert "size" not in payload
        assert "num_inference_steps" not in payload

    def test_sglang_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(steps=20)
        payload = _build_sglang_pixel_generation_payload(self._make_input(opts))
        assert payload["num_inference_steps"] == 20
        # size requires both width and height
        assert "size" not in payload
        assert "guidance_scale" not in payload

    def test_vllm_omni_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=1024,
            height=768,
            steps=28,
            guidance_scale=3.5,
            seed=42,
            negative_prompt="blurry",
        )
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(opts)
        )
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["messages"] == [
            {"role": "user", "content": "A beautiful sunset"}
        ]
        extra = payload["extra_body"]
        assert extra["height"] == 768
        assert extra["width"] == 1024
        assert extra["num_inference_steps"] == 28
        assert extra["guidance_scale"] == 3.5
        assert extra["seed"] == 42
        # negative_prompt is intentionally not supported
        assert "negative_prompt" not in extra

    def test_vllm_omni_payload_no_options(self) -> None:
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(None)
        )
        assert "extra_body" not in payload
        assert payload["messages"][0]["content"] == "A beautiful sunset"

    def test_vllm_omni_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(height=512, width=512)
        payload = _build_vllm_omni_pixel_generation_payload(
            self._make_input(opts)
        )
        extra = payload["extra_body"]
        assert extra["height"] == 512
        assert extra["width"] == 512
        assert "num_inference_steps" not in extra

    def test_vllm_omni_video_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=832,
            height=480,
            steps=50,
            guidance_scale=4.0,
            seed=42,
            negative_prompt="blurry",
            num_frames=81,
        )
        payload = _build_vllm_omni_video_payload(self._make_input(opts))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["width"] == "832"
        assert payload["height"] == "480"
        assert payload["num_inference_steps"] == "50"
        assert payload["guidance_scale"] == "4.0"
        assert payload["seed"] == "42"
        assert payload["negative_prompt"] == "blurry"
        assert payload["num_frames"] == "81"

    def test_vllm_omni_video_payload_no_options(self) -> None:
        payload = _build_vllm_omni_video_payload(self._make_input(None))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert len(payload) == 2

    def test_vllm_omni_video_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(width=480, height=480, num_frames=33)
        payload = _build_vllm_omni_video_payload(self._make_input(opts))
        assert payload["width"] == "480"
        assert payload["height"] == "480"
        assert payload["num_frames"] == "33"
        assert "num_inference_steps" not in payload
        assert "guidance_scale" not in payload

    def test_sglang_video_payload_all_options(self) -> None:
        opts = PixelGenerationImageOptions(
            width=832,
            height=480,
            steps=50,
            guidance_scale=4.0,
            seed=42,
            negative_prompt="blurry",
            num_frames=81,
        )
        payload = _build_sglang_video_payload(self._make_input(opts))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert payload["width"] == 832
        assert payload["height"] == 480
        assert payload["num_inference_steps"] == 50
        assert payload["guidance_scale"] == 4.0
        assert payload["seed"] == 42
        assert payload["negative_prompt"] == "blurry"
        assert payload["num_frames"] == 81
        assert "n" not in payload
        assert "response_format" not in payload

    def test_sglang_video_payload_no_options(self) -> None:
        payload = _build_sglang_video_payload(self._make_input(None))
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["model"] == "black-forest-labs/FLUX.2-dev"
        assert len(payload) == 2

    def test_sglang_video_payload_partial_options(self) -> None:
        opts = PixelGenerationImageOptions(width=832, height=480, num_frames=81)
        payload = _build_sglang_video_payload(self._make_input(opts))
        assert payload["width"] == 832
        assert payload["height"] == 480
        assert payload["num_frames"] == 81
        assert "num_inference_steps" not in payload
        assert "guidance_scale" not in payload


class TestCountGeneratedMedia:
    """Tests for _count_generated_media (OpenResponses response parser)."""

    @staticmethod
    def _wrap(content: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap content items in the OpenResponses message envelope."""
        return {
            "output": [
                {
                    "id": "msg_test_0",
                    "role": "assistant",
                    "content": content,
                    "status": "completed",
                }
            ]
        }

    def test_counts_output_image(self) -> None:
        body = self._wrap(
            [{"type": "output_image", "image_url": "http://x/img.png"}]
        )
        assert _count_generated_media(body) == 1

    def test_counts_output_video(self) -> None:
        # Regression: video responses were previously miscounted as 0, causing
        # every text-to-video request to fail with "No output_image content
        # found in OpenResponses response body." (PERF-2518).
        body = self._wrap(
            [
                {
                    "type": "output_video",
                    "video_url": "http://x/vid.mp4",
                    "format": "mp4",
                    "frames_per_second": 16,
                    "num_frames": 81,
                }
            ]
        )
        assert _count_generated_media(body) == 1

    def test_counts_mixed_image_and_video(self) -> None:
        body = self._wrap(
            [
                {"type": "output_image", "image_url": "http://x/a.png"},
                {"type": "output_video", "video_url": "http://x/b.mp4"},
                {"type": "output_image", "image_url": "http://x/c.png"},
            ]
        )
        assert _count_generated_media(body) == 3

    def test_ignores_non_media_content_types(self) -> None:
        body = self._wrap(
            [
                {"type": "output_text", "text": "hello"},
                {"type": "refusal", "refusal": "no"},
                {"type": "reasoning_summary", "summary": "thinking"},
            ]
        )
        assert _count_generated_media(body) == 0

    def test_empty_output(self) -> None:
        assert _count_generated_media({"output": []}) == 0

    def test_missing_output_key(self) -> None:
        assert _count_generated_media({}) == 0

    def test_malformed_output_not_a_list(self) -> None:
        assert _count_generated_media({"output": "not a list"}) == 0


class TestValidateTaskAndEndpoint:
    """Tests for validate_task_and_endpoint."""

    def test_text_gen_modular_chat_ok(self) -> None:
        validate_task_and_endpoint("text-generation", "/v1/chat/completions")

    def test_text_gen_modular_completions_ok(self) -> None:
        validate_task_and_endpoint("text-generation", "/v1/completions")

    def test_text_gen_responses_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            validate_task_and_endpoint("text-generation", "/v1/responses")

    def test_text_gen_images_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            validate_task_and_endpoint(
                "text-generation", "/v1/images/generations"
            )

    def test_pixel_gen_responses_ok(self) -> None:
        validate_task_and_endpoint("text-to-image", "/v1/responses")

    def test_pixel_gen_images_ok(self) -> None:
        validate_task_and_endpoint("text-to-image", "/v1/images/generations")

    def test_pixel_gen_chat_ok(self) -> None:
        validate_task_and_endpoint("text-to-image", "/v1/chat/completions")

    def test_pixel_gen_completions_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires --endpoint"):
            validate_task_and_endpoint("text-to-image", "/v1/completions")

    def test_pixel_gen_videos_sync_ok(self) -> None:
        validate_task_and_endpoint("text-to-video", "/v1/videos/sync")

    def test_image_to_video_videos_sync_ok(self) -> None:
        # i2v is a video task and may use the video endpoints.
        validate_task_and_endpoint("image-to-video", "/v1/videos/sync")

    def test_text_gen_videos_sync_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            validate_task_and_endpoint("text-generation", "/v1/videos/sync")

    def test_text_to_image_videos_sync_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for"):
            validate_task_and_endpoint("text-to-image", "/v1/videos/sync")

    def test_image_to_image_videos_sync_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for"):
            validate_task_and_endpoint("image-to-image", "/v1/videos/sync")

    def test_pixel_gen_videos_ok(self) -> None:
        validate_task_and_endpoint("text-to-video", "/v1/videos")

    def test_image_to_video_videos_ok(self) -> None:
        validate_task_and_endpoint("image-to-video", "/v1/videos")

    def test_text_gen_videos_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            validate_task_and_endpoint("text-generation", "/v1/videos")

    def test_text_to_image_videos_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for"):
            validate_task_and_endpoint("text-to-image", "/v1/videos")

    def test_image_to_image_videos_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for"):
            validate_task_and_endpoint("image-to-image", "/v1/videos")

    def test_image_to_image_responses_ok(self) -> None:
        validate_task_and_endpoint("image-to-image", "/v1/responses")


class TestMarkCancelledIfPastDeadline:
    """Tests for ``mark_cancelled_if_past_deadline``.

    A request cut off by benchmark end (its non-success result surfaces after
    the deadline) should be reclassified as cancelled rather than failed.
    """

    def test_failed_past_deadline_becomes_cancelled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        past = time.perf_counter_ns() - int(1e9)
        out = RequestFuncOutput(success=False, error="swallowed timeout")
        with caplog.at_level(
            logging.INFO, logger="max.benchmark.benchmark_shared.request"
        ):
            result = mark_cancelled_if_past_deadline(out, past)
        assert result is out
        assert out.cancelled is True
        # The reclassification is logged so the user is aware the request was
        # cut off by benchmark end rather than silently dropped.
        assert any(
            "cut off by benchmark end" in rec.message for rec in caplog.records
        )

    def test_failed_before_deadline_stays_failed(self) -> None:
        future = time.perf_counter_ns() + int(60 * 1e9)
        out = RequestFuncOutput(success=False, error="real failure")
        mark_cancelled_if_past_deadline(out, future)
        assert out.cancelled is False

    def test_failed_unbounded_deadline_stays_failed(self) -> None:
        out = RequestFuncOutput(success=False, error="real failure")
        mark_cancelled_if_past_deadline(out, None)
        assert out.cancelled is False

    def test_success_past_deadline_untouched(self) -> None:
        past = time.perf_counter_ns() - int(1e9)
        out = RequestFuncOutput(success=True)
        mark_cancelled_if_past_deadline(out, past)
        assert out.cancelled is False
