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
"""Tests for the chat-completion route adapter."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from max.experimental.cascade import (
    GenAIChunk,
    GenAIInterface,
    GenAIRequest,
    GenAITextChunk,
    LocalRuntime,
    Modality,
    TextGenOptions,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.experimental.cascade.serve.chat_completions import build_router


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


@pytest.fixture()
async def client(runtime: LocalRuntime) -> AsyncIterator[AsyncClient]:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    app = FastAPI()
    app.include_router(await build_router(pipeline, runtime))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_non_streaming_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "dummy"
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    # The dummy pipeline always emits "A" tokens.
    assert choice["message"]["content"] == "AAA"


@pytest.mark.asyncio
async def test_streaming_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events from the response body.
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: ") :])

    # Last event should be the [DONE] sentinel.
    assert events[-1] == "[DONE]"

    import json

    chunks = [json.loads(e) for e in events[:-1]]
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "dummy"

    # Three content frames, each carrying one "A".
    content_chunks = [
        c for c in chunks if c["choices"][0]["delta"].get("content")
    ]
    assert len(content_chunks) == 3
    for chunk in content_chunks:
        assert chunk["choices"][0]["delta"]["content"] == "A"

    # A terminal frame carries the finish reason.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_multipart_content(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ],
            "max_tokens": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "AA"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_unsupported_content_type(
    client: AsyncClient, stream: bool
) -> None:
    # Streaming must reject just as loudly as non-streaming: the pipeline
    # translates before handing back a stream, so the rejection still lands on
    # the status line rather than truncating a 200 response body.
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x"}}
                    ],
                }
            ],
            "max_tokens": 1,
            "stream": stream,
        },
    )
    assert resp.status_code == 400
    assert "image_url" in resp.json()["detail"]


class _SpyPipeline(GenAIInterface):
    """Records the :class:`GenAIRequest` forwarded by the route."""

    def __init__(self) -> None:
        self.last_request: GenAIRequest | None = None

    def supported_input_modalities(self) -> set[Modality]:
        """Accept text prompts."""
        return {Modality.TEXT}

    def supported_output_modalities(self) -> set[Modality]:
        """Emit assistant text, reasoning, and tool calls."""
        return {Modality.TEXT}

    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        self.last_request = req

        async def _stream() -> AsyncIterator[GenAIChunk]:
            for _ in range(req.text.num_tokens):
                yield GenAITextChunk(text="A")

        return _stream()


@pytest.fixture()
async def spy() -> _SpyPipeline:
    return _SpyPipeline()


@pytest.fixture()
async def spy_client(
    runtime: LocalRuntime, spy: _SpyPipeline
) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    app.include_router(await build_router(spy, runtime))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_forwards_all_sampling_fields(
    spy_client: AsyncClient, spy: _SpyPipeline
) -> None:
    resp = await spy_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "min_tokens": 2,
            "ignore_eos": True,
            "temperature": 0.5,
            "top_k": 7,
            "top_p": 0.9,
            "min_p": 0.1,
            "thinking_temperature": 0.3,
            "seed": 1234,
            "frequency_penalty": 0.25,
            "presence_penalty": -0.5,
            "repetition_penalty": 1.1,
            "stop": ["END", "STOP"],
            "stop_token_ids": [1, 2, 3],
        },
    )
    assert resp.status_code == 200
    assert spy.last_request is not None
    req = spy.last_request.text
    assert req.num_tokens == 4
    assert req.min_new_tokens == 2
    assert req.ignore_eos is True
    assert req.temperature == 0.5
    assert req.top_k == 7
    assert req.top_p == 0.9
    assert req.min_p == 0.1
    assert req.thinking_temperature == 0.3
    assert req.seed == 1234
    assert req.frequency_penalty == 0.25
    assert req.presence_penalty == -0.5
    assert req.repetition_penalty == 1.1
    assert req.stop == ["END", "STOP"]
    assert req.stop_token_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_max_completion_tokens_supersedes_max_tokens(
    spy_client: AsyncClient, spy: _SpyPipeline
) -> None:
    resp = await spy_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "max_completion_tokens": 6,
        },
    )
    assert resp.status_code == 200
    assert spy.last_request is not None
    assert spy.last_request.text.num_tokens == 6


@pytest.mark.asyncio
async def test_stop_string_is_normalized_to_list(
    spy_client: AsyncClient, spy: _SpyPipeline
) -> None:
    resp = await spy_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stop": "END",
        },
    )
    assert resp.status_code == 200
    assert spy.last_request is not None
    assert spy.last_request.text.stop == ["END"]


@pytest.mark.asyncio
async def test_defaults_when_fields_absent(
    spy_client: AsyncClient, spy: _SpyPipeline
) -> None:
    resp = await spy_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    assert spy.last_request is not None
    req = spy.last_request.text
    # Unset sampling fields fall back to "use the model / server default".
    assert req.num_tokens == TextGenOptions.model_fields["num_tokens"].default
    assert req.temperature == 1.0
    assert req.top_k is None
    assert req.top_p is None
    assert req.min_p is None
    assert req.seed is None
    assert req.frequency_penalty is None
    assert req.presence_penalty is None
    assert req.repetition_penalty is None
    assert req.stop is None
    assert req.stop_token_ids is None
