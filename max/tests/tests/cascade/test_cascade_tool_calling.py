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
"""End-to-end reasoning + tool-calling tests over the cascade chat route.

The text-echo pipeline replays a scripted model response verbatim (skipping the
tokenizer), so the whole serve path -- runtime, worker-to-worker streaming,
reasoning/tool parsing, and OpenAI framing -- is exercised without a GPU model.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.pipelines.chat_parser import ChatParserConfig
from max.experimental.cascade.pipelines.text_echo import TextEchoPipeline
from max.experimental.cascade.serve.chat_completions import build_router

# A scripted model response: reasoning, then content, then two tool calls with
# nested JSON arguments.
SCRIPTED = (
    "<think>The user wants weather and time.</think>"
    "Looking those up."
    '<tool_call>get_weather\n{"city": "SF", "opts": {"units": "c"}}</tool_call>'
    '<tool_call>get_time\n{"tz": "PST"}</tool_call>'
)

TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}},
    },
    {
        "type": "function",
        "function": {"name": "get_time", "parameters": {}},
    },
]


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


async def _client(
    runtime: LocalRuntime,
    config: ChatParserConfig,
    emit_reasoning_content: bool = False,
) -> AsyncClient:
    pipeline = TextEchoPipeline(config)
    await pipeline.deploy(runtime)
    app = FastAPI()
    app.include_router(
        await build_router(pipeline, runtime, emit_reasoning_content)
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _config() -> ChatParserConfig:
    return ChatParserConfig(
        reasoning_start="<think>",
        reasoning_end="</think>",
        tool_parser="echo",
    )


@pytest.mark.asyncio
async def test_non_streaming_reasoning_and_tool_calls(
    runtime: LocalRuntime,
) -> None:
    async with await _client(runtime, _config()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [{"role": "user", "content": SCRIPTED}],
                "tools": TOOLS,
                "max_tokens": len(SCRIPTED),
            },
        )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"

    message = choice["message"]
    assert message["reasoning"] == "The user wants weather and time."
    assert message["content"] == "Looking those up."

    tool_calls = message["tool_calls"]
    assert [tc["function"]["name"] for tc in tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "city": "SF",
        "opts": {"units": "c"},
    }
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"tz": "PST"}
    for tc in tool_calls:
        assert tc["type"] == "function"
        assert tc["id"].startswith("call_")


@pytest.mark.asyncio
async def test_streaming_reasoning_and_tool_calls(
    runtime: LocalRuntime,
) -> None:
    async with await _client(runtime, _config()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [{"role": "user", "content": SCRIPTED}],
                "tools": TOOLS,
                "max_tokens": len(SCRIPTED),
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = [
        line[len("data: ") :]
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]

    reasoning = ""
    content = ""
    tool_calls: dict[int, dict[str, str]] = {}
    finish_reason = None
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk"
        choice = chunk["choices"][0]
        delta = choice["delta"]
        if delta.get("reasoning"):
            reasoning += delta["reasoning"]
        if delta.get("content"):
            content += delta["content"]
        for tc in delta.get("tool_calls") or []:
            call = tool_calls.setdefault(
                tc["index"], {"name": "", "arguments": ""}
            )
            fn = tc.get("function") or {}
            if fn.get("name"):
                call["name"] = fn["name"]
            if fn.get("arguments"):
                call["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    assert reasoning == "The user wants weather and time."
    assert content == "Looking those up."
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[1]["name"] == "get_time"
    assert json.loads(tool_calls[0]["arguments"]) == {
        "city": "SF",
        "opts": {"units": "c"},
    }
    assert json.loads(tool_calls[1]["arguments"]) == {"tz": "PST"}

    # A reasoning delta never carries content in the same frame.
    for chunk in chunks:
        delta = chunk["choices"][0]["delta"]
        if delta.get("reasoning"):
            assert not delta.get("content")
            assert not delta.get("tool_calls")


@pytest.mark.asyncio
async def test_reasoning_content_alias(runtime: LocalRuntime) -> None:
    config = ChatParserConfig(
        reasoning_start="<think>",
        reasoning_end="</think>",
    )
    async with await _client(
        runtime, config, emit_reasoning_content=True
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [
                    {"role": "user", "content": "<think>hmm</think>hi"}
                ],
                "max_tokens": 32,
            },
        )
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "hmm"
    assert message["reasoning"] is None
    assert message["content"] == "hi"


@pytest.mark.asyncio
async def test_tools_absent_leaves_markers_as_content(
    runtime: LocalRuntime,
) -> None:
    # Without ``tools`` in the request, tool parsing is disabled even though the
    # pipeline declares a parser -- the markers pass through as plain content.
    async with await _client(runtime, _config()) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [
                    {
                        "role": "user",
                        "content": '<tool_call>a\n{"x": 1}</tool_call>',
                    }
                ],
                "max_tokens": 64,
            },
        )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"] is None
    assert choice["message"]["content"] == '<tool_call>a\n{"x": 1}</tool_call>'
