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
"""Expose a generative pipeline over the OpenAI chat-completions schema.

Owns the translation in both directions: an OpenAI request becomes a
:class:`GenAIRequest`, and the wrapped pipeline's chunks become OpenAI response
bytes. Nothing here knows how any particular model formats reasoning or tool
calls -- the wrapped pipeline has already turned that into typed chunks -- and
nothing above here knows about OpenAI, so the HTTP route is left with only the
FastAPI boilerplate. Everything here is pydantic-to-pydantic: no HTTP types
cross into this module.

The chunk stream is forwarded to an :class:`OpenAIChatFormatter` worker rather
than serialized inline, so the per-chunk JSON encoding runs on the CPU worker
pool instead of the API event loop, and never round-trips the orchestrator.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import Any

from max.experimental.cascade.core import pipeline_method
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAIInterface,
    GenAIRequest,
    GenAITextChunk,
    GenAITool,
    TextGenOptions,
)
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.serve.openai_chat_formatter import (
    OpenAIChatFormatter,
)
from max.pipelines.lib.tool_parsing import maybe_name_from_tool
from max.serve.schemas.openai import CreateChatCompletionRequest

# TODO(GEX-2412): mint a unique id per request.
REQUEST_ID = "chatcmpl-cascade"


def _convert_stop(stop: str | Sequence[str] | None) -> list[str] | None:
    """Normalize the OpenAI ``stop`` field into a list of strings."""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _message_text(content: str | Sequence[Mapping[str, Any]]) -> str:
    """Flatten an OpenAI message content payload into plain text.

    Raises:
        ValueError: If the message carries a non-text content part. The route
            reports it as a ``400``.
    """
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []
    for part in content:
        if part.get("type") == "text" and part.get("text") is not None:
            text_parts.append(part["text"])
        else:
            unsupported_types.append(part.get("type", "unknown"))

    if unsupported_types:
        raise ValueError(
            "Unsupported chat message content part types: "
            + ", ".join(sorted(set(unsupported_types)))
        )

    return "".join(text_parts)


def _convert_tools(request: CreateChatCompletionRequest) -> list[GenAITool]:
    """Convert the request's tool definitions into their generic form.

    Tools without a resolvable name are dropped, matching the MAX Serve route.
    """
    tools: list[GenAITool] = []
    for tool in request.tools or []:
        name = maybe_name_from_tool(tool)
        if not name:
            continue
        function = tool.get("function")
        parameters = None
        description = None
        if isinstance(function, dict):
            if isinstance(function.get("parameters"), dict):
                parameters = function["parameters"]
            if isinstance(function.get("description"), str):
                description = function["description"]
        tools.append(
            GenAITool(name=name, description=description, parameters=parameters)
        )
    return tools


def _text_gen_options(request: CreateChatCompletionRequest) -> TextGenOptions:
    """Map the request's sampling fields onto :class:`TextGenOptions`.

    Passthrough fields share ``TextGenOptions``'s ``None`` defaults, so
    forwarding them verbatim leaves unset ones on the model default. Fields
    whose default differs from "unset" are only overridden when the client
    supplied a value.
    """
    options = TextGenOptions(
        ignore_eos=request.ignore_eos,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        thinking_temperature=request.thinking_temperature,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        repetition_penalty=request.repetition_penalty,
        stop=_convert_stop(request.stop),
        stop_token_ids=request.stop_token_ids,
    )
    # ``max_completion_tokens`` supersedes the legacy ``max_tokens``.
    max_new_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_new_tokens is not None:
        options.num_tokens = max_new_tokens
    if request.min_tokens is not None:
        options.min_new_tokens = request.min_tokens
    if request.temperature is not None:
        options.temperature = request.temperature
    return options


def to_gen_ai_request(request: CreateChatCompletionRequest) -> GenAIRequest:
    """Convert an OpenAI chat-completion request into a cascade request.

    Raises:
        ValueError: If the request carries content this schema cannot express.
    """
    tool_choice = "auto"
    if request.tool_choice in ("none", "required"):
        tool_choice = request.tool_choice
    return GenAIRequest(
        messages=[
            ChatMessage(
                role=message.get("role", ""),
                content=[
                    GenAITextChunk(
                        text=_message_text(message.get("content") or "")
                    )
                ],
            )
            for message in request.messages
        ],
        tools=_convert_tools(request),
        tool_choice=tool_choice,  # type: ignore[arg-type]
        text=_text_gen_options(request),
    )


class OpenAIChatCompletionPipeline(CascadePipeline):
    """Serve a generative pipeline over the OpenAI chat-completions schema."""

    def __init__(
        self, gen_ai: GenAIInterface, emit_reasoning_content: bool = False
    ) -> None:
        """Wrap *gen_ai*, adding a formatter worker for OpenAI chat responses.

        Args:
            gen_ai: An already-deployed pipeline to expose over OpenAI chat.
                :meth:`deploy` only brings up this wrapper's own formatter
                worker; ``gen_ai`` is deployed by whoever owns it.
            emit_reasoning_content: Server-wide reasoning field choice, fixed
                onto the formatter worker here.
        """
        self.gen_ai = gen_ai
        self.formatter = OpenAIChatFormatter(emit_reasoning_content)

    async def chat_completion_sse(
        self, request: CreateChatCompletionRequest
    ) -> AsyncIterator[bytes]:
        """Translate *request*, then open the stream of OpenAI SSE frames.

        Awaiting this does the translation and hands back a stream that has not
        started yet, so a request this schema cannot express still surfaces as a
        rejection rather than a truncated response body: by the time a caller
        starts forwarding frames, the response status is already committed.

        Splitting the translation out here is also what leaves room to move it
        onto a worker, once it grows work worth taking off the event loop (an
        ``image_url`` part that has to be fetched, say).

        Raises:
            ValueError: If the request carries content this schema cannot
                express.
        """
        req = to_gen_ai_request(request)
        return self._sse_frames(req, request.model)

    @pipeline_method
    async def _sse_frames(
        self, req: GenAIRequest, model: str
    ) -> AsyncIterator[bytes]:
        """Stream OpenAI chat SSE frames, serializing on the worker pool.

        The wrapped pipeline's chunk handle flows straight into the formatter
        worker, so both run worker-to-worker inside this one pipeline scope and
        the orchestrator only forwards the finished byte frames.
        """
        chunks = await self.gen_ai._generate_iterator(req)
        async for frame in await self.formatter.format_stream(
            chunks, model, REQUEST_ID, int(time.time())
        ):
            yield frame

    @pipeline_method
    async def chat_completion(
        self, request: CreateChatCompletionRequest
    ) -> Awaitable[bytes]:
        """Translate *request* and serialize a non-streaming OpenAI response.

        Mirrors :meth:`chat_completion_sse` for the non-streaming case: the
        formatter collects the chunk stream and serializes the finished JSON
        body, so the orchestrator only forwards the finished bytes. No split is
        needed here -- the whole body is produced before the caller sees
        anything, so a translation failure is still reportable.

        Raises:
            ValueError: If the request carries content this schema cannot
                express.
        """
        req = to_gen_ai_request(request)
        chunks = await self.gen_ai._generate_iterator(req)
        return await self.formatter.format_complete(
            chunks, request.model, REQUEST_ID, int(time.time())
        )
