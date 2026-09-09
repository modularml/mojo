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
"""Serialize cascade generative-AI chunks as OpenAI chat responses.

Handles both the streaming (SSE frames) and non-streaming (single JSON body)
shapes. The pipeline has already split the model's output into typed chunks, so
all that happens here is a mapping onto OpenAI's wire schema -- plus the
``model_dump_json`` that mapping implies, which is why it runs on a worker.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal

from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.gen_ai import (
    GenAIChunk,
    GenAIReasoningChunk,
    GenAITextChunk,
    GenAIToolCall,
)
from max.serve.schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)

# OpenAI's chat-completion ``finish_reason`` value set.
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

# Terminal frame of an OpenAI streaming response. Pre-encoded because it never
# varies per request.
DONE_SSE: bytes = b"data: [DONE]\n\n"


def _frame(
    delta: ChatCompletionStreamResponseDelta,
    model: str,
    request_id: str,
    created: int,
    finish_reason: FinishReason | None,
) -> bytes:
    r"""Wrap one delta in a ``data: {json}\n\n`` chat-completion SSE frame."""
    chunk = CreateChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model,
        object="chat.completion.chunk",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0, delta=delta, finish_reason=finish_reason
            )
        ],
    )
    return b"data: " + chunk.model_dump_json().encode("utf-8") + b"\n\n"


def _reasoning_delta(
    text: str, emit_reasoning_content: bool
) -> ChatCompletionStreamResponseDelta:
    """Put reasoning under the field this server is configured to emit.

    ``reasoning_content`` is the vLLM/SGLang/DeepSeek alias; ``reasoning`` is
    the OpenAI Responses naming and the default. Split into two explicit
    constructions (rather than ``**{field: ...}``) to keep the delta's typed
    shape.
    """
    if emit_reasoning_content:
        return ChatCompletionStreamResponseDelta(reasoning_content=text)
    return ChatCompletionStreamResponseDelta(reasoning=text)


def _tool_call_delta(call: GenAIToolCall) -> ChoiceDeltaToolCall:
    """Map a tool-call chunk onto the OpenAI streaming shape.

    The opener carries ``id`` + ``type`` + ``function.name``; later fragments
    carry only the ``function.arguments`` diff at the same index.
    """
    function = None
    if call.name is not None or call.arguments is not None:
        function = ChoiceDeltaToolCallFunction(
            name=call.name, arguments=call.arguments
        )
    return ChoiceDeltaToolCall(
        index=call.index,
        id=call.id,
        type="function" if call.id is not None else None,
        function=function,
    )


def chunk_to_delta(
    chunk: GenAIChunk, emit_reasoning_content: bool
) -> ChatCompletionStreamResponseDelta:
    """Map one chunk onto an OpenAI streaming delta.

    Raises:
        ValueError: If the chunk is of a kind OpenAI chat cannot represent.
    """
    if isinstance(chunk, GenAITextChunk):
        return ChatCompletionStreamResponseDelta(content=chunk.text)
    if isinstance(chunk, GenAIReasoningChunk):
        return _reasoning_delta(chunk.text, emit_reasoning_content)
    if isinstance(chunk, GenAIToolCall):
        return ChatCompletionStreamResponseDelta(
            tool_calls=[_tool_call_delta(chunk)]
        )
    raise ValueError(
        f"OpenAI chat completions cannot represent a {chunk.type!r} chunk; "
        "use a route whose schema supports it."
    )


class _ChatAccumulator:
    """Collect a chunk stream into the parts of a non-streaming response."""

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        # Tool calls arrive as fragments keyed by index; arguments concatenate.
        self._calls: dict[int, ChatCompletionMessageToolCall] = {}

    def add(self, chunk: GenAIChunk) -> None:
        if isinstance(chunk, GenAITextChunk):
            self.text += chunk.text
        elif isinstance(chunk, GenAIReasoningChunk):
            self.reasoning += chunk.text
        elif isinstance(chunk, GenAIToolCall):
            self._add_tool_call(chunk)
        else:
            raise ValueError(
                f"OpenAI chat completions cannot represent a {chunk.type!r} "
                "chunk; use a route whose schema supports it."
            )

    def _add_tool_call(self, chunk: GenAIToolCall) -> None:
        call = self._calls.get(chunk.index)
        if call is None:
            call = ChatCompletionMessageToolCall(
                id=chunk.id or "",
                type="function",
                function=ChatCompletionMessageToolCallFunction(
                    name=chunk.name or "", arguments=""
                ),
            )
            self._calls[chunk.index] = call
        if chunk.id is not None:
            call.id = chunk.id
        if chunk.name is not None:
            call.function.name = chunk.name
        if chunk.arguments is not None:
            call.function.arguments += chunk.arguments

    @property
    def tool_calls(self) -> list[ChatCompletionMessageToolCallUnion]:
        return [self._calls[index] for index in sorted(self._calls)]

    def message(
        self, emit_reasoning_content: bool
    ) -> ChatCompletionResponseMessage:
        """Assemble the assistant message this stream added up to."""
        message = ChatCompletionResponseMessage(
            role="assistant", content=self.text or None
        )
        if self.reasoning:
            field = (
                "reasoning_content" if emit_reasoning_content else "reasoning"
            )
            setattr(message, field, self.reasoning)
        if self._calls:
            message.tool_calls = self.tool_calls
        return message


class OpenAIChatFormatter(Worker):
    """Serialize a chunk stream as OpenAI chat completions (stream + complete).

    OpenAI serialization runs a pydantic ``model_dump_json`` -- per chunk on the
    streaming hot path, or once for a non-streaming body -- which would
    otherwise execute on the single GIL-bound API event loop. Running it in a
    CPU worker moves that cost onto the round-robin worker pool, so it
    parallelizes across concurrent requests and stops pacing the decode stream.
    Both entry points chain after the pipeline's parser so chunks flow
    worker-to-worker; the API process only forwards finished bytes.
    """

    def __init__(self, emit_reasoning_content: bool = False) -> None:
        """Build the formatter.

        Args:
            emit_reasoning_content: Emit a thinking model's chain-of-thought
                under ``reasoning_content`` (the vLLM/SGLang/DeepSeek alias)
                rather than ``reasoning`` (the OpenAI Responses naming, and the
                default). A server-wide choice, so it is fixed here rather than
                travelling with each request.
        """
        super().__init__(deploy_hints=["cpu"])
        self.emit_reasoning_content = emit_reasoning_content

    @worker_method()
    async def format_stream(
        self,
        chunk_iter: AsyncIterable[GenAIChunk],
        model: str,
        request_id: str,
        created: int,
    ) -> AsyncIterator[bytes]:
        """Forward a chunk stream as OpenAI SSE frame bytes."""
        saw_tool_call = False
        async for chunk in chunk_iter:
            saw_tool_call |= isinstance(chunk, GenAIToolCall)
            yield _frame(
                chunk_to_delta(chunk, self.emit_reasoning_content),
                model,
                request_id,
                created,
                None,
            )
        yield _frame(
            ChatCompletionStreamResponseDelta(),
            model,
            request_id,
            created,
            "tool_calls" if saw_tool_call else "stop",
        )
        yield DONE_SSE

    @worker_method()
    async def format_complete(
        self,
        chunk_iter: AsyncIterable[GenAIChunk],
        model: str,
        request_id: str,
        created: int,
    ) -> bytes:
        """Collect a chunk stream and serialize the OpenAI chat JSON body."""
        accumulator = _ChatAccumulator()
        async for chunk in chunk_iter:
            accumulator.add(chunk)
        response = CreateChatCompletionResponse(
            id=request_id,
            created=created,
            model=model,
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=accumulator.message(self.emit_reasoning_content),
                    finish_reason=(
                        "tool_calls" if accumulator.tool_calls else "stop"
                    ),
                )
            ],
        )
        return response.model_dump_json().encode("utf-8")
