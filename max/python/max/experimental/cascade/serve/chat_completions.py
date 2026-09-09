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
"""Chat-completion HTTP route for cascade generative-AI pipelines.

Binds :class:`OpenAIChatCompletionPipeline` to a FastAPI path. The pipeline owns
the request and response translation, so this is only the HTTP surface.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from max.experimental.cascade.core import Runtime
from max.experimental.cascade.interfaces.gen_ai import GenAIInterface
from max.experimental.cascade.serve.openai_chat_pipeline import (
    OpenAIChatCompletionPipeline,
)
from max.serve.schemas.openai import CreateChatCompletionRequest


async def build_router(
    pipeline: GenAIInterface,
    runtime: Runtime,
    emit_reasoning_content: bool = False,
) -> APIRouter:
    """Build the OpenAI-style chat-completion route for a generative pipeline.

    The route pairs ``pipeline`` with an :class:`OpenAIChatCompletionPipeline`,
    which is an implementation detail of this adapter, so callers hand over a
    plain :class:`GenAIInterface`.

    Args:
        pipeline: The deployed pipeline to serve.
        runtime: The runtime ``pipeline`` is already deployed on; only the
            wrapper's own formatter worker is deployed here.
        emit_reasoning_content: Emit reasoning under ``reasoning_content``
            instead of ``reasoning``. A server-wide choice, so it is fixed onto
            the formatter worker at deploy time.
    """
    chat = OpenAIChatCompletionPipeline(pipeline, emit_reasoning_content)
    await chat.deploy(runtime)

    router = APIRouter()

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: CreateChatCompletionRequest,
    ) -> Response | StreamingResponse:
        # Awaiting before building the response is what keeps a rejected
        # request a 400: a StreamingResponse commits the status line as soon as
        # it starts.
        try:
            if request.stream:
                return StreamingResponse(
                    await chat.chat_completion_sse(request),
                    media_type="text/event-stream",
                )
            return Response(
                content=await chat.chat_completion(request),
                media_type="application/json",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
