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
"""OpenResponses route adapter for cascade image generation.

We should use https://github.com/mozilla-ai/openresponses-python
once we upgrade to python >= 3.11.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAIChunk,
    GenAIImageChunk,
    GenAIInterface,
    GenAIRequest,
    ImageGenOptions,
)
from max.pipelines.request.open_responses import (
    InputTextContent,
    Message,
    MessageRole,
    MessageStatus,
    OpenResponsesRequestBody,
    OutputImageContent,
    ResponseResource,
)
from max.pipelines.request.provider_options.modality.image import (
    ImageProviderOptions,
)
from pydantic import Field, model_validator
from sse_starlette.sse import EventSourceResponse


class _CascadeRequestBody(OpenResponsesRequestBody):
    """OpenResponses request body that permits streaming.

    The base ``OpenResponsesRequestBody`` rejects ``stream=True`` via a
    validator.  Cascade pipelines support streaming, so this subclass
    overrides that restriction.

    This is a temporary workaround until we use open source schema defs.
    And we should never leak our supported functionality into the types
    themselves going forward.
    """

    stream: bool | None = Field(
        default=None,
        description="If true, stream back partial progress as server-sent events.",
    )

    @model_validator(mode="after")
    def validate_streaming_not_supported(self) -> _CascadeRequestBody:
        """Override: cascade supports streaming, so this is a no-op."""
        return self


def _extract_prompt(body: _CascadeRequestBody) -> str:
    """Extract a text prompt from the request input."""
    if isinstance(body.input, str):
        return body.input

    message = body.input[0]
    if isinstance(message.content, str):
        return message.content

    prompt_parts = [
        content.text
        for content in message.content
        if isinstance(content, InputTextContent)
    ]
    if prompt_parts:
        return "\n".join(prompt_parts)
    raise ValueError("OpenResponses image generation requires text input.")


def _to_gen_ai_request(body: _CascadeRequestBody) -> GenAIRequest:
    """Build a :class:`GenAIRequest` from the OpenResponses request body."""
    image_options = body.provider_options.image or ImageProviderOptions()
    defaults = ImageGenOptions()
    return GenAIRequest(
        messages=[ChatMessage.text("user", _extract_prompt(body))],
        image=ImageGenOptions(
            height=image_options.height or defaults.height,
            width=image_options.width or defaults.width,
            num_steps=(
                image_options.steps
                if image_options.steps is not None
                else defaults.num_steps
            ),
            guidance_scale=(
                image_options.guidance_scale
                if image_options.guidance_scale is not None
                else defaults.guidance_scale
            ),
            seed=body.seed if body.seed is not None else defaults.seed,
            output_format=image_options.output_format.upper(),
        ),
    )


async def _image_bytes(
    chunk_iter: AsyncIterator[GenAIChunk],
) -> AsyncIterator[bytes]:
    """Keep only the generated images from a chunk stream.

    A pipeline may interleave commentary with the images it generates; this
    route's schema carries images alone, so anything else is dropped. A
    URL-valued image chunk is passed over too -- this route inlines image data
    and has nothing to fetch with.
    """
    async for chunk in chunk_iter:
        if isinstance(chunk, GenAIImageChunk) and chunk.data is not None:
            yield chunk.data


def _build_response(
    image_bytes: bytes,
    *,
    model: str,
    output_format: str,
    completed: bool = True,
) -> ResponseResource:
    """Build a ``ResponseResource`` from raw generated image bytes.

    When ``completed`` is False, both the response and its message are marked
    as ``in_progress``, suitable for emitting as an intermediate streaming
    frame. ``response_id`` and ``message_id`` may be supplied to keep ids
    stable across streaming frames; otherwise fresh uuids are minted.
    """
    image_content = OutputImageContent(
        image_data=base64.b64encode(image_bytes).decode("ascii"),
        format=output_format,
    )
    message_status = (
        MessageStatus.completed if completed else MessageStatus.in_progress
    )
    message = Message(
        id=f"msg_{uuid.uuid4().hex}",
        role=MessageRole.assistant,
        content=[image_content],
        status=message_status,
    )
    return ResponseResource(
        id=f"resp_{uuid.uuid4().hex}",
        object="response",
        created_at=int(time.time()),
        status="completed" if completed else "in_progress",
        model=model,
        output=[message],
    )


async def _stream_response(
    image_iter: AsyncIterator[bytes],
    *,
    model: str,
    output_format: str,
) -> AsyncIterator[str]:
    """Forward a stream of image frames as typed SSE events.

    Each upstream frame becomes a :py:class:`ResponseResource` event. All
    frames except the final one are marked ``in_progress``; the final frame
    is marked ``completed`` and is followed by the ``[DONE]`` sentinel. The
    response and message ids are minted once and reused across every frame
    so clients can correlate the in-progress chunks with the final result.
    """
    pending: bytes | None = None
    async for image_bytes in image_iter:
        if pending is not None:
            yield _build_response(
                image_bytes,
                model=model,
                output_format=output_format,
                completed=False,
            ).model_dump_json(exclude_none=True)
        pending = image_bytes
    if pending is not None:
        yield _build_response(
            pending,
            model=model,
            output_format=output_format,
            completed=True,
        ).model_dump_json(exclude_none=True)
    yield "[DONE]"


def build_router(pipeline: GenAIInterface) -> APIRouter:
    """Build OpenResponses routes for a cascade-backed image pipeline."""
    router = APIRouter(prefix="/v1")

    @router.post("/responses", response_model=None)
    async def create_response(
        body: _CascadeRequestBody,
    ) -> ResponseResource | EventSourceResponse:
        image_options = body.provider_options.image or ImageProviderOptions()
        images = _image_bytes(pipeline.generate(_to_gen_ai_request(body)))

        if body.stream:
            return EventSourceResponse(
                _stream_response(
                    images,
                    model=body.model,
                    output_format=image_options.output_format,
                )
            )

        # The final frame is the finished image; earlier ones are intermediate
        # denoising steps a non-streaming client never sees.
        image_bytes = b""
        async for frame in images:
            image_bytes = frame
        return _build_response(
            image_bytes,
            model=body.model,
            output_format=image_options.output_format,
        )

    return router
