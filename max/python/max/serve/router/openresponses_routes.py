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

"""OpenResponses API route handlers.

This module provides a clean implementation of the OpenResponses API standard
without inheriting technical debt from other API endpoints.

Spec: https://www.openresponses.org/reference
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from max.pipelines.context.exceptions import InputError
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import OpenResponsesRequest
from max.pipelines.request.open_responses import (
    OutputAudioContent,
    OutputContent,
    OutputImageContent,
    OutputVideoContent,
    ResponseResource,
)
from max.pipelines.request.provider_options import (
    GeneratedMediaResponseFormat,
)
from max.serve.dependencies import create_request_parser
from max.serve.media import (
    GeneratedMediaStorageLimitExceeded,
    GeneratedMediaStore,
    StoredMediaAsset,
    encode_video_bytes_b64,
)
from starlette.datastructures import State

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")

# Create a reusable dependency for parsing OpenResponses requests
ParseOpenResponsesRequest = Depends(create_request_parser(OpenResponsesRequest))


@router.post("/responses")
async def create_response(
    request: Request,
    open_responses_request: OpenResponsesRequest = ParseOpenResponsesRequest,
) -> JSONResponse:
    """Create a response using the OpenResponses API schema.

    This endpoint provides a clean implementation of the OpenResponses
    standard for generating responses from AI models.

    Args:
        request: The incoming FastAPI request containing OpenResponses data.
        open_responses_request: Parsed and validated OpenResponses request
            (automatically injected via dependency injection). Validation
            includes checking that streaming is not requested.

    Returns:
        A JSONResponse with the generated response data.

    Raises:
        HTTPException: If request parsing or validation fails, including
            if streaming is requested (not currently supported).
    """

    # Request is already parsed and validated via dependency injection
    # (including validation that streaming is not requested)
    logger.debug(
        "OpenResponses request parsed successfully - "
        "request_id=%s, model=%s, stream=%s",
        open_responses_request.request_id.value,
        open_responses_request.body.model,
        open_responses_request.body.stream,
    )
    _validate_requested_model(
        request=request,
        requested_model=open_responses_request.body.model,
    )

    # Generate response using the GeneralPipelineHandler from app state
    logger.debug("Starting response generation")

    # Get the first chunk from the handler (raises StopAsyncIteration if empty)
    generator = request.app.state.handler.next(open_responses_request)
    try:
        final_output = await anext(generator)
        logger.debug(
            "Received chunk - is_done=%s, status=%s",
            final_output.is_done,
            final_output.final_status,
        )

        # Continue consuming chunks until we get is_done=True
        if not final_output.is_done:
            async for chunk in generator:
                logger.debug(
                    "Received chunk - is_done=%s, status=%s",
                    chunk.is_done,
                    chunk.final_status,
                )
                final_output = chunk
                if chunk.is_done:
                    break
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s",
            open_responses_request.request_id.value,
            str(e),
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(e)
        ) from e

    try:
        final_output = await _persist_generated_media(
            request=request,
            open_responses_request=open_responses_request,
            final_output=final_output,
        )
    except GeneratedMediaStorageLimitExceeded as exc:
        raise HTTPException(
            status_code=HTTPStatus.INSUFFICIENT_STORAGE,
            detail=str(exc),
        ) from exc

    # Convert GenerationOutput to ResponseResource format
    response = ResponseResource.from_generation_output(
        final_output, model=open_responses_request.body.model
    )

    logger.debug(
        "Returning response for request_id=%s",
        open_responses_request.request_id.value,
    )
    return JSONResponse(
        content=response.model_dump(mode="json", exclude_none=True),
        status_code=HTTPStatus.OK,
    )


@router.get("/images/{image_id}/content", name="get_generated_image_content")
async def get_generated_image_content(
    request: Request, image_id: str
) -> FileResponse:
    asset = _require_asset(
        _get_media_store(request).get_image(image_id),
        kind="image",
        asset_id=image_id,
    )
    return FileResponse(
        path=asset.path,
        media_type=asset.media_type,
        filename=asset.filename,
    )


@router.get("/videos/{video_id}/content", name="get_generated_video_content")
async def get_generated_video_content(
    request: Request, video_id: str
) -> FileResponse:
    asset = _require_asset(
        _get_media_store(request).get_video(video_id),
        kind="video",
        asset_id=video_id,
    )
    return FileResponse(
        path=asset.path,
        media_type=asset.media_type,
        filename=asset.filename,
    )


@router.get("/audio/{audio_id}/content", name="get_generated_audio_content")
async def get_generated_audio_content(
    request: Request, audio_id: str
) -> FileResponse:
    asset = _require_asset(
        _get_media_store(request).get_audio(audio_id),
        kind="audio",
        asset_id=audio_id,
    )
    return FileResponse(
        path=asset.path,
        media_type=asset.media_type,
        filename=asset.filename,
    )


async def _persist_generated_media(
    request: Request,
    open_responses_request: OpenResponsesRequest,
    final_output: GenerationOutput,
) -> GenerationOutput:
    media_store = getattr(request.app.state, "media_store", None)
    if not isinstance(media_store, GeneratedMediaStore):
        return final_output

    # A waveform cannot be JSON, so audio always leaves as a URL: this route
    # has no field a client could use to ask for it inline, and /v1/audio/speech
    # is the endpoint that hands back the bytes themselves.
    audio_output: list[OutputContent] = []
    persisted_audio = False
    for content in final_output.output:
        if not isinstance(content, OutputAudioContent):
            audio_output.append(content)
            continue

        audio_asset = await media_store.save_audio_content(content)
        audio_output.append(
            content.model_copy(
                update={
                    "audio_url": str(
                        request.url_for(
                            "get_generated_audio_content",
                            audio_id=audio_asset.asset_id,
                        )
                    ),
                    "samples": None,
                }
            )
        )
        persisted_audio = True
    if persisted_audio:
        return final_output.model_copy(update={"output": audio_output})

    video_options = open_responses_request.body.provider_options.video
    if (
        video_options is not None
        and final_output.output
        and len(final_output.output) == 1
        and isinstance(final_output.output[0], OutputVideoContent)
        and final_output.output[0].frames is not None
    ):
        video_response_format = video_options.response_format
        video_content = final_output.output[0]
        frames = video_content.frames
        assert frames is not None

        # Return the video content as a base64-encoded string
        if video_response_format == GeneratedMediaResponseFormat.b64_json:
            video_bytes = media_store.encode_video_content(
                content=video_content,
                frames_per_second=video_options.frames_per_second or 16,
            )
            return final_output.model_copy(
                update={
                    "output": [
                        OutputVideoContent(
                            video_data=encode_video_bytes_b64(video_bytes),
                            format="mp4",
                            frames_per_second=video_options.frames_per_second
                            or 16,
                            num_frames=frames.shape[0],
                            frames=None,
                        )
                    ]
                }
            )

        # Save the video content to the media store
        video_asset = await media_store.save_video_content(
            content=video_content,
            frames_per_second=video_options.frames_per_second or 16,
        )
        video_url = str(
            request.url_for(
                "get_generated_video_content",
                video_id=video_asset.asset_id,
            )
        )
        return final_output.model_copy(
            update={
                "output": [
                    OutputVideoContent(
                        video_url=video_url,
                        format="mp4",
                        frames_per_second=video_options.frames_per_second or 16,
                        num_frames=frames.shape[0],
                        frames=None,
                    )
                ]
            }
        )

    image_options = open_responses_request.body.provider_options.image
    image_response_format = (
        image_options.response_format
        if image_options is not None
        else GeneratedMediaResponseFormat.url
    )
    if image_response_format == GeneratedMediaResponseFormat.b64_json:
        return final_output

    persisted_output: list[OutputContent] = []
    for content in final_output.output:
        if not isinstance(content, OutputImageContent):
            persisted_output.append(content)
            continue

        image_asset = await media_store.save_image_content(content)
        image_url = str(
            request.url_for(
                "get_generated_image_content",
                image_id=image_asset.asset_id,
            )
        )
        persisted_output.append(
            content.model_copy(
                update={
                    "image_url": image_url,
                    "image_data": None,
                }
            )
        )

    return final_output.model_copy(update={"output": persisted_output})


def _get_media_store(request: Request) -> GeneratedMediaStore:
    media_store = getattr(request.app.state, "media_store", None)
    if not isinstance(media_store, GeneratedMediaStore):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Generated media storage is not available.",
        )
    return media_store


def _require_asset(
    asset: StoredMediaAsset | None, kind: str, asset_id: str
) -> StoredMediaAsset:
    if asset is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Unknown generated {kind}: {asset_id}",
        )
    return asset


def _validate_requested_model(request: Request, requested_model: str) -> None:
    app_state: State = request.app.state
    handler = getattr(app_state, "handler", None)
    pipeline_config = getattr(app_state, "pipeline_config", None)

    served_models: list[str] = []
    if handler is not None and getattr(handler, "model_name", None):
        served_models.append(handler.model_name)

    if (
        pipeline_config is not None
        and getattr(pipeline_config, "models", None) is not None
        and getattr(pipeline_config.models, "model_name", None)
    ):
        served_models.append(pipeline_config.models.model_name)

    served_models = list(dict.fromkeys(served_models))
    if not served_models:
        return

    if requested_model not in served_models:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                f"Unknown model '{requested_model}', currently serving "
                f"'{served_models}'."
            ),
        )
