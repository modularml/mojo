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

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

import logging
import socket
import tempfile
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from max.pipelines.context import BaseContext
from max.pipelines.lib import PIPELINE_REGISTRY, MemoryPlan, PipelineConfig
from max.pipelines.modeling.types import (
    PipelineOutput,
    PipelinesFactory,
    PipelineTask,
    PipelineTokenizer,
)
from max.serve._body_size_limit import RequestBodySizeLimitMiddleware
from max.serve._error_envelope import openai_error_body
from max.serve.config import APIType, MetricRecordingMethod, Settings
from max.serve.media import GeneratedMediaStore
from max.serve.pipelines.eplb_stats_rpc import (
    EplbStatsFrontend,
    EplbStatsResetFrontend,
)
from max.serve.pipelines.general_handler import GeneralPipelineHandler
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheFrontend
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.recordreplay.jsonl import JSONLFileRecorder
from max.serve.recordreplay.middleware import RecorderMiddleware
from max.serve.request import register_request
from max.serve.router import (
    kserve_routes,
    openai_routes,
    openresponses_routes,
    sagemaker_routes,
)
from max.serve.router._image_resolution import fetch_media_data_uri
from max.serve.telemetry.common import send_telemetry_log
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface import RequestQueueFull
from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path
from max.serve.worker_interface.lora_queue import LoRAQueue
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface
from uvicorn import Config

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
    APIType.SAGEMAKER: sagemaker_routes,
    APIType.OPENRESPONSES: openresponses_routes,
}

logger = logging.getLogger("max.serve")


def validate_port_is_free(port: int) -> int:
    # check if port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            return port
        except OSError as e:
            raise ValueError(
                f"The network port {port} is already in use"
            ) from e


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    # Pipeline config
    model_factory: PipelinesFactory  # type: ignore
    pipeline_config: PipelineConfig
    tokenizer: PipelineTokenizer[Any, Any, Any]
    task: PipelineTask = PipelineTask.TEXT_GENERATION
    reasoning_parser_name: str | None = None
    temperature: float | None = None
    thinking_temperature: float | None = None
    memory_plan: MemoryPlan | None = None
    """Memory plan the pipeline was sized against; ``None`` only for test
    servers built without one (e.g. echo pipelines)."""


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
    zmq_endpoint_base: str,
) -> AsyncGenerator[None]:
    try:
        if not settings.disable_telemetry:
            send_telemetry_log(
                serving_settings.pipeline_config.models.model_name
            )
    except Exception as e:
        logger.warning("Failed to send telemetry log: %s", e)

    if settings.offline_inference:
        raise ValueError(
            "It is not valid to start the API Server if the server is in offline inference mode"
        )

    logger.info("Starting server...")

    async with AsyncExitStack() as exit_stack:
        media_root = Path(
            exit_stack.enter_context(
                tempfile.TemporaryDirectory(prefix="max_serve_media_")
            )
        )
        app.state.media_store = GeneratedMediaStore(
            media_root,
            max_storage_bytes=(
                settings.generated_media_storage_mb * 1024 * 1024
            ),
        )

        # start telemetry worker and configure Metrics to use it

        metric_client = await exit_stack.enter_async_context(
            start_telemetry_consumer(settings)
        )
        METRICS.configure(client=metric_client)

        # start model worker

        override_architecture: str | None = None

        model_worker_interface = ZmqModelWorkerInterface[
            BaseContext, PipelineOutput
        ](
            serving_settings.task,
            PIPELINE_REGISTRY.retrieve_context_type(
                serving_settings.pipeline_config,
                override_architecture=override_architecture,
                task=serving_settings.task,
            ),
            # Cap the in-transit request backlog to the model worker (HTTP 429
            # when full). ``None`` keeps the queue unbounded.
            request_queue_size=settings.max_queue_size,
        )
        model_worker = await exit_stack.enter_async_context(
            start_model_worker(
                serving_settings.model_factory,
                serving_settings.pipeline_config,
                settings,
                metric_client,
                model_worker_interface=model_worker_interface,
                zmq_endpoint_base=zmq_endpoint_base,
                memory_plan=serving_settings.memory_plan,
            )
        )

        lora_queue: LoRAQueue | None = (
            LoRAQueue(
                zmq_endpoint_base,
                serving_settings.pipeline_config.lora.lora_paths,
            )
            if serving_settings.pipeline_config.lora
            else None
        )

        METRICS.pipeline_load(
            serving_settings.pipeline_config.models.model_name
        )

        pipeline = {
            PipelineTask.TEXT_GENERATION: lambda: TokenGeneratorPipeline(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                model_worker=model_worker,
                reasoning_parser_name=serving_settings.reasoning_parser_name,
                min_chunk_tokens=settings.stream_min_chunk_tokens,
            ),
            PipelineTask.EMBEDDINGS_GENERATION: lambda: TokenGeneratorPipeline(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                model_worker=model_worker,
            ),
            # Pixel generation uses only the OpenResponses API via GeneralPipelineHandler
            PipelineTask.PIXEL_GENERATION: lambda: GeneralPipelineHandler(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                model_worker=model_worker,
                lora_queue=lora_queue,
            ),
            # Audio generation serves /v1/audio/speech and /v1/responses, both
            # of which go through GeneralPipelineHandler.
            PipelineTask.AUDIO_GENERATION: lambda: GeneralPipelineHandler(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                model_worker=model_worker,
                lora_queue=lora_queue,
            ),
        }[serving_settings.task]()

        # Store pipeline (may be GeneralPipelineHandler or modality-specific wrapper)
        # Legacy API routes (OpenAI, KServe, SageMaker) use modality-specific wrappers
        # OpenResponses API uses GeneralPipelineHandler
        app.state.pipeline = pipeline
        app.state.pipeline_config = serving_settings.pipeline_config
        app.state.memory_plan = serving_settings.memory_plan
        # The served task, for routes that only mean something for one of them:
        # every API type is mounted regardless, so a route with no counterpart
        # in the served model has to refuse the request itself.
        app.state.task = serving_settings.task

        # Also store as handler for OpenResponses API route compatibility
        # For the media tasks, this is the same as pipeline
        # For other tasks, we also create a separate handler instance
        if serving_settings.task in (
            PipelineTask.PIXEL_GENERATION,
            PipelineTask.AUDIO_GENERATION,
        ):
            app.state.handler = pipeline
        else:
            app.state.handler = GeneralPipelineHandler(
                model_name=serving_settings.pipeline_config.models.model_name,
                tokenizer=serving_settings.tokenizer,
                model_worker=model_worker,
                lora_queue=lora_queue,
            )

        logger.info(
            f"\n\n{'*' * 80}\n\n"
            f"{f'🚀 Server ready on http://{settings.host}:{settings.port} (Press CTRL+C to quit)'.center(80)}\n\n"
            f"{'*' * 80}\n"
        )

        yield

        logger.info("Shutting down workers...")


def version() -> JSONResponse:
    """Returns max-serve version information."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("max")
        return JSONResponse({"version": package_version})
    except PackageNotFoundError:
        logger.debug("Version could not be reported for max.")
        return JSONResponse({"version": "unknown"})


async def health() -> Response:
    """Health check, tools like lm-eval use this to check for readiness."""
    return Response(status_code=200)


def make_metrics_app() -> Callable[..., Any]:
    from prometheus_client import disable_created_metrics, make_asgi_app

    disable_created_metrics()
    return make_asgi_app()


async def _openai_http_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_body(exc.status_code, str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def _openai_validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    return JSONResponse(
        status_code=422, content=openai_error_body(422, str(exc))
    )


async def _request_queue_full_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Map a full model-worker request queue to HTTP 429.

    ``RequestQueueFull`` is raised at admission (the push to the worker, awaited
    before any response status is committed) by any endpoint that submits to the
    worker, so it is handled centrally here rather than per route. Returns the
    OpenAI ``rate_limit_error`` envelope with a ``Retry-After`` hint; the
    rejection rate is observable via ``maxserve.request_count{code="429"}``.
    """
    assert isinstance(exc, RequestQueueFull)
    request_id = getattr(request.state, "request_id", "<unknown>")
    logger.warning("Request queue full for request %s", request_id)
    return JSONResponse(
        status_code=429,
        content=openai_error_body(
            429, "Server is at capacity. Please retry later."
        ),
        headers={"Retry-After": "1"},
    )


def fastapi_app(
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    zmq_endpoint_base = generate_zmq_ipc_path()

    @asynccontextmanager
    async def lifespan_wrap(app: FastAPI) -> AsyncGenerator[None, None]:
        # Binds the extra arguments so this matches the FastAPI lifespan
        # signature. Used by ASGI test clients (e.g. starlette TestClient).
        # The production entrypoint instead enters `lifespan` directly around
        # `uvicorn` running with `lifespan="off"` (see
        # `serve_api_server_and_model_worker`), so a worker crash tears down
        # the server via task cancellation rather than fragile signal handling.
        async with lifespan(app, settings, serving_settings, zmq_endpoint_base):
            yield

    app = FastAPI(title="MAX Serve", lifespan=lifespan_wrap)
    app.state.zmq_endpoint_base = zmq_endpoint_base

    if settings.max_request_bytes > 0:
        app.add_middleware(
            RequestBodySizeLimitMiddleware,
            max_bytes=settings.max_request_bytes,
        )

    if settings.transaction_recording_file is not None:
        transaction_recording_file = settings.transaction_recording_file
        app.add_middleware(
            RecorderMiddleware,  # type: ignore
            recorder_factory=(
                lambda: JSONLFileRecorder(transaction_recording_file)
            ),
            include_responses=settings.transaction_recording_include_responses,
        )

    if (
        not settings.disable_telemetry
        and settings.metric_recording == MetricRecordingMethod.ASYNCIO
    ):
        app.mount("/metrics", make_metrics_app())

    app.add_api_route("/version", version)
    app.add_api_route("/health", health)

    reset_prefix_cache_frontend = ResetPrefixCacheFrontend(zmq_endpoint_base)

    async def reset_prefix_cache() -> Response:
        """Reset the prefix cache."""
        try:
            model_config = serving_settings.pipeline_config.model
        except ValueError:
            return Response(
                status_code=400,
                content="No main model configured (diffusion pipeline). Ignoring request",
            )
        if not model_config.kv_cache.enable_prefix_caching:
            return Response(
                status_code=400,
                content="Prefix caching is not enabled. Ignoring request",
            )

        reset_prefix_cache_frontend.enqueue_reset_prefix_cache()
        return Response(status_code=200, content="Success")

    app.add_api_route(
        "/reset_prefix_cache", reset_prefix_cache, methods=["POST"]
    )

    eplb_stats_frontend = EplbStatsFrontend(zmq_endpoint_base)

    async def eplb_stats() -> Response:
        """Get the EPLB stats snapshot."""
        if not settings.eplb_profile:
            return Response(
                status_code=404,
                content="EPLB stats profiling is not enabled.",
            )
        try:
            snap = await eplb_stats_frontend.fetch_snapshot()
        except TimeoutError:
            return Response(
                status_code=504,
                content="EPLB stats fetch timed out.",
            )
        return JSONResponse(snap.to_dict())

    app.add_api_route("/max_internal/eplb_stats", eplb_stats, methods=["GET"])

    # reset eplb stat endpoint
    eplb_stats_reset_frontend = EplbStatsResetFrontend(zmq_endpoint_base)

    async def eplb_stats_reset() -> Response:
        """Reset the EP stats accumulator on the worker."""
        if not settings.eplb_profile:
            return Response(
                status_code=404,
                content="EP stats profiling is not enabled.",
            )
        eplb_stats_reset_frontend.enqueue_reset()
        return Response(status_code=200, content="Success")

    app.add_api_route(
        "/max_internal/eplb_stats_reset", eplb_stats_reset, methods=["POST"]
    )
    for api_type in settings.api_types:
        app.include_router(ROUTES[api_type].router)

    app.state.settings = settings

    # The /v1/responses input schema takes data: URIs only, so a client-supplied
    # http(s) image must be fetched and inlined before the body validates. The
    # request library cannot do that itself (it does not depend on max.serve, and
    # a second downloader there would be a second SSRF surface), so hand it the
    # shared resolver, which carries the byte caps and host validation.
    async def fetch_media_data_uri_for_app(url: str) -> str:
        return await fetch_media_data_uri(url, settings)

    app.state.media_data_uri_fetcher = fetch_media_data_uri_for_app

    register_request(app)

    app.add_exception_handler(HTTPException, _openai_http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, _openai_validation_exception_handler
    )
    app.add_exception_handler(
        RequestQueueFull, _request_queue_full_exception_handler
    )

    return app


def fastapi_config(app: FastAPI, server_settings: Settings) -> Config:
    config = Config(
        app=app,
        log_config=None,
        loop="uvloop",
        host=server_settings.host,
        port=server_settings.port,
        timeout_graceful_shutdown=server_settings.graceful_shutdown_timeout_s,
        # uvicorn defaults to closing idle connections after 5s, far below the
        # idle timeout of a pooling client, which makes the server the side
        # that closes and turns the race into client-visible TCP resets.
        timeout_keep_alive=server_settings.http_keepalive_timeout_s,
        # The serving lifespan (model worker, pipeline, telemetry) is entered
        # explicitly by the entrypoint around `server.serve()` so that a worker
        # crash cancels the serving task directly. Keep uvicorn out of the
        # lifespan business to avoid the previous double-SIGINT shutdown hack.
        lifespan="off",
    )

    for route in app.routes:
        logger.debug("Route enabled : %s", route)
    return config
