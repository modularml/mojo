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


"""Utilities for serving api server with model worker."""

import logging
import os
import signal
from types import FrameType

import uvloop
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.logging_utils import log_basic_config, log_pipeline_info
from max.pipelines.modeling.types import PipelineTask
from max.profiler import Tracer
from max.serve.api_server import (
    ServingTokenGeneratorSettings,
    fastapi_app,
    fastapi_config,
    lifespan,
    validate_port_is_free,
)
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.process_control import SubprocessExit
from sse_starlette.sse import unpatch_uvicorn_signal_handler
from uvicorn import Server

logger = logging.getLogger("max._entrypoints")


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """Turns SIGTERM into an exception that unwinds the serving stack.

    The serving context manager (model worker, pipeline, telemetry) wraps
    uvicorn's ``server.serve()``, so its teardown runs when that ``async with``
    block exits. uvicorn re-raises a handled signal once ``serve()`` returns;
    ``SIGTERM``'s default disposition silently terminates the process, which
    would skip that teardown and leak the model worker subprocess. Raising here
    unwinds through the context managers instead, so they tear the worker down.
    (``SIGINT`` already raises ``KeyboardInterrupt`` by default, so it does not
    need this.)
    """
    raise SystemExit(128 + signum)


def serve_api_server_and_model_worker(
    settings: Settings,
    pipeline_args: PipelineArgs,
) -> None:
    # Register custom architectures before any architecture-name lookup. Both
    # retrieve_pipeline_task and retrieve_factory below resolve by name; a custom
    # arch that overrides a built-in must be imported first, or the stale lazy
    # built-in entry is materialized instead (and may fail to import).
    PIPELINE_REGISTRY._import_custom_architectures(
        pipeline_args.runtime.custom_architectures
    )

    # Auto-detect pipeline task from the model architecture if not explicitly set.
    if pipeline_args.task == PipelineTask.UNDEFINED:
        arch_name = pipeline_args.main_architecture_name
        pipeline_args = pipeline_args.model_copy(
            update={"task": PIPELINE_REGISTRY.retrieve_pipeline_task(arch_name)}
        )
        logger.info(
            f"Auto-detected pipeline task: {pipeline_args.task.value} "
            f"(model architecture: {arch_name})"
        )

    override_architecture: str | None = None

    # Load tokenizer and pipeline from PIPELINE_REGISTRY.
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config,
        task=pipeline_args.task,
        override_architecture=override_architecture,
    )
    tokenizer = retrieved.tokenizer
    pipeline_factory = retrieved.factory
    memory_plan = retrieved.memory_plan
    log_basic_config(pipeline_config, memory_plan=memory_plan)
    log_pipeline_info(pipeline_config, memory_plan=memory_plan)

    # Dummy model is for diagnostics and overhead benchmarking
    if os.getenv("MAX_SERVE_DUMMY_MODEL"):
        assert pipeline_config.task == PipelineTask.TEXT_GENERATION, (
            "dummy model only implemented for text gen models"
        )
        logging.warning("Replacing pipeline model with dummy model!")
        pipeline_factory = EchoTokenGenerator

    pipeline_settings = ServingTokenGeneratorSettings(
        model_factory=pipeline_factory,
        pipeline_config=pipeline_config,
        tokenizer=tokenizer,
        task=pipeline_config.task,
        reasoning_parser_name=pipeline_config.runtime.reasoning_parser,
        temperature=pipeline_config.runtime.temperature,
        thinking_temperature=pipeline_config.runtime.thinking_temperature,
        memory_plan=memory_plan,
    )

    # Initialize and serve webserver.
    app = fastapi_app(settings, pipeline_settings)
    config = fastapi_config(app=app, server_settings=settings)
    # If likely to fail, don't waste seconds or minutes loading models
    validate_port_is_free(settings.port)

    server = Server(config)

    async def serve() -> None:
        # Enter the serving context (model worker, pipeline, telemetry) first,
        # then run uvicorn with lifespan="off". Because the context manager and
        # server.serve() run in the same task, a model-worker crash (which
        # surfaces as the worker's TaskGroup cancelling this task) tears down
        # the running server directly, without the fragile self-SIGINT signaling
        # the uvicorn lifespan hook previously relied on.
        try:
            async with lifespan(
                app, settings, pipeline_settings, app.state.zmq_endpoint_base
            ):
                await server.serve()
        except SubprocessExit as e:
            logger.error("Worker crashed (%s), shutting down...", e)
            # quietly unwind the api-server to keep logs cleaner
            # so users can focus on the real error printed by the subprocess
            raise SystemExit(1) from None

    # CLI entry point: install our own SIGTERM handler and don't bother
    # restoring it. uvicorn re-raises a handled signal once serve() returns;
    # SIGTERM's default action would silently kill us before the lifespan
    # teardown above, so raise instead to unwind through it. SIGINT is left
    # alone: its default handler already raises KeyboardInterrupt, which
    # unwinds the same way (and is less surprising for interactive use/tests).
    signal.signal(signal.SIGTERM, _exit_on_signal)

    # sse-starlette patches uvicorn's Server.handle_exit on import so a signal
    # tears down every in-flight EventSourceResponse immediately, which cuts
    # streaming completions off mid-generation with no terminal chunk. Its patch
    # guards against SSE streams that never end; ours always do, and uvicorn
    # already bounds the wait by timeout_graceful_shutdown and cancels whatever
    # is left, so restore uvicorn's handler and let that timeout be the only
    # authority. Must run after the import that installs the patch, hence here
    # rather than at module scope. Upstream replaced this helper with
    # AppStatus.disable_automatic_graceful_drain(), and also learned to find the
    # Server by introspecting the SIGTERM handler, so bumping sse-starlette past
    # 2.1.2 means switching to that call.
    unpatch_uvicorn_signal_handler()

    with Tracer("openai_compatible_frontend_server"):
        uvloop.run(serve())
