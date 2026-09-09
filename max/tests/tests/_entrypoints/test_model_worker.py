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
# Unit tests for model_worker
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib import PipelineConfig
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
)
from max.serve import api_server
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.process_control import SubprocessExit
from max.serve.telemetry.metrics import NoopClient
from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface


@dataclass(frozen=True)
class MockContext(Mock):
    """Mock context that implements BaseContext protocol."""

    request_id: RequestID
    status: GenerationStatus = GenerationStatus.ACTIVE

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self.status.is_done


@pytest.mark.asyncio
async def test_model_worker_propagates_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker context manager."""
    settings = Settings()

    with pytest.raises(ValueError, match="kaboom"):
        async with start_model_worker(
            EchoTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION,
                context_type=TextContext,
            ),
            zmq_endpoint_base=generate_zmq_ipc_path(),
            memory_plan=None,
        ):
            raise ValueError("kaboom")


class MockInvalidTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    ERROR_MESSAGE = "CRASH TEST DUMMY"

    def __init__(self) -> None:
        raise ValueError(MockInvalidTokenGenerator.ERROR_MESSAGE)

    def execute(
        self, inputs: TextGenerationInputs[MockContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_propagates_construction_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests that a worker that crashes during construction surfaces to the parent.

    The child's :py:class:`ValueError` is logged by the subprocess itself; the
    parent only learns that the worker exited with a non-zero code (see
    :py:class:`SubprocessExit`).
    """
    settings = Settings()

    with pytest.raises(SubprocessExit):
        async with start_model_worker(
            MockInvalidTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION,
                context_type=TextContext,
            ),
            metric_client=NoopClient(),
            zmq_endpoint_base=generate_zmq_ipc_path(),
            memory_plan=None,
        ):
            pass


class MockSlowTokenGenerator(
    Pipeline[TextGenerationInputs[MockContext], TextGenerationOutput]
):
    def __init__(self) -> None:
        time.sleep(0.2)

    def execute(
        self, inputs: TextGenerationInputs[MockContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        raise ValueError()

    def release(self, request_id: RequestID) -> None:
        pass


@pytest.mark.asyncio
async def test_model_worker_start_timeout(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings(mw_timeout_s=0.1)

    with pytest.raises(
        TimeoutError, match="Model Worker failed to become ready"
    ):
        async with start_model_worker(
            MockSlowTokenGenerator,
            mock_pipeline_config,
            settings=settings,
            metric_client=NoopClient(),
            model_worker_interface=ZmqModelWorkerInterface(
                PipelineTask.TEXT_GENERATION, context_type=TextContext
            ),
            zmq_endpoint_base=generate_zmq_ipc_path(),
            memory_plan=None,
        ):
            pass


class MockTokenizer(PipelineTokenizer):  # type: ignore
    @property
    def eos_token_ids(self) -> set[int]:
        return set()

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def new_context(self, req: Any) -> Any:
        return None

    async def encode(self, text: str, spectok: bool) -> list[int]:
        return []

    async def decode(self, encoded: Any, **kwargs) -> str:
        return ""


@pytest.mark.asyncio
async def test_lifespan_propagates_worker_exception(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Tests raising in the model worker task."""
    settings = Settings()
    serving_settings = api_server.ServingTokenGeneratorSettings(
        model_factory=MockInvalidTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )

    # The child's ValueError is logged by the subprocess; the parent observes
    # SubprocessExit (the production CLI catches this in
    # serve_api_and_model_worker.py).
    with pytest.raises(SubprocessExit):
        async with api_server.lifespan(
            FastAPI(),
            settings,
            serving_settings,
            generate_zmq_ipc_path(),
        ):
            pass


@pytest.mark.asyncio
async def test_lifespan_startup_crash_skips_server(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """A worker that crashes on startup must not start the server.

    This mirrors the production entrypoint, which enters ``lifespan`` and then
    awaits ``server.serve()`` in the same task (uvicorn runs with
    ``lifespan="off"``). When the model worker fails to come up, entering the
    context manager raises directly, so the server body is never reached and
    the worker exception propagates -- no self-SIGINT required.
    """
    settings = Settings()
    serving_settings = api_server.ServingTokenGeneratorSettings(
        model_factory=MockInvalidTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )

    server_started = False

    with pytest.raises(SubprocessExit):
        async with api_server.lifespan(
            FastAPI(),
            settings,
            serving_settings,
            generate_zmq_ipc_path(),
        ):
            # Stand-in for ``await server.serve()``.
            server_started = True
            await asyncio.Event().wait()

    assert not server_started, "server must not start when the worker crashes"
