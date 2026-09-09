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
"""The fixtures for all tests in this directory."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi import FastAPI
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.lib import (
    MAXModelConfig,
    ModelManifest,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.modeling.types import (
    PipelineTask,
    RequestID,
    TextGenerationInputs,
)
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.telemetry.common import configure_metrics


class SleepyEchoTokenGenerator(EchoTokenGenerator):
    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        # Sleep for 1 ms - otherwise, the echo token generator
        # can break some separation of timescale assumptions
        time.sleep(1e-3)
        return super().execute(inputs)


# This has to be picklable and lambdas are not picklable
def echo_factory():  # noqa: ANN201
    return SleepyEchoTokenGenerator()


@pytest.fixture
def mock_pipeline_config() -> PipelineConfig:
    runtime = PipelineRuntimeConfig.model_construct(
        max_batch_size=1,
    )

    model_config = MAXModelConfig.model_construct(served_model_name="echo")
    return PipelineConfig.model_construct(
        runtime=runtime,
        models=ModelManifest({"main": model_config}),
    )


@pytest.fixture()
def echo_app(mock_pipeline_config: PipelineConfig) -> FastAPI:
    tokenizer = EchoPipelineTokenizer()

    serving_settings = ServingTokenGeneratorSettings(
        model_factory=echo_factory,
        pipeline_config=mock_pipeline_config,
        tokenizer=tokenizer,
    )

    settings = Settings(use_heartbeat=True)
    app = fastapi_app(settings, serving_settings)
    return app


@pytest.fixture(scope="session")
def pipeline_config(request: pytest.FixtureRequest):  # noqa: ANN201
    return request.param


@pytest.fixture(scope="session")
def settings_config(request: pytest.FixtureRequest):  # noqa: ANN201
    """Fixture to control settings configuration"""
    return getattr(request, "param", {"MAX_SERVE_USE_HEARTBEAT": True})


@pytest.fixture(scope="function")
def app(
    pipeline_config: PipelineArgs, settings_config: Mapping[str, Any]
) -> FastAPI:
    """The FastAPI app used to serve the model."""

    pipeline_task = PipelineTask.TEXT_GENERATION
    if pipeline_config.model_path == "sentence-transformers/all-mpnet-base-v2":
        pipeline_task = PipelineTask.EMBEDDINGS_GENERATION

    pipeline_cfg = PipelineConfig.from_args(pipeline_config)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_cfg, task=pipeline_task
    )

    serving_settings = ServingTokenGeneratorSettings(
        model_factory=retrieved.factory,
        pipeline_config=pipeline_cfg,
        tokenizer=retrieved.tokenizer,
        task=pipeline_task,
        memory_plan=retrieved.memory_plan,
    )

    settings = Settings(**settings_config)
    configure_metrics(settings)
    app = fastapi_app(settings, serving_settings)
    return app


@pytest.fixture()
def reset_sse_starlette_appstatus_event() -> None:
    """
    Fixture that resets the appstatus event in the sse_starlette app.

    Should be used on any test that uses sse_starlette to stream events.
    """
    # See https://github.com/sysid/sse-starlette/issues/59
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
