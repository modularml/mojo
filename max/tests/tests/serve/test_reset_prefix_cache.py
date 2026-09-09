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


"""Unit tests for /reset_prefix_cache endpoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    IdentityPipelineTokenizer,
    PipelineConfig,
)
from max.pipelines.modeling.types import PipelineTask, TextGenerationRequest
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import EchoTokenGenerator
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheBackend
from tests.serve.conftest import async_timeout


@dataclass(frozen=True)
class MockTokenizer(IdentityPipelineTokenizer[str]):
    async def new_context(self, request: TextGenerationRequest) -> str:
        return ""


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch PIPELINE_REGISTRY.retrieve_context_type to always return TextContext."""

    def _mock_retrieve_context_type(
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext]:
        return TextContext

    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        _mock_retrieve_context_type,
    )


@pytest.fixture(scope="function")
def app(mock_pipeline_config: PipelineConfig) -> Generator[FastAPI, None, None]:
    """Fixture for a FastAPI app using a given pipeline."""
    serving_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=MockTokenizer(),
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], use_heartbeat=False),
        serving_settings,
    )
    yield app


@pytest_asyncio.fixture
async def test_client(app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """Fixture for a asgi TestClient using a given FastAPI app."""
    async with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("enable_prefix_caching", [True], indirect=True)
@pytest.mark.asyncio
@async_timeout(10)
async def test_reset_prefix_cache(test_client: TestClient) -> None:
    # Don't use hardcoded endpoint, since it collides during parallel test runs
    zmq_endpoint_base = test_client.application.state.zmq_endpoint_base
    reset_prefix_cache_backend = ResetPrefixCacheBackend(
        zmq_endpoint_base=zmq_endpoint_base
    )
    assert not reset_prefix_cache_backend.should_reset_prefix_cache()
    response = await test_client.post("/reset_prefix_cache")
    assert response.status_code == 200
    assert response.text == "Success"
    # this is blocking since it may take some time for the request to arrive
    # to the backend
    assert reset_prefix_cache_backend.should_reset_prefix_cache(blocking=True)
    assert not reset_prefix_cache_backend.should_reset_prefix_cache()


@pytest.mark.parametrize("enable_prefix_caching", [True], indirect=True)
@pytest.mark.asyncio
@async_timeout(10)
async def test_reset_prefix_cache_get_returns_405_error(
    test_client: TestClient,
) -> None:
    # POST is OK but GET is not
    response = await test_client.get("/reset_prefix_cache")
    assert response.status_code == 405
    assert response.text == '{"detail":"Method Not Allowed"}'


@pytest.mark.parametrize("enable_prefix_caching", [False], indirect=True)
@pytest.mark.asyncio
@async_timeout(10)
async def test_reset_prefix_cache_returns_400_error(
    test_client: TestClient,
) -> None:
    response = await test_client.post("/reset_prefix_cache")
    assert response.status_code == 400
    assert response.text == "Prefix caching is not enabled. Ignoring request"
