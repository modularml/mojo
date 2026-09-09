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

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from max.pipelines.context import TextContext
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import PipelineTask
from max.serve._body_size_limit import RequestBodySizeLimitMiddleware
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)

_LIMIT = 1024


def _app_with_limit(max_bytes: int) -> FastAPI:
    """A minimal app that echoes the received body length behind the limiter."""
    app = FastAPI()
    app.add_middleware(RequestBodySizeLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"len": len(body)}

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_under_limit_passes() -> None:
    client = TestClient(_app_with_limit(_LIMIT))
    body = b"x" * (_LIMIT - 1)
    response = client.post("/echo", content=body)
    assert response.status_code == 200
    assert response.json() == {"len": _LIMIT - 1}


def test_at_limit_passes() -> None:
    client = TestClient(_app_with_limit(_LIMIT))
    body = b"x" * _LIMIT
    response = client.post("/echo", content=body)
    assert response.status_code == 200
    assert response.json() == {"len": _LIMIT}


def test_over_limit_content_length_rejected() -> None:
    client = TestClient(_app_with_limit(_LIMIT))
    body = b"x" * (_LIMIT + 1)
    response = client.post("/echo", content=body)
    assert response.status_code == 413


def test_over_limit_chunked_without_content_length_rejected() -> None:
    """A streamed body with no ``Content-Length`` is still bounded."""

    def chunks() -> Iterator[bytes]:
        # Ten chunks well past the limit; httpx sends these chunked (no
        # Content-Length), exercising the received-byte accounting path.
        for _ in range(10):
            yield b"x" * _LIMIT

    client = TestClient(_app_with_limit(_LIMIT))
    response = client.post("/echo", content=chunks())
    assert response.status_code == 413


def test_lying_content_length_still_bounded() -> None:
    """An understated ``Content-Length`` cannot smuggle an oversized body."""

    def chunks() -> Iterator[bytes]:
        for _ in range(10):
            yield b"x" * _LIMIT

    client = TestClient(_app_with_limit(_LIMIT))
    response = client.post(
        "/echo", content=chunks(), headers={"Content-Length": "10"}
    )
    assert response.status_code == 413


def test_zero_disables_limit() -> None:
    client = TestClient(_app_with_limit(0))
    body = b"x" * (_LIMIT * 4)
    response = client.post("/echo", content=body)
    assert response.status_code == 200
    assert response.json() == {"len": _LIMIT * 4}


def test_get_request_unaffected() -> None:
    client = TestClient(_app_with_limit(_LIMIT))
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.fixture(autouse=True)
def patch_pipeline_registry_context_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_server_app_returns_openai_error_envelope(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """The real server app maps an oversized body to the OpenAI 413 envelope."""
    settings = Settings(
        api_types=[APIType.KSERVE],
        use_heartbeat=False,
        max_request_bytes=_LIMIT,
    )
    pipeline_settings = ServingTokenGeneratorSettings(
        model_factory=EchoTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    app = fastapi_app(settings, pipeline_settings)

    # No lifespan needed: the limiter rejects before the route touches the
    # model worker, so requests are issued without entering the app context.
    client = TestClient(app)
    response = client.post(
        "/v2/models/echo/versions/1/infer", content=b"x" * (_LIMIT + 1)
    )
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "413"
    assert error["type"] == "invalid_request_error"
