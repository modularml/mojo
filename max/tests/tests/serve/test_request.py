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
"""Tests for the request-session middleware, especially that the
``maxserve.request_count`` metric is labeled with the real HTTP status code."""

from unittest import mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from max.serve import request as request_module
from max.serve.request import _should_count_request, register_request


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/ok")
    async def ok() -> JSONResponse:
        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/v1/bad-image")
    async def bad_image() -> Response:
        # Mirrors an image-fetch failure surfaced as an HTTP 400 by the route.
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    @app.get("/v1/crash")
    async def crash() -> Response:
        raise RuntimeError("unhandled")

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    register_request(app)
    return app


@pytest.fixture
def request_count(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    counter = mock.Mock()
    monkeypatch.setattr(request_module.METRICS, "request_count", counter)
    return counter


def test_records_success_code(request_count: mock.Mock) -> None:
    with TestClient(_make_app()) as client:
        response = client.get("/v1/ok")
    assert response.status_code == 200
    request_count.assert_called_once_with(200, "/v1/ok")


def test_records_error_code(request_count: mock.Mock) -> None:
    # A non-200 response (e.g. a failed image fetch) must be counted with its
    # real status code, not 200.
    with TestClient(_make_app()) as client:
        response = client.get("/v1/bad-image")
    assert response.status_code == 400
    request_count.assert_called_once_with(400, "/v1/bad-image")


def test_records_500_for_unhandled_exception(request_count: mock.Mock) -> None:
    # ``raise_server_exceptions`` stays at its default: the middleware answers
    # the request itself, so the exception never reaches the transport.
    with TestClient(_make_app()) as client:
        response = client.get("/v1/crash")
    assert response.status_code == 500
    request_count.assert_called_once_with(500, "/v1/crash")


def test_unhandled_exception_returns_openai_error_envelope() -> None:
    """An unhandled route exception is answered in the OpenAI error shape.

    Raising from this middleware instead bypasses the app's ``HTTPException``
    handler -- it runs outside Starlette's exception middleware -- and reaches
    ``ServerErrorMiddleware``, which replies in ``text/plain`` and re-raises,
    closing the connection under a client that was owed a response.
    """
    with TestClient(_make_app()) as client:
        response = client.get("/v1/crash")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "error": {
            "code": "500",
            "message": "Internal server error.",
            "param": "",
            "type": "api_error",
        }
    }


def test_skips_probe_endpoints(request_count: mock.Mock) -> None:
    with TestClient(_make_app()) as client:
        assert client.get("/health").status_code == 200
    request_count.assert_not_called()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/chat/completions", True),
        ("/v1/embeddings", True),
        ("/invocations", True),
        ("/health", False),
        ("/version", False),
        ("/ping", False),
        ("/metrics", False),
        ("/metrics/foo", False),
    ],
)
def test_should_count_request(path: str, expected: bool) -> None:
    assert _should_count_request(path) is expected
