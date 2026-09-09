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
"""FastAPI wrapper used by cascade route adapters."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Iterable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, StreamingResponse

_logger = logging.getLogger(__name__)


class CascadeFastAPI(FastAPI):
    """FastAPI subclass that centralizes the current web-layer integration.

    This consolidates direct usage of both FastAPI and uvicorn in one place
    so the cascade serving layer can be swapped to another framework later
    without rewriting the pipeline route adapters and server entrypoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_exception_handler(
            RequestValidationError, self._log_validation_error
        )

    def streaming_response(
        self,
        content: (
            AsyncIterable[str | bytes | memoryview[int]]
            | Iterable[str | bytes | memoryview[int]]
        ),
        *,
        media_type: str | None = None,
    ) -> StreamingResponse:
        """Create a FastAPI-compatible streaming response."""
        return StreamingResponse(content, media_type=media_type)

    async def _log_validation_error(
        self,
        request: Request,
        exc: Exception,
    ) -> Response:
        """Log request bodies that fail FastAPI validation."""
        assert isinstance(exc, RequestValidationError)
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="replace")
        _logger.warning(
            "Request validation failed for %s %s: errors=%s payload=%s",
            request.method,
            request.url.path,
            exc.errors(),
            body_text,
        )
        return await request_validation_exception_handler(request, exc)

    async def serve(self, host: str, port: int) -> None:
        """Serve this FastAPI app."""
        config = uvicorn.Config(
            self,
            host=host,
            port=port,
            lifespan="off",
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()
