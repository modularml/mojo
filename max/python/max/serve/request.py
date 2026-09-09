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


import logging
import re
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from max.serve._error_envelope import openai_error_body
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch

logger = logging.getLogger("max.serve")

# Liveness/observability endpoints are hit by periodic probes and scrapers.
# Counting them would swamp the ``maxserve.request_count`` metric (which tracks
# API request volume), so they are excluded. ``/metrics`` is a mounted sub-app,
# so its subpaths are excluded too.
_UNCOUNTED_PATH_RE = re.compile(r"/(?:health|version|ping|metrics(?:/.*)?)")


def _should_count_request(path: str) -> bool:
    return _UNCOUNTED_PATH_RE.fullmatch(path) is None


def register_request(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_session(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.request_timer = StopWatch()
        # Record the request against the final HTTP status code. This is the
        # authoritative place to label ``maxserve.request_count`` with the
        # return code: it sees the status of every request, including failures
        # (e.g. a bad image URL) that are rejected before reaching the response
        # generator, and it reflects the code actually sent to the client
        # rather than a value guessed mid-stream.
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except HTTPException as e:
            status_code = e.status_code
            raise  # already wrapped
        except Exception:
            # Returned rather than raised. This middleware runs outside
            # Starlette's ExceptionMiddleware, so a raise here never reaches the
            # app's HTTPException handler; it lands in ServerErrorMiddleware,
            # which answers with a bare text/plain 500 and then re-raises,
            # prompting uvicorn to close the connection out from under a client
            # that was owed a response. ``status_code`` is already 500.
            logger.exception("Exception in request session : %s", request_id)
            return JSONResponse(
                status_code=500,
                content=openai_error_body(500, "Internal server error."),
                headers={"X-Request-ID": request_id},
            )
        finally:
            if _should_count_request(request.url.path):
                METRICS.request_count(status_code, request.url.path)
