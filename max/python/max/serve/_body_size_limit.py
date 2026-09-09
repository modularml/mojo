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

"""ASGI middleware enforcing a maximum request body size.

Without a bound, a single client can force the server to buffer an arbitrarily
large request body in memory (the routes read the whole body before validating
it), so a handful of connections can exhaust host memory. This middleware caps
the accepted body size and rejects anything larger with HTTP 413.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from asgiref import typing as asgi_types
from fastapi import HTTPException

logger = logging.getLogger("max.serve")

# HTTP "Content Too Large".
_HTTP_CONTENT_TOO_LARGE = 413


def _declared_content_length(
    headers: Iterable[tuple[bytes, bytes]],
) -> int | None:
    """Returns the ``Content-Length`` a client declared, if any and valid.

    A missing, duplicated, or malformed header returns ``None``; the body is
    then bounded by counting received bytes instead of trusting the header.
    """
    value: int | None = None
    for name, raw in headers:
        if name.lower() != b"content-length":
            continue
        if value is not None:
            # More than one Content-Length header: ambiguous, so trust none of
            # them and let the received-byte count enforce the limit instead.
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
    return value


class RequestBodySizeLimitMiddleware:
    """Rejects HTTP requests whose body exceeds ``max_bytes``.

    Enforced two ways so a client cannot evade it by omitting or lying about
    ``Content-Length``: an honest oversized ``Content-Length`` is refused before
    any body is read, and the received bytes are counted as they stream in so a
    chunked or mislabeled body is refused once it crosses the limit (bounding
    buffered memory to roughly ``max_bytes`` plus one chunk).

    The limit is raised as an ``HTTPException`` from the wrapped ``receive``
    callable, which executes inside the route handler, so Starlette's exception
    machinery maps it to the shared OpenAI-shaped 413 error envelope just like
    any other ``HTTPException``.
    """

    def __init__(
        self, app: asgi_types.ASGI3Application, *, max_bytes: int
    ) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(
        self,
        scope: asgi_types.Scope,
        receive: asgi_types.ASGIReceiveCallable,
        send: asgi_types.ASGISendCallable,
    ) -> None:
        if self._max_bytes <= 0 or scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        declared = _declared_content_length(scope["headers"])
        received = 0

        async def limited_receive() -> asgi_types.ASGIReceiveEvent:
            nonlocal received
            # Raise from inside receive, not from ``__call__``: only here, in
            # the route handler's frame, does the error reach Starlette's
            # exception handlers and become the shared 413 envelope rather than
            # a bare 500. An honest oversized Content-Length is refused before
            # the first byte; a missing or dishonest one is caught by counting.
            if declared is not None and declared > self._max_bytes:
                raise self._too_large(declared)
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > self._max_bytes:
                    raise self._too_large(received)
            return event

        await self._app(scope, limited_receive, send)

    def _too_large(self, size: int) -> HTTPException:
        logger.warning(
            "Rejecting request body of %d bytes (limit %d bytes)",
            size,
            self._max_bytes,
        )
        return HTTPException(
            status_code=_HTTP_CONTENT_TOO_LARGE,
            detail=(
                f"Request body is too large. The maximum allowed size is "
                f"{self._max_bytes} bytes."
            ),
        )
