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

"""ASGI middleware to record transactions."""

from __future__ import annotations

import contextlib
import datetime
import urllib.parse
from collections.abc import Callable
from typing import overload

from asgiref import typing as asgi_types

from . import schema
from .interfaces import Recorder

__all__ = [
    # Please keep this list alphabetized.
    "LifespanRequiredError",
    "RecorderMiddleware",
]


def _now() -> datetime.datetime:
    """Get the current timestamp, with proper time zone information."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone()


class LifespanRequiredError(Exception):
    """Raised when ASGI lifespan is required but not present."""


_RECORDER_KEY = f"{__name__}.recorder"


class RecorderMiddleware:
    """ASGI middleware to record transactions."""

    _app: asgi_types.ASGI3Application
    _recorder: Recorder | None
    _recorder_factory: (
        Callable[[], contextlib.AbstractContextManager[Recorder]] | None
    )
    _include_responses: bool

    @overload
    def __init__(
        self,
        app: asgi_types.ASGI3Application,
        *,
        recorder: Recorder,
        include_responses: bool = False,
    ) -> None:
        """Create a recorder middleware using a pre-initialized recorder.

        The recorder must be provided as the 'recorder' parameter.

        If include_responses is True, responses are recorded as well.
        Otherwise, only the request will be recorded.
        """

    @overload
    def __init__(
        self,
        app: asgi_types.ASGI3Application,
        *,
        recorder_factory: Callable[
            [], contextlib.AbstractContextManager[Recorder]
        ],
        include_responses: bool = False,
    ) -> None:
        """Create a recorder middleware that manages lifetime of the recorder.

        recorder_factory must be a callable that returns a context manager that
        yields a recorder.  The context manager will be entered upon ASGI
        lifespan start and will be exited upon ASGI lifespan end.

        If include_responses is True, responses are recorded as well.
        Otherwise, only the request will be recorded.
        """

    def __init__(
        self,
        app: asgi_types.ASGI3Application,
        *,
        recorder: Recorder | None = None,
        recorder_factory: Callable[
            [], contextlib.AbstractContextManager[Recorder]
        ]
        | None = None,
        include_responses: bool = False,
    ) -> None:
        if (recorder is not None) + (recorder_factory is not None) != 1:
            raise TypeError(
                "Exactly one of recorder and recorder_factory must be provided"
            )
        self._app = app
        self._recorder = recorder
        self._recorder_factory = recorder_factory
        self._include_responses = include_responses

    async def __call__(
        self,
        scope: asgi_types.Scope,
        receive: asgi_types.ASGIReceiveCallable,
        send: asgi_types.ASGISendCallable,
    ) -> None:
        if scope["type"] == "lifespan" and self._recorder_factory is not None:
            await self._intercept_lifespan(scope, receive, send)
            return

        if scope["type"] != "http":
            # Only HTTP transactions are recorded.
            await self._app(scope, receive, send)
            return

        if self._recorder is not None:
            recorder = self._recorder
        else:
            state_value = scope.get("state", {}).get(_RECORDER_KEY)
            if state_value is None:
                raise LifespanRequiredError(
                    "Transaction recording middleware instantiated in "
                    "lifespan-aware mode, but request occurred outside of "
                    "lifespan."
                )
            assert isinstance(state_value, Recorder)
            recorder = state_value
        context = _TransactionRecordingContext(
            scope=scope,
            send=send,
            receive=receive,
            recorder=recorder,
            include_responses=self._include_responses,
        )
        await self._app(scope, context.receive, context.send)

    async def _intercept_lifespan(
        self,
        scope: asgi_types.LifespanScope,
        receive: asgi_types.ASGIReceiveCallable,
        send: asgi_types.ASGISendCallable,
    ) -> None:
        factory = self._recorder_factory
        assert factory is not None
        exit_stack = contextlib.ExitStack()

        async def intercepting_receive() -> asgi_types.ASGIReceiveEvent:
            event = await receive()
            if event["type"] == "lifespan.startup":
                scope.setdefault("state", {})[_RECORDER_KEY] = (
                    exit_stack.enter_context(factory())
                )
            elif event["type"] == "lifespan.shutdown":
                scope.get("state", {}).pop(_RECORDER_KEY, None)
                exit_stack.close()
            return event

        await self._app(scope, intercepting_receive, send)


class _TransactionRecordingContext:
    """Context for recording a single HTTP transaction."""

    _scope: asgi_types.HTTPScope
    _underlying_send: asgi_types.ASGISendCallable
    _underlying_receive: asgi_types.ASGIReceiveCallable
    _recorder: Recorder
    _include_responses: bool

    _start_timestamp: datetime.datetime
    _request_headers: list[tuple[bytes, bytes]]
    _request_done_timestamp: datetime.datetime | None
    _request_chunks: list[bytes]
    _response_done: bool
    _response_start: schema.ResponseStart | None
    _response_chunks: list[schema.ResponseChunk]
    _recorded: bool

    def __init__(
        self,
        *,
        scope: asgi_types.HTTPScope,
        send: asgi_types.ASGISendCallable,
        receive: asgi_types.ASGIReceiveCallable,
        recorder: Recorder,
        include_responses: bool,
    ) -> None:
        """Create a new transaction recording context."""
        self._scope = scope
        self._underlying_send = send
        self._underlying_receive = receive
        self._recorder = recorder
        self._include_responses = include_responses

        self._start_timestamp = _now()
        self._request_done_timestamp = None
        self._request_chunks = []
        self._response_done = False
        self._response_start = None
        self._response_chunks = []
        self._recorded = False

        # Headers is an Iterable, which may allow only a single iteration (e.g.
        # a generator).  We need to convert it to a sequence for our own
        # purposes, but then also put a copy back into the scope in case we had
        # just consumed it.
        self._request_headers = list(self._scope["headers"])
        self._scope["headers"] = self._request_headers[:]

    async def receive(self) -> asgi_types.ASGIReceiveEvent:
        """Receive an ASGI event."""
        event = await self._underlying_receive()
        if event["type"] == "http.request":
            self._request_chunks.append(event["body"])
            if not event.get("more_body"):
                self._request_done_timestamp = _now()
                self._check_complete()
        return event

    async def send(self, event: asgi_types.ASGISendEvent) -> None:
        """Send an ASGI event."""
        if event["type"] == "http.response.start":
            # In some cases, ASGI applications have all the information
            # they want from the scope directly, and don't ever call
            # receive.  In this case, we artificially set the "request done"
            # time to the "request start" time.  This needs to be done even if
            # we aren't trying to record responses.
            if self._request_done_timestamp is None:
                self._request_done_timestamp = self._start_timestamp
                self._check_complete()
            if self._include_responses:
                # Headers is an Iterable, which may allow only a single
                # iteration (e.g. a generator).  We need to convert it to a
                # sequence for our own purposes, but then also put a copy back
                # into the event in case we had just consumed it.
                headers = list(event["headers"])
                event["headers"] = headers[:]
                self._response_start = schema.ResponseStart(
                    timestamp=_now(), status=event["status"], headers=headers
                )
                # We ignore the presence of trailers (and never record them).
        elif self._include_responses and event["type"] == "http.response.body":
            self._response_chunks.append(
                schema.ResponseChunk(timestamp=_now(), body=event["body"])
            )
            if not event.get("more_body"):
                self._response_done = True
                self._check_complete()
        await self._underlying_send(event)

    def _check_complete(self) -> None:
        """Record the transaction, if we have all requested data."""
        if self._recorded:
            # Don't record a second time.  Getting here in practice is
            # unlikely, but possible -- it would need to be a very weird ASGI
            # application that completes sending its response, and then after
            # having finished its response, decides to read the request.
            return
        if self._request_done_timestamp is None:
            # Request is not done.
            return
        if self._include_responses and not self._response_done:
            # We wanted to record the response, but the response is not done.
            return
        if (raw_path := self._scope.get("raw_path")) is not None:
            path = raw_path.decode("utf-8")
        else:
            path = urllib.parse.quote(self._scope["path"])
        # It's not possible to distinguish between having _no_ query string
        # (question mark not present at all) and having an _empty_ query string
        # (question mark followed by nothing). We assume the former is more
        # likely.
        if query_string := self._scope.get("query_string"):
            path += "?" + query_string.decode("utf-8")
        request = schema.Request(
            start_timestamp=self._start_timestamp,
            end_timestamp=self._request_done_timestamp,
            method=self._scope["method"],
            path=path,
            headers=self._request_headers,
            body=b"".join(self._request_chunks),
        )
        if self._response_start is not None:
            response = schema.Response(
                start=self._response_start, chunks=self._response_chunks
            )
        else:
            response = None
        transaction = schema.Transaction(request=request, response=response)
        self._recorder.record(transaction)
        self._recorded = True
