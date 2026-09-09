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
"""Cascade HTTP transport: client-side :py:class:`Runtime` proxy.

Pickle-over-HTTP analogue of the gRPC transport. Each
:py:class:`Runtime` wire primitive maps to one HTTP endpoint.

:py:class:`HttpRuntimeProxy` is a :py:class:`Runtime` backed by a remote
server at a given ``address``. Picklable: ``__getstate__`` returns only
the address, so :py:class:`Result` / :py:class:`ResultIter` handles
carrying an :py:class:`HttpRuntimeProxy` round-trip cleanly across the
wire.

Session model
-------------

The proxy owns a single :py:class:`aiohttp.ClientSession` opened on
``__aenter__``, shared across every RPC the proxy initiates
(``deploy_worker``, ``call_method``, ``get_result``, ``stream_result``,
``get_metrics``). One TCP / unix connection, pooled.
"""

from __future__ import annotations

import asyncio
import pickle
import struct
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import aiohttp
import numpy as np
from max.experimental.cascade.core import Runtime, Worker


async def _read_framed(
    stream: aiohttp.StreamReader,
) -> AsyncIterator[list[bytes]]:
    """Yield batches of 4-byte length-prefixed envelopes off ``stream``.

    Each batch is the set of complete frames already buffered together when the
    consumer next looked. ``StreamReader.readany`` returns *all* currently
    buffered bytes (blocking only when the buffer is empty), so a single wake
    scoops the whole backlog: while a synchronous detokenize call holds the
    event loop, inbound frames pile up in the socket buffer, and the next drain
    returns them in one go. Under light load a batch is a single frame, so the
    behaviour matches one-at-a-time reading -- the back-pressure is natural.
    """
    buf = bytearray()

    while True:
        # Parse against a sliding offset and compact once per drain. Dropping
        # each frame from the front as it is parsed would memmove the whole
        # remaining backlog per frame, which is quadratic in the batch size --
        # and large batches are exactly what this function exists to serve.
        frames: list[bytes] = []
        pos = 0
        while len(buf) - pos >= 4:
            (length,) = struct.unpack_from(">I", buf, pos)
            if len(buf) - pos < 4 + length:
                break
            frames.append(bytes(buf[pos + 4 : pos + 4 + length]))
            pos += 4 + length
        del buf[:pos]

        if frames:
            yield frames
            continue
        chunk = await stream.readany()
        if not chunk:  # EOF
            if buf:
                # Whatever is left is one truncated frame. Its header, if it
                # arrived at all, says how long the frame should have been.
                expected = (
                    4 + struct.unpack_from(">I", buf, 0)[0]
                    if len(buf) >= 4
                    else 4
                )
                raise asyncio.IncompleteReadError(bytes(buf), expected)
            return
        buf += chunk


def _coalesce_token_arrays(values: list[object]) -> list[object]:
    """Merge a run of 1-D ``int32`` arrays (the token stream) into one array.

    Frames that the transport drained together are handed to the detokenizer as
    a single concatenated array, so it amortizes one HuggingFace ``decode`` over
    many tokens instead of one per token. The gate is deliberately narrow: the
    token stream is the only 1-D ``int32`` ndarray stream, so streamed image
    frames (multi-dim ``uint8``) and every non-array value (text, latent dicts)
    pass through unchanged, one item per frame, and are never corrupted by
    concatenation.
    """
    out: list[object] = []
    run: list[np.ndarray] = []

    def _flush() -> None:
        if len(run) > 1:
            out.append(np.concatenate(run))
        elif run:
            out.append(run[0])
        run.clear()

    for value in values:
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 1
            and value.dtype == np.int32
        ):
            run.append(value)
        else:
            _flush()
            out.append(value)
    _flush()
    return out


# aiohttp's default connection-pool limit is 100. Streaming ``call_method``
# calls hold a connection open for the whole generation, so under concurrent
# serving that cap deadlocks dispatch well before the model saturates (e.g. 600
# in-flight requests starve on 100 connections). ``limit=0`` removes the cap;
# the model worker's own scheduler bounds real concurrency downstream.
_CONNECTION_LIMIT = 0


def _connector_for(address: str) -> aiohttp.BaseConnector:
    """Build the right aiohttp connector for an ``http://`` or ``unix://`` URL."""
    parsed = urlparse(address)
    if parsed.scheme == "http":
        return aiohttp.TCPConnector(limit=_CONNECTION_LIMIT)
    if parsed.scheme == "unix":
        if not parsed.path:
            raise ValueError(f"unix:// address requires a path: {address!r}")
        return aiohttp.UnixConnector(path=parsed.path, limit=_CONNECTION_LIMIT)
    raise ValueError(f"Unsupported address scheme: {address!r}")


def _base_url_for(address: str) -> str:
    """Return the URL base aiohttp uses to form per-request URLs."""
    parsed = urlparse(address)
    if parsed.scheme == "http":
        return address.rstrip("/")
    if parsed.scheme == "unix":
        # aiohttp's UnixConnector dispatches by socket path; the host
        # portion of the URL is only used to satisfy URL parsing.
        return "http://unix"
    raise ValueError(f"Unsupported address scheme: {address!r}")


class HttpRuntimeProxy(Runtime):
    """Client-side :py:class:`Runtime` backed by an HTTP Cascade server.

    ``address`` is either ``http://host:port`` or ``unix:///path/to.sock``.
    """

    def __init__(self, address: str) -> None:
        super().__init__()
        self.address = address
        self._base_url = _base_url_for(address)
        self._session: aiohttp.ClientSession | None = None

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> HttpRuntimeProxy:
        await super().__aenter__()
        self._session = await self.enter_async_context(
            aiohttp.ClientSession(
                connector=_connector_for(self.address),
                connector_owner=True,
            )
        )
        return self

    @asynccontextmanager
    async def session(self) -> AsyncIterator[aiohttp.ClientSession]:
        """Yield a connected :py:class:`aiohttp.ClientSession`.

        Inside an ``async with`` block on this proxy the pooled session
        opened by :py:meth:`__aenter__` is reused. Outside that context
        -- e.g. after :py:class:`HttpRuntimeProxy` has been unpickled
        from a wire-side handle and is used directly without being
        explicitly entered -- this yields a fresh single-use session and
        tears it down on exit. Concurrent users still see a single
        in-flight HTTP request per call.
        """
        if self._session is not None:
            yield self._session
            return
        async with aiohttp.ClientSession(
            connector=_connector_for(self.address),
            connector_owner=True,
        ) as session:
            yield session

    # -- pickle ----------------------------------------------------------

    def __getstate__(self) -> object:
        # Only the address travels; the live session is recreated on the
        # other side of the wire when the proxy is re-entered.
        return {"address": self.address}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["address"])  # type: ignore[misc]

    # -- Runtime wire primitives -----------------------------------------

    async def deploy_worker(self, worker: Worker) -> str:
        """Pickle the worker and register it on the server."""
        # Workers don't carry :py:class:`Result` handles in their state,
        # so plain pickle is fine here.
        async with (
            self.session() as session,
            session.put(
                f"{self._base_url}/worker",
                data=pickle.dumps(worker),
                headers={"Content-Type": "application/pickle"},
            ) as response,
        ):
            response.raise_for_status()
            return (await response.read()).decode()

    @asynccontextmanager
    async def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AsyncIterator[str]:
        """Open a streaming POST that holds the call alive for the scope.

        The server emits the bound ``result_id`` as the first line and
        then holds the response open until the client disconnects. Exit
        of this context manager closes the response, which is what
        signals the server to cancel the in-flight task and release the
        result buffer.
        """
        async with (
            self.session() as session,
            session.post(
                f"{self._base_url}/worker/{worker_id}/{func}",
                data=pickle.dumps((args, kwargs)),
                headers={"Content-Type": "application/pickle"},
            ) as response,
        ):
            response.raise_for_status()
            result_id_line = await response.content.readline()
            yield result_id_line.decode().rstrip("\n")
            # exiting this context deletes the remote result

    async def get_result(self, result_id: str) -> object:
        """Fetch a single result via the proxy's session.

        The HTTP transport pickles values, so they arrive as the native
        Python object with no JSON decoding.
        """
        async with (
            self.session() as session,
            session.get(f"{self._base_url}/result/{result_id}") as response,
        ):
            response.raise_for_status()
            payload = await response.read()
        ok, value = pickle.loads(payload)
        if ok:
            return value
        raise value

    async def stream_result(self, result_id: str) -> AsyncIterator[object]:
        """Stream a result via the proxy's session.

        Pickled items arrive as native objects; no JSON decoding.
        """
        async with (
            self.session() as session,
            session.get(
                f"{self._base_url}/result/{result_id}/stream"
            ) as response,
        ):
            response.raise_for_status()
            async for batch in _read_framed(response.content):
                values: list[object] = []
                for payload in batch:
                    ok, value = pickle.loads(payload)
                    if not ok:
                        # Surface any good items drained ahead of the error
                        # frame, then raise.
                        for item in _coalesce_token_arrays(values):
                            yield item
                        raise value
                    values.append(value)
                for item in _coalesce_token_arrays(values):
                    yield item

    async def get_metrics(self) -> str:
        """Fetch Prometheus exposition text from the server."""
        async with (
            self.session() as session,
            session.get(f"{self._base_url}/metrics") as response,
        ):
            response.raise_for_status()
            return await response.text()
