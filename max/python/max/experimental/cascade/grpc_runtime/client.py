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
"""Cascade gRPC transport: client-side :py:class:`Runtime` proxy.

:py:class:`GrpcRuntimeClient` is a :py:class:`Runtime` backed by a remote
:py:class:`CascadeRuntimeServicer` at a given ``target``. Picklable:
``__getstate__`` returns only the target address, so :py:class:`Result` /
:py:class:`ResultIter` handles carrying a :py:class:`GrpcRuntimeClient`
round-trip cleanly across the wire.

Connection model
----------------

``_own_channel`` is the long-lived gRPC channel from this client to *its own*
server, opened on ``__aenter__`` and closed on ``__aexit__``.

``_get_own_stub()`` is used for every RPC to this client's own server. When
``_own_channel`` is open it reuses the pooled stub directly. When the client was
deserialized from a :py:class:`Result` URL (no context entered), it opens a
fresh per-call channel to ``self.target``, waits for the server, makes the RPC,
and closes.

Resolving a forwarded :py:class:`Result` is just reconstructing a client: a
:py:class:`Result` is a ``(target, result_id)`` pair, so the codec hands the
``target`` to :py:func:`dial`, which constructs an unentered
:py:class:`GrpcRuntimeClient` whose channel opens lazily on first use via the
same ``_get_own_stub()`` per-call path -- no registry, no shared lifecycle.
Both the client (decoding forwarded return values) and the server (decoding
:py:class:`Result` arguments) resolve references through this one function.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

# gRPC core's noisy default logging is suppressed before importing grpc.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import grpc
from max.experimental.cascade.core import Runtime, Worker
from max.experimental.cascade.core.type_hints import CascadeValue

from . import cascade_runtime_v1_pb2 as pb
from . import cascade_runtime_v1_pb2_grpc as rpc
from .codec import (
    decode_error,
    decode_value_slot,
    encode_args,
    encode_kwargs,
    is_remote_ref,
)

logger = logging.getLogger(__name__)


async def _wait_until_alive(stub: rpc.CascadeRuntimeServiceStub) -> None:
    """Poll ``GetAlive`` until the server responds successfully."""
    # TODO(njain) do we want to set up a timeout here?
    while True:
        try:
            response = await stub.GetAlive(pb.Empty())
            if response.alive:
                return
        except grpc.RpcError:
            pass
        await asyncio.sleep(0.5)


class GrpcRuntimeClient(Runtime):
    """Client-side :py:class:`Runtime` that talks to a Cascade gRPC server."""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        # Long-lived channel and stub to the server associated with this client.
        self._own_channel: grpc.aio.Channel | None = None
        self._stub: rpc.CascadeRuntimeServiceStub | None = None

    # -- Lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> GrpcRuntimeClient:
        """Open a long-lived channel to this client's own server and wait for it to come up."""
        await super().__aenter__()
        self._own_channel = grpc.aio.insecure_channel(self.target)
        self._stub = rpc.CascadeRuntimeServiceStub(self._own_channel)
        await _wait_until_alive(self._stub)
        return self

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        """Close the channel to this client's own server."""
        self._stub = None
        if self._own_channel is not None:
            await self._own_channel.close()
            self._own_channel = None
        return await super().__aexit__(*exc_info)

    # -- Pickle -------------------------------------------------------------

    def __getstate__(self) -> object:
        # Only the target URL travels on the wire. The receiver reconstructs a
        # lazy (unentered) client and dials on demand.
        return {"target": self.target}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["target"])  # type: ignore[misc]

    # -- Stub accessor ------------------------------------------------------

    @asynccontextmanager
    async def _get_own_stub(
        self,
    ) -> AsyncIterator[rpc.CascadeRuntimeServiceStub]:
        """Yield a stub for one RPC to this client's own server.

        Analogous to ``session()`` in the HTTP runtime:

        * **Pooled** (``__aenter__`` was called): reuse the long-lived stub on
          ``_own_channel``. Zero extra handshakes.
        * **Lazy / per-call** (client was deserialized from a :py:class:`Result`
          URL and has no open channel): open a fresh channel to ``self.target``,
          wait for the server, yield the stub, then close.
        """
        if self._stub is not None:
            yield self._stub
            return
        channel = grpc.aio.insecure_channel(self.target)
        try:
            stub = rpc.CascadeRuntimeServiceStub(channel)
            await _wait_until_alive(stub)
            yield stub
        finally:
            await channel.close()

    # -- Runtime interface methods ------------------------------------------

    async def deploy_worker(self, worker: Worker) -> str:
        """Pickle worker, deploy it, register it, and return its stable id."""
        async with self._get_own_stub() as stub:
            response = await stub.DeployWorker(
                pb.DeployWorkerRequest(
                    format=pb.DeployWorkerRequest.WORKER_FORMAT_PICKLE,
                    worker_blob=pickle.dumps(worker),
                )
            )
        return response.worker_id

    @asynccontextmanager
    async def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AsyncIterator[str]:
        """Open a server-streaming CallMethod RPC and yield the result_id.

        Reads the server-generated result_id from the first response, then
        holds the stream open for the duration of the caller's ``async with``
        block. Cancelling the stream on exit signals the server to release the
        result buffer (exits its ``local.call_method`` context). The dialed
        channel stays open for the full scope of this context manager.
        """
        async with self._get_own_stub() as stub:
            call = stub.CallMethod(
                pb.CallMethodRequest(
                    worker_id=worker_id,
                    method=func,
                    args=encode_args(args),
                    kwargs=encode_kwargs(kwargs),
                )
            )
            try:
                response = await call.read()
                yield response.result_id
            finally:
                call.cancel()

    async def get_result(self, result_id: str) -> CascadeValue:
        """Fetch and decode a single result from the remote server by id.

        Returns raw JSON-like data (:py:data:`CascadeValue`); the owning
        :py:class:`Result` applies any expected type on top.

        Args:
            result_id: The opaque result identifier returned by
                :py:meth:`call_method`.

        Returns:
            The decoded Python value.

        Raises:
            Exception: If the server returns an error envelope, the encoded
                exception is re-raised on the client.
            RuntimeError: If the server returns an unrecognised outcome kind.
        """
        async with self._get_own_stub() as stub:
            response = await stub.GetResult(
                pb.GetResultRequest(result_id=result_id),
            )
        kind = response.WhichOneof("outcome")
        if kind == "value":
            # We should not receive a ResultRef from the server, since the
            # server should have resolved it itself! Also we don't want to
            # have nested ResultRefs.
            if is_remote_ref(response.value):
                kind = response.value.WhichOneof("kind")
                raise RuntimeError(
                    f"Unexpected {kind!r} ResultReference in get_result"
                )
            return await decode_value_slot(response.value, dial)
        if kind == "error":
            raise decode_error(response.error)
        raise RuntimeError(f"Unexpected GetResult outcome {kind!r}")

    async def stream_result(
        self, result_id: str
    ) -> AsyncIterator[CascadeValue]:
        """Stream items from a bound server-side iterator, decoded one by one.

        Yields raw JSON-like data (:py:data:`CascadeValue`); the owning
        :py:class:`ResultIter` applies any expected element type on top.

        Args:
            result_id: The opaque result identifier for the stream.

        Yields:
            Decoded Python values as they arrive from the server.

        Raises:
            Exception: If the server sends an error frame, the encoded
                exception is re-raised on the client.
            RuntimeError: If the server returns an unrecognised outcome kind.
        """
        async with self._get_own_stub() as stub:
            async for response in stub.StreamResult(
                pb.StreamResultRequest(result_id=result_id)
            ):
                kind = response.WhichOneof("outcome")
                if kind == "value":
                    yield await decode_value_slot(response.value, dial)
                elif kind == "error":
                    raise decode_error(response.error)
                else:
                    raise RuntimeError(
                        f"Unexpected StreamResult outcome {kind!r}"
                    )

    async def get_metrics(self) -> str:
        """Not implemented."""
        raise NotImplementedError


def dial(target: str) -> GrpcRuntimeClient:
    """Resolve a forwarded result reference's ``target`` to a lazily created client.

    We construct a :py:class:`GrpcRuntimeClient` on the dial operation. Its channel
    to ``target`` opens lazily on the first RPC and closes after via `_get_own_stub()`.

    The codec calls this for every ``result_ref`` / ``stream_ref`` it decodes,
    whether that happens on the client (if the worker returns a remote `Result` on another server)
    or the server (if it is passed remote ``Result`` arguments).
    """
    if not target:
        raise ValueError("dial requires a non-empty target")
    return GrpcRuntimeClient(target)
