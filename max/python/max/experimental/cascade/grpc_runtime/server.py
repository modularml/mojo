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
"""Cascade gRPC server: a thin RPC translation layer over a ``LocalRuntime``.

The server hosts an opened :py:class:`LocalRuntime` for the lifetime of the
process. Each RPC translates to a single ``LocalRuntime`` primitive call.

Wire payloads (args, kwargs, return values) round-trip through the
language-portable :py:mod:`codec` ValueSlot envelope; the only pickle on
the wire is the worker blob in ``DeployWorker``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pickle
import signal
from collections.abc import AsyncIterator

import cyclopts
import grpc
from max.experimental.cascade.core import Worker
from max.experimental.cascade.core.local_runtime import LocalRuntime
from prometheus_client import generate_latest

from . import cascade_runtime_v1_pb2 as pb
from . import cascade_runtime_v1_pb2_grpc as rpc
from .client import dial
from .codec import (
    decode_args,
    decode_kwargs,
    encode_error,
    encode_value_slot,
)

logger = logging.getLogger(__name__)


class CascadeRuntimeServicer(rpc.CascadeRuntimeServiceServicer):
    """gRPC servicer that forwards each RPC to its local runtime."""

    def __init__(self, local: LocalRuntime) -> None:
        self._local = local

    async def GetAlive(
        self,
        request: pb.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb.AliveResponse:
        """Liveness check: always reports alive once the server is serving."""
        return pb.AliveResponse(alive=True)

    async def GetMetrics(
        self,
        request: pb.Empty,
        context: grpc.aio.ServicerContext,
    ) -> pb.MetricsResponse:
        """Return Prometheus exposition text for this server."""
        return pb.MetricsResponse(prometheus=generate_latest().decode("utf-8"))

    async def DeployWorker(
        self,
        request: pb.DeployWorkerRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.DeployWorkerResponse:
        """Unpickle a Worker and register it with the local runtime.

        Warning:
            ``DeployWorker`` deserializes a pickle blob from the client, which
            is a remote code execution vector if the server is reachable by an
            untrusted caller. Bind only to a unix-domain socket or a localhost
            port protected by a firewall / network policy, and never expose
            this service on a public interface.
        """
        if request.format != pb.DeployWorkerRequest.WORKER_FORMAT_PICKLE:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Expected pickle but received unsupported worker format: {request.format}",
            )
        worker = pickle.loads(request.worker_blob)
        if not isinstance(worker, Worker):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Expected a Worker object but received: {type(worker).__name__}",
            )
        worker_id = await self._local.deploy_worker(worker)
        return pb.DeployWorkerResponse(worker_id=worker_id)

    async def CallMethod(
        self,
        request: pb.CallMethodRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.CallMethodResponse]:
        """Decode args/kwargs, dispatch the call, and stream back the result_id.

        Keeping the stream open holds the LocalRuntime ``call_method`` context
        alive, which owns the result buffer. When the client cancels the RPC
        (its ``async with grpc_runtime.call_method(...)`` block exits), the
        server receives CancelledError, exits the context, and the buffer is
        released.
        """
        # If an arg is a forwarded Result, decoding dials the peer that owns it
        # (possibly a different server than this one); the value is fetched lazily
        # when the worker awaits the handle.
        arg_types = await self._local.get_func_args_type(
            request.worker_id, request.method
        )
        args = await decode_args(list(request.args), dial, arg_types)
        kwargs = await decode_kwargs(dict(request.kwargs), dial, arg_types)
        async with self._local.call_method(
            request.worker_id,
            request.method,
            args,
            kwargs,
        ) as result_id:
            yield pb.CallMethodResponse(result_id=result_id)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    async def GetResult(
        self,
        request: pb.GetResultRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.GetResultResponse:
        """Fetch a single result by id, encoded as a value or error envelope."""
        try:
            value = await self._local.get_result(request.result_id)
            return pb.GetResultResponse(value=encode_value_slot(value))
        except Exception as exc:
            return pb.GetResultResponse(error=encode_error(exc))

    async def StreamResult(
        self,
        request: pb.StreamResultRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.StreamResultResponse]:
        """Inline server-streaming consumption of a bound stream."""
        try:
            async for item in self._local.stream_result(request.result_id):
                yield pb.StreamResultResponse(
                    value=encode_value_slot(item),
                )
        except Exception as exc:
            yield pb.StreamResultResponse(error=encode_error(exc))


# ---------------------------------------------------------------------------
# Process entry point.
# ---------------------------------------------------------------------------


async def serve_async(bind_addr: str) -> None:
    """Run the Cascade gRPC server bound to ``bind_addr`` until shutdown."""
    shutdown_event = asyncio.Event()
    async with LocalRuntime() as local:
        server = grpc.aio.server()
        rpc.add_CascadeRuntimeServiceServicer_to_server(
            CascadeRuntimeServicer(local), server
        )
        bound_port = server.add_insecure_port(_normalize_addr(bind_addr))
        if not bound_port and not bind_addr.startswith("unix:"):
            raise RuntimeError(f"Failed to bind to address {bind_addr!r}")
        await server.start()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, shutdown_event.set)

        try:
            await shutdown_event.wait()
        finally:
            await server.stop(grace=0)


def serve(bind_addr: str) -> None:
    """Synchronous wrapper invoked by :py:func:`subprocess_manager`."""
    asyncio.run(serve_async(bind_addr))


def _normalize_addr(addr: str) -> str:
    """Normalize a ``unix:``/``host:port`` address for ``add_insecure_port``."""
    addr = addr.strip()
    if not addr:
        raise ValueError("bind address cannot be empty")
    if addr.startswith(("unix:", "unix://")):
        return addr
    if "://" in addr:
        raise ValueError(f"Unsupported gRPC scheme in {addr!r}")
    return addr


cli = cyclopts.App()


@cli.default
def main(bind_addr: str = "127.0.0.1:50051") -> None:
    """Run the Cascade gRPC runtime server bound to ``bind_addr``."""
    serve(bind_addr)
