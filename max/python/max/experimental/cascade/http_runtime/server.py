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
"""Cascade HTTP server: a thin REST translation layer over a ``LocalRuntime``.

The server hosts an opened :py:class:`LocalRuntime` for the lifetime of the
process. Each route translates to a single ``LocalRuntime`` primitive call;
result ids are chosen by the client and used as-is.

Endpoints (one per :py:class:`Runtime` wire primitive):

* ``GET  /alive``                                       health check
* ``GET  /metrics``                                     Prometheus text
* ``PUT  /worker``                                      ``deploy_worker``
* ``POST /worker/{worker_id}/{method}/{result_id}``     ``call_method``
* ``GET  /result/{result_id}``                          ``get_result``
* ``GET  /result/{result_id}/stream``                   ``stream_result``

Wire payloads round-trip as pickled outcome envelopes:
``("ok", value)`` / ``(False, exc)`` / ``("end",)``. The streaming endpoint
emits length-prefixed envelopes (4-byte big-endian length + pickled blob)
in a single chunked response, mirroring the gRPC ``StreamResult``
server-streaming RPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import http
import logging
import pickle
import signal
import struct
from collections.abc import AsyncIterator
from typing import NamedTuple
from urllib.parse import urlparse

import cyclopts
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from max.experimental.cascade.core import Worker
from max.experimental.cascade.core.local_runtime import LocalRuntime
from prometheus_client import make_asgi_app

logger = logging.getLogger(__name__)


def _frame(data: object) -> bytes:
    """Pack a pickled data as a 4-byte length prefix + payload."""
    payload = pickle.dumps(data)
    return struct.pack(">I", len(payload)) + payload


def _build_app(runtime: LocalRuntime, bind_address: str) -> FastAPI:
    r"""Assemble a FastAPI app bound to ``runtime`` listening at ``bind_address``.

    ``bind_address`` is the wire address peers use to reach this server; the
    codec stamps it on outbound refs and short-circuits inbound refs whose
    address matches it directly against ``runtime`` (no HTTP self-loop).
    Refs to other peer addresses ride out via one-shot fetch closures owned
    by the codec -- no shared session cache lives here.
    """
    app = FastAPI()
    app.mount("/metrics", make_asgi_app())

    @app.get("/alive")
    async def get_alive() -> dict[str, bool]:
        return {"alive": True}

    @app.put("/worker")
    async def put_worker(request: Request) -> Response:
        body = await request.body()
        worker = pickle.loads(body)
        if not isinstance(worker, Worker):
            raise HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=(
                    f"Deployed object is not a Worker: {type(worker).__name__}"
                ),
            )
        worker_id = await runtime.deploy_worker(worker)
        return Response(
            worker_id,
            media_type="text/plain",
        )

    @app.post("/worker/{worker_id}/{method}")
    async def post_call(
        worker_id: str, method: str, request: Request
    ) -> StreamingResponse:
        req_body = await request.body()
        args, kwargs = pickle.loads(req_body)

        async def body() -> AsyncIterator[bytes]:
            async with runtime.call_method(
                worker_id, method, args, kwargs
            ) as result_id:
                # ``\n`` terminates the id so the client's ``readline()``
                # returns promptly while the response stays open for the
                # remainder of the call's lifetime.
                yield f"{result_id}\n".encode()
                # Hold the call_method context open until the client
                # disconnects. Starlette is blocked waiting for our
                # __anext__(), so it can't detect the disconnect until
                # we yield or return. Poll the ASGI receive channel
                # ourselves so we exit promptly.
                while not await request.is_disconnected():
                    await asyncio.sleep(0.1)

        return StreamingResponse(body(), media_type="text/plain")

    @app.get("/result/{result_id}")
    async def get_result(result_id: str) -> Response:
        try:
            value = await runtime.get_result(result_id)
            ok = True
        except Exception as exc:
            value = exc
            ok = False
        return Response(
            pickle.dumps((ok, value)),
            media_type="application/pickle",
        )

    @app.get("/result/{result_id}/stream")
    async def stream_result(result_id: str) -> StreamingResponse:
        """Inline server-streaming consumption of a bound stream.

        The local runtime enforces single-consumer binding; a second
        ``stream`` request on the same ``result_id`` raises and is reported
        as a structured error in the first frame.
        """

        async def body() -> AsyncIterator[bytes]:
            try:
                async for item in runtime.stream_result(result_id):
                    yield _frame((True, item))
            except Exception as exc:
                yield _frame((False, exc))

        return StreamingResponse(body(), media_type="application/pickle-stream")

    return app


# ---------------------------------------------------------------------------
# Process entry point.
# ---------------------------------------------------------------------------


async def serve_async(address: str) -> None:
    """Run the Cascade HTTP server bound to ``address`` until shutdown."""
    bind = _normalize_addr(address)
    async with LocalRuntime() as runtime:
        app = _build_app(runtime, address)
        # ``lifespan="off"`` avoids starlette logging a noisy CancelledError
        # traceback when uvicorn is shut down via signal.
        config = uvicorn.Config(
            app,
            **bind._asdict(),
            lifespan="off",
            access_log=False,
        )
        server = uvicorn.Server(config)

        # ``add_signal_handler`` callbacks run on the loop thread, so flipping
        # ``should_exit`` directly is safe; uvicorn's serve loop polls it.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(
                    sig, lambda: setattr(server, "should_exit", True)
                )

        await server.serve()


def serve(address: str) -> None:
    """Synchronous wrapper invoked by :py:func:`subprocess_manager`."""
    asyncio.run(serve_async(address))


class _BindConfig(NamedTuple):
    host: str | None = None
    port: int | None = None
    uds: str | None = None


def _normalize_addr(address: str) -> _BindConfig:
    """Normalize ``http://host:port`` or ``unix://path`` for uvicorn."""
    address = address.strip()
    if not address:
        raise ValueError("address cannot be empty")
    parsed = urlparse(address)
    if parsed.scheme == "http":
        if parsed.hostname is None or parsed.port is None:
            raise ValueError(
                f"HTTP address must be http://host:port, got {address!r}"
            )
        return _BindConfig(host=parsed.hostname, port=parsed.port)
    if parsed.scheme == "unix":
        if not parsed.path:
            raise ValueError(
                f"unix:// address must include a path, got {address!r}"
            )
        return _BindConfig(uds=parsed.path)
    raise ValueError(
        f"address must be http://host:port or unix://path; got {address!r}"
    )


cli = cyclopts.App()


@cli.default
def main(address: str = "http://127.0.0.1:50051") -> None:
    """Run the Cascade HTTP runtime server bound to ``address``."""
    serve(address)


if __name__ == "__main__":
    cli()
