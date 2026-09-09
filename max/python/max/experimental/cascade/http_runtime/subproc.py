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
"""Subprocess-managed local Cascade HTTP server.

:py:class:`SubprocHttpRuntime` boots a Cascade HTTP server in a child
process bound to a unix domain socket and exposes it as a
:py:class:`Runtime`. Useful for tests and single-host deployments.

This lives in its own module so that the heavier server-side deps
(uvicorn, FastAPI, process-control) are only imported when
:py:class:`SubprocHttpRuntime` is actually used, keeping the bare
client import path (:py:mod:`~.client`) lightweight.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import aiohttp
from max.experimental.cascade.core import Runtime, Worker
from max.experimental.cascade.http_runtime import server as http_server
from max.experimental.cascade.http_runtime.client import HttpRuntimeProxy
from max.serve.process_control import subprocess_manager


class SubprocHttpRuntime(Runtime):
    """Lifecycle-owning local HTTP server + opened :py:class:`HttpRuntimeProxy`.

    Entering the ``async with`` block boots a Cascade HTTP worker process
    bound to a unique unix domain socket and yields a proxy connected to
    it. Exiting closes the proxy and cleans up the socket file.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sock_path = f"/tmp/max-{uuid.uuid4().hex[:12]}.sock"
        self._address = f"unix://{self._sock_path}"
        self._delegate: HttpRuntimeProxy | None = None

    def __reduce__(self) -> tuple[type[HttpRuntimeProxy], tuple[str]]:
        return (HttpRuntimeProxy, (self._address,))

    async def __aenter__(self) -> SubprocHttpRuntime:
        await super().__aenter__()
        # daemon=False: the worker hosted here may spin up its own
        # subprocesses (e.g. a MAXModelWorker launches a max.serve model
        # worker), which daemonic processes are not allowed to do.
        proc = await self.enter_async_context(
            subprocess_manager("Cascade HTTP Runtime", daemon=False)
        )
        proc.start(http_server.serve, self._address)
        client = HttpRuntimeProxy(self._address)
        self._delegate = await self.enter_async_context(client)
        self.callback(_unlink_quiet, self._sock_path)
        async with self._delegate.session() as session:
            await _wait_until_alive(session, client._base_url)
        return self

    # -- Runtime forwards ------------------------------------------------

    async def deploy_worker(self, worker: Worker) -> str:
        """Forward to the inner proxy's :py:meth:`HttpRuntimeProxy.deploy_worker`."""
        return await self._proxy().deploy_worker(worker)

    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> Any:
        """Forward to the inner proxy's :py:meth:`HttpRuntimeProxy.call_method`."""
        return self._proxy().call_method(worker_id, func, args, kwargs)

    async def get_result(self, result_id: str) -> object:
        """Forward to the inner proxy's :py:meth:`HttpRuntimeProxy.get_result`."""
        return await self._proxy().get_result(result_id)

    def stream_result(self, result_id: str) -> AsyncIterator[object]:
        """Forward to the inner proxy's :py:meth:`HttpRuntimeProxy.stream_result`."""
        return self._proxy().stream_result(result_id)

    async def get_metrics(self) -> str:
        """Forward to the inner proxy's :py:meth:`HttpRuntimeProxy.get_metrics`."""
        return await self._proxy().get_metrics()

    def _proxy(self) -> HttpRuntimeProxy:
        """Inner proxy; raises if used outside of the ``async with`` block."""
        if self._delegate is None:
            raise RuntimeError(
                "SubprocHttpRuntime used outside of its async-with context"
            )
        return self._delegate


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


async def _wait_until_alive(
    session: aiohttp.ClientSession,
    base_url: str,
    timeout: float = 30.0,
) -> None:
    """Poll ``GET /alive`` until the server responds successfully."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            async with session.get(f"{base_url}/alive") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(
                f"Server at {base_url!r} did not become alive within {timeout}s"
            )
        await asyncio.sleep(0.1)
