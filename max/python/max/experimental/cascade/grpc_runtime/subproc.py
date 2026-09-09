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
"""Subprocess-owned gRPC server wrapper.

:py:class:`SubprocGrpcRuntimeClient` boots a Cascade gRPC server in a child
process and exposes it as a :py:class:`Runtime`. Useful for tests and
single-host deployments.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from max.experimental.cascade.core import Runtime, Worker

from .client import GrpcRuntimeClient


class SubprocGrpcRuntimeClient(Runtime):
    """Lifecycle-owning local gRPC server wrapping a :py:class:`GrpcRuntimeClient`.

    Useful for tests and single-host deployments: ``__aenter__`` boots a
    Cascade gRPC server in a child process bound to a unix domain socket
    and opens a :py:class:`GrpcRuntimeClient` connected to it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sock_path = f"/tmp/max-{uuid.uuid4().hex[:12]}.sock"
        self._bind_addr = f"unix://{self._sock_path}"
        self.target = f"unix:{self._sock_path}"
        self._inner: GrpcRuntimeClient | None = None

    async def __aenter__(self) -> SubprocGrpcRuntimeClient:
        """Boot the gRPC server subprocess and open a connected client."""
        await super().__aenter__()
        # Local import keeps the heavy server deps off the client import path.
        from max.experimental.cascade.grpc_runtime import server as grpc_server
        from max.serve.process_control import subprocess_manager

        # daemon=False: the worker hosted here may spin up its own
        # subprocesses (e.g. a MAXModelWorker launches a max.serve model
        # worker), which daemonic processes are not allowed to do.
        proc = await self.enter_async_context(
            subprocess_manager("Cascade gRPC Runtime", daemon=False)
        )
        proc.start(grpc_server.serve, self._bind_addr)
        self._inner = await self.enter_async_context(
            GrpcRuntimeClient(self.target)
        )
        return self

    async def __aexit__(self, *exc_info: Any) -> bool | None:
        """Shut down the inner runtime and remove the unix socket file."""
        self._inner = None
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        return await super().__aexit__(*exc_info)

    def _get_inner(self) -> GrpcRuntimeClient:
        if self._inner is None:
            raise RuntimeError(
                "SubprocGrpcRuntimeClient used outside of its context"
            )
        return self._inner

    async def deploy_worker(self, worker: Worker) -> str:
        """Delegate worker deployment to the inner :py:class:`GrpcRuntimeClient`."""
        return await self._get_inner().deploy_worker(worker)

    @asynccontextmanager
    async def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AsyncIterator[str]:
        """Delegate method dispatch to the inner :py:class:`GrpcRuntimeClient`."""
        async with self._get_inner().call_method(
            worker_id, func, args, kwargs
        ) as result_id:
            yield result_id

    async def get_result(self, result_id: str) -> Any:
        """Delegate result fetching to the inner :py:class:`GrpcRuntimeClient`."""
        return await self._get_inner().get_result(result_id)

    async def stream_result(self, result_id: str) -> AsyncIterator[Any]:
        """Delegate stream consumption to the inner :py:class:`GrpcRuntimeClient`."""
        async for item in self._get_inner().stream_result(result_id):
            yield item

    async def get_metrics(self) -> str:
        """Not implemented for the subprocess gRPC runtime."""
        raise NotImplementedError
