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
"""Config-driven construction of a hybrid Cascade deployment context.

The "context" a worker deploys into is a
:class:`~max.experimental.cascade.core.Runtime`. :class:`ContextConfig` builds a
device-routed :class:`~max.experimental.cascade.routing.HybridRuntime`:
``cpu``-hinted workers deploy into a round-robin pool of ``cpu`` worker
runtimes and ``gpu``-hinted workers into a pool of ``gpu`` worker runtimes. Each
pool combines locally-launched worker subprocesses (``local_*_workers``) with
connections to already-running remote workers (``remote_*_workers``), over the
selected ``transport``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal

from max.experimental.cascade.core import Runtime
from max.experimental.cascade.deployment.routing import (
    HybridRuntime,
    RoundRobinPool,
)
from max.experimental.cascade.grpc_runtime import (
    GrpcRuntimeClient,
    SubprocGrpcRuntimeClient,
)
from max.experimental.cascade.http_runtime import (
    HttpRuntimeProxy,
    SubprocHttpRuntime,
)
from pydantic import BaseModel, Field

Transport = Literal["http", "grpc"]


class ContextFactory:
    """Build local and remote worker runtimes for one transport family."""

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Build ``count`` locally-launched worker-subprocess runtimes."""
        raise NotImplementedError

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Build client runtimes for already-running remote workers."""
        raise NotImplementedError


class HttpContextFactory(ContextFactory):
    """Build pickle-over-HTTP worker runtimes."""

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Launch ``count`` HTTP worker subprocesses."""
        return [SubprocHttpRuntime() for _ in range(count)]

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Connect to remote HTTP workers by ``http://`` / ``unix://`` URL."""
        runtimes: list[Runtime] = []
        for endpoint in endpoints:
            address = endpoint.strip()
            if not address:
                raise ValueError("Worker endpoint cannot be empty")
            runtimes.append(HttpRuntimeProxy(address))
        return runtimes


class GrpcContextFactory(ContextFactory):
    """Build gRPC worker runtimes."""

    def local_runtimes(self, count: int) -> list[Runtime]:
        """Launch ``count`` gRPC worker subprocesses."""
        return [SubprocGrpcRuntimeClient() for _ in range(count)]

    def remote_runtimes(self, endpoints: Sequence[str]) -> list[Runtime]:
        """Connect to remote gRPC workers by target (``grpc://`` optional)."""
        runtimes: list[Runtime] = []
        for endpoint in endpoints:
            target = endpoint.strip()
            if not target:
                raise ValueError("Worker endpoint cannot be empty")
            runtimes.append(GrpcRuntimeClient(target.removeprefix("grpc://")))
        return runtimes


CONTEXT_FACTORIES: dict[Transport, ContextFactory] = {
    "http": HttpContextFactory(),
    "grpc": GrpcContextFactory(),
}


class ContextConfig(BaseModel):
    """Configuration for building a hybrid Cascade deployment context."""

    transport: Transport = "http"
    local_cpu_workers: int = 2
    local_gpu_workers: int = 0
    remote_cpu_workers: tuple[str, ...] = Field(default_factory=tuple)
    remote_gpu_workers: tuple[str, ...] = Field(default_factory=tuple)

    def build_context(self) -> Runtime:
        """Build the configured device-routed hybrid runtime.

        Raises:
            ValueError: If worker counts are negative or no worker is
                configured for any device class.
        """
        if self.local_cpu_workers < 0 or self.local_gpu_workers < 0:
            raise ValueError("Worker counts must be >= 0")

        factory = CONTEXT_FACTORIES[self.transport]
        cpu_runtimes = factory.local_runtimes(
            self.local_cpu_workers
        ) + factory.remote_runtimes(self.remote_cpu_workers)
        gpu_runtimes = factory.local_runtimes(
            self.local_gpu_workers
        ) + factory.remote_runtimes(self.remote_gpu_workers)

        pools: dict[str, Runtime] = {}
        if cpu_runtimes:
            pools["cpu"] = RoundRobinPool(cpu_runtimes)
        if gpu_runtimes:
            pools["gpu"] = RoundRobinPool(gpu_runtimes)
        if not pools:
            raise ValueError(
                "ContextConfig requires at least one local or remote worker"
            )
        return HybridRuntime(pools)

    @asynccontextmanager
    async def open_context(self) -> AsyncIterator[Runtime]:
        """Open the configured hybrid runtime, owning its lifecycle."""
        async with self.build_context() as runtime:
            yield runtime
