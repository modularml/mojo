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
"""Composite runtimes that route deployments across child runtimes.

:class:`RoundRobinPool` deploys a worker to every runtime in a pool and hands
back a proxy that spreads calls across them round-robin. :class:`HybridRuntime`
routes each worker to a named child runtime (a pool) by matching the worker's
``deploy_hints`` -- e.g. a ``"cpu"``-hinted worker deploys into the ``"cpu"``
pool, a ``"gpu"``-hinted worker into the ``"gpu"`` pool.

Both are :class:`~max.experimental.cascade.core.Runtime` subclasses so they
satisfy ``pipeline.deploy(runtime)``. They are pure routers: ``deploy`` returns
a proxy bound to the leaf (child) runtimes, so the per-call wire primitives run
on the leaves and are never invoked on the router itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from max.experimental.cascade.core import Runtime, Worker, WorkerType


class RoundRobinProxy:
    """Spread attribute access across delegates in round-robin order.

    Each public attribute access advances to the next delegate, so calling a
    worker method through the proxy dispatches to a different underlying
    deployment each time.
    """

    def __init__(self, delegates: Sequence[object]) -> None:
        if not delegates:
            raise ValueError("RoundRobinProxy requires at least one delegate")
        self._delegates = list(delegates)
        self._index = 0

    def _next_delegate(self) -> object:
        delegate = self._delegates[self._index]
        self._index = (self._index + 1) % len(self._delegates)
        return delegate

    # ``Any`` is deliberate: this is a dynamic dispatch proxy, so attribute
    # access forwards to a delegate's arbitrary (worker) method surface.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._next_delegate(), name)


class _RoutingRuntime(Runtime):
    """Base for runtimes that dispatch deployments to child runtimes.

    ``deploy`` returns a proxy bound to the child (leaf) runtimes, so the
    per-call wire primitives (:meth:`call_method`, :meth:`get_result`,
    :meth:`stream_result`) run on the leaves and are never invoked on the
    router. They raise if called to surface any accidental misuse loudly.
    """

    async def deploy_worker(self, worker: Worker) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} deploys via deploy(), not deploy_worker()"
        )

    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AbstractAsyncContextManager[str]:
        raise NotImplementedError(
            f"{type(self).__name__} routes calls through per-leaf proxies"
        )

    def get_result(self, resid: str) -> Awaitable[object]:
        raise NotImplementedError(
            f"{type(self).__name__} routes calls through per-leaf proxies"
        )

    def stream_result(self, resid: str) -> AsyncIterator[object]:
        raise NotImplementedError(
            f"{type(self).__name__} routes calls through per-leaf proxies"
        )


class RoundRobinPool(_RoutingRuntime):
    """Deploy a worker to every child runtime; dispatch calls round-robin."""

    def __init__(self, runtimes: Sequence[Runtime]) -> None:
        super().__init__()
        if not runtimes:
            raise ValueError("RoundRobinPool requires at least one runtime")
        self._runtimes = list(runtimes)

    async def __aenter__(self) -> RoundRobinPool:
        """Open every child runtime; they close when this pool closes."""
        await super().__aenter__()
        self._runtimes = [
            await self.enter_async_context(runtime)
            for runtime in self._runtimes
        ]
        return self

    async def deploy(self, worker: WorkerType) -> WorkerType:
        """Deploy to all child runtimes and return a round-robin proxy."""
        proxies = await asyncio.gather(
            *(runtime.deploy(worker) for runtime in self._runtimes)
        )
        return cast(WorkerType, RoundRobinProxy(list(proxies)))

    async def get_metrics(self) -> str:
        """Aggregate Prometheus text across all child runtimes."""
        parts = await asyncio.gather(
            *(runtime.get_metrics() for runtime in self._runtimes)
        )
        return "\n".join(part for part in parts if part)


class HybridRuntime(_RoutingRuntime):
    """Route each worker to a named child runtime by its ``deploy_hints``."""

    def __init__(self, runtimes: Mapping[str, Runtime]) -> None:
        super().__init__()
        if not runtimes:
            raise ValueError(
                "HybridRuntime requires at least one child runtime"
            )
        self._runtimes = dict(runtimes)

    async def __aenter__(self) -> HybridRuntime:
        """Open every child runtime; they close when this router closes."""
        await super().__aenter__()
        self._runtimes = {
            name: await self.enter_async_context(runtime)
            for name, runtime in self._runtimes.items()
        }
        return self

    def _select(self, worker: Worker) -> str:
        """Pick the child runtime whose name is in the worker's hints."""
        for name in self._runtimes:
            if name in worker.deploy_hints:
                return name
        known = ", ".join(sorted(self._runtimes))
        raise ValueError(
            f"No worker pool for {type(worker).__name__} with deploy_hints "
            f"{worker.deploy_hints!r}. Configured pools: {known}"
        )

    async def deploy(self, worker: WorkerType) -> WorkerType:
        """Deploy the worker to the child runtime its hints select."""
        return await self._runtimes[self._select(worker)].deploy(worker)

    async def get_metrics(self) -> str:
        """Aggregate Prometheus text across all child runtimes."""
        parts = await asyncio.gather(
            *(runtime.get_metrics() for runtime in self._runtimes.values())
        )
        return "\n".join(part for part in parts if part)
