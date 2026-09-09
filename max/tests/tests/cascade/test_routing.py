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
"""Tests for the composite routing runtimes.

Covers round-robin dispatch, pool deployment over multiple child runtimes, and
device-hint routing in :class:`HybridRuntime` -- using in-process
:class:`LocalRuntime` children and recording stubs so the tests stay fast and
launch no subprocesses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager

import pytest
from max.experimental.cascade import (
    LocalRuntime,
    Runtime,
    Worker,
    worker_method,
)
from max.experimental.cascade.core.pipeline_method import (
    _pipeline_method_scope,
)
from max.experimental.cascade.deployment.routing import (
    HybridRuntime,
    RoundRobinPool,
    RoundRobinProxy,
)


class _Recorder:
    """Records how many times its method is called."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    def ping(self) -> str:
        self.calls += 1
        return self.tag


def test_round_robin_proxy_cycles() -> None:
    """Successive accesses cycle through delegates in order."""
    a, b = _Recorder("a"), _Recorder("b")
    proxy = RoundRobinProxy([a, b])
    assert [proxy.ping() for _ in range(4)] == ["a", "b", "a", "b"]
    assert a.calls == 2
    assert b.calls == 2


def test_round_robin_proxy_requires_delegate() -> None:
    """An empty delegate list is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        RoundRobinProxy([])


class _Echo(Worker):
    """Minimal scalar worker."""

    @worker_method()
    async def add(self, a: int, b: int) -> int:
        return a + b


@pytest.mark.asyncio
async def test_pool_runs_calls_over_children() -> None:
    """A pool deploys to every child and returns a round-robin proxy."""
    async with RoundRobinPool([LocalRuntime(), LocalRuntime()]) as pool:
        async with _pipeline_method_scope():
            echo = await pool.deploy(_Echo())
            assert isinstance(echo, RoundRobinProxy)
            assert await (await echo.add(2, 3)) == 5
            assert await (await echo.add(10, 4)) == 14


def test_pool_requires_a_runtime() -> None:
    """An empty pool is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        RoundRobinPool([])


class _RecordingRuntime(Runtime):
    """Minimal runtime that records the workers deployed to it."""

    def __init__(self) -> None:
        super().__init__()
        self.deployed: list[Worker] = []

    async def deploy_worker(self, worker: Worker) -> str:
        self.deployed.append(worker)
        return f"{type(worker).__name__}-{len(self.deployed)}"

    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AbstractAsyncContextManager[str]:
        raise NotImplementedError

    def get_result(self, resid: str) -> Awaitable[object]:
        raise NotImplementedError

    def stream_result(self, resid: str) -> AsyncIterator[object]:
        raise NotImplementedError

    async def get_metrics(self) -> str:
        return ""


class _CpuWorker(Worker):
    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def noop(self) -> int:
        return 0


class _GpuWorker(Worker):
    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def noop(self) -> int:
        return 0


class _TpuWorker(Worker):
    def __init__(self) -> None:
        super().__init__(deploy_hints=["tpu"])

    @worker_method()
    async def noop(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_hybrid_routes_by_deploy_hints() -> None:
    """Each worker deploys to the child runtime its hints select."""
    cpu_rt, gpu_rt = _RecordingRuntime(), _RecordingRuntime()
    async with HybridRuntime({"cpu": cpu_rt, "gpu": gpu_rt}) as hybrid:
        await hybrid.deploy(_CpuWorker())
        await hybrid.deploy(_GpuWorker())
    assert [type(w).__name__ for w in cpu_rt.deployed] == ["_CpuWorker"]
    assert [type(w).__name__ for w in gpu_rt.deployed] == ["_GpuWorker"]


@pytest.mark.asyncio
async def test_hybrid_rejects_unknown_hint() -> None:
    """A worker with no matching pool raises a clear error."""
    async with HybridRuntime({"cpu": _RecordingRuntime()}) as hybrid:
        with pytest.raises(ValueError, match="No worker pool"):
            await hybrid.deploy(_TpuWorker())
