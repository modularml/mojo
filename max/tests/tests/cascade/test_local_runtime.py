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
"""Showcase tests for the LocalRuntime.

Each test demonstrates a specific capability of the cascade LocalRuntime:

- Deploying workers and calling methods
- Awaiting single results
- Streaming results via async generators
- Chaining results between workers (MaybeAsync argument resolution)
- Worker lifecycle management (open/close)
- Error propagation from worker methods
- Concurrent method calls
- Multiple workers coordinating in a pipeline

Each test wraps its body in the framework-internal
``_pipeline_method_scope`` to establish a pipeline-method scope; outside
a scope, proxy method calls would raise. User code reaches the same
scope by decorating an entry point with :py:func:`pipeline_method`. The
calls themselves use two ``await`` s -- one to get the
:py:class:`Result` handle, one to resolve it -- which mirrors what code
inside a ``@pipeline_method``-decorated body sees as well (the inner
await is done automatically by the consuming worker method's
``MaybeAsync`` argument resolution).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from max.experimental.cascade import LocalRuntime, Result, Worker, worker_method
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope

# ---------------------------------------------------------------------------
# Test workers
# ---------------------------------------------------------------------------


class Adder(Worker):
    """Worker that performs arithmetic operations."""

    @worker_method()
    async def add(self, a: int, b: int) -> int:
        return a + b

    @worker_method()
    async def add_three(self, a: int, b: int, c: int) -> int:
        return a + b + c


class Multiplier(Worker):
    """Worker that multiplies values."""

    @worker_method()
    async def multiply(self, a: int, b: int) -> int:
        return a * b


class Counter(Worker):
    """Worker that streams a count of integers."""

    @worker_method()
    async def count_up(self, n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    @worker_method()
    async def count_from(self, start: int, n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield start + i


class StatefulWorker(Worker):
    """Worker with setup/teardown lifecycle demonstrating open()."""

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
        self.closed = False

    @asynccontextmanager
    async def open(self) -> AsyncIterator[StatefulWorker]:
        self.initialized = True
        try:
            yield self
        finally:
            self.closed = True

    @worker_method()
    async def is_ready(self) -> bool:
        return self.initialized


class FailingWorker(Worker):
    """Worker that raises exceptions to demonstrate error propagation."""

    @worker_method()
    async def fail(self, message: str) -> str:
        raise ValueError(message)

    @worker_method()
    async def succeed(self) -> str:
        return "ok"


class SlowAdder(Worker):
    """Worker that sleeps before adding, to demonstrate concurrent resolution."""

    @worker_method()
    async def slow_add(self, a: int, b: int, c: int) -> int:
        await asyncio.sleep(0.1)
        return a + b + c


class StringWorker(Worker):
    """Worker for string transformations, useful in chaining demos."""

    @worker_method()
    async def upper(self, text: str) -> str:
        return text.upper()

    @worker_method()
    async def repeat(self, text: str, n: int) -> str:
        return text * n

    @worker_method()
    async def join(self, parts: list[str]) -> str:
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tests: Basic deployment and method calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_and_call() -> None:
    """Deploy a worker, call a method, and await the result."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        adder = await rt.deploy(Adder())
        handle = await adder.add(2, 3)

        # The first await returned a Result handle, not the value yet.
        assert isinstance(handle, Result)

        # A second await resolves it. Inside a ``@pipeline_method`` body,
        # this second await is done automatically when the handle is
        # passed as an argument to the next worker method.
        assert await handle == 5


@pytest.mark.asyncio
async def test_deploy_multiple_workers() -> None:
    """Deploy multiple workers and call methods on each independently."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        adder = await rt.deploy(Adder())
        multiplier = await rt.deploy(Multiplier())

        add_handle = await adder.add(10, 20)
        assert await add_handle == 30

        multiply_handle = await multiplier.multiply(4, 5)
        assert await multiply_handle == 20


# ---------------------------------------------------------------------------
# Tests: Streaming results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_results() -> None:
    """Async generator methods produce iterable result streams."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        counter = await rt.deploy(Counter())

        items = [item async for item in await counter.count_up(5)]
        assert items == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_streaming_with_arguments() -> None:
    """Streaming methods accept arguments like regular methods."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        counter = await rt.deploy(Counter())

        items = [item async for item in await counter.count_from(10, 3)]
        assert items == [10, 11, 12]


@pytest.mark.asyncio
async def test_empty_stream() -> None:
    """A streaming method that yields zero items produces an empty iteration."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        counter = await rt.deploy(Counter())

        items = [item async for item in await counter.count_up(0)]
        assert items == []


# ---------------------------------------------------------------------------
# Tests: Result chaining (MaybeAsync argument resolution)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_results_between_workers() -> None:
    """Pass a Result handle from one worker as input to another.

    The @worker_method decorator automatically awaits MaybeAsync arguments,
    so you can wire worker outputs directly as inputs to other workers
    without manually awaiting intermediate results.
    """
    async with LocalRuntime() as rt, _pipeline_method_scope():
        adder = await rt.deploy(Adder())
        multiplier = await rt.deploy(Multiplier())

        # adder.add returns a Result[int] (after one await); passed
        # directly to multiplier, whose MaybeAsync arg resolution awaits
        # it on the worker side.
        sum_result = await adder.add(3, 4)  # Result[int] = 7
        product = await multiplier.multiply(sum_result, 2)  # 7 * 2 = 14

        assert await product == 14


@pytest.mark.asyncio
async def test_chain_multiple_steps() -> None:
    """Chain three operations across workers in sequence."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        adder = await rt.deploy(Adder())
        multiplier = await rt.deploy(Multiplier())

        step1 = await adder.add(1, 2)  # 3
        step2 = await multiplier.multiply(step1, step1)  # 3 * 3 = 9
        step3 = await adder.add(step2, 1)  # 9 + 1 = 10

        assert await step3 == 10


@pytest.mark.asyncio
async def test_chain_result_into_stream() -> None:
    """Pass a Result handle as an argument to a streaming method."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        adder = await rt.deploy(Adder())
        counter = await rt.deploy(Counter())

        count = await adder.add(2, 3)  # Result[int] = 5
        items = [item async for item in await counter.count_up(count)]
        assert items == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Tests: Worker lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_lifecycle() -> None:
    """Workers' open() contexts are entered on deploy and exited on shutdown."""
    worker = StatefulWorker()
    assert not worker.initialized

    async with LocalRuntime() as rt:
        async with _pipeline_method_scope():
            proxy = await rt.deploy(worker)
            # open() was called during deploy
            assert worker.initialized
            ready_handle = await proxy.is_ready()
            assert await ready_handle is True

    # After runtime closes, worker's open() context was exited
    assert worker.closed


# ---------------------------------------------------------------------------
# Tests: Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_propagation() -> None:
    """Exceptions raised in worker methods propagate to the awaiter."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        worker = await rt.deploy(FailingWorker())

        with pytest.raises(ValueError, match="something went wrong"):
            fail_handle = await worker.fail("something went wrong")
            await fail_handle


@pytest.mark.asyncio
async def test_error_does_not_poison_runtime() -> None:
    """A failed call does not prevent subsequent successful calls."""
    async with LocalRuntime() as rt, _pipeline_method_scope():
        worker = await rt.deploy(FailingWorker())

        with pytest.raises(ValueError):
            fail_handle = await worker.fail("boom")
            await fail_handle

        # Runtime is still functional
        succeed_handle = await worker.succeed()
        assert await succeed_handle == "ok"


# ---------------------------------------------------------------------------
# Tests: Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls() -> None:
    """Arguments from concurrent slow calls resolve in parallel.

    Three slow_add calls (each sleeping 0.1s) feed their Results into a
    fourth slow_add. If resolution were sequential, the total would be
    ~0.4s; concurrent resolution keeps it around ~0.2s.
    """
    async with LocalRuntime() as rt, _pipeline_method_scope():
        worker = await rt.deploy(SlowAdder())

        r1 = await worker.slow_add(1, 2, 3)  # 6, takes 0.1s
        r2 = await worker.slow_add(4, 5, 6)  # 15, takes 0.1s
        r3 = await worker.slow_add(7, 8, 9)  # 24, takes 0.1s

        # Pass all three Result handles into a single call -- the
        # @worker_method decorator resolves them concurrently.
        start = asyncio.get_event_loop().time()
        total_handle = await worker.slow_add(r1, r2, r3)
        total = await total_handle  # 6+15+24 = 45
        elapsed = asyncio.get_event_loop().time() - start

        assert total == 45
        assert elapsed < 0.3


# ---------------------------------------------------------------------------
# Tests: Multi-worker pipelines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mini_pipeline() -> None:
    """Compose workers into a mini pipeline with chained results.

    Demonstrates the core pattern: build a pipeline by wiring Result
    handles between deployed workers, then await only the final output.
    """
    async with LocalRuntime() as rt, _pipeline_method_scope():
        strings = await rt.deploy(StringWorker())

        # Pipeline: upper("hello") -> repeat(result, 3)
        uppered = await strings.upper("hello")  # Result[str] = "HELLO"
        repeated = await strings.repeat(uppered, 3)  # "HELLOHELLOHELLO"

        assert await repeated == "HELLOHELLOHELLO"


@pytest.mark.asyncio
async def test_deploy_outside_context_raises() -> None:
    """Deploying a worker outside the runtime context raises an error."""
    rt = LocalRuntime()
    with pytest.raises(RuntimeError, match="context not entered"):
        await rt.deploy_worker(Adder())


@pytest.mark.asyncio
async def test_call_outside_pipeline_method_scope_raises() -> None:
    """Calling a proxy method outside a pipeline scope raises."""
    async with LocalRuntime() as rt:
        adder = await rt.deploy(Adder())
        with pytest.raises(RuntimeError, match="No cascade pipeline scope"):
            await adder.add(1, 2)
