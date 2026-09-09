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
import asyncio
import ctypes
import functools
import multiprocessing
import sys
import time
from collections.abc import Awaitable, Callable
from multiprocessing.synchronize import Event
from typing import NoReturn, ParamSpec, TypeVar

import pytest
from max.serve.process_control import SubprocessExit, subprocess_manager

# FIXME: SERVSYS-1275 — subprocess-control tests hang past their timeout on
# contended macOS CI workers, pushing the macOS job past its 2h cap. Skip on
# macOS until the serve-test slowness is addressed.
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="SERVSYS-1275: too slow on macOS CI; exceeds the 2h job timeout",
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


# Simple decorator to make hung test cases fail faster than the bazel 300s timeout
# FIXME: This is copy-pasted from conftest.py, because conftest is importing tons of max.* stuff
# which eventually leads to a pytorch import, which installs a segfault handler for dumping the stack
# Something about that handler causes the subprocess to hang instead of exit, but only in linux CI jobs
# while the test still works fine locally. So we copy the function to dodge the torch import here
def async_timeout(
    timeout: float,
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, _R]]:
    def decorator(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, _R]:
        @pytest.mark.asyncio
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout)

        return wrapper

    return decorator


ctx = multiprocessing.get_context("spawn")


def work(alive: Event, reps: int, pause: float) -> int:
    for _ in range(reps):
        time.sleep(pause)
        alive.set()
    return 123


def run_segfault(alive: Event) -> None:
    # this will segfault, bypassing all subproc try/catch
    # forcing us to monitor process liveness externally
    print(ctypes.string_at(0))
    alive.set()


def run_exception(alive: Event) -> NoReturn:
    alive.set()
    raise ValueError("KABOOM!")


def run_exit(alive: Event) -> NoReturn:
    alive.set()
    sys.exit(1)


@async_timeout(30)
async def test_ready_and_finish() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=3, pause=0)
        await proc.ready(alive, timeout=10)
        # Subprocess return value isn't propagated -- the lifecycle
        # task just completes when the subprocess exits cleanly.
        await asyncio.wait_for(task, timeout=10)


@async_timeout(30)
async def test_cancel() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=5, pause=4)
        await proc.ready(alive, timeout=10)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)


@async_timeout(30)
async def test_heartbeat_good() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=10, pause=0.5)
        await proc.ready(alive, timeout=10)
        proc.watch_heartbeat(alive, timeout=2)
        await asyncio.wait_for(task, timeout=10)


@async_timeout(30)
async def test_heartbeat_bad() -> None:
    with pytest.raises(TimeoutError, match="test1 failed heartbeat check"):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            task = proc.start(work, alive, reps=5, pause=4)
            proc.watch_heartbeat(alive, timeout=0.1)
            await asyncio.wait_for(task, timeout=10)
            raise AssertionError("Should not reach here")


@async_timeout(30)
async def test_exception_propagate() -> None:
    # The subprocess's original exception is logged by the child via the
    # standard `multiprocessing` mechanism but isn't passed back across
    # the process boundary; the parent only learns that the worker
    # exited with a non-zero code.
    with pytest.raises(SubprocessExit):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            proc.start(run_exception, alive)
            await proc.ready(alive, timeout=10)
            await asyncio.sleep(11)
            raise AssertionError("Should not reach here")


@async_timeout(60)
async def test_segfault() -> None:
    # would normally match="Segmentation fault"
    # but ASAN CI turns this into "Aborted" status
    # so we simply check for SubprocessExit
    with pytest.raises(SubprocessExit):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            proc.start(run_segfault, alive)
            await proc.ready(alive, timeout=50)
            raise AssertionError("Should not reach here")


@async_timeout(30)
async def test_hard_exit() -> None:
    with pytest.raises(SubprocessExit, match="1"):
        async with subprocess_manager("test1") as proc:
            alive = ctx.Event()
            proc.start(run_exit, alive)
            await proc.ready(alive, timeout=10)
            await asyncio.sleep(11)
            raise AssertionError("Should not reach here")


@async_timeout(30)
async def test_nested() -> None:
    async with subprocess_manager("test1") as proc:
        alive = ctx.Event()
        task = proc.start(work, alive, reps=3, pause=0)
        await proc.ready(alive, timeout=10)

        async with subprocess_manager("test2") as proc2:
            health2 = ctx.Event()
            task2 = proc2.start(work, health2, reps=3, pause=0)
            await proc2.ready(health2, timeout=10)
            await asyncio.wait_for(task2, timeout=10)

        await asyncio.wait_for(task, timeout=10)
