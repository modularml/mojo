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

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import signal
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import ParamSpec, TypeVar

from max.support._taskgroups import CancelGroup

logger = logging.getLogger("max.serve.process_control")

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Grace periods for forced subprocess teardown. These bound how long the
# event loop blocks during shutdown.
_SIGTERM_GRACE = 5.0
_SIGKILL_GRACE = 1.0


def _event_wait_clear(event: Event, timeout: float) -> None:
    if not event.wait(timeout):
        raise TimeoutError()
    event.clear()


class SubprocessExit(Exception):
    def __init__(self, info: str | int | None):
        if isinstance(info, int):
            try:
                info = signal.strsignal(-info)
            except ValueError:
                pass
        super().__init__(info)


async def _read_ready(fd: int) -> None:
    """async helper to wait for fd to enter readable state"""
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    loop.add_reader(fd, lambda: ready.set_result(True))
    try:
        await ready
    finally:
        loop.remove_reader(fd)


class _DebugSpawnProcess(multiprocessing.context.SpawnProcess):
    """SpawnProcess that keeps the child's stdin open for debugging."""

    # runs in remote process
    def _bootstrap(self, parent_sentinel: int | None = None) -> int:
        multiprocessing.util._close_stdin = lambda: None  # type: ignore[attr-defined]
        return super()._bootstrap(parent_sentinel)  # type: ignore[misc]


async def run_subprocess(
    func: Callable[_P, _R],
    name: str,
    daemon: bool,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> None:
    """async coroutine to run func(*args,**kwargs) in a subprocess.

    ``daemon`` controls the child's ``daemon`` flag. Passing ``True`` matches
    multiprocessing's safety behaviour of auto-terminating the child with the
    parent, but daemonic processes cannot spawn their own children. Callers that
    host a subprocess which itself launches further subprocesses (e.g. a cascade
    worker that spins up a ``max.serve`` model worker) must pass ``False``.
    Teardown is explicit in the ``finally`` block below, so a non-daemon child
    is still reliably reaped on context exit.

    Two entry points to the ``finally:`` cleanup:

    1. **Subprocess exits on its own.** The await wakes up, we reap
       with a bounded :py:meth:`Process.join`, and raise
       :py:class:`SubprocessExit` (for non-zero exit) or return
       normally (exit 0). ``finally:`` runs (mostly no-ops at this
       point) and any exception propagates out of the
       :py:class:`CancelGroup`.
    2. **We get cancelled.** The body finished (cleanly or with an
       exception), :py:class:`CancelGroup` cancels us, and the
       ``finally:`` block does the full SIGTERM/SIGKILL/close dance.

    Cleanup is synchronous on purpose: terminating a subprocess is an
    OS-level op, and making it async (kill tasks, callbacks, etc.) is
    what produced the race that SERVSYS-1270 tracked. Sync cleanup
    blocks the event loop for up to ``_SIGTERM_GRACE + _SIGKILL_GRACE``
    seconds in the worst case, but we're shutting down -- nothing else
    needs the loop.
    """

    proc = _DebugSpawnProcess(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
        name=name,
    )
    try:
        proc.start()
        # this await prevents tying up event loop with blocking join()
        # the "sentinel" file-descriptor tells us when the child exits
        await _read_ready(proc._popen.sentinel)  # type: ignore[attr-defined]
        proc.join()
        if proc.exitcode != 0:
            raise SubprocessExit(proc.exitcode)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=_SIGTERM_GRACE)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=_SIGKILL_GRACE)
        if proc.is_alive():
            logger.error(
                "subprocess %s (pid=%s) would not die; leaking handle",
                proc.name,
                proc.pid,
            )
        else:
            try:
                proc.close()
            except ValueError:
                # Best-effort: process is reaped but multiprocessing's
                # bookkeeping disagrees. Don't let cleanup raise --
                # would replace the in-flight exception (SERVSYS-1270).
                pass


@dataclass
class ProcessManager:
    """Handle yielded by :py:func:`subprocess_manager`.

    Owns one subprocess and any associated watcher tasks (heartbeat,
    etc.). All watchers run inside a shared :py:class:`CancelGroup`, so
    a failure in any of them tears down the body cleanly.

    Example::

        async with subprocess_manager("worker") as proc:
            alive = mp.Event()
            task = proc.start(work, alive)
            await proc.ready(alive, timeout=10)
            proc.watch_heartbeat(alive, timeout=2)
            await task  # blocks until subprocess exits
    """

    name: str
    group: CancelGroup
    daemon: bool = True
    task: asyncio.Task[None] | None = None
    heartbeat: asyncio.Task[None] | None = None

    def start(
        self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> asyncio.Task[None]:
        """Spawns ``func(*args, **kwargs)`` in a subprocess.

        Returns the lifecycle :py:class:`asyncio.Task`. Awaiting it
        blocks until the subprocess exits and surfaces
        :py:class:`SubprocessExit` for any non-zero exit. The
        subprocess's return value and exceptions are *not* propagated
        back to the parent; the subprocess is responsible for logging
        its own failures (see :py:meth:`logger.exception` in the
        worker entry points).
        """
        assert self.task is None, "ProcessManager.start may only be called once"
        self.task = self.group.create_task(
            run_subprocess(func, self.name, self.daemon, *args, **kwargs)
        )

        def task_done(_: asyncio.Task[None]) -> None:
            if self.heartbeat is not None:
                self.heartbeat.cancel()

        self.task.add_done_callback(task_done)

        return self.task

    async def ready(self, event: Event, timeout: float | None) -> None:
        """Waits for ``event`` to be set (then clears it).

        Fails fast if the subprocess exits before the event is set --
        otherwise we'd wait the full timeout for an event that's never
        going to come.

        Args:
            event: The readiness event the subprocess will set.
            timeout: Total wait in seconds. ``None`` waits forever.

        Raises:
            TimeoutError: If ``timeout`` elapses first.
            RuntimeError: If the subprocess exits before the event is
                set; chains the lifecycle task's exception (typically
                :py:class:`SubprocessExit`) as ``__cause__``.
        """
        assert self.task is not None, "call start() before ready()"
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        while True:
            # loop so thread is interruptible for cancellation
            try:
                await loop.run_in_executor(None, _event_wait_clear, event, 1)
                break
            except TimeoutError:
                pass
            if self.task.done():
                # task is done but event never got set, so something went wrong
                raise RuntimeError(
                    f"{self.name} failed to become ready"
                ) from self.task.exception()  # may be None, that's ok
            t1 = time.monotonic()
            if timeout is not None and t1 - t0 > timeout:
                raise TimeoutError(
                    f"{self.name} failed to become ready"
                ) from None

    def watch_heartbeat(
        self, event: Event, timeout: float
    ) -> asyncio.Task[None]:
        """Adds a background watcher that requires ``event`` to keep being set.

        If the subprocess fails to set ``event`` within ``timeout``
        seconds (repeatedly), :py:class:`TimeoutError` is raised, the
        body is cancelled, and the error propagates out of the
        :py:func:`subprocess_manager` block.
        """
        assert self.heartbeat is None, "watch_heartbeat may only be called once"

        async def run_task() -> None:
            try:
                while True:
                    await self.ready(event, timeout)
            except TimeoutError:
                raise TimeoutError(
                    f"{self.name} failed heartbeat check"
                ) from None

        self.heartbeat = self.group.create_task(run_task())
        return self.heartbeat

    def cancel(self) -> None:
        if self.heartbeat and not self.heartbeat.done():
            self.heartbeat.cancel()
        if self.task and not self.task.done():
            self.task.cancel()


@asynccontextmanager
async def subprocess_manager(
    name: str, *, daemon: bool = True
) -> AsyncGenerator[ProcessManager]:
    """Async context manager owning one :py:class:`ProcessManager`.

    Internally uses :py:class:`CancelGroup` to supervise the lifecycle
    task + any watchers; child failures cancel the body, body exit
    cancels the children. The subprocess is always reaped on exit.

    Args:
        name: Human-readable name for the managed subprocess.
        daemon: Whether the managed subprocess runs as a daemon. Pass
            ``daemon=False`` when the subprocess must spawn its own child
            processes (daemonic processes cannot). See
            :py:func:`run_subprocess`.
    """
    logger.info("Starting subprocess: %s", name)
    async with CancelGroup() as tg:
        yield ProcessManager(name=name, group=tg, daemon=daemon)
