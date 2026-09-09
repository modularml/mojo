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

"""Background recorder for GPU diagnostics."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import select
import selectors
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import IO, Any, AnyStr, BinaryIO, Generic, TypeVar, cast

import msgspec
from typing_extensions import Buffer

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    from taskgroup import TaskGroup

from .multi import GPUDiagContext
from .types import GPUStats

_T = TypeVar("_T")


class _Request(msgspec.Struct):
    pass


class _StopRequest(_Request, tag="stop_request"):
    pass


class _Reply(msgspec.Struct):
    pass


class _StopReply(_Reply, tag="stop_reply"):
    path: str


class _Notification(msgspec.Struct):
    pass


class _ReadyNotification(_Notification, tag="ready_notification"):
    pass


class _DieNotification(_Notification, tag="die_notification"):
    cause: str


_AllRequests = _StopRequest
_AllReplies = _StopReply
_AllNotifications = _ReadyNotification | _DieNotification


async def _want_readable(file: IO[AnyStr]) -> None:
    fd = file.fileno()
    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_reader(fd, event.set)
    try:
        await event.wait()
    finally:
        loop.remove_reader(fd)


class _InputChannel(Generic[_T]):
    def __init__(self, file: IO[bytes], type: type[_T]) -> None:
        self._file = file
        if hasattr(self._file, "raw"):
            self._file = self._file.raw
        self._decoder = msgspec.json.Decoder(type=type)
        self._buffer = bytearray()

    def read_sync(self, *, timeout: float | None = None) -> _T | None:
        if b"\n" not in self._buffer:
            if timeout is not None:
                # Use selectors instead of select() to avoid FD_SETSIZE (1024) limit
                sel = selectors.DefaultSelector()
                try:
                    sel.register(self._file, selectors.EVENT_READ)
                    events = sel.select(timeout)
                    if not events:
                        raise TimeoutError
                finally:
                    sel.close()
            self._buffer.extend(self._file.read(select.PIPE_BUF))
        if not self._buffer:
            return None
        index = self._buffer.index(b"\n")
        line = bytes(self._buffer[:index])
        del self._buffer[: index + 1]
        return self._decoder.decode(line)

    async def read_async(self) -> _T | None:
        await _want_readable(self._file)
        return self.read_sync()


class _OutputChannel(Generic[_T]):
    def __init__(self, file: IO[bytes]) -> None:
        self._file = file

    def write(self, item: _T) -> None:
        encoded = msgspec.json.encode(item) + b"\n"
        if len(encoded) > select.PIPE_BUF:
            raise ValueError(
                f"{type(item)} message of {len(encoded)} bytes too big for pipe"
            )
        written = self._file.write(encoded)
        assert written == len(encoded)
        self._file.flush()


def _kill_gracefully(proc: subprocess.Popen[AnyStr]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    else:
        return
    proc.kill()
    proc.wait()


class _RecorderProcess:
    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self.proc = proc
        self.dead = False
        assert proc.stdin is not None
        assert proc.stdout is not None
        self.inch = _InputChannel(
            proc.stdout,
            # Cast is a hack to work around this MyPy error:
            #     error: Argument 2 to "_InputChannel" has incompatible type
            #     "<typing special form>"
            # It's true it's not a real 'type' object, but msgspec knows how to
            # deal with this special form so it's OK.
            cast(
                type[_AllReplies | _AllNotifications],
                _AllReplies | _AllNotifications,
            ),
        )
        self.outch = _OutputChannel[_AllRequests](proc.stdin)

    def _require_live(self) -> None:
        if self.dead:
            raise RuntimeError("Recorder process is dead")

    def kill_gracefully(self) -> None:
        _kill_gracefully(self.proc)
        self.dead = True

    def dying_gracefully(self) -> None:
        """Like kill_gracefully, but assumes process already likely dying."""
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        else:
            self.dead = True
            return
        self.kill_gracefully()

    def handle_die_notification(self, notif: _DieNotification) -> None:
        self.dying_gracefully()
        raise RuntimeError(
            f"Recorder process died with exception: {notif.cause}"
        )

    def read_nonerror(
        self, *, timeout: float | None = None
    ) -> _AllReplies | _AllNotifications:
        self._require_live()
        item = self.inch.read_sync(timeout=timeout)
        if item is None:
            self.dying_gracefully()
            raise RuntimeError("Recorder process died with no message")
        if isinstance(item, _DieNotification):
            self.handle_die_notification(item)
        return item

    def wait_for_ready(self, *, timeout: float) -> None:
        deadline = time.time() + timeout
        while (now := time.time()) < deadline:
            item = self.read_nonerror(timeout=deadline - now)
            if isinstance(item, _ReadyNotification):
                return
        raise TimeoutError(
            f"Recorder process not ready after {timeout} seconds"
        )

    def transact(
        self, request: _AllRequests, *, timeout: float | None = None
    ) -> _AllReplies:
        self._require_live()
        self.outch.write(request)
        if timeout is None:
            deadline = None
        else:
            deadline = time.time() + timeout
        while deadline is None or (now := time.time()) < deadline:
            self._require_live()
            remaining = None if deadline is None else deadline - now
            item = self.read_nonerror(timeout=remaining)
            if isinstance(item, _Reply):
                return item
        raise TimeoutError(
            f"{type(request)} request did not complete in {timeout} seconds"
        )

    def stop(self) -> Path:
        reply = self.transact(_StopRequest(), timeout=5)
        if not isinstance(reply, _StopReply):
            raise RuntimeError(
                f"Unexpected {type(reply)} reply to stop request"
            )
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        else:
            self.dead = True
        if not self.dead:
            self.kill_gracefully()
        return Path(reply.path)

    def exhaust(self, *, timeout: float) -> None:
        deadline = time.time() + timeout
        while (now := time.time()) < deadline:
            if self.dead:
                break
            item = self.inch.read_sync(timeout=deadline - now)
            if item is None:
                self.dying_gracefully()
                return
            if isinstance(item, _DieNotification):
                self.handle_die_notification(item)

    def stop_without_request(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.exhaust(timeout=5)


def _read_stats(path: Path) -> list[dict[str, GPUStats]]:
    decoder = msgspec.json.Decoder(type=dict[str, GPUStats])
    stats: list[dict[str, GPUStats]] = []
    for line in path.open("rb"):
        stats.append(decoder.decode(line))
    return stats


class BackgroundRecorder:
    """Asynchronous GPU metrics collection and data export capabilities.

    The ``BackgroundRecorder`` enables continuous monitoring of GPU performance metrics
    without blocking the main application thread. It automatically samples GPU
    statistics at one-second intervals in a separate process, making it ideal for
    profiling long-running inference sessions or training workloads.

    When used as a context manager, the recorder starts background collection upon
    entry and stops collection upon exit. The collected statistics are then
    available through the `stats` property as a time-series of GPU measurements.

    .. code-block:: python

        from max.profiler.gpu import BackgroundRecorder

        with BackgroundRecorder() as recorder:
            # Run your GPU workload here
            sum(range(1_000_000))

        # Access collected time-series data
        for i, snapshot in enumerate(recorder.stats):
            print(f"Sample {i}: {len(snapshot)} GPUs detected")
            for gpu_id, gpu_stats in snapshot.items():
                print(f"  {gpu_id}: {gpu_stats.memory.used_bytes} bytes used")

    Args:
        interval: How frequently (in seconds) to sample GPU statistics.
            Default is 1 second.
    """

    def __init__(self, *, interval: float = 1.0) -> None:
        if interval < 0:
            raise ValueError("Interval must be non-negative")
        self._proc: _RecorderProcess | None = None
        self._stats: list[dict[str, GPUStats]] | None = None
        self._interval = interval

    @property
    def stats(self) -> list[dict[str, GPUStats]]:
        """Time-series of GPU statistics collected during background recording.

        Returns:
            A list of dictionaries, where each dictionary represents GPU statistics
            at a specific point in time. Each dictionary maps GPU identifiers to
            their corresponding :obj:`GPUStats` objects.

        Raises:
            RuntimeError: If accessed before the recorder context has exited.
        """
        if self._stats is None:
            raise RuntimeError("Recorder has not finished yet")
        return self._stats

    def __enter__(self) -> BackgroundRecorder:
        if self._proc is not None:
            raise RuntimeError("Recorder already running")
        if self._stats is not None:
            raise RuntimeError("Recorder has already been used once")
        raw_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                ".".join(__name__.split(".")[:-1]) + "._bgrec_main",
                str(self._interval),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        proc = _RecorderProcess(raw_proc)
        try:
            # The child re-imports the max package before signaling ready,
            # which can exceed 10s on contended CI workers. A dead child is
            # detected separately (EOF/die notification), so a generous
            # deadline costs nothing on the happy path.
            proc.wait_for_ready(timeout=60)
        except:
            proc.kill_gracefully()
            raise
        self._proc = proc
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._proc is None:
            return
        try:
            if exc_value is None:
                # Graceful stop
                output_path = self._proc.stop()
                self._stats = _read_stats(output_path)
                os.unlink(output_path)
            else:
                # Exception was thrown; tear down faster
                self._proc.stop_without_request()
        finally:
            self._proc.kill_gracefully()
            self._proc = None


class _KeepableTempFile(io.RawIOBase, BinaryIO):
    def __init__(self) -> None:
        self._file: tempfile._TemporaryFileWrapper[bytes] | None = (
            tempfile.NamedTemporaryFile(delete=False)
        )

    def __del__(self) -> None:
        self.close()

    def write(self, data: Buffer) -> int:
        if self._file is None:
            raise RuntimeError("File is closed")
        return self._file.write(data)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            name = self._file.name
            self._file = None
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def close_and_keep(self) -> Path:
        if self._file is None:
            raise RuntimeError("File already closed")
        self._file.close()
        path = Path(self._file.name)
        self._file = None
        return path


async def recorder_async_main() -> None:
    """Runs the background GPU diagnostics recorder event loop.

    Expects the sampling interval in seconds as ``sys.argv[1]``. Communicates
    with the parent process over ``stdin`` and ``stdout`` using msgspec-encoded
    requests and replies, samples GPU statistics at the configured interval,
    and writes JSON snapshots to a temporary file until a stop request is
    received.
    """
    interval = float(sys.argv[1])
    inch = _InputChannel(sys.stdin.buffer, _AllRequests)
    outch = _OutputChannel[_AllReplies | _AllNotifications](sys.stdout.buffer)

    @contextlib.contextmanager
    def die_notifier() -> Generator[None, None, None]:
        try:
            yield
        except asyncio.CancelledError:
            # This is usually intentional from the parent process.
            # Don't confuse it into thinking we crashed during shutdown.
            raise
        except BaseException as e:
            cause_texts = traceback.format_exception_only(type(e), e)
            outch.write(_DieNotification(cause_texts[0].rstrip()[:200]))
            raise

    async def recorder(output_file: IO[bytes]) -> None:
        while True:
            stats = diag.get_stats()
            output_file.write(msgspec.json.encode(stats) + b"\n")
            await asyncio.sleep(interval)

    async with contextlib.AsyncExitStack() as stack:
        stack.enter_context(die_notifier())
        diag = stack.enter_context(GPUDiagContext())
        group = await stack.enter_async_context(TaskGroup())
        output_file = stack.enter_context(_KeepableTempFile())
        recording_task = group.create_task(recorder(output_file))
        outch.write(_ReadyNotification())
        try:
            while req := await inch.read_async():
                if isinstance(req, _StopRequest):
                    recording_task.cancel()
                    try:
                        await recording_task
                    except asyncio.CancelledError:
                        pass
                    outch.write(_StopReply(str(output_file.close_and_keep())))
                    break
                else:
                    raise TypeError(f"Unknown request {req}")
        except KeyboardInterrupt:
            pass
        finally:
            recording_task.cancel()


def recorder_main() -> None:
    """Starts the background GPU diagnostics recorder subprocess."""
    try:
        asyncio.run(recorder_async_main())
    except KeyboardInterrupt:
        pass
