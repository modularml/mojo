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

"""Functionality for replaying recordings."""

from __future__ import annotations

import asyncio
import contextlib
import math
import statistics
import sys
import time
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TextIO

import httpx
import scipy.special
from max.support._taskgroups import TaskGroup

from . import schema

__all__ = [
    # Please keep this list alphabetized.
    "BaseProgressNotifier",
    "ProgressSnapshot",
    "TerminalProgressNotifier",
    "TrackingProgressNotifier",
    "reconstitute_request",
    "replay_recording",
    "replay_transaction",
    "summarize_progress",
    "summarize_progress_stats",
    "summarize_progress_time",
    "summarize_progress_transactions",
]


def reconstitute_request(
    client: httpx.AsyncClient, request: schema.Request
) -> httpx.Request:
    """Turn a stored Request into an HTTPX Request."""
    return client.build_request(
        method=request.method,
        url=request.path,
        content=request.body,
        headers=httpx.Headers(request.headers),
    )


async def replay_transaction(
    client: httpx.AsyncClient, transaction: schema.Transaction
) -> None:
    """Replay a single transaction from a recording."""
    request = reconstitute_request(client, transaction.request)
    response = await client.send(request)
    response.raise_for_status()


class _IncrementalDistributionComputer:
    """Computes normal distribution statistics incrementally."""

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._undivided_var = 0.0

    def add_sample(self, sample: float) -> None:
        # Using Welford's method.  See also:
        # https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        new_count = self._count + 1
        old_mean = self._mean
        new_mean = old_mean + (sample - old_mean) / new_count
        self._count = new_count
        self._mean = new_mean
        self._undivided_var += (sample - old_mean) * (sample - new_mean)

    @property
    def current(self) -> statistics.NormalDist:
        mean = float("nan")
        stdev = float("nan")
        if self._count >= 1:
            mean = self._mean
        if self._count >= 2:
            stdev = math.sqrt(self._undivided_var / (self._count - 1))
        return statistics.NormalDist(mean, stdev)


def _tnmean(a: float) -> float:
    """Mean of the standard normal distribution, truncated to (a, +∞)."""
    # Implementation ported from https://github.com/cossio/TruncatedNormal.jl.
    return math.sqrt(2 / math.pi) / scipy.special.erfcx(a / math.sqrt(2))


def _low_truncated_expectation(
    dist: statistics.NormalDist, lower_bound: float
) -> float:
    return dist.mean + dist.stdev * _tnmean(
        (lower_bound - dist.mean) / dist.stdev
    )


# Please treat this class as-if it were kw_only=True.  I can't trick MyPy into
# accepting kw_only=True due to our minimum Python version, but consider it the
# intent.
@dataclass
class ProgressSnapshot:
    """A snapshot of replay progress at a moment in time."""

    completed_transactions: int
    transactions_in_progress: int
    concurrency: int
    total_transactions: int
    latency_distribution: statistics.NormalDist
    completed_transactions_total_seconds: float
    in_progress_transaction_durations: Iterable[float]
    elapsed_seconds: float

    @property
    def estimated_seconds_remaining(self) -> float:
        """Estimated number of seconds remaining in the replay."""
        if self.completed_transactions == 0:
            # Not even one has completed -- how could we possibly know how long
            # any would take?  Still, provide some kind of guess -- each future
            # task will probably take at least as long as we've taken so far.
            if self.concurrency == 0:
                # But if no workers have even started, we have 0 chance of
                # coming up with anything even remotely sensible, and we should
                # give up completely.
                return float("nan")
            return self.elapsed_seconds * math.ceil(
                self.total_transactions / self.concurrency
            )
        latency = self.latency_distribution.mean
        unstarted_count = (
            self.total_transactions
            - self.completed_transactions
            - self.transactions_in_progress
        )
        # Mental model: Say we have some guess about how long each worker
        # that's currently going is going to take until it's done.
        worker_times = []
        for worker_elapsed in self.in_progress_transaction_durations:
            # Which we compute based on the current duration and the expected
            # duration.  If we have distribution information, we use the
            # expected value of the observed latency distribution conditional
            # on that we've already spent as much time on this task as we have.
            # We fall back to the (one) latency we've observed if that's all we
            # have.
            est_task_dur = latency
            if self.completed_transactions >= 2:
                est_task_dur = _low_truncated_expectation(
                    self.latency_distribution, worker_elapsed
                )
            worker_times.append(max(est_task_dur - worker_elapsed, 0.0))
        # Then, we tack on the time these workers will need to do after they've
        # completed their current work.  If we have more tasks than
        # concurrency, these are likely to be "evenly" distributed among
        # workers, and we can tack that onto the total, so let's consider only
        # "uneven" unstarted tasks.  The workers that finish their current
        # tasks first are going to be the ones to pick these up.
        worker_times.sort(reverse=True)
        del worker_times[self.concurrency :]
        worker_times.extend([0.0] * (self.concurrency - len(worker_times)))
        if unstarted_count > 0:
            even_count, uneven_count = divmod(unstarted_count, self.concurrency)
            for i in range(uneven_count):
                worker_times[-1 - i] += latency
        else:
            even_count = 0
        return even_count * latency + max(worker_times, default=0.0)


def summarize_progress_transactions(progress: ProgressSnapshot) -> str:
    """Provide a string summarizing progress of completed transactions."""
    completed = progress.completed_transactions
    in_prog = progress.transactions_in_progress
    total = progress.total_transactions
    width = len(str(progress.total_transactions))
    conc_width = len(str(progress.concurrency))
    return f"{completed:{width}}+{in_prog:{conc_width}}/{total:{width}}"


def _format_big_duration(seconds: float) -> str:
    """Format a "large" duration into a human-readable string.

    A "large" duration is a duration expected to be in the range of roughly
    several seconds to several hours, for which sub-second information is not
    important.
    """
    if math.isnan(seconds):
        return "UNK"
    if math.isinf(seconds):
        return "∞"
    minutes_f, seconds = divmod(seconds, 60)
    minutes = int(minutes_f)
    hours, minutes = divmod(minutes, 60)
    pieces = []
    if hours:
        pieces.append(f"{hours}h")
        pieces.append(f"{minutes:02d}m")
    elif minutes:
        pieces.append(f"{minutes}m")
    pieces.append(f"{math.floor(seconds):02d}s")
    return "".join(pieces)


def summarize_progress_time(progress: ProgressSnapshot) -> str:
    """Summarize wall timing of current progress."""
    elapsed = _format_big_duration(progress.elapsed_seconds)
    eta = _format_big_duration(progress.estimated_seconds_remaining)
    return f"Elap:{elapsed} ETA:{eta}"


def _format_little_decimal(value: float) -> str:
    """Format a float with a 'reasonable' amount of decimal places.

    >>> _format_little_decimal(0)
    '-0-'
    >>> _format_little_decimal(123.456)
    '123.5'
    >>> _format_little_decimal(12.3456)
    '12.35'
    >>> _format_little_decimal(1.23456)
    '1.23'
    >>> _format_little_decimal(0.123456)
    '0.123'
    >>> _format_little_decimal(0.0123456)
    '1.235e-02'
    """
    if value == 0:
        return "-0-"
    if value > 100:
        return f"{value:.1f}"
    if value > 1:
        return f"{value:.2f}"
    if value > 0.1:
        return f"{value:.3f}"
    return f"{value:.3e}"


def _format_little_duration(seconds: float) -> str:
    """Format a duration expected to be smallish, where sub-second can matter."""
    if math.isnan(seconds):
        return "UNK"
    if seconds == 0:
        return "-0-"
    if seconds > 0.1:
        return f"{_format_little_decimal(seconds)}s"
    if seconds > 1e-3:
        return f"{_format_little_decimal(seconds * 1e3)}ms"
    if seconds > 1e-6:
        return f"{_format_little_decimal(seconds * 1e6)}µs"
    return f"{_format_little_decimal(seconds * 1e9)}ns"


def summarize_progress_stats(progress: ProgressSnapshot) -> str:
    """Summarize statistics of the run into a short string."""
    latency = progress.latency_distribution.mean
    latency_str = _format_little_duration(latency)
    rps_str = _format_little_decimal(progress.concurrency / latency)
    return f"MeanLat {latency_str}; {rps_str} RPS"


def summarize_progress(progress: ProgressSnapshot) -> str:
    """Summarize current progress into a short one-line progress summary."""
    tx_piece = summarize_progress_transactions(progress)
    time_piece = summarize_progress_time(progress)
    stats_piece = summarize_progress_stats(progress)
    return f"{tx_piece} {time_piece}. {stats_piece}"


class BaseProgressNotifier:
    """Something to be notified of progress-related events during replay."""

    __next_worker_id: int

    def __init__(self) -> None:
        """Create a new base progress notifier."""
        self.__next_worker_id = 0

    @contextlib.contextmanager
    def worker_scope(self) -> Iterator[int]:
        """Get an ID to use for worker-specific updates."""
        worker_id = self.__next_worker_id
        self.__next_worker_id += 1
        yield worker_id

    def adjust_total_tasks(self, delta: int) -> None:
        """Notify of a change in the number of tasks."""

    def worker_start_item(self, worker: int, item: int) -> None:
        """Notify that a worker has picked up an item."""

    def worker_finish_item(self, worker: int, item: int) -> None:
        """Notify that a worker has finished an item."""


class TrackingProgressNotifier(BaseProgressNotifier):
    """A progress notifier that tracks statistics during notifications."""

    __start_time: float
    __concurrency: int
    __total_tasks: int
    __completed_tasks: int
    __completed_tasks_total_seconds: float
    __worker_task_start_times: dict[int, float]
    __latency_distribution_computer: _IncrementalDistributionComputer

    def __init__(self) -> None:
        """Create a new tracking progress notifier."""
        super().__init__()
        self.__start_time = time.monotonic()
        self.__concurrency = 0
        self.__total_tasks = 0
        self.__completed_tasks = 0
        self.__completed_tasks_total_seconds = 0
        self.__worker_task_start_times = {}
        self.__latency_distribution_computer = (
            _IncrementalDistributionComputer()
        )

    @contextlib.contextmanager
    def worker_scope(self) -> Iterator[int]:
        """Get an ID to use for worker-specific updates."""
        with super().worker_scope() as worker_id:
            self.__concurrency += 1
            self.progress_changed()
            yield worker_id
            self.__concurrency -= 1
            self.progress_changed()

    def adjust_total_tasks(self, delta: int) -> None:
        """Notify of a change in the number of tasks."""
        self.__total_tasks += delta
        self.progress_changed()
        super().adjust_total_tasks(delta)

    def worker_start_item(self, worker: int, item: int) -> None:
        """Notify that a worker has picked up an item."""
        assert worker not in self.__worker_task_start_times
        self.__worker_task_start_times[worker] = time.monotonic()
        super().worker_start_item(worker, item)

    def worker_finish_item(self, worker: int, item: int) -> None:
        """Notify that a worker has finished an item."""
        now = time.monotonic()
        task_start = self.__worker_task_start_times.pop(worker)
        self.__completed_tasks += 1
        self.__completed_tasks_total_seconds += now - task_start
        self.__latency_distribution_computer.add_sample(now - task_start)
        self.progress_changed(major=True)
        super().worker_finish_item(worker, item)

    @property
    def current_progress(self) -> ProgressSnapshot:
        """A snapshot of the current progress."""
        now = time.monotonic()
        return ProgressSnapshot(
            completed_transactions=self.__completed_tasks,
            transactions_in_progress=len(self.__worker_task_start_times),
            concurrency=self.__concurrency,
            total_transactions=self.__total_tasks,
            latency_distribution=self.__latency_distribution_computer.current,
            completed_transactions_total_seconds=(
                self.__completed_tasks_total_seconds
            ),
            in_progress_transaction_durations=[
                now - task_start
                for task_start in self.__worker_task_start_times.values()
            ],
            elapsed_seconds=now - self.__start_time,
        )

    def progress_changed(self, *, major: bool = False) -> None:
        """Notify that progress has changed.

        Subclasses should implement this to be notified of progress changes.
        """


class TerminalProgressNotifier(TrackingProgressNotifier):
    """A progress notifier that prints periodic messages to the terminal."""

    __stream: TextIO
    __tty: bool
    __printer_task: asyncio.Task[object] | None
    __major_update_event: asyncio.Event

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__()
        if stream is None:
            stream = sys.stderr
        self.__stream = stream
        self.__tty = bool(getattr(stream, "isatty", lambda: False)())
        self.__printer_task = None
        self.__major_update_event = asyncio.Event()

    async def __aenter__(self) -> TerminalProgressNotifier:
        if self.__printer_task is None:
            self.__printer_task = asyncio.create_task(self.__printer())
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: Any,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.__printer_task is not None:
            self.__printer_task.cancel()
            await self.__printer_task
            self.__printer_task = None

    def progress_changed(self, *, major: bool = False) -> None:
        """Notify that progress has changed."""
        if major:
            self.__major_update_event.set()
        super().progress_changed(major=major)

    async def __printer(self) -> None:
        if self.__tty:
            await self.__tty_printer()
        else:
            await self.__nontty_printer()

    async def __nontty_printer(self) -> None:
        try:
            while True:
                await self.__major_update_event.wait()
                self.__major_update_event.clear()
                snapshot = self.current_progress
                print(
                    summarize_progress(snapshot), file=self.__stream, flush=True
                )
        except asyncio.CancelledError:
            pass

    async def __tty_printer(self) -> None:
        last_printed = ""

        def print_update(*, final: bool = False) -> None:
            nonlocal last_printed
            clear_str = ""
            if last_printed:
                clear_str = "\r" + " " * len(last_printed) + "\r"
            snapshot = self.current_progress
            to_print = summarize_progress(snapshot)
            self.__stream.write(clear_str + to_print + ("\n" if final else ""))
            self.__stream.flush()
            last_printed = to_print

        try:
            while True:
                print_update()
                try:
                    await asyncio.wait_for(self.__major_update_event.wait(), 1)
                except asyncio.TimeoutError:
                    pass
                else:
                    self.__major_update_event.clear()
        except asyncio.CancelledError:
            pass
        print_update(final=True)


async def replay_recording(
    recording: schema.Recording,
    *,
    concurrency: int = 1,
    client: httpx.AsyncClient,
    notifier: BaseProgressNotifier | None = None,
    unwind_on_error: bool = False,
) -> None:
    """Replay a whole recording.

    Args:
        concurrency: Number of requests to keep in flight at a time.
        client:
            HTTPX client used to send requests.  You will need to set a
            base_url on this client, as paths provided in requests contain no
            host name on their own.  It is also recommended to set a custom
            timeout, as the HTTPX default of 5 seconds is sometimes too short
            under heavy load.
        notifier: Delegate to be notified of replay progress.
        unwind_on_error:
            If an error or cancellation occurs, tell the progress notifier that
            any in-progress transactions have completed and any queued
            transactions have been dequeued.  This provides a more accurate
            reflection of the current state to the progress notifier if
            multiple separate recordings are being replayed on the same
            notifier at the same time, but can be confusing in the more common
            case where each recording replay runs with a different progress
            notifier.
    """
    if notifier is None:
        notifier = BaseProgressNotifier()
    recording_txns = [
        item for item in recording if isinstance(item, schema.Transaction)
    ]
    recording_iter = iter(enumerate(recording_txns))

    async def worker() -> None:
        with notifier.worker_scope() as worker_id:
            while True:
                try:
                    item_index, item = next(recording_iter)
                except StopIteration:
                    break
                notifier.worker_start_item(worker_id, item_index)
                try:
                    await replay_transaction(client, item)
                finally:
                    if unwind_on_error:
                        notifier.worker_finish_item(worker_id, item_index)
                if not unwind_on_error:
                    notifier.worker_finish_item(worker_id, item_index)

    notifier.adjust_total_tasks(len(recording))
    try:
        async with TaskGroup() as task_group:
            for i in range(concurrency):  # noqa: B007
                task_group.create_task(worker())
    finally:
        if unwind_on_error:
            notifier.adjust_total_tasks(-len(list(recording_iter)))
