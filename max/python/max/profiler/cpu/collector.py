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

"""CPU metrics collector for process-level CPU utilization."""

from __future__ import annotations

import time
from typing import Any

import psutil

from .types import CPUMetrics


def collect_pids_for_port(port: int) -> list[int]:
    """Collects PIDs of processes (and their children) listening on a port.

    Args:
        port: The port number to check.

    Returns:
        A list of PIDs of processes listening on the specified port.
    """
    pids: set[int] = set()

    def add_child_pids(pid: int) -> None:
        pids.add(pid)
        for proc in psutil.process_iter(["pid", "ppid"]):
            if proc.info["ppid"] == pid and proc.info["pid"] not in pids:
                add_child_pids(proc.info["pid"])

    for conn in psutil.net_connections(kind="inet"):
        if (
            conn.laddr.port == port
            and conn.status == psutil.CONN_LISTEN
            and conn.pid
        ):
            add_child_pids(conn.pid)

    return list(pids)


class CPUMetricsCollector:
    """Collects aggregate CPU time across a set of PIDs.

    Use as a context manager around the workload, then call :meth:`get_stats`
    to obtain the :class:`CPUMetrics` summary.

    .. code-block:: python

        import os

        from max.profiler.cpu import CPUMetricsCollector

        collector = CPUMetricsCollector([os.getpid()])
        with collector:
            sum(range(1_000_000))
        metrics = collector.get_stats()

    .. invisible-code-block: python

        assert metrics is not None

    Args:
        pids: The PIDs of the processes to collect CPU times from.
    """

    def __init__(self, pids: list[int]) -> None:
        self.pids: list[int] = pids
        self.clock_start: float | None = None
        self.clock_end: float | None = None
        self.cpu_times_start: dict[int, Any] = {}
        self.cpu_times_end: dict[int, Any] = {}

    def __enter__(self) -> CPUMetricsCollector:
        """Records the start clock and per-PID CPU times.

        Processes that no longer exist record ``None`` and are skipped in
        :meth:`get_stats`.
        """
        self.clock_start = time.monotonic()
        for pid in self.pids:
            try:
                proc = psutil.Process(pid)
                self.cpu_times_start[pid] = proc.cpu_times()
            except psutil.NoSuchProcess:
                self.cpu_times_start[pid] = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Records the stop clock and per-PID CPU times.

        Processes that no longer exist record ``None`` and are skipped in
        :meth:`get_stats`.
        """
        self.clock_end = time.monotonic()
        for pid in self.pids:
            try:
                proc = psutil.Process(pid)
                self.cpu_times_end[pid] = proc.cpu_times()
            except psutil.NoSuchProcess:
                self.cpu_times_end[pid] = None

    def get_stats(self) -> CPUMetrics:
        """Computes and returns CPU metrics aggregated across the tracked PIDs.

        Returns:
            The aggregated user and system CPU times along with their
            utilization percentages over the elapsed wall-clock interval.

        Raises:
            RuntimeError: If the context manager was not used, or if elapsed
                time is non-positive.
        """
        if not self.clock_start or not self.clock_end:
            raise RuntimeError(
                "Must use CPUMetricsCollector as a context manager before calling get_stats"
            )

        user = 0.0
        system = 0.0
        elapsed = self.clock_end - self.clock_start
        if elapsed <= 0:
            raise RuntimeError("Elapsed time must be positive")

        for pid in self.pids:
            if self.cpu_times_start[pid] and self.cpu_times_end[pid]:
                user += (
                    self.cpu_times_end[pid].user
                    - self.cpu_times_start[pid].user
                )
                system += (
                    self.cpu_times_end[pid].system
                    - self.cpu_times_start[pid].system
                )
        return CPUMetrics(
            user=user,
            user_percent=(user / elapsed) * 100,
            system=system,
            system_percent=(system / elapsed) * 100,
            elapsed=elapsed,
        )
