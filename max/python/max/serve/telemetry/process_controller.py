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

import functools
import logging
import multiprocessing
import queue
import signal
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

import prometheus_client
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.process_control import subprocess_manager
from max.serve.telemetry.common import configure_metrics, configure_tracing
from max.serve.telemetry.metrics import (
    MaxMeasurement,
    MetricClient,
    TelemetryFn,
)
from uvicorn import Config, Server

logger = logging.getLogger("max.serve")


def _sync_commit(m: MaxMeasurement) -> None:
    m.commit()


class ProcessMetricClient(MetricClient):
    """Records measurements by shipping them to the telemetry worker process.

    ``transaction_buffer`` / ``transaction_depth`` are deliberately
    unsynchronized. This is safe because of a single-emitter invariant: the
    only process that opens transactions is the model worker, which emits
    metrics from one async event loop, and every transaction body (e.g.
    ``construct_batch`` / ``_publish_metrics``) is fully synchronous — no
    ``await`` inside — so no other measurement can interleave. Do not open a
    ``transaction()`` from a context where measurements are produced
    concurrently (e.g. the API server's per-request handlers) without first
    making this state thread-local; otherwise unrelated measurements would be
    swept into the batch and the buffer mutations would race.
    """

    def __init__(
        self,
        q: Queue[list[MaxMeasurement]],
    ) -> None:
        self.queue = q
        # Within a transaction, measurements accumulate here and are flushed as
        # a single packet when the transaction closes. Depth tracks nesting so
        # only the outermost transaction flushes.
        self.transaction_buffer: list[MaxMeasurement] = []
        self.transaction_depth = 0

    def send_measurement(self, m: MaxMeasurement) -> None:
        if self.transaction_depth > 0:
            self.transaction_buffer.append(m)
            return
        self._flush([m])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transaction_depth += 1
        try:
            yield
        finally:
            self.transaction_depth -= 1
            if self.transaction_depth == 0 and self.transaction_buffer:
                # Hand off a fresh list so the multiprocessing queue serializes
                # a stable snapshot; a subsequent send must not mutate it.
                batch = self.transaction_buffer
                self.transaction_buffer = []
                self._flush(batch)

    def _flush(self, payload: list[MaxMeasurement]) -> None:
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # we would rather lose data than slow the server
            logger.warning(
                f"Telemetry Queue is full.  Dropping {len(payload)} measurements"
            )
        except (ValueError, OSError):
            logger.debug("Telemetry Queue is closed.  Dropping data")

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_reconstruct_client, self.queue)

    def __del__(self) -> None:
        # Flush anything captured by a transaction that never closed (e.g. the
        # client was GC'd mid-block). Best-effort: never raise from __del__.
        if not self.transaction_buffer:
            return
        try:
            self._flush(self.transaction_buffer)
        except Exception:
            logger.debug("Telemetry Queue is unavailable.  Dropping data")
        finally:
            self.transaction_buffer = []


@asynccontextmanager
async def _reconstruct_client(
    q: Queue[list[MaxMeasurement]],
) -> AsyncGenerator[MetricClient, None]:
    yield ProcessMetricClient(q)


@dataclass
class ProcessTelemetryController:
    queue: Queue[list[MaxMeasurement]]

    def Client(self) -> MetricClient:
        return ProcessMetricClient(self.queue)

    def close(self) -> None:
        # Metrics are lossy. Do not let multiprocessing wait at interpreter
        # shutdown for a QueueFeederThread to flush to a worker that may already
        # be gone after a crash.
        self.queue.cancel_join_thread()
        self.queue.close()


@asynccontextmanager
async def start_process_consumer(
    settings: Settings, handle_fn: TelemetryFn | None = None
) -> AsyncGenerator[ProcessTelemetryController, None]:
    if handle_fn is None:
        handle_fn = _sync_commit

    mp = multiprocessing.get_context("spawn")
    async with subprocess_manager("Metrics Worker") as proc:
        metrics_q = mp.Queue()
        controller = ProcessTelemetryController(metrics_q)
        alive = mp.Event()

        try:
            proc.start(init_and_process, settings, metrics_q, alive, handle_fn)

            await proc.ready(alive, settings.telemetry_worker_spawn_timeout)

            if settings.use_heartbeat:
                proc.watch_heartbeat(alive, timeout=5)

            yield controller
        finally:
            controller.close()


def init_and_process(
    settings: Settings,
    metrics_q: Queue[list[MaxMeasurement]],
    alive: Event,
    commit_fn: TelemetryFn,
) -> None:
    """Initialize logging & metrics, and start the metrics server if enabled. This is expected to run from the Telemetry process."""
    configure_metrics(settings)
    configure_tracing(settings)

    if (
        not settings.disable_telemetry
        and settings.metric_recording == MetricRecordingMethod.PROCESS
    ):
        app = prometheus_client.make_asgi_app()
        config = Config(
            app=app,
            host=settings.host,
            port=settings.metrics_port,
            access_log=False,
            log_level="warning",
        )
        server = Server(config)

        def run_server() -> None:
            logger.warning(
                f"Starting ASGI metrics server on port {settings.metrics_port}"
            )
            try:
                server.run()
            except Exception:
                logger.exception("Error running ASGI metrics server")

        # Start the server in a daemon thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    return process_telemetry(metrics_q, alive, commit_fn)


def process_telemetry(
    metrics_q: Queue[list[MaxMeasurement]],
    alive: Event,
    commit_fn: TelemetryFn,
) -> None:
    """Long running function to read from a queue & process each element"""
    should_exit = False

    def signal_handler(*_args: Any) -> None:
        nonlocal should_exit
        should_exit = True

    # Maybe shocking, but SIGINT / SIGTERM do NOT interrupt queue.get()
    # So we need our own signal handler to avoid deadlock bugs on shutdown
    # Eventually, want to pivot to asyncio queue apis and drop these hacks
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not should_exit:
            alive.set()

            try:
                ms = metrics_q.get(block=True, timeout=100e-3)
            except queue.Empty:
                continue

            try:
                for m in ms:
                    commit_fn(m)
            except:
                logger.exception("Error processing telemetry")
    except KeyboardInterrupt:
        pass
