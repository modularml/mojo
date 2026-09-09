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
import queue
import time
from unittest import mock

import pytest
from max.serve.config import Settings
from max.serve.pipelines import telemetry_worker
from max.serve.telemetry import process_controller
from max.serve.telemetry.asyncio_controller import AsyncioMetricClient
from max.serve.telemetry.metrics import MaxMeasurement


@pytest.mark.asyncio
async def test_telemetry_worker() -> None:
    settings = Settings()
    async with telemetry_worker.start_process_consumer(settings) as worker:
        client = worker.Client()
        client.send_measurement(MaxMeasurement("foo", 1))
        client.send_measurement(MaxMeasurement("foo", 2))
        time.sleep(100e-3)
        with pytest.raises(queue.Empty):
            worker.queue.get_nowait()


def _raise_exception(x: MaxMeasurement) -> None:
    """TelemetryFn, but always broken. Only used for tests"""
    raise Exception("I'm always broken")


@pytest.mark.asyncio
async def test_unreliable_handle() -> None:
    settings = Settings()
    async with telemetry_worker.start_process_consumer(
        settings,
        handle_fn=_raise_exception,
    ) as worker:
        client = worker.Client()

        client.send_measurement(MaxMeasurement("foo", 1))
        client.send_measurement(MaxMeasurement("foo", 2))
        client.send_measurement(MaxMeasurement("foo", 3))

        await asyncio.sleep(1)

        with pytest.raises(queue.Empty):
            worker.queue.get_nowait()


@pytest.mark.asyncio
async def test_metric_asyncio_client_emits() -> None:
    q = mock.MagicMock()
    client = AsyncioMetricClient(q)

    # Every measurement is enqueued; there is no level-based dropping.
    client.send_measurement(MaxMeasurement("foo", 1))
    client.send_measurement(MaxMeasurement("foo", 2))
    assert q.put_nowait.call_count == 2


@pytest.mark.asyncio
async def test_metric_process_client_flushes_immediately() -> None:
    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(q)

    # Outside a transaction each measurement is flushed as its own packet.
    client.send_measurement(MaxMeasurement("foo", 1))
    client.send_measurement(MaxMeasurement("foo", 2))
    assert q.put_nowait.call_count == 2
    assert [len(call.args[0]) for call in q.put_nowait.call_args_list] == [1, 1]


@pytest.mark.asyncio
async def test_metric_process_client_transaction_batches() -> None:
    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(q)

    # Inside a transaction, measurements accumulate and flush as one packet
    # when the transaction closes.
    with client.transaction():
        client.send_measurement(MaxMeasurement("foo", 1))
        client.send_measurement(MaxMeasurement("foo", 2))
        client.send_measurement(MaxMeasurement("foo", 3))
        assert q.put_nowait.call_count == 0

    assert q.put_nowait.call_count == 1
    (batch,) = q.put_nowait.call_args.args
    assert [m.value for m in batch] == [1, 2, 3]


@pytest.mark.asyncio
async def test_metric_process_client_nested_transaction_flushes_once() -> None:
    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(q)

    # Only the outermost transaction flushes; a nested transaction must not.
    with client.transaction():
        client.send_measurement(MaxMeasurement("foo", 1))
        with client.transaction():
            client.send_measurement(MaxMeasurement("foo", 2))
            assert q.put_nowait.call_count == 0
        # Still inside the outer transaction: nothing flushed yet.
        assert q.put_nowait.call_count == 0
        client.send_measurement(MaxMeasurement("foo", 3))

    assert q.put_nowait.call_count == 1
    (batch,) = q.put_nowait.call_args.args
    assert [m.value for m in batch] == [1, 2, 3]


@pytest.mark.asyncio
async def test_metric_process_client_transaction_flushes_on_exception() -> None:
    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(q)

    # A body that raises partway through still flushes the already-buffered
    # measurements as one packet (via the finally) and leaves the transaction
    # state clean for the next transaction.
    with pytest.raises(RuntimeError, match="boom"):
        with client.transaction():
            client.send_measurement(MaxMeasurement("foo", 1))
            client.send_measurement(MaxMeasurement("foo", 2))
            raise RuntimeError("boom")

    assert q.put_nowait.call_count == 1
    (batch,) = q.put_nowait.call_args.args
    assert [m.value for m in batch] == [1, 2]
    assert client.transaction_depth == 0
    assert client.transaction_buffer == []
    q = mock.MagicMock()
    controller = process_controller.ProcessTelemetryController(q)

    controller.close()

    assert q.mock_calls == [
        mock.call.cancel_join_thread(),
        mock.call.close(),
    ]
