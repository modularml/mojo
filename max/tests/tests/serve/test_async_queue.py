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

import time
from unittest import mock

import pytest
from max.serve.telemetry.asyncio_controller import (
    AsyncioTelemetryController,
    NotStarted,
)
from max.serve.telemetry.metrics import MaxMeasurement


@pytest.mark.asyncio
async def test_basic_usage() -> None:
    spy = mock.Mock(spec=MaxMeasurement)
    spy2 = mock.Mock(spec=MaxMeasurement)

    atc = AsyncioTelemetryController()

    # Getting a client for a non-started controller should not work
    # we need to be able to serialize clients & still connect
    with pytest.raises(NotStarted):
        atc.Client()

    assert not spy2.commit.called

    # Consumer is running.  call() works
    async with atc:
        client = atc.Client()
        client.send_measurement(spy)
    assert spy.commit.called

    # Consumer is stopped.  call() doesn't work
    with pytest.raises(NotStarted):
        atc.Client().send_measurement(spy2)
    assert not spy2.commit.called


@pytest.mark.asyncio
async def test_fast() -> None:
    """Queuing a metric measurement should be fast"""
    async with AsyncioTelemetryController() as atc:
        client = atc.Client()
        start = time.perf_counter()
        client.send_measurement(MaxMeasurement("maxserve.request_count", 1))
        duration = time.perf_counter() - start
        assert duration < 1e-3


@pytest.mark.asyncio
async def test_shutdown() -> None:
    spy = mock.Mock(spec=MaxMeasurement)
    N = 10
    async with AsyncioTelemetryController() as atc:
        client = atc.Client()
        for i in range(N):  # noqa: B007
            client.send_measurement(spy)
        # we haven't waited long enough for everything to run
        assert spy.commit.call_count < N
    # shutdown should burn through the queue
    assert spy.commit.call_count == N
