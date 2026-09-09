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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from max.serve.config import MetricRecordingMethod, Settings
from max.serve.telemetry.asyncio_controller import start_asyncio_consumer
from max.serve.telemetry.metrics import MetricClient, NoopClient, SyncClient
from max.serve.telemetry.process_controller import start_process_consumer


@asynccontextmanager
async def start_telemetry_consumer(
    settings: Settings,
) -> AsyncGenerator[MetricClient, None]:
    method = settings.metric_recording
    if method == MetricRecordingMethod.NOOP:
        yield NoopClient()

    elif method == MetricRecordingMethod.SYNC:
        yield SyncClient()

    elif method == MetricRecordingMethod.ASYNCIO:
        async with start_asyncio_consumer() as controller:
            yield controller

    elif method == MetricRecordingMethod.PROCESS:
        async with start_process_consumer(settings) as controller:
            yield controller.Client()

    else:
        raise Exception(f"Unrecognized metric_recording: {method}")
