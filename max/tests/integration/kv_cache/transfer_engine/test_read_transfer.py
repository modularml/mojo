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

# NIXL transfers can fail when multiple transfers are in the same bazel pytest target.
# This appears as a invalid file descriptor error.
# As such, this transfer test is alone in this file.

import platform
from queue import Queue
from threading import Thread

import numpy as np
import pytest
from _transfer_engine_helpers import kv_memory
from max.driver import CPU, Device
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)


def _require_supported_transfer_engine_platform() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        pytest.skip(
            "TransferEngine integration tests require Linux x86_64 with UCX support."
        )


def transfer_routine_reader(
    engine: KVTransferEngine,
    remote: KVTransferEngineMetadata,
    queue: Queue,  # type: ignore[type-arg]
    src_idxs: list[int],
    dst_idxs: list[int],
    src_replica_idx: int,
    dst_replica_idx: int,
) -> None:
    transfer_req = engine.initiate_read_transfer(
        remote, src_idxs, dst_idxs, src_replica_idx, dst_replica_idx
    )
    queue.put(transfer_req)
    engine.sync_and_release(transfer_req)


def transfer_routine_receiver(engine: KVTransferEngine, queue: Queue) -> None:  # type: ignore[type-arg]
    transfer_req = queue.get()
    engine.sync_and_release(transfer_req)


@pytest.mark.parametrize("device", [CPU()])
def test_read_transfer_completion_from_both_engines(device: Device) -> None:
    _require_supported_transfer_engine_platform()

    total_num_pages = 3
    elts_per_page = 3
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 10).to(
        device
    )
    blocks_2 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 80).to(
        device
    )

    engine_1 = KVTransferEngine(
        "engine_1",
        [[kv_memory(blocks_1, total_num_pages)]],
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [[kv_memory(blocks_2, total_num_pages)]],
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    queue: Queue[TransferReqData] = Queue()
    src_idxs = [2, 2]
    dst_idxs = [1, 0]
    src_replica_idx = 0
    dst_replica_idx = 0
    reader = Thread(
        target=transfer_routine_reader,
        args=(
            engine_2,
            engine_1.metadata,
            queue,
            src_idxs,
            dst_idxs,
            src_replica_idx,
            dst_replica_idx,
        ),
    )
    writer = Thread(target=transfer_routine_receiver, args=(engine_1, queue))

    reader.start()
    writer.start()

    reader.join()
    writer.join()

    expected_blocks_1 = np.array(
        [10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    expected_blocks_2 = np.array(
        [16, 17, 18, 16, 17, 18, 86, 87, 88],
    )
    assert np.array_equal(blocks_1.to_numpy(), expected_blocks_1)
    assert np.array_equal(blocks_2.to_numpy(), expected_blocks_2)

    engine_2.cleanup()
    engine_1.cleanup()
