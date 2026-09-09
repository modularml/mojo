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

from queue import Queue
from threading import Thread

import numpy as np
import pytest
from _transfer_engine_helpers import kv_group
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)

total_num_pages = 10


def transfer_routine_sender(
    sender_md_queue: Queue[KVTransferEngineMetadata],
    receiver_md_queue: Queue[KVTransferEngineMetadata],
    transfer_queue_0: Queue[TransferReqData],
    transfer_queue_1: Queue[TransferReqData],
    sender_done_queue: Queue[None],
    receiver_done_queue: Queue[None],
) -> None:
    device_0 = Accelerator(0)
    device_1 = Accelerator(1)

    t0 = np.arange(100, dtype=np.float32)
    t1 = np.arange(100, dtype=np.float32) + 1000
    tensors_1 = [
        Buffer.from_numpy(t0).to(device_0),
        Buffer.from_numpy(t1).to(device_1),
    ]

    # DP=1, TP=2 (2 GPUs in one replica)
    engine_1 = KVTransferEngine(
        "engine_1",
        [kv_group(tensors_1, total_num_pages)],
    )

    sender_md_queue.put(engine_1.metadata)
    remote_md = receiver_md_queue.get()
    engine_1.connect(remote_md)

    transfer_0 = engine_1.initiate_send_transfer(
        remote_md,
        src_idxs=[0],
        dst_idxs=[0],
        src_replica_idx=0,
        dst_replica_idx=0,
    )
    transfer_queue_0.put(transfer_0)

    transfer_1 = engine_1.initiate_send_transfer(
        remote_md,
        src_idxs=[3, 4],
        dst_idxs=[3, 4],
        src_replica_idx=0,
        dst_replica_idx=0,
    )
    transfer_queue_1.put(transfer_1)

    engine_1.sync_and_release(transfer_0)
    engine_1.sync_and_release(transfer_1)

    assert np.array_equal(
        tensors_1[0].to_numpy(), np.arange(100, dtype=np.float32)
    )
    assert np.array_equal(
        tensors_1[1].to_numpy(), np.arange(100, dtype=np.float32) + 1000
    )

    sender_done_queue.put(None)
    receiver_done_queue.get()

    engine_1.cleanup()


def transfer_routine_receiver(
    sender_md_queue: Queue[KVTransferEngineMetadata],
    receiver_md_queue: Queue[KVTransferEngineMetadata],
    transfer_queue_0: Queue[TransferReqData],
    transfer_queue_1: Queue[TransferReqData],
    sender_done_queue: Queue[None],
    receiver_done_queue: Queue[None],
) -> None:
    device_2 = Accelerator(2)
    device_3 = Accelerator(3)

    t0 = np.zeros((100,), dtype=np.float32)
    t1 = np.zeros((100,), dtype=np.float32)
    tensors_2 = [
        Buffer.from_numpy(t0).to(device_2),
        Buffer.from_numpy(t1).to(device_3),
    ]

    # DP=1, TP=2 (2 GPUs in one replica)
    engine_2 = KVTransferEngine(
        "engine_2",
        [kv_group(tensors_2, total_num_pages)],
    )

    receiver_md_queue.put(engine_2.metadata)
    remote_md = sender_md_queue.get()
    engine_2.connect(remote_md)

    transfer_0 = transfer_queue_0.get()
    engine_2.sync_and_release(transfer_0)

    transfer_1 = transfer_queue_1.get()
    engine_2.sync_and_release(transfer_1)

    assert np.array_equal(
        tensors_2[0].to_numpy()[:10], np.arange(10, dtype=np.float32)
    ), f"Expected arange(10) in first page, got {tensors_2[0].to_numpy()[:10]}"
    assert np.array_equal(
        tensors_2[1].to_numpy()[:10], np.arange(10, dtype=np.float32) + 1000
    ), (
        f"Expected arange(10)+1000 in first page, got {tensors_2[1].to_numpy()[:10]}"
    )

    elts_per_page = tensors_2[0].num_elements // total_num_pages
    expected_0 = np.arange(100, dtype=np.float32)[
        3 * elts_per_page : 5 * elts_per_page
    ]
    result_0 = tensors_2[0].to_numpy()[3 * elts_per_page : 5 * elts_per_page]
    assert np.array_equal(result_0, expected_0), (
        f"Expected {expected_0} for tensor 0 pages 3-4, got {result_0}"
    )

    expected_1 = (
        np.arange(100, dtype=np.float32)[3 * elts_per_page : 5 * elts_per_page]
        + 1000
    )
    result_1 = tensors_2[1].to_numpy()[3 * elts_per_page : 5 * elts_per_page]
    assert np.array_equal(result_1, expected_1), (
        f"Expected {expected_1} for tensor 1 pages 3-4, got {result_1}"
    )

    receiver_done_queue.put(None)
    sender_done_queue.get()

    engine_2.cleanup()


@pytest.mark.skip(reason="SERVOPT-872: Reenable this test")
def test_multi_tensor_transfer_threaded() -> None:
    """Test transfer between multiple tensors using threading."""
    sender_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    receiver_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    transfer_queue_0: Queue[TransferReqData] = Queue()
    transfer_queue_1: Queue[TransferReqData] = Queue()
    sender_done_queue: Queue[None] = Queue()
    receiver_done_queue: Queue[None] = Queue()

    sender_thread = Thread(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
        ),
    )
    receiver_thread = Thread(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
        ),
    )

    sender_thread.start()
    receiver_thread.start()

    sender_thread.join()
    receiver_thread.join()

    print("\n" + "=" * 80)
    print("Multi-tensor threading test completed successfully!")
    print("=" * 80)
