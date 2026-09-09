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

"""Test DP=2, TP=2 transfers using GPUs.

This test validates that transfer engine correctly handles:

- Multiple DP replicas (DP=2)
- Multiple TP shards per replica (TP=2)
- Per-replica transfers with src_replica_idx and dst_replica_idx parameters
"""

import multiprocessing as mp
import time

import numpy as np
from _transfer_engine_helpers import kv_group
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import KVTransferEngine


def paged(
    total_bytes: int,
    page_values: list[int],
    accelerator_idx: int,
) -> Buffer:
    """Create a buffer with distinct values per page."""
    total_num_pages = len(page_values)
    page_size = total_bytes // total_num_pages
    arr = np.empty(total_bytes, dtype=np.int8)
    for i, v in enumerate(page_values):
        arr[i * page_size : (i + 1) * page_size] = v
    return Buffer.from_numpy(arr).to(Accelerator(accelerator_idx))


def transfer_routine_sender(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue_0: mp.Queue,  # type: ignore[type-arg]
    transfer_queue_1: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
    GB: float,
) -> None:
    """Sender routine for DP=2, TP=2 transfer."""
    # DP=2, TP=2: 4 GPUs total for sender
    # Replica 0: GPU 0, 1
    # Replica 1: GPU 2, 3
    # Each shard gets distinct per-page values so intra-shard scatter bugs are detectable.
    replica_0_tensors = [
        paged(total_bytes, page_values=[10, 11], accelerator_idx=0),
        paged(total_bytes, page_values=[12, 13], accelerator_idx=1),
    ]
    replica_1_tensors = [
        paged(total_bytes, page_values=[20, 21], accelerator_idx=2),
        paged(total_bytes, page_values=[22, 23], accelerator_idx=3),
    ]

    # Create engine with DP=2, TP=2
    engine = KVTransferEngine(
        "sender_engine",
        [
            kv_group(replica_0_tensors, total_num_pages),
            kv_group(replica_1_tensors, total_num_pages),
        ],
    )

    # Connect with receiver
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Transfer from replica 0
    start_time_0 = time.time()
    transfer_req_0 = engine.initiate_send_transfer(
        remote_md,
        src_idxs=[0, 1],
        dst_idxs=[0, 1],
        src_replica_idx=1,
        dst_replica_idx=0,
    )
    transfer_queue_0.put(transfer_req_0)
    engine.sync_and_release(transfer_req_0)
    end_time_0 = time.time()

    # Transfer from replica 1
    start_time_1 = time.time()
    transfer_req_1 = engine.initiate_send_transfer(
        remote_md,
        src_idxs=[0, 1],
        dst_idxs=[0, 1],
        src_replica_idx=0,
        dst_replica_idx=1,
    )
    transfer_queue_1.put(transfer_req_1)
    engine.sync_and_release(transfer_req_1)
    end_time_1 = time.time()

    # Calculate bandwidth for each replica
    total_bytes_transferred = total_bytes * 2  # 2 TP shards

    bw_0 = total_bytes_transferred / (end_time_0 - start_time_0) / GB
    bw_1 = total_bytes_transferred / (end_time_1 - start_time_1) / GB

    print(
        f"[Sender] Replica 0 -> Replica 0: {total_bytes_transferred / GB:.4f} GB "
        f"in {(end_time_0 - start_time_0) * 1000:.2f} ms ({bw_0:.2f} GB/s)"
    )
    print(
        f"[Sender] Replica 0 -> Replica 1: {total_bytes_transferred / GB:.4f} GB "
        f"in {(end_time_1 - start_time_1) * 1000:.2f} ms ({bw_1:.2f} GB/s)"
    )

    # Verify bandwidth is reasonable
    assert bw_0 > 1.0, f"Replica 0 transfer too slow: {bw_0:.2f} GB/s"
    assert bw_1 > 1.0, f"Replica 1 transfer too slow: {bw_1:.2f} GB/s"

    # Verify sender buffers are unchanged
    page_size = total_bytes // total_num_pages
    for shard, page_values in zip(
        replica_0_tensors, [[10, 11], [12, 13]], strict=False
    ):
        result = shard.to_numpy()
        for i, v in enumerate(page_values):
            assert (result[i * page_size : (i + 1) * page_size] == v).all()
    for shard, page_values in zip(
        replica_1_tensors, [[20, 21], [22, 23]], strict=False
    ):
        result = shard.to_numpy()
        for i, v in enumerate(page_values):
            assert (result[i * page_size : (i + 1) * page_size] == v).all()

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue_0: mp.Queue,  # type: ignore[type-arg]
    transfer_queue_1: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Receiver routine for DP=2, TP=2 transfer."""
    # DP=2, TP=2: 4 GPUs total for receiver
    # Replica 0: GPU 1, 3
    # Replica 1: GPU 2, 2
    replica_0_tensors = [
        paged(total_bytes, page_values=[99, 99], accelerator_idx=1),
        paged(total_bytes, page_values=[99, 99], accelerator_idx=3),
    ]
    replica_1_tensors = [
        paged(total_bytes, page_values=[99, 99], accelerator_idx=2),
        paged(total_bytes, page_values=[99, 99], accelerator_idx=2),
    ]

    # Create engine with DP=2, TP=2
    engine = KVTransferEngine(
        "receiver_engine",
        [
            kv_group(replica_0_tensors, total_num_pages),
            kv_group(replica_1_tensors, total_num_pages),
        ],
    )

    # Connect with sender
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Receive transfer for replica 0
    transfer_req_0 = transfer_queue_0.get()
    engine.sync_and_release(transfer_req_0)

    # Receive transfer for replica 1
    transfer_req_1 = transfer_queue_1.get()
    engine.sync_and_release(transfer_req_1)

    # Verify received data at page level.
    # transfer_req_0: sender replica 1 -> receiver replica 0
    #   shard 0 (page_values=[20,21]) -> replica_0_tensors[0]
    #   shard 1 (page_values=[22,23]) -> replica_0_tensors[1]
    # transfer_req_1: sender replica 0 -> receiver replica 1
    #   shard 0 (page_values=[10,11]) -> replica_1_tensors[0]
    #   shard 1 (page_values=[12,13]) -> replica_1_tensors[1]
    page_size = total_bytes // total_num_pages
    for shard, page_values in zip(
        replica_0_tensors, [[20, 21], [22, 23]], strict=False
    ):
        result = shard.to_numpy()
        for i, v in enumerate(page_values):
            assert (result[i * page_size : (i + 1) * page_size] == v).all()
    for shard, page_values in zip(
        replica_1_tensors, [[10, 11], [12, 13]], strict=False
    ):
        result = shard.to_numpy()
        for i, v in enumerate(page_values):
            assert (result[i * page_size : (i + 1) * page_size] == v).all()

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_dp2_tp2_transfer_multiprocessing() -> None:
    """Test DP=2, TP=2 transfer using 8 GPUs (4 for sender, 4 for receiver).

    This test validates:
    - Engine construction with DP=2, TP=2
    - Metadata structure for 2D tensor layout
    - Per-replica transfers using src_replica_idx and dst_replica_idx
    - Correct data transfer for each replica independently
    """
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    transfer_queue_0: mp.Queue = ctx.Queue()  # type: ignore[type-arg]  # For replica 0
    transfer_queue_1: mp.Queue = ctx.Queue()  # type: ignore[type-arg]  # For replica 1
    sender_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    GB = 1024 * 1024 * 1024
    total_bytes = int(0.5 * GB)
    total_num_pages = 2

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            total_bytes,
            GB,
        ),
    )
    receiver_proc = ctx.Process(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            total_bytes,
        ),
    )

    sender_proc.start()
    receiver_proc.start()

    sender_proc.join()
    receiver_proc.join()

    assert sender_proc.exitcode == 0, (
        f"Sender process failed with exit code {sender_proc.exitcode}"
    )
    assert receiver_proc.exitcode == 0, (
        f"Receiver process failed with exit code {receiver_proc.exitcode}"
    )
