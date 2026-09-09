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

"""Test DP=1, TP=1 with 32 concurrent in-flight transfers using GPU 0 and GPU 1.

The payload size is intentionally kept small (512MB) to ensure the test
completes in a reasonable time while still validating concurrent transfer logic.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
from _transfer_engine_helpers import kv_memory
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import KVTransferEngine, TransferReqData


def transfer_routine_sender(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
    GB: float,
) -> None:
    device = Accelerator(0)

    # Fill each page with a distinct value so scatter bugs are detectable.
    # Page i gets value (i + 1).
    page_size = total_bytes // total_num_pages
    blocks_np = np.empty(total_bytes, dtype=np.int8)
    for i in range(total_num_pages):
        blocks_np[i * page_size : (i + 1) * page_size] = i + 1
    blocks = Buffer.from_numpy(blocks_np).to(device)

    # Create engine (DP=1, TP=1)
    engine = KVTransferEngine(
        "engine_1",
        [[kv_memory(blocks, total_num_pages)]],
    )

    # Connect with peer
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    t0 = time.time()
    transfer_reqs: list[TransferReqData] = []

    for idx in range(total_num_pages):
        transfer_req = engine.initiate_send_transfer(
            remote_md, [idx], [idx], src_replica_idx=0, dst_replica_idx=0
        )
        transfer_queue.put(transfer_req)
        transfer_reqs.append(transfer_req)

    for transfer_req in transfer_reqs:
        engine.sync_and_release(transfer_req)

    t1 = time.time()
    bw = total_bytes / (t1 - t0) / GB
    ms = (t1 - t0) * 1000

    print(
        f"[SENDER] Transferring {total_bytes / GB:.2f} GB took {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Verify sender buffer is unchanged
    result = blocks.to_numpy()
    for i in range(total_num_pages):
        assert (result[i * page_size : (i + 1) * page_size] == i + 1).all(), (
            f"Sender page {i} was modified"
        )

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
) -> None:
    device = Accelerator(1)

    blocks_np = np.full(total_bytes, 99, dtype=np.int8)
    blocks = Buffer.from_numpy(blocks_np).to(device)

    # Create engine (DP=1, TP=1)
    engine = KVTransferEngine(
        "engine_2",
        [[kv_memory(blocks, total_num_pages)]],
    )

    # Connect with peer
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    for _ in range(total_num_pages):
        transfer_req = transfer_queue.get()
        engine.sync_and_release(transfer_req)

    # Verify page-level correctness. Each transfer sends page idx → idx,
    # so receiver page i should equal sender page value (i + 1).
    result = blocks.to_numpy()
    page_size = total_bytes // total_num_pages
    for i in range(total_num_pages):
        expected = i + 1
        assert (
            result[i * page_size : (i + 1) * page_size] == expected
        ).all(), f"Receiver page {i} expected value {expected}"

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_send_recv_basic() -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    transfer_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    sender_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    # Transfer parameters
    GB = 1024 * 1024 * 1024
    total_bytes = (
        512 * 1024 * 1024
    )  # 512MB - reduced from 6GB for faster CI runs
    total_num_pages = 32

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue,
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
            transfer_queue,
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
