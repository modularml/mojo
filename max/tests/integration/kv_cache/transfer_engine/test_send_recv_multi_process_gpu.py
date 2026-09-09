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

"""Test DP=1, TP=1 basic send/recv using GPU 0 and GPU 1."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pytest
from _transfer_engine_helpers import kv_memory
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import KVTransferEngine


def transfer_routine_sender(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    src_idxs: list[int],
    dst_idxs: list[int],
    total_bytes: int,
    GB: float,
) -> None:
    # Enable UCX debug logging to verify CUDA IPC transport is selected.
    os.environ["UCX_LOG_LEVEL"] = "debug"

    device = Accelerator(1)

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
    transfer_req = engine.initiate_send_transfer(
        remote_md, src_idxs, dst_idxs, src_replica_idx=0, dst_replica_idx=0
    )
    transfer_queue.put(transfer_req)
    engine.sync_and_release(transfer_req)
    t1 = time.time()
    bw = total_bytes / (t1 - t0) / GB
    ms = (t1 - t0) * 1000

    # This print is consumed by capfd in the test. We can't serialize capfd
    # into the child process to call capfd.disabled() here.
    print(
        f"[Sender] Transferring {total_bytes / GB:.2f} GB took {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Check that the transfer speed is at least 1 GB/s.
    # Note: this alone does not guarantee CUDA IPC is used — d2h→h2d can also
    # exceed 1 GB/s. The cuMemcpyDtoDAsync_v2 log assertion in the test
    # is the authoritative check for transport selection.
    assert bw > 1.0, f"Transfer speed is too low: {bw:.2f} GB/s"

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
    src_idxs: list[int],
    dst_idxs: list[int],
    total_bytes: int,
) -> None:
    device = Accelerator(0)

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
    transfer_req = transfer_queue.get()
    engine.sync_and_release(transfer_req)

    # Verify page-level correctness.
    # Sender page i has value (i + 1). src_idxs[j] maps to dst_idxs[j], so
    # receiver page dst_idxs[j] should equal src_idxs[j] + 1.
    result = blocks.to_numpy()
    page_size = total_bytes // total_num_pages
    for src_idx, dst_idx in zip(src_idxs, dst_idxs, strict=True):
        expected = src_idx + 1
        assert (
            result[dst_idx * page_size : (dst_idx + 1) * page_size] == expected
        ).all(), (
            f"Receiver page {dst_idx} expected value {expected} (from src page {src_idx})"
        )

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_send_recv_basic(capfd: pytest.CaptureFixture[str]) -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    transfer_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    sender_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    # Transfer parameters
    GB = 1024 * 1024 * 1024
    total_bytes = int(1 * GB)
    total_num_pages = 2
    src_idxs = [0, 1]
    dst_idxs = [1, 0]

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            src_idxs,
            dst_idxs,
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
            src_idxs,
            dst_idxs,
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

    out, _err = capfd.readouterr()

    with capfd.disabled():
        print()
        print("-" * 80)
        for line in out.split("\n"):
            if "[Sender]" in line:
                print(line)
        print("-" * 80)

    # Verify CUDA IPC transport is selected. The bw > 1 GB/s check alone is
    # insufficient since d2h->h2d paths can also exceed 1 GB/s.
    assert "UCX  DEBUG cuMemcpyDtoDAsync_v2" in out
