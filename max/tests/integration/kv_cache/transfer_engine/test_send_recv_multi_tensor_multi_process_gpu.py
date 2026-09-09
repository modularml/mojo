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

import multiprocessing as mp
import time

import numpy as np
import pytest
from _transfer_engine_helpers import kv_group
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
    total_bytes: int,
    src_idxs: list[int],
    dst_idxs: list[int],
    GB: float,
) -> None:
    """Multiprocessing sender routine for multi-tensor transfer."""
    device_0 = Accelerator(0)
    device_1 = Accelerator(1)

    # Create large tensors with distinct values for each tensor
    tensor_0_data = np.full(total_bytes, 42, dtype=np.int8)
    tensor_1_data = np.full(total_bytes, 84, dtype=np.int8)
    tensors_1 = [
        Buffer.from_numpy(tensor_0_data).to(device_0),
        Buffer.from_numpy(tensor_1_data).to(device_1),
    ]

    # DP=1, TP=2
    engine_1 = KVTransferEngine(
        "engine_1",
        [kv_group(tensors_1, total_num_pages)],
    )

    sender_md_queue.put(engine_1.metadata)
    remote_md = receiver_md_queue.get()
    engine_1.connect(remote_md)

    # Perform transfers
    start_time = time.time()
    transfer_req = engine_1.initiate_send_transfer(
        remote_md, src_idxs, dst_idxs, src_replica_idx=0, dst_replica_idx=0
    )
    transfer_queue.put(transfer_req)
    engine_1.sync_and_release(transfer_req)
    end_time = time.time()

    transfer_time = end_time - start_time
    # Calculate actual bytes transferred (both tensors for the specified pages)
    bytes_per_tensor = total_bytes * len(src_idxs) // total_num_pages
    total_bytes_transferred = bytes_per_tensor * len(tensors_1)
    bw = total_bytes_transferred / transfer_time / GB
    ms = transfer_time * 1000

    print(
        f"[Sender MP] Transferred {total_bytes_transferred / GB:.4f} GB in {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Check that the transfer speed is at least 1 GB/s
    # We found that CUDA_COPY yields ~.3GB/s while CUDA_IPC yields 100+GB/s
    assert bw > 1.0, f"Transfer speed is too low: {bw:.2f} GB/s"

    # Verify sender data unchanged
    assert np.array_equal(tensors_1[0].to_numpy(), tensor_0_data)
    assert np.array_equal(tensors_1[1].to_numpy(), tensor_1_data)

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine_1.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queue: mp.Queue,  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Multiprocessing receiver routine for multi-tensor transfer."""
    device_2 = Accelerator(2)
    device_3 = Accelerator(3)

    # Create large tensors initialized to different values
    tensor_0_data = np.full(total_bytes, 99, dtype=np.int8)
    tensor_1_data = np.full(total_bytes, 77, dtype=np.int8)
    tensors_2 = [
        Buffer.from_numpy(tensor_0_data).to(device_2),
        Buffer.from_numpy(tensor_1_data).to(device_3),
    ]

    # DP=1, TP=2
    engine_2 = KVTransferEngine(
        "engine_2",
        [kv_group(tensors_2, total_num_pages)],
    )

    receiver_md_queue.put(engine_2.metadata)
    remote_md = sender_md_queue.get()
    engine_2.connect(remote_md)

    transfer_req = transfer_queue.get()
    engine_2.sync_and_release(transfer_req)

    # Verify data received correctly - should now have sender values (42 and 84)
    expected_tensor_0 = np.full(total_bytes, 42, dtype=np.int8)
    expected_tensor_1 = np.full(total_bytes, 84, dtype=np.int8)

    assert np.array_equal(tensors_2[0].to_numpy(), expected_tensor_0), (
        f"Expected tensor 0 to have value 42, got {tensors_2[0].to_numpy()[:10]}"
    )
    assert np.array_equal(tensors_2[1].to_numpy(), expected_tensor_1), (
        f"Expected tensor 1 to have value 84, got {tensors_2[1].to_numpy()[:10]}"
    )

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine_2.cleanup()


def test_multi_tensor_transfer_multiprocessing(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Test transfer between multiple tensors using multiprocessing."""
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    transfer_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    sender_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

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
            total_bytes,
            src_idxs,
            dst_idxs,
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

    # Capture and display output from subprocesses
    out, _err = capfd.readouterr()

    # Display the bandwidth information
    with capfd.disabled():
        print("\n" + "=" * 80)
        print("Multi-tensor multiprocessing test completed successfully!")
        for line in out.split("\n"):
            if "[Sender MP]" in line:
                print(line)
        print("=" * 80)
