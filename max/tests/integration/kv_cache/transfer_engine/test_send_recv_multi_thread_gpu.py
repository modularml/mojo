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
from queue import Queue
from threading import Thread

import numpy as np
from _transfer_engine_helpers import kv_memory
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)


def test_send_recv_basic() -> None:
    # Queues for communication between threads
    sender_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    receiver_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    transfer_queue: Queue[TransferReqData] = Queue()

    # Transfer parameters
    total_num_pages = 3
    src_idxs = [2, 2]
    dst_idxs = [1, 0]
    max_wait_time_s = 10

    # Exit codes
    exit_codes = [-1, -1]

    def transfer_routine_sender() -> None:
        device = Accelerator()

        blocks_np = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
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
        transfer_req = engine.initiate_send_transfer(
            remote_md, src_idxs, dst_idxs, src_replica_idx=0, dst_replica_idx=0
        )
        transfer_queue.put(transfer_req)

        # Wait for transfer to complete, with a timeout
        start_time = time.time()
        is_done = False
        while not is_done and time.time() - start_time < max_wait_time_s:
            is_done = engine.is_complete(transfer_req)
            time.sleep(0.1)

        if not is_done:
            raise TimeoutError(
                f"Transfer did not complete within {max_wait_time_s} seconds"
            )
        engine.cleanup_transfer(transfer_req)

        # Verify results
        expected_blocks = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
        assert np.array_equal(blocks.to_numpy(), expected_blocks)

        # Release resources is skipped since it causes the following error:
        # `flush.c:58   UCX  ERROR req 0x7f274411a280: error during flush: Endpoint timeout`

        # TODO(E2EOPT-299) Reenable cleanup
        # engine.cleanup()

        exit_codes[0] = 0

    def transfer_routine_receiver() -> None:
        device = Accelerator()

        blocks_np = np.array(
            [80, 81, 82, 83, 84, 85, 86, 87, 88],
        )
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

        # Wait for transfer to complete, with a timeout
        transfer_req = transfer_queue.get()
        start_time = time.time()
        is_done = False
        while not is_done and time.time() - start_time < max_wait_time_s:
            is_done = engine.is_complete(transfer_req)
            time.sleep(0.1)

        if not is_done:
            raise TimeoutError(
                f"Transfer did not complete within {max_wait_time_s} seconds"
            )
        engine.cleanup_transfer(transfer_req)

        # Verify results
        expected_blocks = np.array(
            [16, 17, 18, 16, 17, 18, 86, 87, 88],
        )
        assert np.array_equal(blocks.to_numpy(), expected_blocks)

        # Release resources is skipped since it causes the following error:
        # `flush.c:58   UCX  ERROR req 0x7f274411a280: error during flush: Endpoint timeout`

        # TODO(E2EOPT-299) Reenable cleanup
        # engine.cleanup()

        exit_codes[1] = 0

    thread_1 = Thread(target=transfer_routine_sender)
    thread_2 = Thread(target=transfer_routine_receiver)

    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()

    assert exit_codes[0] == 0, "Sender thread failed"
    assert exit_codes[1] == 0, "Receiver thread failed"
