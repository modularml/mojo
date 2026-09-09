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

"""Hermetic KVTransferEngine tests over host DRAM (no GPU).

These run real NIXL transfers — agent creation, metadata exchange, memory
registration, xfer-request creation, completion notifications, teardown —
through the CPU-flavor UCX plugin, so they work on CPU-only CI workers and
give presubmit red/green signal for transfer-engine changes without GPU
hardware or cluster capacity.

Each test creates two engines in one process and moves bytes over UCX's
same-host transports (tcp/shm). Content is sentinel-validated on the
receiver, so a descriptor-layout or page-addressing bug fails loudly.

NIXL transfers can fail when multiple transfers share one process (invalid
file descriptor errors — see test_send_recv.py). The BUILD target sets
``shard_count`` so each test runs in its own process.
"""

from __future__ import annotations

import os
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pytest
from _transfer_engine_helpers import kv_memory
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)


def _configure_cpu_plugin_dir() -> None:
    """Point NIXL_PLUGIN_DIR at the CPU-flavor UCX plugin in the runfiles.

    The upstream nixlPluginManager is a singleton that reads NIXL_PLUGIN_DIR
    once at first agent creation, and only one plugin directory can be active.
    This test is only meaningful against the CPU flavor (the CUDA/ROCm flavors
    cannot dlopen without their GPU driver stacks), so the variable is set
    unconditionally before any engine is constructed.
    """
    test_srcdir = os.environ.get("TEST_SRCDIR")
    if not test_srcdir:
        pytest.skip("TEST_SRCDIR not set; this test must run under bazel test")
    plugin_dir = Path(test_srcdir) / "+http_archive+nixl_upstream" / "cpu"
    if not (plugin_dir / "libplugin_UCX.so").exists():
        raise RuntimeError(
            f"CPU-flavor UCX plugin not found at {plugin_dir}; check the "
            "@nixl_upstream//:cpu/libplugin_UCX.so data dependency"
        )
    os.environ["NIXL_PLUGIN_DIR"] = str(plugin_dir)


_configure_cpu_plugin_dir()


def _sender_routine(
    engine: KVTransferEngine,
    remote: KVTransferEngineMetadata,
    queue: Queue[TransferReqData],
    src_idxs: list[int],
    dst_idxs: list[int],
) -> None:
    transfer_req = engine.initiate_send_transfer(
        remote, src_idxs, dst_idxs, src_replica_idx=0, dst_replica_idx=0
    )
    queue.put(transfer_req)
    engine.sync_and_release(transfer_req)


def _reader_routine(
    engine: KVTransferEngine,
    remote: KVTransferEngineMetadata,
    queue: Queue[TransferReqData],
    src_idxs: list[int],
    dst_idxs: list[int],
) -> None:
    transfer_req = engine.initiate_read_transfer(
        remote, src_idxs, dst_idxs, src_replica_idx=0, dst_replica_idx=0
    )
    queue.put(transfer_req)
    engine.sync_and_release(transfer_req)


def _peer_routine(
    engine: KVTransferEngine, queue: Queue[TransferReqData]
) -> None:
    transfer_req = queue.get()
    engine.sync_and_release(transfer_req)


def _run_transfer_pair(
    initiator: Thread,
    peer: Thread,
) -> None:
    """Run both sides concurrently; a single thread can deadlock because both
    engines must poll for the transfer to progress (see is_complete docs)."""
    initiator.start()
    peer.start()
    initiator.join()
    peer.join()


def test_send_recv_dram() -> None:
    total_num_pages = 3
    elts_per_page = 3
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 10)
    blocks_2 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 80)

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
    sender = Thread(
        target=_sender_routine,
        args=(engine_1, engine_2.metadata, queue, [2, 2], [1, 0]),
    )
    receiver = Thread(target=_peer_routine, args=(engine_2, queue))
    _run_transfer_pair(sender, receiver)

    expected_blocks_1 = np.arange(num_elts, dtype=np.int16) + 10
    expected_blocks_2 = np.array(
        [16, 17, 18, 16, 17, 18, 86, 87, 88], dtype=np.int16
    )
    assert np.array_equal(blocks_1.to_numpy(), expected_blocks_1)
    assert np.array_equal(blocks_2.to_numpy(), expected_blocks_2)

    engine_2.cleanup()
    engine_1.cleanup()


def test_read_transfer_dram() -> None:
    total_num_pages = 3
    elts_per_page = 3
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 10)
    blocks_2 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 80)

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

    # engine_2 pulls pages [2, 2] of engine_1 into its pages [1, 0].
    queue: Queue[TransferReqData] = Queue()
    reader = Thread(
        target=_reader_routine,
        args=(engine_2, engine_1.metadata, queue, [2, 2], [1, 0]),
    )
    source = Thread(target=_peer_routine, args=(engine_1, queue))
    _run_transfer_pair(reader, source)

    expected_blocks_1 = np.arange(num_elts, dtype=np.int16) + 10
    expected_blocks_2 = np.array(
        [16, 17, 18, 16, 17, 18, 86, 87, 88], dtype=np.int16
    )
    assert np.array_equal(blocks_1.to_numpy(), expected_blocks_1)
    assert np.array_equal(blocks_2.to_numpy(), expected_blocks_2)

    engine_2.cleanup()
    engine_1.cleanup()


def test_send_recv_multi_group_dram() -> None:
    """Transfer with an extra tensor group (the spec-decode draft-KV shape).

    The main and extra groups have different bytes_per_page, mirroring a
    heterogeneous target+draft KV cache (e.g. 61-layer MLA target vs 1-layer
    Eagle draft). Both groups share page indices; each group's pages must
    land at that group's own base address and stride on the receiver.
    """
    total_num_pages = 4
    main_elts_per_page = 8
    draft_elts_per_page = 2
    main_elts = total_num_pages * main_elts_per_page
    draft_elts = total_num_pages * draft_elts_per_page

    main_1 = Buffer.from_numpy(np.arange(main_elts, dtype=np.int16) + 100)
    draft_1 = Buffer.from_numpy(np.arange(draft_elts, dtype=np.int16) + 500)
    main_2 = Buffer.from_numpy(np.zeros(main_elts, dtype=np.int16))
    draft_2 = Buffer.from_numpy(np.zeros(draft_elts, dtype=np.int16))

    engine_1 = KVTransferEngine(
        "engine_1",
        [
            [
                kv_memory(main_1, total_num_pages),
                kv_memory(draft_1, total_num_pages),
            ]
        ],
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [
            [
                kv_memory(main_2, total_num_pages),
                kv_memory(draft_2, total_num_pages),
            ]
        ],
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    src_idxs = [1, 3]
    dst_idxs = [2, 0]
    queue: Queue[TransferReqData] = Queue()
    sender = Thread(
        target=_sender_routine,
        args=(engine_1, engine_2.metadata, queue, src_idxs, dst_idxs),
    )
    receiver = Thread(target=_peer_routine, args=(engine_2, queue))
    _run_transfer_pair(sender, receiver)

    main_2_np = main_2.to_numpy()
    draft_2_np = draft_2.to_numpy()
    main_1_np = np.arange(main_elts, dtype=np.int16) + 100
    draft_1_np = np.arange(draft_elts, dtype=np.int16) + 500
    for src_idx, dst_idx in zip(src_idxs, dst_idxs, strict=True):
        assert np.array_equal(
            main_2_np[
                dst_idx * main_elts_per_page : (dst_idx + 1)
                * main_elts_per_page
            ],
            main_1_np[
                src_idx * main_elts_per_page : (src_idx + 1)
                * main_elts_per_page
            ],
        )
        assert np.array_equal(
            draft_2_np[
                dst_idx * draft_elts_per_page : (dst_idx + 1)
                * draft_elts_per_page
            ],
            draft_1_np[
                src_idx * draft_elts_per_page : (src_idx + 1)
                * draft_elts_per_page
            ],
        )
    # Pages not addressed by the transfer stay zero.
    untouched = set(range(total_num_pages)) - set(dst_idxs)
    for page in untouched:
        assert not main_2_np[
            page * main_elts_per_page : (page + 1) * main_elts_per_page
        ].any()

    engine_2.cleanup()
    engine_1.cleanup()


def test_connect_rejects_group_bpp_mismatch_dram() -> None:
    """Engines whose per-group bytes_per_page disagree must fail at connect().

    Guards the ``createXferReq: length mismatch`` class of bug: when two
    engines compute different draft-group bytes_per_page, the error must
    surface at connect() rather than as a failed transfer at runtime.
    """
    total_num_pages = 4
    main_elts = total_num_pages * 8

    main_1 = Buffer.from_numpy(np.zeros(main_elts, dtype=np.int16))
    draft_1 = Buffer.from_numpy(np.zeros(total_num_pages * 2, dtype=np.int16))
    main_2 = Buffer.from_numpy(np.zeros(main_elts, dtype=np.int16))
    draft_2 = Buffer.from_numpy(np.zeros(total_num_pages * 3, dtype=np.int16))

    engine_1 = KVTransferEngine(
        "engine_1",
        [
            [
                kv_memory(main_1, total_num_pages),
                kv_memory(draft_1, total_num_pages),
            ]
        ],
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [
            [
                kv_memory(main_2, total_num_pages),
                kv_memory(draft_2, total_num_pages),
            ]
        ],
    )

    with pytest.raises(ValueError, match="mismatch"):
        engine_1.connect(engine_2.metadata)

    engine_2.cleanup()
    engine_1.cleanup()
