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

"""Jenga's KVConnector hookup, driven by a fake connector.

The real connectors need a GPU, so these use a stand-in that records what it
was asked for and completes only when told to -- which is what lets the
deferred publish be tested: an onloaded page must stay invisible to other
requests until its copy has actually landed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    KVConnector,
    KVConnectorTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
)
from max.pipelines.request.base import RequestID

FULL = "full"
VALUES = "values"
SCALES = "scales"
SLIDING = "sliding"


class FakeTransfer:
    """A transfer that reports complete only once ``synchronize`` is called."""

    def __init__(
        self,
        g0: Mapping[str, Sequence[int]],
        direction: TransferDirection = TransferDirection.LOAD,
    ) -> None:
        self.g0_blocks_per_leaf = g0
        self.direction = direction
        self.done = False

    def is_complete(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class FakeConnector:
    """Serves whatever hashes it has been told it holds, and records calls."""

    name = "FakeConnector"

    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId] | Sequence[str],
        asynchronous: bool = True,
    ) -> None:
        self.leaves = (
            dict(leaves)
            if isinstance(leaves, Mapping)
            else {leaf: KVCacheGroupId.full() for leaf in leaves}
        )
        self.held: set[bytes] = set()
        self.asynchronous = asynchronous
        self.loads: list[tuple[dict[str, list[int]], list[bytes], int]] = []
        self.offloads: list[tuple[dict[str, list[int]], list[bytes]]] = []
        self.transfers: list[FakeTransfer] = []

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        served = 0
        for block_hash in block_hashes:
            if block_hash not in self.held:
                break
            served += 1
        self.loads.append(
            (
                {leaf: list(ids) for leaf, ids in block_ids.items()},
                list(block_hashes),
                served,
            )
        )
        loaded_blocks = {}
        for leaf_id, ids in block_ids.items():
            group_id = self.leaves[leaf_id]
            window_blocks = (
                group_id.blocks_in_window(page_size=1)
                if group_id.is_sliding_window()
                else served
            )
            null_padding = max(0, served - window_blocks)
            loaded_blocks[leaf_id] = [0] * null_padding + list(ids)[
                : served - null_padding
            ]
        transfer = FakeTransfer(loaded_blocks)
        if not self.asynchronous or served == 0:
            transfer.done = True
        self.transfers.append(transfer)
        return transfer

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        self.offloads.append(
            (
                {leaf: list(ids) for leaf, ids in block_ids.items()},
                list(block_hashes),
            )
        )
        self.held.update(block_hashes)
        transfer = FakeTransfer(
            {leaf: list(ids) for leaf, ids in block_ids.items()},
            TransferDirection.OFFLOAD,
        )
        if not self.asynchronous:
            transfer.done = True
        self.transfers.append(transfer)
        return transfer

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        return None

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        return (0, 0)

    def wait_for_loads(self) -> None:
        return None

    def wait_for_offloads(self) -> None:
        return None

    def reset_prefix_cache(self) -> None:
        self.held.clear()

    def shutdown(self) -> None:
        return None

    def reset_metrics(self) -> None:
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=1, total=2)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=3, total=4)


def make_ctx(tokens: Sequence[int]) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
    )


def make_manager(
    connector: KVConnector | None,
    leaves: Sequence[str] = (FULL,),
    num_huge_blocks: int = 16,
    leaf_infos: Mapping[str, KVLeafInfo] | None = None,
) -> JengaBlockManager:
    return JengaBlockManager(
        leaf_infos
        or {leaf: KVLeafInfo(1, KVCacheGroupId.full()) for leaf in leaves},
        num_huge_blocks=num_huge_blocks,
        block_size=1,
        enable_prefix_caching=True,
        num_replicas=1,
        max_num_input_tokens=None,
        num_draft_tokens=0,
        num_draft_tokens_per_step=0,
        connector=connector,
    )


def test_offload_then_onload_defers_publish_until_the_copy_lands() -> None:
    connector = FakeConnector([FULL])
    manager = make_manager(connector)

    # A forward's committed blocks are saved out, and an async offload pins
    # its sources until the write lands.
    first = make_ctx([1, 2, 3, 4])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    assert connector.offloads, "no offload issued"
    _, offloaded_hashes = connector.offloads[-1]
    assert manager.pending_transfers_exist(0), "async offload did not pin"
    manager.release(first)

    # Land the offload: a pinned commit survives reset_prefix_cache, so the
    # device tier is only truly empty once those pins are gone.
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    assert not manager.pending_transfers_exist(0), "offload pins never dropped"
    manager.reset_prefix_cache()
    assert not manager.pools[0].prefix_caches[FULL], "device tier not empty"

    # With the device tier empty, the same prompt must be served by the
    # connector instead.
    connector.held = set(offloaded_hashes)
    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4])
    manager.claim(second)
    transfer = manager.alloc(second)
    assert connector.loads, "no load issued"
    _, asked, served = connector.loads[-1]
    assert served > 0
    assert second.tokens.processed_length == served
    assert not transfer.is_complete(), "async load should still be in flight"

    # The heart of it: nothing may read these pages before the copy lands.
    prefix_cache = manager.pools[0].prefix_caches[FULL]
    assert all(h not in prefix_cache for h in asked[:served]), (
        "onloaded pages were published before their copy landed"
    )
    transfer.synchronize()
    manager.poll_transfers()
    assert any(h in prefix_cache for h in asked[:served]), (
        "onloaded pages were never published after landing"
    )
    assert not manager.pending_transfers_exist(0), "onload pins never dropped"


def test_multi_leaf_sends_distinct_per_leaf_block_ids() -> None:
    """Each leaf is tiled separately, so both must reach the connector.

    A single id list broadcast across leaves -- what a non-Jenga manager can
    get away with -- would address one leaf with the other's page index.
    """
    connector = FakeConnector([VALUES, SCALES])
    manager = make_manager(connector, leaves=(VALUES, SCALES))
    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)
    manager.offload(0)

    assert connector.offloads, "no offload issued"
    block_ids, hashes = connector.offloads[-1]
    assert set(block_ids) == {VALUES, SCALES}
    assert len(block_ids[VALUES]) == len(block_ids[SCALES]) == len(hashes)
    assert block_ids[VALUES] != block_ids[SCALES], (
        "leaves shared a page index; they have separate bid spaces"
    )


def test_device_hit_and_onload_splice_into_one_run() -> None:
    """A partial device hit is extended by the connector, not replaced."""
    connector = FakeConnector([FULL], asynchronous=False)
    manager = make_manager(connector)

    first = make_ctx([1, 2, 3, 4, 5])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    manager.release(first)

    # Keep only the leading block on device; the connector still holds the
    # rest, so the two hits have to meet in the middle.
    manager.reset_prefix_cache()
    connector.held = set(offloaded)
    pool = manager.pools[0]
    assert not pool.prefix_caches[FULL]
    replay = make_ctx([1])
    manager.claim(replay)
    manager.alloc(replay)
    replay.update(9)
    manager.step(replay)
    manager.release(replay)
    assert len(pool.prefix_caches[FULL]) == 1

    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4, 5])
    manager.claim(second)
    manager.alloc(second)

    # The replay's own last token is never hashed, so one block stays uncached.
    reusable = offloaded[:-1]
    asked = connector.loads[-1][1]
    assert manager.metrics.device_blocks_served == 1, (
        "device hit should stop at the one cached block"
    )
    assert asked == list(reusable[1:]), (
        "onload must resume after the device hit"
    )
    assert second.cached_prefix_length == len(reusable)
    assert second.cached_prefix_external_length == len(reusable) - 1
    assert second.tokens.processed_length == len(reusable)
    # One contiguous row: the device page first, then the onloaded ones,
    # then whatever the forward still has to fill.
    row = manager.get_req_blocks_per_leaf(second)[FULL]
    assert row[: len(reusable)] == [
        pool.prefix_caches[FULL][h].bid for h in reusable
    ]


def test_onload_is_skipped_when_the_run_does_not_fit() -> None:
    """Onloading is all-or-nothing: a run the pool cannot hold is dropped."""
    connector = FakeConnector([VALUES, SCALES], asynchronous=False)
    filler = make_manager(
        connector, leaves=(VALUES, SCALES), num_huge_blocks=16
    )
    ctx = make_ctx([1, 2, 3, 4, 5])
    filler.claim(ctx)
    filler.alloc(ctx)
    ctx.update(9)
    filler.step(ctx)
    filler.offload(0)
    assert connector.held

    # Two huge blocks between them cannot carve a 4-page run for both leaves.
    tight = make_manager(connector, leaves=(VALUES, SCALES), num_huge_blocks=3)
    pool = tight.pools[0]
    free_before = len(pool.free_huge_blocks)
    connector.loads.clear()
    replay = make_ctx([1, 2, 3, 4, 5])
    tight.claim(replay)
    with pytest.raises(InsufficientBlocksError):
        tight.alloc(replay)

    assert not connector.loads, "asked for a run the pool cannot hold"
    assert replay.tokens.processed_length == 0
    # Nothing was drawn on the way out.
    assert len(pool.free_huge_blocks) == free_before
    assert not pool.prefix_caches[VALUES]


def test_last_level_cache_only_forces_every_hit_through_the_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmarking flag disables device hits, not device commits."""
    monkeypatch.setenv("MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE", "1")
    connector = FakeConnector([FULL], asynchronous=False)
    manager = make_manager(connector)

    first = make_ctx([1, 2, 3, 4, 5])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    manager.release(first)

    # The device prefix cache still holds the run; the lookup must ignore it.
    pool = manager.pools[0]
    assert all(h in pool.prefix_caches[FULL] for h in offloaded)

    connector.held = set(offloaded)
    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4, 5])
    manager.claim(second)
    manager.alloc(second)

    assert manager.metrics.device_blocks_served == 0
    assert second.cached_prefix_external_length == second.cached_prefix_length
    assert second.cached_prefix_length > 0
    assert connector.loads[-1][1] == list(offloaded[:-1])


def test_alloc_drains_landed_transfers_so_pins_do_not_accumulate() -> None:
    """Offload pins are returned by the next alloc, as in the paged manager.

    Nothing in the serving loop calls ``poll_transfers`` on its own, so an
    alloc that skips the drain leaks every offload source for the whole run.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)
    pool = manager.pools[0]
    free_huge = len(pool.free_huge_blocks)

    for _ in range(4):
        ctx = make_ctx([1, 2, 3, 4])
        manager.claim(ctx)
        manager.alloc(ctx)
        ctx.update(9)
        manager.step(ctx)
        manager.offload(0)
        manager.release(ctx)
        # The copy engine finishes between iterations.
        for pending in connector.transfers:
            pending.synchronize()

    assert manager.pending_transfers_exist(0), "offload never pinned"
    drain = make_ctx([7, 8])
    manager.claim(drain)
    manager.alloc(drain)
    manager.release(drain)

    assert not manager.pending_transfers_exist(0), "alloc did not drain pins"
    assert len(pool.free_huge_blocks) == free_huge, (
        "huge blocks never came back: pinned pages kept them out of the pool"
    )


def test_swa_connector_onload_null_pads_and_skips_prefix_cache_commit() -> None:
    groups = {
        FULL: KVCacheGroupId.full(),
        SLIDING: KVCacheGroupId("sliding_window", 4),
    }
    connector = FakeConnector(groups, asynchronous=False)
    manager = make_manager(
        connector,
        leaf_infos={
            FULL: KVLeafInfo(1, groups[FULL]),
            SLIDING: KVLeafInfo(1, groups[SLIDING]),
        },
    )

    first = make_ctx([1, 2, 3, 4, 5, 6])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    manager.release(first)
    manager.reset_prefix_cache()
    connector.held = set(offloaded)

    replay = make_ctx([1, 2, 3, 4, 5, 6])
    manager.claim(replay)
    manager.alloc(replay)

    block_ids, asked, served = connector.loads[-1]
    assert asked == list(offloaded[:-1])
    assert served == len(asked)
    assert len(block_ids[SLIDING]) == groups[SLIDING].blocks_in_window(1)

    pool = manager.pools[0]
    sliding_blocks = manager.get_req_blocks_per_leaf(replay)[SLIDING]
    null_count = len(asked) - groups[SLIDING].blocks_in_window(1)
    assert sliding_blocks[:null_count] == [0] * null_count
    assert len(pool.prefix_caches[SLIDING]) == len(asked) - null_count
    assert all(
        not block.is_null for block in pool.prefix_caches[SLIDING].values()
    )
    assert len(pool.prefix_caches[FULL]) == len(asked)
