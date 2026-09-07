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

"""Unit tests for ``BlockManager``'s asynchronous KV-connector transfer path.

These run CPU-only and construct a ``BlockManager`` directly with a fake
connector whose ``load``/``offload`` return a *controllable, not-yet-complete*
``KVConnectorTransfer`` (the behavior of the ``rust_tiered`` connector, whose
H2D/D2H run on a separate copy engine). They cover the parts of the async path
that the synchronous connectors never exercise:

- an in-flight onload defers its device-prefix-cache commit and pins the
  destination blocks (a same-batch request must not read them before the copy
  lands),
- ``poll_transfers`` is a no-op while the transfer is incomplete and, once it
  completes, commits the onloaded blocks and unpins them,
- an in-flight offload pins its source blocks without a deferred commit, and
- pinned blocks are held out of the free queue (the primitive behind the
  scheduler's "defer instead of OOM while transfers are in flight" guard).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import cast

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    KVConnectorTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVCacheBlock
from max.pipelines.modeling.types import RequestID


def _b(h: int) -> bytes:
    """Maps an int block hash to its canonical 8-byte big-endian encoding."""
    return h.to_bytes(8, "big", signed=True)


class _ControllableTransfer:
    """A ``KVConnectorTransfer`` whose completion the test flips by hand.

    Models an asynchronous connector's handle: ``is_complete`` returns ``False``
    until the test sets ``complete`` (or calls ``synchronize``), so the block
    manager takes the deferred-commit / pinning branch.
    """

    def __init__(
        self, direction: TransferDirection, g0_blocks: list[int]
    ) -> None:
        self._direction = direction
        self._g0_blocks = {"full": list(g0_blocks)}
        self.complete = False

    @property
    def direction(self) -> TransferDirection:
        return self._direction

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        return self._g0_blocks

    def is_complete(self) -> bool:
        return self.complete

    def synchronize(self) -> None:
        self.complete = True


class _AsyncConnector:
    """A fake external-tier connector that returns in-flight transfers.

    ``host_byte_count.total`` is positive so the block manager runs its
    host-onload path; ``num_blocks_to_load`` controls how many of a ``load``'s
    requested hashes are reported as found. Every returned transfer is
    recorded so a test can flip it complete and then drive ``poll_transfers``.
    """

    def __init__(self) -> None:
        self.num_blocks_to_load = 0
        self.loads: list[_ControllableTransfer] = []
        self.offload_events: list[_ControllableTransfer] = []

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "async-fake"

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        bids = list(block_ids["full"])
        num_loaded = min(len(block_hashes), self.num_blocks_to_load)
        event = _ControllableTransfer(TransferDirection.LOAD, bids[:num_loaded])
        self.loads.append(event)
        return event

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        bids = list(block_ids["full"])
        event = _ControllableTransfer(TransferDirection.OFFLOAD, bids)
        self.offload_events.append(event)
        return event

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None: ...
    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        return (0, 0)

    def wait_for_loads(self) -> None: ...
    def wait_for_offloads(self) -> None: ...
    def shutdown(self) -> None: ...
    def reset_prefix_cache(self) -> None: ...

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=1024 * 4096, total=1024 * 4096)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    def reset_metrics(self) -> None: ...


def _make_block_manager() -> tuple[BlockManager, _AsyncConnector]:
    connector = _AsyncConnector()
    bm = BlockManager(
        total_num_blocks=64,
        block_size=16,
        connector=connector,
        enable_prefix_caching=True,
    )
    return bm, connector


def _make_ctx(bm: BlockManager, request_id: RequestID) -> TextContext:
    """Minimal claimed ctx stub.

    The host-onload path reads only ``ctx.request_id`` and
    ``ctx.dkv_cache_hint``, but the claim is what pins which replica's pool the
    request resolves against.
    """
    ctx = cast(
        TextContext,
        SimpleNamespace(request_id=request_id, dkv_cache_hint=None),
    )
    bm.claim(ctx)
    return ctx


def _commit_device_block(pool: BlockPool, block_hash: int) -> KVCacheBlock:
    """Commit ``block_hash`` as an idle eviction-candidate device block.

    Allocates, commits, then frees a real pool block so it sits in both the
    prefix cache and the free queue at ``ref_cnt == 0`` -- the state a later
    ``touch`` (pin) transitions out of the free queue.
    """
    block, _ = pool.alloc_block()
    pool.commit_into_prefix_cache(_b(block_hash), block)
    pool.free_block(block)
    return block


# -- Onload: deferred commit + pinning --


def test_async_onload_defers_commit_and_pins_blocks() -> None:
    """An incomplete onload pins its blocks and does NOT commit them yet.

    Committing before the H2D lands would let a concurrent same-batch request
    read the destination blocks while they still hold stale bytes, so the commit
    is deferred to ``poll_transfers`` and the blocks are pinned (kept out of the
    eviction / free path) until then.
    """
    bm, connector = _make_block_manager()
    connector.num_blocks_to_load = 2
    pool = bm.device_block_pool
    rid = RequestID("req-onload")
    bm.req_to_hashes[rid] = [_b(1), _b(2)]

    blocks, event, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    assert len(blocks) == 2
    assert not event.is_complete()
    assert bm.pending_transfers_exist()
    # Deferred: not yet visible in the device prefix cache.
    assert _b(1) not in pool.prefix_cache
    assert _b(2) not in pool.prefix_cache
    # Pinned: ref_cnt is 2 (1 from allocation + 1 from the transfer pin), so the
    # blocks are not in the free queue.
    for block in blocks:
        assert block.ref_cnt == 2
        assert block.bid not in pool.free_blocks


def test_poll_transfers_is_noop_while_onload_incomplete() -> None:
    """Polling before the onload lands leaves it pending and uncommitted."""
    bm, connector = _make_block_manager()
    connector.num_blocks_to_load = 2
    pool = bm.device_block_pool
    rid = RequestID("req-poll-incomplete")
    bm.req_to_hashes[rid] = [_b(1), _b(2)]

    _, event, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))
    assert not event.is_complete()

    bm.poll_transfers()

    assert bm.pending_transfers_exist()
    assert _b(1) not in pool.prefix_cache
    assert _b(2) not in pool.prefix_cache


def test_poll_transfers_commits_and_unpins_on_completion() -> None:
    """Once the onload completes, poll commits the blocks and drops the pin."""
    bm, connector = _make_block_manager()
    connector.num_blocks_to_load = 2
    pool = bm.device_block_pool
    rid = RequestID("req-poll-complete")
    bm.req_to_hashes[rid] = [_b(1), _b(2)]

    blocks, event, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    # Complete the transfer, then drain.
    event.synchronize()
    bm.poll_transfers()

    assert not bm.pending_transfers_exist()
    # Now committed into the device prefix cache and reusable.
    assert pool.prefix_cache[_b(1)] is blocks[0]
    assert pool.prefix_cache[_b(2)] is blocks[1]
    # Transfer pin released (back to the allocation's single ref).
    for block in blocks:
        assert block.ref_cnt == 1


def test_partial_onload_frees_surplus_blocks() -> None:
    """A connector that loads fewer blocks than requested frees the surplus.

    The manager allocates one destination block per requested hash, but the
    connector reports only ``num_blocks_to_load`` in ``g0_blocks``; the unused
    destinations must be returned to the pool rather than leaked.
    """
    bm, connector = _make_block_manager()
    connector.num_blocks_to_load = 1  # only the first of two hashes loads
    pool = bm.device_block_pool
    free_before = pool.num_free_blocks
    rid = RequestID("req-partial")
    bm.req_to_hashes[rid] = [_b(1), _b(2)]

    blocks, event, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    assert len(blocks) == 1
    assert len(event.g0_blocks_per_leaf["full"]) == 1
    # Exactly one block is held for the transfer; the surplus was freed.
    assert pool.num_free_blocks == free_before - 1
    assert bm.pending_transfers_exist()


# -- Offload: pin without deferred commit --


def test_async_offload_pins_source_blocks_without_commit() -> None:
    """An incomplete offload pins its source blocks and defers no commit.

    Offloads (D2H) pin the source device blocks so they are not evicted / reused
    mid-copy, but -- unlike onloads -- they carry no ``commit_hashes``, so
    ``poll_transfers`` only unpins them.
    """
    bm, connector = _make_block_manager()
    pool = bm.device_block_pool
    blk1 = _commit_device_block(pool, 111)
    blk2 = _commit_device_block(pool, 222)
    assert blk1.ref_cnt == 0 and blk2.ref_cnt == 0
    bm._pending_offloads = [[[_b(111), _b(222)]]]

    bm.offload()

    assert bm.pending_transfers_exist()
    # Source blocks pinned by the in-flight D2H.
    assert blk1.ref_cnt == 1 and blk2.ref_cnt == 1
    assert blk1.bid not in pool.free_blocks
    assert blk2.bid not in pool.free_blocks

    # Complete the offload and drain: blocks unpinned, no new commit deferred.
    connector.offload_events[-1].complete = True
    bm.poll_transfers()

    assert not bm.pending_transfers_exist()
    assert blk1.ref_cnt == 0 and blk2.ref_cnt == 0
    assert blk1.bid in pool.free_blocks
    assert blk2.bid in pool.free_blocks
