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

"""Unit tests for ``BlockManager.offload`` sequence delivery.

These run CPU-only and construct a ``BlockManager`` directly with a recording
connector — no graph, session, or device memory. They cover the subtle parts of
delivering committed blocks as ordered offload sequences: hash re-resolution to
current device blocks, truncation of a run at the first block evicted since
commit, multi-run ordering, and that the pending queue is drained.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import cast

import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedTransfer,
    KVConnectorTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache import (
    block_manager as block_manager_module,
)
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVCacheBlock
from max.pipelines.modeling.types import RequestID


# Maps an int block hash to its canonical 8-byte big-endian signed encoding
# for readability in test fixtures and assertions.
def _b(h: int) -> bytes:
    return h.to_bytes(8, "big", signed=True)


class RecordingConnector:
    """Connector stub that records ``offload``, ``touch`` and ``load`` calls."""

    def __init__(self) -> None:
        self.offloads: list[tuple[list[int], list[bytes]]] = []
        self.touches: list[tuple[list[bytes], int]] = []
        # The ``hint`` each ``load`` was given, in call order.
        self.load_hints: list[bytes | None] = []
        # Ordered log of ``load``/``touch`` call names, so a test can assert the
        # load-path anchor touch fires AFTER the load (CLIN-1533).
        self.calls: list[str] = []
        # Blocks ``load`` reports as loaded from the host tier (0 == host miss);
        # lets a test drive a cold-G0/warm-host hit without real device memory.
        self.num_blocks_to_load = 0
        self._h2d_bytes_copied = 0
        self._d2h_bytes_copied = 0

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "recording"

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        bids = list(block_ids["full"])
        self.offloads.append((bids, list(block_hashes)))
        return CompletedTransfer(
            TransferDirection.OFFLOAD, leaves=["full"], g0_blocks=bids
        )

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        self.calls.append("touch")
        self.touches.append((list(block_hashes), replica_idx))

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        self.calls.append("load")
        self.load_hints.append(hint)
        bids = list(block_ids["full"])
        num_loaded = min(len(block_hashes), self.num_blocks_to_load)
        return CompletedTransfer(
            TransferDirection.LOAD, leaves=["full"], g0_blocks=bids[:num_loaded]
        )

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
        return ByteCount(free=0, total=0)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=0, total=0)

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_bytes_copied=self._h2d_bytes_copied,
            d2h_bytes_copied=self._d2h_bytes_copied,
        )

    def reset_metrics(self) -> None:
        self._h2d_bytes_copied = 0
        self._d2h_bytes_copied = 0


class _ExternalTierConnector(RecordingConnector):
    """A dKV-style connector that advertises an external tier.

    ``get_full_blocks_from_prefix_cache`` gates the G0 recency ``touch`` behind
    its ``host_byte_count.total == 0`` early-return, so the touch-firing
    tests need a connector whose host block count is positive (a plain
    ``RecordingConnector`` reports 0, i.e. no external tier). Records touches
    like its base so a test can assert on them.
    """

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(
            free=1024 * _FAKE_BYTES_PER_PAGE, total=1024 * _FAKE_BYTES_PER_PAGE
        )


_FAKE_BYTES_PER_PAGE = 64


class _BatchCopyRecorder:
    """Shared counter for fake ``batch_inplace_copy`` submits."""

    def __init__(self) -> None:
        self.copies: list[tuple[int, int]] = []
        # Destination device of each entry in ``copies``, same order. The
        # driver groups a batch by destination device internally, so submit
        # count no longer distinguishes devices -- this does.
        self.dst_device_ids: list[int] = []
        self.batched_calls = 0
        # Optional hook fired at the start of each batch submit so a test can
        # snapshot local-prefix-cache visibility during enqueue.
        self.on_batch_copy: Callable[[], None] | None = None


class _FakePageView:
    """Stand-in for a Buffer page view, carrying its recorder and page index."""

    def __init__(
        self, recorder: _BatchCopyRecorder, block_id: int, device_id: int
    ) -> None:
        self.recorder = recorder
        self.block_id = block_id
        self.device_id = device_id


def _fake_batch_inplace_copy(
    dst_pages: Sequence[_FakePageView], src_pages: Sequence[_FakePageView]
) -> None:
    """Records one submit; stands in for the driver's ``batch_inplace_copy``.

    Every page view in a batch shares one recorder, so the destinations carry
    it in from the call.
    """
    recorder = dst_pages[0].recorder
    if recorder.on_batch_copy is not None:
        recorder.on_batch_copy()
    recorder.batched_calls += 1
    recorder.copies.extend(
        (dst.block_id, src.block_id)
        for dst, src in zip(dst_pages, src_pages, strict=True)
    )
    recorder.dst_device_ids.extend(dst.device_id for dst in dst_pages)


@pytest.fixture(autouse=True)
def fake_batch_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routes the block manager's batched copies at the fake page views.

    ``batch_inplace_copy`` is a module-level driver call rather than a Buffer
    method, so the fake is installed by patching the name ``block_manager``
    resolves.
    """
    monkeypatch.setattr(
        block_manager_module, "batch_inplace_copy", _fake_batch_inplace_copy
    )


class _FakeBuffer:
    """Stand-in for a shard Buffer; ``[block_id, :]`` yields a page view."""

    def __init__(self, recorder: _BatchCopyRecorder, device_id: int) -> None:
        self._recorder = recorder
        self.device = SimpleNamespace(id=device_id)

    def __getitem__(self, key: tuple[int, slice]) -> _FakePageView:
        block_id, _ = key
        return _FakePageView(self._recorder, block_id, self.device.id)


class _FakeKVMemory:
    """CPU stand-in for a ``KVCacheMemory`` unit: records cross-replica D2D
    copies without touching device memory, so a cross-replica prefix-cache hit
    can be exercised CPU-only.

    ``bytes_per_page`` mirrors ``KVCacheMemory.bytes_per_page`` so
    ``_copy_blocks_across_replicas`` can accumulate
    ``cross_replica_bytes_copied``.
    """

    def __init__(
        self,
        *,
        device_ids: Sequence[int] = (0,),
        recorder: _BatchCopyRecorder | None = None,
    ) -> None:
        self._recorder = (
            recorder if recorder is not None else _BatchCopyRecorder()
        )
        self.bytes_per_page = _FAKE_BYTES_PER_PAGE
        self.buffers = [
            _FakeBuffer(self._recorder, device_id=did) for did in device_ids
        ]

    @property
    def copies(self) -> list[tuple[int, int]]:
        return self._recorder.copies

    @property
    def dst_device_ids(self) -> list[int]:
        return self._recorder.dst_device_ids

    @property
    def batched_calls(self) -> int:
        return self._recorder.batched_calls

    @property
    def on_batch_copy(self) -> Callable[[], None] | None:
        return self._recorder.on_batch_copy

    @on_batch_copy.setter
    def on_batch_copy(self, fn: Callable[[], None] | None) -> None:
        self._recorder.on_batch_copy = fn


def _make_block_manager(
    *,
    num_replicas: int = 1,
    unit_device_ids: Sequence[Sequence[int]] = ((0,),),
    connector: RecordingConnector | None = None,
    enable_dp_cross_replica_prefix_copy: bool = True,
) -> tuple[BlockManager, RecordingConnector]:
    connector = connector if connector is not None else RecordingConnector()
    # Multi-replica needs per-replica memory units so a cross-replica hit can
    # materialize via device-to-device copy (fake, CPU-only). All units share
    # one recorder: submits land on destination page views, but tests often
    # assert via a source unit handle.
    #
    # ``unit_device_ids[u]`` is the per-shard device-id list for unit ``u``. Each
    # unit carries every TP shard, so a quantized cache is two units over the
    # same devices (e.g. ((0,1),(0,1))) whether it is MLA-replicated or sharded;
    # a single-kind cache is one unit (e.g. ((0,1),)).
    replica_kv_memory: Sequence[Sequence[KVCacheMemory]] | None = None
    if num_replicas > 1:
        recorder = _BatchCopyRecorder()
        fakes = [
            [
                _FakeKVMemory(device_ids=device_ids, recorder=recorder)
                for device_ids in unit_device_ids
            ]
            for _ in range(num_replicas)
        ]
        replica_kv_memory = cast("Sequence[Sequence[KVCacheMemory]]", fakes)
    bm = BlockManager(
        total_num_blocks=64,
        block_size=16,
        connector=connector,
        enable_prefix_caching=True,
        num_replicas=num_replicas,
        replica_kv_memory=replica_kv_memory,
        enable_dp_cross_replica_prefix_copy=enable_dp_cross_replica_prefix_copy,
    )
    return bm, connector


def _commit(bm: BlockManager, hash_to_bid: dict[bytes, int]) -> None:
    """Place ``hash -> KVCacheBlock(bid)`` entries in the device prefix cache."""
    for block_hash, bid in hash_to_bid.items():
        bm.device_block_pool.prefix_cache[block_hash] = KVCacheBlock(bid)


def _make_ctx(
    bm: BlockManager,
    request_id: RequestID,
    replica_idx: int = 0,
    dkv_cache_hint: bytes | None = None,
) -> TextContext:
    """Minimal ctx stub, claimed so it reads a replica's pool.

    ``get_full_blocks_from_prefix_cache`` reads only ``ctx.request_id`` and
    ``ctx.dkv_cache_hint`` on this path (no tokens/salt/images), so a
    ``SimpleNamespace`` suffices; the claim is what pins which replica the
    request resolves against.
    """
    ctx = cast(
        TextContext,
        SimpleNamespace(request_id=request_id, dkv_cache_hint=dkv_cache_hint),
    )
    bm.claim(ctx, replica_idx)
    return ctx


def _commit_device_block(pool: BlockPool, block_hash: int) -> KVCacheBlock:
    """Commit ``block_hash`` as an idle eviction-candidate device block.

    Unlike :func:`_commit` (which injects a bare ``KVCacheBlock`` used only by
    the offload path), this allocates, commits, then frees a real pool block so
    it sits in both the prefix cache and the free queue at ``ref_cnt == 0`` --
    the realistic state a device prefix-cache *hit* resolves to, and the state
    that lets the hit's ``BlockPool.touch`` exercise the free-queue path without
    corrupting it.
    """
    block, _ = pool.alloc_block()
    pool.commit_into_prefix_cache(_b(block_hash), block)
    pool.free_block(block)
    return block


def test_offload_delivers_run_resolving_hashes_to_bids() -> None:
    bm, connector = _make_block_manager()
    _commit(bm, {_b(111): 5, _b(222): 6, _b(333): 7})
    # One run of three committed blocks.
    bm._pending_offloads = [[[_b(111), _b(222), _b(333)]]]

    bm.offload()

    assert connector.offloads == [([5, 6, 7], [_b(111), _b(222), _b(333)])]
    # Pending queue drained.
    assert bm._pending_offloads == [[]]


def test_offload_truncates_run_at_evicted_block() -> None:
    bm, connector = _make_block_manager()
    # 222 was evicted since commit; the run stops there.
    _commit(bm, {_b(111): 5, _b(333): 7})
    bm._pending_offloads = [[[_b(111), _b(222), _b(333)]]]

    bm.offload()

    assert connector.offloads == [([5], [_b(111)])]


def test_offload_skips_fully_evicted_run() -> None:
    bm, connector = _make_block_manager()
    # First (and only) block of the run is gone -> nothing to deliver.
    _commit(bm, {})
    bm._pending_offloads = [[[_b(111)]]]

    bm.offload()

    assert connector.offloads == []
    assert bm._pending_offloads == [[]]


def test_offload_preserves_multi_run_order() -> None:
    bm, connector = _make_block_manager()
    _commit(bm, {_b(111): 1, _b(222): 2, _b(333): 3, _b(444): 4})
    # Two runs queued across two commits.
    bm._pending_offloads = [
        [
            [_b(111), _b(222)],
            [_b(333), _b(444)],
        ]
    ]

    bm.offload()

    assert connector.offloads == [
        ([1, 2], [_b(111), _b(222)]),
        ([3, 4], [_b(333), _b(444)]),
    ]


def test_reset_metrics_clears_connector_transfer_counters() -> None:
    """Per-batch telemetry must reset connector H2D/D2H counters after sampling.

    Without this, ``get_metrics_aggregated()`` returns lifetime cumulative
    totals and Datadog counter.add() double-counts across batches (MXSERV-203).
    """
    bm, connector = _make_block_manager()
    connector._d2h_bytes_copied = 5
    connector._h2d_bytes_copied = 2

    assert bm.metrics.d2h_bytes_copied == 5
    assert bm.metrics.h2d_bytes_copied == 2

    bm.reset_metrics()

    assert bm.metrics.d2h_bytes_copied == 0
    assert bm.metrics.h2d_bytes_copied == 0

    connector._d2h_bytes_copied = 3
    assert bm.metrics.d2h_bytes_copied == 3
    assert bm.metrics.h2d_bytes_copied == 0


def test_touch_fires_on_device_hit_with_full_root_anchored_hashes() -> None:
    """A G0 device prefix-cache hit touches the FULL root-anchored sequence.

    The request has a 4-block root-anchored prefix whose first two blocks are
    already committed (``num_committed_blocks == 2``), so the device is queried
    for the root-omitting slice ``[333, 444]``. The touch payload must still be
    the full ``[111, 222, 333, 444]`` -- not that slice -- so the prefix root
    stays MRU under dKV's reverse full-attention LRU (the ordering correction,
    CLIN-1533). It fires exactly once and, the whole prefix being on device,
    issues no ``load``. Uses an external-tier connector because the anchor is
    gated on ``host_byte_count.total``.
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-hit")
    bm.req_to_hashes[rid] = [_b(111), _b(222), _b(333), _b(444)]
    # First two blocks already committed => num_committed_blocks == 2.
    bm.req_to_committed_idx[rid] = 2 * bm.block_size
    _commit_device_block(bm.device_block_pool, 333)
    _commit_device_block(bm.device_block_pool, 444)

    device_blocks, _, num_external = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid)
    )

    assert len(device_blocks) == 2  # the two uncommitted device hits
    assert num_external == 0  # served on device, so nothing to attribute out
    assert connector.calls == ["touch"]  # fires once; no host load
    assert connector.touches == [([_b(111), _b(222), _b(333), _b(444)], 0)]


def test_touch_anchor_not_fired_on_fully_cold_request() -> None:
    """Fully cold (no device hit AND no host hit) means no anchor touch.

    Nothing is resident on device and the host tier loads nothing
    (``num_blocks_to_load == 0``), so both ``device_blocks`` and ``host_blocks``
    are empty and the ``if device_blocks or host_blocks`` gate suppresses the
    anchor -- even though the load path ran (``load`` was called). Uses an
    external-tier connector so the ``host_byte_count.total`` gate is passed
    and the empty-result gate is what's exercised.
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-cold")
    bm.req_to_hashes[rid] = [
        _b(111),
        _b(222),
    ]  # nothing on device, nothing in host

    served, _, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    assert served == []  # nothing served
    assert connector.calls == ["load"]  # load ran; gate suppressed the touch
    assert connector.touches == []


def test_touch_fires_on_cross_replica_hit_keyed_to_serving_replica() -> None:
    """A cross-replica device hit still touches the full sequence.

    The blocks are resident only on replica 1's prefix cache; the request runs
    on replica 0, so the hit is served by a device-to-device materialization
    onto replica 0. The touch carries the full root-anchored sequence and is
    keyed to the *serving* replica (0, the client selector) -- not the source
    replica (1).
    """
    bm, connector = _make_block_manager(
        num_replicas=2, connector=_ExternalTierConnector()
    )
    rid = RequestID("req-xrep")
    bm.req_to_hashes[rid] = [_b(111), _b(222)]
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    device_blocks, _, _ = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, replica_idx=0)
    )

    assert len(device_blocks) == 2  # materialized onto replica 0
    assert connector.touches == [([_b(111), _b(222)], 0)]
    # Cross-replica D2D copies, not local device hits (CENG-845).
    assert bm._metrics.cross_replica_blocks_copied == 2
    assert bm._metrics.cross_replica_bytes_copied == 2 * _FAKE_BYTES_PER_PAGE
    assert bm._metrics.device_blocks_served == 0


def test_cross_replica_hit_issues_a_single_batched_copy() -> None:
    """A multi-block cross-replica hit uses one batched D2D call per shard.

    Both pages are resident only on replica 1; the request runs on replica 0.
    The fake memory must record both pages under a single
    ``batch_inplace_copy`` submit, not one copy per page.

    Prefix-cache publish is deferred until after enqueue: during the batched
    copy the destination hashes must not yet be visible in replica 0's local
    cache; after materialization returns, they must be.
    """
    bm, _ = _make_block_manager(num_replicas=2)
    rid = RequestID("req-xrep-batch")
    hashes = [_b(111), _b(222)]
    bm.req_to_hashes[rid] = hashes
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    local_cache = bm.device_block_pools[0].prefix_cache
    visible_during_batch: list[bool] = []
    assert bm._replica_kv_memory is not None
    src_unit = cast(_FakeKVMemory, bm._replica_kv_memory[1][0])

    def _snapshot_local_visibility() -> None:
        visible_during_batch.append(any(h in local_cache for h in hashes))

    src_unit.on_batch_copy = _snapshot_local_visibility

    served, _, _ = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, replica_idx=0)
    )

    assert len(served) == 2  # both pages materialized onto replica 0
    assert bm.metrics.cross_replica_blocks_copied == 2
    assert src_unit.batched_calls == 1  # one batched call, not two
    assert len(src_unit.copies) == 2  # carrying both pages
    # Enqueue-before-publish: hashes must be invisible during batch_copy and
    # visible once materialization returns.
    assert visible_during_batch == [False]
    assert all(h in local_cache for h in hashes)


def test_cross_replica_hit_merges_units_into_one_submit() -> None:
    """Cross-replica copies merge every unit into a single submit.

    MLA-like layout: 2 units each with buffers on devices 0 and 1. Expect one
    ``batch_inplace_copy`` carrying all 8 page pairs, split evenly across the
    two destination devices -- the driver, not the block manager, groups them.
    Enqueue-before-publish still holds.
    """
    unit_device_ids = ((0, 1), (0, 1))
    num_devices = 2
    num_units = len(unit_device_ids)
    bm, _ = _make_block_manager(num_replicas=2, unit_device_ids=unit_device_ids)
    rid = RequestID("req-xrep-merge")
    hashes = [_b(111), _b(222)]
    bm.req_to_hashes[rid] = hashes
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    local_cache = bm.device_block_pools[0].prefix_cache
    visible_during_batch: list[bool] = []
    assert bm._replica_kv_memory is not None
    # Units in a replica share one recorder; either unit surfaces the totals.
    src_unit = cast(_FakeKVMemory, bm._replica_kv_memory[1][0])

    def _snapshot_local_visibility() -> None:
        visible_during_batch.append(any(h in local_cache for h in hashes))

    src_unit.on_batch_copy = _snapshot_local_visibility

    served, _, _ = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, replica_idx=0)
    )

    assert len(served) == 2
    assert bm.metrics.cross_replica_blocks_copied == 2
    assert src_unit.batched_calls == 1  # not one per unit, nor per device
    # The submit includes every unit's page pairs for every device.
    assert len(src_unit.copies) == num_units * num_devices * len(hashes)
    assert sorted(src_unit.dst_device_ids) == sorted(
        [0, 1] * (num_units * len(hashes))
    )
    assert visible_during_batch == [False]
    assert all(h in local_cache for h in hashes)


def test_cross_replica_hit_covers_every_device_in_one_submit() -> None:
    """MHA-like layout reaches both TP devices from the single submit.

    One unit carrying its two TP shards on devices 0 and 1 (non-replicated TP),
    which is what ``to_memory()`` authors for a sharded cache. Every shard's
    pages must be present and attributed to its own destination device; dropping
    a shard would silently leave one device's pages stale.
    """
    unit_device_ids = ((0, 1),)
    num_buffers = sum(len(ids) for ids in unit_device_ids)
    bm, _ = _make_block_manager(num_replicas=2, unit_device_ids=unit_device_ids)
    rid = RequestID("req-xrep-mha")
    hashes = [_b(111), _b(222)]
    bm.req_to_hashes[rid] = hashes
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    assert bm._replica_kv_memory is not None
    src_unit = cast(_FakeKVMemory, bm._replica_kv_memory[1][0])

    served, _, _ = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, replica_idx=0)
    )

    assert len(served) == 2
    assert src_unit.batched_calls == 1
    assert len(src_unit.copies) == num_buffers * len(hashes)
    assert sorted(src_unit.dst_device_ids) == sorted([0, 1] * len(hashes))


def test_cross_replica_copy_disabled_serves_from_external_tier() -> None:
    """With enable_dp_cross_replica_prefix_copy off, a block resident only on
    another replica's device is NOT materialized via a device-to-device copy:
    the device lookup stops at the local miss and the prefix is served from
    the shared external tier instead.
    """
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 2  # both blocks are warm in the tier
    bm, _ = _make_block_manager(
        num_replicas=2,
        connector=connector,
        enable_dp_cross_replica_prefix_copy=False,
    )
    rid = RequestID("req-xrep-off")
    bm.req_to_hashes[rid] = [_b(111), _b(222)]
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    served, _, num_external = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, replica_idx=0)
    )

    assert len(served) == 2  # served, but by the external tier
    # With cross-replica copies off, the peer's device blocks are unreachable
    # and the connector serves the whole prefix, so it is all `external` rather
    # than the `g0` it would have been with the copy enabled.
    assert num_external == 2
    assert connector.calls == ["load", "touch"]  # host load, no device hit
    assert bm.metrics.cross_replica_blocks_copied == 0
    assert bm._replica_kv_memory is not None
    for units in bm._replica_kv_memory:
        assert cast(_FakeKVMemory, units[0]).copies == []  # no D2D issued


def test_cross_replica_copy_disabled_count_is_local_only() -> None:
    """With the flag off, the admission estimate must not count blocks the
    reuse path can no longer serve by device-to-device copy: only the request
    replica's own resident blocks count as device hits.
    """
    bm, _ = _make_block_manager(
        num_replicas=2, enable_dp_cross_replica_prefix_copy=False
    )
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    assert (
        bm._count_full_blocks_from_prefix_cache(
            [_b(111), _b(222)], replica_idx=0
        )
        == 0
    )
    assert (
        bm._count_full_blocks_from_prefix_cache(
            [_b(111), _b(222)], replica_idx=1
        )
        == 2
    )


def test_load_receives_the_requests_cache_hint() -> None:
    """The request's ``dkv_cache_hint`` reaches ``connector.load`` unchanged.

    The manager never reads the hint; it only has to carry it, because the
    routing it drives is parsed on the connector's Rust side (CLIN-1630).
    """
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 2
    bm, _ = _make_block_manager(connector=connector)
    rid = RequestID("req-hinted")
    bm.req_to_hashes[rid] = [_b(111), _b(222)]
    hint = b'{"version":2,"instances":[{"instance_name":"dkv-peer"}]}'

    bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid, dkv_cache_hint=hint)
    )

    assert connector.load_hints == [hint]


def test_load_receives_no_hint_for_an_unhinted_request() -> None:
    """An unhinted request loads with ``hint=None``, today's behavior."""
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 1
    bm, _ = _make_block_manager(connector=connector)
    rid = RequestID("req-unhinted")
    bm.req_to_hashes[rid] = [_b(111)]

    bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    assert connector.load_hints == [None]


def test_touch_anchor_fires_after_load_on_host_only_hit() -> None:
    """Cold-G0 / warm-external hit: the anchor fires AFTER the load, once.

    Nothing is resident on device, so ``device_blocks`` is empty and the whole
    prefix is pulled from the external tier by the load. The anchor is the SOLE
    load-path recency signal, so it must fire AFTER that load -- as the only
    load-path toucher it thereby reserves the last recency stamp and cannot be
    inverted by the load's own touches (CLIN-1533). It fires exactly once, with
    the root-anchored payload, even though there was no device hit.
    """
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 2  # both requested blocks load from host
    bm, _ = _make_block_manager(connector=connector)
    rid = RequestID("req-host")
    bm.req_to_hashes[rid] = [
        _b(111),
        _b(222),
    ]  # nothing committed, nothing on device

    served, _, num_external = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(bm, rid)
    )

    assert len(served) == 2  # both served from the host tier (no device hit)
    assert num_external == 2  # every block is the connector's
    assert connector.calls == ["load", "touch"]  # touch after load, once
    assert connector.touches == [([_b(111), _b(222)], 0)]


def test_touch_anchor_payload_trims_uncached_tail() -> None:
    """Anchor payload is root-anchored: committed prefix in, uncached tail out.

    A 3-block request whose first block is already committed
    (``num_committed_blocks == 1``) hits the device for block 2 only; block 3
    is uncached (absent on device and in the host tier). The touch payload
    must be ``[111, 222]``: it INCLUDES the committed root 111 (omitting it
    re-creates a recency inversion) and EXCLUDES the uncached tail 333 (absent
    server-side, so touching it is a wasted index lookup on a contended path).
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-tail")
    bm.req_to_hashes[rid] = [_b(111), _b(222), _b(333)]
    # First block already committed => num_committed_blocks == 1.
    bm.req_to_committed_idx[rid] = 1 * bm.block_size
    _commit_device_block(bm.device_block_pool, 222)  # 333 stays uncached

    served, _, _ = bm.get_full_blocks_from_prefix_cache(_make_ctx(bm, rid))

    assert len(served) == 1  # only 222 hit; 333 is the uncached tail
    assert connector.touches == [([_b(111), _b(222)], 0)]  # root in, tail out
