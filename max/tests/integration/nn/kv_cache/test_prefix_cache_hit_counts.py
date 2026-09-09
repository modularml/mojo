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

"""Tests for the read-only prefix-cache hit-count query.

Covers the two building blocks added for prefix-aware data-parallel routing:

- ``compute_block_hashes``: pure block hashing that neither reads nor writes
  per-request state and chains onto existing hashes.
- ``BlockManager.count_cached_prefix_blocks`` and the connectors'
  ``count_cached_prefix``: contiguous, tier-ordered (device -> host -> disk)
  hit counting with no side effects on pools, LRUs, or request state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache.block_manager import (
    BlockManager,
    PrefixCacheHits,
    compute_block_hashes,
)
from max.pipelines.modeling.types import RequestID
from test_common.context_utils import create_text_context

BLOCK_SIZE = 8


def _make_ctx(
    tokens: np.ndarray,
    request_id: RequestID = RequestID("req-1"),  # noqa: B008
) -> TextContext:
    """Build a minimal TextContext-like stub (see test_block_manager_sha256)."""
    ctx = SimpleNamespace(
        request_id=request_id,
        tokens=tokens,
        cache_salt=None,
        dkv_cache_hint=None,
    )
    return cast(TextContext, ctx)


def _make_block_manager(
    *,
    connector: object | None = None,
    enable_prefix_caching: bool = True,
) -> BlockManager:
    return BlockManager(
        total_num_blocks=32,
        block_size=BLOCK_SIZE,
        connector=cast(object, connector or NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=enable_prefix_caching,
    )


def _compute_block_hashes(
    bm: BlockManager, ctx: TextContext, existing_hashes: Sequence[bytes]
) -> list[bytes]:
    """Hashes with the manager's own settings, without touching its state."""
    return compute_block_hashes(
        ctx,
        existing_hashes,
        bm.block_size,
        bm.kv_hash_algo,
        bm.kv_hash_seed,
    )


def _seed_device_prefix_cache(
    bm: BlockManager, hashes: Sequence[bytes]
) -> None:
    """Commit blocks with the given hashes into the device prefix cache."""
    for h in hashes:
        block, _ = bm.device_block_pool.alloc_block()
        bm.device_block_pool.commit_into_prefix_cache(h, block)


class _TierStubConnector:
    """KVConnector-shaped stub with host/disk membership sets.

    Exercises the BlockManager -> connector hand-off: records the hashes it
    receives so tests can assert the walk starts where the device prefix
    ended, and asserts the canonical bytes form crossing the boundary.
    """

    def __init__(
        self,
        host_hashes: set[bytes] | None = None,
        disk_hashes: set[bytes] | None = None,
    ) -> None:
        self._host_hashes = host_hashes or set()
        self._disk_hashes = disk_hashes or set()
        self.received_hashes: list[bytes] | None = None

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "TierStubConnector"

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=4 * 4096, total=4 * 4096)

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        assert all(isinstance(h, bytes) for h in block_hashes)
        self.received_hashes = list(block_hashes)
        num_host_hits = 0
        num_disk_hits = 0
        for h in block_hashes:
            if h in self._host_hashes:
                num_host_hits += 1
            elif h in self._disk_hashes:
                num_disk_hits += 1
            else:
                break
        return (num_host_hits, num_disk_hits)

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> int:
        raise NotImplementedError("must not be called by count paths")

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
    ) -> None:
        raise NotImplementedError("must not be called by count paths")


class _ReusableTierStubConnector:
    """KVConnector-shaped stub that actually serves blocks from its host tier.

    Separate from :class:`_TierStubConnector`, whose ``load`` raises on purpose
    to prove the count path never transfers. This one drives the *reuse* path,
    so it implements ``load`` and ``touch``.
    """

    def __init__(self, host_hashes: set[bytes] | None = None) -> None:
        self._host_hashes = host_hashes or set()
        self.touched: list[bytes] | None = None

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        return "ReusableTierStubConnector"

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=8 * 4096, total=8 * 4096)

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        num_host = 0
        for h in block_hashes:
            if h not in self._host_hashes:
                break
            num_host += 1
        return (num_host, 0)

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> CompletedTransfer:
        # Serve the leading run this stub holds; the manager frees the surplus
        # staging blocks past what we report as loaded.
        bids = list(block_ids["full"])
        num_loaded = 0
        for h in block_hashes:
            if h not in self._host_hashes:
                break
            num_loaded += 1
        return CompletedTransfer(
            TransferDirection.LOAD,
            leaves=["full"],
            g0_blocks=bids[:num_loaded],
        )

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        self.touched = list(block_hashes)

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        raise NotImplementedError("this stub does not exercise offload")


# ---------------------------------------------------------------------------
# compute_block_hashes: pure hashing
# ---------------------------------------------------------------------------


def test_compute_block_hashes_is_side_effect_free() -> None:
    bm = _make_block_manager()
    # 33 tokens => 32 hashable (last reserved) => 4 full blocks of 8.
    ctx = _make_ctx(np.arange(33, dtype=np.int32))

    hashes = _compute_block_hashes(bm, ctx, [])

    assert len(hashes) == 4
    assert ctx.request_id not in bm.req_to_hashes
    assert ctx.request_id not in bm.req_to_blocks
    assert ctx.request_id not in bm.req_to_committed_idx


def test_compute_block_hashes_matches_stateful_path() -> None:
    tokens = np.arange(33, dtype=np.int32)

    bm = _make_block_manager()
    pure = _compute_block_hashes(bm, _make_ctx(tokens, RequestID("req-A")), [])

    bm.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-B")))
    stateful = bm.req_to_hashes[RequestID("req-B")]

    assert pure == stateful


def test_compute_block_hashes_chains_onto_existing() -> None:
    """Hashes computed incrementally chain to the same values as one shot."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager()
    ctx = _make_ctx(tokens)

    full = _compute_block_hashes(bm, ctx, [])
    continuation = _compute_block_hashes(bm, ctx, full[:2])

    assert continuation == full[2:]


def test_compute_block_hashes_partial_block_returns_empty() -> None:
    bm = _make_block_manager()
    # 8 tokens => 7 hashable => no full block of 8.
    ctx = _make_ctx(np.arange(8, dtype=np.int32))

    assert _compute_block_hashes(bm, ctx, []) == []


# ---------------------------------------------------------------------------
# count_cached_prefix_blocks: device tier
# ---------------------------------------------------------------------------


def test_count_device_hits_contiguous_prefix() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])

    _seed_device_prefix_cache(bm, hashes[:3])

    hits = bm.count_cached_prefix_blocks(hashes)
    assert hits == PrefixCacheHits(device_blocks=3)
    assert hits.total_blocks == 3


def test_count_stops_at_device_gap() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])

    # Seed blocks 0 and 2, leaving a gap at block 1. With a NullConnector
    # (no external tiers) the walk must stop at the gap: block 2 is cached
    # but not reachable as part of the contiguous prefix.
    _seed_device_prefix_cache(bm, [hashes[0], hashes[2]])

    hits = bm.count_cached_prefix_blocks(hashes)
    assert hits == PrefixCacheHits(device_blocks=1)


def test_count_with_no_hits_is_zero() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])

    assert bm.count_cached_prefix_blocks(hashes) == PrefixCacheHits()


def test_count_respects_prefix_caching_disabled() -> None:
    bm = _make_block_manager(enable_prefix_caching=False)
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])

    assert bm.count_cached_prefix_blocks(hashes) == PrefixCacheHits()


# ---------------------------------------------------------------------------
# reuse_blocks_from_prefix_cache: device_blocks_served metric (CENG-845)
# ---------------------------------------------------------------------------


def test_device_hit_increments_device_blocks_served() -> None:
    """A local device prefix-cache hit is counted as ``device_blocks_served``.

    Regression guard for the G0 tier-attribution counter: the increment must
    fire on the true local-hit branch inside
    ``_get_full_blocks_from_device_prefix_cache``, not on that method's full
    return value (which also carries cross-replica-copied blocks) -- a
    single-replica local hit must never also tick
    ``cross_replica_blocks_copied``/``cross_replica_bytes_copied``.
    """
    bm = _make_block_manager()

    num_prompt_tokens = 2 * BLOCK_SIZE + 1
    ctx = create_text_context(np.arange(num_prompt_tokens))
    bm.claim(ctx)
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[bytes]", list(bm.req_to_hashes[ctx.request_id]))
    assert len(hashes) == 2

    _seed_device_prefix_cache(bm, hashes)

    skip_amount, event = bm.reuse_blocks_from_prefix_cache(ctx)

    assert skip_amount == 2 * BLOCK_SIZE
    assert event.is_complete()
    assert bm._metrics.device_blocks_served == 2
    assert bm._metrics.cross_replica_blocks_copied == 0
    assert bm._metrics.cross_replica_bytes_copied == 0


def test_device_hit_attributes_no_cached_prefix_to_the_external_tier() -> None:
    """CLIN-1785: a pure device hit leaves ``cached_prefix_external_length`` 0.

    This is the field the scheduler subtracts from ``cached_prefix_length`` to
    tag ``maxserve.cache.hits`` per tier, so a non-zero value here would charge
    an on-device hit to the connector and understate the device share. Asserted
    on the admission path rather than on ``get_full_blocks_from_prefix_cache``'s
    block count, because the scheduler reads the context field, not the count.
    """
    bm = _make_block_manager()

    num_prompt_tokens = 2 * BLOCK_SIZE + 1
    ctx = create_text_context(np.arange(num_prompt_tokens))
    bm.claim(ctx)
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[bytes]", list(bm.req_to_hashes[ctx.request_id]))
    _seed_device_prefix_cache(bm, hashes)

    skip_amount, _ = bm.reuse_blocks_from_prefix_cache(ctx)

    assert skip_amount == 2 * BLOCK_SIZE
    assert ctx.cached_prefix_length == 2 * BLOCK_SIZE
    assert ctx.cached_prefix_external_length == 0
    # The whole cached prefix is therefore attributed to the device tier.
    assert (
        ctx.cached_prefix_length - ctx.cached_prefix_external_length
        == 2 * BLOCK_SIZE
    )


def test_external_tier_hit_is_attributed_away_from_the_device() -> None:
    """CLIN-1785: a connector-served prefix lands in ``external``, not ``g0``.

    This is the only place the non-zero split is produced end to end. Without
    it, dropping the ``cached_prefix_external_length`` assignment in
    ``reuse_blocks_from_prefix_cache`` leaves every test green while ``g0``
    silently absorbs the connector's blocks -- which is the pre-change reading
    and the exact misattribution this metric exists to prevent.

    Block 0 is on device and block 1 only in the connector's host tier, so the
    cached prefix spans both tiers and the split must divide it 1:1.
    """
    connector = _ReusableTierStubConnector()
    bm = _make_block_manager(connector=connector)

    ctx = create_text_context(np.arange(2 * BLOCK_SIZE + 1))
    bm.claim(ctx)
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[bytes]", list(bm.req_to_hashes[ctx.request_id]))
    assert len(hashes) == 2

    _seed_device_prefix_cache(bm, hashes[:1])
    connector._host_hashes = {hashes[1]}

    skip_amount, _ = bm.reuse_blocks_from_prefix_cache(ctx)

    assert skip_amount == 2 * BLOCK_SIZE
    assert ctx.cached_prefix_length == 2 * BLOCK_SIZE
    # One block from the connector, one from the device prefix cache.
    assert ctx.cached_prefix_external_length == BLOCK_SIZE
    assert (
        ctx.cached_prefix_length - ctx.cached_prefix_external_length
        == BLOCK_SIZE
    )


def test_a_full_miss_zeroes_both_cached_prefix_fields() -> None:
    """Both fields are written together, so neither can go stale.

    A request admitted with nothing cached must leave the pair at ``(0, 0)``:
    if only ``cached_prefix_length`` were reset, a later admission would
    subtract a previous request's external length and misattribute the split.
    """
    bm = _make_block_manager()

    ctx = create_text_context(np.arange(2 * BLOCK_SIZE + 1))
    bm.claim(ctx)
    bm.compute_hashes_for_request(ctx)

    skip_amount, _ = bm.reuse_blocks_from_prefix_cache(ctx)

    assert skip_amount == 0
    assert ctx.cached_prefix_length == 0
    assert ctx.cached_prefix_external_length == 0


def test_count_is_read_only() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])
    _seed_device_prefix_cache(bm, hashes[:2])

    num_free_before = bm.device_block_pool.num_free_blocks
    prefix_cache_before = dict(bm.device_block_pool.prefix_cache)

    first = bm.count_cached_prefix_blocks(hashes)
    second = bm.count_cached_prefix_blocks(hashes)

    assert first == second == PrefixCacheHits(device_blocks=2)
    assert bm.device_block_pool.num_free_blocks == num_free_before
    assert bm.device_block_pool.prefix_cache == prefix_cache_before
    assert ctx.request_id not in bm.req_to_hashes


# ---------------------------------------------------------------------------
# count_cached_prefix_blocks: continuation into connector tiers
# ---------------------------------------------------------------------------


def test_count_continues_into_host_and_disk_tiers() -> None:
    connector = _TierStubConnector()
    bm = _make_block_manager(connector=connector)
    ctx = _make_ctx(np.arange(41, dtype=np.int32))  # 5 full blocks
    hashes = _compute_block_hashes(bm, ctx, [])

    # Block 0 on device, block 1 on host, block 2 on disk, block 3 missing,
    # block 4 on host (unreachable past the gap).
    _seed_device_prefix_cache(bm, hashes[:1])
    connector._host_hashes = {hashes[1], hashes[4]}
    connector._disk_hashes = {hashes[2]}

    hits = bm.count_cached_prefix_blocks(hashes)

    assert hits == PrefixCacheHits(
        device_blocks=1, host_blocks=1, disk_blocks=1
    )
    assert hits.total_blocks == 3
    # The connector must only be asked about the run after the device prefix.
    assert connector.received_hashes == list(hashes[1:])


def test_count_all_device_hits_skips_connector() -> None:
    connector = _TierStubConnector()
    bm = _make_block_manager(connector=connector)
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = _compute_block_hashes(bm, ctx, [])
    _seed_device_prefix_cache(bm, hashes)

    hits = bm.count_cached_prefix_blocks(hashes)

    assert hits == PrefixCacheHits(device_blocks=4)
    assert connector.received_hashes is None


# ---------------------------------------------------------------------------
# Connector implementations
# ---------------------------------------------------------------------------


def test_null_connector_counts_nothing() -> None:
    assert NullConnector().count_cached_prefix([b"\x00" * 8]) == (0, 0)


# The host/disk tier's own host-then-disk walk lives in Rust now; it is covered
# by the kv-tier-connector crate's unit tests and
# ``internal/dkv/test_rust_tiered_connector_gpu.py``.
