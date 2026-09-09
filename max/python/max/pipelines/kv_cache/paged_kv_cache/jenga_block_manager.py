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
"""KVCache management based on the Jenga paper.

This module implements the JengaBlockManager on top of the JengaBlockPool.
It is used to manage the allocation and release of blocks to requests.
It also manages the prefix cache hits for the requests.

We use a two level huge-little block hierarchy to allocate the blocks among the
different caches. This allows the memory to be fungible between the caches.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import BlockCount, KVConnector
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from .block_manager import (
    CompletedTransfer,
    KVConnectorTransfer,
    PrefixCacheHits,
    _compute_seq_len,
    _resolve_only_use_kv_connector_last_level_cache,
    compute_block_hashes,
)
from .block_utils import InsufficientBlocksError, KVHashAlgo, LittleKVCacheBlock
from .jenga_block_pool import JengaBlockPool
from .kv_group_coordinator import (
    FullKVGroupCoordinator,
    KVGroupCoordinatorInterface,
    SlidingWindowKVGroupCoordinator,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class _PendingTransfer:
    """An in-flight async connector transfer and the pages it pins.

    ``blocks`` are pinned per leaf until the copy lands, so nothing evicts or
    reuses them mid-copy. ``commit_hashes`` is set only for onloads, whose
    prefix-cache publish is deferred until the H2D has actually landed.
    """

    event: KVConnectorTransfer
    blocks: dict[str, list[LittleKVCacheBlock]]
    commit_hashes: list[bytes] | None = None


@dataclass(frozen=True)
class _PageCopy:
    """One leaf's page, fetched from another replica's device memory.

    ``src_replica`` is per page so leaves of the same hit may come from
    different replicas.
    """

    leaf_id: str
    dst_bid: int
    src_bid: int
    src_replica: int


def create_kv_group_coordinator(
    pools: Sequence[JengaBlockPool],
    leaf_ids: Sequence[str],
    group_id: KVCacheGroupId,
    page_size: int,
) -> KVGroupCoordinatorInterface:
    """Returns the coordinator matching the group's attention pattern."""
    if group_id.is_sliding_window():
        return SlidingWindowKVGroupCoordinator(
            pools=pools,
            leaf_ids=leaf_ids,
            group_id=group_id,
            page_size=page_size,
            window_size=group_id.window_size,
        )
    return FullKVGroupCoordinator(
        pools=pools, leaf_ids=leaf_ids, group_id=group_id
    )


@dataclass(frozen=True)
class KVLeaf:
    """One cache's share of the pool, and the pages each request holds in it.

    Every leaf of a group is written in lockstep, so a request's row is the
    same length in all of them.
    """

    leaf_id: str
    group_id: KVCacheGroupId


@dataclass(frozen=True)
class KVLeafInfo:
    """How one cache tiles a huge block, and which group it belongs to.

    ``ratio`` is the number of little blocks per huge block.
    """

    ratio: int
    group_id: KVCacheGroupId


class JengaBlockManager:
    """Assigns blocks to requests and manages prefix cache hits."""

    def __init__(
        self,
        leaf_infos: Mapping[str, KVLeafInfo],
        num_huge_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        num_replicas: int = 1,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        max_num_input_tokens: int | None = None,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
        connector: KVConnector | None = None,
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]] | None = None,
        enable_dp_cross_replica_prefix_copy: bool = True,
    ) -> None:
        self._block_size = block_size
        self._enable_prefix_caching = enable_prefix_caching
        self._only_use_kv_connector_last_level_cache = (
            _resolve_only_use_kv_connector_last_level_cache()
        )
        self._kv_hash_algo = kv_hash_algo
        self._kv_hash_seed = kv_hash_seed
        self._max_num_input_tokens = max_num_input_tokens
        self._num_draft_tokens = num_draft_tokens
        self._num_draft_tokens_per_step = num_draft_tokens_per_step
        self._metrics = KVCacheMetrics()
        self._num_replicas = num_replicas

        # Per-replica device memory, keyed by leaf, used to copy committed
        # prefix blocks between replicas. None when the caller has no buffers.
        self._replica_kv_memory = replica_kv_memory
        self._cross_replica_copy_enabled = (
            enable_dp_cross_replica_prefix_copy
            and num_replicas > 1
            and replica_kv_memory is not None
        )

        ratios = {leaf_id: leaf.ratio for leaf_id, leaf in leaf_infos.items()}
        self.pools = [
            JengaBlockPool(num_huge_blocks, ratios) for _ in range(num_replicas)
        ]

        self._leaves = {
            leaf_id: KVLeaf(
                leaf_id,
                group_id=leaf_info.group_id,
            )
            for leaf_id, leaf_info in leaf_infos.items()
        }

        # Deduplicate in first-appearance order rather than through a set:
        # groups are claimed and scanned in this order, so a set would make
        # which cache gets which page depend on the run's hash seed.
        group_ids = dict.fromkeys(
            leaf.group_id for leaf in self._leaves.values()
        )
        self._groups: dict[KVCacheGroupId, KVGroupCoordinatorInterface] = {
            group_id: create_kv_group_coordinator(
                self.pools,
                [
                    leaf.leaf_id
                    for leaf in self._leaves.values()
                    if leaf.group_id == group_id
                ],
                group_id,
                self._block_size,
            )
            for group_id in group_ids
        }

        self._req_to_hashes: dict[RequestID, list[bytes]] = {}
        self._req_to_committed_idx: dict[RequestID, int] = {}
        self._req_to_replica: dict[RequestID, int] = {}

        # State for the KVConnector.
        self._connector = connector

        self._pending_transfers: list[list[_PendingTransfer]] = [
            [] for _ in range(num_replicas)
        ]
        # Runs of newly committed hashes awaiting an `offload` call.
        self._pending_offloads: list[list[list[bytes]]] = [
            [] for _ in range(num_replicas)
        ]

    # ============================================================================
    # Request Lifecycle APIs
    # ============================================================================

    @traced
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""
        req_id = ctx.request_id
        existing = self._req_to_replica.get(req_id)
        if existing is not None:
            raise ValueError(
                f"Request is already claimed, on replica {existing}: {req_id}"
            )
        self._req_to_replica[req_id] = replica_idx
        self._req_to_hashes[req_id] = []
        self._req_to_committed_idx[req_id] = 0
        for group in self._groups.values():
            group.claim(req_id)

    @property
    def groups(self) -> Mapping[KVCacheGroupId, KVGroupCoordinatorInterface]:
        """The cache groups this manager owns."""
        return self._groups

    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is registered with the block manager."""
        return ctx.request_id in self._req_to_replica

    @traced
    def release(self, ctx: TextContext) -> None:
        """Frees every page the request holds, in every cache."""
        req_id = ctx.request_id
        replica_idx = self._replica_of(ctx)

        for group in self._groups.values():
            group.release(req_id, replica_idx)

        del self._req_to_replica[req_id]
        del self._req_to_hashes[req_id]
        del self._req_to_committed_idx[req_id]

    # ============================================================================
    # Allocation & Reuse APIs
    # ============================================================================

    @traced
    def alloc(self, ctx: TextContext) -> KVConnectorTransfer:
        """Gives every cache the pages the next forward needs.

        Raises:
            InsufficientBlocksError: If the pool cannot serve all of the
                request's caches at once, in which case it draws nothing.
        """
        replica_idx = self._replica_of(ctx)

        # Drain landed transfers first: their pinned pages are only returned
        # here, so skipping this leaks every offload source for the run.
        self.poll_transfers()

        transfer = self._reuse_blocks_from_prefix_cache(ctx, replica_idx)

        self._metrics.input_tokens += ctx.tokens.active_length

        # Check if we have enough blocks available to satisfy the demand.
        pool = self.pools[replica_idx]
        num_required_blocks = self._num_required_blocks(ctx)
        demand: dict[str, int] = {}
        for group in self._groups.values():
            demand.update(
                group.blocks_to_allocate(ctx.request_id, num_required_blocks)
            )
        if not pool.can_satisfy_demand(demand):
            raise InsufficientBlocksError(
                f"Serving {demand} needs more huge blocks than are available"
            )

        for group in self._groups.values():
            group.grow(ctx.request_id, num_required_blocks, replica_idx)

        return transfer

    @traced
    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Claims a dummy request and points it at the replica's null page."""
        self.claim(ctx, replica_idx)
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
        )
        num_required_blocks = ceildiv(seq_len, self._block_size)
        for group in self._groups.values():
            group.grow_with_padding(
                ctx.request_id, num_required_blocks, replica_idx
            )

    @traced
    def step(self, ctx: TextContext) -> None:
        """Records what the forward just wrote, and slides every window."""
        replica_idx = self._replica_of(ctx)
        pool = self.pools[replica_idx]
        if self._enable_prefix_caching:
            self._commit_blocks_into_prefix_cache(ctx, pool)

        num_filled_blocks = self._num_filled_blocks(ctx)
        for group in self._groups.values():
            group.advance(ctx.request_id, num_filled_blocks, replica_idx)

    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Counts the number of prefix cache hits for a request per replica.

        Read-only. With cross-replica copies on, a block held by any replica
        counts as a hit, without checking there is room to copy it in.

        Returns:
            A list of PrefixCacheHits for each replica.
        """
        desired_hashes = self._compute_block_hashes(ctx, [])
        hit_counts: list[PrefixCacheHits] = []
        for replica_idx in range(self._num_replicas):
            num_hit_blocks = self._find_longest_device_prefix_cache_hit(
                desired_hashes, replica_idx, self._cross_replica_copy_enabled
            )
            # Ask the connector to load the hashes that are remaining.
            (num_hit_host_blocks, num_hit_disk_blocks) = (
                self._connector.count_cached_prefix(
                    desired_hashes[num_hit_blocks:]
                )
                if self._connector is not None
                else (0, 0)
            )
            hit_counts.append(
                PrefixCacheHits(
                    device_blocks=num_hit_blocks,
                    host_blocks=num_hit_host_blocks,
                    disk_blocks=num_hit_disk_blocks,
                )
            )
        return hit_counts

    def reset_prefix_cache(self) -> None:
        """Drops every commit no request is holding, in every cache."""
        for pool in self.pools:
            pool.reset_prefix_cache()
        if self._connector is not None:
            self._connector.reset_prefix_cache()

    # ============================================================================
    # KVConnector
    # ============================================================================

    @traced
    def offload(self, replica_idx: int = 0) -> None:
        """Offloads the recently produced KV states to the connector."""
        connector = self._connector
        if connector is None:
            return
        pool = self.pools[replica_idx]
        for hashes in self._pending_offloads[replica_idx]:
            src: dict[str, list[LittleKVCacheBlock]] = {
                leaf_id: [] for leaf_id in self._leaves
            }
            block_hashes: list[bytes] = []
            for block_hash in hashes:
                if any(
                    block_hash not in pool.prefix_caches[leaf_id]
                    for leaf_id in self._leaves
                ):
                    # Evicted from at least one leaf since it was committed, so
                    # the row is no longer whole: truncate the run here.
                    break
                for leaf_id in self._leaves:
                    block = pool.prefix_caches[leaf_id][block_hash]
                    src[leaf_id].append(block)
                block_hashes.append(block_hash)
            if not block_hashes:
                continue
            bids = {
                leaf_id: [b.bid for b in bids] for leaf_id, bids in src.items()
            }
            event = connector.offload(
                bids,
                block_hashes,
                replica_idx=replica_idx,
            )
            # An asynchronous connector reads these pages on its own engine, so
            # pin them until the D2H lands. A synchronous one is already done.
            if not event.is_complete():
                self._track_transfer(event, src, replica_idx)
        self._pending_offloads[replica_idx].clear()

    def poll_transfers(self) -> None:
        """Drains completed async transfers on the scheduler thread.

        For each pending async transfer, check if it has completed. If so, we
        may commit the hashes into the prefix cache and then unpin the blocks.
        """
        for replica_idx, pending_list in enumerate(self._pending_transfers):
            if not pending_list:
                continue
            pool = self.pools[replica_idx]
            still_pending: list[_PendingTransfer] = []
            for pending in pending_list:
                if not pending.event.is_complete():
                    still_pending.append(pending)
                    continue
                if pending.commit_hashes is not None:
                    self._commit_onloaded_blocks(
                        pool, pending.blocks, pending.commit_hashes
                    )
                for leaf_blocks in pending.blocks.values():
                    for block in leaf_blocks:
                        pool.free_block(block)
            self._pending_transfers[replica_idx] = still_pending

    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether any async transfer is in flight on the replica."""
        return bool(self._pending_transfers[replica_idx])

    @traced
    def _lookup_connector_prefix_cache_hit(
        self,
        desired: Sequence[bytes],
        replica_idx: int,
        hint: bytes | None,
    ) -> tuple[int, dict[str, list[LittleKVCacheBlock]], KVConnectorTransfer]:
        """Loads the desired hashes from the connector's prefix cache.

        Fresh device pages are allocated for the hashes the connector can
        serve and filled by its ``load``. The connector may serve some but not
        all of the desired hashes.

        ``hint`` is the request's raw ``dkv_cache_hint``, passed through to the
        connector; see :meth:`KVConnector.load`.

        Eg:
        ```
        > desired_hashes = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]
        > staging_blocks = {
        >   'sliding_window_group(1024)': [42, 43, 44, 45],
        >   'full_group': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        > }
        > event = connector.load(staging_blocks, desired_hashes)
        > # Cache hit on 8 of 10 hashes
        > assert event.g0_blocks_per_leaf == {
        >   'sliding_window_group(1024)': [0, 0, 0, 0, 42, 43, 44, 45],
        >   'full_group': [1, 2, 3, 4, 5, 6, 7, 8]
        > }
        ```

        Returns:
            A tuple containing:
                The number of blocks loaded from the connector
                The blocks that the contents are being loaded into, per leaf
                The transfer tracking the copy
        """
        connector = self._connector
        empty: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [] for leaf_id in self._leaves
        }
        if connector is None or not desired:
            return 0, empty, CompletedTransfer.load(list(self._leaves))

        pool = self.pools[replica_idx]

        # Only try to load from connector if we have enough device blocks to
        # hold the desired hashes.
        num_blocks_needed = {
            leaf_id: group.num_blocks_needed_for_connector_load(len(desired))
            for group in self._groups.values()
            for leaf_id in group.leaf_ids
        }
        # If there are insufficient blocks available, we will be unable to schedule
        # this request. Return zero connector cache hits and let the caller raise
        # InsufficientBlocksError after releasing all resources owned by this request.
        if not pool.can_satisfy_demand(num_blocks_needed):
            return 0, empty, CompletedTransfer.load(list(self._leaves))

        staging_blocks = {
            leaf_id: [pool.alloc_block(leaf_id) for _ in range(num_blocks)]
            for leaf_id, num_blocks in num_blocks_needed.items()
        }
        event = connector.load(
            {
                leaf_id: [b.bid for b in leaf_blocks]
                for leaf_id, leaf_blocks in staging_blocks.items()
            },
            desired,
            replica_idx=replica_idx,
            hint=hint,
        )
        # Note that for SWA groups, we expect the connector to pad the blocks
        # with 0 to denote the null blocks. As such, the length of the blocks
        # for each leaf should be the same.
        unique_num_loaded = {
            len(blocks) for blocks in event.g0_blocks_per_leaf.values()
        }
        if len(unique_num_loaded) != 1:
            raise ValueError(
                "Expected all leaves to have the same number of loaded blocks, "
                f"but got {event.g0_blocks_per_leaf} from KVConnector.load(...)"
            )
        num_loaded = unique_num_loaded.pop()

        # Give the surplus blocks back.
        for leaf_id in self._leaves:
            all_bids = {b.bid for b in staging_blocks[leaf_id]}
            loaded_bids = {bid for bid in event.g0_blocks_per_leaf[leaf_id]}
            unused = all_bids - loaded_bids
            for bid in unused:
                block = pool.block(leaf_id, bid)
                pool.free_block(block)

        if num_loaded == 0:
            return 0, empty, CompletedTransfer.load(list(self._leaves))

        logger.debug(
            f"KVConnector loaded {num_loaded} / {len(desired)} hashes. Blocks: {event.g0_blocks_per_leaf}"
        )
        loaded_blocks = {
            leaf_id: [pool.block(leaf_id, bid) for bid in blocks]
            for leaf_id, blocks in event.g0_blocks_per_leaf.items()
        }

        loaded_hashes = list(desired[:num_loaded])
        if event.is_complete():
            self._commit_onloaded_blocks(pool, loaded_blocks, loaded_hashes)
        else:
            self._track_transfer(
                event, loaded_blocks, replica_idx, commit_hashes=loaded_hashes
            )
        return num_loaded, loaded_blocks, event

    def _commit_onloaded_blocks(
        self,
        pool: JengaBlockPool,
        blocks: Mapping[str, list[LittleKVCacheBlock]],
        hashes: Sequence[bytes],
    ) -> None:
        """Publishes landed onload pages, skipping any hash already in cache."""
        for leaf_id, leaf_blocks in blocks.items():
            prefix_cache = pool.prefix_caches[leaf_id]
            for block, block_hash in zip(leaf_blocks, hashes, strict=True):
                if (
                    block.block_hash is None
                    and block_hash not in prefix_cache
                    and not block.is_null
                ):
                    pool.commit_into_prefix_cache(block_hash, block)

    def _track_transfer(
        self,
        event: KVConnectorTransfer,
        blocks: Mapping[str, list[LittleKVCacheBlock]],
        replica_idx: int,
        commit_hashes: list[bytes] | None = None,
    ) -> None:
        """Tracks an async connector transfer and the pages it pins.

        The pin (a ``touch``) keeps the pages out of the eviction and free
        paths while the copy engine is still reading or writing them.

        The KVCache will poll the transfer via ``poll_transfers``. When the
        transfer completes, the pages will be unpinned.
        """
        if not any(blocks.values()):
            return
        pool = self.pools[replica_idx]
        for leaf_blocks in blocks.values():
            for block in leaf_blocks:
                pool.touch(block)
        self._pending_transfers[replica_idx].append(
            _PendingTransfer(
                event=event,
                blocks={
                    leaf_id: list(leaf_blocks)
                    for leaf_id, leaf_blocks in blocks.items()
                },
                commit_hashes=commit_hashes,
            )
        )

    # ============================================================================
    # Misc
    # ============================================================================

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns the block manager's metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Resets the block manager's metrics to zero."""
        self._metrics = KVCacheMetrics()

    @property
    def effective_max_seq_length(self) -> int | None:
        """Returns the longest single-request sequence every leaf could serve simultaneously.

        ``None`` if there is no finite bound (every leaf is sliding-window
        and each window fits the budget). Binary search over
        :meth:`_fits_in_cache`, which is monotonic in ``seq_len``: a leaf's
        block requirement never decreases -- it grows for full attention,
        and plateaus once a sliding window is fully covered. That makes this
        equivalent to the largest ``seq_len`` for which every leaf still
        fits, without needing to simulate how the shared huge-block budget
        gets partitioned across leaves.
        """
        # Nothing can outrun a single leaf handed the entire budget at the
        # most generous ratio, so that is a safe ceiling to search up to.
        # Still fitting AT the ceiling means there is no finite bound: every
        # leaf must be sliding-window, since only their demand plateaus.
        upper_bound = (
            self.huge_block_count().total
            * self._block_size
            * max(self.pools[0].cache_ratios.values())
        )
        search_space = upper_bound + 2
        idx = bisect_left(
            range(search_space),
            True,
            key=lambda seq_len: not self._fits_in_cache(seq_len),
        )
        return (idx - 1) if idx < search_space else None

    def _blocks_demanded(self, seq_len: int) -> dict[str, int]:
        """Returns the pages each leaf holds for a ``seq_len``-token request."""
        demand = {}
        for leaf_id, leaf in self._leaves.items():
            num_blocks = ceildiv(seq_len, self._block_size)
            if leaf.group_id.is_sliding_window():
                # A window only keeps its most recent tokens resident, so its
                # demand stops growing once the window itself is covered.
                num_blocks = min(
                    num_blocks,
                    ceildiv(leaf.group_id.window_size, self._block_size),
                )
            demand[leaf_id] = num_blocks
        return demand

    def _fits_in_cache(self, seq_len: int) -> bool:
        """Whether an empty pool could serve one ``seq_len``-token request."""
        return self.pools[0].can_satisfy_demand(
            self._blocks_demanded(seq_len), at_capacity=True
        )

    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the pages the request holds, per leaf.

        Distinct from :meth:`PagedKVCacheManagerInterface.get_req_blocks`
        (a single flat ``list[int]``, sized for one leaf): Jenga's caches
        aren't interchangeable, so this returns one list per leaf instead.
        """
        self._replica_of(ctx)
        return {
            leaf_id: [
                block.bid
                for block in self._groups[leaf.group_id].blocks_of(
                    ctx.request_id
                )[leaf_id]
            ]
            for leaf_id, leaf in self._leaves.items()
            if leaf.group_id in self._groups
        }

    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Returns block IDs the request holds for the first leaf.

        TODO: Delete this method after refactoring downstream callers.
        """
        return next(iter(self.get_req_blocks_per_leaf(ctx).values()))

    def huge_block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the huge-block occupancy for the given replica.

        ``total`` excludes the null block (huge block 0), which every cache
        shares and which is never allocable.
        """
        pool = self.pools[replica_idx]
        return BlockCount(
            free=len(pool.free_huge_blocks), total=len(pool.huge_blocks)
        )

    def little_block_count(self, replica_idx: int = 0) -> dict[str, BlockCount]:
        """Returns each leaf's little-block occupancy for the given replica."""
        pool = self.pools[replica_idx]
        total_huge_blocks = len(pool.huge_blocks)
        return {
            leaf_id: BlockCount(
                free=pool.num_free_blocks(leaf_id),
                total=total_huge_blocks * pool.cache_ratios[leaf_id],
            )
            for leaf_id in self._leaves
        }

    # ============================================================================
    # Internal
    # ============================================================================

    def _replica_of(self, ctx: TextContext) -> int:
        """Returns the replica the request was claimed on."""
        replica_idx = self._req_to_replica.get(ctx.request_id)
        if replica_idx is None:
            raise ValueError(
                f"Request is not claimed, so it holds no pages to work with: "
                f"{ctx.request_id}"
            )
        return replica_idx

    def _num_required_blocks(self, ctx: TextContext) -> int:
        """Returns how many pages the next forward needs the request to hold."""
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
            max_num_input_tokens=self._max_num_input_tokens,
        )
        return ceildiv(seq_len, self._block_size)

    def _num_filled_blocks(self, ctx: TextContext) -> int:
        """Returns how many of the request's blocks a forward has filled."""
        return ctx.tokens.processed_length // self._block_size

    @traced
    def _compute_block_hashes(
        self, ctx: TextContext, existing_hashes: Sequence[bytes]
    ) -> list[bytes]:
        return compute_block_hashes(
            ctx,
            existing_hashes,
            self._block_size,
            self._kv_hash_algo,
            self._kv_hash_seed,
        )

    @traced
    def _compute_hashes_for_request(self, ctx: TextContext) -> list[bytes]:
        """Extends the request's hash chain to cover its newest full blocks."""
        hashes = self._req_to_hashes[ctx.request_id]
        hashes.extend(self._compute_block_hashes(ctx, hashes))
        return hashes

    def _find_longest_device_prefix_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool,
    ) -> int:
        """Returns how many blocks every group can serve at once."""
        # Each group answers under the run the others already allowed, so the
        # run is settled once every group has accepted it in turn.
        groups = list(self._groups.values())
        accepted = 0
        turn = 0
        while desired_hashes and accepted < len(groups):
            num_hit_blocks = groups[turn].longest_cache_hit(
                desired_hashes, replica_idx, allow_cross_replica
            )
            accepted = (
                accepted + 1 if num_hit_blocks == len(desired_hashes) else 1
            )
            desired_hashes = desired_hashes[:num_hit_blocks]
            turn = (turn + 1) % len(groups)

        return len(desired_hashes)

    def _lookup_device_prefix_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> tuple[dict[str, list[LittleKVCacheBlock]], int]:
        """Finds the longest run of ``desired_hashes`` the device cache holds.

        Returns:
            The hit pages of each leaf, and how many blocks long the run is.
            The caller splices the pages onto the request.
        """
        if self._only_use_kv_connector_last_level_cache:
            return {leaf_id: [] for leaf_id in self._leaves}, 0

        num_hit_blocks = self._find_longest_device_prefix_cache_hit(
            desired_hashes, replica_idx, self._cross_replica_copy_enabled
        )

        if self._cross_replica_copy_enabled:
            self._copy_prefix_from_peers(
                desired_hashes[:num_hit_blocks], replica_idx
            )
            # The copy is best effort, so re-read what actually landed. Every
            # page that fit is local now, and one that did not shortens the
            # hit instead of breaking it.
            num_hit_blocks = self._find_longest_device_prefix_cache_hit(
                desired_hashes, replica_idx, False
            )

        hit_hashes = desired_hashes[:num_hit_blocks]
        hit_blocks: dict[str, list[LittleKVCacheBlock]] = {}
        for group in self._groups.values():
            hit_blocks.update(
                group.claim_hit_blocks(
                    hit_hashes,
                    replica_idx,
                )
            )

        return hit_blocks, num_hit_blocks

    def _copy_prefix_from_peers(
        self, hit_hashes: Sequence[bytes], replica_idx: int
    ) -> None:
        """Copies the pages of ``hit_hashes`` that only peer replicas hold.

        Best effort. The caller re-reads the prefix cache afterwards, so a
        page that does not fit shortens the hit rather than corrupting it.
        Every page taken is held until the end, so allocating for a later one
        cannot evict an earlier one.
        """
        pool = self.pools[replica_idx]
        held: list[LittleKVCacheBlock] = []
        copies: list[_PageCopy] = []
        staged: list[tuple[bytes, LittleKVCacheBlock]] = []
        try:
            for group in self._groups.values():
                for block_hash in group.claimable_hashes(hit_hashes):
                    src = group.find_replica_with_hash(
                        block_hash, replica_idx, allow_cross_replica=True
                    )
                    if src is None:
                        break
                    try:
                        for leaf_id in group.leaf_ids:
                            local = pool.prefix_caches[leaf_id].get(block_hash)
                            if local is not None:
                                pool.touch(local)
                                held.append(local)
                                continue
                            dst = pool.alloc_block(leaf_id)
                            held.append(dst)
                            copies.append(
                                _PageCopy(
                                    leaf_id=leaf_id,
                                    dst_bid=dst.bid,
                                    src_bid=self.pools[src]
                                    .prefix_caches[leaf_id][block_hash]
                                    .bid,
                                    src_replica=src,
                                )
                            )
                            staged.append((block_hash, dst))
                    except InsufficientBlocksError:
                        # No room left; this group copies no further. A hash
                        # left half-committed is harmless: every page that
                        # landed holds the right bytes for its leaf, and a hit
                        # needs every leaf, so the re-read just will not count
                        # it.
                        break

            num_bytes = self._submit_page_copies(replica_idx, copies)

            # Publish only once the copies are enqueued: a committed hash is
            # visible to every request, so its page must already be filled.
            for block_hash, dst in staged:
                pool.commit_into_prefix_cache(block_hash, dst)
            self._metrics.cross_replica_bytes_copied += num_bytes
        finally:
            for block in held:
                pool.free_block(block)

    def _submit_page_copies(
        self, dst_replica: int, copies: Sequence[_PageCopy]
    ) -> int:
        """Batch-copies pages from other replicas into ``dst_replica``.

        Every page view is built before the call so host-side getitem work
        does not sit between the peer copies. Destinations all live on
        ``dst_replica``, so mixed sources still cost one submission.

        Returns:
            How many bytes the copy moves. Each leaf is measured at its own
            page size, which is the only comparable unit here: one block of
            the prefix is a page in every leaf of its group, and those pages
            are not the same size.
        """
        if not copies:
            return 0
        assert self._replica_kv_memory is not None

        num_bytes = 0
        dst_pages: list[Buffer] = []
        src_pages: list[Buffer] = []
        for page in copies:
            src_unit = self._replica_kv_memory[page.src_replica][page.leaf_id]
            dst_unit = self._replica_kv_memory[dst_replica][page.leaf_id]
            # Every TP shard is fanned out with its own point-to-point copy.
            for src_buf, dst_buf in zip(
                src_unit.buffers, dst_unit.buffers, strict=True
            ):
                dst_pages.append(dst_buf[page.dst_bid, :])
                src_pages.append(src_buf[page.src_bid, :])
            num_bytes += src_unit.bytes_per_page * len(src_unit.buffers)

        batch_inplace_copy(dst_pages, src_pages)
        return num_bytes

    def _reuse_blocks_from_prefix_cache(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> KVConnectorTransfer:
        """Splices the longest prefix-cache hit into the request.

        The device hit is extended by whatever the connector's external tiers
        still hold; both runs are spliced on the same way, so they skip the
        same tokens and are committed at the same index.

        Returns:
            The transfer tracking the onload's copy, already complete when
            nothing was onloaded.
        """
        # Only try to reuse blocks if the ctx is fresh (ie: no tokens are processed)
        if not self._enable_prefix_caching or ctx.tokens.processed_length != 0:
            return CompletedTransfer.load(list(self._leaves))

        self._compute_hashes_for_request(ctx)

        committed_blocks = (
            self._req_to_committed_idx[ctx.request_id] // self._block_size
        )
        desired_hashes = self._req_to_hashes[ctx.request_id][committed_blocks:]

        hit_blocks, num_hit_blocks = self._lookup_device_prefix_cache_hit(
            desired_hashes, replica_idx
        )
        # Ask the connector to load the hashes that are remaining.
        num_loaded, loaded_blocks, transfer = (
            self._lookup_connector_prefix_cache_hit(
                desired_hashes[num_hit_blocks:],
                replica_idx,
                hint=ctx.dkv_cache_hint,
            )
        )
        num_reused = num_hit_blocks + num_loaded

        self._metrics.device_blocks_served += num_hit_blocks
        self._metrics.cache_tokens += num_reused * self._block_size
        ctx.cached_prefix_length = num_reused * self._block_size
        ctx.cached_prefix_external_length = num_loaded * self._block_size

        if num_reused == 0:
            return transfer

        # The hit resumes at the committed index, so whatever the request holds
        # beyond it belongs to a chunk that is about to be re-planned.
        self._release_uncommitted_blocks(ctx, replica_idx)

        for group in self._groups.values():
            group.extend(ctx.request_id, hit_blocks, loaded_blocks, replica_idx)

        committed_idx = (
            self._req_to_committed_idx[ctx.request_id]
            + num_reused * self._block_size
        )
        self._req_to_committed_idx[ctx.request_id] = committed_idx

        skip_amount = committed_idx - ctx.tokens.processed_length
        ctx.tokens.skip_processing(skip_amount)
        assert ctx.tokens.active_length >= 1, (
            "No active tokens after prefix caching! A 100% prefix cache hit "
            "leaves nothing to compute logits from, so compute_block_hashes "
            "must never hash the last token."
        )
        return transfer

    def _release_uncommitted_blocks(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Drops the blocks past the committed index, in every cache."""
        committed_idx = self._req_to_committed_idx[ctx.request_id]
        num_committed_blocks = committed_idx // self._block_size

        for group in self._groups.values():
            group.shrink_to_fit(
                ctx.request_id, num_committed_blocks, replica_idx
            )

        delta = ctx.tokens.processed_length - committed_idx
        if delta > 0:
            ctx.tokens.rewind_processing(delta)
        elif delta < 0:
            ctx.tokens.skip_processing(-delta)

    def _commit_blocks_into_prefix_cache(
        self, ctx: TextContext, pool: JengaBlockPool
    ) -> None:
        """Publishes the blocks the forward filled to the prefix caches."""
        req_hashes = self._compute_hashes_for_request(ctx)
        first_block = (
            self._req_to_committed_idx[ctx.request_id] // self._block_size
        )

        last_block = min(self._num_filled_blocks(ctx), len(req_hashes))

        for leaf in self._leaves.values():
            paged = self._groups.get(leaf.group_id)
            if paged is None:
                continue
            req_blocks = paged.blocks_of(ctx.request_id)[leaf.leaf_id]
            for block_idx in range(first_block, last_block):
                block = req_blocks[block_idx]
                # A twin block already serving this hash means the bytes are
                # already published, so the request adopts it and drops its own.
                twin = pool.get_or_commit_into_prefix_cache(
                    req_hashes[block_idx], block
                )
                if twin is not None:
                    req_blocks[block_idx] = twin

        self._req_to_committed_idx[ctx.request_id] = (
            last_block * self._block_size
        )

        # Queue the newly committed run for the next `offload`. Every leaf
        # commits the same hashes in lockstep, so one run covers them all.
        if self._connector is not None and last_block > first_block:
            self._pending_offloads[self._replica_of(ctx)].append(
                list(req_hashes[first_block:last_block])
            )
