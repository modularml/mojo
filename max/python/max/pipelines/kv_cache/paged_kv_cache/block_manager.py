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

"""Block manager for PagedAttention KVCache.

Handles allocating new blocks for requests as well as prefix caching/reuse.
This is done very efficiently and largely avoids Python memory allocations.

This logic is largely borrowed from vLLM v1:
- https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_manager.py#L1
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_utils.py#L1
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TokenHashOverride,
)
from max.pipelines.kv_cache.kv_connector import (
    CompletedTransfer,
    KVConnector,
    KVConnectorTransfer,
)
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from .block_pool import BlockPool
from .block_utils import (
    InsufficientBlocksError,
    KVCacheBlock,
    KVHashAlgo,
    hash_request_tokens,
)

logger = logging.getLogger("max.pipelines")


def compute_block_hashes(
    ctx: TextContext,
    existing_hashes: Sequence[bytes],
    block_size: int,
    kv_hash_algo: KVHashAlgo,
    kv_hash_seed: bytes | None,
) -> list[bytes]:
    """Computes block hashes for the request beyond ``existing_hashes``.

    Unlike :meth:`compute_hashes_for_request`, this reads and writes no
    per-request state, so it is safe to call for requests that are not
    (and may never be) claimed on this replica — e.g. when computing
    prefix-cache overlap for data-parallel routing.

    Args:
        ctx: The request context.
        existing_hashes: Hashes already computed for the request's leading
            blocks; new hashes chain onto the last entry. Pass an empty
            sequence to hash from the start of the prompt.
        block_size: How many tokens one block holds.
        kv_hash_algo: Which hash the keys are built with.
        kv_hash_seed: Seed isolating one deployment's keys from another's, or
            ``None`` for the unseeded default.

    Returns:
        Hashes for the newly hashed full blocks; empty if no additional
        full block is hashable.
    """
    num_hashed_tokens = len(existing_hashes) * block_size
    # We do not compute the hash for the last token because it is ineligible
    # for prefix caching. This is because 100% prefix cache hit is illegal
    # and will result in a 0 input tokens for the request. Hence the minus 1.
    num_hashable_tokens = len(ctx.tokens) - 1
    num_unhashed_tokens = num_hashable_tokens - num_hashed_tokens
    if num_unhashed_tokens < block_size:
        return []

    parent_hash_value: bytes | None = None
    if len(existing_hashes) > 0:
        parent_hash_value = existing_hashes[-1]

    unhashed_tokens = ctx.tokens[num_hashed_tokens:num_hashable_tokens]

    token_hash_overrides: list[TokenHashOverride] = []
    if isinstance(ctx, TextAndVisionContext):
        for img in ctx.images:
            if img.image_hash is None:
                raise ValueError(
                    "hash_request_tokens requires `image_hash` to be present. Found None."
                )
            token_hash_overrides.append(
                TokenHashOverride(
                    token_idx=img.start_idx,
                    token_hash=img.image_hash,
                    source="image",
                )
            )
        token_hash_overrides.extend(ctx.token_hash_overrides)

    return hash_request_tokens(
        token_ids=unhashed_tokens,
        block_size=block_size,
        parent_hash=parent_hash_value,
        prefix_length=num_hashed_tokens,
        token_hash_overrides=token_hash_overrides,
        algo=kv_hash_algo,
        seed=kv_hash_seed,
        salt=ctx.cache_salt,
    )


@dataclass
class _PendingTransfer:
    """A device-side async transfer tracked on the main (scheduler) thread.

    ``blocks`` are the device blocks pinned for the transfer's duration (a
    load's H2D destinations, an offload's D2H sources); they are unpinned when
    ``event`` completes. ``commit_hashes`` is set for onloads only: the onloaded
    blocks are committed into the device prefix cache on completion (deferred so
    a concurrent request cannot read them before the H2D lands), keyed by these
    hashes in ``blocks`` order. Only asynchronous connectors (``rust_tiered``)
    produce these; synchronous connectors' transfers are already complete and
    never tracked.
    """

    event: KVConnectorTransfer
    blocks: list[KVCacheBlock]
    commit_hashes: list[bytes] | None = None


def _compute_seq_len(
    ctx: TextContext,
    num_draft_tokens: int,
    num_draft_tokens_per_step: int = 0,
    max_num_input_tokens: int | None = None,
) -> int:
    # Each term accounts for one category of tokens that need a KV slot:
    #
    #   ctx.tokens                    : prompt + tokens generated so far
    #   maybe_accepted_draft_tokens   : draft tokens being verified in the
    #                                   *previous* batch (overlap scheduler);
    #                                   conservative: assume all are accepted
    #   2 * num_draft_tokens          : drafts to verify *next* batch
    #                                   + drafts written *during* that batch
    #   1                             : one regular decode step
    #   -1                            : the last generated token has no KV
    #                                   entry yet (overlap's in-flight
    #                                   placeholder, or the just-sampled token)
    #
    # Block-draft correction (DFlash): the draft model's ``forward_block``
    # writes ``num_draft_tokens_per_step + 1`` positions in a single batched
    # call, starting at ``bumped_cache_length = pre_cache_length +
    # commit_lengths``. Compared to the autoregressive-draft accounting above
    # (which assumes one draft KV per step), that's an extra position past the
    # bonus token — exactly the slot the ``- 1`` here was reclaiming under the
    # "last generated token has no KV entry" optimization. For block drafts
    # that bonus position *does* get a KV entry (forward_block writes it as
    # part of the speculative tail), so we add it back. Guard on
    # ``num_draft_tokens > 0`` so a disabled request (0, 0) never matches this
    # by coincidence.
    block_draft_extra = (
        1
        if num_draft_tokens > 0
        and num_draft_tokens_per_step == num_draft_tokens
        else 0
    )
    # Avoid allocating blocks for the entire sequence when chunked prefill is used.
    # This is critically important for SWA.
    active_length = ctx.tokens.active_length
    if max_num_input_tokens is not None:
        active_length = min(active_length, max_num_input_tokens)
    seq_len = ctx.tokens.processed_length + active_length
    seq_len += (
        len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + 2 * num_draft_tokens
        + 1
        + block_draft_extra
        - 1
    )
    return seq_len


def _resolve_only_use_kv_connector_last_level_cache() -> bool:
    """Resolve whether to only use the KVConnector last level cache.

    When this is set, the device prefix cache will be disabled. All KVCache hits
    will strictly be served from the KVConnector. This is primarily used for
    testing and benchmarking the performance of the KVConnector. Do NOT use this
    flag in production.

    With the local connector, the last level cache is the host memory. With the
    tiered connector, the last level cache is the disk.
    """
    enabled = os.getenv(
        "MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE", "0"
    ).lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if enabled:
        logger.info(
            "Detected MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE flag, only using KVConnector prefix cache."
        )
    return enabled


@dataclass(frozen=True)
class PrefixCacheHits:
    """Per-tier counts of a request's contiguous cached prefix on one replica.

    Counts are blocks, not tokens, and describe one contiguous run from the
    start of the request's block hash chain: the leading ``device_blocks``
    are resident in the device prefix cache, and ``host_blocks`` plus
    ``disk_blocks`` continue that run from the connector's external tiers.
    """

    device_blocks: int = 0
    """Leading blocks resident in the device prefix cache."""

    host_blocks: int = 0
    """Blocks continuing the run that are resident in the host tier."""

    disk_blocks: int = 0
    """Blocks continuing the run that are resident in the disk tier."""

    @property
    def total_blocks(self) -> int:
        """Total contiguous cached blocks across all tiers."""
        return self.device_blocks + self.host_blocks + self.disk_blocks


class BlockManager:
    """Manages allocation and deallocation of paged KV cache blocks.

    A single ``BlockManager`` is responsible for every data-parallel (DP)
    replica. Device (GPU) memory is physically partitioned per replica, so the
    manager owns one :class:`BlockPool` per replica (``device_block_pools``).
    A request lives on exactly one replica for its lifetime; replica-scoped
    methods take a ``replica_idx`` (defaulting to ``0`` for the common
    single-replica case).

    The device prefix cache is shared across replicas in the sense that a
    lookup for a request on replica ``B`` can hit a block physically resident
    on replica ``A``: matching pages are copied device-to-device onto ``B``
    (a single ``batch_inplace_copy``). The
    ``enable_dp_cross_replica_prefix_copy`` config flag turns these copies
    off, in which case cross-replica reuse falls through to the shared
    external tier instead. External tiers (host/disk) are reached through a
    single ``KVConnector`` shared by every replica; each ``load``/``offload``
    passes the ``replica_idx`` so the connector can select that replica's
    device buffers.
    """

    @traced
    def __init__(
        self,
        total_num_blocks: int,
        block_size: int,
        connector: KVConnector,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
        *,
        num_replicas: int = 1,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]] | None = None,
        enable_dp_cross_replica_prefix_copy: bool = True,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("BlockManager requires at least one replica")

        self.total_num_blocks = total_num_blocks
        self.block_size = block_size
        self.num_replicas = num_replicas

        self.kv_hash_algo: KVHashAlgo = kv_hash_algo
        self.kv_hash_seed: bytes | None = kv_hash_seed

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # A single connector for external cache tiers (host memory, disk, dKV),
        # shared across every replica. It owns host memory / a host block pool
        # and the H2D/D2H transfers; ``load``/``offload`` take a ``replica_idx``
        # to select the device endpoint.
        self.connector = connector

        # Per-replica offload-ready device memory units, used to copy committed
        # prefix blocks device-to-device between replicas. Required (non-None)
        # when ``num_replicas > 1`` and prefix caching is on.
        self._replica_kv_memory: list[list[KVCacheMemory]] | None = (
            [list(units) for units in replica_kv_memory]
            if replica_kv_memory is not None
            else None
        )

        # Whether a cross-replica device prefix-cache hit may be served by a
        # device-to-device copy. Requires per-replica device memory handles;
        # the enable_dp_cross_replica_prefix_copy config flag turns it off so
        # that cross-replica reuse falls through to the shared external tier
        # instead.
        self._cross_replica_copy_enabled = (
            self._replica_kv_memory is not None
            and enable_dp_cross_replica_prefix_copy
        )

        # Ordered offload sequences pending delivery to each replica's
        # connector. Each entry is one contiguous run of newly-committed block
        # hashes, in prefix order.
        self._pending_offloads: list[list[list[bytes]]] = [
            [] for _ in range(self.num_replicas)
        ]

        # In-flight async transfers (host/disk onloads and offloads) per
        # replica, each pinning its device blocks until it completes. Only
        # asynchronous connectors (``rust_tiered``) populate this; drained on
        # the main thread by ``poll_transfers``.
        self._pending_transfers: list[list[_PendingTransfer]] = [
            [] for _ in range(self.num_replicas)
        ]

        # One pool of device blocks per replica.
        self.device_block_pools: list[BlockPool] = [
            BlockPool(
                total_num_blocks,
                enable_runtime_checks=enable_runtime_checks,
            )
            for _ in range(self.num_replicas)
        ]

        # Mapping from request ID to the replica it is assigned to. A request
        # lives on a single replica for its whole lifetime.
        self.req_to_replica: dict[RequestID, int] = {}

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: dict[RequestID, list[KVCacheBlock]] = defaultdict(
            list
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_hashes: dict[RequestID, list[bytes]] = defaultdict(list)

        # Mapping from request ID to committed index (number of tokens
        # committed into the prefix cache). This replaces reliance on
        # the context's committed_idx.
        self.req_to_committed_idx: dict[RequestID, int] = defaultdict(int)

        # Metrics for the KV cache.
        self._metrics = KVCacheMetrics()

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

        # Whether to only use the KVConnector last level cache.
        # When this is set, the device prefix cache will be disabled. This is
        # primarily used for testing and benchmarking the performance of the
        # KVConnector.
        self._only_use_kv_connector_last_level_cache = (
            _resolve_only_use_kv_connector_last_level_cache()
        )

    @property
    def device_block_pool(self) -> BlockPool:
        """The replica-0 device block pool (single-replica convenience)."""
        return self.device_block_pools[0]

    @traced
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""
        request_id = ctx.request_id
        existing = self.req_to_replica.get(request_id)
        if existing is not None:
            raise ValueError(
                f"Request is already claimed, on replica {existing}: "
                f"{request_id}"
            )
        self.req_to_replica[request_id] = replica_idx

    def _replica_of(self, ctx: TextContext) -> int:
        """Returns the replica the request was claimed on."""
        replica_idx = self.req_to_replica.get(ctx.request_id)
        if replica_idx is None:
            raise ValueError(
                f"Request is not claimed, so it holds no pages to work with: "
                f"{ctx.request_id}"
            )
        return replica_idx

    @traced
    def step(self, ctx: TextContext) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(ctx)

        self.assert_runtime_invariants(ctx)

    @traced
    def compute_hashes_for_request(
        self,
        ctx: TextContext,
    ) -> None:
        """Computes the block hashes for the request."""
        hashes = self.req_to_hashes[ctx.request_id]
        new_hashes = compute_block_hashes(
            ctx,
            hashes,
            self.block_size,
            self.kv_hash_algo,
            self.kv_hash_seed,
        )
        hashes.extend(new_hashes)

    @traced
    def reuse_blocks_from_prefix_cache(
        self,
        ctx: TextContext,
    ) -> tuple[int, KVConnectorTransfer]:
        """Reuses blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW.

        Args:
            ctx: The request context.

        Returns:
            ``(skip_amount, event)`` where ``skip_amount`` is the number of
            tokens skipped due to prefix-cache reuse (0 when no blocks were
            reused) and ``event`` is the async onload transfer for the reused
            prefix -- an already-complete :class:`CompletedTransfer` when nothing
            was onloaded asynchronously (the common case: device hits and
            synchronous connectors), so callers can just poll ``is_complete()``.
        """
        # Reject an unclaimed request here rather than deeper in: the early
        # returns below skip every other call that would resolve its replica.
        self._replica_of(ctx)
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching or ctx.tokens.active_length == 1:
            return 0, CompletedTransfer.load()

        # Identify a request's first admission so we record one cache-hit
        # observation per request, not one per chunked-prefill chunk.
        is_first_admission = ctx.tokens.processed_length == 0

        req_blocks = self.req_to_blocks[ctx.request_id]

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Query prefix cache for full blocks.
        prefix_cache_blocks, load_event, num_external_blocks = (
            self.get_full_blocks_from_prefix_cache(ctx)
        )

        if len(prefix_cache_blocks) > 0:
            # Update metrics.
            self._metrics.cache_tokens += (
                len(prefix_cache_blocks) * self.block_size
            )

            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            prev_committed_idx = self.req_to_committed_idx[ctx.request_id]
            new_committed_idx = (
                prev_committed_idx + len(prefix_cache_blocks) * self.block_size
            )
            self.req_to_committed_idx[ctx.request_id] = new_committed_idx

            skip_amount = new_committed_idx - ctx.tokens.processed_length
            ctx.tokens.skip_processing(skip_amount)
            assert ctx.tokens.active_length >= 1, (
                "No active tokens after prefix caching! "
                "We should never get 100% prefix cache hit rate. "
                "Something went wrong!"
            )
            if is_first_admission:
                ctx.cached_prefix_length = skip_amount
                external_tokens = num_external_blocks * self.block_size
                # The connector's blocks are the tail of the committed prefix,
                # and `release_uncommitted_blocks` above re-synced
                # `processed_length` to the committed index, so `skip_amount`
                # covers every block just spliced in. Assert rather than clamp:
                # a clamp would silently under-report `external` and inflate
                # `g0` if that ever stopped holding.
                assert external_tokens <= skip_amount, (
                    f"external prefix {external_tokens} tokens exceeds the "
                    f"{skip_amount} tokens skipped"
                )
                ctx.cached_prefix_external_length = external_tokens
            return skip_amount, load_event

        if is_first_admission:
            ctx.cached_prefix_length = 0
            ctx.cached_prefix_external_length = 0
        return 0, load_event

    @traced
    def _count_full_blocks_from_prefix_cache(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Returns the count of device blocks with the desired hashes.

        A hash counts as a device hit if it is resident in *any* replica's
        device prefix cache, because a cross-replica hit is served by a
        device-to-device copy onto ``replica_idx`` rather than a recompute.
        When cross-replica copies are unavailable or disabled (see
        ``enable_dp_cross_replica_prefix_copy``), only ``replica_idx``'s
        own cache counts, matching what the reuse path can actually serve.
        """
        local_cache = self.device_block_pools[replica_idx].prefix_cache
        device_prefix_cache_hits = []
        desired_host_hashes = []
        for hash_value in desired_hashes:
            if self._cross_replica_copy_enabled:
                _, block = self._find_block_in_any_replica(
                    hash_value, replica_idx
                )
            else:
                block = local_cache.get(hash_value)
            if block is not None:
                # Device hashes with prefix cache hit (local or cross-replica)
                device_prefix_cache_hits.append(hash_value)
            else:
                # Record potential host hash
                desired_host_hashes.append(hash_value)

        # Ignoring host cache hits in this calculation as it may be expensive
        # to compute. Eg: due to querying an external process.
        device_prefix_cache_hit_count = len(device_prefix_cache_hits)

        return device_prefix_cache_hit_count

    def _find_block_in_any_replica(
        self, block_hash: bytes, preferred_replica: int
    ) -> tuple[int, KVCacheBlock | None]:
        """Finds a committed block for ``block_hash`` on any replica.

        The preferred (local) replica is checked first so that local hits never
        incur a cross-replica copy. Returns ``(replica_idx, block)`` for a hit,
        or ``(preferred_replica, None)`` when no replica has the block.
        """
        local_cache = self.device_block_pools[preferred_replica].prefix_cache
        block = local_cache.get(block_hash)
        if block is not None:
            return preferred_replica, block
        for replica_idx in range(self.num_replicas):
            if replica_idx == preferred_replica:
                continue
            block = self.device_block_pools[replica_idx].prefix_cache.get(
                block_hash
            )
            if block is not None:
                return replica_idx, block
        return preferred_replica, None

    @traced
    def _get_full_blocks_from_device_prefix_cache(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> list[KVCacheBlock]:
        """Returns device blocks on ``replica_idx`` with the desired hashes.

        Blocks resident in ``replica_idx``'s own prefix cache are reused
        directly. Blocks committed on a *different* replica are materialized
        onto ``replica_idx`` via a device-to-device copy into a freshly
        allocated block, which is then committed into the local prefix cache so
        subsequent requests on this replica hit locally (SERVOPT-1500). When
        cross-replica copies are disabled via
        ``enable_dp_cross_replica_prefix_copy``, the chain stops at the
        first local miss so the external tier (host/disk) can serve the rest.
        """
        if self._only_use_kv_connector_last_level_cache:
            return []

        local_pool = self.device_block_pools[replica_idx]
        local_cache = local_pool.prefix_cache

        blocks: list[KVCacheBlock] = []
        # Cross-replica copies grouped by source replica: one batched
        # device-to-device transfer per group rather than one round trip per
        # page. Commit into the local prefix cache is deferred until after
        # enqueue so a concurrent lookup cannot hit an empty destination page.
        pending_copies: dict[int, list[tuple[bytes, KVCacheBlock, int]]] = {}
        for block_hash in desired_hashes:
            local_block = local_cache.get(block_hash)
            if local_block is not None:
                local_pool.touch(local_block)
                blocks.append(local_block)
                self._metrics.device_blocks_served += 1
                continue

            # Local miss: a cross-replica hit can only be served when
            # device-to-device copies are enabled; otherwise stop the prefix
            # chain here and let the external tier serve the rest.
            if not self._cross_replica_copy_enabled:
                break

            # Look for the block on another replica.
            src_replica, src_block = self._find_block_in_any_replica(
                block_hash, replica_idx
            )
            if src_block is None:
                break

            # A cross-replica hit needs a free local block to copy into. If
            # none is available, stop the prefix chain here (it must remain
            # contiguous).
            if local_pool.num_free_blocks == 0:
                break

            # Allocate and queue the copy; commit only after enqueue (below)
            # so others cannot hit an empty page.
            dst_block = self.allocate_device_block(replica_idx)
            pending_copies.setdefault(src_replica, []).append(
                (block_hash, dst_block, src_block.bid)
            )
            blocks.append(dst_block)
            self._metrics.cross_replica_blocks_copied += 1

        # Enqueue batched D2D copies, then publish destinations into the local
        # prefix cache so subsequent requests on this replica hit locally.
        for src_replica, pending in pending_copies.items():
            self._copy_blocks_across_replicas(
                dst_replica=replica_idx,
                src_replica=src_replica,
                block_id_pairs=[
                    (dst_block.bid, src_block_id)
                    for _, dst_block, src_block_id in pending
                ],
            )
        for pending in pending_copies.values():
            for block_hash, dst_block, _ in pending:
                local_pool.commit_into_prefix_cache(block_hash, dst_block)

        return blocks

    def _copy_blocks_across_replicas(
        self,
        dst_replica: int,
        src_replica: int,
        block_id_pairs: Sequence[tuple[int, int]],
    ) -> None:
        """Batch-copies pages from ``src_replica`` to ``dst_replica``.

        Pages across every unit go into one ``batch_inplace_copy``; the driver
        groups them by destination device and submits one batch per device.

        All page views are built before the call so getitem host work does not
        sit between the peer copies.

        ``block_id_pairs`` is a sequence of ``(dst_block_id, src_block_id)``.
        """
        assert self._replica_kv_memory is not None
        src_units = self._replica_kv_memory[src_replica]
        dst_units = self._replica_kv_memory[dst_replica]
        if not block_id_pairs:
            return

        dst_pages: list[Buffer] = []
        src_pages: list[Buffer] = []
        for src_unit, dst_unit in zip(src_units, dst_units, strict=True):
            # Every shard is fanned out with an independent point-to-point copy
            # (no broadcast collective).
            for src_buf, dst_buf in zip(
                src_unit.buffers, dst_unit.buffers, strict=True
            ):
                for dst_id, src_id in block_id_pairs:
                    dst_pages.append(dst_buf[dst_id, :])
                    src_pages.append(src_buf[src_id, :])
            self._metrics.cross_replica_bytes_copied += (
                src_unit.bytes_per_page
                * len(src_unit.buffers)
                * len(block_id_pairs)
            )

        batch_inplace_copy(dst_pages, src_pages)

    @traced
    def _get_full_blocks_from_host_prefix_cache(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
        *,
        hint: bytes | None,
    ) -> tuple[list[KVCacheBlock], KVConnectorTransfer]:
        """Onloads device blocks with the desired hashes from the connector.

        The device blocks are newly allocated and initialized with the contents
        of the host/external blocks via the connector's ``load``.

        For a synchronous / stream-ordered connector the H2D is already ordered
        ahead of the forward, so the onloaded blocks are committed into the
        device prefix cache immediately and the (already-complete) transfer is
        returned unchanged.

        For an asynchronous connector (``rust_tiered``) the H2D runs on a
        separate copy engine: the destination blocks are pinned and their
        prefix-cache commit is deferred until the copy lands (``poll_transfers``)
        so a concurrent request in the same batch cannot read them early.

        ``hint`` is the request's raw ``dkv_cache_hint``, passed through to the
        connector; see :meth:`KVConnector.load`.

        Returns:
            ``(loaded_blocks, event)``; ``event`` is an already-complete
            :class:`CompletedTransfer` when nothing was onloaded or the onload
            was synchronous, otherwise the in-flight transfer.
        """
        connector = self.connector
        pool = self.device_block_pools[replica_idx]
        if not desired_hashes:
            return [], CompletedTransfer.load()

        # Limit by available device blocks.
        num_hashes_to_load = min(len(desired_hashes), pool.num_free_blocks)
        desired_hashes = desired_hashes[:num_hashes_to_load]
        blocks = [
            self.allocate_device_block(replica_idx)
            for _ in range(num_hashes_to_load)
        ]

        # Query connector for available blocks from host cache.
        block_ids = [b.bid for b in blocks]
        event = connector.load(
            {leaf_id: block_ids for leaf_id in connector.leaves},
            desired_hashes,
            replica_idx=replica_idx,
            hint=hint,
        )

        # The connector may load fewer blocks than requested; its event reports
        num_loaded = len(next(iter(event.g0_blocks_per_leaf.values())))
        for surplus_block in blocks[num_loaded:]:
            pool.free_block(surplus_block)
        loaded_blocks = blocks[:num_loaded]
        loaded_hashes = list(desired_hashes[:num_loaded])

        if not loaded_blocks:
            return [], CompletedTransfer.load()

        if event.is_complete():
            # Synchronous / stream-ordered connector (dKV): the
            # H2D is already ordered ahead of the forward, so commit into the
            # device prefix cache now.
            for block, block_hash in zip(
                loaded_blocks, loaded_hashes, strict=True
            ):
                if block_hash in pool.prefix_cache:
                    # With this env var set, we may transfer blocks already
                    # resident in the device prefix cache; skip the commit.
                    assert self._only_use_kv_connector_last_level_cache
                    continue
                pool.commit_into_prefix_cache(block_hash, block)
        else:
            # Asynchronous connector (``rust_tiered``): pin the destination
            # blocks and defer their prefix-cache commit until the H2D lands, so
            # a concurrent request cannot read them before the data arrives.
            self._track_transfer(
                event, loaded_blocks, replica_idx, commit_hashes=loaded_hashes
            )
        return loaded_blocks, event

    @traced
    def count_cached_prefix_blocks(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> PrefixCacheHits:
        """Counts contiguous leading blocks resident in this replica's caches.

        Walks ``block_hashes`` in prefix order through the device prefix
        cache and then the connector's external tiers (host, then disk per
        block), mirroring the reuse order of
        :meth:`get_full_blocks_from_prefix_cache`, and stops at the first
        block found in no tier.

        Unlike the reuse path this is strictly read-only: no blocks are
        allocated or onboarded, no LRU state is touched, and no per-request
        state is created, so it is safe to call for requests that are not
        (and may never be) claimed on this replica — e.g. for prefix-aware
        data-parallel routing. Counts reflect index presence only and ignore
        transient staging constraints the reuse path enforces (e.g. free
        device blocks to load into).

        Args:
            block_hashes: The request's block hash chain, in prefix order.
            replica_idx: The replica whose caches are probed.

        Returns:
            Per-tier counts of the contiguous cached prefix.
        """
        if not self.enable_prefix_caching:
            return PrefixCacheHits()

        num_device_hits = 0
        if not self._only_use_kv_connector_last_level_cache:
            device_prefix_cache = self.device_block_pools[
                replica_idx
            ].prefix_cache
            for block_hash in block_hashes:
                if self._cross_replica_copy_enabled:
                    _, blk = self._find_block_in_any_replica(
                        block_hash, replica_idx
                    )
                    present = blk is not None
                else:
                    present = block_hash in device_prefix_cache
                if not present:
                    break
                num_device_hits += 1

        remaining = block_hashes[num_device_hits:]
        num_host_hits = 0
        num_disk_hits = 0
        if remaining:
            num_host_hits, num_disk_hits = self.connector.count_cached_prefix(
                remaining
            )

        return PrefixCacheHits(
            device_blocks=num_device_hits,
            host_blocks=num_host_hits,
            disk_blocks=num_disk_hits,
        )

    @traced
    def get_full_blocks_from_prefix_cache(
        self, ctx: TextContext
    ) -> tuple[list[KVCacheBlock], KVConnectorTransfer, int]:
        """Gets the computed (cached) blocks for the request.

        Note that the computed blocks must be full.

        Returns:
            ``(blocks, event, num_external_blocks)`` where ``event`` is the async
            onload transfer for the host-tier portion, or an already-complete
            :class:`CompletedTransfer` when nothing was onloaded asynchronously
            (device / cross-replica hits are served synchronously).
            ``num_external_blocks`` is how many trailing entries of ``blocks``
            the connector served rather than the device prefix cache, for
            per-tier hit attribution.
        """
        assert self.enable_prefix_caching

        replica_idx = self._replica_of(ctx)
        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        uncommitted_hashes = req_hashes[num_committed_blocks:]

        # query the device prefix cache for full blocks
        device_blocks = self._get_full_blocks_from_device_prefix_cache(
            uncommitted_hashes, replica_idx
        )

        if self.connector.name == "NullConnector":
            return device_blocks, CompletedTransfer.load(), 0

        # remove the hashes that were found in the device prefix cache
        uncommitted_hashes = uncommitted_hashes[len(device_blocks) :]

        # query the host prefix cache for full blocks via connector
        host_blocks, load_event = self._get_full_blocks_from_host_prefix_cache(
            uncommitted_hashes, replica_idx, hint=ctx.dkv_cache_hint
        )

        # refresh the lru status of all hit hashes associated with the request.
        hit_hashes = req_hashes[
            : num_committed_blocks + len(device_blocks) + len(host_blocks)
        ]
        if hit_hashes:
            self.connector.touch(hit_hashes, replica_idx=replica_idx)

        return device_blocks + host_blocks, load_event, len(host_blocks)

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: TextContext,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            ctx: TextContext.
        """
        replica_idx = self._replica_of(ctx)
        pool = self.device_block_pools[replica_idx]
        req_blocks = self.req_to_blocks[ctx.request_id]
        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )

        # Count the number of tokens for which we know the values of and align
        # to the block size.
        num_computed_blocks = ctx.tokens.processed_length // self.block_size

        # Commit blocks into the prefix cache.
        for block_idx in range(num_committed_blocks, num_computed_blocks):
            block = req_blocks[block_idx]
            block_hash = req_hashes[block_idx]

            new_block = pool.get_or_commit_into_prefix_cache(block_hash, block)
            # If the block is already int the prefix cache, we skip the commit.
            # Then we overwrite the req blocks with the existing block that contains
            # the same contents.
            if new_block is not None:
                req_blocks[block_idx] = new_block

        # Queue the newly-committed blocks as one ordered offload sequence.
        if num_computed_blocks > num_committed_blocks:
            self._pending_offloads[replica_idx].append(
                req_hashes[num_committed_blocks:num_computed_blocks]
            )

        # Bump the committed index.
        self.req_to_committed_idx[ctx.request_id] = (
            num_computed_blocks * self.block_size
        )

    def offload(self, replica_idx: int = 0) -> None:
        """Offload the pending sequences to the replica's connector.

        Each pending sequence is delivered as one ordered ``offload`` call.
        Hashes are re-resolved to their current device blocks here; if a block
        was evicted since it was committed, the run is truncated at that point.
        """
        prefix_cache = self.device_block_pools[replica_idx].prefix_cache
        connector = self.connector
        for hashes in self._pending_offloads[replica_idx]:
            block_ids = []
            block_hashes = []
            src_blocks = []
            for block_hash in hashes:
                if block_hash not in prefix_cache:
                    # Block evicted since commit; truncate the run here.
                    break
                block = prefix_cache[block_hash]
                block_ids.append(block.bid)
                block_hashes.append(block_hash)
                src_blocks.append(block)
            if block_hashes:
                event = connector.offload(
                    {leaf_id: block_ids for leaf_id in connector.leaves},
                    block_hashes,
                    replica_idx=replica_idx,
                )
                # Asynchronous connector: pin the device source blocks until the
                # D2H lands so they are not evicted / reused mid-copy. Synchronous
                # connectors report the offload already complete (no pin).
                if not event.is_complete():
                    self._track_transfer(event, src_blocks, replica_idx)
        self._pending_offloads[replica_idx].clear()

    def _track_transfer(
        self,
        event: KVConnectorTransfer,
        blocks: list[KVCacheBlock],
        replica_idx: int,
        commit_hashes: list[bytes] | None = None,
    ) -> None:
        """Pins ``blocks`` and records an in-flight transfer to drain later.

        Pinning (a ``ref_cnt`` bump via ``touch``) keeps the blocks out of the
        eviction / free path until ``poll_transfers`` observes ``event``
        complete and unpins them. Only asynchronous connectors reach here; a
        no-op for an empty block list.
        """
        if not blocks:
            return
        pool = self.device_block_pools[replica_idx]
        for block in blocks:
            pool.touch(block)
        self._pending_transfers[replica_idx].append(
            _PendingTransfer(
                event=event, blocks=blocks, commit_hashes=commit_hashes
            )
        )

    def poll_transfers(self) -> None:
        """Drains completed async transfers on the main (scheduler) thread.

        For each completed transfer: commits any deferred onload blocks into the
        device prefix cache (now safe for cross-request reuse) and unpins the
        transfer's device blocks. Cheap to call every scheduler iteration -- an
        ``is_complete`` poll per in-flight transfer. A no-op unless an
        asynchronous connector (``rust_tiered``) is in use.
        """
        for replica_idx, pending_list in enumerate(self._pending_transfers):
            if not pending_list:
                continue
            pool = self.device_block_pools[replica_idx]
            still_pending: list[_PendingTransfer] = []
            for pending in pending_list:
                if not pending.event.is_complete():
                    still_pending.append(pending)
                    continue
                if pending.commit_hashes is not None:
                    for block, block_hash in zip(
                        pending.blocks, pending.commit_hashes, strict=True
                    ):
                        # Guard against a concurrent commit of the same hash
                        # (an ignored same-block onload race).
                        if (
                            block.block_hash is None
                            and block_hash not in pool.prefix_cache
                        ):
                            pool.commit_into_prefix_cache(block_hash, block)
                for block in pending.blocks:
                    pool.free_block(block)
            self._pending_transfers[replica_idx] = still_pending

    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether any async transfer is in flight on the replica."""
        return bool(self._pending_transfers[replica_idx])

    def release(self, ctx: TextContext) -> None:
        """Release the blocks for the request.

        Raises:
            ValueError: If the request is not claimed, which includes a second
                release: the claim is what names the pool the pages return to,
                so there is no such thing as releasing without one.
        """
        request_id = ctx.request_id
        pool = self.device_block_pools[self._replica_of(ctx)]
        blocks = self.req_to_blocks[request_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            pool.free_block(block)

        self.req_to_blocks.pop(request_id, None)
        self.req_to_hashes.pop(request_id, None)
        self.req_to_replica.pop(request_id, None)

        # Committed idx is only used with the prefix cache
        # therefore this may not always be in the dict.
        if request_id in self.req_to_committed_idx:
            del self.req_to_committed_idx[request_id]

    @traced
    def allocate_new_blocks(
        self,
        ctx: TextContext,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
    ) -> None:
        """Allocate new blocks for a request to accommodate additional tokens.

        Calculates the number of additional blocks needed based on the current sequence
        length, then allocates them from the device block pool.
        Validates that there are sufficient free blocks available and that the current
        blocks can accommodate the completed tokens.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_draft_tokens: Total draft tokens generated per speculative
                iteration. Zero for non-speculative decode.
            num_draft_tokens_per_step: Number of draft KV positions written
                per draft forward. Zero when speculative decoding is
                disabled; one for autoregressive drafts (``eagle``, ``mtp``);
                equal to ``num_draft_tokens`` for block drafts (``dflash``).
                Used by ``_compute_seq_len`` to size the cache for block
                drafts, whose ``forward_block`` writes one extra position
                past the bonus token.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        replica_idx = self._replica_of(ctx)
        pool = self.device_block_pools[replica_idx]

        # It is impossible to schedule this request, even if it was the only req
        # and could use the entire KV cache.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        total_kv_slots = self.total_num_blocks * self.block_size
        seq_len = (
            len(ctx.tokens)
            + len(ctx.spec_decoding_state.draft_tokens_to_verify)
            + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        )
        if seq_len > total_kv_slots:
            raise InsufficientBlocksError(
                f"Insufficient KV pages for a single request with {seq_len} tokens.\n"
                f"The KVCache has {self.total_num_blocks} pages with page size {self.block_size}. This is only enough to support {total_kv_slots} tokens.\n"
                "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
            )

        # Update metrics.
        self._metrics.input_tokens += ctx.tokens.active_length

        # Determine number of new blocks to allocate.
        num_new_blocks = self.num_blocks_to_allocate(
            ctx,
            num_draft_tokens,
            num_draft_tokens_per_step,
        )

        # Verify that processed tokens fit within the currently allocated blocks.
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        processed_length = ctx.tokens.processed_length
        assert ctx.tokens.processed_length <= (
            num_current_blocks * self.block_size
        ), (
            f"Expected at least {ceildiv(processed_length, self.block_size)} "
            f"blocks to store KV for {processed_length} committed tokens, but "
            f"only {num_current_blocks} are assigned."
        )

        # Check that we have enough free blocks to allocate the new blocks.
        if num_new_blocks > pool.num_free_blocks:
            free = pool.num_free_blocks
            in_use = self.total_num_blocks - free
            raise InsufficientBlocksError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue"
                f" (only {free} available; {in_use}/{self.total_num_blocks} blocks"
                f" currently in use)"
            )

        # Allocate new blocks.
        for _ in range(num_new_blocks):
            new_block = self.allocate_device_block(replica_idx)
            current_blocks.append(new_block)

    @traced
    def num_blocks_to_allocate(
        self,
        ctx: TextContext,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
    ) -> int:
        """Calculates the number of new blocks to allocate for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_draft_tokens: Total draft tokens generated per speculative
                iteration. Zero for non-speculative decode.
            num_draft_tokens_per_step: Number of draft KV positions written
                per draft forward. Zero when speculative decoding is
                disabled; one for autoregressive drafts (``eagle``, ``mtp``);
                equal to ``num_draft_tokens`` for block drafts (``dflash``).
                Used by ``_compute_seq_len`` to size the cache for block
                drafts, whose ``forward_block`` writes one extra position
                past the bonus token.

        Returns:
            The number of new blocks to allocate.
        """
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        current_seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens,
            num_draft_tokens_per_step,
        )
        num_required_blocks = ceildiv(current_seq_len, self.block_size)
        num_new_blocks = num_required_blocks - num_current_blocks

        return max(num_new_blocks, 0)

    @traced
    def allocate_device_block(self, replica_idx: int = 0) -> KVCacheBlock:
        """Allocates a single block from the replica's device block pool."""
        new_block, _ = self.device_block_pools[replica_idx].alloc_block()
        return new_block

    def release_uncommitted_blocks(
        self,
        ctx: TextContext,
    ) -> None:
        """Release the uncommitted blocks for the request."""
        pool = self.device_block_pools[self._replica_of(ctx)]
        req_blocks = self.req_to_blocks[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            pool.free_block(block)
        delta = (
            ctx.tokens.processed_length
            - self.req_to_committed_idx[ctx.request_id]
        )
        if delta > 0:
            ctx.tokens.rewind_processing(delta)
        elif delta < 0:
            ctx.tokens.skip_processing(-delta)

    def register_dummy_request(self, ctx: TextContext) -> None:
        """Maps a dummy request to the replica pool's reserved null block."""
        request_id = ctx.request_id
        assert self.req_to_blocks[request_id] == []
        self.req_to_blocks[request_id] = [
            self.device_block_pools[self._replica_of(ctx)].null_block
        ]

    @traced
    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Get the block ids for a request."""
        self._replica_of(ctx)
        return [block.bid for block in self.req_to_blocks[ctx.request_id]]

    @traced
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches for all replicas.

        Note: Host prefix cache reset is handled by the connector.
        """
        for pool in self.device_block_pools:
            pool.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns combined metrics for this manager and its connector."""
        return self._metrics + self.connector.metrics

    def reset_metrics(self) -> None:
        """Resets block-manager and connector transfer metrics to zero."""
        self._metrics = KVCacheMetrics()
        self.connector.reset_metrics()

    @traced
    def assert_runtime_invariants(self, ctx: TextContext) -> None:
        """Asserts runtime invariants when runtime checks are enabled."""
        if not self.enable_runtime_checks:
            return

        # Get the active block ids, partitioned by the replica that owns each
        # request's blocks.
        active_block_ids_by_replica: list[list[int]] = [
            [] for _ in range(self.num_replicas)
        ]
        for request_id, blocks in self.req_to_blocks.items():
            req_replica = self.req_to_replica.get(request_id, 0)
            for block in blocks:
                active_block_ids_by_replica[req_replica].append(block.bid)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0

        # Blocks pinned by an in-flight async transfer are held out of the free
        # queue while their copy lands, and an offload's sources outlive the
        # request that owned them, so they are neither free nor request-active.
        # Count them here or the pool's free + active == total check trips
        # whenever an asynchronous connector has a transfer in flight.
        for replica_idx_, pending_list in enumerate(self._pending_transfers):
            for pending in pending_list:
                for block in pending.blocks:
                    active_block_ids_by_replica[replica_idx_].append(block.bid)
                    assert block.ref_cnt > 0

        # Check that each block pool is consistent
        for pool, active_block_ids in zip(
            self.device_block_pools,
            active_block_ids_by_replica,
            strict=True,
        ):
            pool.assert_runtime_invariants(active_block_ids)

        # Get the request hashes and blocks
        req_hashes = self.req_to_hashes[ctx.request_id]
        req_blocks = self.req_to_blocks[ctx.request_id]

        # Check that the number of committed blocks for request is correct
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        num_committed = 0
        for block in req_blocks:
            if block.block_hash is None:
                break
            num_committed += 1
        assert num_committed == num_committed_blocks

        # Check that the req block hashes are consistent with req blocks
        for hash_value, block in zip(req_hashes, req_blocks, strict=False):
            assert block.block_hash is None or block.block_hash == hash_value
