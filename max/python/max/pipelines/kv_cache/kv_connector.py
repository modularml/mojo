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

"""Connector protocol and transfer handle for external KV cache tiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics


class TransferDirection(str, Enum):
    """Whether a KV connector transfer is an onload or an offload."""

    LOAD = "load"
    OFFLOAD = "offload"


@dataclass(frozen=True, slots=True)
class BlockCount:
    """A point-in-time snapshot of a block pool's occupancy.

    Used for the device (G0) pools, whose blocks are the unit the manager
    actually allocates in. The connector's external tiers report
    :class:`ByteCount` instead.
    """

    free: int
    total: int

    @property
    def used(self) -> int:
        """Returns the number of blocks currently in use."""
        return self.total - self.free

    @property
    def used_pct(self) -> float:
        """Returns the percentage of blocks currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0`` (e.g. a tier with no configured
        capacity), rather than dividing by zero.
        """
        return 100 * self.used / self.total if self.total else 0.0

    @property
    def free_pct(self) -> float:
        """Returns the percentage of blocks not currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0``, rather than dividing by zero.
        """
        return 100 * self.free / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class ByteCount:
    """A point-in-time snapshot of an external cache tier's occupancy, in bytes.

    Bytes rather than blocks because a connector's host and disk tiers are
    byte budgets that the operator sizes in bytes (``host_offload_max_gb``,
    ``disk_offload_max_gb``), and because their block width need not match the
    device's -- an MLA-replicated unit is stored once on the host and broadcast
    back on load. Bytes also rate directly against PCIe and disk bandwidth.
    """

    free: int
    total: int

    @property
    def used(self) -> int:
        """Returns the bytes currently in use."""
        return self.total - self.free

    @property
    def used_pct(self) -> float:
        """Returns the percentage of bytes currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0`` (e.g. a tier with no configured
        capacity), rather than dividing by zero.
        """
        return 100 * self.used / self.total if self.total else 0.0

    @property
    def free_pct(self) -> float:
        """Returns the percentage of bytes not currently in use, in ``[0, 100]``.

        ``0`` when ``total`` is ``0``, rather than dividing by zero.
        """
        return 100 * self.free / self.total if self.total else 0.0


@runtime_checkable
class KVConnectorTransfer(Protocol):
    """Handle for one KV connector transfer (an onload or an offload).

    Returned by :meth:`KVConnector.load` and :meth:`KVConnector.offload`. It lets
    the manager overlap a transfer with GPU compute: the transfer owns the
    ``g0`` device block ids it touches (a load's H2D destinations, an offload's
    D2H sources), and the manager keeps those blocks pinned until
    :meth:`is_complete` returns ``True``, then unpins them exactly once (see the
    scheduler's ``poll_transfers`` loop).

    Two completion models cross this handle:

    * Synchronous / stream-ordered connectors (dKV) issue their copies on -- or
      GPU-ordered ahead of -- the forward stream and return
      :class:`CompletedTransfer`. ``is_complete`` is immediately ``True``, so the
      manager commits the reused prefix at once and never holds the request out
      of a batch.
    * Asynchronous connectors (the Rust ``rust_tiered`` connector) issue their
      copies on a separate copy engine and return a handle whose
      ``is_complete`` flips only once the copy lands. The manager pins the blocks,
      defers committing an onloaded prefix, and cordons the request out of the
      batch until then -- so the GPU runs other ready work while the copy is in
      flight.

    ``is_complete`` must be a cheap, side-effect-free poll (a plain atomic /
    ``cudaEventQuery``-style check), safe to call every scheduler iteration.
    """

    @property
    def direction(self) -> TransferDirection:
        """Whether this transfer is a ``load`` (onload) or an ``offload``."""
        ...

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        """Device (G0) block ids this transfer pins until it completes, per leaf.

        Each leaf should have the same number of blocks. For SWA groups, the list
        of blocks may be null padded with block_id=0.
        """
        ...

    def is_complete(self) -> bool:
        """Returns whether the transfer has completed. Never blocks."""
        ...

    def synchronize(self) -> None:
        """Blocks until the transfer completes. Used only at drain/shutdown."""
        ...


class CompletedTransfer:
    """An already-complete :class:`KVConnectorTransfer`.

    Returned by synchronous / stream-ordered connectors (dKV):
    their copies ride the forward stream or are GPU-ordered ahead of it, so from
    the manager's perspective the transfer is already done -- no pinning, no
    deferred commit, no cordoning. ``g0_blocks_per_leaf`` still reports the device blocks
    the connector loaded (fewer than requested is allowed), which the manager
    uses to trim any surplus staging blocks.
    """

    def __init__(
        self,
        direction: TransferDirection,
        leaves: Sequence[str] | None = None,
        g0_blocks: Sequence[int] | None = None,
    ) -> None:
        self._direction: TransferDirection = direction

        leaves = leaves or []
        g0_blocks = g0_blocks or []
        self._g0_blocks_per_leaf = {leaf_id: g0_blocks for leaf_id in leaves}

    @property
    def direction(self) -> TransferDirection:
        """The transfer direction (``load`` or ``offload``)."""
        return self._direction

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        """The device blocks the connector loaded/offloaded, per leaf."""
        return self._g0_blocks_per_leaf

    def is_complete(self) -> bool:
        """Always ``True``: this transfer is already complete."""
        return True

    def synchronize(self) -> None:
        """No-op: this transfer is already complete."""
        return

    @classmethod
    def load(cls, leaves: Sequence[str] | None = None) -> CompletedTransfer:
        """Create a completed load transfer."""
        return cls(TransferDirection.LOAD, leaves)

    @classmethod
    def offload(cls, leaves: Sequence[str] | None = None) -> CompletedTransfer:
        """Create a completed offload transfer."""
        return cls(TransferDirection.OFFLOAD, leaves)


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via load/offload methods.

    All block hashes crossing this Protocol are in canonical bytes form:
    8 big-endian bytes for ahash64-family algos (including ``sha256_64``),
    32 bytes for full SHA-256 digests. The block hasher produces this
    canonical form directly, so callers pass the hashes through unchanged;
    a connector that needs a narrower wire encoding (e.g. dKV's 64-bit key)
    validates and converts at its own boundary.

    Required call ordering per inference step:
      1. connector.load()            # post loads on the main stream
      2. connector.wait_for_loads()  # order loads before the forward pass
      3. connector.offload()         # kick off this step's offloads
      4. [model executes]
      5. connector.wait_for_offloads()  # settle offloads posted this step

    ``wait_for_loads`` guarantees the forward pass reads loaded data, but not
    necessarily by blocking the host until it lands. A stream-ordered connector
    may instead enqueue a cross-stream wait so the compute stream is GPU-ordered
    after the loads and return without a host sync (the data can still be in
    flight on return, ordered ahead of the forward pass on the device). A
    host-polled connector blocks until the data has landed. Either way the model
    in step 4 sees the loaded KV.

    ``wait_for_offloads`` likewise need not block the host. A stream-ordered
    connector may defer marking each block readable until its copy lands, polled
    without a host sync, so a block offloaded this step can become readable on a
    later step. Correctness holds: a block is never published before its bytes
    are written.
    """

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        """Returns a mapping from leaf_id to group_id serviced by the connector."""
        ...

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVConnectorTransfer:
        """Load data from external cache into device blocks.

        Args:
            block_ids: Device block IDs to load data into per leaf. Each leaf
                may have a different number of blocks depending on the group type.
                For example, a full attn group may need 100 pages to load 100 hashes
                while a sliding window group may only need 8 pages.
            block_hashes: Hashes to load data for, in canonical bytes form
                (8 big-endian bytes for ahash64-family, 32 bytes for
                SHA-256).
            replica_idx: DP replica whose device buffers receive the loaded
                blocks. The external tier itself is replica-agnostic (keyed by
                hash); this only selects the H2D destination.
            hint: The request's ``dkv_cache_hint`` as raw JSON bytes, or
                ``None`` when it carried none. Opaque to the manager: only the
                dKV connector reads it, to route blocks to the peer that holds
                them. It never affects correctness, since an unusable hint
                costs a cache miss and nothing else.

        Returns:
            A :class:`KVConnectorTransfer` for the H2D copy. ``g0_blocks_per_leaf``
            reports the device blocks actually loaded (a prefix of the requested
            ``block_ids``; fewer than requested is allowed).
            Synchronous connectors return a :class:`CompletedTransfer`;
            asynchronous ones return a handle the manager polls before reading
            the loaded KV.
            Note that all leaves must have the same number of blocks. This number
            should be equal to the number of loaded hashes. For sliding window
            groups, the list of blocks may be null padded with block_id=0.
        """
        ...

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offload the device blocks to the external cache.

        The blocks form one ordered sequence. Every connector keys blocks
        purely by hash, so the order carries no parentage.

        Args:
            block_ids: Device block IDs to offload per leaf.
            block_hashes: Hashes for the blocks being offloaded, in prefix
                order. Canonical bytes form (8 big-endian bytes for
                ahash64-family, 32 bytes for SHA-256).
            replica_idx: DP replica whose device buffers source the offloaded
                blocks. The external tier itself is replica-agnostic.

        Returns:
            A :class:`KVConnectorTransfer` for the D2H copy; ``g0_blocks_per_leaf`` are
            the device source blocks the manager keeps pinned until it lands.
            Synchronous connectors return a :class:`CompletedTransfer`.
        """
        ...

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        """Refresh the external tier's recency for blocks served from device (G0).

        Best-effort and fire-and-forget: returns immediately, processes
        asynchronously, ignores the result, and never raises into the caller.
        A block served from the on-device prefix cache issues no other
        external-tier traffic, so without this its external-tier LRU recency
        can freeze and the tier can evict a block that is still hot on device.
        There is no companion barrier; a missed touch costs at most a later
        refetch, never correctness. No-op by default.

        Contract: pass the complete set in sequence order from the true root --
        the full sequence for a full-attention group, the full active window
        for a sliding-window group. Never a root-omitting slice: a partial
        touch reserves a later recency stamp and inverts eviction order (the
        omitted root ages below the touched subset and evicts first). Missing
        keys are tolerated, so it is always safe to pass the whole sequence.

        Args:
            block_hashes: Hashes of the device-served blocks, in canonical
                bytes form (8 big-endian bytes for ahash64-family, 32 bytes
                for SHA-256). Root-anchored and in sequence order (see the
                contract above).
            replica_idx: DP replica that served the blocks. The external tier
                is replica-agnostic (keyed by hash); this only selects the
                client.
        """
        return

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        """Counts contiguous leading blocks resident in this connector's tiers.

        Walks ``block_hashes`` in prefix order, counting blocks the connector
        holds in its external tiers, and stops at the first block found in no
        tier. Implementations must be strictly read-only: no transfers,
        allocations, or LRU updates. Counts reflect index presence only and
        may ignore transient constraints that the ``load`` path enforces
        (e.g. free staging blocks required to onboard a disk hit).

        Args:
            block_hashes: Block hashes in prefix order, in canonical bytes
                form (see the class docstring).

        Returns:
            ``(num_host_blocks, num_disk_blocks)`` counted along the
            contiguous run. Connectors without a cheap local index (e.g.
            remote block stores) return ``(0, 0)``.
        """
        return (0, 0)

    def wait_for_loads(self) -> None:
        """Order all posted loads before the forward pass.

        .. deprecated::
            Superseded by the :class:`KVConnectorTransfer` model: asynchronous
            connectors report load completion through
            :meth:`KVConnectorTransfer.is_complete` (the manager's
            ``poll_transfers`` loop plus the scheduler's cordon), so the forward
            never reads KV that has not landed without any pre-forward barrier.
            Retained only for the dKV connector, which still posts its READs in
            :meth:`load` and orders them here; a no-op for every other connector.

        Called before the forward pass. Connectors that report completion
        through :class:`KVConnectorTransfer` need no work here. The dKV connector
        does one of two things by transport: for a co-located (same-host) load it
        enqueues a cross-stream CUDA event wait so the compute stream is
        GPU-ordered after the H2D copies and returns without a host sync (the
        copy may still be draining, ordered ahead of the forward pass); for a
        remote NIXL load it host-polls the off-stream RDMA to completion. No-op
        by default.
        """
        return

    def wait_for_offloads(self) -> None:
        """Settle offloads posted since the last call.

        .. deprecated::
            The post-forward counterpart of :meth:`wait_for_loads`; see its note.
            Asynchronous connectors settle offloads through
            :meth:`KVConnectorTransfer.is_complete` / ``poll_transfers``. Retained
            only for the dKV connector; a no-op for every other connector.

        Called after the forward pass. No-op by default. For a co-located
        (same-host) offload the dKV connector defers marking the block readable
        until its D2H copy lands, polled without a host sync, so the block can
        become readable on a later step; for a remote NIXL offload it host-polls
        the RDMA to completion and marks the block readable inline. A block is
        never marked readable before its bytes land.
        """
        return

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        return

    # Optional properties with default implementations
    @property
    def host_byte_count(self) -> ByteCount:
        """Host tier occupancy in bytes. Empty (0 of 0) if not applicable."""
        return ByteCount(free=0, total=0)

    @property
    def disk_byte_count(self) -> ByteCount:
        """Disk tier occupancy in bytes. Empty (0 of 0) if not applicable."""
        return ByteCount(free=0, total=0)

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()

    def reset_metrics(self) -> None:
        """Reset per-batch transfer counters after the scheduler samples them."""
        return
