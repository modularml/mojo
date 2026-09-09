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
"""Cache group coordinators: one per set of leaves written in lockstep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.modeling.types import RequestID

from .block_utils import LittleKVCacheBlock
from .jenga_block_pool import JengaBlockPool

__all__ = [
    "FullKVGroupCoordinator",
    "KVGroupCoordinatorInterface",
    "SlidingWindowKVGroupCoordinator",
]


@dataclass(frozen=True)
class KVGroupCoordinatorInterface:
    """Finds and claims the prefix-cache hit one group of caches can serve.

    The leaves of a group are written in lockstep, so a hash is only reusable
    when every one of them holds it, and how deep the group can resume depends
    on how far back its attention reads.
    """

    pools: Sequence[JengaBlockPool]
    leaf_ids: Sequence[str]
    group_id: KVCacheGroupId

    rows: dict[RequestID, dict[str, list[LittleKVCacheBlock]]] = field(
        default_factory=dict, kw_only=True
    )
    """Each live request's blocks, per leaf."""

    def _holds_every_leaf(self, block_hash: bytes, replica_idx: int) -> bool:
        """Whether every cache of the group has committed ``block_hash``.

        The leaves are written in lockstep, so a hash only some of them hold
        is unusable.
        """
        return all(
            block_hash in self.pools[replica_idx].prefix_caches[leaf_id]
            for leaf_id in self.leaf_ids
        )

    def find_replica_with_hash(
        self,
        block_hash: bytes,
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int | None:
        """The replica to serve ``block_hash`` to ``replica_idx`` from.

        ``replica_idx`` is checked first, so a hash it already holds is never
        copied. ``allow_cross_replica`` then widens the search to the other
        replicas, whose pages are copied over before use; without it a local
        miss is simply a miss.

        Returns:
            The replica to read the block from, or None when no replica the
            search covered holds it.
        """
        if self._holds_every_leaf(block_hash, replica_idx):
            return replica_idx
        if not allow_cross_replica:
            return None
        holders = [
            candidate
            for candidate in range(len(self.pools))
            if candidate != replica_idx
            and self._holds_every_leaf(block_hash, candidate)
        ]
        if not holders:
            return None
        # Taking the first holder would make the lowest-indexed one serve every
        # copy of a popular prefix. Keying the choice on the hash and the
        # destination splits the reads across the holders, and across the ranks
        # reading the same block, while staying stable per block.
        rotation = int.from_bytes(block_hash, "little") + replica_idx
        return holders[rotation % len(holders)]

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Which of ``desired_hashes`` this group would claim as a hit."""
        raise NotImplementedError("Subclasses must implement this method.")

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int:
        """Returns how many of ``desired_hashes`` this group can reuse.

        Counted from the start, so the answer is a prefix length.

        Asked again about a prefix it just returned, a group has to return
        that same length. The manager cycles through the groups, shortening
        the run to each answer until they all agree, so a group that
        shrinks a prefix it already accepted would take the run to nothing.

        Args:
            desired_hashes: The blocks the request wants, starting at its
                committed index.
            replica_idx: Which replica's pool to read.
            allow_cross_replica: Whether hashes held only by another replica
                count as hits.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Claims the blocks for the given hashes."""
        raise NotImplementedError("Subclasses must implement this method.")

    def claim(self, req_id: RequestID) -> None:
        """Starts an empty row in every leaf of the group."""
        self.rows[req_id] = {leaf_id: [] for leaf_id in self.leaf_ids}

    def release(self, req_id: RequestID, replica_idx: int) -> None:
        """Frees every block the request holds in this group."""
        pool = self.pools[replica_idx]
        for blocks in self.rows.pop(req_id, {}).values():
            # Free in reverse so the tail is evicted before the shared head.
            for block in reversed(blocks):
                pool.free_block(block)

    def blocks_of(
        self, req_id: RequestID
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Returns the request's blocks, per leaf."""
        return self.rows[req_id]

    def extend(
        self,
        req_id: RequestID,
        hit_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        loaded_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        replica_idx: int,
    ) -> None:
        """Appends the blocks a prefix hit found to the end of each row.

        The device blocks come first, then the ones an external tier
        loaded, matching the order their hashes were hashed in.

        Args:
            req_id: The request to extend.
            hit_blocks: Blocks found in this device's cache, per leaf.
            loaded_blocks: Blocks loaded from an external tier, per leaf.
            replica_idx: Which pool the blocks came from.
        """
        row = self.rows[req_id]
        for leaf_id in self.leaf_ids:
            row[leaf_id].extend([*hit_blocks[leaf_id], *loaded_blocks[leaf_id]])

    def grow_with_padding(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Points every row at the null block."""
        pool = self.pools[replica_idx]
        self.rows[req_id] = {
            leaf_id: [pool.null_little_blocks[leaf_id]] * num_required_blocks
            for leaf_id in self.leaf_ids
        }

    def shrink_to_fit(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Drops the blocks past the committed index."""
        pool = self.pools[replica_idx]
        for req_blocks in self.rows[req_id].values():
            assert len(req_blocks) >= num_committed_blocks
            for _ in range(len(req_blocks) - num_committed_blocks):
                pool.free_block(req_blocks.pop())

    def _num_blocks_to_allocate(
        self, row: Sequence[LittleKVCacheBlock], num_required_blocks: int
    ) -> int:
        """Returns how many blocks the row still needs."""
        return max(num_required_blocks - len(row), 0)

    def blocks_to_allocate(
        self, req_id: RequestID, num_required_blocks: int
    ) -> dict[str, int]:
        """Returns how many blocks each leaf must be given."""
        row = self.rows[req_id]
        return {
            leaf_id: self._num_blocks_to_allocate(
                row[leaf_id], num_required_blocks
            )
            for leaf_id in self.leaf_ids
        }

    def grow(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Allocates the blocks every row still needs."""
        pool = self.pools[replica_idx]
        for leaf_id in self.leaf_ids:
            req_blocks = self.rows[req_id][leaf_id]
            for _ in range(
                self._num_blocks_to_allocate(req_blocks, num_required_blocks)
            ):
                req_blocks.append(pool.alloc_block(leaf_id))

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the blocks the group no longer reads, nulling their slots."""
        raise NotImplementedError("Subclasses must implement this method.")

    def num_blocks_needed_for_connector_load(self, num_hashes: int) -> int:
        """The number of blocks needed to service a connector cache hit for all hashes."""
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class FullKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group whose caches read their whole history."""

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int:
        """Returns the run of committed hashes from the root."""
        for num_hit_blocks, block_hash in enumerate(desired_hashes):
            if (
                self.find_replica_with_hash(
                    block_hash, replica_idx, allow_cross_replica
                )
                is None
            ):
                return num_hit_blocks
        return len(desired_hashes)

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Every hash: this group reads its whole history."""
        return desired_hashes

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts every block of the hit: the group reads its whole history."""
        pool = self.pools[replica_idx]
        rows: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [] for leaf_id in self.leaf_ids
        }
        for block_hash in desired_hashes:
            for leaf_id in self.leaf_ids:
                block = pool.prefix_caches[leaf_id][block_hash]
                pool.touch(block)
                rows[leaf_id].append(block)
        return rows

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Keeps every block: this group reads its whole history."""
        return

    def num_blocks_needed_for_connector_load(self, num_hashes: int) -> int:
        """The full group needs one block per hash."""
        return num_hashes


@dataclass(frozen=True)
class SlidingWindowKVGroupCoordinator(KVGroupCoordinatorInterface):
    """This group needs ``blocks_in_window`` sized run to serve a cache hit."""

    window_size: int
    page_size: int

    @property
    def _blocks_in_window(self) -> int:
        return self.group_id.blocks_in_window(self.page_size)

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int:
        """Returns the longest windowed cache hit we can serve.

        Computing eligible Prefix Cache hits for sliding window differs greatly
        from full attn. Recall that the window size includes the query token.
        Say the query token is idx=42 and the window size is 10. This means
        that the query token will attend to tokens from idx=32 to idx=41.

        For a concrete example:

        [X]: Token is in Prefix Cache
         . : Token is not in Prefix Cache
         ^ : Eligible Prefix Cache hit

          Tokens [A]  [B]   .   [D]  [E]  [F]   .    .   [I]  [J]  [K]  [L]  [M]
        w_size=1  ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^
        w_size=2  ^    ^         ^    ^    ^              ^    ^    ^    ^    ^
        w_size=3  ^    ^              ^    ^                   ^    ^    ^    ^
        w_size=4  ^    ^                   ^                        ^    ^    ^
        w_size=5  ^    ^                                                 ^    ^
        w_size=6  ^    ^                                                      ^
        w_size=7  ^    ^

        Notice that as window_size increases, the number of indices eligible for
        a cache hit decreases. Additionally, we can count consecutive runs of
        window_size-1 tokens to determine eligibility. For example, [DEF] is a
        run of 3 tokens so token F is a valid cache hit for w_size=4 and below.

        Additionally, partial window cache hits is possible if the run starts from
        the start of sequence. For example, [A] and [AB] are valid cache hits for
        any window size.

        Also window_size=1 is a degenerate case where we always get 100% cache
        hit rate since the query token does not attend to any historical tokens.
        """
        # This is a degenerate case. When window_size=1, we always get 100%
        # cache hit rate.
        if self._blocks_in_window == 0:
            return len(desired_hashes)

        run = 0
        for idx in range(len(desired_hashes) - 1, -1, -1):
            if (
                self.find_replica_with_hash(
                    desired_hashes[idx], replica_idx, allow_cross_replica
                )
                is None
            ):
                # The run is broken. Reset the run counter.
                run = 0
                continue
            run += 1
            # If the run is at least than the window size, we have a complete window.
            if run >= self._blocks_in_window:
                return idx + run
        # No complete window. The surviving run, if any, ends at index 0.
        # We can skip the blocks_in_window check in this case.
        return run

    def claimable_hashes(
        self, desired_hashes: Sequence[bytes]
    ) -> Sequence[bytes]:
        """Only the window: this group has slid past everything below it."""
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        return desired_hashes[low:]

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts the window ending at the hit and nulls every slot below it."""
        pool = self.pools[replica_idx]
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        if not all(
            self._holds_every_leaf(block_hash, replica_idx)
            for block_hash in desired_hashes[low:]
        ):
            low = len(desired_hashes)

        rows: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [pool.null_little_blocks[leaf_id]] * low
            for leaf_id in self.leaf_ids
        }
        for block_hash in desired_hashes[low:]:
            for leaf_id in self.leaf_ids:
                block = pool.prefix_caches[leaf_id][block_hash]
                pool.touch(block)
                rows[leaf_id].append(block)
        return rows

    def advance(
        self,
        req_id: RequestID,
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the blocks below the window, nulling their slots."""
        pool = self.pools[replica_idx]
        first_needed = max(0, num_committed_blocks - self._blocks_in_window)
        for leaf_id in self.leaf_ids:
            req_blocks = self.rows[req_id][leaf_id]
            null_block = pool.null_little_blocks[leaf_id]
            for idx in range(first_needed - 1, -1, -1):
                if req_blocks[idx].is_null:
                    break
                pool.free_block(req_blocks[idx])
                req_blocks[idx] = null_block

    def num_blocks_needed_for_connector_load(self, num_hashes: int) -> int:
        """The number of blocks needed to service a connector cache hit for all hashes."""
        return min(num_hashes, self._blocks_in_window)

    def extend(
        self,
        req_id: RequestID,
        hit_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        loaded_blocks: Mapping[str, Sequence[LittleKVCacheBlock]],
        replica_idx: int,
    ) -> None:
        """Drops the device blocks when the loaded run starts with a null.

        A null first loaded block means that block sits below the window,
        and so does everything before it, including every device block. The
        group will never read them again, so free them rather than hold
        pages nothing can use.
        """
        pool = self.pools[replica_idx]
        row = self.rows[req_id]
        for leaf_id in self.leaf_ids:
            hit = list(hit_blocks[leaf_id])
            loaded = loaded_blocks[leaf_id]
            if loaded and loaded[0].is_null:
                for block in hit:
                    pool.free_block(block)
                hit = [pool.null_little_blocks[leaf_id]] * len(hit)
            row[leaf_id].extend([*hit, *loaded])
