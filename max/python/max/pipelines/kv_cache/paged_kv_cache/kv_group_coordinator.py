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
from dataclasses import dataclass

from max.nn.kv_cache import KVCacheGroupId

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

    def is_in_prefix_cache(self, block_hash: bytes, replica_idx: int) -> bool:
        """Whether every cache of the group has committed ``block_hash``."""
        return all(
            block_hash in self.pools[replica_idx].prefix_caches[leaf_id]
            for leaf_id in self.leaf_ids
        )

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> int:
        """Returns how many of ``desired_hashes`` this group could resume from.

        Args:
            desired_hashes: The blocks the request wants, from its committed
                index up.
            replica_idx: Which pool to read.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Claims the blocks for the given hashes."""
        raise NotImplementedError("Subclasses must implement this method.")

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Returns the pages the group's attention just slid past.

        Args:
            rows: The request's blocks, per leaf of this group, mutated in
                place: a released slot is overwritten with the null block so
                the row stays as long as the request's block count.
            num_committed_blocks: How far the request's committed prefix
                reaches, which is what the window is measured back from.
            replica_idx: Which pool the pages return to.
        """
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
    ) -> int:
        """Returns the run of committed hashes from the root."""
        for num_hit_blocks, block_hash in enumerate(desired_hashes):
            if not self.is_in_prefix_cache(block_hash, replica_idx):
                return num_hit_blocks
        return len(desired_hashes)

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

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Keeps every page: this group reads its whole history."""
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
            if not self.is_in_prefix_cache(desired_hashes[idx], replica_idx):
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

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts the window ending at the hit and nulls every slot below it."""
        pool = self.pools[replica_idx]
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        if not all(
            self.is_in_prefix_cache(block_hash, replica_idx)
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

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the pages below the window, nulling their slots."""
        pool = self.pools[replica_idx]
        first_needed = max(0, num_committed_blocks - self._blocks_in_window)
        for leaf_id in self.leaf_ids:
            req_blocks = rows[leaf_id]
            null_block = pool.null_little_blocks[leaf_id]
            for idx in range(first_needed - 1, -1, -1):
                if req_blocks[idx].is_null:
                    break
                pool.free_block(req_blocks[idx])
                req_blocks[idx] = null_block

    def num_blocks_needed_for_connector_load(self, num_hashes: int) -> int:
        """The number of blocks needed to service a connector cache hit for all hashes."""
        return min(num_hashes, self._blocks_in_window)
