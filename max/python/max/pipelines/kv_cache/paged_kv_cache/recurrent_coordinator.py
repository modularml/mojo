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
"""The cache group whose entry is a recurrent state rather than a span."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from .block_utils import InsufficientBlocksError, LittleKVCacheBlock
from .kv_group_coordinator import KVGroupCoordinatorInterface

__all__ = ["RecurrentKVGroupCoordinator"]


@dataclass(frozen=True)
class RecurrentKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group whose entry is one state, held in the last slot of a row.

    ``row[-1]`` is the block the recurrence runs in, and it carries no hash.
    Every block behind it is null or a published checkpoint. The recurrence
    reads and writes one block, so reaching a boundary publishes the block it
    ran in and copies it into a successor rather than snapshotting it.
    """

    page_size: int = 0
    """Tokens per page: the granularity a state can be committed at."""

    def _live(
        self, row: Sequence[LittleKVCacheBlock]
    ) -> LittleKVCacheBlock | None:
        """Returns the block the recurrence runs in, if the row has one."""
        if not row or row[-1].block_hash is not None:
            return None
        return row[-1]

    def _resumed_from(
        self, row: Sequence[LittleKVCacheBlock]
    ) -> LittleKVCacheBlock | None:
        """Returns the published block a hit left for this request to read."""
        for block in reversed(row):
            if block.is_null:
                continue
            if block.block_hash is not None:
                return block
        return None

    # ============================================================================
    # The hit
    # ============================================================================

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool = False,
    ) -> int:
        """Returns the deepest boundary a published state stands at."""
        for idx in range(len(desired_hashes) - 1, -1, -1):
            if self._holds_every_leaf(desired_hashes[idx], replica_idx):
                return idx + 1
        return 0

    def claim_hit_blocks(
        self, desired_hashes: Sequence[bytes], replica_idx: int
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Takes the block at the granted block count, nulling the slots behind it.

        Returns empty rows when no published state stands there.
        """
        pool = self.pools[replica_idx]
        matched = bool(desired_hashes) and self._holds_every_leaf(
            desired_hashes[-1], replica_idx
        )
        if not matched:
            return {leaf_id: [] for leaf_id in self.leaf_ids}
        rows: dict[str, list[LittleKVCacheBlock]] = {}
        for leaf_id in self.leaf_ids:
            null_block = pool.null_little_blocks[leaf_id]
            row = [null_block] * (len(desired_hashes) - 1)
            block = pool.prefix_caches[leaf_id][desired_hashes[-1]]
            pool.touch(block)
            row.append(block)
            rows[leaf_id] = row
        return rows

    # ============================================================================
    # Demand
    # ============================================================================

    def _num_blocks_to_allocate(
        self, row: Sequence[LittleKVCacheBlock], num_required_blocks: int
    ) -> int:
        """Returns one block until the row holds a live one, then none."""
        return 0 if self._live(row) is not None else 1

    def grow(
        self, req_id: RequestID, num_required_blocks: int, replica_idx: int
    ) -> None:
        """Draws the live block if the request has none, and pads the row out."""
        pool = self.pools[replica_idx]
        drawn: dict[str, LittleKVCacheBlock] = {}
        try:
            for leaf_id in self.leaf_ids:
                row = self.rows[req_id][leaf_id]
                if self._num_blocks_to_allocate(row, num_required_blocks):
                    drawn[leaf_id] = pool.alloc_block(leaf_id)
        except InsufficientBlocksError:
            for block in drawn.values():
                pool.free_block(block)
            raise InsufficientBlocksError(
                f"No blocks left for the recurrent state of {req_id}"
            ) from None

        for leaf_id in self.leaf_ids:
            row = self.rows[req_id][leaf_id]
            null_block = pool.null_little_blocks[leaf_id]
            live = drawn.get(leaf_id) or row.pop()
            while len(row) < max(num_required_blocks - 1, 0):
                row.append(null_block)
            row.append(live)

    def shrink_to_fit(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Refits the row to the committed blocks, keeping the live block last.

        Frees every other block in the row.
        """
        pool = self.pools[replica_idx]
        for leaf_id in self.leaf_ids:
            row = self.rows[req_id][leaf_id]
            live = row.pop() if self._live(row) is not None else None
            for block in reversed(row):
                pool.free_block(block)
            row.clear()
            if live is None:
                continue
            null_block = pool.null_little_blocks[leaf_id]
            while len(row) < max(num_committed_blocks - 1, 0):
                row.append(null_block)
            row.append(live)

    # ============================================================================
    # The blocks a forward runs in
    # ============================================================================

    def live_blocks(self, req_id: RequestID) -> dict[str, int] | None:
        """Returns the block per leaf the recurrence runs in, if one is drawn."""
        row = self.rows.get(req_id)
        if row is None:
            return None
        live = {leaf_id: self._live(row[leaf_id]) for leaf_id in self.leaf_ids}
        if any(block is None for block in live.values()):
            return None
        return {leaf_id: block.bid for leaf_id, block in live.items()}  # type: ignore[union-attr]

    def advance(
        self, req_id: RequestID, num_committed_blocks: int, replica_idx: int
    ) -> None:
        """Frees every published block behind the live one, nulling its slot."""
        pool = self.pools[replica_idx]
        for leaf_id in self.leaf_ids:
            row = self.rows[req_id][leaf_id]
            null_block = pool.null_little_blocks[leaf_id]
            live = self._live(row)
            for idx, block in enumerate(row):
                if block is live or block.is_null:
                    continue
                if block.block_hash is None:
                    continue
                pool.free_block(block)
                row[idx] = null_block

    # ============================================================================
    # Checkpoints
    # ============================================================================

    @traced
    def resume(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int | None, int]]:
        """Returns the block this forward resumes its state from, per leaf.

        A published block when the row holds one: what a prefix hit claimed,
        or the predecessor a checkpoint published. ``None`` when the request has
        processed nothing and matched no hit, meaning the block is wiped
        instead, since a drawn block holds whatever its last request wrote.

        Empty once the request runs in a block it has written itself.
        """
        del replica_idx  # the blocks name themselves; the caller holds the pool
        row = self.rows.get(ctx.request_id)
        runs_in = self.live_blocks(ctx.request_id)
        if row is None or runs_in is None:
            return {}
        sources: dict[str, tuple[int | None, int]] = {}
        for leaf_id in self.leaf_ids:
            published = self._resumed_from(row[leaf_id])
            if published is None and ctx.tokens.processed_length > 0:
                continue
            src = None if published is None else published.bid
            sources[leaf_id] = (src, runs_in[leaf_id])
        return sources

    @traced
    def checkpoint(
        self, ctx: TextContext, replica_idx: int
    ) -> Mapping[str, tuple[int, int]]:
        """Publishes the block just run in and fills the one that succeeds it.

        The block the forward ran in already holds the state the boundary
        hash names, so it is published where that hash lands rather than
        snapshotted. The request continues in a freshly drawn block, copied
        from it. Runs before the commit, so the copy reads a block nothing
        has freed yet.

        Empty unless the request has processed at least one full block and
        the row holds no unpublished predecessor already.
        """
        row = self.rows.get(ctx.request_id)
        if row is None:
            return {}
        if any(
            row[leaf_id] and row[leaf_id][-1].is_null
            for leaf_id in self.leaf_ids
        ):
            return {}  # padding a batch out; there is no state to keep
        ran_in = self.live_blocks(ctx.request_id)
        if ran_in is None:
            return {}

        num_committed_blocks = ctx.tokens.processed_length // self.page_size
        if num_committed_blocks == 0:
            return {}
        if any(
            not block.is_null and block.block_hash is None
            for leaf_id in self.leaf_ids
            for block in row[leaf_id][:-1]
        ):
            return {}  # one checkpoint at a time

        # Admission reserves the successor alongside the block being
        # published, so a failure here is an accounting bug, not pressure.
        pool = self.pools[replica_idx]
        drawn = {
            leaf_id: pool.alloc_block(leaf_id) for leaf_id in self.leaf_ids
        }

        copies: dict[str, tuple[int, int]] = {}
        for leaf_id in self.leaf_ids:
            r = row[leaf_id]
            while len(r) <= num_committed_blocks:
                r.insert(len(r) - 1, pool.null_little_blocks[leaf_id])
            # The block that ran moves to the slot its hash indexes; the
            # successor takes over as the one the recurrence runs in.
            r[num_committed_blocks - 1] = r[-1]
            r[-1] = drawn[leaf_id]
            copies[leaf_id] = (ran_in[leaf_id], drawn[leaf_id].bid)
        return copies

    def _is_committable(
        self, row: Sequence[LittleKVCacheBlock], block_idx: int
    ) -> bool:
        """The live block is still being written, so it cannot be published."""
        if row[block_idx] is self._live(row):
            return False
        return super()._is_committable(row, block_idx)

    @traced
    def forward_blocks(
        self, batch: Sequence[TextContext], num_blocks: Sequence[int]
    ) -> dict[str, list[list[int]]]:
        """Returns the block each request's recurrence runs in."""
        plans: dict[str, list[list[int]]] = {
            leaf_id: [] for leaf_id in self.leaf_ids
        }
        for ctx in batch:
            runs_in = self.live_blocks(ctx.request_id)
            if runs_in is None:
                raise ValueError(
                    f"{ctx.request_id} has no state blocks; alloc must run"
                    " before its inputs are built"
                )
            for leaf_id in self.leaf_ids:
                plans[leaf_id].append([runs_in[leaf_id]])
        return plans
