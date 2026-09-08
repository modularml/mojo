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
"""Tests for the KVGroupCoordinator"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.pipelines.kv_cache.paged_kv_cache.block_utils import LittleKVCacheBlock
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    create_kv_group_coordinator,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    JengaBlockPool,
)
from max.pipelines.kv_cache.paged_kv_cache.kv_group_coordinator import (
    FullKVGroupCoordinator,
    KVGroupCoordinatorInterface,
    SlidingWindowKVGroupCoordinator,
)

BLOCK_SIZE = 4

VALUES = "attn/values"
SCALES = "attn/scales"
RATIOS = {VALUES: 8, SCALES: 32}

# Deliberately not a multiple of the block size: the window a model asks for
# rarely lands on a page boundary.
WINDOW = 17
WINDOW_BLOCKS = 4


def make_pool(num_huge_blocks: int = 64) -> JengaBlockPool:
    return JengaBlockPool(num_huge_blocks, RATIOS)


def block_keys(count: int) -> list[bytes]:
    """Returns one distinct hash per block index."""
    return [f"block-{idx}".encode() for idx in range(count)]


def commit(
    pool: JengaBlockPool, leaf_ids: Sequence[str], keys: Sequence[bytes]
) -> None:
    """Commits a block per key in every leaf, then drops the references.

    The blocks are left committed but unreferenced, which is what a finished
    request leaves behind and what a later hit revives.
    """
    for leaf_id in leaf_ids:
        blocks = [pool.alloc_block(leaf_id) for _ in keys]
        for block, key in zip(blocks, keys, strict=False):
            pool.commit_into_prefix_cache(key, block)
        for block in reversed(blocks):
            pool.free_block(block)


def full_group(
    pool: JengaBlockPool, leaf_ids: Sequence[str] = (VALUES,)
) -> KVGroupCoordinatorInterface:
    return create_kv_group_coordinator(
        [pool], list(leaf_ids), KVCacheGroupId.full(), BLOCK_SIZE
    )


def sliding_group(
    pool: JengaBlockPool,
    window: int = WINDOW,
    block_size: int = BLOCK_SIZE,
    leaf_ids: Sequence[str] = (VALUES,),
) -> KVGroupCoordinatorInterface:
    return create_kv_group_coordinator(
        [pool],
        list(leaf_ids),
        KVCacheGroupId("sliding_window", window),
        block_size,
    )


def bids(blocks: Sequence[LittleKVCacheBlock]) -> list[int]:
    return [block.bid for block in blocks]


def test_dispatch_matches_the_attention_pattern() -> None:
    pool = make_pool()
    assert isinstance(full_group(pool), FullKVGroupCoordinator)
    assert isinstance(sliding_group(pool), SlidingWindowKVGroupCoordinator)


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        # When window_size=1, we hit a degenerate case where we always get 100%
        # cache hit rate since the query token does not attend to any historical
        # tokens.
        (1, 0),
        (BLOCK_SIZE, 1),
        (BLOCK_SIZE + 1, 1),
        (BLOCK_SIZE + 2, 2),
        (WINDOW, WINDOW_BLOCKS),
    ],
)
def test_blocks_in_window_excludes_the_query_token(
    window: int, expected: int
) -> None:
    group = sliding_group(make_pool(), window=window)
    assert isinstance(group, SlidingWindowKVGroupCoordinator)
    assert group._blocks_in_window == expected


@pytest.mark.parametrize(
    ("num_hashes", "expected"),
    [(0, 0), (2, 2), (10, WINDOW_BLOCKS)],
)
def test_sliding_connector_load_staging_is_capped_to_its_window(
    num_hashes: int, expected: int
) -> None:
    group = sliding_group(make_pool())
    assert group.num_blocks_needed_for_connector_load(num_hashes) == expected


def test_a_hash_is_present_only_when_every_leaf_holds_it() -> None:
    """The leaves are written together, so a half-present block is unusable."""
    pool = make_pool()
    keys = block_keys(2)
    group = full_group(pool, [VALUES, SCALES])

    commit(pool, [VALUES], keys)
    assert not group.is_in_prefix_cache(keys[0], 0)

    commit(pool, [SCALES], keys)
    assert group.is_in_prefix_cache(keys[0], 0)


def test_full_hit_is_the_run_from_the_root() -> None:
    pool = make_pool()
    keys = block_keys(6)
    commit(pool, [VALUES], keys)

    assert full_group(pool).longest_cache_hit(keys, 0) == 6


def test_full_hit_stops_at_the_first_gap() -> None:
    """A gap below the resume point is a gap the forward would read."""
    pool = make_pool()
    keys = block_keys(6)
    commit(pool, [VALUES, SCALES], keys)
    pool.uncommit_block(pool.prefix_caches[SCALES][keys[2]])

    assert full_group(pool, [VALUES, SCALES]).longest_cache_hit(keys, 0) == 2


def test_full_claim_adopts_the_whole_prefix() -> None:
    pool = make_pool()
    keys = block_keys(3)
    commit(pool, [VALUES, SCALES], keys)

    rows = full_group(pool, [VALUES, SCALES]).claim_hit_blocks(keys, 0)

    for leaf_id in (VALUES, SCALES):
        assert bids(rows[leaf_id]) == [
            pool.prefix_caches[leaf_id][key].bid for key in keys
        ]
        assert all(block.ref_cnt == 1 for block in rows[leaf_id])


def test_window_cache_hit() -> None:
    pool = make_pool()
    keys = block_keys(8)
    commit(pool, [VALUES], keys[4:7])

    assert (
        sliding_group(pool, window=4, block_size=1).longest_cache_hit(keys, 0)
        == 7
    )


def test_a_run_reaching_the_root_is_a_complete_window() -> None:
    """A short run is usable at index 0: there is no history under it."""
    pool = make_pool()
    keys = block_keys(5)
    commit(pool, [VALUES], keys[:2])

    assert sliding_group(pool).longest_cache_hit(keys, 0) == 2


def test_a_run_that_stops_short_is_not_a_window() -> None:
    """A run floating in the middle leaves the blocks under it missing."""
    pool = make_pool()
    keys = block_keys(8)
    commit(pool, [VALUES], keys[5:7])

    assert sliding_group(pool).longest_cache_hit(keys, 0) == 0


def test_no_cache_hit_with_empty_prefix_cache() -> None:
    pool = make_pool()
    keys = block_keys(10)

    assert sliding_group(pool).longest_cache_hit(keys, 0) == 0


def test_sliding_window_with_window_size_1_gets_cache_hit_anywhere() -> None:
    pool = make_pool()
    keys = block_keys(10)

    # Despite nothing being committed, we still get maximal cache hit since the
    # query token does not attend to any historical tokens when window_size=1.
    assert (
        sliding_group(pool, window=1, block_size=1).longest_cache_hit(keys, 0)
        == 10
    )


def test_sliding_claim_nulls_everything_below_the_window() -> None:
    """The row keeps its length so an index still means the same tokens."""
    pool = make_pool()
    keys = block_keys(7)
    commit(pool, [VALUES], keys)

    row = sliding_group(pool).claim_hit_blocks(keys, 0)[VALUES]

    assert len(row) == 7
    low = 7 - WINDOW_BLOCKS
    assert all(block.is_null for block in row[:low])
    assert bids(row[low:]) == [
        pool.prefix_caches[VALUES][key].bid for key in keys[low:]
    ]


def test_sliding_claim_refuses_a_window_it_did_not_validate() -> None:
    """A window missing a block is recomputed whole, not adopted in part.

    Another group can shorten the hit after this one scanned, landing the
    window somewhere this group never looked.
    """
    pool = make_pool()
    keys = block_keys(8)
    commit(pool, [VALUES], keys[4:])

    # The hit was 8 when this group found its window; a shortened hit of 6 puts
    # the window at blocks 2 through 5, and blocks 2 and 3 were never committed.
    row = sliding_group(pool).claim_hit_blocks(keys[:6], 0)[VALUES]

    assert len(row) == 6
    assert all(block.is_null for block in row)


def test_claiming_revives_parked_pages() -> None:
    """A hit takes back bytes that were up for grabs."""
    pool = make_pool()
    keys = block_keys(3)
    commit(pool, [VALUES], keys)
    parked = len(pool.free_huge_blocks)

    row = full_group(pool).claim_hit_blocks(keys, 0)

    assert len(pool.free_huge_blocks) == parked - 1
    assert all(block.ref_cnt == 1 for block in row[VALUES])


def test_null_pad_is_a_no_op_for_an_unbounded_group() -> None:
    pool = make_pool()
    rows = {VALUES: [pool.alloc_block(VALUES) for _ in range(6)]}
    before = list(rows[VALUES])

    full_group(pool).null_pad_blocks(
        rows, num_committed_blocks=6, replica_idx=0
    )

    assert rows[VALUES] == before


def test_null_pad_returns_the_pages_below_the_window() -> None:
    pool = make_pool()
    rows = {VALUES: [pool.alloc_block(VALUES) for _ in range(10)]}
    free_before = pool.num_free_blocks(VALUES)

    sliding_group(pool).null_pad_blocks(
        rows, num_committed_blocks=10, replica_idx=0
    )

    slid_past = 10 - WINDOW_BLOCKS
    assert [block.is_null for block in rows[VALUES]] == [True] * slid_past + [
        False
    ] * WINDOW_BLOCKS
    assert pool.num_free_blocks(VALUES) == free_before + slid_past


def test_null_pad_keeps_a_window_that_has_not_slid() -> None:
    pool = make_pool()
    rows = {VALUES: [pool.alloc_block(VALUES) for _ in range(WINDOW_BLOCKS)]}

    sliding_group(pool).null_pad_blocks(
        rows, num_committed_blocks=WINDOW_BLOCKS, replica_idx=0
    )

    assert not any(block.is_null for block in rows[VALUES])


def test_null_pad_stops_at_the_first_null() -> None:
    """Running twice must not hand the same page back twice."""
    pool = make_pool()
    rows = {VALUES: [pool.alloc_block(VALUES) for _ in range(10)]}
    group = sliding_group(pool)

    group.null_pad_blocks(rows, num_committed_blocks=10, replica_idx=0)
    settled = list(rows[VALUES])
    free_after_first = pool.num_free_blocks(VALUES)

    group.null_pad_blocks(rows, num_committed_blocks=10, replica_idx=0)

    assert rows[VALUES] == settled
    assert pool.num_free_blocks(VALUES) == free_after_first


def test_only_the_group_leaves_are_touched() -> None:
    """A coordinator reads and writes its own group's caches, nothing else."""
    pool = make_pool()
    keys = block_keys(3)
    commit(pool, [VALUES, SCALES], keys)

    rows = full_group(pool, [VALUES]).claim_hit_blocks(keys, 0)

    assert set(rows) == {VALUES}
    assert all(pool.prefix_caches[SCALES][key].ref_cnt == 0 for key in keys)
