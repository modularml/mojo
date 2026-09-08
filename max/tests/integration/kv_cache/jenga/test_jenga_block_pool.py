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
"""Tests for the fungible huge-block pool shared by every flat KV cache."""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    JengaBlockPool,
)

GLOBAL = "global/values"
SLIDING = "sliding/values"
SCALES = "global/scales"


def check_invariants(pool: JengaBlockPool) -> None:
    """Asserts the pool's structural invariants.

    A huge block is either parked in ``free_huge_blocks`` with none of its
    little blocks in circulation, or claimed by one cache, in which case each
    of that cache's little blocks in it is referenced or free-listed, never
    both and never neither. The null blocks are in neither state: they back
    dummy requests and never take part in allocation.
    """
    assert pool.null_huge_block not in pool.free_huge_blocks
    for cache_id, null in pool.null_little_blocks.items():
        assert null not in pool.free_little_blocks[cache_id]

    for huge_block in pool.huge_blocks:
        parked = huge_block in pool.free_huge_blocks
        if parked:
            assert huge_block.ref_cnt == 0
        for cache_id, blocks in huge_block.little_blocks.items():
            queue = pool.free_little_blocks[cache_id]
            # A parked huge block's little blocks stay directly allocable on
            # its owning cache's free list (see the class docstring), so
            # "claimed" here means typed to this cache, parked or not.
            claimed = huge_block.little_block_type == cache_id
            for block in blocks:
                if not claimed:
                    assert block.ref_cnt == 0
                    assert block not in queue
                else:
                    assert (block.ref_cnt > 0) != (block in queue)

    for cache_id, prefix_cache in pool.prefix_caches.items():
        for block_hash, block in prefix_cache.items():
            assert block.block_hash == block_hash
            assert block.cache_id == cache_id
            # Retyping the bytes out from under a committed block must drop it.
            assert block.huge_block.little_block_type == cache_id


def test_constructor() -> None:
    ratios = {GLOBAL: 5, SLIDING: 2}
    pool = JengaBlockPool(num_huge_blocks=43, cache_ratios=ratios)

    # Huge block 0 is the null block, so 42 of the 43 are allocatable.
    assert len(pool.huge_blocks) == 42
    assert len(pool.little_blocks[GLOBAL]) == 5 * 42
    assert len(pool.little_blocks[SLIDING]) == 2 * 42

    for cache_id, ratio in ratios.items():
        blocks = pool.little_blocks[cache_id]
        # Block 0 of every cache is its null block, so real ids start at the
        # first block of the first huge block.
        assert [block.bid for block in blocks] == list(range(ratio, 43 * ratio))
        # A little block's bid is its page index into its cache's own view of
        # the pool buffer, so the blocks of huge block h are contiguous there.
        for huge_bid, huge_block in enumerate(pool.huge_blocks):
            expected = blocks[huge_bid * ratio : (huge_bid + 1) * ratio]
            assert huge_block.little_blocks[cache_id] == expected
            assert all(block.huge_block is huge_block for block in expected)
        assert len(pool.free_little_blocks[cache_id]) == 0

    check_invariants(pool)


def test_invalid_constructor() -> None:
    # A pool of one huge block is all null block and nothing to hand out.
    for num_huge_blocks in [0, 1]:
        with pytest.raises(
            ValueError, match="num_huge_blocks must be at least 2"
        ):
            JengaBlockPool(
                num_huge_blocks=num_huge_blocks, cache_ratios={GLOBAL: 1}
            )
    with pytest.raises(ValueError, match="cache_ratios must be non-empty"):
        JengaBlockPool(num_huge_blocks=42, cache_ratios={})
    with pytest.raises(ValueError, match="cache_ratios must be positive"):
        JengaBlockPool(
            num_huge_blocks=42, cache_ratios={GLOBAL: 1, SCALES: -42}
        )


def test_alloc() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=3, cache_ratios={GLOBAL: 42, SCALES: 43, SLIDING: 3}
    )

    blocks = [pool.alloc_block(SLIDING) for _ in range(6)]
    with pytest.raises(
        InsufficientBlocksError, match="No free blocks available"
    ):
        pool.alloc_block(SLIDING)

    assert [block.bid for block in blocks] == list(range(3, 9))
    assert all(block.ref_cnt == 1 for block in blocks)
    assert all(block.is_null is False for block in blocks)
    assert all(block.cache_id == SLIDING for block in blocks)

    assert [block.huge_block.bid for block in blocks] == [1, 1, 1, 2, 2, 2]

    check_invariants(pool)


def test_caches_split_the_pool_by_live_demand() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=6, cache_ratios={GLOBAL: 4, SLIDING: 1}
    )

    # No knob decides the split: whichever cache asks takes the next huge block.
    global_blocks = [pool.alloc_block(GLOBAL) for _ in range(8)]
    sliding_blocks = [pool.alloc_block(SLIDING) for _ in range(2)]

    assert len(pool.free_huge_blocks) == 1
    assert {block.huge_block.bid for block in global_blocks} == {1, 2}
    assert {block.huge_block.bid for block in sliding_blocks} == {3, 4}

    check_invariants(pool)

    for block in [*global_blocks, *sliding_blocks]:
        pool.free_block(block)

    # With the pool idle, either cache can have all of it.
    global_blocks = [pool.alloc_block(GLOBAL) for _ in range(20)]
    with pytest.raises(
        InsufficientBlocksError, match="No free blocks available"
    ):
        pool.alloc_block(GLOBAL)

    for block in global_blocks:
        pool.free_block(block)

    sliding_blocks = [pool.alloc_block(SLIDING) for _ in range(5)]
    with pytest.raises(
        InsufficientBlocksError, match="No free blocks available"
    ):
        pool.alloc_block(SLIDING)

    check_invariants(pool)


def test_fragmentation_can_cause_oom() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=5, cache_ratios={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    pool.alloc_block("a")
    pool.alloc_block("b")
    pool.alloc_block("c")
    pool.alloc_block("d")

    # "a" or "e" needs a whole new huge page, which is not available
    for cache_id in ["a", "e"]:
        with pytest.raises(
            InsufficientBlocksError, match="No free blocks available"
        ):
            pool.alloc_block(cache_id)

    # Can allocate little block for "b", "c", and "d"
    for block in ["b", "c", "d"]:
        pool.alloc_block(block)


def test_partly_freed_huge_block_stays_with_its_cache() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=3, cache_ratios={GLOBAL: 2, SLIDING: 2}
    )

    blocks = [pool.alloc_block(GLOBAL) for _ in range(2)]
    pool.free_block(blocks[0])

    # One live reference keeps the bytes typed, so the other cache cannot have
    # them however much it wants them.
    assert len(pool.free_huge_blocks) == 1
    assert len(pool.free_little_blocks[GLOBAL]) == 1
    assert pool.num_free_blocks(SLIDING) == 2
    check_invariants(pool)

    pool.free_block(blocks[1])
    assert len(pool.free_huge_blocks) == 2
    # The huge block went idle, but its little blocks stay on the global
    # cache's free list -- still directly allocable without a reclaim.
    assert len(pool.free_little_blocks[GLOBAL]) == 2
    assert pool.num_free_blocks(SLIDING) == 4


def test_freeing_a_pristine_block_is_reused_before_a_committed_one() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 2})

    blocks = [pool.alloc_block(GLOBAL) for _ in range(2)]
    pool.commit_into_prefix_cache(b"prefix", blocks[0])
    # Free the committed block first, then the pristine one -- despite that
    # order, the pristine one is reused first.
    pool.free_block(blocks[0])
    pool.free_block(blocks[1])

    next_block = pool.alloc_block(GLOBAL)

    assert next_block is blocks[1]
    assert blocks[0].block_hash == b"prefix"
    check_invariants(pool)


def test_idle_huge_block_can_be_retyped() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=2, cache_ratios={GLOBAL: 2, SLIDING: 1}
    )

    block = pool.alloc_block(GLOBAL)
    pool.free_block(block)
    retyped = pool.alloc_block(SLIDING)

    assert pool.huge_blocks[0].little_block_type == SLIDING
    assert retyped.cache_id == SLIDING
    assert block.ref_cnt == 0
    # The stolen-from cache must not still think its parked-and-typed huge
    # block counts as headroom now that it's gone.
    assert pool.num_free_blocks(GLOBAL) == 0
    check_invariants(pool)


def test_prefer_claim_huge_avoids_evicting_a_commit() -> None:
    pool = JengaBlockPool(num_huge_blocks=3, cache_ratios={GLOBAL: 2})

    blocks = [pool.alloc_block(GLOBAL) for _ in range(2)]
    pool.commit_into_prefix_cache(b"apple", blocks[0])
    pool.free_block(blocks[0])
    pool.commit_into_prefix_cache(b"banana", blocks[1])
    pool.free_block(blocks[1])

    # Huge block 2 is still virgin, so the next alloc claims it instead of
    # evicting either commit.
    new = pool.alloc_block(GLOBAL)

    assert new.huge_block.bid == 2
    assert pool.prefix_caches[GLOBAL] == {
        b"apple": blocks[0],
        b"banana": blocks[1],
    }
    check_invariants(pool)


def test_retyping_evicts_what_the_outgoing_cache_committed() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=2, cache_ratios={GLOBAL: 2, SLIDING: 1}
    )

    block = pool.alloc_block(GLOBAL)
    pool.commit_into_prefix_cache(b"prefix", block)
    pool.free_block(block)
    # Parked, but still serving its hash: the global cache gets it back for
    # free if it asks before anyone else claims the bytes.
    assert pool.prefix_caches[GLOBAL] == {b"prefix": block}

    pool.alloc_block(SLIDING)

    assert pool.prefix_caches[GLOBAL] == {}
    assert block.block_hash is None
    check_invariants(pool)


def test_prefix_hit_reclaims_a_parked_huge_block() -> None:
    pool = JengaBlockPool(num_huge_blocks=3, cache_ratios={GLOBAL: 2})

    block = pool.alloc_block(GLOBAL)
    pool.commit_into_prefix_cache(b"prefix", block)
    pool.free_block(block)
    assert len(pool.free_huge_blocks) == 2

    pool.touch(pool.prefix_caches[GLOBAL][b"prefix"])

    assert block.ref_cnt == 1
    assert block.block_hash == b"prefix"
    # Reclaiming the huge block puts its sibling back in circulation too.
    assert len(pool.free_huge_blocks) == 1
    sibling = pool.little_blocks[GLOBAL][1]
    assert sibling in pool.free_little_blocks[GLOBAL]
    check_invariants(pool)


def test_allocating_over_a_committed_block_evicts_it() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 2})

    blocks = [pool.alloc_block(GLOBAL) for _ in range(2)]
    pool.commit_into_prefix_cache(b"apple", blocks[0])
    pool.free_block(blocks[0])
    pool.commit_into_prefix_cache(b"banana", blocks[1])
    pool.free_block(blocks[1])
    assert pool.prefix_caches[GLOBAL] == {
        b"apple": blocks[0],
        b"banana": blocks[1],
    }

    new = pool.alloc_block(GLOBAL)

    # Allocating a new global block evicts a single oldest entry from global
    # prefix cache.
    assert new is blocks[0]
    assert new.block_hash is None
    assert pool.prefix_caches[GLOBAL] == {b"banana": blocks[1]}
    check_invariants(pool)


def test_allocating_from_different_cache_over_a_committed_block_evicts_it() -> (
    None
):
    pool = JengaBlockPool(
        num_huge_blocks=2, cache_ratios={GLOBAL: 2, SLIDING: 1}
    )

    blocks = [pool.alloc_block(GLOBAL) for _ in range(2)]
    pool.commit_into_prefix_cache(b"apple", blocks[0])
    pool.free_block(blocks[0])
    pool.commit_into_prefix_cache(b"banana", blocks[1])
    pool.free_block(blocks[1])
    assert pool.prefix_caches[GLOBAL] == {
        b"apple": blocks[0],
        b"banana": blocks[1],
    }

    _ = pool.alloc_block(SLIDING)

    # Allocating a single sliding block evicts the whole global prefix cache.
    assert blocks[0].block_hash is None
    assert blocks[1].block_hash is None
    assert pool.prefix_caches[GLOBAL] == {}
    check_invariants(pool)


def test_get_or_commit_dedupes_onto_the_committed_block() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 2})

    first = pool.alloc_block(GLOBAL)
    assert pool.get_or_commit_into_prefix_cache(b"prefix", first) is None

    duplicate = pool.alloc_block(GLOBAL)
    assert pool.get_or_commit_into_prefix_cache(b"prefix", duplicate) is first

    assert first.ref_cnt == 2
    # The duplicate's bytes are redundant, so they go back to the pool.
    assert duplicate.ref_cnt == 0
    assert duplicate in pool.free_little_blocks[GLOBAL]
    check_invariants(pool)


def test_get_or_commit_is_a_noop_for_an_already_committed_block() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 1})

    block = pool.alloc_block(GLOBAL)
    pool.commit_into_prefix_cache(b"prefix", block)

    assert pool.get_or_commit_into_prefix_cache(b"prefix", block) is None
    assert block.ref_cnt == 1


def test_caches_do_not_share_a_prefix_cache() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=3, cache_ratios={GLOBAL: 1, SLIDING: 1}
    )

    # The same tokens hash the same way for every cache, but each cache stores
    # different bytes for them, so a hash names one block per cache.
    global_block = pool.alloc_block(GLOBAL)
    sliding_block = pool.alloc_block(SLIDING)
    pool.commit_into_prefix_cache(b"prefix", global_block)
    pool.commit_into_prefix_cache(b"prefix", sliding_block)

    assert pool.prefix_caches[GLOBAL][b"prefix"] is global_block
    assert pool.prefix_caches[SLIDING][b"prefix"] is sliding_block
    check_invariants(pool)


def test_every_cache_gets_a_null_block_at_id_zero() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=3, cache_ratios={GLOBAL: 2, SLIDING: 1}
    )

    # Dummy and padding requests point at block 0, which is why real blocks
    # start at id `ratio` and real huge blocks at id 1.
    assert pool.null_huge_block.bid == 0
    for cache_id, null in pool.null_little_blocks.items():
        assert null.bid == 0
        assert null.is_null
        assert null.cache_id == cache_id
        assert null.huge_block is pool.null_huge_block
        assert null not in pool.little_blocks[cache_id]
        assert 0 not in [block.bid for block in pool.little_blocks[cache_id]]
    assert 0 not in [huge_block.bid for huge_block in pool.huge_blocks]

    # The null block costs no capacity, since it is outside the pool.
    assert pool.num_free_blocks(GLOBAL) == 4
    check_invariants(pool)


def test_the_null_block_is_never_allocated_or_freed() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 1})
    null = pool.null_little_blocks[GLOBAL]
    pinned = null.ref_cnt

    # Exhausting the pool hands out its one real block, not the null one.
    assert pool.alloc_block(GLOBAL) is not null
    with pytest.raises(
        InsufficientBlocksError, match="No free blocks available"
    ):
        pool.alloc_block(GLOBAL)

    # Requests share the null block, so freeing it must not put it in
    # circulation -- however many dummy requests release it.
    for _ in range(3):
        pool.free_block(null)

    assert null.ref_cnt == pinned
    assert null not in pool.free_little_blocks[GLOBAL]
    check_invariants(pool)


def test_the_null_block_cannot_be_committed() -> None:
    pool = JengaBlockPool(num_huge_blocks=2, cache_ratios={GLOBAL: 1})

    # Its bytes are shared by every dummy request, so no hash describes them.
    with pytest.raises(AssertionError):
        pool.commit_into_prefix_cache(
            b"prefix", pool.null_little_blocks[GLOBAL]
        )

    assert pool.prefix_caches[GLOBAL] == {}


def test_can_satisfy_demand_charges_every_cache_to_one_budget() -> None:
    # 4 allocatable huge blocks: 8 GLOBAL pages or 16 SCALES pages on their
    # own, but only 3 huge blocks' worth once both are asked for at once.
    pool = JengaBlockPool(
        num_huge_blocks=5, cache_ratios={GLOBAL: 2, SCALES: 4}
    )
    assert pool.num_free_blocks(GLOBAL) == 8
    assert pool.num_free_blocks(SCALES) == 16

    assert pool.can_satisfy_demand({GLOBAL: 8})
    assert pool.can_satisfy_demand({SCALES: 16})
    # 8 GLOBAL pages take all 4 huge blocks, leaving none for the scales.
    assert not pool.can_satisfy_demand({GLOBAL: 8, SCALES: 1})
    assert pool.can_satisfy_demand({GLOBAL: 6, SCALES: 4})


def test_can_satisfy_demand_credits_pages_already_carved() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=3, cache_ratios={GLOBAL: 2, SCALES: 4}
    )
    # Carve one huge block for each cache and hand a page back, so both have
    # free pages while no huge block is left to claim.
    for cache_id in (GLOBAL, SCALES):
        pool.free_block(pool.alloc_block(cache_id))

    # Each cache alone would retype the other's parked huge block, so on its
    # own it counts far more than the pages both can hold at once.
    assert pool.num_free_blocks(GLOBAL) == 4
    assert pool.num_free_blocks(SCALES) == 8
    assert pool.can_satisfy_demand({GLOBAL: 2, SCALES: 4})
    assert not pool.can_satisfy_demand({GLOBAL: 3, SCALES: 4})
    assert not pool.can_satisfy_demand({GLOBAL: 4, SCALES: 8})
    check_invariants(pool)


def test_can_satisfy_demand_retypes_surplus_parked_pages() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=5, cache_ratios={GLOBAL: 2, SCALES: 4}
    )
    scales_blocks = [pool.alloc_block(SCALES) for _ in range(8)]
    for block in scales_blocks:
        pool.free_block(block)

    # One parked SCALES huge block satisfies its demand. The other can be
    # retyped with the two pristine blocks to satisfy GLOBAL's demand.
    assert pool.can_satisfy_demand({GLOBAL: 6, SCALES: 4})
    check_invariants(pool)


def test_can_satisfy_demand_at_capacity_ignores_what_is_out() -> None:
    pool = JengaBlockPool(
        num_huge_blocks=5, cache_ratios={GLOBAL: 2, SCALES: 4}
    )
    demand = {GLOBAL: 6, SCALES: 4}
    assert pool.can_satisfy_demand(demand)

    # Drain the pool: the live answer flips, the geometric one does not.
    for _ in range(8):
        pool.alloc_block(GLOBAL)

    assert not pool.can_satisfy_demand(demand)
    assert pool.can_satisfy_demand(demand, at_capacity=True)
    # Still bounded by the geometry -- 4 huge blocks is all there ever is.
    assert not pool.can_satisfy_demand({GLOBAL: 8, SCALES: 1}, at_capacity=True)
