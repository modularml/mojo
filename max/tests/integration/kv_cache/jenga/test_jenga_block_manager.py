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
"""Tests for the Jenga block manager."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.kv_connector import BlockCount
from max.pipelines.kv_cache.paged_kv_cache.block_manager import PrefixCacheHits
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
)
from max.pipelines.request.base import RequestID

FULL = "full"
VALUES = "values"
SCALES = "scales"
SLIDING = "sliding"


def full(ratio: int = 1) -> KVLeafInfo:
    return KVLeafInfo(ratio, KVCacheGroupId.full())


def sliding(ratio: int = 1, window: int = 10) -> KVLeafInfo:
    """A windowed cache holding ``ceildiv(window - 1, block_size)`` blocks.

    The query token takes one of the window's slots, so only ``window - 1``
    earlier tokens have to stay resident.
    """
    return KVLeafInfo(ratio, KVCacheGroupId("sliding_window", window))


def make_manager(
    leaf_infos: Mapping[str, KVLeafInfo],
    num_huge_blocks: int = 999,
    *,
    block_size: int = 1,
    enable_prefix_caching: bool = False,
    num_replicas: int = 1,
    max_num_input_tokens: int | None = None,
    num_draft_tokens: int = 0,
    num_draft_tokens_per_step: int = 0,
) -> JengaBlockManager:
    return JengaBlockManager(
        dict(leaf_infos),
        num_huge_blocks=num_huge_blocks,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        num_replicas=num_replicas,
        max_num_input_tokens=max_num_input_tokens,
        num_draft_tokens=num_draft_tokens,
        num_draft_tokens_per_step=num_draft_tokens_per_step,
    )


def make_ctx(num_tokens: int) -> TextContext:
    """Returns a request whose prompt is ``num_tokens`` distinct tokens."""
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.arange(num_tokens, dtype=np.int64)),
    )


def make_ctx_with_tokens(tokens: Sequence[int]) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
    )


def decode(bm: JengaBlockManager, ctx: TextContext, token: int = 42) -> None:
    """Runs one decode step: allocate, forward, record what it wrote."""
    bm.alloc(ctx)
    ctx.update(token)
    bm.step(ctx)


def uncommit(
    bm: JengaBlockManager, leaf_id: str, bids: set[int] | None = None
) -> None:
    """Drops pages from a leaf's prefix cache, all of them by default.

    Stands in for the eviction a busy pool would do on its own, so a test can
    pick exactly which blocks a later request finds missing.
    """
    pool = bm.pools[0]
    for block in list(pool.prefix_caches[leaf_id].values()):
        if bids is None or block.bid in bids:
            pool.uncommit_block(block)


# ===--------------------------------------------------------------------=== #
# Basic Functionality
# ===--------------------------------------------------------------------=== #


def test_claim_and_alloc() -> None:
    bm = make_manager({FULL: full(ratio=1)}, block_size=1)
    ctx = make_ctx(num_tokens=3)

    bm.claim(ctx)
    bm.alloc(ctx)

    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2, 3]

    for _ in range(4):
        ctx.update(42)
    bm.alloc(ctx)

    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2, 3, 4, 5, 6, 7]


def test_release() -> None:
    # 3 of the 4 huge blocks are allocable; the 4th backs the null block.
    bm = make_manager({FULL: full(ratio=1)}, block_size=1, num_huge_blocks=4)
    ctxA = make_ctx(num_tokens=3)
    ctxB = make_ctx(num_tokens=3)

    # Allocate blocks for A
    bm.claim(ctxA)
    bm.alloc(ctxA)
    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [1, 2, 3]

    # No blocks left for B
    assert bm.huge_block_count().free == 0
    bm.claim(ctxB)
    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctxB)

    # Once we release A, there is space recovered
    bm.release(ctxA)
    assert bm.huge_block_count().free == 3

    # Allocate blocks for B. A's blocks were never committed, so each release
    # puts one at the *front* of the free list -- releasing tail first (3,
    # then 2, then 1) hands them back out in 1, 2, 3 order.
    bm.alloc(ctxB)
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [1, 2, 3]


def test_release_recycles_the_tail_first() -> None:
    """A released request's tail blocks are the first to be reused.

    Whoever shares this request's prefix wants its head, so the head has to be
    the last thing overwritten. That ordering is what makes the pool's reuse
    order least-recently-useful rather than arbitrary.
    """
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=5,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=4)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [1, 2, 3, 4]
    bm.release(ctxA)

    # An unrelated request takes A's last two blocks, evicting what they held.
    ctxB = make_ctx_with_tokens([99, 98])
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [4, 3]

    # A's first block was the last to be overwritten, so it is still there for
    # a request that shares its prefix.
    ctxC = make_ctx(num_tokens=2)
    bm.claim(ctxC)
    bm.alloc(ctxC)
    assert bm.get_req_blocks_per_leaf(ctxC)[FULL] == [1, 2]
    assert ctxC.tokens.processed_length == 1


def test_ratio() -> None:
    bm = make_manager({FULL: full(ratio=3)}, block_size=1, num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)

    bm.claim(ctx)
    # 1 of the 10 huge blocks is reserved for the null block.
    assert bm.huge_block_count().free == 9
    bm.alloc(ctx)
    # 1 huge block is used for the new request.
    assert bm.huge_block_count().free == 8

    # With ratio 3, the first non-null block is the 3rd one.
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [3, 4, 5]


def test_block_size() -> None:
    bm = make_manager({FULL: full(ratio=1)}, block_size=4)
    ctx = make_ctx(num_tokens=3)

    # Need 1 block to hold 3 tokens.
    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1]

    # 4 tokens still fits in 1 block.
    ctx.update(42)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1]

    # 5 tokens needs 2 blocks.
    ctx.update(42)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2]


def test_huge_block_shared_by_different_req() -> None:
    bm = make_manager({FULL: full(ratio=3)}, block_size=1, num_huge_blocks=4)
    ctxA = make_ctx(num_tokens=5)
    ctxB = make_ctx(num_tokens=4)

    bm.claim(ctxA)
    bm.claim(ctxB)

    # Notice that huge_page=2 corresponds to little_pages=6,7,8 and they are
    # split between the two requests.
    assert bm.huge_block_count().free == 3
    bm.alloc(ctxA)
    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [3, 4, 5, 6, 7]
    assert bm.huge_block_count().free == 1
    bm.alloc(ctxB)
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [8, 9, 10, 11]
    assert bm.huge_block_count().free == 0


def test_little_block_cannot_be_shared_by_different_req() -> None:
    # 3 blocks of 10 tokens each is enough for 30 tokens
    bm = make_manager({FULL: full(ratio=1)}, block_size=10, num_huge_blocks=4)

    # 10 + 20 = 30 tokens is ok
    # ctxA gets 1 block, ctxB gets 2 blocks
    ctxA = make_ctx(num_tokens=10)
    ctxB = make_ctx(num_tokens=20)
    bm.claim(ctxA)
    bm.claim(ctxB)
    bm.alloc(ctxA)
    bm.alloc(ctxB)
    bm.release(ctxA)
    bm.release(ctxB)

    # 15 + 15 = 30 tokens is not ok
    # ctxA gets 2 blocks, ctxB needs another 2 blocks
    ctxA = make_ctx(num_tokens=15)
    ctxB = make_ctx(num_tokens=15)
    bm.claim(ctxA)
    bm.claim(ctxB)
    bm.alloc(ctxA)
    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctxB)
    assert bm.huge_block_count().free == 1
    # A's blocks are 2 and 3, not 1 and 2: the round above released its blocks
    # tail first (3, then 2, then 1), and each release puts an uncommitted
    # block at the *front* of the free list, so the pool hands them back out
    # in 2, 3, 1 order.
    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [2, 3]
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == []


def test_oom() -> None:
    # We can fit 2*5*(4-1) = 30 tokens
    bm = make_manager({FULL: full(ratio=2)}, block_size=5, num_huge_blocks=4)

    # 30 tokens is ok
    ctx = make_ctx(num_tokens=30)
    bm.claim(ctx)
    bm.alloc(ctx)
    bm.release(ctx)

    # 31 tokens is unsatisfiable
    ctx = make_ctx(num_tokens=31)
    bm.claim(ctx)
    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctx)


def test_dont_double_claim() -> None:
    bm = make_manager({FULL: full()})
    ctx = make_ctx(num_tokens=3)

    bm.claim(ctx)
    with pytest.raises(ValueError, match="Request is already claimed"):
        bm.claim(ctx)


def test_dont_use_req_before_claim() -> None:
    bm = make_manager({FULL: full()})
    ctx = make_ctx(num_tokens=3)
    with pytest.raises(ValueError, match="Request is not claimed"):
        bm.get_req_blocks_per_leaf(ctx)
    with pytest.raises(ValueError, match="Request is not claimed"):
        bm.alloc(ctx)
    with pytest.raises(ValueError, match="Request is not claimed"):
        bm.step(ctx)
    with pytest.raises(ValueError, match="Request is not claimed"):
        bm.release(ctx)


def test_chunked_prefill() -> None:
    bm = make_manager({FULL: full(ratio=1)}, block_size=1, num_huge_blocks=10)
    ctx = make_ctx(num_tokens=6)
    bm.claim(ctx)

    # Each forward covers 2 tokens, so the request only ever holds the blocks
    # the forwards so far have needed, not the whole prompt's worth.
    ctx.tokens.chunk(2)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2]
    ctx.update(42)
    bm.step(ctx)

    ctx.tokens.chunk(2)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2, 3, 4]
    ctx.update(42)
    bm.step(ctx)

    # The last chunk needs no trimming: 2 tokens are left.
    assert ctx.tokens.active_length == 2
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[FULL] == [1, 2, 3, 4, 5, 6]


# ===--------------------------------------------------------------------=== #
# Different Ratios
# ===--------------------------------------------------------------------=== #


def test_values_and_scales() -> None:
    # Each huge page has either 1 value or 10 scales.
    bm = make_manager(
        {VALUES: full(ratio=1), SCALES: full(ratio=10)},
        block_size=1,
        num_huge_blocks=4,
    )

    # We can fit two requests with 1 token
    # huge_page=1 is values, huge_page=2 is scales, huge_page=3 is values
    assert bm.huge_block_count().free == 3
    # little_block_count's total is fixed by the 3 allocable huge blocks'
    # ratio-converted capacity; free shrinks as requests claim blocks.
    assert bm.little_block_count() == {
        VALUES: BlockCount(free=3, total=3),
        SCALES: BlockCount(free=30, total=30),
    }

    ctx = make_ctx(num_tokens=1)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[VALUES] == [1]
    assert bm.get_req_blocks_per_leaf(ctx)[SCALES] == [20]
    assert bm.huge_block_count().free == 1
    assert bm.little_block_count() == {
        VALUES: BlockCount(free=1, total=3),
        SCALES: BlockCount(free=19, total=30),
    }

    # The second request takes the last huge block for its values, which leaves
    # scales with the 8 little blocks it already holds and nothing to grow into.
    ctx = make_ctx(num_tokens=1)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[VALUES] == [3]
    assert bm.get_req_blocks_per_leaf(ctx)[SCALES] == [21]
    assert bm.huge_block_count().free == 0
    assert bm.little_block_count() == {
        VALUES: BlockCount(free=0, total=3),
        SCALES: BlockCount(free=8, total=30),
    }

    # OOM. There is no more room for values, but there is still room for scales.
    ctx = make_ctx(num_tokens=1)
    bm.claim(ctx)
    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctx)


def test_fp8_spec_dec() -> None:
    bm = make_manager(
        {
            "target/values": full(ratio=1),
            "target/scales": full(ratio=10),
            "draft/values": full(ratio=2),
            "draft/scales": full(ratio=20),
        },
        block_size=128,
        num_huge_blocks=5,
    )
    ctx = make_ctx(num_tokens=128)
    bm.claim(ctx)
    bm.alloc(ctx)

    # A huge block serves one cache at a time, so four caches take four of
    # them even though each wants a single page. Each cache numbers its pages
    # in its own space, at its own ratio: huge block h holds ids
    # [h * ratio, (h + 1) * ratio).
    assert bm.get_req_blocks_per_leaf(ctx) == {
        "target/values": [1],  # huge block 1, 1 page each
        "target/scales": [20],  # huge block 2, 10 pages each
        "draft/values": [6],  # huge block 3, 2 pages each
        "draft/scales": [80],  # huge block 4, 20 pages each
    }
    assert bm.huge_block_count().free == 0


# ===--------------------------------------------------------------------=== #
# SWA Alone
# ===--------------------------------------------------------------------=== #


def test_swa_frees_the_blocks_it_slid_past() -> None:
    # A window of 4 covers the query token plus 3 earlier ones, so 3 blocks of
    # history stay resident.
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=10,
    )
    ctx = make_ctx(num_tokens=6)

    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[SLIDING] == [1, 2, 3, 4, 5, 6]

    # Nothing is released until a forward has filled the blocks, which is what
    # ctx.update records.
    ctx.update(42)
    bm.step(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[SLIDING] == [0, 0, 0, 4, 5, 6]
    assert bm.huge_block_count().free == 6


def test_swa_frees_the_blocks_it_slid_past_with_block_size_larger_than_1() -> (
    None
):
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=3)},
        block_size=2,
        num_huge_blocks=10,
    )
    ctx = make_ctx(num_tokens=6)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[SLIDING] == [1, 2, 3]

    # The next query is at token 6, and a window of 3 has it read back to token
    # 4, which lives in block 2. Blocks 0 and 1 are below that, so they go. A
    # released slot is nulled where it stands rather than shifting the row.
    ctx.update(42)
    bm.step(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[SLIDING] == [0, 0, 3]
    assert bm.huge_block_count().free == 8


def test_swa_reuses_the_blocks_it_freed() -> None:
    """The window walks forward through a pool far smaller than the sequence.

    Five allocable blocks back a request that runs to fifty-odd tokens, because
    a page the window slid past is back in the pool before the next one is
    needed.
    """
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=6,
    )
    ctx = make_ctx(num_tokens=2)

    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)
    bm.step(ctx)
    assert bm.get_req_blocks_per_leaf(ctx)[SLIDING] == [1, 2]

    for _ in range(50):
        decode(bm, ctx)
        row = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
        assert len([bid for bid in row if bid != 0]) == 3

    # The row still spans every block index, and every page in it came from the
    # five the pool can back. Each is uncommitted when freed, so it goes back
    # onto the *front* of the free list and is the next one reused.
    row = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
    assert len(row) == 52
    assert row[-3:] == [2, 3, 4]


def test_swa_reuses_a_window_it_still_holds() -> None:
    """A hit does not have to reach the root when nothing reads that far back."""
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=6)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    assert bm.get_req_blocks_per_leaf(ctxA)[SLIDING] == [0, 0, 0, 4, 5, 6]
    bm.release(ctxA)

    # The blocks below the window were freed, but freeing keeps their commits,
    # so the window ending at block 5 is still there to adopt.
    ctxB = make_ctx(num_tokens=6)
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert ctxB.tokens.processed_length == 5
    assert bm.get_req_blocks_per_leaf(ctxB)[SLIDING] == [0, 0, 3, 4, 5, 7]


def test_swa_resumes_below_a_hole_in_its_history() -> None:
    """A hole spoils the window above it but not the prefix below it.

    Blocks 3 and 4 survive, but the window ending there also needs block 2, so
    that resume point is out. The run from block 0 up to the hole is usable
    though: the request holds those blocks itself, so they complete the window
    of whatever it recomputes next.
    """
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=6)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    uncommit(bm, SLIDING, {3})

    ctxB = make_ctx(num_tokens=6)
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert ctxB.tokens.processed_length == 2
    # Block 3 is uncommitted but was never evicted from circulation when it
    # was freed, so it is reused directly instead of costing a fresh claim.
    assert bm.get_req_blocks_per_leaf(ctxB)[SLIDING] == [1, 2, 3, 7, 8, 9]


def test_swa_refuses_a_window_it_cannot_complete() -> None:
    """Two thirds of a window is worth nothing: attention reads all of it.

    A model with an full cache could still resume here, one window early,
    and let the forward refill the window as it recomputes -- the full
    cache holds the tokens being skipped. With every cache windowed nothing
    holds them, so the only resume points are the ones where a whole window
    survives, or the prefix run the request itself continues.
    """
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=6)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    # Leaves blocks 4 and 5 committed: a two-block run in the middle, reaching
    # neither a full window nor the root.
    uncommit(bm, SLIDING, {1, 2, 3})

    ctxB = make_ctx(num_tokens=6)
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert ctxB.tokens.processed_length == 0
    # Blocks 1-3 are uncommitted but were never evicted from circulation when
    # they were freed, so they are reused directly before any fresh claim.
    assert bm.get_req_blocks_per_leaf(ctxB)[SLIDING] == [3, 2, 1, 7, 8, 9]


# ===--------------------------------------------------------------------=== #
# Prefix Cache
# ===--------------------------------------------------------------------=== #


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_prefix_caching(enable_prefix_caching: bool) -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        enable_prefix_caching=enable_prefix_caching,
    )
    ctxA = make_ctx_with_tokens([42, 42, 42, 98, 98, 98])
    ctxB = make_ctx_with_tokens([42, 42, 42, 99, 99, 99])

    bm.claim(ctxA)
    bm.alloc(ctxA)
    # A block is only committed once a forward has filled it, which is what
    # ctx.update records.
    ctxA.update(42)
    bm.step(ctxA)

    bm.claim(ctxB)
    bm.alloc(ctxB)

    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [1, 2, 3, 4, 5, 6]
    if enable_prefix_caching:
        # The two prompts share their first three tokens, so B reuses those
        # blocks and computes the rest.
        assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [1, 2, 3, 7, 8, 9]
        assert ctxB.tokens.processed_length == 3
    else:
        assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [7, 8, 9, 10, 11, 12]
        assert ctxB.tokens.processed_length == 0


def test_lru_evicts_the_tail_of_a_released_prefix() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=5,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=4)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [1, 2, 3, 4]
    bm.release(ctxA)

    # The pool is full, so serving anything at all costs a committed block. The
    # one it takes is the tail of A's prefix, the least useful to keep.
    ctxB = make_ctx_with_tokens([99])
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [4]
    bm.release(ctxB)

    # A request repeating A's prompt therefore hits 3 of its 4 blocks.
    ctxC = make_ctx(num_tokens=4)
    bm.claim(ctxC)
    bm.alloc(ctxC)
    assert ctxC.tokens.processed_length == 3
    assert bm.get_req_blocks_per_leaf(ctxC)[FULL] == [1, 2, 3, 4]


def test_reset_prefix_cache_drops_commits_nobody_holds() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=4)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    bm.reset_prefix_cache()

    ctxB = make_ctx(num_tokens=4)
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert ctxB.tokens.processed_length == 0
    # Dropping the commit does not evict the block from circulation, so A's
    # blocks are reused directly rather than costing a fresh claim.
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [4, 3, 2, 1]


def test_reset_prefix_cache_keeps_commits_a_request_holds() -> None:
    """A live request's blocks cannot be handed out, so their commits stay."""
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=4)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)

    bm.reset_prefix_cache()

    ctxB = make_ctx(num_tokens=4)
    bm.claim(ctxB)
    bm.alloc(ctxB)
    assert ctxB.tokens.processed_length == 3
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [1, 2, 3, 5]


# ===--------------------------------------------------------------------=== #
# Full + SWA
# ===--------------------------------------------------------------------=== #


def test_full_and_swa_share_one_block_index() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
    )
    ctx = make_ctx(num_tokens=6)

    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6],
        SLIDING: [7, 8, 9, 10, 11, 12],
    }

    ctx.update(42)
    bm.step(ctx)
    # Both rows still span every block index, so index i means the same tokens
    # in both. The sliding cache just stops backing the ones it cannot read.
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6],
        SLIDING: [0, 0, 0, 10, 11, 12],
    }


def test_full_and_swa_resume_from_the_same_block() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=6)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    ctxB = make_ctx(num_tokens=6)
    bm.claim(ctxB)
    bm.alloc(ctxB)

    # Both caches resume at block 5: the full one adopts the whole prefix,
    # the sliding one only the part its attention reads.
    assert ctxB.tokens.processed_length == 5
    assert bm.get_req_blocks_per_leaf(ctxB) == {
        FULL: [1, 2, 3, 4, 5, 13],
        SLIDING: [0, 0, 9, 10, 11, 14],
    }


def test_a_dropped_sliding_window_blocks_all_reuse() -> None:
    """A block index means the same tokens everywhere, so the groups agree.

    The full cache holds the whole prefix and the sliding one holds
    nothing, so there is no point both can resume from and the request starts
    over. The full cache's depth cannot carry the sliding one: its blocks
    feed its own caches, and resuming on its strength would leave the sliding
    caches attending null pages where their window should be.
    """
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctxA = make_ctx(num_tokens=6)
    bm.claim(ctxA)
    bm.alloc(ctxA)
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    pool = bm.pools[0]
    for block in list(pool.prefix_caches[SLIDING].values()):
        pool.uncommit_block(block)

    ctxB = make_ctx(num_tokens=6)
    bm.claim(ctxB)
    bm.alloc(ctxB)

    assert ctxB.tokens.processed_length == 0
    # Nothing adopted anywhere, so full draws a fresh row -- its own released
    # blocks are still committed, and claiming a virgin huge block beats
    # evicting them. Sliding's released blocks were uncommitted above, so it
    # reuses them directly instead of claiming anything new.
    assert bm.get_req_blocks_per_leaf(ctxB) == {
        FULL: [13, 14, 15, 16, 17, 18],
        SLIDING: [9, 8, 7, 12, 11, 10],
    }


def test_partial_window_hit_at_start() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=100)},
        block_size=10,
        enable_prefix_caching=True,
    )

    # Add tokens 0-1000 to prefix cache
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    sliding_blocks = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
    ctx.update(42)
    bm.step(ctx)
    bm.release(ctx)

    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    # Window is 890-990
    # 900-1000 is not viable since we have to keep one input token
    assert ctx.tokens.processed_length == 990

    # Introduce gap at 930-940
    uncommit(bm, SLIDING, {sliding_blocks[93]})
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    # Window is 830-930
    assert ctx.tokens.processed_length == 930

    # Delete 50-1000
    uncommit(bm, SLIDING, set(sliding_blocks[5:]))
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    # Window is 0-50 (this is ok despite range being < window_size)
    assert ctx.tokens.processed_length == 50

    uncommit(bm, SLIDING, set(sliding_blocks[1:]))
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    # Window is 0-10 (this is ok despite range being < window_size)
    assert ctx.tokens.processed_length == 10


def test_window_hit_with_irregular_ratio() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=500)},
        block_size=128,
        enable_prefix_caching=True,
    )

    # Notice that window size is 500, but block size is 128.
    # This means that we need at most cdiv(500, 128) = 4 blocks to cover window
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    sliding_blocks = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
    ctx.update(42)
    bm.step(ctx)
    bm.release(ctx)
    assert len(bm.pools[0].prefix_caches[SLIDING]) == 7

    #               0      1        2        3        4        5        6
    # blocks cover [0-128, 128-256, 256-384, 384-512, 512-640, 640-768, 768-896]
    #                                       <           window=500             >
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 896

    #               0      1        2        3        4        -        6
    # blocks cover [0-128, 128-256, 256-384, 384-512, 512-640, 640-768, 768-896]
    #                     <            window=500            >
    uncommit(bm, SLIDING, {sliding_blocks[5]})
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 640

    #               0      1        -        3        4        -        6
    # blocks cover [0-128, 128-256, 256-384, 384-512, 512-640, 640-768, 768-896]
    #              <  window=256  >
    uncommit(bm, SLIDING, {sliding_blocks[2]})
    ctx = make_ctx(num_tokens=1000)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 256


def test_alphabet() -> None:
    """The caches must agree, so the answer is the deepest point in both.

    The full cache can resume at any block up to I. The sliding cache can only
    resume where two committed blocks sit together, which is after D, after I,
    or after J. The deepest point in both is after I.
    """
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=3)},
        block_size=1,
        num_huge_blocks=30,
        enable_prefix_caching=True,
    )

    #          A     B     C     D     E     F     G     H     I     J
    #   Full  [A]   [B]   [C]   [D]   [E]   [F]   [G]   [H]   [I]    .
    #   Hits   ^     ^     ^     ^     ^     ^     ^     ^     ^
    #
    #    SWA   .     .    [C]   [D]    .    [F]    .    [H]   [I]   [J]
    #   Hits                     ^                             ^
    #
    #  Union                     ^                             ^ best_match=9
    alphabet = "ABCDEFGHIJ"
    ctx = make_ctx(num_tokens=len(alphabet))
    bm.claim(ctx)
    bm.alloc(ctx)
    full_blocks = bm.get_req_blocks_per_leaf(ctx)[FULL]
    sliding_blocks = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
    assert full_blocks == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sliding_blocks == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    ctx.update(42)
    bm.step(ctx)
    sliding_blocks_after = bm.get_req_blocks_per_leaf(ctx)[SLIDING]
    assert sliding_blocks_after == [0, 0, 0, 0, 0, 0, 0, 0, 19, 20]
    bm.release(ctx)

    to_uncommit = {full_blocks[alphabet.index("J")]}
    uncommit(bm, FULL, to_uncommit)
    to_uncommit = {sliding_blocks[alphabet.index(ch)] for ch in "ABEG"}
    uncommit(bm, SLIDING, to_uncommit)
    assert len(bm.pools[0].prefix_caches[FULL]) == 9
    assert len(bm.pools[0].prefix_caches[SLIDING]) == 6

    # Nine blocks reused.
    ctx = make_ctx(num_tokens=len(alphabet))
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 9
    # J's blocks are uncommitted but were never evicted from circulation, so
    # both caches reuse them directly instead of a fresh claim.
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        SLIDING: [0, 0, 0, 0, 0, 0, 0, 18, 19, 17],
    }
    bm.release(ctx)

    #          A     B     C     D     E     F     G     H     I     J
    #   Full  [A]   [B]   [C]   [D]   [E]   [F]   [G]   [H]   [I]    .
    #   Hits   ^     ^     ^     ^     ^     ^     ^     ^     ^
    #
    #    SWA   .     .    [C]   [D]    .    [F]    .    [H]    .    [J]
    #   Hits                     ^
    #
    #  Union                     ^ best_match=4

    to_uncommit = {sliding_blocks[alphabet.index("I")]}
    uncommit(bm, SLIDING, to_uncommit)
    assert len(bm.pools[0].prefix_caches[FULL]) == 9
    assert len(bm.pools[0].prefix_caches[SLIDING]) == 5

    # Four blocks reused. The full cache still reaches I, but a resume point
    # there needs a window the sliding cache no longer holds, and its depth
    # cannot stand in for one: the blocks below the resume point feed its own
    # caches, so recomputing from I would leave those queries reading nulls.
    ctx = make_ctx(num_tokens=len(alphabet))
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 4
    # Blocks 10 (full's J) and 17 (sliding's G) are uncommitted but were never
    # evicted from circulation, so they are reused directly ahead of a fresh
    # claim for the rest.
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 10, 21, 22, 23, 24, 25],
        SLIDING: [0, 0, 13, 14, 17, 26, 27, 28, 29, 16],
    }


def test_reset_prefix_cache_purges_all_caches() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=4)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctx = make_ctx(num_tokens=6)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)
    bm.step(ctx)
    bm.release(ctx)

    assert len(bm.pools[0].prefix_caches[FULL]) == 6
    assert len(bm.pools[0].prefix_caches[SLIDING]) == 6
    bm.reset_prefix_cache()
    assert len(bm.pools[0].prefix_caches[FULL]) == 0
    assert len(bm.pools[0].prefix_caches[SLIDING]) == 0


# ===--------------------------------------------------------------------=== #
# Exotic Configs (eg: SWA + SWA + Full)
# ===--------------------------------------------------------------------=== #

NEAR = "near"
FAR = "far"


def test_multi_window() -> None:
    bm = make_manager(
        {
            FULL: full(ratio=1),
            NEAR: sliding(ratio=1, window=3),
            FAR: sliding(ratio=1, window=5),
        },
        block_size=1,
        enable_prefix_caching=True,
    )
    ctx = make_ctx(num_tokens=8)

    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6, 7, 8],
        NEAR: [9, 10, 11, 12, 13, 14, 15, 16],
        FAR: [17, 18, 19, 20, 21, 22, 23, 24],
    }
    ctx.update(42)
    bm.step(ctx)
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6, 7, 8],
        # NEAR evicted 6 blocks as they fall out of the window
        NEAR: [0, 0, 0, 0, 0, 0, 15, 16],
        # FAR evicted 4 blocks as they fall out of the window
        FAR: [0, 0, 0, 0, 21, 22, 23, 24],
    }

    ctx = make_ctx(num_tokens=8)
    bm.claim(ctx)
    bm.alloc(ctx)

    # Prefix caching works as intended
    assert ctx.tokens.processed_length == 7
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 6, 7, 25],
        NEAR: [0, 0, 0, 0, 0, 14, 15, 26],
        FAR: [0, 0, 0, 20, 21, 22, 23, 27],
    }


def test_two_windows_converge_on_prefix_cache_hit() -> None:
    bm = make_manager(
        {
            FULL: full(ratio=1),
            NEAR: sliding(ratio=1, window=3),
            FAR: sliding(ratio=1, window=4),
        },
        block_size=1,
        enable_prefix_caching=True,
    )
    alphabet = "ABCDEFGHIJKL"
    ctxA = make_ctx(num_tokens=len(alphabet))
    bm.claim(ctxA)
    bm.alloc(ctxA)
    full_blocks = bm.get_req_blocks_per_leaf(ctxA)[FULL]
    near_blocks = bm.get_req_blocks_per_leaf(ctxA)[NEAR]
    far_blocks = bm.get_req_blocks_per_leaf(ctxA)[FAR]
    ctxA.update(42)
    bm.step(ctxA)
    bm.release(ctxA)

    #         A    B    C    D    E    F    G    H    I    J    K    L
    #  Full  [A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]  [I]  [J]  [K]   .
    #  Hits   ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^
    #
    # SWA=3  [A]  [B]  [C]  [D]  [E]  [F]   .   [H]  [I]   .   [K]  [L]
    #  Hits   ^    ^    ^    ^    ^    ^              ^              ^
    #
    # SWA=4  [A]   .   [C]  [D]  [E]   .    .   [H]  [I]  [J]  [K]  [L]
    #  Hits   ^                   ^                        ^    ^    ^
    #
    # Union   ^                   ^ best_match=5

    uncommit(bm, FULL, {full_blocks[alphabet.index("L")]})
    uncommit(bm, NEAR, {near_blocks[alphabet.index(ch)] for ch in "GJ"})
    uncommit(bm, FAR, {far_blocks[alphabet.index(ch)] for ch in "BFG"})

    ctx = make_ctx(num_tokens=len(alphabet))
    bm.claim(ctx)
    bm.alloc(ctx)
    assert ctx.tokens.processed_length == 5
    # Full's block 12 (L) and near's block 22 (J) are uncommitted but were
    # never evicted from circulation, so each cache reuses its own directly
    # ahead of a fresh claim for the rest.
    assert bm.get_req_blocks_per_leaf(ctx) == {
        FULL: [1, 2, 3, 4, 5, 12, 37, 38, 39, 40, 41, 42],
        NEAR: [0, 0, 0, 16, 17, 22, 43, 44, 45, 46, 47, 48],
        FAR: [0, 0, 27, 28, 29, 49, 50, 51, 52, 53, 54, 55],
    }


# ===--------------------------------------------------------------------=== #
# get_req_blocks
# ===--------------------------------------------------------------------=== #


def test_get_req_blocks_returns_first_leaf_blocks() -> None:
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=10)},
        block_size=1,
    )
    ctx = make_ctx(num_tokens=3)
    bm.claim(ctx)
    bm.alloc(ctx)

    per_leaf = bm.get_req_blocks_per_leaf(ctx)
    assert bm.get_req_blocks(ctx) == next(iter(per_leaf.values()))


def test_get_req_blocks_empty_when_claimed_not_allocated() -> None:
    bm = make_manager({FULL: full(ratio=1)}, block_size=1)
    ctx = make_ctx(num_tokens=3)
    bm.claim(ctx)

    assert bm.get_req_blocks(ctx) == []
    assert not bm.get_req_blocks(ctx)


# ===--------------------------------------------------------------------=== #
# get_prefix_cache_hit_counts
# ===--------------------------------------------------------------------=== #


def test_get_prefix_cache_hit_counts_reports_device_hits() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        enable_prefix_caching=True,
    )
    ctx_a = make_ctx(num_tokens=5)
    bm.claim(ctx_a)
    bm.alloc(ctx_a)
    ctx_a.update(42)
    bm.step(ctx_a)
    bm.release(ctx_a)

    ctx_b = make_ctx(num_tokens=5)
    hits = bm.get_prefix_cache_hit_counts(ctx_b)

    assert len(hits) == 1
    assert hits[0].device_blocks > 0
    assert hits[0].host_blocks == 0
    assert hits[0].disk_blocks == 0

    assert bm.get_prefix_cache_hit_counts(
        make_ctx_with_tokens([100, 101, 102, 103, 104])
    ) == [PrefixCacheHits()]


def test_get_prefix_cache_hit_counts_is_per_replica() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=8,
        num_replicas=2,
        enable_prefix_caching=True,
    )
    ctx_a = make_ctx(num_tokens=4)
    bm.claim(ctx_a, replica_idx=0)
    bm.alloc(ctx_a)
    ctx_a.update(42)
    bm.step(ctx_a)
    bm.release(ctx_a)

    ctx_b = make_ctx(num_tokens=4)
    hits = bm.get_prefix_cache_hit_counts(ctx_b)

    assert len(hits) == 2
    assert hits[0].device_blocks > 0
    assert hits[1].device_blocks == 0


# ===--------------------------------------------------------------------=== #
# Data Parallelism
# ===--------------------------------------------------------------------=== #
#
# Each replica gets its own pool, so a page id names a page within a replica
# and says nothing across them. Two requests on different replicas holding
# page 1 are holding different memory.


def test_replicas_draw_from_their_own_pool() -> None:
    """The same page ids on two replicas are different pages."""
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=4,
        num_replicas=2,
    )
    ctxA = make_ctx(num_tokens=3)
    ctxB = make_ctx(num_tokens=3)

    bm.claim(ctxA, replica_idx=0)
    bm.alloc(ctxA)

    # Replica 0 is spent and replica 1 has not been touched.
    assert bm.huge_block_count(0).free == 0
    assert bm.huge_block_count(1).free == 3

    bm.claim(ctxB, replica_idx=1)
    bm.alloc(ctxB)

    assert bm.get_req_blocks_per_leaf(ctxA)[FULL] == [1, 2, 3]
    assert bm.get_req_blocks_per_leaf(ctxB)[FULL] == [1, 2, 3]
    assert bm.huge_block_count(1).free == 0


def test_a_dry_replica_does_not_borrow_from_its_neighbor() -> None:
    """A request is refused on a full replica however idle the others are."""
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=4,
        num_replicas=2,
    )
    ctxA = make_ctx(num_tokens=3)
    bm.claim(ctxA, replica_idx=0)
    bm.alloc(ctxA)

    ctxB = make_ctx(num_tokens=3)
    bm.claim(ctxB, replica_idx=0)
    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctxB)

    # The neighbor's pages were never a candidate.
    assert bm.huge_block_count(1).free == 3


def test_a_request_stays_on_the_replica_that_claimed_it() -> None:
    """The claim pins the replica, so no later call has to name it again."""
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=8,
        num_replicas=2,
    )
    ctx = make_ctx(num_tokens=3)
    bm.claim(ctx, replica_idx=1)

    decode(bm, ctx)

    assert bm.huge_block_count(0).free == 7
    assert bm.huge_block_count(1).free == 4

    bm.release(ctx)

    assert bm.huge_block_count(1).free == 7


def test_replicas_keep_separate_prefix_caches() -> None:
    """A prefix warmed on one replica is unknown to the others.

    The hashes are content-derived and so identical across replicas, but they
    are looked up in the pool the request was claimed on, whose pages are the
    only ones it can adopt.
    """
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        num_replicas=2,
        enable_prefix_caching=True,
    )
    warm = make_ctx(num_tokens=4)
    bm.claim(warm, replica_idx=0)
    decode(bm, warm)
    bm.release(warm)

    assert len(bm.pools[0].prefix_caches[FULL]) == 4
    assert len(bm.pools[1].prefix_caches[FULL]) == 0

    # Same prompt, other replica: nothing to adopt, so it prefills from zero.
    elsewhere = make_ctx(num_tokens=4)
    bm.claim(elsewhere, replica_idx=1)
    bm.alloc(elsewhere)

    assert elsewhere.tokens.processed_length == 0
    assert bm.get_req_blocks_per_leaf(elsewhere)[FULL] == [1, 2, 3, 4]

    # Same prompt, same replica: three blocks adopted, one drawn for the tail.
    again = make_ctx(num_tokens=4)
    bm.claim(again, replica_idx=0)
    bm.alloc(again)

    assert again.tokens.processed_length == 3
    assert bm.get_req_blocks_per_leaf(again)[FULL] == [1, 2, 3, 5]


def test_reset_prefix_cache_clears_every_replica() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        num_replicas=2,
        enable_prefix_caching=True,
    )
    for replica_idx in (0, 1):
        ctx = make_ctx(num_tokens=4)
        bm.claim(ctx, replica_idx=replica_idx)
        decode(bm, ctx)
        bm.release(ctx)
        assert len(bm.pools[replica_idx].prefix_caches[FULL]) == 4

    bm.reset_prefix_cache()

    assert len(bm.pools[0].prefix_caches[FULL]) == 0
    assert len(bm.pools[1].prefix_caches[FULL]) == 0


# ===--------------------------------------------------------------------=== #
# DP Padding Dummies
# ===--------------------------------------------------------------------=== #


def test_a_dummy_draws_no_pages() -> None:
    """Padding dummies exist to equalize batch shapes, not to compute, so
    they must not spend the pool a real request needs."""
    bm = make_manager({FULL: full(ratio=1), SLIDING: sliding(ratio=2)})
    free_before = bm.huge_block_count(0).free

    dummy = make_ctx(num_tokens=1)
    bm.alloc_dummy(dummy)

    assert bm.get_req_blocks_per_leaf(dummy) == {FULL: [0], SLIDING: [0]}
    assert bm.huge_block_count(0).free == free_before


def test_a_dummy_points_at_its_own_replica_null_page() -> None:
    bm = make_manager({FULL: full(ratio=1)}, num_replicas=2)
    dummy = make_ctx(num_tokens=1)
    bm.alloc_dummy(dummy, replica_idx=1)

    assert (
        bm._leaves[FULL].req_to_blocks[dummy.request_id][0]
        is bm.pools[1].null_little_blocks[FULL]
    )


def test_releasing_a_dummy_gives_nothing_back() -> None:
    """The null page is shared by every dummy, so freeing one must not put it
    into circulation for the next allocation to hand out."""
    bm = make_manager({FULL: full(ratio=1)}, num_huge_blocks=4)
    dummy = make_ctx(num_tokens=1)
    bm.alloc_dummy(dummy)
    free_before = bm.huge_block_count(0).free

    bm.release(dummy)

    assert bm.huge_block_count(0).free == free_before
    assert not bm.contains(dummy)


# ===--------------------------------------------------------------------=== #
# Speculative Decoding
# ===--------------------------------------------------------------------=== #


def test_drafts_are_sized_into_the_allocation() -> None:
    """A speculative step verifies drafts and writes more behind them, all of
    which need a KV slot the allocation has to have drawn up front."""
    plain = make_manager({FULL: full(ratio=1)}, block_size=1)
    spec = make_manager({FULL: full(ratio=1)}, block_size=1, num_draft_tokens=3)

    for bm, expected in ((plain, 3), (spec, 9)):
        ctx = make_ctx(num_tokens=3)
        bm.claim(ctx)
        bm.alloc(ctx)
        assert len(bm.get_req_blocks_per_leaf(ctx)[FULL]) == expected


def test_block_drafts_get_the_bonus_position() -> None:
    """A block draft writes one position past the bonus token in a single
    batched forward, which autoregressive-draft accounting does not reserve."""
    autoregressive = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_draft_tokens=3,
        num_draft_tokens_per_step=1,
    )
    block_draft = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_draft_tokens=3,
        num_draft_tokens_per_step=3,
    )

    counts = []
    for bm in (autoregressive, block_draft):
        ctx = make_ctx(num_tokens=3)
        bm.claim(ctx)
        bm.alloc(ctx)
        counts.append(len(bm.get_req_blocks_per_leaf(ctx)[FULL]))

    assert counts == [9, 10]


# ===--------------------------------------------------------------------=== #
# Metrics
# ===--------------------------------------------------------------------=== #


def test_metrics_split_prompt_tokens_into_hits_and_misses() -> None:
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    warm = make_ctx(num_tokens=4)
    bm.claim(warm, replica_idx=0)
    decode(bm, warm)
    bm.release(warm)

    assert bm.metrics.cache_tokens == 0
    assert bm.metrics.input_tokens == 4

    bm.reset_metrics()

    # Same prompt: three blocks are adopted and only the tail is recomputed.
    again = make_ctx(num_tokens=4)
    bm.claim(again)
    bm.alloc(again)

    assert bm.metrics.cache_tokens == 3
    assert bm.metrics.input_tokens == 1
    assert bm.metrics.device_blocks_served == 3
    assert bm.metrics.prompt_tokens == 4
    assert bm.metrics.cache_hit_rate == 0.75
    assert again.cached_prefix_length == 3


def test_a_miss_records_a_zero_length_cached_prefix() -> None:
    """The scheduler reads ``cached_prefix_length`` per admission; leaving it
    unset would drop the request from the batch's hit-rate entirely."""
    bm = make_manager(
        {FULL: full(ratio=1)},
        block_size=1,
        num_huge_blocks=20,
        enable_prefix_caching=True,
    )
    ctx = make_ctx(num_tokens=4)
    bm.claim(ctx)
    bm.alloc(ctx)

    assert ctx.cached_prefix_length == 0
    assert bm.metrics.cache_tokens == 0
    assert bm.metrics.device_blocks_served == 0


def test_a_hit_counts_a_position_once_not_once_per_leaf() -> None:
    """Two leaves storing the same position are one reused position.

    ``device_blocks_served`` and ``cache_tokens`` measure the same prefix, so
    counting a leaf's row as its own block would make the two disagree by a
    factor of however many caches the model happens to have.
    """
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=99)},
        block_size=1,
        num_huge_blocks=40,
        enable_prefix_caching=True,
    )
    warm = make_ctx(num_tokens=4)
    bm.claim(warm)
    decode(bm, warm)
    bm.release(warm)
    bm.reset_metrics()

    again = make_ctx(num_tokens=4)
    bm.claim(again)
    bm.alloc(again)

    assert bm.metrics.device_blocks_served == 3
    assert bm.metrics.cache_tokens == 3


def test_reset_metrics_zeroes_the_counters() -> None:
    bm = make_manager({FULL: full(ratio=1)}, block_size=1)
    ctx = make_ctx(num_tokens=4)
    bm.claim(ctx)
    bm.alloc(ctx)
    assert bm.metrics.input_tokens == 4

    bm.reset_metrics()

    assert bm.metrics == KVCacheMetrics()


# ===--------------------------------------------------------------------=== #
# effective_max_seq_length
# ===--------------------------------------------------------------------=== #


def test_effective_max_seq_length_single_full_leaf() -> None:
    """One leaf gets every allocable huge block (block 0 is the null block),
    so capacity grows linearly."""
    bm = make_manager({FULL: full(ratio=1)}, block_size=4, num_huge_blocks=5)
    assert bm.effective_max_seq_length == 4 * 4


def test_effective_max_seq_length_balances_across_full_leaves() -> None:
    """Huge blocks go to whichever leaf has the least capacity so far."""
    bm = make_manager(
        {"a": full(ratio=1), "b": full(ratio=1)},
        block_size=1,
        num_huge_blocks=4,
    )
    # 3 allocable huge blocks: a=1,b=0 -> a=1,b=1 -> a=2,b=1.
    assert bm.effective_max_seq_length == 1


def test_effective_max_seq_length_weighs_by_ratio() -> None:
    """A higher ratio grows a leaf's capacity faster per huge block."""
    bm = make_manager(
        {"a": full(ratio=1), "b": full(ratio=3)},
        block_size=1,
        num_huge_blocks=4,
    )
    # 3 allocable huge blocks: a=1,b=0 -> a=1,b=3 -> a=2,b=3.
    assert bm.effective_max_seq_length == 2


def test_effective_max_seq_length_sliding_window_saturates_to_none() -> None:
    """A sliding-window-only geometry has no ceiling once windows fill.

    Once every leaf's window is fully covered, more blocks add no more
    capacity, so there is no finite bound on how long a sequence it can
    serve: the eviction rotates old blocks out as the window slides.
    """
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=3)},
        block_size=1,
        num_huge_blocks=10,
    )
    assert bm.effective_max_seq_length is None


def test_effective_max_seq_length_sliding_window_below_saturation() -> None:
    """Below saturation, a sliding-window leaf still grows like a full one."""
    bm = make_manager(
        {SLIDING: sliding(ratio=1, window=3)},
        block_size=1,
        num_huge_blocks=3,
    )
    # 2 allocable huge blocks, short of the 3 needed to saturate the window.
    assert bm.effective_max_seq_length == 2


def test_effective_max_seq_length_full_leaf_is_the_bottleneck_once_swa_caps() -> (
    None
):
    """A saturated sliding leaf drops out; the full leaf alone sets the bound."""
    bm = make_manager(
        {FULL: full(ratio=1), SLIDING: sliding(ratio=1, window=2)},
        block_size=1,
        num_huge_blocks=6,
    )
    # 5 allocable huge blocks: full=1,sliding=0 -> full=1,sliding=1
    # -> full=2,sliding=1 -> full=2,sliding capped (None) -> full=3.
    assert bm.effective_max_seq_length == 3
