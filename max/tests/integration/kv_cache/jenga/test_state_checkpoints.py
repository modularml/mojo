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
"""Tests for recurrent-state checkpoints in the Jenga block manager."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVLeafRegion,
    PagedKVLeafRegion,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.kv_cache.paged_kv_cache.recurrent_coordinator import (
    RecurrentKVGroupCoordinator,
)
from max.pipelines.request.base import RequestID

BLOCK_SIZE = 4

FULL = "full"
CONV = "state/conv"
REC = "state/rec"


STATE_PARAMS_REGIONS = (
    RecurrentStateRegion(
        leaf_id=CONV, num_layers=2, row_shape=(4,), dtype=DType.bfloat16
    ),
    RecurrentStateRegion(
        leaf_id=REC, num_layers=3, row_shape=(2,), dtype=DType.bfloat16
    ),
)
"""Two state leaves with *different* layer counts."""

STATE_PARAMS = RecurrentStateParams(
    devices=[DeviceRef.CPU()], regions=STATE_PARAMS_REGIONS
)
"""The tree the group factory reads its geometry out of."""


STATE_LEAVES = {
    FULL: KVLeafInfo(1, KVCacheGroupId.full()),
    CONV: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    REC: KVLeafInfo(1, KVCacheGroupId.recurrent()),
}


def state_group(bm: JengaBlockManager) -> RecurrentKVGroupCoordinator:
    """The group under test."""
    group = bm.groups[KVCacheGroupId.recurrent()]
    assert isinstance(group, RecurrentKVGroupCoordinator)
    return group


def live(bm: JengaBlockManager, ctx: TextContext) -> dict[str, int] | None:
    """The blocks this request's recurrence runs in, drawing them if new."""
    return state_group(bm).live_blocks(ctx.request_id)


def resume(
    bm: JengaBlockManager, ctx: TextContext
) -> Mapping[str, tuple[int | None, int]]:
    """The block each leaf's next forward resumes its state from."""
    return state_group(bm).resume(ctx, 0)


def checkpoint(
    bm: JengaBlockManager, ctx: TextContext
) -> Mapping[str, tuple[int, int]]:
    """The blocks the forward that just ran should be copied between."""
    return state_group(bm).checkpoint(ctx, 0)


def make_manager(
    num_huge_blocks: int = 999,
    *,
    enable_prefix_caching: bool = True,
) -> JengaBlockManager:
    pools = create_pools(STATE_LEAVES, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
        groups=create_groups(STATE_LEAVES, pools, BLOCK_SIZE),
        leaves=STATE_PARAMS.leaves(),
    )


def make_ctx(num_tokens: int, *, offset: int = 0) -> TextContext:
    """A request whose prompt is ``num_tokens`` tokens from ``offset`` up."""
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(
            np.arange(offset, offset + num_tokens, dtype=np.int64)
        ),
    )


def forward(bm: JengaBlockManager, ctx: TextContext, token: int = 42) -> None:
    """Runs one step the way the pipeline does."""
    bm.alloc(ctx)
    resume(bm, ctx)
    ctx.update(token)
    # The checkpoint runs before the commit, so the copy reads a block nothing
    # has published or freed.
    checkpoint(bm, ctx)
    bm.step(ctx)


def published(bm: JengaBlockManager) -> list[bytes]:
    return list(bm.pools[0].prefix_caches[CONV])


# ===--------------------------------------------------------------------=== #
# The blocks a forward runs in
# ===--------------------------------------------------------------------=== #


def test_a_claimed_request_runs_in_blocks_of_every_state_leaf() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    blocks = live(bm, ctx)
    assert blocks is not None
    assert set(blocks) == {CONV, REC}


def test_a_fresh_request_resumes_from_a_wiped_block() -> None:
    # A drawn block holds whatever its last request wrote, and the kernels
    # read the incoming state unconditionally, so it has to be zeroed.
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    runs_in = live(bm, ctx)
    assert runs_in is not None
    assert resume(bm, ctx) == {
        leaf_id: (None, runs_in[leaf_id]) for leaf_id in runs_in
    }, "no source block means the rows are wiped"


def test_a_request_that_has_run_carries_its_own_state_forward() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    forward(bm, ctx)

    assert resume(bm, ctx) == {}, (
        "a second forward continues in the block the first one wrote"
    )


def test_the_blocks_are_kept_across_a_forward_that_publishes_nothing() -> None:
    # Drawn once and written in place; only publishing moves a request on.
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)
    first = live(bm, ctx)

    bm.alloc(ctx)

    assert live(bm, ctx) == first


def test_a_pool_with_nothing_to_spare_refuses_the_request() -> None:
    # A state block holds the only copy of something no recomputation can
    # rebuild, so there is nothing to degrade to.
    bm = make_manager(num_huge_blocks=3)
    ctx = make_ctx(8)
    bm.claim(ctx)

    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctx)


# ===--------------------------------------------------------------------=== #
# Publishing and reuse
# ===--------------------------------------------------------------------=== #


def test_cached_kv_without_a_state_is_not_reusable() -> None:
    # One cache_length serves every layer, so KV the recurrence has no state
    # for cannot be resumed from.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    forward(bm, first)
    for block in list(bm.pools[0].prefix_caches[CONV].values()):
        bm.pools[0].uncommit_block(block)
    bm.release(first)

    second = make_ctx(16)
    bm.claim(second)
    bm.alloc(second)

    assert second.cached_prefix_length == 0


def test_a_published_state_licenses_a_hit_at_its_block() -> None:
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        forward(bm, first)
    bm.release(first)
    assert published(bm)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    # The first request ran on past its prompt, so the boundary at 16 was
    # committed by a later step and the repeat resumes at it.
    assert second.cached_prefix_length == 16


def test_resuming_reads_the_published_state_where_it_lies() -> None:
    # The matched blocks may be matched again, so the request reads them and
    # writes its own rather than writing theirs.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        forward(bm, first)
    bm.release(first)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    resumed = resume(bm, second)
    assert resumed, "a hit must give the forward a block to copy in"
    runs_in = live(bm, second)
    assert runs_in is not None
    for leaf_id, (src, dst) in resumed.items():
        assert src is not None, "the hit named a block to read"
        assert dst == runs_in[leaf_id]
        assert src != dst, (
            "the matched blocks may be matched again, so the request copies"
            " them into its own rather than writing theirs"
        )


def test_a_forward_publishes_kv_but_not_state() -> None:
    # A state is published at a boundary chosen for it, not by every forward.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)
    bm.step(ctx)

    assert bm.pools[0].prefix_caches[FULL], "KV should commit every block"
    assert not published(bm), "a forward should publish no state"


def test_a_checkpoint_moves_the_request_onto_a_successor() -> None:
    # The block that ran holds the state its boundary hash names, so it is
    # published where it lies and the request continues elsewhere.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    before = live(bm, ctx)
    ctx.update(42)
    checkpoint(bm, ctx)

    assert live(bm, ctx) != before


# ===--------------------------------------------------------------------=== #
# Checkpoints
# ===--------------------------------------------------------------------=== #


def test_a_forward_ending_on_a_boundary_checkpoints() -> None:
    # The block that ran holds the state at the boundary the forward ended
    # on, which is the one boundary anyone can name.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    copies = checkpoint(bm, ctx)
    assert set(copies) == {region.leaf_id for region in STATE_PARAMS_REGIONS}
    leaves = STATE_PARAMS.leaves()
    for leaf_id, (src, dst) in copies.items():
        assert src != dst, "a successor is a block of its own"
        rows = leaves[leaf_id].bound_row_copies(src, dst)
        for src_rows, dst_rows in rows.values():
            assert len(src_rows) == len(dst_rows), (
                "a block folds to the same rows either way"
            )


def test_a_forward_ending_mid_block_rotates_nothing() -> None:
    # Shorter than one block: there is no boundary behind it to name.
    bm = make_manager()
    ctx = make_ctx(3)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert checkpoint(bm, ctx) == {}


def test_a_forward_ending_past_a_boundary_checkpoints_nothing() -> None:
    # Past a boundary and off it; the case above is num_blocks zero.
    bm = make_manager()
    ctx = make_ctx(18)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert ctx.tokens.processed_length % BLOCK_SIZE != 0, "fixture must be off"
    assert ctx.tokens.processed_length // BLOCK_SIZE > 0, "and past a boundary"
    assert checkpoint(bm, ctx) == {}


def test_one_checkpoint_is_outstanding_at_a_time() -> None:
    # The row holds a published block until its boundary is committed.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert checkpoint(bm, ctx) != {}
    assert checkpoint(bm, ctx) == {}


def test_the_block_published_is_the_one_the_forward_ran_in() -> None:
    # Nothing is handed to the cache that the request is still writing: the
    # successor it moves to is untouched, and the block it left is not.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ran_in = live(bm, ctx)
    assert ran_in is not None
    ctx.update(42)

    copies = checkpoint(bm, ctx)

    for leaf_id, (src, _) in copies.items():
        assert src == ran_in[leaf_id]


def test_a_checkpoint_is_published_by_the_step_that_reaches_it() -> None:
    # The checkpoint moves the block out of the slot the recurrence runs in
    # before the commit scans the row, so the boundary it names is
    # committable straight away rather than a step later.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    forward(bm, ctx)

    assert published(bm), "the step that reaches the boundary names it"


def test_a_boundary_reached_while_generating_is_checkpointed() -> None:
    # A chat turn's prompt is the last turn's plus its answer, so what is
    # generated here is the prefix the next turn extends.
    bm = make_manager()
    ctx = make_ctx(6)
    bm.claim(ctx)
    forward(bm, ctx)

    kept = 0
    for _ in range(10):
        forward(bm, ctx)
        kept = len(published(bm))

    assert kept >= 2, "generation keeps advancing the boundary it publishes"


def test_nothing_is_published_when_prefix_caching_is_off() -> None:
    # The blocks a recurrence runs in are still drawn: it cannot run without
    # them.
    bm = make_manager(enable_prefix_caching=False)
    ctx = make_ctx(16)
    bm.claim(ctx)
    for _ in range(4):
        forward(bm, ctx)

    assert live(bm, ctx) is not None
    assert not published(bm)


def test_release_commits_nothing() -> None:
    # A checkpoint is committed by a later step, so one still outstanding at
    # release is never published.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    forward(bm, ctx)
    before = len(published(bm))

    bm.release(ctx)

    assert len(published(bm)) == before


# ============================================================================
# The run every group accepts at once
# ============================================================================

HYBRID_SLIDING = "sliding"

HYBRID_LEAVES = {
    FULL: KVLeafInfo(1, KVCacheGroupId.full()),
    HYBRID_SLIDING: KVLeafInfo(
        1, KVCacheGroupId("sliding_window", 3 * BLOCK_SIZE)
    ),
    CONV: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    REC: KVLeafInfo(1, KVCacheGroupId.recurrent()),
}
"""All three kinds of group in one manager."""


def hybrid_leaves() -> dict[str, KVLeafRegion]:
    """The paged leaves alongside the state ones the params object owns."""
    paged = {
        leaf_id: PagedKVLeafRegion(
            leaf_id=leaf_id,
            group_id=info.group_id,
            bytes_per_page=1,
            page_size=BLOCK_SIZE,
        )
        for leaf_id, info in HYBRID_LEAVES.items()
        if not info.group_id.is_recurrent()
    }
    return {**paged, **STATE_PARAMS.leaves()}


def make_hybrid_manager(num_huge_blocks: int = 999) -> JengaBlockManager:
    pools = create_pools(HYBRID_LEAVES, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        block_size=BLOCK_SIZE,
        groups=create_groups(HYBRID_LEAVES, pools, BLOCK_SIZE),
        leaves=hybrid_leaves(),
    )


def publish(bm: JengaBlockManager, leaf_id: str, keys: Sequence[bytes]) -> None:
    """Commits one block per key in a single leaf, then drops the references."""
    pool = bm.pools[0]
    blocks = [pool.alloc_block(leaf_id) for _ in keys]
    for block, key in zip(blocks, keys, strict=True):
        pool.commit_into_prefix_cache(key, block)
    for block in reversed(blocks):
        pool.free_block(block)


@pytest.mark.parametrize("seed", range(16))
def test_a_hybrid_hit_settles_on_the_deepest_run_every_group_accepts(
    seed: int,
) -> None:
    """The settling loop finds the deepest run no group shortens.

    The expected run is the deepest prefix every group accepts at once,
    computed by brute force since the loop's own answer is under test.
    """
    rng = random.Random(seed)
    for _ in range(200):
        bm = make_hybrid_manager()
        keys = [f"k{idx}".encode() for idx in range(rng.randint(0, 10))]
        # Each leaf publishes its own subset, so the groups disagree about
        # how deep a hit they can serve.
        for leaf_id in HYBRID_LEAVES:
            publish(
                bm,
                leaf_id,
                [key for key in keys if rng.random() < 0.7],
            )

        groups = list(bm.groups.values())
        expected = max(
            n
            for n in range(len(keys), -1, -1)
            if all(g.longest_cache_hit(keys[:n], 0) == n for g in groups)
        )
        assert (
            bm._find_longest_device_prefix_cache_hit(keys, 0, False) == expected
        ), f"seed={seed} keys={len(keys)}"
