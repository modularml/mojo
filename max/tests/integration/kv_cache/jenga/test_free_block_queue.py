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
"""Tests for the intrusive free list that every block pool allocates from.

The list is intrusive -- the links live on the blocks themselves -- so its
bookkeeping can drift out of step with the links it maintains. Every test here
walks the links and compares them against what the queue reports, which is the
guarantee its callers (``BlockPool``, ``JengaBlockPool``) take for granted.
"""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


def linked_bids(queue: FreeKVCacheBlockQueue) -> list[int]:
    """Walks the list head to tail, checking it against the queue's own state."""
    bids: list[int] = []
    block = queue.free_list_head
    while block is not None:
        assert len(bids) <= len(queue), "list is cyclic or longer than reported"
        bids.append(block.bid)
        if block.next_free_block is not None:
            assert block.next_free_block.prev_free_block is block
        block = block.next_free_block

    assert len(bids) == len(queue)
    assert set(bids) == queue.free_blocks
    assert len(set(bids)) == len(bids), "block linked twice"
    if bids:
        assert queue.free_list_head is not None
        assert queue.free_list_tail is not None
        assert queue.free_list_head.bid == bids[0]
        assert queue.free_list_tail.bid == bids[-1]
        assert queue.free_list_head.prev_free_block is None
        assert queue.free_list_tail.next_free_block is None
    else:
        assert queue.free_list_head is None
        assert queue.free_list_tail is None
    return bids


def make_blocks(count: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(bid) for bid in range(count)]


def test_queue_starts_in_block_id_order() -> None:
    queue = FreeKVCacheBlockQueue(make_blocks(4))

    assert linked_bids(queue) == [0, 1, 2, 3]


def test_empty_queue() -> None:
    queue = FreeKVCacheBlockQueue()

    assert linked_bids(queue) == []
    with pytest.raises(ValueError, match="No free blocks available"):
        queue.popleft()


def test_popleft_hands_out_the_least_recently_used_block() -> None:
    queue = FreeKVCacheBlockQueue(make_blocks(3))

    assert [queue.popleft().bid for _ in range(3)] == [0, 1, 2]
    assert linked_bids(queue) == []


def test_append_puts_a_block_back_at_the_tail() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.append(queue.popleft())

    # Reuse makes a block the most recently used, so it is now last in line.
    assert linked_bids(queue) == [1, 2, 0]


def test_remove_from_the_middle() -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[1])
    queue.remove(blocks[2])

    assert linked_bids(queue) == [0, 3]
    assert blocks[1].prev_free_block is None
    assert blocks[1].next_free_block is None


@pytest.mark.parametrize("index", [0, 3])
def test_remove_from_an_end(index: int) -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[index])

    assert index not in linked_bids(queue)
    assert len(queue) == 3


def test_remove_the_last_block_empties_the_queue() -> None:
    blocks = make_blocks(1)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[0])

    assert linked_bids(queue) == []
    queue.append(blocks[0])
    assert linked_bids(queue) == [0]


def test_membership_tracks_the_list() -> None:
    blocks = make_blocks(2)
    queue = FreeKVCacheBlockQueue(blocks)

    assert blocks[0] in queue
    queue.remove(blocks[0])
    assert blocks[0] not in queue
    queue.append(blocks[0])
    assert blocks[0] in queue


def test_queues_over_the_same_block_ids_are_independent() -> None:
    # Two pools of the same size hold different blocks under the same ids, so
    # membership is by identity of the queue, not by id alone.
    first = FreeKVCacheBlockQueue(make_blocks(2))
    second = FreeKVCacheBlockQueue(make_blocks(2))

    first.popleft()

    assert linked_bids(first) == [1]
    assert linked_bids(second) == [0, 1]


def test_churn_preserves_the_list() -> None:
    """Interleaved removes and appends, the pattern a pool actually produces."""
    blocks = make_blocks(6)
    queue = FreeKVCacheBlockQueue(blocks)
    live: list[KVCacheBlock] = []

    for _ in range(3):
        for _ in range(4):
            live.append(queue.popleft())
            linked_bids(queue)
        # Requests free their blocks tail first, which is why a pool reverses
        # them before handing them back.
        while live:
            queue.append(live.pop())
            linked_bids(queue)

    assert len(queue) == 6
    assert sorted(linked_bids(queue)) == [0, 1, 2, 3, 4, 5]
