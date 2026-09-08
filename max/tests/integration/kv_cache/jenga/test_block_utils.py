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
"""Tests for the intrusive free list every block pool allocates from."""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


def make_blocks(count: int) -> list[KVCacheBlock]:
    return [KVCacheBlock(bid) for bid in range(count)]


def queue_order(queue: FreeKVCacheBlockQueue) -> list[int]:
    """Walks the queue head to tail, checking it against its own bookkeeping.

    The queue is intrusive, so its bookkeeping can drift out of step with the
    links it maintains; every test here compares the two, which is the
    guarantee its callers take for granted.
    """
    bids: list[int] = []
    block = queue.free_list_head
    prev = None
    while block is not None:
        assert len(bids) <= len(queue), (
            "queue is cyclic or longer than reported"
        )
        bids.append(block.bid)
        assert block in queue, f"linked block {block.bid} is not marked free"
        assert block.prev_free_block is prev
        prev = block
        block = block.next_free_block

    assert len(bids) == len(queue)
    assert len(set(bids)) == len(bids), "block linked twice"
    assert queue.free_list_tail is prev
    return bids


def test_queue_starts_in_block_id_order() -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    assert queue_order(queue) == [0, 1, 2, 3]


def test_empty_queue() -> None:
    queue = FreeKVCacheBlockQueue()

    assert queue_order(queue) == []
    with pytest.raises(ValueError, match="No free blocks available"):
        queue.popleft()
    assert queue.peek_front() is None


def test_popleft_hands_out_the_least_recently_used_block() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    popped = [queue.popleft().bid for _ in range(3)]

    assert popped == [0, 1, 2]
    assert queue_order(queue) == []


def test_append_puts_a_block_back_at_the_tail() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    block = queue.popleft()
    queue.append(block)

    # Reuse makes a block the most recently used, so it is now last in line.
    assert queue_order(queue) == [1, 2, 0]


def test_appendleft_puts_a_block_back_at_the_head() -> None:
    blocks = make_blocks(3)
    queue = FreeKVCacheBlockQueue(blocks)

    block = queue.popleft()
    queue.appendleft(block)

    # Unlike append, the block is popped again before the others.
    assert queue_order(queue) == [0, 1, 2]


def test_appendleft_on_an_empty_queue_becomes_both_ends() -> None:
    blocks = make_blocks(1)
    queue = FreeKVCacheBlockQueue()

    queue.appendleft(blocks[0])

    assert queue.free_list_head is blocks[0]
    assert queue.free_list_tail is blocks[0]
    assert queue_order(queue) == [0]


def test_append_and_appendleft_interleave_correctly() -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue()

    queue.append(blocks[0])
    queue.appendleft(blocks[1])
    queue.append(blocks[2])
    queue.appendleft(blocks[3])

    assert queue_order(queue) == [3, 1, 0, 2]


def test_remove_from_the_middle() -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[1])
    queue.remove(blocks[2])

    assert queue_order(queue) == [0, 3]
    assert blocks[1].prev_free_block is None
    assert blocks[1].next_free_block is None


@pytest.mark.parametrize("bid", [0, 3])
def test_remove_from_an_end(bid: int) -> None:
    blocks = make_blocks(4)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[bid])

    assert bid not in queue_order(queue)
    assert len(queue) == 3


def test_remove_the_last_block_empties_the_queue() -> None:
    blocks = make_blocks(1)
    queue = FreeKVCacheBlockQueue(blocks)

    queue.remove(blocks[0])

    assert queue_order(queue) == []
    queue.append(blocks[0])
    assert queue_order(queue) == [0]


def test_membership_tracks_the_queue() -> None:
    blocks = make_blocks(2)
    queue = FreeKVCacheBlockQueue(blocks)

    assert blocks[0] in queue
    queue.remove(blocks[0])
    assert blocks[0] not in queue
    queue.append(blocks[0])
    assert blocks[0] in queue


def test_queues_over_the_same_block_ids_are_independent() -> None:
    # Two pools of the same size hold different blocks under the same ids, so
    # membership follows the block objects a queue holds, not the id alone.
    first_blocks = make_blocks(2)
    first = FreeKVCacheBlockQueue(first_blocks)
    second_blocks = make_blocks(2)
    second = FreeKVCacheBlockQueue(second_blocks)

    first.popleft()

    assert queue_order(first) == [1]
    assert queue_order(second) == [0, 1]


def test_churn_preserves_the_queue() -> None:
    # Interleaved removes and appends, the pattern a pool actually produces.
    blocks = make_blocks(6)
    queue = FreeKVCacheBlockQueue(blocks)
    live: list[KVCacheBlock] = []

    for _ in range(3):
        for _ in range(4):
            live.append(queue.popleft())
            queue_order(queue)
        # Requests free their blocks tail first, which is why a pool reverses
        # them before handing them back.
        while live:
            queue.append(live.pop())
            queue_order(queue)

    assert len(queue) == 6
