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

"""Device specific block pool for PagedAttention KVCache.

Supports allocating/freeing blocks and maps block hashes to committed blocks.

This logic is largely borrowed from vLLM v1:
- https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_manager.py#L1
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_utils.py#L1

TODO E2EOPT-116: Port block_pool.py and block_utils.py to Mojo
"""

from __future__ import annotations

import logging

from max.profiler import traced

from .block_utils import FreeKVCacheBlockQueue, KVCacheBlock

logger = logging.getLogger("max.pipelines")


class BlockPool:
    """A pool of fixed-size memory blocks for the paged KV cache."""

    @traced
    def __init__(
        self,
        total_num_blocks: int,
        enable_runtime_checks: bool = False,
    ) -> None:
        self.total_num_blocks = total_num_blocks
        self.enable_runtime_checks = enable_runtime_checks

        # A Block pool of all kv-cache blocks.
        self.pool: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.total_num_blocks)
        ]

        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.pool)

        # Mapping from block hash to a committed block.
        # A committed block is a full block with a block hash that can be shared
        # between requests for prefix caching. The cached block may be used by
        # running requests or in the free_block_queue that could potentially
        # be evicted.
        self.prefix_cache: dict[bytes, KVCacheBlock] = {}

        # Placeholder block for dummy / padding requests. It will never be freed.
        self.null_block = KVCacheBlock(
            self.total_num_blocks, is_null=True, ref_cnt=42
        )

    @traced
    def commit_into_prefix_cache(
        self,
        block_hash: bytes,
        block: KVCacheBlock,
    ) -> None:
        """Commit a block into the prefix cache."""
        if block.is_null:
            raise ValueError("Cannot commit null block into prefix cache")

        assert block.block_hash is None
        block.block_hash = block_hash

        # Commit the block into the prefix cache.
        assert block_hash not in self.prefix_cache
        self.prefix_cache[block_hash] = block

    def get_or_commit_into_prefix_cache(
        self,
        block_hash: bytes,
        block: KVCacheBlock,
    ) -> KVCacheBlock | None:
        """Get or commit a block into the prefix cache.

        If there already exists a committed block with the same hash, we return
        the already committed block. Otherwise, we commit the provided block
        into the prefix cache and return None.
        """
        hash_value = block_hash
        if hash_value in self.prefix_cache:
            # Check if a block with the same hash is already committed.
            # If so, we reuse the already committed block.
            prefix_cache_block = self.prefix_cache[hash_value]
            if block.bid == prefix_cache_block.bid:
                return None

            self.touch(prefix_cache_block)

            # Free the block we currently have.
            assert block.block_hash is None
            self.free_block(block)

            return prefix_cache_block

        self.commit_into_prefix_cache(block_hash, block)
        return None

    @traced
    def uncommit_block(self, block: KVCacheBlock) -> None:
        """Evict a block from the prefix cache."""
        assert block.block_hash is not None
        hash_value = block.block_hash

        # Nothing to do if it is not committed.
        if hash_value not in self.prefix_cache:
            return

        del self.prefix_cache[hash_value]
        block.block_hash = None

    @traced
    def alloc_block(self) -> tuple[KVCacheBlock, bytes | None]:
        """Allocates a block from the free block queue."""
        # First allocate block
        curr_block = self.free_block_queue.popleft()
        assert curr_block.ref_cnt == 0

        # If the block is committed into prefix cache, evict it.
        block_hash = curr_block.block_hash
        if block_hash is not None:
            self.uncommit_block(curr_block)

        curr_block.ref_cnt += 1
        assert curr_block.block_hash is None
        return curr_block, block_hash

    @traced
    def free_block(self, block: KVCacheBlock) -> None:
        """Frees a block by decreasing its reference count.

        If the reference count is 0, the block is added to the free block queue.
        A block can be in both the prefix cache and the free block queue at the
        same time.
        """
        if block.is_null:
            return

        block.ref_cnt -= 1
        assert block.ref_cnt >= 0
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)

    @traced
    def touch(self, block: KVCacheBlock) -> None:
        """Increases the block's reference count by 1 and may remove it from the free queue.

        Used when a block is hit by another request with the same prefix.
        """
        # ref_cnt=0 means this block is in the free list (i.e. eviction
        # candidate), so remove it.
        if block.ref_cnt == 0:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1

    @property
    def free_blocks(self) -> set[int]:
        """Get the set of free blocks."""
        return self.free_block_queue.free_blocks

    @property
    def num_free_blocks(self) -> int:
        """Get the number of free blocks."""
        return self.free_block_queue.num_free_blocks

    def reset_prefix_cache(self) -> None:
        """Reset the prefix cache."""
        blocks_to_purge = []
        for hash, block in self.prefix_cache.items():
            assert block.block_hash is not None
            if block.ref_cnt > 0:
                continue
            block.block_hash = None
            blocks_to_purge.append(hash)
        # Delete separately to avoid modifying the dictionary size while iterating
        for hash in blocks_to_purge:
            del self.prefix_cache[hash]
        logger.info(f"Purged {len(blocks_to_purge)} blocks from prefix cache")

    @traced
    def assert_runtime_invariants(self, active_bids: list[int]) -> None:
        """Asserts runtime invariants when runtime checks are enabled."""
        if not self.enable_runtime_checks:
            return

        # Check that all blocks in the prefix cache are committed.
        for block_hash, block in self.prefix_cache.items():
            assert block.block_hash is not None
            assert block.block_hash == block_hash

        # Check that the total number of blocks is correct.
        assert (
            self.num_free_blocks + len(set(active_bids))
            == self.total_num_blocks
        )
