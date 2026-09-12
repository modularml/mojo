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
"""The fungible page pool that every flat KV cache draws from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import lcm

from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from .block_utils import (
    FreeHugeKVCacheBlockQueue,
    FreeLittleKVCacheBlockQueue,
    HugeKVCacheBlock,
    InsufficientBlocksError,
    LittleKVCacheBlock,
)

# A padded page may spend at most this share of itself on bytes the kernel
# never reads. Generous enough to rescue a coprime geometry, tight enough that
# a leaf never pays for more padding than data.
_MAX_PADDING_FRACTION = 0.25

# The search will not consider a huge block larger than this. Past it the
# allocation quantum is coarse enough that stranding beats any padding it saves.
_MAX_HUGE_PAGE_BYTES = 128 * 1024 * 1024

# How many little pages of one cache a huge block is considered to hold. Bounds
# the candidate search; ratios above this imply a page so much smaller than the
# block that padding it tightly is free anyway.
_MAX_TILING_RATIO = 256


def compute_jenga_ratios(
    available_bytes: int,
    cache_sizes: Mapping[str, int],
    include_null_block: bool = True,
) -> tuple[int, int, dict[str, int]]:
    """Fits a byte budget to a huge block geometry every cache tiles exactly.

    A huge block is the least common multiple of the caches' page sizes, so it
    holds a whole number of pages of each: ``ratios[cache_id]`` of them. The
    budget therefore backs ``num_huge_blocks`` huge blocks -- that is,
    ``num_huge_blocks * ratios[cache_id]`` pages of each cache -- the first of
    which is the null block every cache shares.

    Args:
        available_bytes: The per-device KV budget the pool may occupy.
        cache_sizes: Each cache's page size in bytes.
        include_null_block: Whether to include the null block.

    Returns:
        How many huge blocks the budget holds, size of each huge block in bytes,
        and how many pages of each cache it holds.

    Raises:
        ValueError: If the arguments are not positive, or if the budget is too
            small to hold a null block and one allocatable block.
    """
    if len(cache_sizes) == 0:
        raise ValueError(f"cache_sizes must be non-empty, found {cache_sizes}")
    if any(size <= 0 for size in cache_sizes.values()):
        raise ValueError(f"cache_sizes must be positive, found {cache_sizes}")
    if available_bytes <= 0:
        raise ValueError(
            f"available_bytes must be positive, found {available_bytes}"
        )

    huge_page_bytes = lcm(*cache_sizes.values())
    num_huge_blocks = available_bytes // huge_page_bytes

    pages = ", ".join(
        f"{cache_id}={to_human_readable_bytes(size)}"
        for cache_id, size in cache_sizes.items()
    )
    if include_null_block and num_huge_blocks < 2:
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to "
            f"build a pool. A huge block is the least common multiple of the "
            f"page sizes ({pages}), so it takes "
            f"{to_human_readable_bytes(huge_page_bytes)}, and the pool needs "
            f"at least two of them -- "
            f"{to_human_readable_bytes(2 * huge_page_bytes)} -- because huge "
            f"block 0 is the null page every cache shares."
        )
    if num_huge_blocks < 1:
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to "
            f"build a pool. A huge block is the least common multiple of the "
            f"page sizes ({pages}), so it takes "
            f"{to_human_readable_bytes(huge_page_bytes)}, and the pool needs "
            f"at least one of them."
        )

    return (
        num_huge_blocks,
        huge_page_bytes,
        {
            cache_id: huge_page_bytes // size
            for cache_id, size in cache_sizes.items()
        },
    )


@dataclass(frozen=True)
class JengaGeometry:
    """A huge-block geometry every cache tiles exactly, once padded.

    :func:`compute_jenga_ratios` demands exact tiling, which makes the huge
    block the least common multiple of the page sizes -- cheap when they divide
    each other, ruinous when a coprime factor survives into one of them (an odd
    layer count, a head width that is not a power of two). Padding each page up
    to a divisor of a *searched* huge block buys tractability back: a leaf
    spends a few percent of its page on bytes the kernel never reads, and the
    huge block stays small enough to allocate against a real budget.

    """

    num_huge_blocks: int
    """How many huge blocks the budget holds, counting the null block when one
    was requested."""

    huge_page_bytes: int
    """Size of one huge block."""

    ratios: dict[str, int]
    """How many little pages of each cache one huge block holds."""

    padded_sizes: dict[str, int]
    """Each cache's page size after padding. Every value divides
    ``huge_page_bytes`` and is a whole number of that cache's rows."""

    def padding_fraction(self, cache_id: str, data_bytes: int) -> float:
        """Returns the share of ``cache_id``'s page that is padding."""
        return self.padded_sizes[cache_id] / data_bytes - 1


def _smallest_tiling_page(
    huge_page_bytes: int, data_bytes: int, row_bytes: int
) -> int | None:
    """Returns the smallest page holding ``data_bytes`` that tiles the block.

    A page must divide ``huge_page_bytes``, so a huge block holds a whole
    number of them, and must be a whole number of rows, because
    ``PagedKVCache`` expresses paging as row arithmetic -- ``_stride()`` divides
    the page stride by ``num_heads * head_size`` and that division has to be
    exact, or there is no row index for the start of a page.

    Searching the ratio rather than the page keeps this cheap: the ratio cannot
    exceed ``huge_page_bytes // data_bytes``, a few hundred at the sizes we
    allocate, and the largest valid ratio yields the smallest page.

    Returns:
        The page size, or ``None`` when no divisor of ``huge_page_bytes``
        qualifies.
    """
    for ratio in range(huge_page_bytes // data_bytes, 0, -1):
        if huge_page_bytes % ratio:
            continue
        page = huge_page_bytes // ratio
        if page % row_bytes == 0:
            return page
    return None


def _fit_huge_page(
    huge_page_bytes: int,
    cache_sizes: Mapping[str, int],
    row_sizes: Mapping[str, int],
    max_padding_fraction: float,
) -> dict[str, int] | None:
    """Pads every cache onto ``huge_page_bytes``, or gives up on it.

    Returns ``None`` if any cache cannot tile the block, or would have to spend
    more than ``max_padding_fraction`` of its page on padding to do so.
    """
    padded: dict[str, int] = {}
    for cache_id, data_bytes in cache_sizes.items():
        page = _smallest_tiling_page(
            huge_page_bytes, data_bytes, row_sizes[cache_id]
        )
        if page is None or page / data_bytes - 1 > max_padding_fraction:
            return None
        padded[cache_id] = page
    return padded


def _geometry_cost(
    available_bytes: int,
    huge_page_bytes: int,
    cache_sizes: Mapping[str, int],
    padded_sizes: Mapping[str, int],
    include_null_block: bool,
) -> float | None:
    """Returns the share of the budget a geometry cannot put cache data in.

    Two terms, and they pull against each other -- which is why the huge block
    is chosen on their sum rather than on padding alone. A larger block lets
    every page land closer above its data (less padding) but quantises the
    budget more coarsely, so it strands a bigger remainder and spends more on
    the null block. Minimising padding alone walks straight into the coarse end.

    Returns ``None`` when the budget cannot hold the blocks the pool needs.
    """
    num_huge_blocks = available_bytes // huge_page_bytes
    if num_huge_blocks < (2 if include_null_block else 1):
        return None

    padding = sum(
        padded_sizes[cache_id] / data_bytes - 1
        for cache_id, data_bytes in cache_sizes.items()
    ) / len(cache_sizes)
    unusable = available_bytes - num_huge_blocks * huge_page_bytes
    if include_null_block:
        unusable += huge_page_bytes
    return padding + unusable / available_bytes


def _huge_page_candidates(
    cache_sizes: Mapping[str, int],
    row_sizes: Mapping[str, int],
    ceiling: int,
) -> list[int]:
    """Returns the huge-block sizes worth pricing, smallest first.

    Every page is a whole number of its own rows and divides the huge block, so
    the block is necessarily a multiple of ``lcm(row_sizes)`` -- which is the
    grid the search runs on. On that grid the sizes that matter are the ones
    where some cache tiles *tightly*: for each cache and each small ratio, the
    smallest grid point at or above ``ratio * page``. That keeps the candidate
    set in the low thousands while still containing the optimum, since a
    geometry is only as good as its worst-padded leaf.
    """
    grid = lcm(*row_sizes.values())
    floor = max(cache_sizes.values())
    candidates = set()
    for data_bytes in cache_sizes.values():
        for ratio in range(1, _MAX_TILING_RATIO + 1):
            size = ceildiv(ratio * data_bytes, grid) * grid
            if size < floor:
                continue
            if size > ceiling:
                break
            candidates.add(size)
    return sorted(candidates)


def plan_jenga_geometry(
    available_bytes: int,
    cache_sizes: Mapping[str, int],
    row_sizes: Mapping[str, int],
    *,
    include_null_block: bool = True,
    max_padding_fraction: float = _MAX_PADDING_FRACTION,
) -> JengaGeometry:
    """Fits a byte budget to a huge-block geometry, padding pages to get there.

    The exact-tiling geometry :func:`compute_jenga_ratios` computes is priced
    first and wins whenever it is competitive, so a model whose page sizes
    already divide each other is allocated exactly as before. When a coprime
    factor makes that least common multiple too coarse to allocate -- the case
    that takes a vision tower pooled with a text cache to a 23.7 GiB huge block
    -- the search pads each page up to a divisor of a smaller block instead.

    Args:
        available_bytes: The per-device KV budget the pool may occupy.
        cache_sizes: Each cache's page size in bytes, before padding.
        row_sizes: The granularity each cache's padded page must be a whole
            number of -- a row, ``num_heads * head_size * dtype_size``, widened
            where a kernel needs more alignment than a row gives.
        include_null_block: Whether to include the null block.
        max_padding_fraction: The most of a page any one cache may spend on
            padding. Caps the search rather than the result: a geometry needing
            more is simply not considered.

    Returns:
        The geometry to allocate.

    Raises:
        ValueError: If the arguments are not positive or disagree on cache ids,
            or if no geometry fits the budget.
    """
    if len(cache_sizes) == 0:
        raise ValueError(f"cache_sizes must be non-empty, found {cache_sizes}")
    if any(size <= 0 for size in cache_sizes.values()):
        raise ValueError(f"cache_sizes must be positive, found {cache_sizes}")
    if any(size <= 0 for size in row_sizes.values()):
        raise ValueError(f"row_sizes must be positive, found {row_sizes}")
    if cache_sizes.keys() != row_sizes.keys():
        raise ValueError(
            f"cache_sizes and row_sizes must cover the same caches, found "
            f"{sorted(cache_sizes)} and {sorted(row_sizes)}"
        )
    for cache_id, data_bytes in cache_sizes.items():
        if data_bytes % row_sizes[cache_id]:
            raise ValueError(
                f"cache {cache_id!r} holds {data_bytes} bytes per page, which "
                f"is not a whole number of its {row_sizes[cache_id]}-byte "
                f"rows. A page the kernel addresses by row cannot be a "
                f"fraction of one."
            )
    if available_bytes <= 0:
        raise ValueError(
            f"available_bytes must be positive, found {available_bytes}"
        )

    best: tuple[float, int, dict[str, int]] | None = None

    exact = lcm(*cache_sizes.values())
    exact_cost = _geometry_cost(
        available_bytes, exact, cache_sizes, cache_sizes, include_null_block
    )
    if exact_cost is not None:
        best = (exact_cost, exact, dict(cache_sizes))

    ceiling = min(_MAX_HUGE_PAGE_BYTES, available_bytes // 2)
    for huge_page_bytes in _huge_page_candidates(
        cache_sizes, row_sizes, ceiling
    ):
        padded = _fit_huge_page(
            huge_page_bytes, cache_sizes, row_sizes, max_padding_fraction
        )
        if padded is None:
            continue
        cost = _geometry_cost(
            available_bytes,
            huge_page_bytes,
            cache_sizes,
            padded,
            include_null_block,
        )
        # Ties go to the smaller block: same cost, finer allocation quantum.
        if cost is not None and (best is None or cost < best[0]):
            best = (cost, huge_page_bytes, padded)

    if best is None:
        pages = ", ".join(
            f"{cache_id}={to_human_readable_bytes(size)}"
            for cache_id, size in cache_sizes.items()
        )
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to build "
            f"a pool from the page sizes ({pages}), even allowing each page to "
            f"pad by {max_padding_fraction:.0%} to tile a smaller huge block. "
            f"Exact tiling would take "
            f"{to_human_readable_bytes(exact)} per huge block."
        )

    _cost, huge_page_bytes, padded_sizes = best
    return JengaGeometry(
        num_huge_blocks=available_bytes // huge_page_bytes,
        huge_page_bytes=huge_page_bytes,
        ratios={
            cache_id: huge_page_bytes // size
            for cache_id, size in padded_sizes.items()
        },
        padded_sizes=padded_sizes,
    )


class JengaBlockPool:
    """A pool of huge blocks, each subdividable into one cache's little blocks.

    Every cache tiles the same bytes at its own page size, so a huge block is
    ``cache_ratios[cache_id]`` blocks of cache ``cache_id``. Huge block 0 is
    spent on the null block (``N``) that dummy and padding requests share,
    which leaves the rest of it (``.``) unusable, and starts real ids at 1 and
    at ``ratio``::

      huge block     |     0     |     1     |     2     |     3     |
      global  (x4)   | N| .| .| .| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
      sliding (x2)   |  N  |  .  |  2  |  3  |  4  |  5  |  6  |  7  |

    A little block's ``bid`` is thus its index in its own cache's id space.
    ``num_huge_blocks`` counts huge block 0, so a pool needs at least two of
    them to hand anything out.

    Those views alias, so a huge block serves one cache at a time -- its
    ``little_block_type`` -- and at most one row of each column exists. Here
    the global cache holds huge block 1, the sliding cache holds 2, and 3 is
    free for either of them to claim::

      huge block     |     0     |     1     |     2     |     3     |
      global  (x4)   | N| .| .| .| 4| 5| 6| 7| - - - - - | - - - - - |
      sliding (x2)   |  N  |  .  | - - - - - |  4  |  5  | - - - - - |

    Bytes change hands only while nothing references them, so the split
    between caches follows live demand rather than a knob. A huge block is
    therefore always in one of two states::

      parked                                      claimed by cache c
      +-------------------------------+           +--------------------------+
      | ref_cnt == 0                  |   claim   | ref_cnt >= 1             |
      | in free_huge_blocks           |  ------>  | little_block_type == c   |
      | any cache may claim it        |  <------  | only c allocates from it |
      | commits still in prefix cache |   park    |                          |
      +-------------------------------+           +--------------------------+

    Parked means no request holds any of its little blocks, so the bytes are
    up for grabs. Claimed means one cache owns them: each of that cache's
    little blocks in the huge block is either referenced by a request or
    queued in that cache's free list as an eviction candidate, never both and
    never neither.

    ``alloc_block`` claims a parked block when its cache has no free little
    block left, and ``touch`` claims one back when a prefix hit takes its
    reference count up from 0. ``free_block`` parks the block again as soon as
    its last reference goes away, which is what lets another cache reuse the
    bytes.

    Parking keeps commits: a parked block's little blocks stay in their cache's
    prefix cache, so that cache can reclaim it and still hit them. Only another
    cache claiming the bytes evicts them.
    """

    def __init__(
        self, num_huge_blocks: int, cache_ratios: Mapping[str, int]
    ) -> None:
        if num_huge_blocks < 2:
            raise ValueError(
                "num_huge_blocks must be at least 2, since huge block 0 is the "
                f"null block, found {num_huge_blocks}"
            )
        if len(cache_ratios) == 0:
            raise ValueError(
                f"cache_ratios must be non-empty, found {cache_ratios}"
            )
        if any(ratio <= 0 for ratio in cache_ratios.values()):
            raise ValueError(
                f"cache_ratios must be positive, found {cache_ratios}"
            )

        self.cache_ratios = cache_ratios
        self.num_huge_blocks = num_huge_blocks
        # Huge block 0 is the null block, so the allocatable ones start at 1.
        self.huge_blocks: list[HugeKVCacheBlock] = [
            HugeKVCacheBlock(idx) for idx in range(1, num_huge_blocks)
        ]
        self.little_blocks: dict[str, list[LittleKVCacheBlock]] = {
            cache_id: [
                LittleKVCacheBlock(
                    idx, cache_id, self.huge_blocks[idx // ratio - 1]
                )
                for idx in range(ratio, ratio * num_huge_blocks)
            ]
            for cache_id, ratio in cache_ratios.items()
        }
        for bid, huge_block in enumerate(self.huge_blocks):
            huge_block.little_blocks = {
                cache_id: self.little_blocks[cache_id][
                    bid * ratio : bid * ratio + ratio
                ]
                for cache_id, ratio in cache_ratios.items()
            }

        # Every huge block starts untyped, so no cache has any little block in
        # circulation yet.
        self.free_little_blocks = {
            cache_id: FreeLittleKVCacheBlockQueue() for cache_id in cache_ratios
        }
        self.free_huge_blocks = FreeHugeKVCacheBlockQueue(self.huge_blocks)
        self.prefix_caches: dict[str, dict[bytes, LittleKVCacheBlock]] = {
            cache_id: {} for cache_id in cache_ratios
        }
        # How many parked huge blocks are still typed to each cache. Their
        # little blocks are already counted in free_little_blocks, so
        # num_free_blocks must not also count the huge block for that cache.
        self._parked_and_typed: dict[str, int] = {
            cache_id: 0 for cache_id in cache_ratios
        }

        # The block dummy and padding requests point at. Its reference count is
        # pinned so no path can free it, evict it, or hand it out.
        self.null_huge_block = HugeKVCacheBlock(0)
        self.null_little_blocks = {
            cache_id: LittleKVCacheBlock(
                0, cache_id, self.null_huge_block, ref_cnt=42, is_null=True
            )
            for cache_id in cache_ratios
        }
        self.null_huge_block.little_blocks = {
            cache_id: [block]
            for cache_id, block in self.null_little_blocks.items()
        }

    def alloc_block(self, cache_id: str) -> LittleKVCacheBlock:
        """Returns a fresh block of ``cache_id``, claiming huge blocks as needed.

        Raises:
            InsufficientBlocksError: If the pool has no bytes left to serve this
                cache.
        """
        free_little_block_queue = self.free_little_blocks[cache_id]

        # If no free little blocks are available, allocate a huge block and
        # split it into little blocks. Also prefer claiming a huge block even
        # when a little block is available, to avoid evicting a commit.
        if len(free_little_block_queue) == 0 or self._prefer_claim_huge(
            cache_id
        ):
            if len(self.free_huge_blocks) == 0:
                raise InsufficientBlocksError(
                    f"No free blocks available for {cache_id}"
                )
            huge_block = self.free_huge_blocks.popleft()
            # If the huge block is already typed to a different cache, it is
            # parked, so all of that cache's little blocks in it are
            # unreferenced and still sitting on that cache's free list.
            # Remove them there before handing the bytes over.
            old_type = huge_block.little_block_type
            if old_type is not None:
                # A huge block parked under cache_id keeps its little blocks on
                # cache_id's own free list, so we would have popped one of those
                # rather than reaching for a huge block at all.
                assert old_type != cache_id
                # The huge block just left free_huge_blocks, so it stops
                # counting toward old_type's parked-and-typed total.
                self._parked_and_typed[old_type] -= 1
                old_free = self.free_little_blocks[old_type]
                for sibling in huge_block.little_blocks[old_type]:
                    old_free.remove(sibling)
            self._claim_huge_block(huge_block, cache_id)

        little_block = free_little_block_queue.popleft()
        # This may be a little block of a huge block that's still parked (see
        # the class docstring): referencing it directly un-parks the huge
        # block without going through _claim_huge_block, since its little
        # blocks never left this cache's free list in the first place.
        huge_block = little_block.huge_block
        if huge_block in self.free_huge_blocks:
            self.free_huge_blocks.remove(huge_block)
            self._parked_and_typed[cache_id] -= 1
        # Handing out a committed block evicts it: its bytes are about to be
        # overwritten, so it can no longer serve its hash.
        self.uncommit_block(little_block)
        little_block.ref_cnt += 1
        return little_block

    def _prefer_claim_huge(self, cache_id: str) -> bool:
        """Whether claiming a huge block beats evicting a committed little one."""
        free_little_block_queue = self.free_little_blocks[cache_id]
        if len(free_little_block_queue) == 0 or len(self.free_huge_blocks) == 0:
            return False
        little_candidate = free_little_block_queue.peek_front()
        huge_candidate = self.free_huge_blocks.peek_front()
        assert little_candidate is not None
        assert huge_candidate is not None
        return (
            little_candidate.block_hash is not None
            and huge_candidate.little_block_type is None
        )

    def _claim_huge_block(
        self, huge_block: HugeKVCacheBlock, cache_id: str
    ) -> None:
        """Puts an unreferenced huge block's little blocks in circulation.

        Retyping to a different cache hands the bytes over, so whatever the
        outgoing cache had committed in them is evicted first. Reclaiming for
        the same cache keeps those commits, so a prefix hit can still serve
        them.
        """
        assert huge_block.ref_cnt == 0
        prev_type = huge_block.little_block_type
        if huge_block in self.free_huge_blocks:
            self.free_huge_blocks.remove(huge_block)
            if prev_type is not None:
                self._parked_and_typed[prev_type] -= 1

        if prev_type == cache_id:
            # Reclaiming our own parked huge block: its little blocks never
            # left free_little_blocks[cache_id], so there is nothing left to
            # put back in circulation.
            return

        if prev_type is not None:
            for block in huge_block.little_blocks[prev_type]:
                self.uncommit_block(block)

        huge_block.little_block_type = cache_id
        # Reversed so the front-appended pristine blocks come off in
        # ascending bid order, not descending.
        for little_block in reversed(huge_block.little_blocks[cache_id]):
            if little_block.block_hash is None:
                self.free_little_blocks[cache_id].appendleft(little_block)
            else:
                self.free_little_blocks[cache_id].append(little_block)

    def uncommit_block(self, block: LittleKVCacheBlock) -> None:
        """Drops a block from its cache's prefix cache, if it is committed."""
        if block.block_hash is None:
            return
        del self.prefix_caches[block.cache_id][block.block_hash]
        block.block_hash = None

    def commit_into_prefix_cache(
        self, block_hash: bytes, block: LittleKVCacheBlock
    ) -> None:
        """Makes a filled block reusable by anyone hashing the same tokens."""
        assert not block.is_null, "Null blocks should not be committed"
        assert block.block_hash is None
        prefix_cache = self.prefix_caches[block.cache_id]
        assert block_hash not in prefix_cache
        prefix_cache[block_hash] = block
        block.block_hash = block_hash

    def free_block(self, block: LittleKVCacheBlock) -> None:
        """Drops one reference, parking the huge block once it holds none."""
        if block.is_null:
            return

        block.ref_cnt -= 1
        assert block.ref_cnt >= 0
        if block.ref_cnt == 0:
            free_block_queue = self.free_little_blocks[block.cache_id]
            if block.block_hash is None:
                free_block_queue.appendleft(block)
            else:
                free_block_queue.append(block)

            huge_block = block.huge_block
            if huge_block.ref_cnt == 0:
                # The whole huge block went idle, so park it where any cache
                # can claim it. Its little blocks stay on this cache's own
                # free list -- still directly allocable without a reclaim --
                # and the block stays typed: whoever claims it next needs to
                # know whose commits its bytes still back, to evict them only
                # if the bytes change hands.
                assert huge_block.little_block_type == block.cache_id
                self._parked_and_typed[block.cache_id] += 1
                self.free_huge_blocks.append(huge_block)

    def get_or_commit_into_prefix_cache(
        self, block_hash: bytes, block: LittleKVCacheBlock
    ) -> LittleKVCacheBlock | None:
        """Commits a block, or returns the twin already holding its bytes.

        Returns:
            The committed block to use instead of ``block``, which has been
            freed, or ``None`` if ``block`` itself now serves the hash.
        """
        assert not block.is_null, "Null blocks should not be committed"
        prefix_cache = self.prefix_caches[block.cache_id]
        if block_hash in prefix_cache:
            # Check if a block with the same hash is already committed.
            # If so, we reuse the already committed block.
            prefix_cache_block = prefix_cache[block_hash]
            if block.bid == prefix_cache_block.bid:
                return None

            self.touch(prefix_cache_block)

            # Free the block we currently have.
            assert block.block_hash is None
            self.free_block(block)

            return prefix_cache_block

        self.commit_into_prefix_cache(block_hash, block)
        return None

    def touch(self, block: LittleKVCacheBlock) -> None:
        """Takes a reference on a block, reviving it if it was out of use."""
        if block.is_null:
            return

        # Reviving a block whose bytes changed hands would hand back another
        # cache's tensor, so only its own cache's commits are touchable.
        assert block.huge_block.little_block_type == block.cache_id

        # ref_cnt=0 means this block is out of circulation: either an eviction
        # candidate in its cache's free queue, or parked with the rest of an
        # idle huge block, which referencing it takes back for this cache.
        if block.ref_cnt == 0:
            if block.huge_block in self.free_huge_blocks:
                self._claim_huge_block(block.huge_block, block.cache_id)
            self.free_little_blocks[block.cache_id].remove(block)

        block.ref_cnt += 1

    def block(self, cache_id: str, bid: int) -> LittleKVCacheBlock:
        """Returns the little block ``bid`` of ``cache_id``."""
        # Bid 0 is the null block, so a cache's own blocks start at its ratio.
        if bid == 0:
            return self.null_little_blocks[cache_id]
        return self.little_blocks[cache_id][bid - self.cache_ratios[cache_id]]

    def num_free_blocks(self, cache_id: str) -> int:
        """Returns how many more blocks of ``cache_id`` the pool can still serve."""
        # Parked huge blocks already typed to cache_id contribute nothing
        # extra: their little blocks are already counted in
        # free_little_blocks[cache_id], so only the other free huge blocks
        # (pristine, or typed to a different cache) would yield new ones.
        claimable_huge_blocks = (
            len(self.free_huge_blocks) - self._parked_and_typed[cache_id]
        )
        num_little_blocks = len(self.free_little_blocks[cache_id])
        return (
            claimable_huge_blocks * self.cache_ratios[cache_id]
            + num_little_blocks
        )

    def can_satisfy_demand(
        self, demand: dict[str, int], at_capacity: bool = False
    ) -> bool:
        """Returns whether the pool can allocate the demanded number of little blocks.

        ``at_capacity`` asks the same of a pool that has handed nothing out
        yet, making the answer a property of the pool's geometry rather than
        of what it currently holds.
        """
        if at_capacity:
            claimable = len(self.huge_blocks)
            carved: dict[str, int] = dict.fromkeys(demand, 0)
        else:
            carved = {
                cache_id: len(self.free_little_blocks[cache_id])
                for cache_id in demand
            }
            # A parked huge block stays typed only while its cache needs it to
            # satisfy demand. The remaining idle pages may be retyped.
            claimable = len(self.free_huge_blocks)
            for cache_id, num_blocks in demand.items():
                ratio = self.cache_ratios[cache_id]
                parked = self._parked_and_typed[cache_id]
                fixed_free_blocks = carved[cache_id] - parked * ratio
                parked_to_keep = min(
                    parked,
                    ceildiv(
                        max(0, num_blocks - fixed_free_blocks),
                        ratio,
                    ),
                )
                claimable -= parked_to_keep
        # A huge block is carved for exactly one cache, so the demands compete
        # for the same claimable huge blocks: each cache's shortfall is
        # converted at its own ratio and charged against a shared budget.
        # Asking each cache on its own with num_free_blocks would instead let
        # every one of them believe it has room while together they overrun
        # the pool.
        for cache_id, num_blocks in demand.items():
            shortfall = num_blocks - carved[cache_id]
            if shortfall > 0:
                claimable -= ceildiv(shortfall, self.cache_ratios[cache_id])
        return claimable >= 0

    def reset_prefix_cache(self) -> dict[str, int]:
        """Drops every commit no request is holding, in every cache.

        A commit a request still references survives, because its block cannot
        be handed out while it is in use.

        Returns:
            How many blocks were purged from each cache's prefix cache.
        """
        purged: dict[str, int] = {}
        for cache_id, prefix_cache in self.prefix_caches.items():
            unreferenced = [
                block_hash
                for block_hash, block in prefix_cache.items()
                if block.ref_cnt == 0
            ]
            for block_hash in unreferenced:
                prefix_cache.pop(block_hash).block_hash = None
            purged[cache_id] = len(unreferenced)
        return purged
