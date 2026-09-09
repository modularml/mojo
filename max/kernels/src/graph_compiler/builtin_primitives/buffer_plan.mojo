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

from std.collections import BitSet, Array
from std.math import align_up, ceildiv


def _compute_unshareable[
    N: Int,
](can_share: Array[Int, N * N], out result: BitSet[N]):
    """Compute which allocations cannot share memory with any other allocation.

    An allocation i is unshareable if no other allocation j has a
    non-overlapping lifetime with it. Two allocations can share memory when
    can_share[i * N + j] == 1, meaning their lifetimes do not overlap.
    """
    result = {}
    result.set_all()

    for i in range(N):
        for j in range(N):
            if i != j and can_share[i * N + j]:
                result.clear(i)
                break


def _compute_shareable_rows[
    N: Int,
](can_share: Array[Int, N * N], out result: Array[BitSet[N], N]):
    """Compute per-allocation sharing bitsets from the sharing matrix.

    result[i].test(j) == True iff can_share[i * N + j] == 1, i.e. allocations
    i and j have non-overlapping lifetimes and may share a memory block.
    """

    @always_inline
    def result_init(i: Int) {imm can_share} -> BitSet[N]:
        var row: BitSet[N] = {}
        for j in range(N):
            if can_share[i * N + j]:
                row.set(j)
        return row^

    result = Array[_, N](fill_with=result_init)


def _maximum(alignments: Array[Int, _]) -> Int:
    """Return the maximum value in an Array of Ints."""
    var result = 0
    for align in alignments:
        result = max(result, align)
    return result


struct BufferPlanStats(ImplicitlyCopyable, Writable):
    """Statistics produced by the buffer planning algorithm.

    Holds the key metrics describing how effectively the planner used memory,
    and implements `Writable` so it can be passed directly to a `Logger` or
    any other writer.
    """

    var num_allocs: Int
    """Total number of allocations planned."""
    var num_reused: Int
    """Number of allocations that reused an existing memory block."""
    var watermark: Int
    """High watermark: total bytes in the memory pool (final pool size)."""
    var requested: Int
    """Total bytes requested across all individual allocations."""

    def __init__(
        out self,
        num_allocs: Int,
        num_reused: Int,
        watermark: Int,
        requested: Int,
    ):
        self.num_allocs = num_allocs
        self.num_reused = num_reused
        self.watermark = watermark
        self.requested = requested

    def write_to(self, mut writer: Some[Writer]):
        var reuse_pct = Float64(self.num_reused) / Float64(
            self.num_allocs
        ) * 100 if self.num_allocs > 0 else Float64(0)
        var memory_saved = self.requested - self.watermark
        var savings_pct = Float64(memory_saved) / Float64(
            self.requested
        ) * 100 if self.requested > 0 else Float64(0)
        t"""mgp.buffer.plan stats:
  Total Allocations: {self.num_allocs}
  Reused Allocations: {self.num_reused} ({reuse_pct}%)
  Total Allocation Size: {self.requested} bytes
  Final Pool Size: {self.watermark} bytes
  Memory Saved: {memory_saved} bytes ({savings_pct}%)""".write_to(
            writer
        )


struct MemoryBlock[N: Int](Copyable, Movable):
    """Memory block used by buffer planning algorithm."""

    var offset: Int
    var size: Int
    # Set of allocation indices that can still be assigned to this block.
    # An allocation j is in this set iff j can share memory with every
    # allocation already assigned to this block.
    var shareable: BitSet[Self.N]

    def __init__(
        out self, offset: Int, size: Int, var shareable: BitSet[Self.N]
    ):
        self.offset = offset
        self.size = size
        self.shareable = shareable^


struct BufferPlanState[
    num_allocs: Int,
    //,
    alignments: Array[Int, num_allocs],
    can_share: Array[Int, num_allocs * num_allocs],
](Movable):
    # Computed at comptime: which allocations interfere with all others and
    # therefore can never share a memory block with any other allocation.
    comptime unshareable = _compute_unshareable[Self.num_allocs](Self.can_share)
    # Per-allocation sharing bitsets: shareable_rows[i].test(j) == True iff
    # allocations i and j can share a memory block.
    comptime shareable_rows = _compute_shareable_rows[Self.num_allocs](
        Self.can_share
    )
    # If every allocation is unshareable, skip the greedy block-reuse logic
    # entirely and use the fast linear-allocation path.
    comptime enable_sharing = len(Self.unshareable) < Self.num_allocs
    comptime max_alignment = _maximum(Self.alignments)

    var blocks: List[MemoryBlock[Self.num_allocs]]
    var allocated: Int

    # Computed allocation offsets for all allocations.
    var offsets: Array[Int, Self.num_allocs]
    var pool_size: Int
    # Sum of all allocation sizes passed to allocate_greedy.
    var requested: Int
    # Number of allocations that reused an existing memory block.
    var num_reused: Int

    # The set of per-allocation shareable bitsets.
    var shareable_sets: Array[BitSet[Self.num_allocs], Self.num_allocs]

    def __init__(
        out self,
    ):
        comptime if Self.enable_sharing:
            self.blocks = List[MemoryBlock[Self.num_allocs]](
                capacity=Self.num_allocs
            )
            self.shareable_sets = materialize[Self.shareable_rows]()
        else:
            self.blocks = {}
            self.shareable_sets = {uninitialized = True}

        self.allocated = 0
        self.pool_size = 0
        self.requested = 0
        self.num_reused = 0
        self.offsets = Array[Int, Self.num_allocs](fill=0)

    @always_inline
    def take_results(
        deinit self,
    ) -> Tuple[Int, Array[Int, Self.num_allocs]]:
        assert self.allocated == Self.num_allocs
        return self.pool_size, self.offsets.copy()

    @always_inline
    def stats(self) -> BufferPlanStats:
        """Returns a lightweight snapshot of planning statistics for logging."""
        return BufferPlanStats(
            Self.num_allocs, self.num_reused, self.pool_size, self.requested
        )

    def find_block(
        self,
        index: Int,
        alloc_size: Int,
        out result: Int,
    ):
        result = -1

        comptime if not Self.enable_sharing:
            return

        var best_size = Int.MAX

        for block_idx in range(len(self.blocks)):
            if (
                alloc_size > self.blocks[block_idx].size
                or self.blocks[block_idx].size >= best_size
            ):
                continue

            if self.blocks[block_idx].shareable.test(index):
                best_size = self.blocks[block_idx].size
                result = block_idx

    @always_inline
    def append_result(mut self, index: Int, value: Int):
        self.offsets[index] = value
        self.allocated += 1

    @always_inline
    def shareable_set(self, index: Int) -> BitSet[Self.num_allocs]:
        assert Self.enable_sharing, "unable to get shareable set"
        return self.shareable_sets[index].copy()

    def allocate_new_block(mut self, index: Int, alloc_size: Int):
        var new_offset = align_up(self.pool_size, self.max_alignment)

        comptime if Self.enable_sharing:
            self.blocks.append(
                MemoryBlock[Self.num_allocs](
                    offset=new_offset,
                    size=alloc_size,
                    shareable=self.shareable_set(index),
                )
            )

        self.append_result(index, new_offset)
        self.pool_size = new_offset + alloc_size

    def try_reuse_block(mut self, result_idx: Int, alloc_size: Int):
        var best_block_idx = self.find_block(result_idx, alloc_size)
        if best_block_idx >= 0:
            # Intersect the block's shareable set with the precomputed row for
            # result_idx. Since the diagonal is zero, result_idx is
            # automatically cleared (it is now a member, not a future
            # candidate).
            self.blocks[best_block_idx].shareable = self.blocks[
                best_block_idx
            ].shareable.intersection(self.shareable_set(result_idx))
            self.append_result(result_idx, self.blocks[best_block_idx].offset)
            self.num_reused += 1
        else:
            self.allocate_new_block(result_idx, alloc_size)

    @always_inline
    def allocate_greedy[start: Int = 0](mut self, sizes: Array[Int, _]):
        comptime if not Self.enable_sharing:
            # No allocations can be shared; skip the greedy search entirely.
            for i, size in enumerate(sizes):
                self.requested += size
                self.allocate_new_block(i + start, size)
        else:
            comptime for i in range(sizes.length):
                var alloc_size = sizes[i]
                comptime result_idx = i + start
                self.requested += alloc_size

                comptime if Self.unshareable.test(result_idx):
                    # This allocation cannot share with any other; skip search.
                    self.allocate_new_block(result_idx, alloc_size)
                else:
                    self.try_reuse_block(result_idx, alloc_size)
