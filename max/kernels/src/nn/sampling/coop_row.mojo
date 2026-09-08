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
"""Provides cross-block reductions for row-parallel sampling.

Blocks exchange values through global memory and fold them in rank order so
every block can safely branch on the same result.
"""

from std.atomic import Atomic, Ordering, fence
from max.gpu import thread_idx
from max.gpu.primitives.warp import _ReduceFn
from std.os import abort
from std.sys.info import has_amd_gpu_accelerator, is_amd_gpu
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.sync import barrier


@always_inline
def _device_scope() -> StaticString:
    comptime if is_amd_gpu():
        return "agent"
    else:
        return "device"


comptime _DEVICE_SCOPE = _device_scope()
comptime _coop_atomic = Atomic[Int32, scope=_DEVICE_SCOPE]

# Eight lanes fit the six cutoff statistics in one aligned store.
comptime COOP_SLOT_FLOATS = 8

# This padding aligns each row's publish slots to 32 bytes.
comptime _COOP_CONTROL_WORDS = 8

# This bound turns a missing peer into an error instead of a deadlock.
comptime _COOP_SPIN_LIMIT = 1 << 24

# Larger groups spend more time synchronizing than they save on memory work.
comptime _MAX_COOP_GROUP = 16


def coop_row_words(batch_size: Int, group_size: Int) -> Int:
    """Returns the workspace size in `Int32` words.

    Args:
        batch_size: Number of rows in the launch.
        group_size: Blocks cooperating on each row.

    Returns:
        The number of words to allocate. The launcher must initialize them to
        zero before each launch.
    """
    return batch_size * (
        _COOP_CONTROL_WORDS + 2 * group_size * COOP_SLOT_FLOATS
    )


def coop_group_size(
    ctx: DeviceContext, batch_size: Int, block_size: Int, n_vec: Int
) raises -> Int:
    """Chooses a resident AMD block group for each row.

    Every block must be resident because the barrier spins until all blocks
    arrive. The launch therefore uses at most one block per compute unit. The
    timeout catches contention that prevents a group from becoming resident.

    Args:
        ctx: Device the launch targets.
        batch_size: Number of rows in the launch.
        block_size: Threads per block.
        n_vec: Vectors in one row.

    Returns:
        Blocks per row, or `1` when the row should not be split.
    """
    comptime if not has_amd_gpu_accelerator():
        return 1
    if batch_size <= 0:
        return 1
    var cu_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var group_size = 1
    var c = 2
    while c <= _MAX_COOP_GROUP:
        if batch_size * c <= cu_count and n_vec >= c * block_size:
            group_size = c
        c *= 2
    return group_size


struct CoopRow[group_size: Int](TrivialRegisterPassable):
    """Tracks one block's position and phase within a row group."""

    var rank: Int
    var _row: Int
    var _round: Int
    var _phase: Int

    @always_inline
    def __init__(out self, row: Int, rank: Int):
        self.rank = rank
        self._row = row
        self._round = 0
        self._phase = 0

    @always_inline
    def _counter(
        self, workspace: UnsafePointer[Int32, MutAnyOrigin]
    ) -> UnsafePointer[Int32, MutAnyOrigin]:
        return workspace + self._row * (
            _COOP_CONTROL_WORDS + 2 * Self.group_size * COOP_SLOT_FLOATS
        )

    @always_inline
    def _slots(
        self, workspace: UnsafePointer[Int32, MutAnyOrigin]
    ) -> UnsafePointer[Float32, MutAnyOrigin]:
        return (self._counter(workspace) + _COOP_CONTROL_WORDS).bitcast[
            Float32
        ]()

    @always_inline
    def _arrive_and_wait(
        mut self, workspace: UnsafePointer[Int32, MutAnyOrigin]
    ):
        """Synchronizes the group with device-scoped release and acquire fences.

        Thread 0 updates the counter. Block barriers make the acquired peer
        data visible to the remaining threads. A timeout aborts before any
        block can read stale data.
        """
        self._round += 1
        var target = Int32(self._round * Self.group_size)
        var counter = self._counter(workspace)
        barrier()
        if thread_idx.x == 0:
            fence[ordering=Ordering.RELEASE, scope=_DEVICE_SCOPE]()
            var seen = (
                _coop_atomic.fetch_add[ordering=Ordering.RELAXED](
                    counter, Int32(1)
                )
                + 1
            )
            var spins = 0
            while seen < target and spins < _COOP_SPIN_LIMIT:
                seen = _coop_atomic.load[ordering=Ordering.RELAXED](counter)
                spins += 1
            if seen < target:
                debug_assert(
                    False,
                    "coop_row barrier timed out; blocks may not be co-resident",
                )
                abort()
            fence[ordering=Ordering.ACQUIRE, scope=_DEVICE_SCOPE]()
        barrier()

    @always_inline
    def sync(mut self, workspace: UnsafePointer[Int32, MutAnyOrigin]):
        """Makes each block's preceding global writes visible to its peers."""
        self._arrive_and_wait(workspace)

    @always_inline
    def gather[
        width: Int
    ](
        mut self,
        workspace: UnsafePointer[Int32, MutAnyOrigin],
        table: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
        vals: SIMD[.float32, width],
    ):
        """Publishes `vals` and copies each rank's slot into shared memory.

        Calls alternate between two slot sets so a faster block cannot
        overwrite data that its peers are still reading.
        `table[r * width + j]` receives lane `j` from rank `r`.
        """
        comptime assert (
            width <= COOP_SLOT_FLOATS
        ), "a publish cannot exceed one slot"
        var tx = Int(thread_idx.x)
        var slots = self._slots(workspace)
        var slot_base = self._phase * Self.group_size * COOP_SLOT_FLOATS
        self._phase ^= 1

        if tx == 0:
            slots.store(slot_base + self.rank * COOP_SLOT_FLOATS, vals)

        self._arrive_and_wait(workspace)

        if tx < Self.group_size:
            table.store(
                tx * width,
                slots.load[width=width](slot_base + tx * COOP_SLOT_FLOATS),
            )
        barrier()

    @always_inline
    def combine[
        width: Int, combine_fn: _ReduceFn
    ](
        mut self,
        workspace: UnsafePointer[Int32, MutAnyOrigin],
        table: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
        vals: SIMD[.float32, width],
    ) -> SIMD[.float32, width]:
        """Combines block-reduced vectors in rank order.

        Every block returns bit-identical values and may safely branch on the
        result.
        """
        self.gather[width](workspace, table, vals)
        var acc = table.load[width=width](0)
        comptime for r in range(1, Self.group_size):
            acc = combine_fn(acc, table.load[width=width](r * width))
        return acc
