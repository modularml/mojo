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

"""Provides tile schedulers for multi-head attention kernels on NVIDIA GPUs.

Defines the `MHATileScheduler` trait and concrete schedulers (`TileScheduler`,
`TransientScheduler`, `QueuedTileScheduler`) that map work tiles to thread
blocks, along with supporting state and summary types used by the persistent
attention kernel.
"""

from std.collections import OptionalReg

from std.atomic import Atomic

import max.gpu.primitives.warp as warp
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host.info import H100
from max.gpu import block_idx, thread_idx
from max.gpu.primitives.id import cluster_dim
from max.gpu.sync import barrier, named_barrier
from nn.attention.gpu.nvidia.common import NullPointer, OptionalPointer

from std.builtin.device_passable import DevicePassable


@fieldwise_init
struct WorkInfo(TrivialRegisterPassable, Writable):
    """Holds the coordinates and validity of a single work tile.

    Each work tile is identified by its offset within the prompt, head index,
    and batch index, plus a flag indicating whether the tile is in bounds.
    """

    # (query_offset, head_idx, sequence idx in batch)
    var prompt_offset: UInt32
    var head_idx: UInt32
    var prompt_idx: UInt32
    # Currently each work tile travser entire cache length.
    # TODO: Add starting kv index in cache len dim
    # var kv_start: UInt32 = 0
    # var kv_end: UInt32 = 0
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    def is_valid(self) -> Bool:
        return self.is_valid_tile

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.prompt_offset,
            ", ",
            self.head_idx,
            ", ",
            self.prompt_idx,
            ", ",
            self.is_valid_tile,
            ")",
        )


struct SeqInfo(TrivialRegisterPassable):
    """Describes a sequence's length and starting offset for a work tile.

    Carries the sequence length, start-of-sequence offset, and the originating
    `WorkInfo` coordinates used by the attention kernel to index into the KV
    cache.
    """

    var seq_len: UInt32
    var start_of_seq: UInt32
    var prompt_offset: UInt32
    var head_idx: UInt32
    var prompt_idx: UInt32

    @always_inline
    def __init__(
        out self, seq_len: UInt32, start_of_seq: UInt32, work: WorkInfo
    ):
        self.seq_len = seq_len
        self.start_of_seq = start_of_seq
        self.prompt_offset = work.prompt_offset
        self.head_idx = work.head_idx
        self.prompt_idx = work.prompt_idx

    @always_inline
    def is_valid(self) -> Bool:
        return self.seq_len > self.prompt_offset

    @staticmethod
    @always_inline
    def create[
        ValidLengthType: OptionalPointer,
        //,
    ](
        work: WorkInfo,
        valid_length: ValidLengthType,
        max_seq_len: UInt32,
    ) -> SeqInfo:
        var batch_idx: UInt32 = work.prompt_idx

        comptime if not ValidLengthType.is_null:
            # treat valid_lengths as a input_row_offsets
            var ptr = rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](
                valid_length.value()
            )
            var seq = ptr.load[width=2](batch_idx)
            var start_of_seq = warp.broadcast(seq[0])
            var end_of_seq = warp.broadcast(seq[1])
            var seq_len = end_of_seq - start_of_seq
            return SeqInfo(seq_len, start_of_seq, work)
        else:
            var seq_len = max_seq_len
            return SeqInfo(seq_len, 0, work)


@fieldwise_init
struct MHASchedulerSynchronization(TrivialRegisterPassable):
    """Enumerates synchronization modes for advancing the MHA scheduler.

    Controls which threads participate in the barrier when advancing to the
    next work tile: `NONE` for TMA-only paths, `PRODUCER` for copy-async paths,
    and `ALL` when every thread must synchronize.
    """

    var _value: Int32

    comptime NONE = Self(0)  # use for TMA
    comptime PRODUCER = Self(1)  # use for copy-async
    comptime ALL = Self(2)  # use when all threads are synced
    comptime DEFAULT = Self.PRODUCER  # default is currently copy-async

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# This class is constructed within the fully inlined kernel,
# so unneeded fields can be optimized away.
struct MHATileState(TrivialRegisterPassable):
    """Tracks the mutable per-CTA state of the tile scheduler during kernel execution.

    Holds the current linear work-tile index, a shared-memory pointer used to
    communicate the active index across threads, and the maximum valid index
    bounding the work grid.
    """

    # Linear work tile index i.e. idx-th work among all possible workload.
    var idx: UInt32

    @__allow_legacy_any_origin_fields
    var sidx_ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED]
    var max_idx: UInt32

    @always_inline
    def __init__(
        out self,
        idx: UInt32,
        sidx_ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED],
        max_idx: UInt32,
    ):
        self.idx = idx
        self.sidx_ptr = sidx_ptr
        self.max_idx = max_idx

    @always_inline
    def is_valid(self, idx: UInt32) -> Bool:
        return idx < self.max_idx

    @always_inline
    def is_valid(self) -> Bool:
        return self.is_valid(self.idx)


struct MHATileSummary[ValidLengthType: OptionalPointer](
    TrivialRegisterPassable
):
    """Summarizes the dimensions and valid-length metadata of the MHA work grid.

    Stores the batch size, maximum number of prompt tiles, optional per-batch
    sequence length offsets, and the maximum sequence length, providing the
    mapping from linear work-tile indices to `(prompt_tile, head, batch)`
    coordinates.

    Parameters:
        ValidLengthType: The optional pointer type carrying per-batch
            sequence length offsets (inferred).
    """

    # Number of sequences in batch.
    var batch_size: UInt32
    # Maximum num tiles.
    var max_num_prompt_tiles: UInt32
    var valid_length: Self.ValidLengthType
    var max_seq_len: UInt32

    @always_inline
    def __init__(
        out self,
        batch_size: UInt32,
        max_num_prompt_tiles: UInt32,
        valid_length: Self.ValidLengthType,
        max_seq_len: UInt32,
    ):
        self.batch_size = batch_size
        self.max_num_prompt_tiles = max_num_prompt_tiles
        self.valid_length = valid_length
        self.max_seq_len = max_seq_len

    @always_inline
    def _index_to_coords[
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        """Map the thread block's index to coordinates of work tile."""

        comptime if schedule == MHASchedule.PROMPT_ROTATE:
            return self._index_to_coords_prompt_rotate[num_heads](idx)

        return self._index_to_coords_default[num_heads](idx)

    @always_inline
    def _index_to_coords_default[
        num_heads: UInt32
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        # First dim, offset in prompt length
        #
        # The goal is to keep kv in l2 cache.
        # As kv is constant with prompt_tile_idx,
        # this changes fastest.
        # kv changes for every `group` iterations of
        # head-idx, and for every iteration of `prompt-idx`.
        # Thus, we have head_idx vary fastest.
        #
        # self.idx's max-value = self.max_num_prompt_tiles*num_heads*batch_size
        var quotient, prompt_tile_idx = divmod(idx, self.max_num_prompt_tiles)
        # max value = num_heads-1
        # head index
        # changes kv whenever head_idx//group changes
        # max value = batch_size-1
        # prompt index
        # changes kv
        var prompt_idx, head_idx = divmod(quotient, num_heads)

        return (prompt_tile_idx, head_idx, prompt_idx)

    @always_inline
    def _index_to_coords_prompt_rotate[
        num_heads: UInt32
    ](self, idx: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
        # First dim, offset in prompt length
        var quotient, prompt_tile_idx = divmod(idx, self.max_num_prompt_tiles)
        # head index
        # prompt index
        var prompt_idx, head_idx = divmod(quotient, num_heads)
        # Switch the traverse direction in prompt for odd head.
        prompt_tile_idx = (
            prompt_tile_idx if head_idx % 2
            == 0 else self.max_num_prompt_tiles - 1 - prompt_tile_idx
        )

        return (prompt_tile_idx, head_idx, prompt_idx)

    @always_inline
    def get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> WorkInfo:
        var prompt_tile_idx, head_idx, prompt_idx = self._index_to_coords[
            num_heads, schedule
        ](idx)
        var is_valid = (
            prompt_tile_idx < self.max_num_prompt_tiles
            and head_idx < num_heads
            and prompt_idx < self.batch_size
        )

        return WorkInfo(
            prompt_tile_idx * tile_shape,
            head_idx,
            prompt_idx,
            is_valid,
        )

    @always_inline
    def unsafe_get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> WorkInfo:
        var prompt_tile_idx, head_idx, prompt_idx = self._index_to_coords[
            num_heads, schedule
        ](idx)

        assert prompt_tile_idx < self.max_num_prompt_tiles
        assert head_idx < num_heads
        assert prompt_idx < self.batch_size

        return WorkInfo(
            prompt_tile_idx * tile_shape,
            head_idx,
            prompt_idx,
            True,
        )

    @always_inline
    def max_idx(self, num_heads: UInt32) -> UInt32:
        return self.max_num_prompt_tiles * self.batch_size * num_heads

    @always_inline
    def get_current_work_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: MHATileState) -> WorkInfo:
        return self.get_current_work_info[tile_shape, num_heads, schedule](
            idx.idx
        )

    @staticmethod
    @always_inline
    def grid_dim[
        num_heads: UInt32
    ](max_num_prompt_tiles: UInt32, batch_size: UInt32) -> Tuple[Int, Int, Int]:
        return (Int(max_num_prompt_tiles), Int(num_heads), Int(batch_size))

    @always_inline
    def seq_info(self, work: WorkInfo) -> SeqInfo:
        return SeqInfo.create(work, self.valid_length, self.max_seq_len)

    @always_inline
    def unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, idx: UInt32) -> SeqInfo:
        var work = self.unsafe_get_current_work_info[
            tile_shape, num_heads, schedule
        ](idx)
        return SeqInfo.create(work, self.valid_length, self.max_seq_len)

    @always_inline
    def unsafe_seq_info[
        tile_shape: UInt32,
        num_heads: UInt32,
        schedule: MHASchedule,
    ](self, state: MHATileState) -> SeqInfo:
        return self.unsafe_seq_info[tile_shape, num_heads, schedule](state.idx)


trait MHATileScheduler(Copyable, DevicePassable, TrivialRegisterPassable):
    """Describes a schedule for the persistent MHA kernel.

    A tile scheduler maps work tiles to thread blocks, advances the per-CTA
    state through the work grid across kernel iterations, and reports the grid
    dimensions required for launch.
    """

    comptime may_advance: Bool
    comptime mha_schedule: MHASchedule

    def get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        """Returns the current `WorkInfo`.

        Parameters:
            ValidLengthType: The optional pointer type carrying per-batch
                sequence length offsets (inferred).

        Args:
            ts: The tile summary describing the work grid.
            state: The per-CTA scheduler state whose current index to
                resolve.
        """
        ...

    @always_inline
    def advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        """Advance state to the next work item.

        `func` must return a `Bool` indicating whether there is more work.
        Returns `True` if there is more work.

        Parameters:
            ValidLengthType: The optional pointer type carrying per-batch
                sequence length offsets (inferred).
            producer: Whether the calling CTA is the producer thread for
                copy-async paths.
            sync: Which threads participate in the barrier when advancing
                (defaults to `MHASchedulerSynchronization.DEFAULT`).

        Args:
            ts: The tile summary describing the work grid.
            state: The mutable per-CTA scheduler state to advance.
            pipeline_idx: The pipeline stage index for storing the shared
                work index.
        """
        ...

    @staticmethod
    @always_inline
    def grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        """Return the grid_dim required for the kernel.

        Args:
            batch_size: Number of sequences in the batch.
            max_num_prompt_tiles: Maximum number of prompt tiles along the
                sequence dimension.
        """
        ...

    @always_inline
    def initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        """Create the initial state object.

        Parameters:
            ValidLengthType: The optional pointer type carrying per-batch
                sequence length offsets (inferred).

        Args:
            ptr: Shared-memory pointer for communicating the active work
                index across threads.
            tile_summary: The tile summary describing the work grid
                dimensions.
        """
        ...

    @always_inline
    def unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        ...


@fieldwise_init
struct MHASchedule(TrivialRegisterPassable):
    """Enumerates the scheduling strategy for mapping work tiles to thread blocks.

    `DEFAULT` orders tiles to maximize KV-cache locality; `PROMPT_ROTATE`
    reverses the prompt-tile traversal direction for odd-numbered heads to
    spread L2 cache pressure.
    """

    var _value: Int32

    comptime DEFAULT = Self(0)
    comptime PROMPT_ROTATE = Self(1)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


struct TransientScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    flip_prompt_idx: Bool,
    pair_cta: Bool = False,
    splitk_partitions: UInt32 = 1,
](Defaultable, MHATileScheduler, TrivialRegisterPassable):
    """Implements a non-persistent (transient) tile scheduler for the MHA kernel.

    Each CTA processes exactly one work tile identified by its `block_idx`,
    with optional prompt-index flipping, pair-CTA clustering, or split-K
    partitioning to widen the launch grid.

    Parameters:
        tile_shape: Size of each query tile along the sequence dimension,
            in tokens.
        num_heads: Number of attention heads mapped across the grid's `y`
            dimension.
        flip_prompt_idx: Whether to reverse the prompt-tile traversal
            direction so that the last tile is processed first.
        pair_cta: Whether to cluster two CTAs per query tile for a 2-SM
            launch width (defaults to `False`).
        splitk_partitions: Number of split-K partitions used to widen the
            launch grid along `x` (defaults to `1`).
    """

    comptime may_advance: Bool = False
    comptime mha_schedule: MHASchedule = MHASchedule.DEFAULT

    # CTAs per launch cluster: the pair-CTA 2-SM width (1 or 2) times the
    # num_q==1 split-K partition count. Each cluster owns one Q-tile, so the
    # grid is widened by this factor and `block_idx.x` is divided by it to
    # recover the tile index. `pair_cta` and split-K are mutually exclusive,
    # so this is `2` for pair-CTA, `P` for split-K, and `1` otherwise.
    comptime cluster_size: UInt32 = (
        UInt32(2) if Self.pair_cta else UInt32(1)
    ) * Self.splitk_partitions

    comptime device_type: AnyType = Self

    # Runtime split-K partition count for the WORKSPACE (traditional/unfused)
    # split-K path, which keeps `splitk_partitions == 1` at comptime (no launch
    # cluster) but over-launches grid.x by a runtime `P`. Defaults to 1, so for
    # every non-workspace config (non-split, pair-CTA, cluster split-K) the tile
    # divisor and flip below fold back to their prior comptime values —
    # byte-identical. `get_seq_info` sets it from `partition.num_partitions()`.
    var num_partitions: UInt32

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return (
            "TransientScheduler[tile_shape = "
            + String(Self.tile_shape)
            + ", num_heads = "
            + String(Self.num_heads)
            + ", flip_prompt_idx = "
            + String(Self.flip_prompt_idx)
            + ", pair_cta = "
            + String(Self.pair_cta)
            + ", splitk_partitions = "
            + String(Self.splitk_partitions)
            + "]"
        )

    @always_inline
    def __init__(out self):
        self.num_partitions = 1

    @always_inline
    def __init__(out self, num_partitions: UInt32):
        self.num_partitions = num_partitions

    @always_inline
    def get_current_work_info(self, num_prompt_tiles: UInt32) -> WorkInfo:
        # Each cluster of `cluster_size` CTAs owns one Q-tile; recover the tile
        # index by dividing out the cluster. For the dynamic split-K path the
        # cluster size is chosen at LAUNCH (one compiled kernel covers
        # P in {2,4,8}), so divide by the runtime `cluster_dim.x` (a one-time
        # u32 divide per CTA, off the hot path). Pair-CTA / non-split keep the
        # comptime `cluster_size`, which folds to a shift (`>> 1` for pair-CTA,
        # identity at 1) — zero blast radius for every non-split-K config.
        var raw_idx: UInt32
        comptime if Self.splitk_partitions > 1 and not Self.pair_cta:
            raw_idx = UInt32(block_idx.x) // UInt32(cluster_dim.x)
        else:
            # `cluster_size` folds to a shift (pair-CTA `>> 1`, identity at 1);
            # `num_partitions` is 1 except on the WORKSPACE split-K path, where it
            # is the runtime `P` that over-launched grid.x. Dividing by it maps
            # `block_idx.x` -> a VALID prompt tile in `[0, real_tiles)` (the
            # partition index is recovered separately as `block_idx.x % P` in the
            # warps). For every non-workspace config `num_partitions == 1`, so this
            # is byte-identical to the prior `// cluster_size`.
            raw_idx = UInt32(block_idx.x) // (
                Self.cluster_size * self.num_partitions
            )
        # Un-inflate the prompt-tile count for the flip: on the workspace path the
        # scheduler tile space was multiplied by `num_partitions`, so the real
        # per-partition tile count is `num_prompt_tiles // num_partitions` (== the
        # original count for every other config, leaving the flip unchanged).
        var real_num_prompt_tiles: UInt32 = (
            num_prompt_tiles // self.num_partitions
        )
        var prompt_tile_idx: UInt32
        comptime if Self.flip_prompt_idx:
            prompt_tile_idx = real_num_prompt_tiles - 1 - raw_idx
        else:
            prompt_tile_idx = raw_idx
        return WorkInfo(
            prompt_tile_idx * Self.tile_shape,
            UInt32(block_idx.y),
            UInt32(block_idx.z),
            True,
        )

    @always_inline
    def get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return self.get_current_work_info(ts.max_num_prompt_tiles)

    @always_inline
    def advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        return None

    @staticmethod
    @always_inline
    def grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # One cluster of `cluster_size` CTAs per Q-tile, so widen x by the
        # cluster size (×2 for pair-CTA, ×P for split-K, ×1 otherwise). This is
        # the `MHATileScheduler`-trait signature; the dynamic split-K launch
        # uses the 3-arg overload below to widen by the RUNTIME partition count.
        return (
            Int(max_num_prompt_tiles * Self.cluster_size),
            Int(Self.num_heads),
            Int(batch_size),
        )

    @staticmethod
    @always_inline
    def grid_dim(
        batch_size: UInt32,
        max_num_prompt_tiles: UInt32,
        num_partitions: UInt32,
    ) -> Tuple[Int, Int, Int]:
        # Dynamic split-K (Stage B): the cluster size is chosen at LAUNCH (one
        # compiled kernel covers P in {2,4,8}), so widen x by the RUNTIME
        # `num_partitions` rather than the comptime `cluster_size` (= P_MAX
        # here). This MUST agree with the device tile divisor (`block_idx.x //
        # cluster_dim.x`). Pair-CTA / non-split fall back to the comptime
        # `cluster_size`, so this overload is correct to call unconditionally.
        var x_width: UInt32
        comptime if Self.splitk_partitions > 1 and not Self.pair_cta:
            x_width = max_num_prompt_tiles * num_partitions
        else:
            x_width = max_num_prompt_tiles * Self.cluster_size
        return (
            Int(x_width),
            Int(Self.num_heads),
            Int(batch_size),
        )

    @always_inline
    def initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        return MHATileState(0, ptr, 1)

    @always_inline
    def unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return SeqInfo.create(
            self.get_current_work_info(ts.max_num_prompt_tiles),
            ts.valid_length,
            ts.max_seq_len,
        )


struct TileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    num_ctas: UInt32 = UInt32(H100.sm_count),
    schedule: MHASchedule = MHASchedule.DEFAULT,
](Defaultable, MHATileScheduler, TrivialRegisterPassable):
    """Implements a persistent tile scheduler that cycles CTAs through work tiles.

    Each CTA begins at its `block_idx` and strides by `num_ctas` on every
    advance, reusing the same SMs across multiple work tiles to amortize launch
    overhead and improve cache locality.

    Parameters:
        tile_shape: Size of each query tile along the sequence dimension,
            in tokens.
        num_heads: Number of attention heads mapped across the grid's `y`
            dimension.
        num_ctas: Number of CTAs to launch for the persistent kernel, where
            each CTA strides by this count between tiles (defaults to the
            H100 SM count).
        schedule: Strategy for mapping work tiles to thread blocks
            (defaults to `MHASchedule.DEFAULT`).
    """

    comptime may_advance: Bool = True
    comptime mha_schedule: MHASchedule = Self.schedule

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return (
            "TileScheduler[tile_shape = "
            + String(Self.tile_shape)
            + ", num_heads = "
            + String(Self.num_heads)
            + ", num_ctas = "
            + String(Self.num_ctas)
            + ", schedule = "
            + String(Self.schedule._value)
            + "]"
        )

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @always_inline
    def fetch_next_work(
        self,
        ts: MHATileSummary,
        mut state: MHATileState,
    ) -> WorkInfo:
        state.idx += Self.num_ctas
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    @always_inline
    def advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        state.idx += Self.num_ctas
        if not state.is_valid(state.idx):
            return None
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    @staticmethod
    @always_inline
    def grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        var bx, by, bz = MHATileSummary[NullPointer[.uint32]].grid_dim[
            Self.num_heads
        ](max_num_prompt_tiles, batch_size)
        var size = min(Int(Self.num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    def initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        return MHATileState(
            UInt32(block_idx.x), ptr, tile_summary.max_idx(Self.num_heads)
        )

    @always_inline
    def unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)


struct QueuedTileScheduler[
    tile_shape: UInt32,
    num_heads: UInt32,
    /,
    decoding: Bool,
    num_ctas: UInt32 = UInt32(H100.sm_count),
    schedule: MHASchedule = MHASchedule.DEFAULT,
](DevicePassable, MHATileScheduler, TrivialRegisterPassable):
    """
    If `decoding == False`, then `num_heads` is `q_num_heads`.
    If `decoding == True`, then `num_heads` is `kv_num_heads`.

    Parameters:
        tile_shape: Size of each query tile along the sequence dimension.
        num_heads: Number of attention heads (`q_num_heads` when not
            decoding, `kv_num_heads` when decoding).
        decoding: Whether the kernel is in the decoding phase.
        num_ctas: Number of CTAs to launch (defaults to the H100 SM
            count).
        schedule: Strategy for mapping work tiles to thread blocks
            (defaults to `MHASchedule.DEFAULT`).
    """

    # Linear work tile index i.e. idx-th work among all possible workload.
    @__allow_legacy_any_origin_fields
    var gidx_ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.GLOBAL]

    comptime may_advance: Bool = True
    comptime mha_schedule: MHASchedule = Self.schedule

    @always_inline
    def __init__(
        out self,
        gidx_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    ):
        self.gidx_ptr = gidx_ptr.address_space_cast[.GLOBAL]()

    @always_inline
    def get_current_work_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> WorkInfo:
        return ts.get_current_work_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @always_inline
    def advance[
        ValidLengthType: OptionalPointer,
        //,
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](
        self,
        ts: MHATileSummary[ValidLengthType],
        mut state: MHATileState,
        pipeline_idx: UInt32,
    ) -> OptionalReg[SeqInfo]:
        """The parameter `func` must return a `Bool` indicating whether the `WorkInfo` arg is valid.
        This function returns whether the current idx corresponds to a valid `WorkInfo`.
        Note that if `MHASchedulerSynchronization` is `NONE`, then we assume it is only called by `thread_idx.x==0`.

        Parameters:
            ValidLengthType: The optional pointer type carrying per-batch
                sequence length offsets (inferred).
            producer: Whether the calling CTA is the producer thread for
                copy-async paths.
            sync: Which threads participate in the barrier when advancing
                (defaults to `MHASchedulerSynchronization.DEFAULT`).

        Args:
            ts: The tile summary describing the work grid.
            state: The mutable per-CTA scheduler state to advance.
            pipeline_idx: The pipeline stage index for storing the shared
                work index.
        """

        comptime if producer:
            if thread_idx.x == 0:
                var idx: UInt32
                while True:
                    idx = Atomic.fetch_add(self.gidx_ptr, 1)
                    if not state.is_valid(idx):
                        comptime if sync == MHASchedulerSynchronization.NONE:
                            state.idx = idx
                            state.sidx_ptr.store(offset=pipeline_idx, val=idx)
                            return None

                        else:
                            break
                    var seq_info: SeqInfo = ts.unsafe_seq_info[
                        Self.tile_shape, Self.num_heads, Self.schedule
                    ](idx)

                    comptime if not Self.decoding:
                        if seq_info.is_valid():
                            comptime if sync == MHASchedulerSynchronization.NONE:
                                state.idx = idx
                                state.sidx_ptr.store(
                                    offset=pipeline_idx, val=idx
                                )
                                # tma with producer doesn't need to sync
                                return seq_info
                            else:
                                break

                state.sidx_ptr.store(offset=pipeline_idx, val=idx)

            # producer needs to sync before loading
            comptime if sync == MHASchedulerSynchronization.PRODUCER:
                named_barrier[128,](id=1)

        comptime if sync == MHASchedulerSynchronization.ALL:
            barrier()

        # when !ALL, consumers rely on `async_copy_arrive`
        state.idx = warp.broadcast(state.sidx_ptr.load(pipeline_idx))
        if not state.is_valid():
            return None
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state)

    @staticmethod
    @always_inline
    def grid_dim(
        batch_size: UInt32, max_num_prompt_tiles: UInt32
    ) -> Tuple[Int, Int, Int]:
        # NOTE: mha_sm90 assumes `grid_dim` limits the grid
        # size for persistent kernels, so that it doesn't
        # need to check the first `work_info` for validity.
        var bx, by, bz = MHATileSummary[NullPointer[.uint32]].grid_dim[
            Self.num_heads
        ](max_num_prompt_tiles, batch_size)
        var size = min(Int(Self.num_ctas), bx * by * bz)
        return (size, 1, 1)

    @always_inline
    def initial_state[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self,
        ptr: UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED],
        tile_summary: MHATileSummary[ValidLengthType],
    ) -> MHATileState:
        var state = MHATileState(
            UInt32(block_idx.x), ptr, tile_summary.max_idx(Self.num_heads)
        )

        if thread_idx.x == 0:
            state.sidx_ptr.store(state.idx)
        return state

    @always_inline
    def unsafe_seq_info[
        ValidLengthType: OptionalPointer,
        //,
    ](
        self, ts: MHATileSummary[ValidLengthType], state: MHATileState
    ) -> SeqInfo:
        return ts.unsafe_seq_info[
            Self.tile_shape, Self.num_heads, Self.schedule
        ](state.idx)

    # `trait DevicePassable` implementation
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: The device specific type encoder.
            target: The target address to store the device type.
        """
        encoder.encode(self, target)

    @no_inline
    @staticmethod
    def get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "QueuedTileScheduler[tile_shape = ",
            String(Self.tile_shape),
            ", num_heads = ",
            String(Self.num_heads),
            ", decoding = ",
            String(Self.decoding),
            ", num_ctas = ",
            String(Self.num_ctas),
            ", schedule = ",
            String(Self.schedule._value),
            "]",
        )
