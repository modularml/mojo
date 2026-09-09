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

"""Provides the persistent tile scheduler for GPU matmul kernels.

Maps thread blocks to output tiles across the launch grid using 1D, 2D-wave,
or DeepSeek scheduling strategies, and exposes the `TileScheduler`,
`WorkInfo`, `MatmulSchedule`, and `RasterOrder` types used by persistent
matmul kernels.
"""

from std.math import ceildiv
from std.math.uutils import udivmod

from max.gpu import block_idx, grid_dim

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList

from ...utils_gpu import block_swizzle


@fieldwise_init
struct RasterOrder(Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Rasterization order for mapping thread blocks to output tiles.

    Controls the traversal direction of the 2D output tile grid. `AlongN`
    rasterizes across columns first; `AlongM` rasterizes across rows first.
    The choice affects L2 cache reuse patterns and should match cluster
    multicast configuration.
    """

    var _value: Int32

    comptime AlongN = Self(0)
    comptime AlongM = Self(1)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self._value == 0:
            writer.write("rasterN")
        else:
            writer.write("rasterM")


@fieldwise_init
struct WorkInfo(TrivialRegisterPassable, Writable):
    """Descriptor for one output tile assigned to a thread block.

    Carries the (m, n) tile coordinates in the output matrix, the starting
    K-tile index for split-K kernels, the number of K tiles to process, and
    a validity flag indicating whether the tile falls within the problem bounds.
    """

    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    var num_k_tiles: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    comptime INVALID_WORK_INFO = Self(0, 0, 0, 0, False)

    @always_inline
    def __init__(
        out self,
    ):
        self.m = 0
        self.n = 0
        self.k_start = 0
        self.num_k_tiles = 0
        self.is_valid_tile = False

    @always_inline
    def is_valid(self) -> Bool:
        return self.is_valid_tile

    @always_inline
    def is_final_split(self, k_tiles_per_output_tile: UInt32) -> Bool:
        return (self.k_start + self.num_k_tiles) == k_tiles_per_output_tile

    @always_inline
    def get_k_start(self) -> UInt32:
        return self.k_start

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.k_start,
            ", ",
            self.num_k_tiles,
            ", ",
            self.is_valid_tile,
            ")",
        )


@fieldwise_init
struct MatmulSchedule(TrivialRegisterPassable):
    """Tile scheduling strategy for GPU matmul kernels.

    Selects how thread blocks are mapped to output tiles across the grid.
    `TILE1D` uses a simple linear mapping; `TILE2D` adds wave-level 2D
    swizzling for better L2 locality; `DS_SCHEDULER` uses the DeepSeek
    persistent block scheduler; `NONE` disables the scheduler (non-persistent
    kernels).
    """

    var _value: Int32

    comptime NONE = Self(0)
    comptime TILE1D = Self(1)
    comptime TILE2D = Self(2)
    comptime DS_SCHEDULER = Self(3)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


struct TileScheduler[
    problem_shape: IndexList[3],
    tile_shape: IndexList[3],
    grid_shape: IndexList[2],
    cluster: IndexList[3] = Index(1, 1, 1),
    raster_dim: UInt32 = 1,
    schedule: MatmulSchedule = MatmulSchedule.TILE2D,
](TrivialRegisterPassable):
    """Persistent tile scheduler for GPU matmul output tiles.

    Maps thread blocks to (m, n) output tile coordinates using 1D, 2D-wave,
    or DeepSeek scheduling strategies. Supports block clusters with multicast
    and advances through tiles in a persistent kernel loop via `fetch_next_work`.

    Parameters:
        problem_shape: Static (M, N, K) dimensions of the matmul problem.
        tile_shape: Static (BM, BN, BK) tile sizes.
        grid_shape: (grid_x, grid_y) number of thread blocks in the launch grid.
        cluster: Block cluster shape (default 1×1×1 for no clustering).
        raster_dim: Axis along which to rasterize (0 = M, 1 = N).
        schedule: Tile scheduling strategy to use.
    """

    # grid_shape[0], [1] map to x, y, to N and M in output matrix.
    # tile_shape[0], [1] map to M and N
    # wave_shape[0], [1] map to M and N
    comptime wave_shape = Index[dtype=DType.uint32](
        Self.tile_shape[0] * Self.grid_shape[1],
        Self.tile_shape[1] * Self.grid_shape[0],
    )
    # This has to match the grid dimension for the kernel launch.
    comptime num_grids: UInt32 = UInt32(Self.grid_shape[0] * Self.grid_shape[1])
    var idx: UInt32
    var prob_shape: IndexList[3]  # M x N x K
    var num_waves_m: UInt32
    var num_waves_n: UInt32
    var log_num_waves_n: FastDiv[.uint32]

    # Member variables for DeepSeek Scheduler
    var current_iter: Int  # Tracks the scheduler's progress across kernel launches
    var num_aligned_m_blocks: UInt32  # Number of blocks needed for the M dimension
    var num_blocks: UInt32  # Total number of blocks for non-masked types

    comptime kNum1DBlocksPerGroup: UInt32 = 16
    comptime kNumNBlocks: UInt32 = UInt32(
        ceildiv(Self.problem_shape[1], Self.tile_shape[1])
    )

    @always_inline
    def __init__(out self, prob_shape: IndexList[3]):
        comptime if Self.schedule == MatmulSchedule.TILE2D:
            comptime assert _check_cluster(
                Self.cluster, Self.raster_dim
            ), "Only support block cluster in along raster dimension."

        if Self.schedule == MatmulSchedule.DS_SCHEDULER:
            comptime assert (
                Self.cluster[0] == Self.cluster[1] == Self.cluster[2] == 1
            ), "Currently multicasting is not supported for DeepSeek Scheduler"

        self.prob_shape = prob_shape
        self.num_waves_m = UInt32(
            ceildiv(self.prob_shape[0], Self.wave_shape[0])
        )
        self.num_waves_n = UInt32(
            ceildiv(self.prob_shape[1], Self.wave_shape[1])
        )
        self.log_num_waves_n = FastDiv[.uint32](Int(self.num_waves_n))

        self.current_iter = -1
        self.num_aligned_m_blocks = UInt32(
            ceildiv(prob_shape[0], Self.tile_shape[0])
        )
        self.num_blocks = self.num_aligned_m_blocks * Self.kNumNBlocks

        comptime if Self.raster_dim == 0:  # rasterize along M
            self.idx = UInt32(block_idx.x) * UInt32(grid_dim.y) + UInt32(
                block_idx.y
            )
        else:
            self.idx = UInt32(block_idx.x) + UInt32(grid_dim.x) * UInt32(
                block_idx.y
            )

    @always_inline
    def get_current_work_info(mut self) -> WorkInfo:
        comptime if Self.schedule == MatmulSchedule.DS_SCHEDULER:
            var m_block_idx: UInt32 = 0
            var n_block_idx: UInt32 = 0
            var is_valid = self._get_next_block(m_block_idx, n_block_idx)
            var m: UInt32 = m_block_idx * UInt32(Self.tile_shape[0])
            var n: UInt32 = n_block_idx * UInt32(Self.tile_shape[1])

            return WorkInfo(
                m,
                n,
                0,
                UInt32(ceildiv(Self.problem_shape[2], Self.tile_shape[2])),
                is_valid,
            )
        else:
            var m, n = self._index_to_mn()
            var is_valid = m < self.prob_shape[0] and n < self.prob_shape[1]
            return WorkInfo(
                UInt32(m),
                UInt32(n),
                0,
                UInt32(ceildiv(self.prob_shape[2], Self.tile_shape[2])),
                is_valid,
            )

    @always_inline
    def advance(mut self):
        self.idx += Self.num_grids

    @always_inline
    def fetch_next_work(mut self) -> WorkInfo:
        comptime if Self.schedule == MatmulSchedule.DS_SCHEDULER:
            return self.fetch_next_work_ds()
        else:
            self.advance()
            return self.get_current_work_info()

    @always_inline
    def _index_to_mn(self) -> Tuple[Int, Int]:
        """Map the thread block's index to coordinates of work tile."""

        comptime if Self.schedule == MatmulSchedule.TILE2D:
            return self._index_to_mn_tile2d()

        return self._index_to_mn_tile1d()

    @always_inline
    def _index_to_mn_tile1d(self) -> Tuple[Int, Int]:
        # Grid dim as if there is no persist kernel
        var logical_grid_dim = Index[dtype=DType.uint32](
            ceildiv(self.prob_shape[1], Self.tile_shape[1]),
            ceildiv(self.prob_shape[0], Self.tile_shape[0]),
        )

        var by, bx = udivmod(Int(self.idx), logical_grid_dim[0])
        var block_xy_swizzle = block_swizzle(
            Index[dtype=DType.uint32](bx, by), logical_grid_dim
        )

        var m: Int = block_xy_swizzle[1] * Self.tile_shape[0]
        var n: Int = block_xy_swizzle[0] * Self.tile_shape[1]

        return (m, n)

    @always_inline
    def _index_to_mn_tile2d(self) -> Tuple[Int, Int]:
        # We consider a sweep on busy SMs a wave, not all SMs
        comptime log_num_grids = FastDiv[.uint32](Int(Self.num_grids))
        comptime log_grid_shape = FastDiv[.uint32](Self.grid_shape[0])

        comptime FastUInt = Scalar[FastDiv[.uint32].uint_type]

        var num_waves_executed = FastUInt(self.idx) / log_num_grids
        var idx_in_wave = FastUInt(self.idx) % log_num_grids

        var num_waves_executed_m = (
            FastUInt(num_waves_executed) / self.log_num_waves_n
        )
        var num_waves_executed_n = (
            FastUInt(num_waves_executed) % self.log_num_waves_n
        )

        # The wave maps to a BM x grid_shape[1] by BN x grid_shape[0]
        # submatrix in C.
        var wave_m = num_waves_executed_m * FastUInt(Self.wave_shape[0])
        var wave_n = num_waves_executed_n * FastUInt(Self.wave_shape[1])

        var m_in_wave = FastUInt(idx_in_wave) / log_grid_shape
        var n_in_wave = FastUInt(idx_in_wave) % log_grid_shape

        return (
            Int(wave_m + m_in_wave * FastUInt(Self.tile_shape[0])),
            Int(wave_n + n_in_wave * FastUInt(Self.tile_shape[1])),
        )

    @always_inline
    def num_output_tiles(self) -> Int:
        return ceildiv(self.prob_shape[0], Self.wave_shape[0]) * ceildiv(
            self.prob_shape[1], Self.wave_shape[1]
        )

    @always_inline
    def fetch_next_work_ds(mut self) -> WorkInfo:
        var m_block_idx: UInt32 = 0
        var n_block_idx: UInt32 = 0
        var is_valid = self._get_next_block(m_block_idx, n_block_idx)

        var m: UInt32 = m_block_idx * UInt32(Self.tile_shape[0])
        var n: UInt32 = n_block_idx * UInt32(Self.tile_shape[1])
        # Only support K starting from 0 for now.
        return WorkInfo(
            m,
            n,
            0,
            UInt32(ceildiv(Self.problem_shape[2], Self.tile_shape[2])),
            is_valid,
        )

    # Calculates swizzled M and N block indices for better cache utilization
    @always_inline
    def _get_swizzled_block_idx(
        self, num_m_blocks: UInt32, block_idx: Int
    ) -> Tuple[UInt32, UInt32]:
        """
        Calculates swizzled (m_block_idx, n_block_idx) based on the overall block_idx.
        The swizzling pattern depends on kIsTMAMulticastOnA.
        Returns a tuple (m_block_idx, n_block_idx).
        """

        var m_block_idx: UInt32
        var n_block_idx: UInt32

        # Swizzle for better L2 usages
        var primary_num_blocks = num_m_blocks
        comptime secondary_num_blocks = Self.kNumNBlocks
        comptime num_blocks_per_group = secondary_num_blocks * Self.kNum1DBlocksPerGroup
        var group_idx = UInt32(block_idx) / num_blocks_per_group
        var first_block_idx = group_idx * Self.kNum1DBlocksPerGroup
        var in_group_idx = UInt32(block_idx) % num_blocks_per_group
        var num_blocks_in_group = min(
            Self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx
        )

        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
        n_block_idx = in_group_idx / num_blocks_in_group

        return (m_block_idx, n_block_idx)

    # Gets the next (m_block_idx, n_block_idx) pair for the current thread block to process
    @always_inline
    def _get_next_block(
        mut self, mut m_block_idx: UInt32, mut n_block_idx: UInt32
    ) -> Bool:
        """
        Calculates and returns the next (m_block_idx, n_block_idx) pair for the
        calling thread block. Returns None if no more blocks are available.
        This implements the core logic of the persistent block scheduler.
        """

        self.current_iter += 1
        var next_block_idx = self.current_iter * grid_dim.x + block_idx.x

        # Check if the calculated index exceeds the total number of blocks
        if next_block_idx >= Int(self.num_blocks):
            return False  # No more work

        # Get swizzled indices based on the total number of aligned M blocks
        m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            self.num_aligned_m_blocks, next_block_idx
        )
        return True


def _check_cluster(cluster_dims: IndexList[3], raster_dim: UInt32) -> Bool:
    """Check if block cluster is along the raster dimension."""

    comptime for i in range(3):
        if cluster_dims[i] > 1 and i != Int(raster_dim):
            return False

    return True
