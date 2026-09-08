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

"""Provides AMD GPU building blocks for structured matrix multiply-accumulate kernels.

Defines thread roles, pipeline-stage shared memory layouts, workgroup and warp
barriers, and the MMA configuration and tile operator used to drive AMD tensor
core matrix multiplication.
"""

from std.sys import align_of, simd_width_of
from max.gpu import WARP_SIZE
from max.gpu.compute.mma import mma
from std.itertools import product
from layout import Layout, LayoutTensor
from layout.layout import blocked_product
from layout.int_tuple import product as prod
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg, TensorCore
from linalg.structuring import SMemTile, RegTile
from std.atomic import Atomic, Ordering
from std.sys._assembly import inlined_assembly
from std.utils import IndexList, StaticTuple

comptime _workgroup_atomic = Atomic[Int32, scope="workgroup"]


trait Enum(TrivialRegisterPassable):
    """Defines a comparable enum-like trait exposing an integer value and equality.
    """

    @always_inline
    def value(self) -> Int:
        ...

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.value() == other.value()

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    def __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline
    def __isnot__(self, other: Self) -> Bool:
        return self != other


@fieldwise_init
struct ThreadRole(Enum, Writable):
    """Represents the role a thread plays in a producer-consumer pipeline.

    Defines whether a thread produces data, consumes data, or performs both
    roles during software-pipelined matrix multiply execution.
    """

    var _value: Int

    @always_inline
    def value(self) -> Int:
        return self._value

    comptime PRODUCER = Self(0)
    comptime CONSUMER = Self(1)
    comptime PRODUCER_CONSUMER = Self(2)

    @always_inline
    def write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))


@__parameter
@always_inline
def pipeline_layout[layout: Layout, pipeline_stages: Int]() -> Layout:
    """Builds a 2D layout extended with a pipeline-stage dimension.

    Combines the given 2D layout with a row-major stage dimension of length
    `pipeline_stages` so that each stage owns a full copy of the tile layout.

    Parameters:
        layout: The base 2D tile layout to replicate per stage.
        pipeline_stages: Number of pipeline stages in the buffer.

    Returns:
        A layout suitable for staging multiple copies of the tile in shared memory.
    """

    comptime assert layout.rank() == 2
    return blocked_product(
        materialize[layout](),
        Layout.row_major(1, pipeline_stages),
        coalesce_output=True,
    )


# TODO: replace with Fabio's implementation
struct SMemBuffer[
    dtype: DType,
    layout: Layout,
    pipeline_stages: Int,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
](TrivialRegisterPassable):

    """Manages shared memory and returns 2D tile slices of the buffer.

    Parameters:
        dtype: Element data type stored in the shared memory buffer.
        layout: 2D layout of a single pipeline stage's tile in shared memory.
        pipeline_stages: Number of pipeline stages buffered in shared memory.
        BM: Block tile height in rows; must equal the row extent of `layout`.
        BN: Block tile width in columns; must equal the column extent of
            `layout`.
        WM: Warp tile height in rows; must evenly divide `BM`.
        WN: Warp tile width in columns; must evenly divide `BN`.
    """

    comptime SMemTile = SMemTile[
        Self.dtype,
        pipeline_layout[Self.layout, Self.pipeline_stages](),
        alignment=128,
    ]

    comptime BlockTileType = Self.SMemTile.TileType[Self.BM, Self.BN]
    comptime WarpTileType = Self.BlockTileType.TileType[Self.WM, Self.WN]

    @__allow_legacy_any_origin_fields
    var buffer: Self.SMemTile

    @always_inline
    def __init__(out self):
        comptime assert Self.layout.rank() == 2, "layout must be 2D"

        comptime assert (
            prod(Self.layout.shape[0]) == Self.BM
            and prod(Self.layout.shape[1]) == Self.BN
        ), "shared memory rows must match block_rows and columns must match BN"

        comptime assert (
            Self.BM % Self.WM == 0 and Self.BN % Self.WN == 0
        ), "BM and BN must be a multiple of WM and WN"

        self.buffer = Self.SMemTile.stack_allocation()

    @always_inline
    def get_tile(self, stage: Int) -> Self.BlockTileType:
        return self.buffer.tile[Self.BM, Self.BN](0, stage)


struct AMDSharedMemoryBarrier(TrivialRegisterPassable):
    """Implements a workgroup-level shared memory barrier for AMD GPUs.

    Tracks an atomic counter in shared memory that producers increment and
    consumers wait on to coordinate software-pipelined tile loads.
    """

    var __repr: Int32

    @always_inline
    def initialize[origin: MutOrigin](ref[AddressSpace.SHARED, origin] self):
        self.__repr = 0

    @always_inline
    def value[
        origin: MutOrigin
    ](ref[AddressSpace.SHARED, origin] self) -> Int32:
        var bar = UnsafePointer(to=self.__repr).address_space_cast[.SHARED]()
        return _workgroup_atomic.load[ordering=Ordering.ACQUIRE](bar)

    @always_inline
    def increment[
        origin: MutOrigin
    ](ref[AddressSpace.SHARED, origin] self, warp_id: Int):
        var bar = UnsafePointer(to=self.__repr).address_space_cast[.SHARED]()
        _workgroup_atomic.store[ordering=Ordering.RELEASE](
            bar, _workgroup_atomic.load[ordering=Ordering.ACQUIRE](bar) + 1
        )

    @always_inline
    def wait_until_greater_or_equal_to[
        origin: MutOrigin
    ](ref[AddressSpace.SHARED, origin] self, v: Int32):
        while self.value() < v:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()


struct AMDWarpSharedMemoryBarrier[size: Int](TrivialRegisterPassable):
    """Implements a per-warp shared memory barrier for AMD GPUs.

    Maintains a per-warp counter tuple so that each warp can independently track
    progress, and consumers wait on the summed value across all warps.

    Parameters:
        size: Number of warps tracked by the barrier.
    """

    var __repr: StaticTuple[Int32, Self.size]

    @always_inline
    def initialize(ref[AddressSpace.SHARED, MutAnyOrigin] self):
        self.__repr = StaticTuple[Int32, Self.size](fill=0)

    @always_inline
    def value(ref[AddressSpace.SHARED] self) -> Int32:
        var sum: Int32 = 0

        comptime for i in range(Self.size):
            sum += self.__repr[i]
        return sum

    @always_inline
    def increment(ref[AddressSpace.SHARED, MutAnyOrigin] self, warp_id: Int):
        var bar = rebind[
            UnsafePointer[Int32, MutAnyOrigin, address_space=.SHARED]
        ](Pointer(to=self.__repr))
        bar[warp_id] += 1

    @always_inline
    def wait_until_greater_or_equal_to(ref[AddressSpace.SHARED] self, v: Int32):
        while self.value() < v:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()


struct MMAConfig[
    InType: DType,
    OutType: DType,
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
](TrivialRegisterPassable):
    """Configures matrix multiply-accumulate parameters for AMD tensor cores.

    Computes the register and K-group sizing derived from the MMA shape and
    input type so that callers can query the adjusted K dimension used per
    tensor core load for operands A and B.

    Parameters:
        InType: Data type of the input operands.
        OutType: Data type of the accumulator output.
        mma_shape: The [M, N, K] shape of a single MMA operation.
        transpose_b: Whether matrix B is loaded transposed.
    """

    comptime mma = TensorCore[
        Self.OutType,
        Self.InType,
        Self.mma_shape,
        Self.transpose_b,
    ]()

    comptime simd_width = simd_width_of[Self.InType]()
    comptime registers_per_thread_a = num_matrix_reg[
        Self.mma_shape[0], Self.mma_shape[2]
    ]()
    comptime registers_per_thread_b = num_matrix_reg[
        Self.mma_shape[1], Self.mma_shape[2]
    ]()

    comptime k_group_size_a = Self.simd_width // Self.registers_per_thread_a
    comptime k_group_size_b = Self.simd_width // Self.registers_per_thread_b

    @staticmethod
    @always_inline
    def adjusted_mma_k_shape_a() -> Int:
        return Self.mma_shape[2] * Self.k_group_size_a

    @staticmethod
    @always_inline
    def adjusted_mma_k_shape_b() -> Int:
        return Self.mma_shape[2] * Self.k_group_size_b


struct AmdTileOperator[
    InType: DType,
    OutType: DType,
    warp_block_layout_a: Layout,
    warp_block_layout_b: Layout,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = None,
    transpose_b: Bool = True,
](TrivialRegisterPassable):
    """Manages tensor core operations for matrix multiplication on AMD GPUs.

    This operator handles loading matrix fragments from shared memory to registers
    and performing matrix multiply-accumulate operations using tensor cores.

    Parameters:
        InType: Input data type.
        OutType: Output data type.
        warp_block_layout_a: Layout for matrix A warp tiles.
        warp_block_layout_b: Layout for matrix B warp tiles.
        mma_shape: Shape of the MMA operation [M, N, K].
        swizzle: Optional swizzle pattern for memory access.
        transpose_b: Whether matrix B is transposed.

    Requirements:
        - warp_block_layout_a.shape[0] must be divisible by mma_shape[0]
        - warp_block_layout_b.shape[0] must be divisible by mma_shape[1]
        - warp_block_layout_a.shape[1] must be divisible by mma_shape[2]
        - warp_block_layout_b.shape[1] must be divisible by mma_shape[2]
        - The K dimension must align such that num_k_tiles is divisible by k_group_size
    """

    comptime simd_width = simd_width_of[Self.InType]()
    comptime _type_alignment = align_of[SIMD[Self.InType, Self.simd_width]]()

    # Create tensor core instance
    comptime tensor_core = TensorCore[
        Self.OutType,
        Self.InType,
        Self.mma_shape,
        Self.transpose_b,
    ]()

    comptime num_m_mmas = prod(
        Self.warp_block_layout_a.shape[0]
    ) // Self.mma_shape[0]
    comptime num_n_mmas = prod(
        Self.warp_block_layout_b.shape[0]
    ) // Self.mma_shape[1]

    comptime _out_frag_rows = Self.num_m_mmas * Self.num_n_mmas
    comptime _out_frag_cols = Self.tensor_core.c_reg_type.length

    comptime _out_layout = Layout.row_major(
        Self._out_frag_rows, Self._out_frag_cols
    )

    comptime WK = prod(Self.warp_block_layout_a.shape[1])
    comptime num_k_tiles = Self.WK // Self.mma_shape[2]

    comptime _registers_per_thread_a = num_matrix_reg[
        Self.mma_shape[0], Self.mma_shape[2]
    ]()
    comptime _registers_per_thread_b = num_matrix_reg[
        Self.mma_shape[1], Self.mma_shape[2]
    ]()
    comptime k_group_size_a = Self.simd_width // Self._registers_per_thread_a
    comptime k_group_size_b = Self.simd_width // Self._registers_per_thread_b

    comptime _k_tiles_per_simd_a = Self.num_k_tiles // Self.k_group_size_a
    comptime _k_tiles_per_simd_b = Self.num_k_tiles // Self.k_group_size_b

    # Total number of K tiles for MMA operations
    comptime total_k_tiles = Self.num_k_tiles
    comptime out_frag_size = Self.mma_shape[0] * Self.mma_shape[1] // WARP_SIZE

    comptime _in_layout[
        num_mmas: Int,
        _k_tiles_per_simd: Int,
    ] = Layout.row_major(_k_tiles_per_simd * num_mmas, Self.simd_width)

    comptime ARegTile = RegTile[
        Self.InType, Self._in_layout[Self.num_m_mmas, Self._k_tiles_per_simd_a]
    ]

    comptime BRegTile = RegTile[
        Self.InType, Self._in_layout[Self.num_n_mmas, Self._k_tiles_per_simd_b]
    ]

    comptime OutRegTile = LayoutTensor[
        Self.OutType,
        Self._out_layout,
        MutAnyOrigin,
        alignment=Self._type_alignment,
        address_space=.LOCAL,
    ]

    comptime OutRegTileFragmentType = Self.OutRegTile.TileType[
        Self._out_frag_rows, Self._out_frag_cols
    ]

    # Register storage for matrix data
    @__allow_legacy_any_origin_fields
    var _a_reg_tile: Self.ARegTile

    @__allow_legacy_any_origin_fields
    var _b_reg_tile: Self.BRegTile

    @__allow_legacy_any_origin_fields
    var out_reg_tile: Self.OutRegTile

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.simd_width >= Self._registers_per_thread_a
            and Self.simd_width >= Self._registers_per_thread_b
        ), (
            "simd_width must be greater than or equal to required mma"
            " fragments size"
        )

        comptime assert (
            Self.num_k_tiles % Self.k_group_size_a == 0
        ), "num_k_tiles must be divisible by k_group_size"

        comptime assert (
            Self._k_tiles_per_simd_a == Self._k_tiles_per_simd_b
        ), "k_tiles_per_simd must be equal for A and B"

        self._a_reg_tile = Self.ARegTile.stack_allocation()
        self._b_reg_tile = Self.BRegTile.stack_allocation()

        # Initialize output accumulator to zero
        self.out_reg_tile = Self.OutRegTile.stack_allocation().fill(0)

    @always_inline
    def a_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.ARegTile.TileType[Self.num_m_mmas, Self.simd_width]:
        """Get A register tile for a specific K tile.

        Args:
            k_tile_idx: Index along the K dimension selecting which K-tile
                slice of the A register buffer to return.
        """
        return self._a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
            k_tile_idx, 0
        )

    @always_inline
    def b_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.BRegTile.TileType[Self.num_n_mmas, Self.simd_width]:
        """Get B register tile for a specific K tile.

        Args:
            k_tile_idx: Index along the K dimension selecting which K-tile
                slice of the B register buffer to return.
        """
        return self._b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
            k_tile_idx, 0
        )

    @always_inline
    def reset_accumulator(self):
        """Reset the accumulator to zero for a new tile computation."""
        _ = self.out_reg_tile.fill(0)

    # Helper aliases for K-tile indexing
    comptime k_tile_group_index[
        k_tile_idx: Int
    ] = k_tile_idx // Self.k_group_size_a

    comptime k_tile_fragment_index[
        k_tile_idx: Int
    ] = k_tile_idx % Self.k_group_size_a

    @always_inline
    def load_tile_fragment[
        k_tile_idx: Int
    ](self, smem_tile_a: LayoutTensor, smem_tile_b: LayoutTensor):
        """Load fragments from shared memory to registers for a specific K tile.

        Parameters:
            k_tile_idx: K-tile index (0 to total_k_tiles-1).

        Args:
            smem_tile_a: Shared memory tile for matrix A.
            smem_tile_b: Shared memory tile for matrix B.
        """
        comptime group_idx = Self.k_tile_group_index[k_tile_idx]
        comptime fragment_idx = Self.k_tile_fragment_index[k_tile_idx]

        # Only load if this is the first fragment in the group
        # (tensor core loads k_group_size tiles at once)
        comptime if fragment_idx == 0:
            Self.tensor_core.load_a[swizzle=Self.swizzle](
                smem_tile_a,
                self._a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
                    group_idx, 0
                ).vectorize[1, Self.simd_width](),
                group_idx,
            )

            Self.tensor_core.load_b[swizzle=Self.swizzle](
                smem_tile_b,
                self._b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
                    group_idx, 0
                ).vectorize[1, Self.simd_width](),
                group_idx,
            )

    @always_inline
    def mma_compute[k_tile_idx: Int](self):
        """Perform matrix multiply-accumulate for a specific K tile.

        This method assumes fragments are already loaded via load_tile_fragment.

        Parameters:
            k_tile_idx: K-tile index (0 to total_k_tiles-1).
        """
        comptime group_idx = Self.k_tile_group_index[k_tile_idx]
        comptime fragment_idx = Self.k_tile_fragment_index[k_tile_idx]

        var c_slice = self.out_reg_tile

        # Get the tiles for this group
        var a_tile = self.a_reg_tile(group_idx)
        var b_tile = self.b_reg_tile(group_idx)

        # Perform MMA for this specific fragment within the group
        comptime for mma_m_idx in range(Self.num_m_mmas):
            var a_fragment = a_tile.tile[1, Self._registers_per_thread_a](
                mma_m_idx, fragment_idx
            )

            comptime for mma_n_idx in range(Self.num_n_mmas):
                var b_fragment = b_tile.tile[1, Self._registers_per_thread_b](
                    mma_n_idx, fragment_idx
                )

                # Storage scheme is column major for efficient write-back
                var c_fragment = c_slice.tile[1, Self._registers_per_thread_a](
                    mma_n_idx * Self.num_m_mmas + mma_m_idx, 0
                )

                mma(
                    c_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                    b_fragment.vectorize[1, Self._registers_per_thread_b]()[
                        0, 0
                    ],
                    a_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                    c_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                )
