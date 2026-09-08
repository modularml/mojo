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

"""TileWriter module for efficient tile writing in GPU matrix multiplication.

This module provides utilities for writing tiles to memory using different
mechanisms and destinations:

1. Register → Shared Memory: Uses st.matrix hardware instruction for efficient
   storage of WGMMA accumulator results to shared memory with swizzling.

2. Register → Global Memory: Direct stores from register tiles to global memory
   with optional epilogue processing and bounds checking.

3. Shared Memory → Global Memory: Hardware-accelerated TMA stores or regular
   stores for efficient 2D tile transfers from shared to global memory.

Two main traits abstract these writing mechanisms:
- TileWriter: For shared memory → global memory transfers
- RegTileWriter: For register → memory (shared or global) transfers
"""

from layout.tma_async import TMATensorTile
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    MixedLayout,
    DefaultEngine,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tensor_engine import TensorEngine
from layout.tile_io import copy_sram_to_dram
from max.gpu.memory import fence_async_view_proxy
from std.collections import OptionalReg
from ....structuring import RegTile
from structured_kernels.smem_types import reg_tile_to_tile_tensor
from layout.swizzle import Swizzle
from max.gpu import lane_id
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE

from max.gpu.compute.mma import st_matrix
from std.memory import bitcast
from layout.tensor_core_async import st_matrix_n_layout, st_matrix_m_layout
from ....utils import elementwise_epilogue_type, elementwise_compute_lambda_type
from std.utils.index import IndexList
from std.sys import align_of, size_of
from layout.tile_io import copy_local_to_dram
import std.itertools
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from std.bit import log2_floor
from std.math.uutils import ufloordiv


# Import ThreadInfo from matmul_output
struct ThreadInfo(TrivialRegisterPassable):
    """Thread identification within the warp group."""

    var warp_id: Int
    var lane_id: Int
    var lane_row: UInt32
    var lane_col: UInt32

    def __init__(
        out self,
        warp_id: Int,
        lane_id: Int,
        lane_row: UInt32,
        lane_col: UInt32,
    ):
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.lane_row = lane_row
        self.lane_col = lane_col

    @always_inline
    @staticmethod
    def from_warp_group_idx(warp_group_thread_idx: Int) -> ThreadInfo:
        """Create ThreadInfo from a warp group thread index.

        Args:
            warp_group_thread_idx: Thread index within the warp group.

        Returns:
            ThreadInfo struct with computed warp_id, lane_id, lane_row, and lane_col.
        """
        var warp_id = ufloordiv(warp_group_thread_idx, WARP_SIZE)
        var lid = lane_id()
        var lane_row, lane_col = divmod(UInt32(lid), 4)
        return ThreadInfo(warp_id, lid, lane_row, lane_col)


struct TileCoordinates(TrivialRegisterPassable):
    """Helper struct for managing tile coordinate offsets.

    This struct encapsulates corner and split coordinates used in epilogue
    processing and provides a clean interface for coordinate transformations.
    """

    var corner: IndexList[2]
    var split: IndexList[2]

    @always_inline
    def __init__(out self, corner: IndexList[2], split: IndexList[2]):
        """Initialize tile coordinates.

        Args:
            corner: Corner coordinates offset.
            split: Split coordinates offset.
        """
        self.corner = corner
        self.split = split

    @always_inline
    def adjust(self, base_coords: IndexList[2]) -> IndexList[2]:
        """Add corner and split offsets to base coordinates.

        Args:
            base_coords: Base tile coordinates.

        Returns:
            Adjusted coordinates with corner and split offsets applied.
        """
        return IndexList[2](
            base_coords[0] + self.corner[0] + self.split[0],
            base_coords[1] + self.corner[1] + self.split[1],
        )


trait SMemTileWriter(TrivialRegisterPassable):
    """Base trait for tile writing mechanisms in matrix multiplication.

    This trait defines the interface for writing tiles from shared memory to global memory,
    abstracting over different hardware mechanisms.
    """

    comptime _dtype: DType

    @always_inline
    def write_tile(
        self,
        src: TileTensor[mut=True, Self._dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """Write a tile from shared memory to global memory.

        Args:
            src: Source tile in shared memory (must be 128-byte aligned).
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        ...


struct TileWriterTMA[
    tma_origin: ImmOrigin,
    dtype: DType,
    tma_rank: Int,
    tile_shape: IndexList[tma_rank],
    desc_shape: IndexList[tma_rank],
    //,
](SMemTileWriter, TrivialRegisterPassable):
    """TMA-based tile writer for hardware-accelerated memory transfers.

    This writer uses NVIDIA's Tensor Memory Accelerator (TMA) for efficient
    2D tile transfers from shared to global memory.

    Parameters:
        tma_origin: Origin type for the TMA operation.
        dtype: Data type of the elements being written.
        tma_rank: Rank of the TMA tile (number of dimensions).
        tile_shape: Shape of the TMA tile for async store operations.
        desc_shape: Shape described by the TMA descriptor.
    """

    comptime _dtype = Self.dtype

    comptime TMATensorTilePtr = Pointer[
        TMATensorTile[
            Self.dtype, Self.tma_rank, Self.tile_shape, Self.desc_shape
        ],
        Self.tma_origin,
    ]
    var tma_op: Self.TMATensorTilePtr

    @always_inline
    def __init__(
        out self,
        tma_op: Self.TMATensorTilePtr,
    ):
        """Initialize the TMA tile writer.

        Args:
            tma_op: Pointer to the TMA tensor descriptor.
        """
        self.tma_op = tma_op

    @always_inline
    def write_tile(
        self,
        src: TileTensor[mut=True, Self._dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """Write a tile using TMA hardware acceleration.

        Performs an asynchronous TMA store from shared memory to global memory.
        The operation includes proper fencing and synchronization.

        Args:
            src: Source tile in shared memory.
            coords: Tile coordinates (col, row) in element space.

        Note:
            Coordinates are expected in (N, M) order for column-major output.
        """
        # Ensure all prior async operations are visible before TMA store
        fence_async_view_proxy()

        # Perform the async store
        self.tma_op[].async_store(src, (coords[0], coords[1]))

        # Commit and wait for completion
        self.tma_op[].commit_group()
        self.tma_op[].wait_group()


struct TileWriterThreadwise[
    dtype: DType,
    dst_layout: TensorLayout,
    dst_origin: MutOrigin,
    dst_engine: TensorEngine,
    dst_linear_idx_type: DType,
    //,
    thread_layout: MixedLayout,
    simd_size: Int,
    half_tile: Bool = False,  # Handle masked x2 case,
    swapAB: Bool = False,
](SMemTileWriter, TrivialRegisterPassable):
    """Writes shared-memory tiles to global memory using per-thread vectorized stores.

    Implements `SMemTileWriter` without hardware TMA: each thread reads a
    SIMD-width chunk from a swizzled shared-memory tile and writes it directly
    to the destination global tensor. Supports an optional A/B swap mapping and
    a half-tile mode for the x2 masked-consumer case.

    Parameters:
        dtype: Data type of the source and destination tiles (inferred).
        dst_layout: Layout of the destination global tensor (inferred).
        dst_origin: Origin type of the destination global tensor (inferred).
        dst_engine: Engine of the destination global tensor (inferred).
        dst_linear_idx_type: Linear index type for the destination tensor
            (inferred).
        thread_layout: Layout mapping threads across the tile for vectorized
            stores.
        simd_size: SIMD vector width, in elements, used for vectorized stores.
        half_tile: Whether to write only half the tile for the masked x2
            consumer case (defaults to False).
        swapAB: Whether to transpose the A and B matrix mapping (defaults to
            False).
    """

    comptime _dtype = Self.dtype

    comptime DstType = TileTensor[
        mut=True,
        Self.dtype,
        LayoutType=Self.dst_layout,
        origin=Self.dst_origin,
        Engine=Self.dst_engine,
        address_space=.GENERIC,
        linear_idx_type=Self.dst_linear_idx_type,
    ]
    var dst: Self.DstType
    var thread_idx: Int

    @always_inline
    def __init__(
        out self,
        dst: Self.DstType,
        thread_idx: Int,
    ):
        """Initialize the threadwise tile writer.

        Args:
            dst: Destination tensor in global memory.
            thread_idx: Thread index within the consumer warp group.
        """
        self.dst = dst
        self.thread_idx = thread_idx

    @always_inline
    def write_tile(
        self,
        src: TileTensor[mut=True, Self._dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """Write a tile using thread-distributed stores.

        Each thread writes a portion of the tile with proper swizzling
        for optimal memory access patterns.

        Args:
            src: Source tile in shared memory.
            coords: Tile indices (row_tile, col_tile) in the destination matrix.
        """
        # For the threadwise writer, coords are always (0, 0) because the
        # destination tile is already pre-extracted as the workgroup tile.
        # We directly use self.dst as the destination.

        comptime swizzle = make_ldmatrix_swizzle[
            Self._dtype,
            Self.dst_layout.static_shape[1],
            log2_floor(16 // size_of[Self._dtype]()),
        ]()

        comptime if Self.half_tile:
            if Self.swapAB:
                comptime threads_per_row = Self.thread_layout.static_shape[1]
                comptime num_threads = Int(Self.thread_layout.size())

                comptime dst_height = Self.dst_layout.static_shape[0] // 2
                comptime dst_width = Self.dst_layout.static_shape[1]

                # Slice both source and destination to half height internally
                var masked_src = src.slice((0, dst_height), (0, dst_width))

                var masked_dst = self.dst.slice((0, dst_height), (0, dst_width))

                var casted_thread_idx = self.thread_idx

                var rows = Int(self.dst.dim[0]())
                var cols = Int(self.dst.dim[1]()) // Self.simd_size

                var row_check = casted_thread_idx // threads_per_row < rows
                var col_check = (casted_thread_idx % threads_per_row) < cols

                comptime half_thread_layout = row_major[
                    Self.thread_layout.static_shape[0] // 2,
                    Self.thread_layout.static_shape[1],
                ]()

                if row_check and col_check:
                    copy_sram_to_dram[
                        thread_layout=half_thread_layout,
                        swizzle=swizzle,
                    ](
                        masked_dst.vectorize[1, Self.simd_size](),
                        masked_src.vectorize[1, Self.simd_size](),
                    )

            else:
                # Handle masked x2 case - write only half the tile width
                # Get compile-time layout dimensions
                comptime dst_height = Self.dst_layout.static_shape[0]
                comptime dst_width = Self.dst_layout.static_shape[1] // 2

                # Slice both source and destination to half width internally
                var masked_src = src.slice((0, dst_height), (0, dst_width))

                var masked_dst = self.dst.slice((0, dst_height), (0, dst_width))

                # Compute half-width thread layout
                comptime half_thread_layout = row_major[
                    Self.thread_layout.static_shape[0],
                    Self.thread_layout.static_shape[1] // 2,
                ]()

                # Only first half of threads participate
                comptime num_threads = Self.thread_layout.size()
                if self.thread_idx < (num_threads // 2):
                    copy_sram_to_dram[
                        thread_layout=half_thread_layout,
                        swizzle=swizzle,
                    ](
                        masked_dst.vectorize[1, Self.simd_size](),
                        masked_src.vectorize[1, Self.simd_size](),
                    )
        else:
            # Normal case - write full tile
            copy_sram_to_dram[
                thread_layout=Self.thread_layout,
                swizzle=swizzle,
            ](
                self.dst.vectorize[1, Self.simd_size](),
                src.vectorize[1, Self.simd_size](),
            )


trait RegTileWriter(TrivialRegisterPassable):
    """Base trait for tile writing mechanisms in matrix multiplication.

    This trait defines the interface for writing register tiles to memory
    (either shared memory or global memory).
    """

    @always_inline
    def write_tile(
        self,
        c_reg_tile: RegTile,
        coords: Tuple[Int, Int],
    ) capturing -> None:
        """Write a register tile to memory.

        Args:
            c_reg_tile: Source register tile containing accumulator values.
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        ...


struct FragmentToSMemWriter[
    c_type: DType,
    c_tile_layout: TensorLayout,
    //,
    tile_n_size: Int,  # Size of each tile in N dimension (e.g., TMA_BN)
    num_m_mmas: Int,
    num_consumer: Int,
    half_tile: Bool,
    WG_BM: Int,  # Warp group M dimension
    WG_BN: Int,  # Warp group N dimension
    sub_wg_id: Int,  # Sub warp group ID in N dimension
    swapAB: Bool = False,
](RegTileWriter):
    """Writes WGMMA accumulator results from registers to shared memory using st.matrix.

    Stores 16-byte fragments with swizzling to avoid bank conflicts. Sub-warp groups
    divide N-dimension work, each handling a portion of WG_BN output tiles.

    Parameters:
        c_type: Output data type (must be bfloat16 for st.matrix).
        c_tile_layout: Layout of the entire shared memory region.
        tile_n_size: Width of each output tile (typically TMA_BN).
        num_m_mmas: Number of MMA operations in M dimension.
        num_consumer: Number of consumer warp groups.
        half_tile: Special mode for handling partial tiles.
        WG_BM: Warp group tile height.
        WG_BN: Warp group tile width.
        sub_wg_id: Which portion of WG_BN this instance handles.
        swapAB: Whether to swap the A and B matrices.
    """

    comptime st_matrix_swizzle = make_ldmatrix_swizzle[
        Self.c_type,
        Self.tile_n_size if not Self.swapAB else Self.WG_BN,
        log2_floor(16 // size_of[Self.c_type]()),
    ]()

    comptime st_matrix_layout_regular = st_matrix_n_layout[
        Self.c_type, Self.tile_n_size, Self.num_m_mmas, Self.num_consumer
    ]()

    comptime st_matrix_layout_transpose = st_matrix_m_layout[
        Self.c_type, Self.tile_n_size, Self.num_m_mmas, Self.num_consumer
    ]()

    comptime st_matrix_rt_layout_type = RuntimeLayout[
        Self.st_matrix_layout_regular if not Self.swapAB else Self.st_matrix_layout_transpose,
        element_type=.int32,
        linear_idx_type=.int32,
    ]

    comptime st_matrix_tile_layout_regular = row_major[
        Self.WG_BM, Self.tile_n_size
    ]()
    comptime st_matrix_tile_layout_swapAB = row_major[
        Self.tile_n_size, Self.WG_BN
    ]()

    @__allow_legacy_any_origin_fields
    var c_tile: TileTensor[
        mut=True,
        Self.c_type,
        LayoutType=Self.c_tile_layout,
        origin=MutAnyOrigin,
        address_space=.SHARED,
    ]
    var warp_group_thread_idx: Int
    var local_warp_group_idx: Int
    var st_matrix_rt_layout: Self.st_matrix_rt_layout_type

    @always_inline
    def __init__(
        out self,
        c_tile: TileTensor[
            mut=True,
            Self.c_type,
            LayoutType=Self.c_tile_layout,
            origin=MutAnyOrigin,
            address_space=.SHARED,
        ],
        warp_group_thread_idx: Int,
        local_warp_group_idx: Int,
    ):
        """Initialize the fragment writer.

        Args:
            c_tile: Shared memory tile to write to.
            warp_group_thread_idx: Thread index within the warp group.
            local_warp_group_idx: Sub-warp group index (divides N-dimension work).
        """
        self.c_tile = c_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.st_matrix_rt_layout = Self.st_matrix_rt_layout_type()

    @always_inline
    def _compute_swizzled_offset[n_frag: Int, m_frag: Int](self) -> Int32:
        """Compute swizzled offset for st.matrix to avoid bank conflicts.

        Parameters:
            n_frag: Fragment index in N dimension within tile.
            m_frag: Fragment index in M dimension (MMA tile index).

        Returns:
            Swizzled offset for st.matrix instruction.
        """
        var layout_coords = RuntimeTuple[
            IntTuple(
                UNKNOWN_VALUE,
                IntTuple(n_frag, m_frag, UNKNOWN_VALUE),
            )
        ](
            self.warp_group_thread_idx,
            n_frag,
            m_frag,
            self.local_warp_group_idx,
        )
        var linear_idx = self.st_matrix_rt_layout(layout_coords)
        return self.st_matrix_swizzle(linear_idx)

    @always_inline
    def _store_fragment[
        elements_per_op: Int,  # 8 for normal mode, 4 for x2 mode
        m_frag: Int,
        n_frag: Int,
    ](
        self,
        smem_tile: TileTensor[
            mut=True,
            Self.c_type,
            origin=MutAnyOrigin,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        data: SIMD[Self.c_type, elements_per_op],
    ) -> None:
        """Store register data to shared memory using st.matrix instruction.

        Parameters:
            elements_per_op: Elements per st.matrix operation (4 or 8).
            m_frag: Fragment index in M dimension (0 to num_m_mmas-1).
            n_frag: Fragment index in N dimension within tile.

        Args:
            smem_tile: Target shared memory tile.
            data: Register data to store.
        """
        comptime packed_width = elements_per_op // 2  # BF16 pairs packed as float32

        # Pack BF16 pairs into float32 (hardware requirement)
        var packed_data = bitcast[.float32, packed_width](data)

        # Get swizzled offset for bank conflict avoidance
        var swizzled_offset = self._compute_swizzled_offset[n_frag, m_frag]()

        # Execute st.matrix hardware instruction
        st_matrix[simd_width=packed_width, transpose=Self.swapAB](
            smem_tile._storage + swizzled_offset, packed_data
        )

    @always_inline
    def write_tile(
        self,
        c_reg_tile: RegTile,
        coords: Tuple[Int, Int],
    ) capturing -> None:
        """Write accumulator tile from registers to shared memory.

        Args:
            c_reg_tile: Register tile containing MMA results.
            coords: Tile position (row_idx, col_idx) in output.
        """
        # Locate destination tile in shared memory
        var tile_linear_idx = coords[0] + coords[1]

        # Elements per tile: rows * cols
        # For normal: WG_BM rows, tile_n_size cols
        # For swapAB: WG_BN rows, tile_n_size cols
        comptime elements_per_tile_reg = Self.WG_BM * Self.tile_n_size
        comptime elements_per_tile_swapAB = Self.WG_BN * Self.tile_n_size
        comptime elements_per_tile = elements_per_tile_reg if not Self.swapAB else elements_per_tile_swapAB
        # Create a flat TileTensor view of the target shared-memory tile.
        comptime flat_tile_layout = row_major[1, elements_per_tile]()
        var dest_tile_flat = TileTensor[
            mut=True,
            Self.c_type,
            LayoutType=type_of(flat_tile_layout),
            origin=MutAnyOrigin,
            address_space=.SHARED,
        ](
            self.c_tile._storage + tile_linear_idx * elements_per_tile,
            flat_tile_layout,
        )

        # SMem fragment view for st.matrix operations
        # st.matrix configuration
        comptime elements_per_store = 4 * (1 if Self.half_tile else 2)
        comptime reg_fragment_scale = 2 if Self.half_tile else 1

        comptime ST_MATRIX_WIDTH_BYTES = 16  # Fragment size: st.matrix operates on 16-byte chunks

        # Sub-warp group offset: each sub-warp handles a portion of the tile
        # For normal: sub_wg_id * WG_BN
        # For swapAB: sub_wg_id * WG_BM
        comptime sub_wg_offset_reg = Self.WG_BN
        comptime sub_wg_offset_swapAB = Self.WG_BM
        comptime sub_wg_offset = sub_wg_offset_reg if not Self.swapAB else sub_wg_offset_swapAB

        var n_fragment_base = (
            coords[1] * Self.tile_n_size + Self.sub_wg_id * sub_wg_offset
        ) // ST_MATRIX_WIDTH_BYTES

        comptime if Self.swapAB:
            var smem_frag = dest_tile_flat.reshape(
                Self.st_matrix_tile_layout_swapAB
            )
            comptime for m_frag, n_frag in std.itertools.product(
                range(Self.num_m_mmas),
                range(Self.tile_n_size // ST_MATRIX_WIDTH_BYTES),
            ):
                var reg_fragment_idx = reg_fragment_scale * (
                    n_fragment_base + n_frag
                )
                var reg_fragment = c_reg_tile.tile[1, elements_per_store](
                    m_frag, reg_fragment_idx
                )
                var frag_data = reg_fragment.load[elements_per_store](
                    0, 0
                ).cast[Self.c_type]()
                self._store_fragment[elements_per_store, m_frag, n_frag](
                    smem_frag, frag_data
                )
        else:
            var smem_frag = dest_tile_flat.reshape(
                Self.st_matrix_tile_layout_regular
            )
            comptime for m_frag, n_frag in std.itertools.product(
                range(Self.num_m_mmas),
                range(Self.tile_n_size // ST_MATRIX_WIDTH_BYTES),
            ):
                var reg_fragment_idx = reg_fragment_scale * (
                    n_fragment_base + n_frag
                )
                var reg_fragment = c_reg_tile.tile[1, elements_per_store](
                    m_frag, reg_fragment_idx
                )
                var frag_data = reg_fragment.load[elements_per_store](
                    0, 0
                ).cast[Self.c_type]()
                self._store_fragment[elements_per_store, m_frag, n_frag](
                    smem_frag, frag_data
                )


struct RegisterToGMemWriter[
    c_type: DType,
    dst_layout: TensorLayout,
    dst_origin: MutOrigin,
    dst_engine: TensorEngine,
    dst_linear_idx_type: DType,
    //,
    wgmma_shape: IndexList[3],
    num_consumer: Int,
    N: Int,  # Matrix N dimension
    epilogue_fn: Optional[elementwise_epilogue_type] = None,
    compute_lambda_fn: Optional[elementwise_compute_lambda_type] = None,
    check_runtime_bounds: Bool = False,  # New parameter for N-dimension bounds checking
    swapAB: Bool = False,
](RegTileWriter):
    """Writer for transferring accumulator registers directly to global memory.

    This writer handles the direct copy from register tiles to global memory
    tiles, with proper thread distribution and alignment. It supports optional
    epilogue processing, compute lambda transformations, and bounds checking.

    Parameters:
        c_type: Output data type.
        dst_layout: Layout of the destination tensor.
        dst_origin: Origin type of the destination tensor.
        dst_engine: Engine of the destination tensor.
        dst_linear_idx_type: Linear index type for destination tensor.
        wgmma_shape: Shape of the WGMMA operation [M, N, K].
        num_consumer: Number of consumer warp groups.
        N: Matrix N dimension.
        epilogue_fn: Optional epilogue function (mutates value in place).
        compute_lambda_fn: Optional compute lambda function (returns new value).
        check_runtime_bounds: Whether to perform bounds checking on N dimension.
        swapAB: Whether to swap the A and B matrices.
    Note:
        At most one of epilogue_fn or compute_lambda_fn should be set.
    """

    comptime c_frag_size = Self.wgmma_shape[0] * Self.wgmma_shape[
        1
    ] // WARPGROUP_SIZE
    comptime num_n_frag_mat = Self.wgmma_shape[1] // 8
    comptime num_m_frag_mat = Self.wgmma_shape[0] // 4 // 8
    comptime num_frag_mats = Self.num_n_frag_mat * Self.num_m_frag_mat

    var thread_info: ThreadInfo

    comptime DstType = TileTensor[
        mut=True,
        Self.c_type,
        LayoutType=Self.dst_layout,
        origin=Self.dst_origin,
        Engine=Self.dst_engine,
        address_space=.GENERIC,
        linear_idx_type=Self.dst_linear_idx_type,
    ]
    var dst: Self.DstType
    var num_m_mmas: Int
    var tile_coords: OptionalReg[TileCoordinates]
    var max_row: OptionalReg[UInt32]

    @always_inline
    def __init__(
        out self,
        dst: Self.DstType,
        warp_group_thread_idx: Int,
        num_m_mmas: Int,
        tile_coords: OptionalReg[TileCoordinates] = None,
        max_row: OptionalReg[UInt32] = None,
    ):
        """Initialize the register-to-global-memory writer.

        Args:
            dst: Destination tensor in global memory.
            warp_group_thread_idx: Thread index within the warp group.
            num_m_mmas: Number of MMA tiles in M dimension.
            tile_coords: Optional tile coordinates for epilogue processing.
            max_row: Optional maximum valid M coordinate (for epilogue).
        """
        comptime assert (Self.epilogue_fn is None) or (
            Self.compute_lambda_fn is None
        ), "Only one of epilogue_fn or compute_lambda_fn should be set"

        # Store destination tensor
        self.dst = dst
        self.num_m_mmas = num_m_mmas
        self.tile_coords = tile_coords
        self.max_row = max_row

        # Extract thread information
        self.thread_info = ThreadInfo.from_warp_group_idx(warp_group_thread_idx)

    @always_inline
    def _get_mma_id(self, m_mma: Int, n_mma: Int) -> Int:
        """Calculate MMA tile ID from M and N indices.

        Args:
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.

        Returns:
            The linearized MMA tile ID.
        """
        return n_mma * self.num_m_mmas + m_mma

    @always_inline
    def write_tile(
        self,
        c_reg_tile: RegTile,
        coords: Tuple[Int, Int],
    ) capturing -> None:
        """Write a single MMA tile from registers to global memory.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        var m_mma = coords[0]
        var n_mma = coords[1]
        var mma_id = self._get_mma_id(m_mma, n_mma)

        comptime if Self.check_runtime_bounds or Self.swapAB:
            # Element-by-element with runtime bounds checking
            self._write_with_runtime_bounds(c_reg_tile, m_mma, n_mma, mma_id)
        elif Self.epilogue_fn is not None or Self.compute_lambda_fn is not None:
            # Vectorized with epilogue/compute_lambda
            self._write_with_transform(c_reg_tile, m_mma, n_mma, mma_id)
        else:
            # Direct vectorized copy
            self._write_direct_vectorized(c_reg_tile, m_mma, n_mma, mma_id)

    @always_inline
    def _write_direct_vectorized(
        self,
        c_reg_tile: RegTile,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Direct vectorized copy without transformations."""
        # Get the warp's portion of the tile
        var warp_tile = self.dst.tile[
            Self.wgmma_shape[0] // 4, Self.wgmma_shape[1]
        ](Coord(m_mma * 4 + self.thread_info.warp_id, n_mma))

        # Get the corresponding register fragment
        var c_frag = c_reg_tile.tile[1, Self.c_frag_size](mma_id, 0)

        # Direct copy using hardware layout
        copy_local_to_dram[row_major[8, 4]()](
            warp_tile.vectorize[1, 2](),
            reg_tile_to_tile_tensor(c_frag).vectorize[1, 2](),
        )

    @always_inline
    def _write_with_transform(
        self,
        c_reg_tile: RegTile,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Vectorized write with epilogue or compute_lambda transformation."""
        # Get warp tile and coordinates
        var warp_tile, warp_tile_coords, _ = self.dst.tile_with_offset[
            Self.wgmma_shape[0] // 4, Self.wgmma_shape[1]
        ](Coord(m_mma * 4 + self.thread_info.warp_id, n_mma))

        # Calculate global coordinates
        var warp_coords_base = IndexList[2](
            warp_tile_coords[0], warp_tile_coords[1]
        )
        var warp_coords = self.tile_coords.value().adjust(warp_coords_base)

        # Distribute fragments across threads
        var c_reg_frag = c_reg_tile.vectorize[1, 2]()
        var gmem_frag, gmem_offset_coords_raw, _ = warp_tile.vectorize[
            1, 2
        ]().distribute_with_offset[row_major[8, 4]()](self.thread_info.lane_id)

        var gmem_offset_coords = IndexList[2](
            gmem_offset_coords_raw[0], gmem_offset_coords_raw[1] * 2
        )
        var coords = gmem_offset_coords + warp_coords
        var max_row = self.max_row.value()

        comptime num_vecs = type_of(gmem_frag).LayoutType.static_product

        # Process all vectors
        comptime for frag_idx in range(num_vecs):
            var dst_idx = Int(
                type_of(gmem_frag).LayoutType()(Coord(Idx[frag_idx], Idx[0]))
            )
            var dst_m_offset, dst_n_offset = divmod(dst_idx, Self.N)
            var m = coords[0] + dst_m_offset
            var n = coords[1] + dst_n_offset

            # Bounds check and apply transformation
            if m < Int(max_row) and n < Self.N:
                Self._apply_transform_and_store[frag_idx](
                    gmem_frag, c_reg_frag, mma_id, m, n
                )

    @always_inline
    @staticmethod
    def _apply_transform_and_store[
        frag_idx: Int
    ](
        gmem_frag: TileTensor[mut=True, Self.c_type, ...],
        c_reg_frag: RegTile,
        mma_id: Int,
        m: Int,
        n: Int,
    ) capturing:
        """Apply epilogue or compute_lambda and store result."""
        comptime alignment = align_of[SIMD[Self.c_type, 2]]()

        comptime if Self.epilogue_fn:
            comptime epilogue = Self.epilogue_fn.value()
            epilogue[alignment=alignment](
                (m, n),
                c_reg_frag[mma_id, frag_idx].cast[Self.c_type](),
            )
        else:  # compute_lambda
            comptime compute_lambda = Self.compute_lambda_fn.value()
            var reg_val = compute_lambda[alignment=alignment](
                (m, n),
                c_reg_frag[mma_id, frag_idx].cast[Self.c_type](),
            )
            gmem_frag.store(Coord(Idx[frag_idx], Idx[0]), reg_val)

    @always_inline
    def _write_with_runtime_bounds(
        self,
        c_reg_tile: RegTile,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Element-by-element with full runtime bounds checking.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.
            mma_id: Linearized MMA tile ID.
        """

        comptime warp_tile_size_m = Self.wgmma_shape[
            0
        ] // 4 if not Self.swapAB else Self.wgmma_shape[1]
        comptime warp_tile_size_n = Self.wgmma_shape[
            1
        ] if not Self.swapAB else Self.wgmma_shape[0] // 4

        var coord_0 = (
            m_mma * 4 + self.thread_info.warp_id
        ) if not Self.swapAB else n_mma
        var coord_1 = n_mma if not Self.swapAB else (
            m_mma * 4 + self.thread_info.warp_id
        )

        # Get warp tile with bounds checking
        var warp_tile, warp_tile_coords_raw, _ = self.dst.tile_with_offset[
            warp_tile_size_m, warp_tile_size_n
        ](Coord(coord_0, coord_1))

        var warp_tile_coords = rebind[IndexList[2]](warp_tile_coords_raw)
        if self.tile_coords:
            warp_tile_coords = self.tile_coords.value().adjust(warp_tile_coords)

        # Process fragment matrices
        comptime for m_frag, n_frag in std.itertools.product(
            range(Self.num_m_frag_mat), range(Self.num_n_frag_mat)
        ):
            comptime frag_mat_id = n_frag * Self.num_m_frag_mat + m_frag

            # Fragment tile position in warp tile
            # For normal: (m_frag, n_frag)
            # For swapAB: (n_frag, m_frag) - transposed
            var frag_row = m_frag if not Self.swapAB else n_frag
            var frag_col = n_frag if not Self.swapAB else m_frag
            var frag_mat_gmem = warp_tile.tile[8, 8](frag_row, frag_col)

            # Get runtime output bounds from the destination TileTensor.
            var max_row = UInt32(self.dst.dim[0]())
            var max_col = UInt32(self.dst.dim[1]())

            comptime for i in range(2):
                # Bounds check coordinates
                # For normal: row = lane_row, col = lane_col * 2 + i
                # For swapAB: row = lane_col * 2 + i, col = lane_row (transposed)
                var lane_row_idx = self.thread_info.lane_row
                var lane_col_idx = self.thread_info.lane_col * 2 + UInt32(i)
                var check_row = (
                    lane_row_idx if not Self.swapAB else lane_col_idx
                )
                var check_col = (
                    lane_col_idx if not Self.swapAB else lane_row_idx
                )

                if check_row < max_row and check_col < max_col:
                    var reg_val = c_reg_tile[mma_id, frag_mat_id * 2 + i].cast[
                        Self.c_type
                    ]()

                    @__parameter
                    def epilogue_coordinates() -> Tuple[Int, Int]:
                        comptime if Self.swapAB:
                            # In swapAB mode, coordinates are transposed
                            return (
                                warp_tile_coords[0]
                                + Int(
                                    UInt32(n_frag * 8)
                                    + self.thread_info.lane_col * 2
                                    + UInt32(i)
                                ),
                                warp_tile_coords[1]
                                + Int(
                                    UInt32(m_frag * 8)
                                    + self.thread_info.lane_row
                                ),
                            )
                        else:
                            return (
                                warp_tile_coords[0]
                                + Int(
                                    UInt32(m_frag * 8)
                                    + self.thread_info.lane_row
                                ),
                                warp_tile_coords[1]
                                + Int(
                                    UInt32(n_frag * 8)
                                    + self.thread_info.lane_col * 2
                                    + UInt32(i)
                                ),
                            )

                    comptime if Self.epilogue_fn:
                        comptime epilogue = Self.epilogue_fn.value()
                        var frag_m, frag_n = epilogue_coordinates()
                        epilogue[alignment=align_of[Scalar[Self.c_type]]()](
                            (frag_m, frag_n), reg_val
                        )
                    else:
                        comptime if Self.compute_lambda_fn:
                            comptime compute_lambda = Self.compute_lambda_fn.value()
                            var frag_m, frag_n = epilogue_coordinates()
                            reg_val = compute_lambda[
                                alignment=align_of[Scalar[Self.c_type]]()
                            ]((frag_m, frag_n), reg_val)

                        # Store coordinates
                        # For normal: (lane_row, lane_col * 2 + i)
                        # For swapAB: (lane_col * 2 + i, lane_row) - transposed
                        var store_row = Int(
                            lane_row_idx
                        ) if not Self.swapAB else Int(lane_col_idx)
                        var store_col = Int(
                            lane_col_idx
                        ) if not Self.swapAB else Int(lane_row_idx)
                        frag_mat_gmem[store_row, store_col] = rebind[
                            type_of(frag_mat_gmem[store_row, store_col])
                        ](reg_val)
