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

"""TMA tile loader for SM100 matrix multiplication.

Provides a wrapper around TMA async_multicast_load operations, following
the SM90 TileLoaderTMA pattern. Orchestration logic (k-group iteration,
expect_bytes, barrier management) is handled by the kernel, not the loader.

Usage:
    # In kernel - create separate A and B loaders
    var a_loader = ATileLoaderType(Pointer(to=a_tma_op), ctx.a_multicast_mask)
    var b_loader = BTileLoaderType(Pointer(to=b_tma_op), ctx.b_multicast_mask)

    # Load tiles using TileTensor
    a_loader.load(a_tile, barrier, k_coord, m_coord)
    b_loader.load(b_tile, barrier, k_coord, n_coord)

    # TileTensor tiles are passed directly to TMA ops
"""

from layout import (
    TensorLayout,
    TileTensor,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

# Import TileTensor types for overloaded load methods
from structured_kernels.tile_types import SMemTile2D, TmaOpType

from std.utils.index import IndexList


struct TileLoaderTMA[
    tma_origin: ImmOrigin,
    dtype: DType,
    tma_rank: Int,
    tile_shape: IndexList[tma_rank],
    desc_shape: IndexList[tma_rank],
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA-based tile loader for SM100.

    Wraps a TMA descriptor and multicast mask for efficient tile loading.
    The load method issues async_multicast_load with proper CTA group handling.

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type.
        tma_rank: Rank of the TMA tile/descriptor shapes.
        tile_shape: TMA tile shape as IndexList.
        desc_shape: TMA descriptor shape as IndexList.
        cta_group: CTA group size (1 or 2 for SM100 2-SM MMA).
    """

    comptime TmaOp = TMATensorTile[
        Self.dtype, Self.tma_rank, Self.tile_shape, Self.desc_shape
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    # TMA descriptor pointer (referencing grid constant)
    var tma_op: Self.TmaOpPtr
    # Multicast mask for cluster distribution
    var multicast_mask: UInt16

    @always_inline
    def __init__(out self, tma_op: Self.TmaOpPtr, multicast_mask: UInt16):
        """Initialize the TMA tile loader.

        Args:
            tma_op: Pointer to TMA descriptor (grid constant).
            multicast_mask: Multicast mask for cluster distribution.
        """
        self.tma_op = tma_op
        self.multicast_mask = multicast_mask

    @always_inline
    def load[
        dim0: Int,
        dim1: Int,
        /,
        alignment: Int = 128,
    ](
        self,
        dest: SMemTile2D[Self.dtype, dim0, dim1, alignment=alignment],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: Int,
        row_coord: Int,
    ):
        """Load a tile using TMA hardware acceleration.

        Issues an async multicast load from global memory to shared memory.
        Coordinates are in element units (not tile units).

        Parameters:
            dim0: Number of rows in the destination SMEM tile (inferred).
            dim1: Number of columns in the destination SMEM tile (inferred).
            alignment: Byte alignment of the destination SMEM tile (defaults to 128).

        Args:
            dest: Destination SMEM TileTensor tile.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        # TileTensor overload of async_multicast_load - no conversion needed
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )

    @always_inline
    def load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype, LayoutType, MutAnyOrigin, address_space=.SHARED
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: Int,
        row_coord: Int,
    ):
        """Load a TileTensor tile with variadic shape/stride types using TMA.

        This overload accepts TileTensor tiles with swizzled layouts (created via
        internal_k_major) and passes them to the TMA operation.

        Parameters:
            LayoutType: Layout type of the destination `TileTensor`.

        Args:
            dest: Destination SMEM TileTensor tile with swizzled layout.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )


# =============================================================================
# TileLoader -- TileTensor-native TMA loader (new Layout types)
# =============================================================================


struct TileLoader[
    tma_origin: ImmOrigin,
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA tile loader parameterized on new Layout types.

    Uses TmaOpType to derive the TMATensorTile type from new Layout.
    Accepts TileTensor destinations.

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type.
        tile_layout: Layout of the tile loaded into shared memory.
        desc_layout: Layout of the TMA descriptor.
        cta_group: CTA group size (1 or 2 for SM100 2-SM MMA).
    """

    comptime TmaOp = TmaOpType[Self.dtype, Self.tile_layout, Self.desc_layout]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr
    var multicast_mask: UInt16

    @always_inline
    def __init__[
        tma_op_type: AnyType
    ](
        out self,
        tma_op: Pointer[tma_op_type, Self.tma_origin],
        multicast_mask: UInt16,
    ):
        """Accepts any TMA pointer. Rebinds to the loader's derived type.

        Parameters:
            tma_op_type: Compile-time type of the passed TMA descriptor pointer.

        Args:
            tma_op: Pointer to the TMA descriptor.
            multicast_mask: Multicast mask for cluster distribution.
        """
        self.tma_op = rebind[Self.TmaOpPtr](tma_op)
        self.multicast_mask = multicast_mask

    @always_inline
    def load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype, LayoutType, MutAnyOrigin, address_space=.SHARED
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        k_coord: Int,
        row_coord: Int,
    ):
        """Load a tile using TMA async multicast load.

        Parameters:
            LayoutType: Layout type of the destination TileTensor.

        Args:
            dest: Destination SMEM TileTensor tile.
            barrier: Memory barrier for TMA completion signaling.
            k_coord: K dimension coordinate in global memory (elements).
            row_coord: Row coordinate (M for A, N for B) in global memory (elements).
        """
        self.tma_op[].async_multicast_load[Self.cta_group](
            dest, barrier, (k_coord, row_coord), self.multicast_mask
        )


# =============================================================================
# ScalesLoader -- TileTensor-native scales loader (new Layout types)
# =============================================================================


struct ScalesLoader[
    tma_origin: ImmOrigin,
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout = tile_layout,
    /,
    *,
    cta_group: Int,
](TrivialRegisterPassable):
    """TMA scales loader parameterized on new Layout types.

    Uses TmaOpType to derive the TMATensorTile type from new Layout.
    Uses async_copy (no multicast). Coordinate order is
    (row_coord, k_coord) matching scales tensor layout.

    Parameters:
        tma_origin: Origin of the TMA descriptor pointer.
        dtype: Element data type.
        tile_layout: Layout of the scales tile loaded into shared memory.
        desc_layout: Layout of the TMA descriptor (defaults to `tile_layout`).
        cta_group: CTA group size (1 or 2 for SM100 2-SM MMA).
    """

    comptime TmaOp = TmaOpType[Self.dtype, Self.tile_layout, Self.desc_layout]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]

    var tma_op: Self.TmaOpPtr

    @always_inline
    def __init__[
        tma_op_type: AnyType
    ](out self, tma_op: Pointer[tma_op_type, Self.tma_origin]):
        """Accepts any TMA pointer. Rebinds to the loader's derived type.

        Parameters:
            tma_op_type: Compile-time type of the passed TMA descriptor pointer.

        Args:
            tma_op: Pointer to the TMA descriptor.
        """
        self.tma_op = rebind[Self.TmaOpPtr](tma_op)

    @always_inline
    def load[
        LayoutType: TensorLayout
    ](
        self,
        dest: TileTensor[
            Self.dtype, LayoutType, MutAnyOrigin, address_space=.SHARED
        ],
        ref[AddressSpace.SHARED] barrier: SharedMemBarrier,
        row_coord: Int,
        k_coord: Int,
    ):
        """Load scales using TMA async copy.

        Parameters:
            LayoutType: Layout type of the destination TileTensor.

        Args:
            dest: Destination SMEM TileTensor tile for scales.
            barrier: Memory barrier for TMA completion signaling.
            row_coord: Row coordinate in global memory (elements).
            k_coord: K dimension coordinate in global memory (elements).
        """
        self.tma_op[].async_copy[Self.cta_group](
            dest, barrier, (row_coord, k_coord)
        )
