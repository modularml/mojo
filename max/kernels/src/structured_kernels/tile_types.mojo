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
"""Native TileTensor types for SM100 structured kernels.

This module provides TileTensor-based tile types for SM100 structured kernels.
All SMEM storage uses TileTensor natively. TileTensors are passed directly
to TMA and MMA via TileTensor overloads.

Usage:

    # Create tile with a layout
    comptime my_layout = row_major[64, 32]()
    comptime MyTile = SMemTile[.float16, my_layout]

    # TileTensors are passed directly to TMA/MMA
    tma_op.async_copy(tile, barrier, coords)
"""

from std.sys import size_of

from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    DefaultEngine,
    Idx,
    LayoutTensor,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.coord import _Flattened
from layout.tma_async import (
    TMATensorTile,
    TMATensorTileIm2col,
    create_tensor_tile,
)
from layout.tile_layout import Layout
from std.utils.index import IndexList
from std.memory import unsafe_stack_allocation
from std.utils.index import IndexList

# Core matrix constant from tensor_core_async.mojo
comptime _CM_NUM_ROWS = 8


# ============================================================================
# TilePayload - Marker trait for tile payload types
# ============================================================================


trait TilePayload(TrivialRegisterPassable):
    """Trait for tile payload types. Must be extend TrivialRegisterPassable."""

    pass


# ============================================================================
# Swizzle Mode Conversion
# ============================================================================


comptime swizzle_mode_to_bytes[
    swizzle_mode: TensorMapSwizzle
] = 128 if swizzle_mode == TensorMapSwizzle.SWIZZLE_128B else (
    64 if swizzle_mode
    == TensorMapSwizzle.SWIZZLE_64B else (
        32 if swizzle_mode == TensorMapSwizzle.SWIZZLE_32B else 16
    )
)
"""Convert TensorMapSwizzle enum to swizzle size in bytes.

SWIZZLE_NONE returns 16, matching ``TensorMapSwizzle.bytes()``
(formula: ``(2**value) * 16``, value=0 gives 16).

Parameters:
    swizzle_mode: The TensorMapSwizzle enum value.

Returns:
    The swizzle size in bytes (16, 32, 64, or 128).
"""


# ============================================================================
# Internal Swizzled Layout Aliases
# ============================================================================
# These create internal Layout types that match the swizzled structures from
# tile_layout_k_major. Using internal Layout allows passing through struct
# parameters while preserving type information for .to_layout_tensor().
#
# The key insight: new Layout (from _layout.mojo) has compile-time type
# parameters (shape_types, stride_types) that are preserved through struct
# chains, unlike LegacyLayout (from layout.mojo) which uses runtime IntTuple.

# Internal swizzled layout for K-major access with configurable swizzle.
# This layout is coalesce-equivalent to tile_layout_k_major_typed for the
# first 2 modes (which is what MMA descriptor creation needs).
# Migration to tile_layout_k_major_typed is tracked as Phase 4 work.
comptime internal_k_major[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_bytes: Int,
] = Layout(
    Coord(
        Coord(Idx[BM // _CM_NUM_ROWS], Idx[_CM_NUM_ROWS]),
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()],
            Idx[BK * size_of[dtype]() // swizzle_bytes],
        ),
    ),
    Coord(
        Coord(
            Idx[swizzle_bytes // size_of[dtype]()],
            Idx[(BM // _CM_NUM_ROWS) * (swizzle_bytes // size_of[dtype]())],
        ),
        Coord(Idx[1], Idx[0]),
    ),
)

# Convenience aliases for common swizzle sizes
comptime internal_k_major_128B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 128]

comptime internal_k_major_64B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 64]

comptime internal_k_major_32B[
    dtype: DType, BM: Int, BK: Int
] = internal_k_major[dtype, BM, BK, 32]

# Internal row-major layout (no swizzle) - just delegates to row_major
comptime internal_k_major_none[
    dtype: DType,
    BM: Int,
    BK: Int,
] = row_major[BM, BK]()


# ============================================================================
# Internal SF layout for scale factors
# ============================================================================
#
# This layout matches tile_sf_layout_k_major (from tensor_core_async.mojo)
# and is used for both storage allocation and TileTensor type parameters.
# MMA extracts the layout directly from TileTensor's compile-time type params.
#
# Layout structure: ((32, tiles_m), ((4, 4), tiles_k))
# - K-major ordering: K-tiles placed with stride 512 (32*16)
# - M-tiles placed after all K content with stride dim1*32

# SF atom constants (from tile_sf_layout_k_major in tensor_core_async.mojo)
comptime _SF_ATOM_M_0 = 32
comptime _SF_ATOM_M_1 = 4
comptime _SF_ATOM_K = 4
comptime _SF_ATOM_SIZE = _SF_ATOM_M_1 * _SF_ATOM_K  # 16 = atom size in K dimension
comptime _SF_MN_GROUP_SIZE = _SF_ATOM_M_0 * _SF_ATOM_M_1  # 128

# Internal SF layout for scale factor STORAGE
# Takes dim0/dim1 which are the already-computed SFA/SFB dimensions
# The size (dim0 * dim1) matches tile_sf_layout_k_major.size()
# Supports tiling in both M and K dimensions:
# - tiles_m = dim0 // 32: number of M-tiles
# - tiles_k = dim1 // 16: number of K-tiles
comptime internal_sf_k_major[
    dim0: Int,  # (BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
    dim1: Int,  # (SF_BK // (SF_ATOM_K * vec_sf_size)) * (SF_ATOM_M[1] * SF_ATOM_K)
    # where SF_BK = SF_K_GROUP_SIZE * num_sf_k_tiles (NOT raw BK)
] = Layout(
    # Shape: ((32, tiles_m), ((4, 4), tiles_k))
    # When tiles_m = 1 or tiles_k = 1, structure is preserved but factor is 1
    Coord(
        Coord(Idx[_SF_ATOM_M_0], Idx[dim0 // _SF_ATOM_M_0]),
        Coord(
            Coord(Idx[_SF_ATOM_M_1], Idx[_SF_ATOM_K]),
            Idx[dim1 // _SF_ATOM_SIZE],
        ),
    ),
    # Stride: ((16, dim1*32), ((1, 4), 512))
    # Mode 0: (16, dim1*32) = (atom_stride, M-tile stride)
    #         M-tile stride = num_k_tiles * k_tile_stride = (dim1/16) * 512 = dim1 * 32
    # Mode 1: ((1, 4), 512) = ((inner_strides), K-tile stride = 32*16)
    # K-major: K-tiles are placed with stride = atom M size (32*16=512)
    Coord(
        Coord(
            Idx[_SF_ATOM_SIZE],
            Idx[dim1 * _SF_ATOM_M_0],
        ),
        Coord(
            Coord(Idx[1], Idx[_SF_ATOM_M_1]),
            Idx[_SF_ATOM_M_0 * _SF_ATOM_SIZE],
        ),
    ),
)


# ============================================================================
# sf_tile_dim0/dim1 -- Compute SF tile dimensions from matmul tile parameters
# ============================================================================

# dim0 for internal_sf_k_major: (BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
# BM is the M-dimension (or align_up(MMA_N, SF_MN_GROUP_SIZE) for SFB).
comptime sf_tile_dim0[BM: Int] = (BM // _SF_MN_GROUP_SIZE) * _SF_ATOM_M_0

# dim1 for internal_sf_k_major:
#   (sf_bk // (SF_ATOM_K * vec_sf_size)) * (SF_ATOM_M[1] * SF_ATOM_K)
# sf_bk = SF_K_GROUP_SIZE * num_sf_k_tiles (NOT raw BK).
comptime sf_tile_dim1[sf_bk: Int, vec_sf_size: Int] = (
    sf_bk // (_SF_ATOM_K * vec_sf_size)
) * (_SF_ATOM_M_1 * _SF_ATOM_K)


# ============================================================================
# Core TileTensor type for shared memory
# ============================================================================

comptime SMemTile[
    dtype: DType,
    layout: Layout,
    *,
    alignment: Int = 128,
] = TileTensor[
    dtype,
    Layout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutAnyOrigin,
    address_space=.SHARED,
]
"""Shared memory tile using TileTensor with a Layout.

The Layout parameter preserves swizzle information, enabling
.to_layout_tensor() to produce correctly swizzled LayoutTensors.

Parameters:
    dtype: The data type of tile elements.
    layout: The full layout including swizzle information.
    alignment: Memory alignment (default 128 for shared memory).
"""


# ============================================================================
# GMEMLayout1D -- 1D layout type for flat arrays (dynamic shape, stride 1)
# ============================================================================

comptime GMEMLayout1D = Layout[
    Coord[Int64].element_types,
    Coord[ComptimeInt[1]].element_types,
]
"""1D layout for flat global memory arrays.

Shape is dynamic (Scalar), stride is 1 (ComptimeInt[1]).
Rank is provably 1 at compile time.
"""


# ============================================================================
# static_row_major -- 2D row-major layout with fully static dimensions
# ============================================================================

comptime static_row_major[dim0: Int, dim1: Int] = Layout[
    Coord[ComptimeInt[dim0], ComptimeInt[dim1]].element_types,
    Coord[ComptimeInt[dim1], ComptimeInt[1]].element_types,
]
"""2D row-major layout with fully static dimensions.

Equivalent to `LegacyLayout.row_major(dim0, dim1)` but using new Layout
types with rank=2 provable at compile time.
"""

comptime _StridedLayout[dim0: Int, dim1: Int, stride0: Int] = Layout[
    Coord[ComptimeInt[dim0], ComptimeInt[dim1]].element_types,
    Coord[ComptimeInt[stride0], ComptimeInt[1]].element_types,
]
"""2D layout with explicit stride for dim0.

Used for sub-tiles that are strided views into wider rows. Shape is
[dim0, dim1] but stride is [stride0, 1] where stride0 >= dim1.
"""


def _strided_layout[
    dim0: Int, dim1: Int, stride0: Int
]() -> _StridedLayout[dim0, dim1, stride0]:
    return Layout(
        Coord(ComptimeInt[dim0](), ComptimeInt[dim1]()),
        Coord(ComptimeInt[stride0](), ComptimeInt[1]()),
    )


# ============================================================================
# _to_index_list -- Extract IndexList of shapes from a TensorLayout
# ============================================================================


@__parameter
def _to_index_list[L: TensorLayout]() -> IndexList[L.rank]:
    """Extract static shapes from a TensorLayout into an IndexList.

    Works for any rank. TMA layouts are always fully static.
    """
    var result = IndexList[L.rank]()

    comptime for i in range(L.rank):
        result[i] = L.static_shape[i]

    return result


def _to_index_list[rank: Int, L: TensorLayout]() -> IndexList[rank]:
    """Extract static shapes from a TensorLayout into an IndexList with explicit rank.

    Used when the compiler can't prove the TensorLayout's rank matches
    the expected rank.
    """
    comptime assert L.rank == rank, "TensorLayout rank must match explicit rank"
    var result = IndexList[rank]()

    comptime for i in range(rank):
        result[i] = L.static_shape[i]

    return result


# ============================================================================
# tma_desc_layout -- Compute TMA descriptor layout from tile layout + swizzle
# ============================================================================

# TMA descriptor layouts have the same shape as tile layouts except the
# last dimension is replaced with swizzle_bytes // element_size. All
# strides are 1 (flat layout for the TMA descriptor).
comptime tma_desc_layout_2d[
    dtype: DType,
    tile_dim0: Int,
    swizzle: TensorMapSwizzle,
] = Layout[
    Coord[
        ComptimeInt[tile_dim0], ComptimeInt[swizzle.bytes() // size_of[dtype]()]
    ].element_types,
    Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
]
"""2D TMA descriptor layout: [dim0, swizzle_elems], strides [1, 1]."""

comptime tma_desc_layout_3d[
    dtype: DType,
    tile_dim0: Int,
    tile_dim1: Int,
    swizzle: TensorMapSwizzle,
] = Layout[
    Coord[
        ComptimeInt[tile_dim0],
        ComptimeInt[tile_dim1],
        ComptimeInt[swizzle.bytes() // size_of[dtype]()],
    ].element_types,
    Coord[ComptimeInt[1], ComptimeInt[1], ComptimeInt[1]].element_types,
]
"""3D TMA descriptor layout: [dim0, dim1, swizzle_elems], strides [1,1,1]."""


# Variant of tma_desc_layout_3d that takes the innermost dim size explicitly
# instead of deriving it from the swizzle atom size. The default helper is
# bound to swizzle.bytes() // size_of[dtype](), which matches the per-swizzle
# upper bound for swizzled modes. For SWIZZLE_NONE, however, the driver only
# requires boxDim[0] * elem_size to be a multiple of 16 bytes (no upper cap
# beyond the 256-element limit), so callers may want a larger innermost dim
# matching their actual SMEM tile width. Use this helper for those cases.
comptime tma_desc_layout_3d_explicit_inner[
    tile_dim0: Int,
    tile_dim1: Int,
    inner_elems: Int,
] = Layout[
    Coord[
        ComptimeInt[tile_dim0],
        ComptimeInt[tile_dim1],
        ComptimeInt[inner_elems],
    ].element_types,
    Coord[ComptimeInt[1], ComptimeInt[1], ComptimeInt[1]].element_types,
]
"""3D TMA descriptor layout with explicit innermost dim (in elements)."""

comptime tma_desc_layout_4d[
    dtype: DType,
    tile_dim0: Int,
    tile_dim1: Int,
    tile_dim2: Int,
    swizzle: TensorMapSwizzle,
] = Layout[
    Coord[
        ComptimeInt[tile_dim0],
        ComptimeInt[tile_dim1],
        ComptimeInt[tile_dim2],
        ComptimeInt[swizzle.bytes() // size_of[dtype]()],
    ].element_types,
    Coord[
        ComptimeInt[1], ComptimeInt[1], ComptimeInt[1], ComptimeInt[1]
    ].element_types,
]
"""4D TMA descriptor layout: [d0,d1,d2,swizzle_elems], strides all 1."""

comptime tma_desc_layout_5d[
    dtype: DType,
    tile_dim0: Int,
    tile_dim1: Int,
    tile_dim2: Int,
    tile_dim3: Int,
    swizzle: TensorMapSwizzle,
] = Layout[
    Coord[
        ComptimeInt[tile_dim0],
        ComptimeInt[tile_dim1],
        ComptimeInt[tile_dim2],
        ComptimeInt[tile_dim3],
        ComptimeInt[swizzle.bytes() // size_of[dtype]()],
    ].element_types,
    Coord[
        ComptimeInt[1],
        ComptimeInt[1],
        ComptimeInt[1],
        ComptimeInt[1],
        ComptimeInt[1],
    ].element_types,
]
"""5D TMA descriptor layout: [d0,d1,d2,d3,swizzle_elems], strides all 1."""


# ============================================================================
# TMA op type from new Layout -- derive TMATensorTile with legacy layouts
# ============================================================================

comptime TmaOpType[
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout,
] = TMATensorTile[
    dtype,
    tile_layout.rank,
    _to_index_list[tile_layout](),
    _to_index_list[tile_layout.rank, desc_layout](),
]
"""TMATensorTile type derived from new Layout types.

Single source of truth: new Layout types determine the TMATensorTile
type parameters via _to_index_list.
"""

comptime TmaOpTypeIm2col[
    dtype: DType,
    tile_layout: TensorLayout,
    desc_layout: TensorLayout,
] = TMATensorTileIm2col[
    dtype,
    tile_layout.rank,
    _to_index_list[tile_layout](),
    _to_index_list[tile_layout.rank, desc_layout](),
]
"""TMATensorTileIm2col type derived from new Layout types.

Same as TmaOpType but for im2col TMA (used by conv2d activation loads).
"""


def create_tma_tile[
    tma_tile_layout: TensorLayout,
    tma_desc_layout: TensorLayout,
    tile_shape: IndexList[tma_tile_layout.rank],
    *,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor[mut=False, ...]) raises -> TmaOpType[
    tensor.dtype, tma_tile_layout, tma_desc_layout
]:
    """Create a TMATensorTile using new Layout types.

    Extracts IndexList shapes from new Layout types internally, calls
    create_tensor_tile, and returns TMATensorTile. No LegacyLayout
    is exposed to callers -- the conversion is encapsulated here.

    Parameters:
        tma_tile_layout: Tile layout as new TensorLayout.
        tma_desc_layout: Descriptor layout as new TensorLayout.
        tile_shape: Physical tile dimensions for the TMA descriptor.
        swizzle_mode: TMA swizzle mode.

    Args:
        ctx: Device context for TMA descriptor creation.
        tensor: Source tensor in global memory.

    Returns:
        A TMATensorTile (DevicePassable) for use with enqueue_function.
    """
    return create_tensor_tile[
        tile_shape,
        swizzle_mode=swizzle_mode,
        __tile_shape=_to_index_list[tma_tile_layout](),
        __desc_shape=_to_index_list[tma_tile_layout.rank, tma_desc_layout](),
    ](ctx, tensor)


def create_tma_tile[
    tma_tile_layout: TensorLayout,
    tma_desc_layout: TensorLayout,
    tile_shape: IndexList[tma_tile_layout.rank],
    *,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    unpack_fp4: Bool = False,
](ctx: DeviceContext, tensor: TileTensor[mut=False, ...]) raises -> TmaOpType[
    tensor.dtype, tma_tile_layout, tma_desc_layout
]:
    """TileTensor overload of create_tma_tile.

    Calls create_tensor_tile directly with TileTensor, bypassing
    LayoutTensor entirely. The TileTensor just needs ptr and layout
    shape/stride accessors, which work on any TileTensor including
    reshaped views.

    Parameters:
        tma_tile_layout: Tile layout as new TensorLayout.
        tma_desc_layout: Descriptor layout as new TensorLayout.
        tile_shape: Physical tile dimensions for the TMA descriptor.
        swizzle_mode: TMA swizzle mode.
        unpack_fp4: When True, `tensor` is nibble-packed E2M1 held as `uint8`
            and the copy pads it into shared memory so a K extent spans one
            byte per element.

    Args:
        ctx: Device context for TMA descriptor creation.
        tensor: Source TileTensor in global memory.

    Returns:
        A TMATensorTile (DevicePassable) for use with enqueue_function.
    """
    return create_tensor_tile[
        tile_shape,
        swizzle_mode=swizzle_mode,
        __tile_shape=_to_index_list[tma_tile_layout](),
        __desc_shape=_to_index_list[tma_tile_layout.rank, tma_desc_layout](),
        unpack_fp4=unpack_fp4,
    ](ctx, tensor)


# ============================================================================
# GMEMTile -- TileTensor type for global memory kernel parameters
# ============================================================================

comptime GMEMTile[
    dtype: DType,
    tt_layout: TensorLayout,
    Engine: TensorEngine = DefaultEngine[element_width=1],
] = TileTensor[dtype, tt_layout, MutAnyOrigin, Engine=Engine]
"""Global memory TileTensor for global memory kernel parameters.

Used for kernel parameter types, replacing LayoutTensor parameters.
"""


# ============================================================================
# Compile-time Accessors for SMemTile
# ============================================================================
# These comptimes allow accessing shape and stride values at compile time.
# Usage:
#   comptime shape0 = SMemTileShape[0, type_of(tile)]
#   comptime stride1 = SMemTileStride[1, type_of(tile)]

comptime SMemTileShape[
    idx: Int,
    Tile: TileTensor,
] = Tile.static_shape[idx]
"""Get compile-time shape value at index from a TileTensor type.

Parameters:
    idx: The dimension index.
    Tile: The TileTensor type (use type_of(tile)).

Returns:
    The static shape value, or -1 if runtime-determined.
"""

comptime SMemTileStride[
    idx: Int,
    Tile: TileTensor,
] = Tile.static_stride[idx]
"""Get compile-time stride value at index from a TileTensor type.

Parameters:
    idx: The dimension index.
    Tile: The TileTensor type (use type_of(tile)).

Returns:
    The static stride value, or -1 if runtime-determined.
"""

# Backward-compatible alias: SMemTile2D[dtype, dim0, dim1] -> SMemTile[dtype, row_major[dim0, dim1]()]
comptime SMemTile2D[
    dtype: DType,
    dim0: Int,
    dim1: Int,
    *,
    alignment: Int = 128,
] = SMemTile[dtype, row_major[dim0, dim1](), alignment=alignment]
"""Backward-compatible alias for SMemTile with explicit 2D dimensions."""


# ============================================================================
# SMemTileArrayWithLayout - Array of tiles with explicit Layout
# ============================================================================


# TODO: This type should correctly propagate mutability.
struct SMemTileArrayWithLayout[
    dtype: DType,
    tile_layout: Layout,
    num_tiles: Int,
    alignment: Int = 128,
](TrivialRegisterPassable):
    """Array of TileTensor tiles with explicit Layout (preserves swizzle info).

    Unlike SMemTileArray2D which uses row_major internally, this type preserves
    the full layout information from TMA swizzling, enabling .to_layout_tensor()
    to produce correctly swizzled LayoutTensors.

    Parameters:
        dtype: Tile element data type.
        tile_layout: The full layout including swizzle information.
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment (default 128 for shared memory).

    Example:
        comptime swizzled_layout = tile_layout_k_major[dtype, BM, BK, ...]()
        comptime MyArray = SMemTileArrayWithLayout[dtype, swizzled_layout, 4]

        var array = MyArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor with swizzled layout
        var lt = tile.to_layout_tensor()  # Correctly swizzled!
    """

    # The TileTensor-based tile type with correct layout
    comptime Tile = SMemTile[Self.dtype, Self.tile_layout]

    # Size calculations - use layout.product() for element count
    comptime tile_size: Int = Self.tile_layout.product()
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = Array[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArrayWithLayout pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    def __init__(
        out self,
        # TODO: this should correctly propagate mutability.
        unsafe_ptr: UnsafePointer[Scalar[Self.dtype], _, address_space=.SHARED],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = rebind[type_of(self.ptr)](unsafe_ptr)

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Parameters:
            T: Index value type, must be convertible to `Int`.

        Args:
            index: The tile index.

        Returns:
            A TileTensor with correct swizzled layout at the given index.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return Self.Tile(tile_ptr.as_unsafe_any_origin(), Self.tile_layout)

    def slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArrayWithLayout[
            Self.dtype, Self.tile_layout, length, Self.alignment
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArrayWithLayout representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    def stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArrayWithLayout backed by stack-allocated shared memory.
        """
        var ptr = unsafe_stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment=Self.alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


# ============================================================================
# SMemTileArray - Array of tiles with variadic shape/stride types
# ============================================================================


# TODO: This type should correctly propagate mutability.
struct SMemTileArray[
    dtype: DType,
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
    num_tiles: Int,
    alignment: Int = 128,
](TrivialRegisterPassable):
    """Array of TileTensor tiles with variadic shape/stride type parameters.

    This is the TileTensor equivalent of the LayoutTensor-based SMemTileArray
    in structuring.mojo. By taking shape_types and stride_types directly as
    variadic type parameters, this preserves full compile-time type information
    including swizzle patterns.

    Parameters:
        dtype: Tile element data type.
        shape_types: List of shape types from Layout (preserves compile-time info).
        stride_types: List of stride types from Layout (preserves compile-time info).
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment (default 128 for shared memory).

    Example:
        comptime a_layout = internal_k_major[dtype, BM, BK, 128]
        comptime ATileArray = SMemTileArray[
            dtype,
            a_layout.shape_types,
            a_layout.stride_types,
            num_pipeline_stages,
        ]

        var array = ATileArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor with correct swizzled layout
    """

    # The TileTensor-based tile type with correct shape/stride types
    comptime Tile = TileTensor[
        Self.dtype,
        Layout[shape_types=Self.shape_types, stride_types=Self.stride_types],
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    # Layout type for constructing tiles
    comptime TileLayout = Layout[
        shape_types=Self.shape_types,
        stride_types=Self.stride_types,
    ]

    # Flattened shape types (leaf scalars only, handles nested Coords).
    comptime _flat_shape_types = _Flattened[*Self.shape_types]

    # Size calculations using static shape product.
    comptime tile_size: Int = Coord[*Self._flat_shape_types].static_product
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = Array[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArray pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    def __init__(
        out self,
        # TODO: This should correctly propagate mutability
        unsafe_ptr: UnsafePointer[Scalar[Self.dtype], _, address_space=.SHARED],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = rebind[type_of(self.ptr)](unsafe_ptr)

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Parameters:
            T: Index value type, must be convertible to `Int`.

        Args:
            index: The tile index.

        Returns:
            A TileTensor with correct swizzled layout at the given index.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        # Construct layout from shape/stride types (all compile-time known)
        var layout = Self.TileLayout(
            Coord[*Self.shape_types](),
            Coord[*Self.stride_types](),
        )
        return Self.Tile(tile_ptr.as_unsafe_any_origin(), layout)

    def slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray[
            Self.dtype,
            Self.shape_types,
            Self.stride_types,
            length,
            Self.alignment,
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArray representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    def stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArray backed by stack-allocated shared memory.
        """
        var ptr = unsafe_stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment=Self.alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


# ============================================================================
# TileTensor Usage Patterns
# ============================================================================
#
# TileTensor is used natively throughout SM100 kernels. TMA and MMA
# operations accept TileTensors directly via overloaded methods.
#
# PEER TILE CREATION (offset pointer with same layout)
#
#    TileTensor requires both ptr and layout for construction:
#
#        var peer_tile = type_of(tile)(
#            tile.ptr + offset,
#            tile.layout,
#        )
#


# ============================================================================
# SMemTileArray - Array of tiles in shared memory (TileTensor-based)
# ============================================================================


# TODO: This type should correctly propagate mutability.
struct SMemTileArray2D[
    dtype: DType,
    dim0: Int,
    dim1: Int,
    num_tiles: Int,
    swizzle_bytes: Int = 128,
    alignment: Int = 128,
](TrivialRegisterPassable):
    """Array of TileTensor tiles in shared memory with swizzled K-major layout.

    The tiles use `internal_k_major` layout with configurable swizzle, matching
    the SM100 TMA swizzle pattern. This preserves swizzle information in the
    TileTensor type while using simple dimension-based parameters.

    Parameters:
        dtype: Tile element data type.
        dim0: First dimension (rows, e.g., BM or BN).
        dim1: Second dimension (columns, e.g., BK).
        num_tiles: Number of tiles in the array.
        swizzle_bytes: Swizzle size in bytes (128, 64, or 32). Must be > 0.
        alignment: Memory alignment (default 128 for shared memory).

    Note:
        For tiles without swizzle, use SMemTileArrayWithLayout with row_major.

    Example:
        comptime MyArray = SMemTileArray2D[.float16, 64, 32, 4, 128, 128]

        var array = MyArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor with swizzled layout
    """

    # The TileTensor-based tile type with swizzled layout for SM100
    # swizzle_bytes must be > 0; for row_major, use SMemTileArrayWithLayout
    comptime Tile = SMemTile[
        Self.dtype,
        internal_k_major[Self.dtype, Self.dim0, Self.dim1, Self.swizzle_bytes],
        alignment=Self.alignment,
    ]

    # Size calculations
    comptime tile_size: Int = Self.dim0 * Self.dim1
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = Array[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArray2D pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    def __init__(
        out self,
        # TODO: This should correctly propagate mutability
        unsafe_ptr: UnsafePointer[Scalar[Self.dtype], _, address_space=.SHARED],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = rebind[type_of(self.ptr)](unsafe_ptr)

    # The internal layout matching the Tile type
    comptime tile_layout = internal_k_major[
        Self.dtype, Self.dim0, Self.dim1, Self.swizzle_bytes
    ]

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Parameters:
            T: Index value type, must be convertible to `Int`.

        Args:
            index: The tile index.

        Returns:
            A TileTensor-based tile at the given index with swizzled layout.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return Self.Tile(
            tile_ptr.as_unsafe_any_origin(),
            Self.tile_layout,
        )

    @always_inline
    def get_with_layout[
        tile_layout: Layout, T: Intable
    ](self, index: T) -> SMemTile[
        Self.dtype, tile_layout, alignment=Self.alignment
    ]:
        """Get tile at the given index with a specified layout.

        This method allows getting tiles with a swizzled layout for MMA
        operations, where the layout information is needed for correct
        K-iteration offsets.

        Parameters:
            tile_layout: The layout to use (e.g., swizzled layout for MMA).
            T: Index type (must be Intable).

        Args:
            index: The tile index.

        Returns:
            A TileTensor with the specified layout at the given index.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return SMemTile[Self.dtype, tile_layout, alignment=Self.alignment](
            tile_ptr.as_unsafe_any_origin(), tile_layout
        )

    def slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray2D[
            Self.dtype, Self.dim0, Self.dim1, length, Self.alignment
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArray2D representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    def stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArray2D backed by stack-allocated shared memory.
        """
        var ptr = unsafe_stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment=Self.alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


# ============================================================================
# SMemTileArray2DRowMajor - TileTensor array with row_major layout (no swizzle)
# ============================================================================


# TODO: This type should correctly propagate mutability.
struct SMemTileArray2DRowMajor[
    dtype: DType,
    dim0: Int,
    dim1: Int,
    num_tiles: Int,
    alignment: Int = 128,
](TrivialRegisterPassable):
    """Array of TileTensor tiles in shared memory with row_major layout.

    Unlike SMemTileArray2D which uses swizzled internal_k_major layout, this
    type uses simple row_major layout. Suitable for 1D vectors (like A-scales)
    or output tiles where swizzling is not needed.

    Parameters:
        dtype: Tile element data type.
        dim0: First dimension (rows).
        dim1: Second dimension (columns).
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment (default 128 for shared memory).

    Example:
        comptime MyArray = SMemTileArray2DRowMajor[.float32, 1, 64, 4]

        var array = MyArray.stack_allocation()
        var tile = array[0]  # Returns TileTensor with row_major layout
    """

    # The TileTensor-based tile type with simple row_major layout
    comptime Tile = SMemTile[
        Self.dtype,
        row_major[Self.dim0, Self.dim1](),
        alignment=Self.alignment,
    ]

    # The internal layout matching the Tile type
    comptime tile_layout = row_major[Self.dim0, Self.dim1]()

    # Size calculations
    comptime tile_size: Int = Self.dim0 * Self.dim1
    comptime num_elements: Int = Self.tile_size * Self.num_tiles
    comptime storage_size: Int = Self.num_elements * size_of[Self.dtype]()

    # Storage type for stack allocation
    comptime Storage = Array[Scalar[Self.dtype], Self.num_elements]

    # Pointer to the array data
    var ptr: UnsafePointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from inline storage.

        Args:
            storage: The inline storage array.

        Returns:
            A new SMemTileArray2DRowMajor pointing to the storage.
        """
        return Self(storage.unsafe_ptr())

    def __init__(
        out self,
        # TODO: This should correctly propagate mutability
        unsafe_ptr: UnsafePointer[Scalar[Self.dtype], _, address_space=.SHARED],
    ):
        """Initialize with a shared memory pointer.

        Args:
            unsafe_ptr: Pointer to shared memory storage.
        """
        self.ptr = rebind[type_of(self.ptr)](unsafe_ptr)

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Parameters:
            T: Index value type, must be convertible to `Int`.

        Args:
            index: The tile index.

        Returns:
            A TileTensor-based tile at the given index with row_major layout.
        """
        var tile_ptr = self.ptr + Self.tile_size * Int(index)
        return Self.Tile(
            tile_ptr.as_unsafe_any_origin(),
            Self.tile_layout,
        )

    def slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray2DRowMajor[
            Self.dtype, Self.dim0, Self.dim1, length, Self.alignment
        ],
    ):
        """Get a slice of the array.

        Parameters:
            length: The length of the slice.

        Args:
            start: The starting index.

        Returns:
            A new SMemTileArray2DRowMajor representing the slice.
        """
        return type_of(result)(self.ptr + Self.tile_size * start)

    @always_inline
    @staticmethod
    def stack_allocation() -> Self:
        """Allocate the array on the stack (in shared memory).

        Returns:
            A new SMemTileArray2DRowMajor backed by stack-allocated shared memory.
        """
        var ptr = unsafe_stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment=Self.alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)
