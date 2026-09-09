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
"""Tensor Core Async Module.

This module provides high-performance abstractions for utilizing NVIDIA's Tensor Cores
to perform asynchronous matrix multiplication operations. It implements optimized memory
layouts and access patterns for efficient tensor core computations.

Key components:
- Layout creation functions for K-major and MN-major memory arrangements
- Swizzling support for improved memory access patterns
- WGMMA (Warp Group Matrix Multiply-Accumulate) descriptor generation
- TensorCoreAsync struct with methods for asynchronous matrix multiplication

The module supports various data types, matrix dimensions, and memory configurations,
enabling efficient implementation of deep learning primitives and other tensor operations
that can leverage hardware acceleration.

Performance features:
- Asynchronous execution model to overlap computation and memory access
- Support for different swizzling modes to optimize memory bandwidth
- Efficient register and shared memory utilization
- Support for multi-warp group execution

This implementation is specifically optimized for NVIDIA GPUs with Tensor Core support.
"""
from std.sys import size_of, bit_width_of
from std.sys._assembly import inlined_assembly

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout.coord import Coord, Idx
from layout import IntTuple, Layout, LayoutTensor, TileTensor
from layout.layout import (
    MakeLayoutList,
    composition,
    downcast,
    logical_divide,
    logical_product,
    make_layout,
    right_inverse,
    tile_to_shape,
    upcast,
)
from layout.tile_layout import Layout as TileLayout

from std.utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
# WGMMA shared memory layout                                                   #
# ===-----------------------------------------------------------------------===#
#
# TODO: add more context for WGMMA, core matrix. Assuming reader know them for now
#
#
# -------------------------------
# M/N x K, K-major, w/o swizzling
# -------------------------------
#
# Consider core matrix cm_M x cm_K, where cm_M = 8 and cm_K = 16 // size_of[type]()
#
#    !!!! core matrix one contiguous chunk in row-major(cm_M, cm_K) !!!!
#
# E.g.
#                 | core matrix 00 | core matrix 01 |
#                 | core matrix 10 | core matrix 11 |
#
# The elements are stored in shared memory as
#
#       | core matrix 00 | core matrix 10 | core matrix 01 | core matrix 11 |
#
# and each core matrix ij is a contiguous 128B with row_major(cm_M, cn_K) layout.
#
# The share memory tile is logically mapped to a BM x BK sub-matrix in global memory,
# Without swizzling, the tile layout is
#
#     ((cm_M,  BM // cm_M), (cm_K, BK // cm_K))
#   : ((cm_K, cm_M * cm_K), (   1, BM  * cm_K))
#
# coalesceable to (but we use the above like cutlass)
#
#     (  BM, (cm_K, BK // cm_K))
#   : (cm_K, (   1, BM  * cm_K))
#
#
# WGMMA descriptor layout:
#
# B16 => no swizzle
#
# B16  : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1,LBO)) where x = 1
# B32  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 2
# B64  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 4
# B128 : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((xT,SBO),(1, T )) where x = 8
#
# cm_M = cm_N = 8
# cm_K = T = 16 // size_of[type]()
# When there is swizzle, there is the swizzle mode constraint:
# `swizzle_mode.bytes() = 16x = 16 xT/T = 16 xT/(16 / size_of[type]()) = xT * size_of[type]().
#
# Tiled descriptors:
#
# B16  : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2k)):((T,SBO),(1,LBO))
#
# When the layout is dense, and the core matrices are tiled in column major
# like above comment. We have `SBO = cm_M * T = cm_M * cm_K` and `LBO = (cm_M *
# m) * T = BM * T = BM * cm_K`. The minimal dense layout is then `BM = 8` and
# `BK = T` which is exactly the core matrix layout.
#
#
# B32  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 2
# B64  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 4
# B128 : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2k)):((xT,SBO),(1, T )) where x = 8
#
# When the layout is dense, we have the unique solution `xT = T*2k = BK`, `SBO
# = cm_M * BK`. The minimal dense layout is then `m = 1` and `2k = x`.
#
# ----------------------------
# K x M/N, MN-major, siwzzling
# ----------------------------
#
# MN-major layouts are hard to reason. We port cutlass' three canonical
# layouts with some refactorization:
#
# B32   : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
# B64   : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
# B128  : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
#
# T = 16B // size_of[type]()
# m = BM  // (2T or 4T or 8T)
# k = BK  // 8
#
# We simplify them to
#
# B32   : Swizzle<1,4,3> o smem_ptr o ((2T,m),(8,k)):((1,LBO),(2T,SBO))
# B64   : Swizzle<2,4,3> o smem_ptr o ((4T,m),(8,k)):((1,LBO),(4T,SBO))
# B128  : Swizzle<3,4,3> o smem_ptr o ((8T,m),(8,k)):((1,LBO),(8T,SBO))
#
# `2/4/8 * T` is generalized as `swizzle.bytes() // size_of[type]()`.


@always_inline
def _supported_mma_shape[
    mma_shape: IndexList[3],
]() -> Bool:
    """Checks if a given MMA shape is supported for tensor core operations.

    This function validates the dimensions of the MMA shape against known tensor core
    operation constraints. It ensures that the shape is compatible with the expected
    dimensions for tensor core operations.

    Parameters:
        mma_shape: The shape of the MMA operation.

    Returns:
        `Bool` - True if the MMA shape is supported, False otherwise.
    """

    # Ideally this check should be input/output type dependent as mma_shape depends on input/output types
    # (https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmwgmmamma_async-nvvmwgmmammaasyncop).
    comptime if mma_shape[0] == 64 and mma_shape[2] == 8:
        return (
            mma_shape[1] % 8 == 0 and mma_shape[1] >= 8 and mma_shape[1] <= 256
        )
    elif mma_shape[0] == 64 and mma_shape[2] == 16:
        return (
            mma_shape[1] % 8 == 0 and mma_shape[1] >= 8 and mma_shape[1] <= 256
        )
    elif mma_shape[0] == 64 and mma_shape[2] == 32:
        return (
            mma_shape[1] % 8 == 0 and mma_shape[1] >= 8 and mma_shape[1] <= 256
        )
    else:
        return False


# Core matrix dimensions
# Each core matrix has 8 rows and 16 bytes per row.
comptime _CM_NUM_ROWS = 8


comptime _CM_ROW_BYTES = 16
comptime _CM_ROW_BITS = 128

# WGMMA's K dim has 32 bytes.
comptime WGMMA_K_BYTES = 32
"""Size of WGMMA K dimension in bytes."""

comptime _CM_LAYOUT_BITS = Layout.row_major(_CM_NUM_ROWS, _CM_ROW_BITS)
comptime _CM_TILE_STRIDE = IntTuple(1, _CM_ROW_BITS)


@always_inline
def warpgroup_fence[
    accum_type: DType,
    accum_layout: Layout,
    //,
](accum: LayoutTensor[accum_type, accum_layout, address_space=.LOCAL, ...]):
    """Code motion fence to ensure the registers of the WGMMA instruction do not get touched by anything.

    This has no impact on kernel correctness. It serves purely as an NVVM code motion barrier,
    preventing other operations from modifying the WGMMA instruction's
    registers during execution of the WGMMA instruction batch.

    Parameters:
        accum_type: Element data type of the tensor.
        accum_layout: Register layout of the accumulator.

    Args:
        accum: A LayoutTensor with the accum_type and accum_layout.

    """
    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for warpgroup fence"

    @always_inline
    def _warpgroup_fence_operand(reg: Scalar[accum_type]):
        inlined_assembly["", NoneType, constraints="+f", has_side_effect=True](
            reg
        )

    comptime for i in range(accum_layout.size()):
        _warpgroup_fence_operand(accum.ptr[i])


# constructs core matrix or "minimal dense" layout in bytes as described in file
# header.
def _select_k_atom_bits[
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    return Layout.row_major(_CM_NUM_ROWS, swizzle_mode.bytes() * 8)


def select_k_atom[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    """Creates a core matrix layout for tensor core operations.

    Constructs the fundamental atomic layout for tensor core operations based on the
    specified data type and swizzle mode. This layout represents the minimal dense
    matrix structure that can be efficiently processed by tensor cores.

    Parameters:
        dtype: Element data type of the tensor.
        swizzle_mode: Memory access pattern swizzling mode.

    Returns:
        `Layout` - A core matrix layout optimized for tensor core operations.
    """
    comptime a = _select_k_atom_bits[swizzle_mode]()
    return upcast(materialize[a](), bit_width_of[dtype]())


def _checked_tile_shape[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    BM: Int,
    BK: Int,
]() -> IntTuple:
    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime k_bytes = BK * size_of[dtype]()
        comptime assert (k_bytes % swizzle_mode.bytes()) == 0, (
            "K dim "
            + String(k_bytes)
            + " doesn't match "
            + String(swizzle_mode)
        )
        # swizzled WGMMA cannot be tiled in K if we constraint the layout to 2D.

    return [BM, BK]


# Helper: swizzle width in elements for a given dtype and swizzle mode.
comptime _sw_K[
    dtype: DType, swizzle_mode: TensorMapSwizzle
] = swizzle_mode.bytes() // size_of[dtype]()

# Outer stride for the K dimension in tile_layout_k_major_typed.
# When BK == sw_K (outer K dim has shape 1), the stride is 0 to match
# the compact layout produced by tile_to_shape / tile_layout_k_major().
comptime _outer_k_stride[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = 0 if BK == _sw_K[dtype, swizzle_mode] else BM * _sw_K[dtype, swizzle_mode]

# Outer stride for the M dimension in tile_layout_k_major_typed, following the
# same shape-1 rule as _outer_k_stride.
comptime _outer_m_stride[
    dtype: DType, BM: Int, swizzle_mode: TensorMapSwizzle
] = 0 if BM == _CM_NUM_ROWS else _CM_NUM_ROWS * _sw_K[dtype, swizzle_mode]


comptime tile_layout_k_major_typed[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
] = TileLayout(
    Coord(
        Coord(
            Idx[_CM_NUM_ROWS],
            Idx[BM // _CM_NUM_ROWS],
        ),
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[BK // _sw_K[dtype, swizzle_mode]],
        ),
    ),
    Coord(
        Coord(
            Idx[_sw_K[dtype, swizzle_mode]],
            Idx[_outer_m_stride[dtype, BM, swizzle_mode]],
        ),
        Coord(
            Idx[1],
            Idx[_outer_k_stride[dtype, BM, BK, swizzle_mode]],
        ),
    ),
)
"""K-major typed Layout for tensor core operations.

Shape ``((CM, BM/CM), (sw_K, BK/sw_K))``, stride ``((sw_K, CM*sw_K), (1, BM*sw_K))``
where CM=8 and sw_K = swizzle_mode.bytes() / sizeof(dtype).
An outer dimension of extent 1 carries stride 0 (compact), matching
`tile_layout_k_major()`.

Parameters:
    dtype: Element data type of the tensor.
    BM: Size of the M dimension in the tile.
    BK: Size of the K dimension in the tile.
    swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
"""


comptime tile_layout_mn_major_typed[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
] = tile_layout_k_major_typed[dtype, k_dim, mn_dim, swizzle_mode].transpose()
"""MN-major typed Layout for tensor core operations.

Equivalent to ``tile_layout_k_major_typed[dtype, k_dim, mn_dim, swizzle_mode].transpose()``.

Parameters:
    dtype: Element data type of the tensor.
    mn_dim: Size of the MN dimension.
    k_dim: Size of the K dimension.
    swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
"""


# Scale-factor atom geometry, mirrored from `tile_sf_layout_k_major`. The
# equivalence test keeps the two definitions in step.
comptime _SF_ATOM_M0 = 32
comptime _SF_ATOM_M1 = 4
comptime _SF_ATOM_K = 4
comptime _SF_MN_GROUP_SIZE = _SF_ATOM_M0 * _SF_ATOM_M1
comptime _SF_ATOM_SIZE = _SF_ATOM_M0 * _SF_ATOM_M1 * _SF_ATOM_K


def _checked_sf_mn_tiles[BM: Int]() -> Int:
    """Returns the number of scale-factor atoms along the MN dimension.

    Parameters:
        BM: Size of the M dimension in the tile.

    Returns:
        The atom count `BM / 128`.

    Constraints:
        BM must be a positive multiple of the 128-row scale-factor group.
    """
    comptime is_grouped = BM > 0 and BM % _SF_MN_GROUP_SIZE == 0
    comptime assert is_grouped, "BM must be a positive multiple of 128"
    return BM // _SF_MN_GROUP_SIZE


def _checked_sf_k_tiles[BK: Int, SF_SCALE_SIZE: Int]() -> Int:
    """Returns the number of scale-factor atoms along the K dimension.

    Parameters:
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.

    Returns:
        The atom count `BK / (4 * SF_SCALE_SIZE)`.

    Constraints:
        BK must be a positive multiple of `4 * SF_SCALE_SIZE`.
    """
    comptime k_group = _SF_ATOM_K * SF_SCALE_SIZE
    comptime is_grouped = BK > 0 and BK % k_group == 0
    comptime assert (
        is_grouped
    ), "BK must be a positive multiple of 4*SF_SCALE_SIZE"
    return BK // k_group


comptime _sf_mn_tiles[BM: Int] = _checked_sf_mn_tiles[BM]()

comptime _sf_k_tiles[BK: Int, SF_SCALE_SIZE: Int] = _checked_sf_k_tiles[
    BK, SF_SCALE_SIZE
]()

comptime _sf_mn_stride[
    BM: Int, BK: Int, SF_SCALE_SIZE: Int
] = 0 if _sf_mn_tiles[BM] == 1 else _sf_k_tiles[
    BK, SF_SCALE_SIZE
] * _SF_ATOM_SIZE

comptime _sf_k_stride[BK: Int, SF_SCALE_SIZE: Int] = 0 if _sf_k_tiles[
    BK, SF_SCALE_SIZE
] == 1 else _SF_ATOM_SIZE


comptime tile_sf_layout_k_major_typed[
    BM: Int,
    BK: Int,
    SF_SCALE_SIZE: Int,
] = TileLayout(
    Coord(
        Coord(
            Idx[_SF_ATOM_M0],
            Idx[_sf_mn_tiles[BM]],
        ),
        Coord(
            Coord(Idx[_SF_ATOM_M1], Idx[_SF_ATOM_K]),
            Idx[_sf_k_tiles[BK, SF_SCALE_SIZE]],
        ),
    ),
    Coord(
        Coord(
            Idx[_SF_ATOM_M1 * _SF_ATOM_K],
            Idx[_sf_mn_stride[BM, BK, SF_SCALE_SIZE]],
        ),
        Coord(
            Coord(Idx[1], Idx[_SF_ATOM_M1]),
            Idx[_sf_k_stride[BK, SF_SCALE_SIZE]],
        ),
    ),
)
"""K-major typed Layout for tensor core scale factors.

Closed form of ``tile_sf_layout_k_major()``: the ``(32, (4, 4))`` scale-factor
atom repeated over ``BM/128`` MN atoms and ``BK/(4*SF_SCALE_SIZE)`` K atoms,
K-atom-minor. An atom count of 1 carries stride 0, matching the compact layout
produced by `tile_to_shape`.

Unlike `tile_sf_layout_k_major()`, which silently drops a mode when an atom
count rounds down to 0, this rejects tile extents that do not cover a whole
number of atoms.

Parameters:
    BM: Size of the M dimension in the tile.
    BK: Size of the K dimension in the tile.
    SF_SCALE_SIZE: Number of elements in a scale factor vector.
"""


def tile_layout_k_major[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    page_dense: Bool = False,
]() -> Layout:
    """Creates a K-major layout for tensor core operations.

    Constructs a layout optimized for K-major access patterns in tensor core operations,
    with optional swizzling for improved memory access patterns.

    New code should prefer `tile_layout_k_major_typed`, which builds the same
    layout as a `TensorLayout` and only falls back to this legacy `Layout` at
    the boundary of an API that still requires one.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
        page_dense: When `True`, return the native chunk-inner (row-major atoms)
            k-major layout instead of the default chunk-outer form. The two are
            byte-distinct when `BK` spans more than one swizzle atom
            (`num_chunks >= 2`); the chunk-inner form lays each page contiguously
            so one TMA fills it (the SM100 row-major page-fold path, K-side
            analog of the `tile_layout_mn_major` `page_dense` branch).
            SWIZZLE_128B only.

    Returns:
        `Layout` - A K-major layout configured for the specified dimensions and swizzle mode.
    """
    comptime if page_dense:
        # Page-dense (chunk-inner, row-major atoms) k-major layout, proven by
        # `test_qk_kmajor_mma_spike.mojo`. Each `gran`-wide swizzle atom is
        # innermost (stride 1); atom-rows (the MN/`BM` axis) step by
        # `num_chunks * _CM_NUM_ROWS * gran` (OUTER); the chunk atoms within an
        # atom-row step by `_CM_NUM_ROWS * gran` (INNER). Each
        # `_CM_NUM_ROWS × gran` atom stays dense, so SWIZZLE_128B applies per
        # atom. Fed through `tile_to_descriptor` + `_create_mma_desc_pair` this
        # yields SBO = num_chunks*_CM_NUM_ROWS*gran*size_of (=2048 B, the page
        # stride, double the chunk-outer 1024) and LBO = 16 (unchanged).
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense k-major layout is only defined for SWIZZLE_128B"
        comptime gran = swizzle_mode.bytes() // size_of[dtype]()
        comptime num_chunks = BK // gran
        return Layout(
            [
                [_CM_NUM_ROWS, BM // _CM_NUM_ROWS],
                [gran, num_chunks],
            ],
            [
                [gran, num_chunks * _CM_NUM_ROWS * gran],
                [1, _CM_NUM_ROWS * gran],
            ],
        )
    else:
        comptime atom = select_k_atom[dtype, swizzle_mode]()
        comptime new_shape = _checked_tile_shape[dtype, swizzle_mode, BM, BK]()
        return tile_to_shape(materialize[atom](), new_shape)


def tile_sf_layout_k_major[
    BM: Int,
    BK: Int,
    SF_SCALE_SIZE: Int,
]() -> Layout:
    """Creates a K-major layout for tensor core scale factors.

    Constructs a layout for K-major access patterns for scale factors.

    Parameters:
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.

    Returns:
        `Layout` - A K-major layout configured for the specified dimensions and scale factor size.
    """

    comptime SF_ATOM_M = (32, 4)
    comptime SF_ATOM_K = 4
    comptime SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128

    comptime sf_atom = Layout(
        IntTuple(SF_ATOM_M[0], IntTuple(SF_ATOM_M[1], SF_ATOM_K)),
        IntTuple(SF_ATOM_M[1] * SF_ATOM_K, IntTuple(1, SF_ATOM_M[1])),
    )
    comptime sf_layout = tile_to_shape(
        sf_atom,
        [
            (BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0],
            (BK // (SF_ATOM_K * SF_SCALE_SIZE)) * (SF_ATOM_M[1] * SF_ATOM_K),
        ],
        IntTuple(2, 1),
    )
    return materialize[sf_layout]()


def tile_to_descriptor[
    dtype: DType,
    layout: Layout,
    is_k_major: Bool = True,
]() -> Layout:
    """Transforms a layout into a WGMMA descriptor-compatible layout.

    Converts a standard layout into a form that can be used with WGMMA descriptors,
    handling both K-major and MN-major layouts differently.

    Parameters:
        dtype: Element data type of the tensor.
        layout: Input layout to transform.
        is_k_major: Whether the layout is K-major (True) or MN-major (False).

    Returns:
        `Layout - A transformed layout compatible with WGMMA descriptors.
    """

    comptime if is_k_major:
        # Tile a layout to ((8,m),(T,2)) shape to match the K-major wgmma descriptor
        comptime T = _CM_ROW_BYTES // size_of[dtype]()
        comptime tiler = MakeLayoutList(Layout(_CM_NUM_ROWS), Layout(T))
        return logical_divide(materialize[layout](), materialize[tiler]())
    else:
        # We are not using atom layout for MN-major layouts.
        return materialize[layout]()


def tile_layout_mn_major[
    dtype: DType,
    mn_dim: Int,
    k_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    page_dense: Bool = False,
]() -> Layout:
    """Creates an MN-major layout for tensor core operations.

    Constructs a unit layout optimized for MN-major access patterns in shared memory,
    with optional swizzling for improved memory access patterns.

    New code should prefer `tile_layout_mn_major_typed`, which builds the same
    layout as a `TensorLayout` and only falls back to this legacy `Layout` at
    the boundary of an API that still requires one.

    Parameters:
        dtype: Element data type of the tensor.
        mn_dim: Size of the MN dimension.
        k_dim: Size of the K dimension.
        swizzle_mode: Memory access pattern swizzling mode (default: SWIZZLE_NONE).
        page_dense: When `True`, return the native chunk-inner (row-major atoms)
            layout instead of the transpose-of-k-major (chunk-outer) form. The
            two are byte-distinct when `k_dim` spans more than one swizzle atom;
            the chunk-inner form lays each page contiguously so one TMA fills it
            (the SM100 row-major page-fold path). SWIZZLE_128B only.

    Returns:
        `Layout` - An MN-major layout configured for the specified dimensions and swizzle mode.

    Note:
        This returns the "unit" layout; the actual shared memory layout can be a multiple of this unit.
        Currently only supports SWIZZLE_NONE and SWIZZLE_128B modes.
    """

    comptime if page_dense:
        # Native (pre-`#73811`) mn-major layout, recovered verbatim from
        # `fe239ba77a3`. Chunk-inner (row-major atoms): each `row_len`-wide
        # swizzle atom is innermost (stride 1); the mn axis steps by
        # `_CM_NUM_ROWS * row_len`; the k axis steps by `row_len` within a core
        # matrix and `_CM_NUM_ROWS * mn_dim` across core matrices. Each
        # `_CM_NUM_ROWS × row_len` atom stays dense (NOT the
        # swizzle-incompatible element-row-contiguous form).
        comptime assert (
            swizzle_mode == TensorMapSwizzle.SWIZZLE_128B
        ), "page_dense mn-major layout is only defined for SWIZZLE_128B"
        comptime row_len = swizzle_mode.bytes() // size_of[dtype]()
        return Layout(
            [
                [row_len, mn_dim // row_len],
                [_CM_NUM_ROWS, k_dim // _CM_NUM_ROWS],
            ],
            [
                [1, _CM_NUM_ROWS * row_len],
                [row_len, _CM_NUM_ROWS * mn_dim],
            ],
        )
    else:
        return tile_layout_k_major[
            dtype, k_dim, mn_dim, swizzle_mode
        ]().transpose()


def wgmma_c_thread_layout[C: Layout]() -> Layout:
    """Returns the thread layout component for WGMMA C matrix.

    Generates the first mode of the WGMMA C layout, which maps thread coordinates
    to linearized indices in the output matrix.

    Parameters:
        C: The layout of the C matrix.

    Returns:
        `Layout` - A layout mapping thread coordinates to linearized indices.
    """
    return Layout(
        [4, 8, 4],
        [comptime (C([0, 2])), comptime (C([1, 0])), comptime (C([16, 0]))],
    )


def wgmma_output_layout[mma_n: Int, C: Layout]() -> Layout:
    """Returns the output layout component for WGMMA C matrix.

    Generates the second mode of the WGMMA C layout, which maps output vector
    coordinates to linearized indices in the output matrix.

    Parameters:
        mma_n: The N dimension of the WGMMA instruction.
        C: The layout of the C matrix.

    Returns:
        `Layout` - A layout mapping output vector coordinates to linearized indices.
    """
    return Layout(
        [2, 2, mma_n // 8],
        [comptime (C([0, 1])), comptime (C([8, 0])), comptime (C([0, 8]))],
    )


def wgmma_c_layout[mma_m: Int, mma_n: Int, C: Layout]() -> List[Layout]:
    """Generates three layouts for mapping WGMMA C matrix coordinates.

    This function creates three layout mappings that are essential for working with WGMMA
    (Warp Group Matrix Multiply-Accumulate) operations:

    1. A projection layout that maps linearized indices to row coordinates (i)
    2. A projection layout that maps linearized indices to column coordinates (j)
    3. A composite layout that maps thread and vector coordinates to linearized indices
       across multiple MMA tiles

    These layouts are particularly useful for operations like attention masking and
    matrix multiplication epilogues, where register values need to be mapped to the
    coordinate system of the C matrix.

    Parameters:
        mma_m: The M dimension (rows) of a single WGMMA instruction, must be 64.
        mma_n: The N dimension (columns) of a single WGMMA instruction, must be multiple of 8.
        C: The layout of the C matrix within a thread block.

    Returns:
        `List[Layout]` - A list containing three layouts:
            1. proj_i: Maps linearized indices to row coordinates
            2. proj_j: Maps linearized indices to column coordinates
            3. TV_tile_to_idx: Maps thread/vector/tile coordinates to linearized indices

    Note:
        This function enforces constraints on the WGMMA dimensions and ensures the C matrix
        dimensions are compatible with the WGMMA instruction size.
    """
    comptime err = "C = " + String(C) + ", mma_m = " + String(
        mma_m
    ) + ", mma_n = " + String(mma_n)
    comptime assert mma_m == 64, err
    comptime assert mma_n % 8 == 0, err
    comptime M = C.shape[0].value()
    comptime N = C.shape[1].value()
    comptime assert M % mma_m == 0, err
    comptime assert N % mma_n == 0, err
    comptime num_m_mma = M // mma_m
    comptime num_n_mma = N // mma_n
    # idx -> col(i, j)
    comptime inv_c = right_inverse(C)
    # idx -> col(i, j) -> i
    comptime proj_i = composition(Layout([M, N], [1, 0]), inv_c)
    # idx -> col(i, j) -> j
    comptime proj_j = composition(Layout([M, N], [0, 1]), inv_c)
    # ((lane_j, lane_i, warp_id), (vec_12, value_i, value_j)) -> idx
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png
    comptime T_to_idx = wgmma_c_thread_layout[C]()
    comptime V_to_idx = wgmma_output_layout[mma_n, C]()
    comptime TV_to_idx = make_layout(T_to_idx, V_to_idx)
    comptime tiler = Layout.col_major(num_m_mma, num_n_mma)
    comptime TV_tile_to_idx = logical_product(TV_to_idx, tiler)
    return [
        materialize[proj_i](),
        materialize[proj_j](),
        materialize[TV_tile_to_idx](),
    ]


def st_matrix_n_atom[num_stmatrix: Int]() -> Layout:
    """Creates a layout for N-major `st_matrix` atom in the context of WGMMA C
    matrix.

    The domain of this layout is the warp group local thread index. Thus, the
    layout takes [0, 128) as input and returns an offset for a logical array
    with an element size of 128-bit.

    Parameters:
        num_stmatrix: Number of N-dimension tiles in the C matrix.

    Returns:
        `Layout` - A layout that maps warp group local thread index to an offset
        for a logical array with an element size of 128-bit.
    """
    # C with the granularity of 128-bit per element
    comptime C = Layout.row_major(64, 2 * num_stmatrix)
    return Layout(
        [16, 2, 4],
        [comptime (C([1, 0])), comptime (C([0, 1])), comptime (C([16, 0]))],
    )


def st_matrix_m_atom[num_stmatrix: Int, num_consumer: Int]() -> Layout:
    """Creates a layout for M-major `st_matrix` atom in the context of WGMMA C
    matrix.

    The domain of this layout is the warp group local thread index. Thus, the
    layout takes [0, 128) as input and returns an offset for a logical array
    with an element size of 128-bit.

    Assume num_consumer = 2, and num_stmatrix = 2 then a single atom for one warp would look like this
    Each block contains the thread_idx, each thread idx will hold the address of the next 128-bit fragment.

    |  0  |  8  |
    |  1  |  9  |
    |  2  | 10  |
    | ... | ... |
    |  7  | 15  |

    | 16  | 24  |
    | 17  | 25  |
    | 18  | 26  |
    | ... | ... |
    | 23  | 31  |

    All 4 warps in the warp group will then be laid out next to each other

    |  w1  |  w2  | w3  | w4  |

    Parameters:
        num_stmatrix: Number of N-dimension tiles in the C matrix.
        num_consumer: Number of consumers.

    Returns:
        `Layout` - A layout that maps warp group local thread index to an offset
        for a logical array with an element size of 128-bit.
    """
    # C with the granularity of 128-bit per element
    comptime C = Layout.row_major(2 * num_stmatrix, 8 * num_consumer)
    return Layout(
        [8, 2, 2, 4],
        [
            comptime (C([1, 0])),
            comptime (C([0, 1])),
            comptime (C([8, 0])),
            comptime (C([0, 2])),
        ],
    )


def st_matrix_n_layout[
    c_type: DType, WG_BN: Int, num_m_mmas: Int, num_consumer: Int
]() -> Layout:
    """Creates a layout for N-major `st_matrix` in the context of WGMMA C
    matrix.

    The layout modes are: the warp group local thread index, the N-dimension
    tiling size `WG_BN // 16`, the number of MMA tiles `num_m_mmas` in the
    M-dimension, and the number of consumers `num_consumer`. The output is an
    offset for a logical array with the element type `c_type`.

    Parameters:
        c_type: Data type of the C matrix.
        WG_BN: Size of the K dimension in the C matrix in shared memory.
        num_m_mmas: Number of MMA tiles in the M dimension.
        num_consumer: Number of consumers.

    Returns:
        `Layout` - A layout that maps warp group local thread index to an offset
        for a logical array with the element type `c_type`.
    """
    comptime n_stmatrix = WG_BN // 16
    comptime atom = st_matrix_n_atom[n_stmatrix]()
    comptime b128_layout = logical_product(
        atom, Layout.col_major(n_stmatrix, num_m_mmas, num_consumer)
    )
    return downcast(materialize[b128_layout](), 128 // (8 * size_of[c_type]()))


def st_matrix_m_layout[
    c_type: DType, WG_BM: Int, num_m_mmas: Int, num_consumer: Int
]() -> Layout:
    """Creates a layout for M-major `st_matrix` in the context of WGMMA C
    matrix. This meant to be used with swapAB, since the C
    matrix must be transposed during the write phase. This must also be used
    in conjunction with st_matrix transposed modifier.

    The M-dimension tiling size `WG_BM // 16`, the number of MMA tiles `num_m_mmas`
    in the N-dimension, and the number of consumers `num_consumer`. The output is an
    offset for a logical array with the element type `c_type`.

    Parameters:
        c_type: Data type of the C matrix.
        WG_BM: Size of the K dimension in the C matrix in shared memory.
        num_m_mmas: Number of MMA tiles in the M dimension.
        num_consumer: Number of consumers.

    Returns:
        `Layout` - A layout that maps warp group local thread index to an offset
        for a logical array with the element type `c_type`.
    """
    comptime n_stmatrix = WG_BM // 16
    comptime atom = st_matrix_m_atom[n_stmatrix, num_consumer]()
    comptime b128_layout = logical_product(
        atom, Layout.row_major(n_stmatrix, num_m_mmas, num_consumer)
    )

    return downcast(materialize[b128_layout](), 128 // (8 * size_of[c_type]()))


def _wgmma_descriptor[
    dtype: DType,
    //,
    layout: Layout,
    is_k_major: Bool = True,
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    addr: ImmPointer[Scalar[dtype], address_space=.SHARED, ...]
) -> WGMMADescriptor[dtype]:
    # Conform to canonical layout.
    comptime assert (
        layout.rank() == 2 and layout[0].rank() == 2 and layout[1].rank() == 2
    ), "shared memory tile layout should have structure (rank-2, rank-2)."

    comptime shape00 = layout[0].shape[0].value()
    comptime shape11 = layout[1].shape[1].value()
    comptime stride01 = layout[0].stride[1].value()
    comptime stride11 = layout[1].stride[1].value()

    comptime if is_k_major:
        comptime assert (
            shape00 == 8 and shape11 % 2 == 0
        ), "Tile shape must be ((8, _), (_, multiple of 2)), get " + String(
            layout
        )

        # Ignore 4 LSB.
        comptime SBO = (stride01 * size_of[dtype]()) >> 4
        comptime LBO = (stride11 * size_of[dtype]()) >> 4

        return WGMMADescriptor.create[SBO, LBO, swizzle](addr)

    comptime no_swizzle = swizzle == TensorMapSwizzle.SWIZZLE_NONE

    # Swizzle and non-swizzle modes switch SBO and LBO based on
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=bar%2520sync#asynchronous-warpgroup-level-majorness-supported-by-strides
    comptime SBO = (
        (stride01 if no_swizzle else stride11) * size_of[dtype]()
    ) >> 4
    comptime LBO = (
        (stride11 if no_swizzle else stride01) * size_of[dtype]()
    ) >> 4

    return WGMMADescriptor.create[SBO, LBO, swizzle](addr)


def _lhs_descriptor[
    dtype: DType,
    layout: Layout,
    //,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[dtype, layout, address_space=.SHARED, ...]
) -> WGMMADescriptor[tensor.dtype]:
    comptime BM = layout[0].size()
    comptime BK = layout[1].size()
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    comptime canonical_layout_flat = tile_layout_k_major[
        dtype, BM, canonical_K, swizzle_mode
    ]()
    comptime canonical_layout = tile_to_descriptor[
        dtype, canonical_layout_flat, True
    ]()
    return _wgmma_descriptor[
        layout=canonical_layout, is_k_major=True, swizzle=swizzle_mode
    ](tensor.ptr)


def _rhs_descriptor[
    dtype: DType,
    layout: Layout,
    //,
    transposed: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    tensor: LayoutTensor[dtype, layout, address_space=.SHARED, ...]
) -> WGMMADescriptor[tensor.dtype]:
    comptime BN = layout[0].size()
    comptime BK = layout[1].size()
    comptime canonical_K = swizzle_mode.bytes() // size_of[
        dtype
    ]() if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE else BK
    comptime canonical_layout_flat = tile_layout_k_major[
        dtype, BN, canonical_K, swizzle_mode
    ]() if transposed else layout
    comptime canonical_layout = tile_to_descriptor[
        dtype, canonical_layout_flat, transposed
    ]()
    return _wgmma_descriptor[
        layout=canonical_layout, is_k_major=transposed, swizzle=swizzle_mode
    ](tensor.ptr)


# TODO(KERN-1301): Layouts are calculated for 64x8x8 instruction
def _output_register_size[mma_shape: IndexList[3]]() -> Int:
    comptime assert _supported_mma_shape[mma_shape](), (
        "WGMMA operation of shape '" + String(mma_shape) + "' is not supported"
    )
    return mma_shape[0] * mma_shape[1] // 128


@always_inline
def _convert_cfrags_to_tuple[
    c_type: DType, c_frag_size: Int
](
    c_frags: LayoutTensor[c_type, _, address_space=.LOCAL, ...],
) -> StaticTuple[
    Scalar[c_type], c_frag_size
]:
    var c_frags_in_tuple = StaticTuple[Scalar[c_type], c_frag_size]()

    comptime for i in range(c_frag_size):
        c_frags_in_tuple[i] = rebind[Scalar[c_type]](c_frags[0, i])

    return c_frags_in_tuple


@always_inline
def _convert_cfrags_to_simd[
    c_type: DType, c_frag_size: Int
](
    c_frags_in_tuple: StaticTuple[Scalar[c_type], c_frag_size],
    c_frags: LayoutTensor[mut=True, c_type, _, address_space=.LOCAL, ...],
):
    comptime for i in range(c_frag_size):
        c_frags[0, i] = c_frags_in_tuple[i]


struct TensorCoreAsync[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    mma_shape: IndexList[3],
    /,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = False,
](Defaultable):
    """High-performance asynchronous tensor core operations for matrix multiplication.

    This struct provides methods for utilizing NVIDIA's Tensor Cores for asynchronous
    matrix multiplication operations, with support for various data types and swizzling
    configurations.

    Parameters:
        c_type: Data type of the output matrix C.
        a_type: Data type of the input matrix A.
        b_type: Data type of the input matrix B.
        mma_shape: Dimensions for the matrix multiply-accumulate (MMA) operation as [M, N, K].
        a_swizzle: Swizzling mode for matrix A (default: SWIZZLE_NONE).
        b_swizzle: Swizzling mode for matrix B (default: SWIZZLE_NONE).
        transpose_b: Whether to transpose matrix B (default: False).
    """

    @always_inline
    def __init__(out self):
        """Initialize the `TensorCoreAsync` instance.

        Ensures that the provided MMA shape is supported.

        Note:
            Fails to compile if `mma_shape` is not supported.
        """
        comptime assert _supported_mma_shape[Self.mma_shape](), (
            "WGMMA operation of shape '"
            + String(Self.mma_shape)
            + "' is not supported"
        )

    @staticmethod
    @always_inline
    def wgmma[
        num_warp_groups: Int = 1,
        scale_c: Int = 1,
        scale_a: Int = 1,
        scale_b: Int = 1,
        num_k_iters: Optional[Int] = None,
    ](
        a_smem_tile: LayoutTensor[
            Self.a_type, _, _, address_space=.SHARED, ...
        ],
        b_smem_tile: LayoutTensor[
            Self.b_type, _, _, address_space=.SHARED, ...
        ],
        c_reg_tile: LayoutTensor[
            mut=True,
            Self.c_type,
            _,
            _,
            address_space=.LOCAL,
            ...,
        ],
        wg_idx: Int = 0,
    ):
        """Perform asynchronous matrix multiplication using warp group matrix multiply-accumulate (WGMMA).

        This method handles the case where both A and B matrices are in shared memory.

        Parameters:
            num_warp_groups: Number of warp groups to distribute work across (default: 1).
            scale_c: Scale factor for matrix C. Valid values are 1 or 0 (default: 1).
            scale_a: Scale factor for matrix A. Valid values are 1 or -1 (default: 1).
            scale_b: Scale factor for matrix B. Valid values are 1 or -1 (default: 1).
            num_k_iters: Number of iterations for the K dimension. This is useful to save computation when we pad shared memory. (default: None which is just `a_smem_layout[1].size() // mma_shape[2]`).

        Args:
            a_smem_tile: Matrix A in shared memory.
            b_smem_tile: Matrix B in shared memory.
            c_reg_tile: Output matrix C in register memory.
            wg_idx: Warp group index for multi-warp group scenarios (default: 0).
        """
        comptime assert scale_c == 1 or scale_c == 0
        comptime assert scale_a == 1 or scale_a == -1
        comptime assert scale_b == 1 or scale_b == -1
        comptime a_smem_layout = a_smem_tile.layout
        comptime b_smem_layout = b_smem_tile.layout

        # TODO: refactor once atom layout is simplified
        comptime BM = a_smem_layout[0].size()
        comptime BN = b_smem_layout[0].size()
        comptime BK = a_smem_layout[1].size()

        # Canonical layouts conform to WGMMA's layout requirement e.g.
        # K-major layout requires BK = swizzle.bytes() // size_of[T}().
        comptime a_canonical_K = Self.a_swizzle.bytes() // size_of[
            Self.a_type
        ]() if Self.a_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        comptime a_canonical_layout_flat = tile_layout_k_major[
            Self.a_type, BM, a_canonical_K, Self.a_swizzle
        ]()
        comptime a_canonical_layout = tile_to_descriptor[
            Self.a_type, a_canonical_layout_flat, True
        ]()
        comptime b_canonical_K = Self.b_swizzle.bytes() // size_of[
            Self.b_type
        ]() if Self.b_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        comptime b_canonical_layout_flat = tile_layout_k_major[
            Self.b_type, BN, b_canonical_K, Self.b_swizzle
        ]() if Self.transpose_b else b_smem_layout
        comptime b_canonical_layout = tile_to_descriptor[
            Self.b_type, b_canonical_layout_flat, Self.transpose_b
        ]()

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        # TODO: use layout algebra like `tile_to_shape` here.
        comptime a_shape00 = a_canonical_layout[0].shape[0].value()
        comptime a_stride01 = a_canonical_layout[0].stride[1].value()
        comptime a_stride11 = a_canonical_layout[1].stride[1].value()
        comptime b_shape00 = b_canonical_layout[0].shape[0].value()
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        comptime assert Self.mma_shape[0] % a_shape00 == 0
        comptime assert Self.mma_shape[1] % b_shape00 == 0

        # fmt: off
        # Strides between WGMMA tiles
        comptime a_m_stride = a_stride01 * (Self.mma_shape[0] // a_shape00) * size_of[Self.a_type]()
        comptime b_n_stride = b_stride01 * (Self.mma_shape[1] // b_shape00) * size_of[Self.b_type]()
        # K dim is stepped by 2 core matrices.
        comptime a_k_stride = a_stride11 * 2 * size_of[Self.a_type]()
        comptime b_k_stride = b_stride11 * 2 * size_of[Self.b_type]()

        comptime num_m_mmas = a_canonical_layout[0].size() // Self.mma_shape[0] // num_warp_groups
        comptime num_n_mmas = b_canonical_layout[0].size() // Self.mma_shape[1]
        comptime num_k_mmas = num_k_iters.or_else(a_smem_layout[1].size() // Self.mma_shape[2])

        # Number of wgmma per canonical layout. There can be multiple canonical layouts
        # per K dim e.g. BF16 128B swizzle has BK = 64 while input K = 128.
        comptime a_num_k_mmas_per_tile = a_canonical_K // Self.mma_shape[2]
        comptime b_num_k_mmas_per_tile = b_canonical_K // Self.mma_shape[2] if Self.transpose_b else num_k_mmas
        # fmt: on

        var a_desc = _wgmma_descriptor[
            a_canonical_layout, True, Self.a_swizzle
        ](a_smem_tile.ptr)
        var b_desc = _wgmma_descriptor[
            b_canonical_layout, Self.transpose_b, Self.b_swizzle
        ](b_smem_tile.ptr)

        comptime if num_warp_groups > 1:
            a_desc += a_m_stride * num_m_mmas * wg_idx

        comptime layout_b = "col" if Self.transpose_b else "row"
        comptime c_frag_size = Self.mma_shape[0] * Self.mma_shape[1] // 128

        comptime for k_mma in range(num_k_mmas):
            comptime scale_d = scale_c if k_mma == 0 else 1

            # Offsets when K is multiple of canonical layouts.
            comptime a_offset_bytes = (
                k_mma // a_num_k_mmas_per_tile
            ) * a_canonical_layout.size() * size_of[Self.a_type]()
            comptime b_offset_bytes = (
                k_mma // b_num_k_mmas_per_tile
            ) * b_canonical_layout.size() * size_of[
                Self.b_type
            ]() if Self.transpose_b else 0

            comptime a_k_mma_offset = (
                k_mma % a_num_k_mmas_per_tile
            ) * a_k_stride
            comptime b_k_mma_offset = (
                k_mma % b_num_k_mmas_per_tile
            ) * b_k_stride

            comptime for m_mma in range(num_m_mmas):
                comptime a_offset = m_mma * a_m_stride + a_k_mma_offset + a_offset_bytes
                var a_desc_m = a_desc + a_offset

                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    comptime b_offset = n_mma * b_n_stride + b_k_mma_offset + b_offset_bytes
                    var b_desc_n = b_desc + b_offset

                    var c_frags = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

                    var c_frags_in_tuple = _convert_cfrags_to_tuple[
                        Self.c_type, c_frag_size
                    ](c_frags)

                    var c_frags_out_tuple = wgmma_async[
                        Self.mma_shape[0],
                        Self.mma_shape[1],
                        Self.mma_shape[2],
                        a_type=Self.a_type,
                        b_type=Self.b_type,
                        layout_b=layout_b,
                        scale_d=scale_d,
                        scale_a=scale_a,
                        scale_b=scale_b,
                    ](a_desc_m, b_desc_n, c_frags_in_tuple)

                    _convert_cfrags_to_simd[Self.c_type, c_frag_size](
                        c_frags_out_tuple, c_frags
                    )

    @staticmethod
    @always_inline
    def wgmma[
        num_warp_groups: Int = 1,
        scale_c: Int = 1,
        scale_a: Int = 1,
        scale_b: Int = 1,
        num_k_iters: Optional[Int] = None,
    ](
        a_smem_tile: TileTensor[
            mut=True, Self.a_type, address_space=.SHARED, ...
        ],
        b_smem_tile: TileTensor[
            mut=True, Self.b_type, address_space=.SHARED, ...
        ],
        c_reg_tile: LayoutTensor[
            mut=True,
            Self.c_type,
            _,
            _,
            address_space=.LOCAL,
            ...,
        ],
        wg_idx: Int = 0,
    ):
        """Perform asynchronous matrix multiplication with TileTensor inputs.

        This overload handles the case where both A and B matrices are in
        shared memory as TileTensor values. It converts at this boundary so
        SM90 kernels can stay TileTensor-native while the WGMMA implementation
        continues to reuse the legacy descriptor path internally.

        Parameters:
            num_warp_groups: Number of warp groups to distribute work across.
            scale_c: Scale factor for matrix C. Valid values are 1 or 0.
            scale_a: Scale factor for matrix A. Valid values are 1 or -1.
            scale_b: Scale factor for matrix B. Valid values are 1 or -1.
            num_k_iters: Number of K-dimension iterations to execute.

        Args:
            a_smem_tile: Matrix A tile in shared memory.
            b_smem_tile: Matrix B tile in shared memory.
            c_reg_tile: Accumulator tile in register memory.
            wg_idx: Warp group index for multi-warp-group execution.
        """
        Self.wgmma[
            num_warp_groups,
            scale_c,
            scale_a,
            scale_b,
            num_k_iters,
        ](
            a_smem_tile.to_layout_tensor(),
            b_smem_tile.to_layout_tensor(),
            c_reg_tile,
            wg_idx,
        )

    @staticmethod
    @always_inline
    def wgmma(
        a_frag_tile: LayoutTensor[Self.a_type, _, address_space=.LOCAL, ...],
        b_smem_tile: LayoutTensor[Self.b_type, _, address_space=.SHARED, ...],
        c_reg_tile: LayoutTensor[
            mut=True,
            Self.c_type,
            _,
            address_space=.LOCAL,
            ...,
        ],
    ):
        """Perform asynchronous matrix multiplication using warp group matrix multiply-accumulate (WGMMA).

        This overloaded method handles the case where matrix A is in register memory and matrix B
        is in shared memory.

        Args:
            a_frag_tile: Matrix A in register memory.
            b_smem_tile: Matrix B in shared memory.
            c_reg_tile: Output matrix C in register memory.
        """
        comptime BN = b_smem_layout[0].size()
        comptime BK = b_smem_layout[1].size()
        comptime b_smem_layout = b_smem_tile.layout
        comptime b_canonical_K = Self.b_swizzle.bytes() // size_of[
            Self.b_type
        ]() if Self.b_swizzle != TensorMapSwizzle.SWIZZLE_NONE else BK
        comptime b_canonical_layout_flat = tile_layout_k_major[
            Self.b_type, BN, b_canonical_K, Self.b_swizzle
        ]() if Self.transpose_b else b_smem_layout
        comptime b_canonical_layout = tile_to_descriptor[
            Self.b_type, b_canonical_layout_flat, Self.transpose_b
        ]()

        # Layout modes are always (MN, K) transpose or not.
        # Note that shape00 may not equal core matrix dim for MN-major layouts.
        # TODO: use layout algebra like `tile_to_shape` here.
        comptime b_shape00 = b_canonical_layout[0].shape[0].value()
        comptime b_stride01 = b_canonical_layout[0].stride[1].value()
        comptime b_stride11 = b_canonical_layout[1].stride[1].value()
        # Strides between WGMMA tiles
        comptime assert Self.mma_shape[1] % b_shape00 == 0, (
            "b_shape00 = "
            + String(b_shape00)
            + ", mma_shape[1] = "
            + String(Self.mma_shape[1])
        )
        # fmt: off
        comptime b_n_stride = b_stride01 * (Self.mma_shape[1] // b_shape00) * size_of[Self.b_type]()
        # K dim is stepped by 2 core matrices.
        comptime b_k_stride = b_stride11 * 2 * size_of[Self.b_type]()
        comptime assert b_k_stride > 0

        comptime num_n_mmas = b_smem_layout[0].size() // Self.mma_shape[1]
        comptime num_k_mmas = b_smem_layout[1].size() // Self.mma_shape[2]
        comptime num_m_mmas = a_frag_tile.layout[0].shape[0].value() // num_k_mmas

        comptime b_num_k_mmas_per_tile = b_canonical_K // Self.mma_shape[2] if Self.transpose_b else num_k_mmas
        # fmt: on

        comptime assert b_n_stride > 0 or (
            b_n_stride == 0 and num_n_mmas == 1
        ), "b_smem_layout = " + String(b_smem_layout)

        # Vectorize each wgmma's fragment size.
        comptime a_frag_size = Self.mma_shape[0] * Self.mma_shape[2] // 128
        comptime c_frag_size = Self.mma_shape[0] * Self.mma_shape[1] // 128
        var a_frags = a_frag_tile.vectorize[1, a_frag_size]()
        var c_frags = c_reg_tile.vectorize[1, c_frag_size]()
        comptime assert (
            type_of(c_frags).layout.size() == num_m_mmas * num_n_mmas
        ), (
            "C fragments' size: "
            + String(type_of(c_frags).layout.size())
            + "\nDoesn't match the total number of wgmmas\n= num_m_mmas *"
            " num_n_mmas: "
            + String(num_m_mmas)
            + " * "
            + String(num_n_mmas)
            + ".\na_frag_tile.layout[0].shape[0].value() = "
            + String(a_frag_tile.layout[0].shape[0].value())
            + "\nnum_k_mmas = "
            + String(num_k_mmas)
            + "\nb_smem_layout = "
            + String(b_smem_layout)
        )

        var b_desc = _wgmma_descriptor[
            b_canonical_layout, Self.transpose_b, Self.b_swizzle
        ](b_smem_tile.ptr)
        comptime layout_b = "col" if Self.transpose_b else "row"

        comptime for k_mma in range(num_k_mmas):
            comptime b_offset_bytes = (
                k_mma // b_num_k_mmas_per_tile
            ) * b_canonical_layout.size() * size_of[
                Self.b_type
            ]() if Self.transpose_b else 0
            comptime b_k_mma_offset = (
                k_mma % b_num_k_mmas_per_tile
            ) * b_k_stride

            comptime for m_mma in range(num_m_mmas):
                var a_frag = a_frags[m_mma + k_mma * num_m_mmas, 0]

                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # a_desc_m = a_desc + m_mma * a_m_stride + k_mma * a_k_stride
                    comptime offset = n_mma * b_n_stride + b_k_mma_offset + b_offset_bytes
                    var b_desc_n = b_desc + offset

                    c_frags[mma_id, 0] = wgmma_async[
                        Self.mma_shape[0],
                        Self.mma_shape[1],
                        Self.mma_shape[2],
                        a_type=Self.a_type,
                        b_type=Self.b_type,
                        layout_b=layout_b,
                    ](
                        a_frag,
                        b_desc_n,
                        c_frags[mma_id, 0],
                    )

    @staticmethod
    @always_inline
    def arrive():
        """Ensures memory consistency by creating a fence for WGMMA operations.

        This method should be called before committing a group to ensure all
        shared memory accesses are properly aligned and visible.
        """
        wgmma_fence_aligned()

    @staticmethod
    @always_inline
    def commit_group():
        """Commits the current warp group for execution.

        This synchronizes the warp group and commits all pending WGMMA operations
        that have been previously issued.
        """
        wgmma_commit_group_sync()

    @staticmethod
    @always_inline
    def wait_group[group: Int = 0]():
        """Waits for the completion of a specific warp group's operations.

        This method blocks until all WGMMA operations from the specified group are complete.

        Parameters:
            group: The group ID to wait for (default: 0).
        """
        wgmma_wait_group_sync[group]()
