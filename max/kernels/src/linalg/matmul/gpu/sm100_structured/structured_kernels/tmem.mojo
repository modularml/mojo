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
"""Tensor Memory (TMEM) abstractions for SM100 Blackwell GPUs.

TMEM is dedicated memory for MMA accumulators, separate from registers and
shared memory. This module provides type-safe abstractions:

- TmemAllocation: Manages TMEM lifecycle (alloc/dealloc)
- TmemTensor: Layout-parameterized typed view over TMEM accumulators
- TmemStage: Represents a pipeline stage for accumulator buffering
- TmemAddress: Simple address wrapper for TMEM load operations
"""

from layout import Layout
from std.sys import size_of

from max.gpu.sync import syncwarp
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)

from max.gpu.primitives.cluster import block_rank_in_cluster
from layout.tma_async import SharedMemBarrier
from linalg.structuring import SMemArray
from structured_kernels.pipeline import SM100_PIPELINE_WAIT_TICKS
from .config import OutputPipelineConfig


struct TmemAllocation[
    cta_group: Int,
    max_cols: Int = 512,
](TrivialRegisterPassable):
    """Handle to allocated Tensor Memory.

    Lifecycle: allocate() → use → release_lock() → wait → deallocate()

    Parameters:
        cta_group: Cooperating CTAs (1 or 2).
        max_cols: TMEM columns (512 for SM100).
    """

    comptime SmemAddrStorage = SMemArray[UInt32, 1]

    var addr: UInt32

    @always_inline
    def __init__(out self, addr: UInt32):
        self.addr = addr

    @staticmethod
    @always_inline("nodebug")
    def allocate(smem_addr: Self.SmemAddrStorage) -> Self:
        """Allocate TMEM (MMA warp). Address stored in smem for epilogue.

        Args:
            smem_addr: Shared memory slot that receives the allocated
                TMEM address for cross-warp sharing with the epilogue.
        """
        tcgen05_alloc[Int32(Self.cta_group)](
            smem_addr.ptr, UInt32(Self.max_cols)
        )
        syncwarp()
        return Self(smem_addr.ptr[0])

    @staticmethod
    @always_inline
    def from_shared(smem_addr: Self.SmemAddrStorage) -> Self:
        """Get handle from existing allocation (epilogue warp).

        Args:
            smem_addr: Shared memory slot holding the TMEM address
                previously written by `allocate`.
        """
        return Self(smem_addr.ptr[0])

    @always_inline
    def release_lock(self):
        """Release allocation lock before waiting for epilogue."""
        tcgen05_release_allocation_lock[Int32(Self.cta_group)]()

    @always_inline
    def deallocate(self):
        """Free TMEM after epilogue completion."""
        tcgen05_dealloc[Int32(Self.cta_group)](self.addr, UInt32(Self.max_cols))


# TMEM Address Encoding (SM100 Blackwell)
# =========================================
# TMEM addresses encode row and column offsets in a packed format:
#
#   Address = [row_offset : 16 bits] [column_offset : 16 bits]
#
# SM100 MMA accumulators span 32 rows × N columns per tile:
#   - Upper fragment: rows 0-15  (accessed at base address)
#   - Lower fragment: rows 16-31 (accessed at base + TMEM_LOWER_ROW_OFFSET)
#
# The value 16 << 16 encodes "row 16, column 0" as the starting offset
# for the lower half of the accumulator.
comptime TMEM_LOWER_ROW_OFFSET: UInt32 = 16 << 16


struct TmemAddress(TrivialRegisterPassable):
    """Simple TMEM address wrapper for load/store operations.

    Encapsulates TMEM address encoding for accumulator fragment access.
    SM100 MMA accumulators are organized as 32 rows, split into:
      - Upper fragment (rows 0-15): accessed via upper_addr()
      - Lower fragment (rows 16-31): accessed via lower_addr()

    The lower fragment address adds TMEM_LOWER_ROW_OFFSET (16 << 16) to
    encode the row offset in the upper 16 bits of the address.

    Usage:
        var tmem = TmemAddress(base_offset)

        # Load operations
        var upper = tmem.load_upper[dtype, size]()
        var lower = tmem.load_lower[dtype, size]()
        TmemAddress.wait_load()

        # Store operations
        tmem.store_upper[dtype, size](upper_frag)
        tmem.store_lower[dtype, size](lower_frag)
        TmemAddress.wait_store()

        # Low-level address access for custom operations
        raw_upper = tmem.upper_addr()
        raw_lower = tmem.lower_addr()
    """

    var addr: UInt32

    @always_inline
    def __init__(out self, addr: Int):
        """Create TmemAddress from integer column address.

        Args:
            addr: TMEM column address as a plain integer.
        """
        self.addr = UInt32(addr)

    @always_inline
    def __init__(out self, addr: UInt32):
        """Create TmemAddress from hardware address (UInt32).

        Args:
            addr: Raw TMEM hardware address as produced by
                `tcgen05_alloc`.
        """
        self.addr = addr

    @always_inline
    def __add__(self, offset: Int) -> Self:
        """Create new TmemAddress with column offset added.

        Args:
            offset: Number of TMEM columns to advance the address by.
        """
        return Self(Int(self.addr) + offset)

    @always_inline
    def upper_addr(self) -> UInt32:
        """Raw address for upper fragment (rows 0-15)."""
        return self.addr

    @always_inline
    def lower_addr(self) -> UInt32:
        """Raw address for lower fragment (rows 16-31)."""
        return self.addr + TMEM_LOWER_ROW_OFFSET

    @always_inline
    def load_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> Array[Scalar[dtype], width]:
        """Load upper accumulator fragment (rows 0-15).

        Parameters:
            dtype: Element type to load from TMEM.
            width: Total number of elements per row in the returned
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits loaded per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the load pattern (defaults
                to 1).
        """
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.upper_addr())

    @always_inline
    def load_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self) -> Array[Scalar[dtype], width]:
        """Load lower accumulator fragment (rows 16-31).

        Parameters:
            dtype: Element type to load from TMEM.
            width: Total number of elements per row in the returned
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits loaded per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the load pattern (defaults
                to 1).
        """
        return tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=dtype,
            pack=False,
            width=width,
        ](self.lower_addr())

    @always_inline
    def store_upper[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: Array[Scalar[dtype], width]):
        """Store upper accumulator fragment (rows 0-15).

        Parameters:
            dtype: Element type to store to TMEM.
            width: Total number of elements per row in the `data`
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits stored per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the store pattern (defaults
                to 1).

        Args:
            data: `Array` of accumulator data to store.
        """
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.upper_addr(), data)

    @always_inline
    def store_lower[
        dtype: DType,
        width: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 1,
    ](self, data: Array[Scalar[dtype], width]):
        """Store lower accumulator fragment (rows 16-31).

        Parameters:
            dtype: Element type to store to TMEM.
            width: Total number of elements per row in the `data`
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits stored per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the store pattern (defaults
                to 1).

        Args:
            data: `Array` of accumulator data to store.
        """
        tcgen05_st[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            pack=False,
        ](self.lower_addr(), data)

    @staticmethod
    @always_inline
    def wait_store():
        """Wait for TMEM store operations to complete."""
        tcgen05_store_wait()

    @staticmethod
    @always_inline
    def wait_load():
        """Wait for TMEM load operations to complete."""
        tcgen05_load_wait()


# =============================================================================
# TmemTensor - Layout-parameterized typed view over TMEM
# =============================================================================


struct TmemTensor[
    dtype: DType,
    layout: Layout,
    *,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Typed tensor view over Tensor Memory (TMEM) for MMA accumulators.

    Provides a typed abstraction for TMEM with:
    - Type safety: dtype and layout known at compile time
    - Fragment access: upper (rows 0-15) and lower (rows 16-31)
    - MMA integration: offset() returns raw address for MMA operations

    The layout parameter captures the logical accumulator shape (M × N),
    enabling future extensions like custom tiling patterns or multi-tile
    accumulator management.

    Parameters:
        dtype: Accumulator data type (typically float32).
        layout: Logical layout of the accumulator tile (M × N).
        cta_group: CTA cooperation level (1 or 2).

    Example:
        # Create typed TMEM view with (64, 128) accumulator layout
        comptime layout = Layout.row_major(64, 128)
        var tmem = TmemTensor[.float32, layout](col_offset)

        # Use with MMA operations (returns raw UInt32 offset)
        mma_op.mma(a_tile, b_tile, tmem.offset(), init_c=True)

        # Load fragments for epilogue
        var upper = tmem.load_upper[repeat=4]()
        var lower = tmem.load_lower[repeat=4]()
        TmemTensor.wait_load()
    """

    # SM100 tcgen05 fragment parameters
    comptime data_paths = 16
    comptime bits = 256
    comptime frag_size = (Self.data_paths * (Self.bits // 32)) // 32

    # Lower fragment required unless cta_group=1 and M=64
    comptime tile_m = Self.layout.shape[0].value()
    comptime is_lower_required = not (Self.cta_group == 1 and Self.tile_m == 64)

    var col_addr: Int

    @always_inline
    def __init__(out self, col_addr: Int):
        """Create TMEM tensor view at the given column address.

        Args:
            col_addr: TMEM column address for the accumulator tile.
        """
        self.col_addr = col_addr

    @always_inline
    def __init__(out self, addr: TmemAddress):
        """Create TMEM tensor view from a TmemAddress.

        Args:
            addr: `TmemAddress` whose column address becomes the tensor
                view base.
        """
        self.col_addr = Int(addr.addr)

    @always_inline
    def offset(self) -> Int:
        """TMEM column address for this tensor."""
        return self.col_addr

    @always_inline
    def address(self) -> TmemAddress:
        """Get TmemAddress for low-level fragment operations."""
        return TmemAddress(self.col_addr)

    @always_inline
    def load_upper[
        repeat: Int = 1,
    ](self) -> Array[Scalar[Self.dtype], Self.frag_size * repeat]:
        """Load upper accumulator fragment (rows 0-15).

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            Array containing the upper fragment data.
        """
        return self.address().load_upper[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ]()

    @always_inline
    def load_lower[
        repeat: Int = 1,
    ](self) -> Array[Scalar[Self.dtype], Self.frag_size * repeat]:
        """Load lower accumulator fragment (rows 16-31).

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            Array containing the lower fragment data.
        """
        return self.address().load_lower[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ]()

    @always_inline
    def store_upper[
        repeat: Int = 1,
    ](self, data: Array[Scalar[Self.dtype], Self.frag_size * repeat]):
        """Store upper accumulator fragment (rows 0-15).

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            data: Array containing the data to store.
        """
        self.address().store_upper[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ](data)

    @always_inline
    def store_lower[
        repeat: Int = 1,
    ](self, data: Array[Scalar[Self.dtype], Self.frag_size * repeat]):
        """Store lower accumulator fragment (rows 16-31).

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            data: Array containing the data to store.
        """
        self.address().store_lower[
            Self.dtype,
            Self.frag_size * repeat,
            Self.data_paths,
            Self.bits,
            repeat,
        ](data)

    # ========== Unified Fragment Access ==========

    comptime Fragments = TmemFragments[
        Self.dtype,
        Self.frag_size,
        is_lower_required=Self.is_lower_required,
        data_paths=Self.data_paths,
        bits=Self.bits,
    ]

    @always_inline
    def load_fragments[
        repeat: Int = 1,
    ](self) -> TmemFragments[
        Self.dtype,
        Self.frag_size * repeat,
        is_lower_required=Self.is_lower_required,
    ]:
        """Load both upper and lower fragments in one call.

        Handles is_lower_required automatically based on layout.

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Returns:
            TmemFragments containing upper and (conditionally) lower data.
        """
        return TmemFragments[
            Self.dtype,
            Self.frag_size,
            is_lower_required=Self.is_lower_required,
        ].load[repeat](self.address())

    @always_inline
    def store_fragments[
        repeat: Int = 1,
    ](
        self,
        frags: TmemFragments[
            Self.dtype,
            Self.frag_size * repeat,
            is_lower_required=Self.is_lower_required,
        ],
    ):
        """Store both upper and lower fragments in one call.

        Handles is_lower_required automatically based on layout.

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            frags: TmemFragments containing upper and (conditionally) lower data.
        """
        frags.store[repeat](self.address())

    @staticmethod
    @always_inline
    def wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()

    @staticmethod
    @always_inline
    def wait_store():
        """Wait for TMEM store operations to complete."""
        TmemAddress.wait_store()


# =============================================================================
# TmemFragments - Paired upper/lower accumulator fragments
# =============================================================================


struct TmemFragments[
    dtype: DType,
    frag_size: Int,
    *,
    is_lower_required: Bool = True,
    data_paths: Int = 16,
    bits: Int = 256,
](Copyable, Movable):
    """Paired upper/lower accumulator fragments from TMEM.

    Encapsulates the SM100 TMEM row-split hardware detail:
    - Upper fragment: rows 0-15 (always present)
    - Lower fragment: rows 16-31 (only when is_lower_required=True)

    The is_lower_required flag is determined by:
    - False when cta_group=1 and MMA_M=64 (fits in 16 rows)
    - True otherwise (needs both halves)

    Parameters:
        dtype: Fragment data type (typically float32).
        frag_size: Elements per fragment (derived from data_paths and bits).
        is_lower_required: Whether lower fragment is needed.
        data_paths: SM100 data paths (typically 16).
        bits: Bits per fragment load (typically 256).

    Example:
        # Load both fragments in one call
        var frags = TmemFragments[.float32, 16].load(tmem_addr)

        # Work with fragments
        frags.upper = process(frags.upper)
        frags.lower = process(frags.lower)

        # Store both fragments
        frags.store(tmem_addr)
        TmemFragments.wait_store()
    """

    var upper: Array[Scalar[Self.dtype], Self.frag_size]
    var lower: Array[Scalar[Self.dtype], Self.frag_size]

    @always_inline
    def __init__(out self):
        """Initialize with zero fragments."""
        self.upper = Array[Scalar[Self.dtype], Self.frag_size](
            fill=Scalar[Self.dtype](0)
        )
        self.lower = Array[Scalar[Self.dtype], Self.frag_size](
            fill=Scalar[Self.dtype](0)
        )

    @always_inline
    def __init__(
        out self,
        upper: Array[Scalar[Self.dtype], Self.frag_size],
        lower: Array[Scalar[Self.dtype], Self.frag_size],
    ):
        """Initialize with provided fragments.

        Args:
            upper: `Array` of accumulator data for rows 0-15.
            lower: `Array` of accumulator data for rows 16-31.
        """
        self.upper = upper.copy()
        self.lower = lower.copy()

    @staticmethod
    @always_inline
    def load[
        repeat: Int = 1
    ](tmem: TmemAddress) -> TmemFragments[
        Self.dtype,
        Self.frag_size * repeat,
        is_lower_required=Self.is_lower_required,
    ]:
        """Load fragments from TMEM address.

        Loads upper fragment always; loads lower only if required.

        Parameters:
            repeat: Number of times to repeat the load pattern.

        Args:
            tmem: TMEM address to load from.

        Returns:
            TmemFragments containing upper and (optionally) lower data.
        """
        comptime width = Self.frag_size * repeat
        var result = TmemFragments[
            Self.dtype, width, is_lower_required=Self.is_lower_required
        ]()
        result.upper = tmem.load_upper[
            Self.dtype, width, Self.data_paths, Self.bits, repeat
        ]()

        comptime if Self.is_lower_required:
            result.lower = tmem.load_lower[
                Self.dtype, width, Self.data_paths, Self.bits, repeat
            ]()

        return result^

    @always_inline
    def store[repeat: Int = 1](self, tmem: TmemAddress):
        """Store fragments to TMEM address.

        Stores upper fragment always; stores lower only if required.

        Parameters:
            repeat: Number of times to repeat the store pattern.

        Args:
            tmem: TMEM address to store to.
        """
        tmem.store_upper[
            Self.dtype, Self.frag_size, Self.data_paths, Self.bits, repeat
        ](self.upper)

        comptime if Self.is_lower_required:
            tmem.store_lower[
                Self.dtype, Self.frag_size, Self.data_paths, Self.bits, repeat
            ](self.lower)

    @always_inline
    def cast[
        target_dtype: DType
    ](self) -> TmemFragments[
        target_dtype, Self.frag_size, is_lower_required=Self.is_lower_required
    ]:
        """Cast fragments to a different dtype.

        Parameters:
            target_dtype: Destination element type for the cast fragments.
        """
        var result = TmemFragments[
            target_dtype,
            Self.frag_size,
            is_lower_required=Self.is_lower_required,
        ]()

        # Cast in SIMD chunks of at least 4 bytes for efficient
        # hardware cast instructions.
        comptime cast_width = 4 // size_of[Scalar[target_dtype]]()
        comptime for _chunk in range(Self.frag_size // cast_width):
            comptime offset = _chunk * cast_width
            var src = SIMD[Self.dtype, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = self.upper[offset + _j]
            var dst = src.cast[target_dtype]()
            comptime for _j in range(cast_width):
                result.upper[offset + _j] = dst[_j]

        comptime if Self.is_lower_required:
            comptime for _chunk in range(Self.frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.dtype, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = self.lower[offset + _j]
                var dst = src.cast[target_dtype]()
                comptime for _j in range(cast_width):
                    result.lower[offset + _j] = dst[_j]

        return result^

    @staticmethod
    @always_inline
    def wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()

    @staticmethod
    @always_inline
    def wait_store():
        """Wait for TMEM store operations to complete."""
        TmemAddress.wait_store()


# =============================================================================
# TmemArrayType - Array of tiles in Tensor Memory
# =============================================================================


struct TmemArrayType[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    *,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Array of tiles in Tensor Memory (TMEM).

    Similar to SMemArray but for TMEM-resident tiles. Provides indexed
    access to a contiguous array of TmemTensor tiles.

    Parameters:
        dtype: Element dtype for tiles.
        layout: Layout of each tile.
        num_tiles: Number of tiles in the array.
        cta_group: CTA group size (1 or 2).

    Compile-time constants:
        Tile: TmemTensor type for each tile.
        tile_stride: Columns per tile (derived from layout.size()).
        num_cols: Total TMEM columns used (num_tiles × tile_stride).
    """

    comptime Tile = TmemTensor[
        Self.dtype, Self.layout, cta_group=Self.cta_group
    ]
    # TMEM addresses are in column units, so stride is N dimension (shape[1])
    comptime tile_stride = Self.layout.shape[1].value()
    comptime num_cols = Self.num_tiles * Self.tile_stride

    var base_addr: Int

    @always_inline
    def __init__(out self, base_addr: Int):
        """Initialize array at the given TMEM base address.

        Args:
            base_addr: Starting TMEM column address for the array.
        """
        self.base_addr = base_addr

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at the given index.

        Parameters:
            T: Int-like type used for the tile index.

        Args:
            index: Tile index in the range `[0, num_tiles)`.
        """
        return Self.Tile(self.base_addr + Int(index) * Self.tile_stride)


# =============================================================================
# BlockScaledTmem - TMEM region for block-scaled MMA operations
# =============================================================================


struct BlockScaledTmem[
    # Accumulator configuration
    accum_dtype: DType,
    MMA_M: Int,
    MMA_N: Int,
    num_accum_stages: Int,
    # Scaling factor configuration
    sf_dtype: DType,
    BM: Int,  # Block M dimension (for SFA)
    num_pipeline_stages: Int,
    *,
    cta_group: Int = 1,
    total_cols: Int = 512,
    num_sf_k_tiles: Int = 1,
    SFB_N: Int = MMA_N,
](TrivialRegisterPassable):
    """TMEM region for block-scaled matmul with typed tile accessors.

    Manages the TMEM address space for block-scaled MMA operations,
    providing typed TmemTensor access to:
    - Accumulator tiles (one per output pipeline stage)
    - SFA scaling factor tiles (one per k-iteration)
    - SFB scaling factor tiles (one per k-iteration)

    Memory layout (512 columns total):
    ┌────────────────────────────────────────────────────────────┐
    │ Accumulators     │ SFA Scales        │ SFB Scales         │
    │ (stages × MMA_N) │ (iters × cols)    │ (iters × cols)     │
    └────────────────────────────────────────────────────────────┘

    Parameters:
        accum_dtype: Accumulator data type (typically float32).
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension (also stage stride for accumulators).
        num_accum_stages: Number of accumulator pipeline stages.
        sf_dtype: Scaling factor data type.
        BM: Block M dimension (for SFA sizing).
        num_pipeline_stages: Number of k-iteration pipeline stages.
        cta_group: CTA group size (1 or 2).
        total_cols: Total TMEM columns (512 for SM100).
        num_sf_k_tiles: Scaling factor tiles per K-iteration.
            MXFP8 uses 1 (one SF vector per K-tile).
            NVFP4 uses 4 (multiple SF vectors per K-tile).
        SFB_N: SFB N dimension for TMEM layout. Defaults to MMA_N.
            Set to align_up(MMA_N, SF_MN_GROUP_SIZE) when
            MMA_N < SF_MN_GROUP_SIZE so the TMEM tile is wide enough
            for the SMEM-to-TMEM copy (which always writes a full
            SF_MN_GROUP_SIZE group).
    """

    # Tile layouts (stride derived automatically from layout.size())
    # Each SFA/SFB tile in TMEM covers num_sf_k_tiles SF vectors,
    # so the column width is num_sf_k_tiles * (dim // 32).
    comptime accum_layout = Layout.row_major(Self.MMA_M, Self.MMA_N)
    comptime sfa_layout = Layout.row_major(
        1, Self.num_sf_k_tiles * (Self.BM // 32)
    )
    comptime sfb_layout = Layout.row_major(
        1, Self.num_sf_k_tiles * (Self.SFB_N // 32)
    )

    # Array types for each TMEM region
    comptime AccumArray = TmemArrayType[
        Self.accum_dtype,
        Self.accum_layout,
        num_tiles=Self.num_accum_stages,
        cta_group=Self.cta_group,
    ]
    comptime SFAArray = TmemArrayType[
        Self.sf_dtype,
        Self.sfa_layout,
        num_tiles=Self.num_pipeline_stages,
        cta_group=Self.cta_group,
    ]
    comptime SFBArray = TmemArrayType[
        Self.sf_dtype,
        Self.sfb_layout,
        num_tiles=Self.num_pipeline_stages,
        cta_group=Self.cta_group,
    ]

    # Tile types (for convenience)
    comptime AccumTile = Self.AccumArray.Tile
    comptime SFATile = Self.SFAArray.Tile
    comptime SFBTile = Self.SFBArray.Tile

    # Region base offsets (compile-time constants)
    comptime accum_offset = 0
    comptime sfa_offset = Self.AccumArray.num_cols
    comptime sfb_offset = Self.sfa_offset + Self.SFAArray.num_cols
    comptime used_cols = Self.sfb_offset + Self.SFBArray.num_cols

    var base_addr: Int

    @always_inline
    def __init__(out self, base_addr: Int):
        """Create TMEM region view at the given base address.

        Args:
            base_addr: Base TMEM column address for the region.
        """
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = base_addr

    @always_inline
    def __init__(out self, addr: TmemAddress):
        """Create TMEM region view from a TmemAddress.

        Args:
            addr: `TmemAddress` whose column address becomes the region
                base.
        """
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = Int(addr.addr)

    @always_inline
    def __init__[
        cta: Int, max_cols: Int
    ](out self, alloc: TmemAllocation[cta, max_cols]):
        """Create TMEM region view from a TmemAllocation.

        Parameters:
            cta: CTA group size of the source allocation (1 or 2).
            max_cols: Maximum TMEM columns of the source allocation
                (512 for SM100).

        Args:
            alloc: The `TmemAllocation` whose address becomes the region
                base.
        """
        comptime assert (
            Self.used_cols <= Self.total_cols
        ), "Block-scaled TMEM region exceeds capacity"
        self.base_addr = Int(alloc.addr)

    @always_inline
    def accum_tiles(self) -> Self.AccumArray:
        """Get array of accumulator tiles."""
        return Self.AccumArray(self.base_addr + Self.accum_offset)

    @always_inline
    def sfa_tiles(self) -> Self.SFAArray:
        """Get array of SFA scaling factor tiles."""
        return Self.SFAArray(self.base_addr + Self.sfa_offset)

    @always_inline
    def sfb_tiles(self) -> Self.SFBArray:
        """Get array of SFB scaling factor tiles."""
        return Self.SFBArray(self.base_addr + Self.sfb_offset)

    # Convenience accessors (delegate to arrays)
    @always_inline
    def accum[T: Intable](self, stage: T) -> Self.AccumTile:
        """Get accumulator tile for the given pipeline stage.

        Parameters:
            T: Int-like type for the stage index.

        Args:
            stage: Pipeline stage index into the accumulator array.
        """
        return self.accum_tiles()[stage]

    @always_inline
    def sfa[T: Intable](self, index: T) -> Self.SFATile:
        """Get SFA scaling factor tile for the given k-iteration index.

        Parameters:
            T: Int-like type for the index value.

        Args:
            index: K-iteration index into the SFA tile array.
        """
        return self.sfa_tiles()[index]

    @always_inline
    def sfb[T: Intable](self, index: T) -> Self.SFBTile:
        """Get SFB scaling factor tile for the given k-iteration index.

        Parameters:
            T: Int-like type for the index value.

        Args:
            index: K-iteration index into the SFB tile array.
        """
        return self.sfb_tiles()[index]


# =============================================================================
# TmemStage - Pipeline stage wrapper for accumulator buffering
# =============================================================================


struct TmemStage[
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """A pipeline stage within TMEM for accumulator buffering.

    Used by OutputTilePipeline to manage MMA→Epilogue synchronization.
    MMA writes to one stage while epilogue reads from another.

    Wraps TmemAddress with stage-specific offset calculation:
      - offset(): Column address for this stage (base + index * stride)
      - address(): TmemAddress for this stage (for load/store ops)
      - tensor[layout](): Get typed TmemTensor view

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime num_stages = Self.opc.num_stages
    comptime stage_stride = Self.opc.stage_stride_cols
    comptime cta_group = Self.opc.cta_group

    var base_addr: Int
    var index: Int

    @always_inline
    def __init__(out self, base_addr: Int, index: Int):
        self.base_addr = base_addr
        self.index = index

    @always_inline
    def __init__(out self, addr: TmemAddress, index: Int):
        """Create stage from TmemAddress and stage index.

        Args:
            addr: `TmemAddress` whose column address becomes the stage
                base.
            index: Pipeline stage index (for barrier signaling).
        """
        self.base_addr = Int(addr.addr)
        self.index = index

    @always_inline
    def __init__[
        cta: Int, max_cols: Int
    ](out self, alloc: TmemAllocation[cta, max_cols], index: Int):
        """Create stage from TmemAllocation and stage index.

        Parameters:
            cta: CTA group size of the source allocation (1 or 2).
            max_cols: Maximum TMEM columns of the source allocation
                (512 for SM100).

        Args:
            alloc: `TmemAllocation` whose address becomes the stage base.
            index: Pipeline stage index (for barrier signaling).
        """
        self.base_addr = Int(alloc.addr)
        self.index = index

    @staticmethod
    @always_inline
    def from_offset(offset: Int, index: Int) -> Self:
        """Create stage from pre-computed offset (for legacy pipeline compatibility).

        Use this when the caller has already computed the TMEM offset
        (e.g., `base + stage * stride`) and just needs to wrap it.

        The index is preserved for barrier signaling, and we back-calculate
        the base_addr such that offset() = base + index * stride = offset.

        Args:
            offset: Pre-computed TMEM column offset for this stage.
            index: Pipeline stage index (for barrier signaling).

        Returns:
            TmemStage with offset() returning the given value.
        """
        # Back-calculate base_addr so that base_addr + index * stride = offset
        # base_addr = offset - index * stride
        var base_addr = offset - index * Self.stage_stride
        return Self(base_addr, index)

    @always_inline
    def offset(self) -> Int:
        """TMEM column address for this stage."""
        return self.base_addr + self.index * Self.stage_stride

    @always_inline
    def address(self) -> TmemAddress:
        """Get TmemAddress for this stage's offset."""
        return TmemAddress(self.offset())

    @always_inline
    def tensor[
        accum_dtype: DType,
        accum_layout: Layout,
    ](self) -> TmemTensor[accum_dtype, accum_layout, cta_group=Self.cta_group]:
        """Get typed TmemTensor view of this stage's accumulator.

        Parameters:
            accum_dtype: Accumulator data type.
            accum_layout: Logical accumulator layout (M × N).

        Returns:
            TmemTensor providing typed access to the accumulator.
        """
        return TmemTensor[accum_dtype, accum_layout, cta_group=Self.cta_group](
            self.base_addr + self.index * Self.stage_stride
        )

    @always_inline
    def load_upper[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> Array[Scalar[dtype], frag_size]:
        """Load upper accumulator fragment (rows 0-15).

        Parameters:
            dtype: Element type to load from TMEM.
            frag_size: Total number of elements per row in the returned
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits loaded per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the load pattern (defaults
                to 4).
        """
        return self.address().load_upper[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @always_inline
    def load_lower[
        dtype: DType,
        frag_size: Int,
        data_paths: Int = 16,
        bits: Int = 256,
        repeat: Int = 4,
    ](self) -> Array[Scalar[dtype], frag_size]:
        """Load lower accumulator fragment (rows 16-31).

        Parameters:
            dtype: Element type to load from TMEM.
            frag_size: Total number of elements per row in the returned
                `Array`.
            data_paths: Number of SM100 TMEM data paths (defaults to 16).
            bits: Bits loaded per data path per repeat (defaults to 256).
            repeat: Number of times to repeat the load pattern (defaults
                to 4).
        """
        return self.address().load_lower[
            dtype, frag_size, data_paths, bits, repeat
        ]()

    @staticmethod
    @always_inline
    def wait_load():
        """Wait for TMEM load operations to complete."""
        TmemAddress.wait_load()


# =============================================================================
# TmemDeallocBarrier - TMEM deallocation synchronization
# =============================================================================


struct TmemDeallocBarrier[cta_group: Int](TrivialRegisterPassable):
    """TMEM deallocation synchronization barrier.

    Handles cluster-aware synchronization patterns for TMEM deallocation,
    supporting both single-CTA and multi-CTA (cta_group=2) configurations.

    Parameters:
        cta_group: Number of cooperating CTAs (1 or 2).
    """

    comptime BarrierStorage = SMemArray[SharedMemBarrier, 1]

    var barrier: Self.BarrierStorage

    def __init__(out self, barrier: Self.BarrierStorage):
        """Initialize with shared memory barrier array.

        Args:
            barrier: Shared memory barrier storage used to coordinate
                deallocation across warps and CTAs.
        """
        self.barrier = barrier

    @always_inline
    def signal_peer(self):
        """Signal peer CTA in cluster (cta_group=2 only)."""

        comptime if Self.cta_group == 2:
            _ = self.barrier.ptr[].arrive_cluster(block_rank_in_cluster() ^ 1)

    @always_inline
    def signal_self(self):
        """Signal own arrival at barrier."""
        _ = self.barrier.ptr[].arrive()

    @always_inline
    def wait[ticks: Optional[UInt32] = None](self):
        """Wait for barrier completion.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns) forwarded to
                `SharedMemBarrier.wait`. `None` (default) preserves the
                immediate-return spin for every existing caller.
        """
        self.barrier.ptr[].wait[ticks=ticks]()

    @always_inline
    def complete_dealloc[
        max_cols: Int = 512
    ](self, tmem: TmemAllocation[Self.cta_group, max_cols]):
        """Complete TMEM deallocation sequence (MMA warp side).

        Releases the allocation lock, waits for epilogue completion,
        then deallocates the TMEM.

        Parameters:
            max_cols: TMEM columns to deallocate (defaults to 512 for
                SM100).

        Args:
            tmem: `TmemAllocation` to release and deallocate.
        """
        tmem.release_lock()
        # SM100 warp-specialized matmul: hardware-suspend the MMA warp while it
        # waits for the epilogue to consume TMEM, instead of busy-spinning.
        # See SM100_PIPELINE_WAIT_TICKS.
        self.wait[ticks=SM100_PIPELINE_WAIT_TICKS]()
        tmem.deallocate()

    @always_inline
    def signal_complete(self):
        """Signal TMEM consumption complete (Epilogue warp side).

        For cta_group=2, signals peer CTA first, then signals self.
        """
        self.signal_peer()
        self.signal_self()
