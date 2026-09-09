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
"""Unified Pipeline Storage Framework for SM100 Structured Kernels.

This module provides a single-source-of-truth framework for pipeline storage,
where stage count determines barrier count, and tile storage type determines
the SMEM layout for input tiles.

All tile storage uses TileTensor natively. Conversion to LayoutTensor only
happens at external API boundaries (TMA, MMA) using the {ptr} syntax or
explicit LayoutTensor construction.

Design Principles
-----------------
1. **Single Source of Truth**: Stage count parameterizes barrier count
2. **Single Source of Truth**: Tile storage types define array types once
3. **TileTensor Native**: All SMEM tiles use TileTensor
4. **Composable**: SMEM structs compose storage objects
5. **Extensible**: Easy to add new storage types
6. **Escape Hatch**: Raw storage access when framework doesn't fit

Architecture
------------
```
┌─────────────────────────────────────────────────────────────────────┐
│  Tile Storage (defines tile arrays and storage)                     │
│                                                                     │
│  StandardTileStorage[a_type, b_type, a_shape, b_shape, ...]        │
│      ├── ATileArray = SMemTileArray2D[...]  # TileTensor-based     │
│      ├── BTileArray = SMemTileArray2D[...]  # TileTensor-based     │
│      ├── var a_tiles_storage                                        │
│      ├── var b_tiles_storage                                        │
│      └── def a_tiles(), b_tiles()  # Returns TileTensor             │
│                                                                     │
│  BlockScaledTileStorage[..., sfa_type, sfb_type, dims, ...]        │
│  BlockwiseFP8TileStorage[..., a_scales_type, dims, ...]            │
│  OutputTileStorage[c_type, c_layout, num_stages]                   │
├─────────────────────────────────────────────────────────────────────┤
│  Pipeline Storage (defines barriers)                                │
│                                                                     │
│  InputPipelineStorage[num_stages, Payload]                         │
│      └── var barriers: BarrierPair[num_stages]                     │
│                                                                     │
│  OutputPipelineStorage[num_stages]                                 │
│  ClcPipelineStorage[num_stages]                                    │
│  TmemDeallocStorage                                                │
├─────────────────────────────────────────────────────────────────────┤
│  SMEM composes both:                                                │
│                                                                     │
│  struct MySmem:                                                     │
│      var tiles: StandardTileStorage[...]      # Tile storage       │
│      var output_tiles: OutputTileStorage[...] # Output tiles       │
│      var input_pipeline: InputPipelineStorage[...]  # Barriers     │
│      var output_pipeline: OutputPipelineStorage[...]                │
│      var clc_pipeline: ClcPipelineStorage[...]                     │
└─────────────────────────────────────────────────────────────────────┘
```

Example Usage
-------------
```
struct MyKernelSmem[config: MyConfig]:
    # Tile storage (single source of truth for tile types)
    comptime Tiles = StandardTileStorage[
        config.a_type, config.b_type,
        IndexList[2](config.BM, config.BK),  # A tile shape
        IndexList[2](config.BN, config.BK),  # B tile shape
        config.num_pipeline_stages,
    ]
    var tiles: Self.Tiles

    # Output tile storage (separate stage count)
    comptime OutputTiles = OutputTileStorage[
        config.c_type, config.c_layout, config.num_output_stages
    ]
    var output_tiles: Self.OutputTiles

    # Pipeline storage (barriers)
    var input_pipeline: InputPipelineStorage[...]
    var output_pipeline: OutputPipelineStorage[...]

    # Accessors delegate to composed storage
    def a_tiles(ref[SHARED] self) -> Self.Tiles.ATileArray:
        return self.tiles.a_tiles()  # Returns TileTensor

    def c_tiles(ref[SHARED] self) -> Self.OutputTiles.CTileArray:
        return self.output_tiles.c_tiles()
```

Extensibility
-------------
To add a new tile storage type:
1. Create a new struct with comptime type aliases and storage fields
2. Add accessors that construct tile arrays from storage
3. Use in SMEM via composition

Escape Hatch
------------
When the framework doesn't fit:
1. Use raw SMemArray for custom tile layouts
2. Use RawBarrierStorage for non-standard barrier patterns
3. Add custom storage fields to SMEM struct
"""

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout
from layout.tma_async import SharedMemBarrier
from layout.tensor_core_async import tile_layout_k_major_typed
from std.utils.index import IndexList

# SMemArray for barriers (non-tile arrays), SMemPtr for barrier pointers
from .smem_types import SMemArray, SMemPtr

# LayoutTensor-based SMemTileArray for C output tiles (used by epilogue)


# TileTensor-based tile arrays for input tiles (A, B, scales)
from .tile_types import (
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    SMemTileArrayWithLayout,
    TilePayload,
    internal_sf_k_major,
)

# Import variadic types for SMemTileArray parameters

comptime MbarPtr = SMemPtr[SharedMemBarrier]

from .pipeline import ProducerConsumerPipeline


# =============================================================================
# Tile Storage - Single source of truth for input tile layouts
# =============================================================================


struct StandardTileStorage[
    a_type: DType,
    b_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    # Pipeline stages
    num_pipeline_stages: Int,
]:
    """Storage for standard matmul tiles (A and B).

    This is the single source of truth for tile array types and storage.
    SMEM structs embed this rather than defining tile arrays separately.

    All tiles use TileTensor natively. Convert to LayoutTensor at TMA/MMA
    boundaries using {ptr} syntax or explicit construction.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_shape: Tile shape for A tiles as (rows, cols).
        b_shape: Tile shape for B tiles as (rows, cols).
        num_pipeline_stages: Number of pipeline stages (determines array depth).
    """

    # TileTensor-based array types (native, explicit dimensions)
    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]

    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage

    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor (TileTensor-based)."""
        return Self.ATileArray(self.a_tiles_storage.unsafe_ptr())

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor (TileTensor-based)."""
        return Self.BTileArray(self.b_tiles_storage.unsafe_ptr())


struct BlockScaledTileStorage[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    # C tile dimensions kept separate for type-level compatibility with TileWriter
    c_dim0: Int,
    c_dim1: Int,
    sfa_shape: IndexList[2],  # Scale factor A dims
    sfb_shape: IndexList[2],  # Scale factor B dims
    # Pipeline stages
    num_pipeline_stages: Int,
    num_output_stages: Int,
]:
    """Storage for block-scaled matmul tiles (A, B, C, SFA, SFB).

    Single source of truth for block-scaled tile arrays and storage.

    All tiles use TileTensor natively. LayoutTensor conversion
    is kept for SMemEpilogueWriter compatibility.

    IMPORTANT: Field order preserves SMEM layout compatibility: a, b, c, sfa, sfb.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        c_type: Data type for C matrix tiles.
        sfa_type: Data type for A scale factor tiles.
        sfb_type: Data type for B scale factor tiles.
        a_shape: Tile shape for A tiles as (rows, cols).
        b_shape: Tile shape for B tiles as (rows, cols).
        c_dim0: First dimension for C tiles (OutputM).
        c_dim1: Second dimension for C tiles (OutputN).
        sfa_shape: Tile shape for SFA tiles.
        sfb_shape: Tile shape for SFB tiles.
        num_pipeline_stages: Number of input pipeline stages.
        num_output_stages: Number of output pipeline stages.
    """

    # C tile layout derived from dimensions (row_major for TMA compatibility)
    comptime c_tile_layout = Layout.row_major(Self.c_dim0, Self.c_dim1)

    # TileTensor-based array types (native, explicit dimensions)
    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    # TileTensor-based for C storage (row_major layout - no swizzle for C tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
    ]
    # SF tiles use internal_sf_k_major layout (matches tile_sf_layout_k_major).
    # MMA extracts layout directly from TileTensor type parameters.
    comptime sfa_layout = internal_sf_k_major[
        Self.sfa_shape[0], Self.sfa_shape[1]
    ]
    comptime sfb_layout = internal_sf_k_major[
        Self.sfb_shape[0], Self.sfb_shape[1]
    ]
    comptime SFATileArray = SMemTileArrayWithLayout[
        Self.sfa_type,
        Self.sfa_layout,
        Self.num_pipeline_stages,
        128,
    ]
    comptime SFBTileArray = SMemTileArrayWithLayout[
        Self.sfb_type,
        Self.sfb_layout,
        Self.num_pipeline_stages,
        128,
    ]

    # Field order preserves SMEM layout: a, b, c, sfa, sfb
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var sfa_tiles_storage: Self.SFATileArray.Storage
    var sfb_tiles_storage: Self.SFBTileArray.Storage

    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return Self.ATileArray(self.a_tiles_storage.unsafe_ptr())

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return Self.BTileArray(self.b_tiles_storage.unsafe_ptr())

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return Self.CTileArray(self.c_tiles_storage.unsafe_ptr())

    @always_inline
    def sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        """Get SFA tile array accessor."""
        return Self.SFATileArray(self.sfa_tiles_storage.unsafe_ptr())

    @always_inline
    def sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        """Get SFB tile array accessor."""
        return Self.SFBTileArray(self.sfb_tiles_storage.unsafe_ptr())


struct BlockwiseFP8TileStorage[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    # C tile dimensions kept separate for type-level compatibility with TileWriter
    c_dim0: Int,
    c_dim1: Int,
    a_scales_shape: IndexList[2],  # (1, BM)
    # Pipeline stages
    num_pipeline_stages: Int,
    num_output_stages: Int,
]:
    """Storage for blockwise FP8 matmul tiles (A, B, C, A-scales).

    Single source of truth for blockwise FP8 tile arrays and storage.
    B-scales are read directly from global memory during epilogue.

    All tiles use TileTensor natively. LayoutTensor conversion
    is kept for SMemEpilogueWriter compatibility.

    IMPORTANT: Field order preserves SMEM layout compatibility: a, b, c, a_scales.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        c_type: Data type for C matrix tiles.
        a_scales_type: Data type for A scale tiles.
        a_shape: Tile shape for A tiles as (rows, cols).
        b_shape: Tile shape for B tiles as (rows, cols).
        c_dim0: First dimension for C tiles (OutputM).
        c_dim1: Second dimension for C tiles (OutputN).
        a_scales_shape: Tile shape for A scale tiles as (rows, cols).
        num_pipeline_stages: Number of input pipeline stages.
        num_output_stages: Number of output pipeline stages.
    """

    # C tile layout derived from dimensions (row_major for TMA compatibility)
    comptime c_tile_layout = Layout.row_major(Self.c_dim0, Self.c_dim1)

    # TileTensor-based array types for A/B (native, explicit dimensions)
    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    # TileTensor-based for C storage (row_major layout - no swizzle for C tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
    ]
    # A-scales are 1D vectors (1 x BM) - use row_major, NOT swizzled
    # Swizzled layout would corrupt indexing for 1D vectors
    comptime AScalesTileArray = SMemTileArray2DRowMajor[
        Self.a_scales_type,
        Self.a_scales_shape[0],
        Self.a_scales_shape[1],
        Self.num_pipeline_stages,
    ]

    # Field order preserves SMEM layout: a, b, c, a_scales
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var a_scales_tiles_storage: Self.AScalesTileArray.Storage

    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return Self.ATileArray(self.a_tiles_storage.unsafe_ptr())

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return Self.BTileArray(self.b_tiles_storage.unsafe_ptr())

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return Self.CTileArray(self.c_tiles_storage.unsafe_ptr())

    @always_inline
    def a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return Self.AScalesTileArray(self.a_scales_tiles_storage.unsafe_ptr())


struct OutputTileStorage[
    c_type: DType,
    c_dim0: Int,
    c_dim1: Int,
    num_output_stages: Int,
]:
    """Storage for output tiles (C matrix).

    Single source of truth for output tile array and storage.
    Separate from input tiles since output has different stage count.

    All tiles use TileTensor natively. LayoutTensor conversion
    is kept for SMemEpilogueWriter compatibility.

    Parameters:
        c_type: Data type for C matrix tiles.
        c_dim0: First dimension for C tiles (OutputM).
        c_dim1: Second dimension for C tiles (OutputN).
        num_output_stages: Number of output pipeline stages.
    """

    # C tile layout derived from dimensions (row_major for TMA compatibility)
    comptime c_tile_layout = Layout.row_major(Self.c_dim0, Self.c_dim1)

    # TileTensor-based for C storage (row_major layout - no swizzle for C tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type, Self.c_dim0, Self.c_dim1, Self.num_output_stages, 128
    ]
    var c_tiles_storage: Self.CTileArray.Storage

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return Self.CTileArray(self.c_tiles_storage.unsafe_ptr())


# =============================================================================
# Barrier Storage - Foundational building block
# =============================================================================


struct BarrierPair[num_stages: Int]:
    """Storage for a producer-consumer barrier pair (full + empty).

    Each stage has two barriers:
    - full[i]: Producer signals when stage i is filled
    - empty[i]: Consumer signals when stage i is consumed

    Parameters:
        num_stages: Number of pipeline stages (ring buffer depth).
    """

    comptime Array = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    var storage: Self.Array.Storage

    @always_inline
    def barriers(ref[AddressSpace.SHARED] self) -> Self.Array:
        """Get barrier array accessor."""
        return Self.Array(self.storage)

    @always_inline
    def ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get raw barrier pointer for initialization or custom usage."""
        return self.barriers().ptr

    @always_inline
    def create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create a runtime pipeline from this barrier storage."""
        return ProducerConsumerPipeline[Self.num_stages](self.barriers().ptr)

    # Note: Barrier initialization is left to the kernel code as patterns vary.
    # Use ptr() to get raw access for initialization.


# =============================================================================
# Input Pipeline Storage - For TMA → MMA tile transfer
# =============================================================================


struct InputPipelineStorage[
    num_stages: Int,
    Payload: TilePayload,
]:
    """Unified storage for input tile pipeline (barriers + payload).

    Bundles barrier storage with tile payload storage, ensuring they're
    always consistent. The pipeline can only be created from matching storage.

    Parameters:
        num_stages: Number of pipeline stages.
        Payload: Tile payload type (defines what's in each stage).

    Example:
        ```
        struct MySmem[...]:
            var input: InputPipelineStorage[
                4,  # 4 stages
                StandardTilePayload[float16, float16, a_layout, b_layout],
            ]

            def get_pipeline(ref[SHARED] self):
                return self.input.create_pipeline()
        ```
    """

    # Type alias for barrier array (exposed for type-level access)
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    # Barrier storage (derived from stage count)
    var barriers: BarrierPair[Self.num_stages]

    # Payload storage would go here, but Payload types currently
    # don't define their own storage. For now, tile storage remains
    # separate in SMEM. This is the extensibility point for future.
    #
    # TODO: When Payload types define Storage, add:
    # var payload: Payload.Engine[num_stages]

    @always_inline
    def create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create runtime pipeline from this storage."""
        return self.barriers.create_pipeline()

    @always_inline
    def barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Escape hatch: Get raw barrier pointer for custom initialization."""
        return self.barriers.barriers().ptr


# =============================================================================
# Output Pipeline Storage - For MMA → Epilogue TMEM transfer
# =============================================================================


struct OutputPipelineStorage[num_stages: Int]:
    """Unified storage for output/accumulator pipeline.

    For MMA → Epilogue synchronization. TMEM stages are allocated
    dynamically, so this only stores barriers.

    Parameters:
        num_stages: Number of accumulator pipeline stages.
    """

    # Type alias for barrier array (exposed for type-level access)
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    var barriers: BarrierPair[Self.num_stages]

    @always_inline
    def create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create runtime pipeline from this storage."""
        return self.barriers.create_pipeline()

    @always_inline
    def barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Escape hatch: Get raw barrier pointer."""
        return self.barriers.barriers().ptr


# =============================================================================
# CLC Pipeline Storage - For scheduler coordination
# =============================================================================


struct ClcPipelineStorage[num_stages: Int]:
    """Storage for CLC (Cluster Launch Control) scheduler pipeline.

    CLC has a different barrier pattern:
    - full/empty: Standard producer-consumer for work items
    - throttle: Rate limiting barriers (2 per stage)
    - response: CLC response storage (UInt128 per stage)

    Parameters:
        num_stages: Number of CLC pipeline stages.
    """

    # Standard full/empty barriers
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages]
    var full_storage: Self.BarrierArray.Storage
    var empty_storage: Self.BarrierArray.Storage

    # Throttle barriers (2 per stage for rate limiting)
    comptime ThrottleArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]
    var throttle_storage: Self.ThrottleArray.Storage

    # CLC response storage
    comptime ResponseArray = SMemArray[UInt128, Self.num_stages]
    var response_storage: Self.ResponseArray.Storage

    @always_inline
    def full(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.full_storage)

    @always_inline
    def empty(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.empty_storage)

    @always_inline
    def throttle(ref[AddressSpace.SHARED] self) -> Self.ThrottleArray:
        return Self.ThrottleArray(self.throttle_storage)

    @always_inline
    def response(ref[AddressSpace.SHARED] self) -> Self.ResponseArray:
        return Self.ResponseArray(self.response_storage)


# =============================================================================
# TMEM Deallocation Storage - Single barrier for TMEM lifecycle
# =============================================================================


struct TmemDeallocStorage:
    """Storage for TMEM deallocation synchronization.

    Single barrier + address storage for TMEM lifecycle management.
    """

    comptime BarrierArray = SMemArray[SharedMemBarrier, 1]
    comptime AddrArray = SMemArray[UInt32, 1]

    var barrier_storage: Self.BarrierArray.Storage
    var addr_storage: Self.AddrArray.Storage

    @always_inline
    def barrier(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.barrier_storage)

    @always_inline
    def addr(ref[AddressSpace.SHARED] self) -> Self.AddrArray:
        return Self.AddrArray(self.addr_storage)


# =============================================================================
# Source Tile Storage - For residual/skip connection input (tensor C)
# =============================================================================


struct SourceTileStorage[
    src_type: DType,
    src_shape: IndexList[2],  # (OutputM, OutputN)
    num_epi_load_stages: Int,
]:
    """Storage for source tensor C tiles (residual/skip connection input).

    Used by the epilogue load warp to pre-fetch source tensor C via TMA,
    enabling overlap with MMA computation for residual operations like
    D = Conv(A,B) + beta*C or D = MatMul(A,B) + beta*C.

    Parameters:
        src_type: Data type for source tiles (same as output type).
        src_shape: Tile shape for source tiles as (OutputM, OutputN).
        num_epi_load_stages: Number of epilogue load pipeline stages.
    """

    # Source tile layout (row_major for TMA compatibility, matches output)
    comptime src_tile_layout = Layout.row_major(
        Self.src_shape[0], Self.src_shape[1]
    )

    # TileTensor-based for source storage and access
    comptime SrcTileArray = SMemTileArray2DRowMajor[
        Self.src_type,
        Self.src_shape[0],
        Self.src_shape[1],
        Self.num_epi_load_stages,
        128,
    ]

    var src_tiles_storage: Self.SrcTileArray.Storage

    @always_inline
    def src_tiles(ref[AddressSpace.SHARED] self) -> Self.SrcTileArray:
        """Get source tile array accessor (TileTensor-based)."""
        return Self.SrcTileArray(self.src_tiles_storage.unsafe_ptr())


# =============================================================================
# Epilogue Load Pipeline Storage - For EpilogueLoad → Epilogue transfer
# =============================================================================


struct EpiLoadPipelineStorage[num_stages: Int]:
    """Storage for epilogue load pipeline (source C loading).

    For EpilogueLoad warp → Epilogue warps synchronization.
    The epilogue load warp loads source tensor C into SMEM, and the
    epilogue warps consume it for residual operations.

    Producer: EpilogueLoad warp (1 warp, 32 threads)
    Consumer: Epilogue warps (4 warps, 128 threads)

    Parameters:
        num_stages: Number of epilogue load pipeline stages (typically 2).
    """

    # Type alias for barrier array (exposed for type-level access)
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    var barriers: BarrierPair[Self.num_stages]

    @always_inline
    def create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create runtime pipeline from this storage."""
        return self.barriers.create_pipeline()

    @always_inline
    def barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Escape hatch: Get raw barrier pointer."""
        return self.barriers.barriers().ptr


# =============================================================================
# Load Order Barrier Storage - For MainLoad → EpilogueLoad coordination
# =============================================================================


struct LoadOrderBarrierStorage:
    """Storage for load order barrier (mainloop → epilogue load coordination).

    This single barrier coordinates the mainloop load warp with the epilogue
    load warp, ensuring the epilogue load doesn't start before the mainloop
    has issued its prologue TMA operations.

    Protocol:
    1. Mainloop load warp issues prologue loads
    2. Mainloop load warp calls arrive() on this barrier
    3. Epilogue load warp waits on this barrier before starting

    This prevents TMA resource contention between mainloop and epilogue loads.
    """

    comptime BarrierArray = SMemArray[SharedMemBarrier, 1]

    var barrier_storage: Self.BarrierArray.Storage

    @always_inline
    def barrier(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        """Get the load order barrier."""
        return Self.BarrierArray(self.barrier_storage)

    @always_inline
    def ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get raw barrier pointer for initialization."""
        return self.barrier().ptr


# =============================================================================
# Escape Hatch - Raw barrier storage for custom patterns
# =============================================================================


struct RawBarrierStorage[count: Int]:
    """Escape hatch: Raw barrier storage for custom patterns.

    Use this when the standard pipeline storage doesn't fit your needs.
    You're responsible for initialization and synchronization semantics.

    Parameters:
        count: Total number of barriers to allocate.

    Example:
        ```
        # Custom barrier layout for specialized synchronization
        struct MyCustomSmem:
            var custom_barriers: RawBarrierStorage[8]

            def init_custom(ref[SHARED] self):
                ptr = self.custom_barriers.ptr()
                # Custom initialization...
        ```
    """

    comptime Array = SMemArray[SharedMemBarrier, Self.count]

    var storage: Self.Array.Storage

    @always_inline
    def barriers(ref[AddressSpace.SHARED] self) -> Self.Array:
        return Self.Array(self.storage)

    @always_inline
    def ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get raw pointer for custom usage."""
        return self.barriers().ptr


# =============================================================================
# SmemPipelineBundle - Composed pipeline storage + barrier accessors
# =============================================================================


struct SmemPipelineBundle[
    num_group_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_clc_pipeline_stages: Int,
    Payload: TilePayload,
    num_epilogue_load_stages: Int = 0,
]:
    """Composed pipeline storage with unified barrier accessors.

    Bundles InputPipelineStorage, OutputPipelineStorage, ClcPipelineStorage,
    TmemDeallocStorage, and optionally EpiLoadPipelineStorage into a single
    composed struct, eliminating ~60 lines of duplicated pipeline declarations,
    barrier type aliases, and barrier accessor methods from each SMEM struct.

    Parameters:
        num_group_pipeline_stages: Number of grouped pipeline stages for input.
        num_accum_pipeline_stages: Number of accumulator pipeline stages.
        num_clc_pipeline_stages: Number of CLC scheduler pipeline stages.
        Payload: Tile payload type (e.g. StandardTilePayload, BlockScaledTilePayload).
        num_epilogue_load_stages: Number of epilogue load stages (0 = disabled).
    """

    # ========== Pipeline Storage Types ==========
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages, Self.Payload
    ]
    comptime OutputPipeline = OutputPipelineStorage[
        Self.num_accum_pipeline_stages
    ]
    comptime ClcPipeline = ClcPipelineStorage[Self.num_clc_pipeline_stages]
    comptime TmemDeallocPipeline = TmemDeallocStorage
    comptime EpiLoadPipeline = EpiLoadPipelineStorage[
        Self.num_epilogue_load_stages
    ]

    # ========== Barrier Type Aliases ==========
    comptime InputBarriers = Self.InputPipeline.BarrierArray
    comptime AccumBarriers = Self.OutputPipeline.BarrierArray
    comptime ClcBarriers = Self.ClcPipeline.BarrierArray
    comptime ClcThrottleBarriers = Self.ClcPipeline.ThrottleArray
    comptime ClcResponse = Self.ClcPipeline.ResponseArray
    comptime TmemDealloc = Self.TmemDeallocPipeline.BarrierArray
    comptime TmemAddr = Self.TmemDeallocPipeline.AddrArray
    comptime EpiLoadBarriers = Self.EpiLoadPipeline.BarrierArray

    # ========== Storage Fields ==========
    var input_pipeline: Self.InputPipeline
    var output_pipeline: Self.OutputPipeline
    var clc_pipeline: Self.ClcPipeline
    var tmem_dealloc_pipeline: Self.TmemDeallocPipeline
    var epi_load_pipeline: Self.EpiLoadPipeline

    # ========== Barrier Accessors ==========
    @always_inline
    def input_barriers(ref[AddressSpace.SHARED] self) -> Self.InputBarriers:
        """Returns input tile pipeline barriers."""
        return self.input_pipeline.barriers.barriers()

    @always_inline
    def accum_barriers(ref[AddressSpace.SHARED] self) -> Self.AccumBarriers:
        """Returns accumulator pipeline barriers."""
        return self.output_pipeline.barriers.barriers()

    @always_inline
    def clc_full(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        """Returns CLC full barriers."""
        return self.clc_pipeline.full()

    @always_inline
    def clc_empty(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        """Returns CLC empty barriers."""
        return self.clc_pipeline.empty()

    @always_inline
    def clc_throttle(ref[AddressSpace.SHARED] self) -> Self.ClcThrottleBarriers:
        """Returns CLC throttle barriers."""
        return self.clc_pipeline.throttle()

    @always_inline
    def clc_response(ref[AddressSpace.SHARED] self) -> Self.ClcResponse:
        """Returns CLC response storage."""
        return self.clc_pipeline.response()

    @always_inline
    def tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        """Returns TMEM deallocation barrier."""
        return self.tmem_dealloc_pipeline.barrier()

    @always_inline
    def tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        """Returns TMEM address storage."""
        return self.tmem_dealloc_pipeline.addr()

    @always_inline
    def epilogue_load_barriers(
        ref[AddressSpace.SHARED] self,
    ) -> Self.EpiLoadBarriers:
        """Returns epilogue load pipeline barriers."""
        return self.epi_load_pipeline.barriers.barriers()

    @always_inline
    def epilogue_load_barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Returns epilogue load pipeline barrier pointer."""
        return self.epi_load_pipeline.barrier_ptr()


# =============================================================================
# SmemPipelineBundleNoClc - Pipeline bundle without CLC scheduler
# =============================================================================


struct SmemPipelineBundleNoClc[
    num_group_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    Payload: TilePayload,
]:
    """Composed pipeline storage without CLC scheduler.

    Used by kernels with 3-warp specialization (Load, MMA, Epilogue) that
    don't use a scheduler warp (e.g. Grouped1D1DSmem).

    Parameters:
        num_group_pipeline_stages: Number of grouped pipeline stages for input.
        num_accum_pipeline_stages: Number of accumulator pipeline stages.
        Payload: Tile payload type.
    """

    # ========== Pipeline Storage Types ==========
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages, Self.Payload
    ]
    comptime OutputPipeline = OutputPipelineStorage[
        Self.num_accum_pipeline_stages
    ]
    comptime TmemDeallocPipeline = TmemDeallocStorage

    # ========== Barrier Type Aliases ==========
    comptime InputBarriers = Self.InputPipeline.BarrierArray
    comptime AccumBarriers = Self.OutputPipeline.BarrierArray
    comptime TmemDealloc = Self.TmemDeallocPipeline.BarrierArray
    comptime TmemAddr = Self.TmemDeallocPipeline.AddrArray

    # ========== Storage Fields ==========
    var input_pipeline: Self.InputPipeline
    var output_pipeline: Self.OutputPipeline
    var tmem_dealloc_pipeline: Self.TmemDeallocPipeline

    # ========== Barrier Accessors ==========
    @always_inline
    def input_barriers(ref[AddressSpace.SHARED] self) -> Self.InputBarriers:
        """Returns input tile pipeline barriers."""
        return self.input_pipeline.barriers.barriers()

    @always_inline
    def accum_barriers(ref[AddressSpace.SHARED] self) -> Self.AccumBarriers:
        """Returns accumulator pipeline barriers."""
        return self.output_pipeline.barriers.barriers()

    @always_inline
    def tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        """Returns TMEM deallocation barrier."""
        return self.tmem_dealloc_pipeline.barrier()

    @always_inline
    def tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        """Returns TMEM address storage."""
        return self.tmem_dealloc_pipeline.addr()


# =============================================================================
# SmemLayouts - Common SMEM layout definitions for matmul-family kernels
# =============================================================================


struct SmemLayouts[
    a_type: DType,
    b_type: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    OutputM: Int,
    OutputN: Int,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
]:
    """Common SMEM layout definitions for matmul-family kernels.

    Centralizes the A/B/C tile layout computation including the
    transpose-conditional B layout logic, eliminating ~10 lines of
    duplicated layout definitions from each SMEM struct.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        BM: Block tile M dimension.
        BN: Block tile N dimension.
        BK: Block tile K dimension.
        OutputM: Output tile M dimension.
        OutputN: Output tile N dimension.
        a_swizzle: Swizzle mode for A tiles.
        b_swizzle: Swizzle mode for B tiles.
        transpose_b: Whether B is transposed (K-major).
    """

    # Typed layouts (source of truth, new Layout type).
    # Both are K-major: A is inherently K-major, and B is K-major because
    # SM100 MMA (MmaOpSM100_SS) requires transpose_b=True.
    comptime a_smem_layout_typed = tile_layout_k_major_typed[
        Self.a_type, Self.BM, Self.BK, Self.a_swizzle
    ]
    comptime b_smem_layout_typed = tile_layout_k_major_typed[
        Self.b_type, Self.BN, Self.BK, Self.b_swizzle
    ]

    # Element counts derived from typed layouts.
    comptime a_tile_elems: Int = Self.a_smem_layout_typed.static_product
    comptime b_tile_elems: Int = Self.b_smem_layout_typed.static_product
