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

"""Shared memory layout for blockwise FP8 SM100 matmul.

This module provides the SMEM struct for blockwise FP8 matmul kernels where:
- A-scales are loaded via TMA and stored in SMEM (1D: 1 x BM per stage)
- B-scales are read directly from global memory (not stored in SMEM)
- Scaling is applied post-MMA in CUDA cores, not within the MMA unit

Unlike block-scaled matmul, blockwise FP8 uses register-based accumulation
across K iterations, with scales applied per-iteration.

The tile storage, derived constants, layouts, and accessors are factored into
BlockwiseFP8TileCore and shared with BlockwiseFP8_1D2DSmem. Each SMEM struct
is a thin wrapper that adds the appropriate pipeline bundle.
"""

from std.utils.index import IndexList

from ..structured_kernels.config import MatmulConfig
from structured_kernels.pipeline_storage import (
    BlockwiseFP8TileStorage,
    SmemPipelineBundle,
)
from ..structured_kernels.tile_pipeline import BlockwiseFP8TilePayload


# =============================================================================
# BlockwiseFP8TileCore - Shared tile storage, constants, and accessors
# =============================================================================


struct BlockwiseFP8TileCore[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Core tile storage for blockwise FP8 matmul SMEM structs.

    Contains derived constants, layouts, tile storage, tile accessors, and
    size utilities. Shared between BlockwiseFP8Smem (CLC) and
    BlockwiseFP8_1D2DSmem (no CLC).

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        a_scales_type: Element type of the A blockwise scales.
        transpose_b: Whether B is accessed transposed.
        config: Matmul tile shapes, MMA shapes, and pipeline stage counts.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]
    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages = Self.config.num_output_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages

    # Layout definitions removed — A/B tiles use TileTensor (SMemTileArray2D)
    # with internal_k_major layout. C tiles use SMemTileArray2DRowMajor.
    # Legacy SmemLayouts is no longer needed here.

    # ========== Tile Storage ==========
    comptime Tiles = BlockwiseFP8TileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        Self.OutputM,
        Self.OutputN,
        IndexList[2](1, Self.BM),  # A-scales shape
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Tile array type aliases
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime AScalesTileArray = Self.Tiles.AScalesTileArray

    # Tile payload type alias (used by pipeline bundles)
    comptime Payload = BlockwiseFP8TilePayload[
        Self.a_type,
        Self.b_type,
        Self.a_scales_type,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        IndexList[2](1, Self.BM),  # A-scales shape
        Self.num_pipeline_stages,
    ]

    # ========== Tile Storage Field ==========
    var tiles: Self.Tiles

    # ========== Tile Accessors ==========
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return self.tiles.a_tiles()

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return self.tiles.b_tiles()

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return self.tiles.c_tiles()

    @always_inline
    def a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return self.tiles.a_scales_tiles()

    # ========== Size Utilities ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.ATileArray.num_elements + Self.BTileArray.num_elements

    @staticmethod
    @always_inline
    def a_scales_pipeline_size() -> Int:
        """Total size of A-scales tiles for all pipeline stages (in elements).
        """
        return Self.AScalesTileArray.num_elements

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+A-scales+C) in elements."""
        return (
            Self.ab_pipeline_size()
            + Self.a_scales_pipeline_size()
            + Self.c_output_size()
        )


# =============================================================================
# BlockwiseFP8Smem - SMEM wrapper with CLC pipeline
# =============================================================================


struct BlockwiseFP8Smem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """SMEM struct for blockwise FP8 matmul with CLC scheduler pipeline.

    Thin wrapper over BlockwiseFP8TileCore + SmemPipelineBundle.

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        a_scales_type: Element type of the A blockwise scales.
        transpose_b: Whether B is accessed transposed.
        config: Matmul tile shapes, MMA shapes, and pipeline stage counts.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockwiseFP8TileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== Storage Fields ==========
    var core: Self.Core

    # ========== Pipeline Storage ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.Core.num_group_pipeline_stages,
        Self.Core.num_accum_pipeline_stages,
        Self.config.num_clc_pipeline_stages,
        Self.Core.Payload,
    ]
    var pipelines: Self.Pipelines

    # ========== Tile Accessors (forwarding) ==========
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.ATileArray:
        """Get A tile array accessor."""
        return self.core.a_tiles()

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.BTileArray:
        """Get B tile array accessor."""
        return self.core.b_tiles()

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.CTileArray:
        """Get C tile array accessor."""
        return self.core.c_tiles()

    @always_inline
    def a_scales_tiles(
        ref[AddressSpace.SHARED] self,
    ) -> Self.Core.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return self.core.a_scales_tiles()

    # ========== Size Utilities (forwarding) ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.Core.ab_pipeline_size()

    @staticmethod
    @always_inline
    def a_scales_pipeline_size() -> Int:
        """Total size of A-scales tiles for all pipeline stages (in elements).
        """
        return Self.Core.a_scales_pipeline_size()

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.Core.c_output_size()

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+A-scales+C) in elements."""
        return Self.Core.total_tile_size()
