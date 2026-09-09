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

"""Shared memory layout for block-scaled SM100 matmul.

Provides A/B/C tile storage plus scaling factor tile storage (SFA, SFB)
following MXFP8 layout conventions. Also includes pipeline barriers and TMEM
state.

The tile storage, derived constants, layouts, and accessors are factored into
BlockScaledTileCore and shared with GroupedBlockScaledSmem and Grouped1D1DSmem.
Each SMEM struct is a thin wrapper that adds the appropriate pipeline bundle.
"""

from std.math import align_up
from layout.tensor_core_async import tile_sf_layout_k_major
from std.utils.index import IndexList

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from structured_kernels.pipeline_storage import (
    BlockScaledTileStorage,
    SmemPipelineBundle,
)
from ..structured_kernels.tile_pipeline import BlockScaledTilePayload


# =============================================================================
# BlockScaledTileCore - Shared tile storage, constants, and accessors
# =============================================================================


struct BlockScaledTileCore[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """Core tile storage for block-scaled matmul SMEM structs.

    Contains derived constants, layouts, tile storage, tile accessors, and
    size utilities. Shared between BlockScaledSmem (CLC),
    GroupedBlockScaledSmem (CLC + TMA descriptors), and Grouped1D1DSmem (no CLC).

    Parameters:
        a_type: Element type of the A operand matrix.
        b_type: Element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        sfa_dtype: Element type of the A operand scaling factors.
        sfb_dtype: Element type of the B operand scaling factors.
        transpose_b: Whether the B operand is stored in transposed layout.
        config: Block-scaled matmul configuration providing tile shapes,
            pipeline stage counts, and scaling factor layout parameters.
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
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size
    # This determines how many K elements each scaling factor covers
    comptime SF_K_GROUP_SIZE = sf_k_group_size[Self.config]()

    # SF layouts use config.vec_sf_size (MXFP8=32, NVFP4=16) and num_sf_k_tiles
    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        align_up(Self.MMA_N, SF_MN_GROUP_SIZE),
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    # SF tile dimensions (computed via shared helper functions)
    comptime SF_BK = sf_bk[Self.config]()
    comptime SFA_DIM0 = sfa_dim0[Self.config]()
    comptime SFA_DIM1 = sfa_dim1[Self.config]()
    comptime SFB_DIM0 = sfb_dim0[Self.config]()
    comptime SFB_DIM1 = sfb_dim1[Self.config]()

    # ========== Tile Storage ==========
    comptime Tiles = BlockScaledTileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        Self.OutputM,
        Self.OutputN,
        IndexList[2](Self.SFA_DIM0, Self.SFA_DIM1),  # SFA shape
        IndexList[2](Self.SFB_DIM0, Self.SFB_DIM1),  # SFB shape
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Tile array type aliases
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime SFATileArray = Self.Tiles.SFATileArray
    comptime SFBTileArray = Self.Tiles.SFBTileArray

    # Tile payload type alias (used by pipeline bundles)
    comptime Payload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        IndexList[2](Self.SFA_DIM0, Self.SFA_DIM1),  # SFA shape
        IndexList[2](Self.SFB_DIM0, Self.SFB_DIM1),  # SFB shape
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
    def sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        """Get SFA tile array accessor."""
        return self.tiles.sfa_tiles()

    @always_inline
    def sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.tiles.sfb_tiles()

    # ========== Size Utilities ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.ATileArray.num_elements + Self.BTileArray.num_elements

    @staticmethod
    @always_inline
    def sf_pipeline_size() -> Int:
        """Total size of SFA+SFB tiles for all pipeline stages (in elements)."""
        return Self.SFATileArray.num_elements + Self.SFBTileArray.num_elements

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return (
            Self.ab_pipeline_size()
            + Self.sf_pipeline_size()
            + Self.c_output_size()
        )


# =============================================================================
# BlockScaledSmem - SMEM wrapper with CLC pipeline
# =============================================================================


struct BlockScaledSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """SMEM struct for block-scaled matmul with CLC scheduler pipeline.

    Thin wrapper over BlockScaledTileCore + SmemPipelineBundle.

    Parameters:
        a_type: Element type of the A operand matrix.
        b_type: Element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        sfa_dtype: Element type of the A operand scaling factors.
        sfb_dtype: Element type of the B operand scaling factors.
        transpose_b: Whether the B operand is stored in transposed layout.
        config: Block-scaled matmul configuration providing tile shapes,
            pipeline stage counts, and scaling factor layout parameters.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockScaledTileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
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
    def sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFATileArray:
        """Get SFA tile array accessor."""
        return self.core.sfa_tiles()

    @always_inline
    def sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.core.sfb_tiles()

    # ========== Size Utilities (forwarding) ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.Core.ab_pipeline_size()

    @staticmethod
    @always_inline
    def sf_pipeline_size() -> Int:
        """Total size of SFA+SFB tiles for all pipeline stages (in elements)."""
        return Self.Core.sf_pipeline_size()

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.Core.c_output_size()

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return Self.Core.total_tile_size()


# =============================================================================
# Scaling Factor Dimension Helpers
# =============================================================================


@always_inline
def sf_k_group_size[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SF_K_GROUP_SIZE from config.

    Parameters:
        config: Block-scaled matmul configuration providing the scaling
            factor vector size used to compute the K group size.
    """
    return SF_ATOM_K * config.vec_sf_size


@always_inline
def sf_bk[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SF_BK from config.

    Parameters:
        config: Block-scaled matmul configuration providing the scaling
            factor K tile count and vector size.
    """
    return sf_k_group_size[config]() * config.num_sf_k_tiles


@always_inline
def sfa_dim0[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFA first dimension from config.

    Parameters:
        config: Block-scaled matmul configuration providing the block
            tile M dimension.
    """
    return (config.block_tile_shape[0] // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]


@always_inline
def sfa_dim1[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFA second dimension from config.

    Parameters:
        config: Block-scaled matmul configuration providing the block
            tile K dimension and scaling factor parameters.
    """
    return (sf_bk[config]() // (SF_ATOM_K * config.vec_sf_size)) * (
        SF_ATOM_M[1] * SF_ATOM_K
    )


@always_inline
def sfb_dim0[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFB first dimension from config.

    Parameters:
        config: Block-scaled matmul configuration providing the MMA N
            dimension and scaling factor parameters.
    """
    return (
        align_up(config.mma_shape[1], SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE
    ) * SF_ATOM_M[0]


@always_inline
def sfb_dim1[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFB second dimension from config.

    Parameters:
        config: Block-scaled matmul configuration providing the block
            tile K dimension and scaling factor parameters.
    """
    return (sf_bk[config]() // (SF_ATOM_K * config.vec_sf_size)) * (
        SF_ATOM_M[1] * SF_ATOM_K
    )
