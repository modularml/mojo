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

"""Shared memory layout for SM100 Conv2D kernel.

This module provides the Conv2dSmem struct which defines the shared memory
organization for the Conv2D fprop kernel. The layout is similar to
B200MatmulSmem but uses conv-specific naming.

SMEM Organization:
- Activation tiles (from im2col): Multi-stage pipelined
- Filter tiles: Multi-stage pipelined
- Output tiles: Double-buffered for epilogue
- Pipeline barriers: For producer-consumer synchronization
- CLC barriers: For work scheduling
- TMEM storage: For accumulator address sharing
"""


from layout import Layout
from std.utils.index import IndexList
from layout.tensor_core_async import tile_layout_k_major_typed

# Import pipeline storage from matmul structured kernels
from structured_kernels.pipeline_storage import (
    StandardTileStorage,
    OutputTileStorage,
    SourceTileStorage,
    EpiLoadPipelineStorage,
    LoadOrderBarrierStorage,
    SmemPipelineBundle,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline import (
    StandardTilePayload,
)

from .conv_config import Conv2dConfig


# =============================================================================
# Conv2dSmem - Shared memory layout for Conv2D kernel
# =============================================================================


struct Conv2dSmem[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    *,
    config: Conv2dConfig[act_type, filter_type, out_type],
]:
    """Shared memory layout for SM100 Conv2D fprop kernel.

    This struct manages shared memory allocation for:
    - Activation tiles (after im2col transformation)
    - Filter tiles
    - Output tiles for accumulation
    - Synchronization barriers

    The layout mirrors B200MatmulSmem but with conv-specific semantics:
    - A tiles = im2col'd activation (M x K where M = N*H*W, K = C*R*S)
    - B tiles = filter (transposed, K x N where K = C*R*S, N = K_out)
    - C tiles = output (M x N)

    Parameters:
        act_type: Activation data type.
        filter_type: Filter data type.
        out_type: Output data type.
        config: Kernel configuration.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages: Int = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages: Int = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages: Int = Self.config.num_accum_pipeline_stages
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages

    # ========== Layout Definitions ==========
    # Activation tiles use K-major layout (im2col'd)
    comptime act_smem_elements = tile_layout_k_major_typed[
        Self.act_type,
        Self.BM,
        Self.BK,
        swizzle_mode=Self.config.a_swizzle,
    ].static_product

    # Filter tiles use K-major layout (transposed GEMM B)
    comptime filter_smem_elements = tile_layout_k_major_typed[
        Self.filter_type,
        Self.BN,
        Self.BK,
        swizzle_mode=Self.config.b_swizzle,
    ].static_product

    # Output tiles use row-major layout
    comptime out_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # ========== Tile Storage ==========
    # Reuse StandardTileStorage from matmul (same structure)
    comptime InputTiles = StandardTileStorage[
        Self.act_type,
        Self.filter_type,
        # Activation tile dimensions (BM x BK)
        IndexList[2](Self.BM, Self.BK),
        # Filter tile dimensions (BN x BK)
        IndexList[2](Self.BN, Self.BK),
        Self.num_pipeline_stages,
    ]

    comptime OutputTiles = OutputTileStorage[
        Self.out_type,
        Self.OutputM,
        Self.OutputN,
        Self.num_output_stages,
    ]

    # Source tile storage for residual operations (D = Conv + beta*C).
    #
    # The epilogue iterates `num_inner_stages = MMA_N / OutputN` sub-tiles
    # per block tile (e.g. 4 for the 1-SM default: BN=128 / OutputN=32).
    # Each sub-tile of source must be present before the matching
    # epilogue stage runs. To match the CUTLASS
    # `sm100_epilogue_tma_warpspecialized` pattern (one TMA load per
    # sub-tile, pipelined across `StagesC` SMEM buffers), size the
    # source SMEM as `num_inner_stages` buffers of `(OutputM, OutputN)`.
    # A previous (BM, OutputN) single-buffer layout caused stages > 0 to
    # read uninitialized SMEM (MODELS-1484).
    comptime num_epi_load_stages: Int = (
        Self.config.mma_shape[1] // Self.OutputN
    )
    comptime SourceTiles = SourceTileStorage[
        Self.out_type,  # Source C has same type as output D
        IndexList[2](Self.OutputM, Self.OutputN),
        Self.num_epi_load_stages,
    ]

    # Re-export tile array types
    comptime ActTileArray = Self.InputTiles.ATileArray
    comptime FilterTileArray = Self.InputTiles.BTileArray
    comptime OutTileArray = Self.OutputTiles.CTileArray
    comptime SrcTileArray = Self.SourceTiles.SrcTileArray  # Source C tiles (TileTensor)

    # ========== Storage Fields ==========
    var input_tiles: Self.InputTiles
    var output_tiles: Self.OutputTiles
    var source_tiles: Self.SourceTiles

    # ========== Tile Accessors ==========
    @always_inline
    def act_tiles(ref[AddressSpace.SHARED] self) -> Self.ActTileArray:
        """Get activation tiles (im2col'd)."""
        return self.input_tiles.a_tiles()

    @always_inline
    def filter_tiles(ref[AddressSpace.SHARED] self) -> Self.FilterTileArray:
        """Get filter tiles."""
        return self.input_tiles.b_tiles()

    @always_inline
    def out_tiles(ref[AddressSpace.SHARED] self) -> Self.OutTileArray:
        """Get output tiles."""
        return self.output_tiles.c_tiles()

    @always_inline
    def src_tiles(ref[AddressSpace.SHARED] self) -> Self.SrcTileArray:
        """Get source C tiles (for residual operations)."""
        return self.source_tiles.src_tiles()

    # ========== Pipeline Storage (Composed Bundle) ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.num_group_pipeline_stages,
        Self.num_accum_pipeline_stages,
        Self.num_clc_pipeline_stages,
        StandardTilePayload[
            Self.act_type,
            Self.filter_type,
            IndexList[2](Self.BM, Self.BK),
            IndexList[2](Self.BN, Self.BK),
            Self.num_pipeline_stages,
        ],
    ]
    var pipelines: Self.Pipelines

    # ========== Conv2D-specific Pipeline Storage ==========
    # Epilogue load pipeline (EpilogueLoad warp → Epilogue warps)
    # Synchronizes source C loading with epilogue consumption
    comptime EpiLoadPipeline = EpiLoadPipelineStorage[Self.num_epi_load_stages]

    # Load order barrier (MainLoad warp → EpilogueLoad warp)
    # Ensures epilogue loads don't start before mainloop prologue completes
    comptime LoadOrderBarrier = LoadOrderBarrierStorage

    var epi_load_pipeline: Self.EpiLoadPipeline
    var load_order_barrier: Self.LoadOrderBarrier

    # ========== Conv2D-specific Barrier Type Aliases ==========
    comptime EpiLoadBarriers = Self.EpiLoadPipeline.BarrierArray
    comptime LoadOrderBarriers = Self.LoadOrderBarrier.BarrierArray

    # ========== Conv2D-specific Barrier Accessors ==========
    @always_inline
    def epi_load_barriers(
        ref[AddressSpace.SHARED] self,
    ) -> Self.EpiLoadBarriers:
        """Get epilogue load pipeline barriers.

        Used for synchronization between EpilogueLoad warp (producer)
        and Epilogue warps (consumers) for source C tensor loading.
        """
        return self.epi_load_pipeline.barriers.barriers()

    @always_inline
    def get_load_order_barrier(
        ref[AddressSpace.SHARED] self,
    ) -> Self.LoadOrderBarriers:
        """Get load order barrier.

        Used to coordinate MainLoad warp with EpilogueLoad warp, ensuring
        epilogue loads don't start before mainloop prologue completes.
        """
        return self.load_order_barrier.barrier()
