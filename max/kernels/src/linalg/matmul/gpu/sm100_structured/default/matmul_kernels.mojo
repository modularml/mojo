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

"""SM100 Default Matmul Kernel - Standard FP8/BF16 warp-specialized kernel.

This module contains the default SM100 matmul kernel implementation:
- B200MatmulSmem: Shared memory layout for the kernel
- BlackwellMatmulSM100Kernel: Main kernel struct with run() and run_splitk()
- BlackwellMatmulSM100FallbackKernel: Simple fallback kernel

Shared components (WarpRole, KernelContext) are in kernel_common.mojo.
Output pipeline (TileWriter, copy_accum_to_gmem) is in output_writer.mojo.
Low-level epilogue components (TMAStoreExecutor, etc.) are in epilogue_components.mojo.

The kernel implements a warp-specialized architecture:
- Scheduler warp: CLC-based tile scheduling
- TMA Load warp: Async memory transfers
- MMA warp: Tensor core operations with TMEM accumulators
- Epilogue warps: Output from TMEM to GMEM via TileWriter
"""

from std.math import ceildiv
from std.sys import align_of, size_of

from std.memory import UnsafePointer
from max.gpu import WARP_SIZE, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, lane_id
from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
    fence_mbarrier_init,
    CacheEviction,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    PDLLevel,
    wait_on_dependent_grids,
)
from max.gpu.sync import async_copy_arrive, syncwarp
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
    coord,
    row_major,
)
from layout.tile_layout import Layout as _NewLayout
from structured_kernels.tile_types import (
    SMemTile as TTSMemTile,
    SMemTileArray2DRowMajor,
    TmaOpType,
    static_row_major,
    tma_desc_layout_3d,
    tma_desc_layout_3d_explicit_inner,
)
from layout.tile_layout import _IntToComptimeInt
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)
from layout.tma_async import SharedMemBarrier

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..structured_kernels.config import MatmulConfig, OutputPipelineConfig
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    MbarPtr,
    StandardTilePayload,
    ProducerTiles,
    ConsumerTiles,
    OutputTilePipeline,
)
from structured_kernels.barriers import WarpGroupBarrier
from structured_kernels.pipeline_storage import (
    StandardTileStorage,
    OutputTileStorage,
    SmemPipelineBundle,
    SmemLayouts,
)
from structured_kernels.pipeline import ProducerConsumerPipeline
from ..structured_kernels.tmem import (
    TmemAllocation,
    TmemTensor,
    TmemDeallocBarrier,
)
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)
from ..structured_kernels.tile_loader import TileLoader
from ..structured_kernels.tile_scheduler import TileScheduler, WorkIterator
from ..structured_kernels.tile_scheduler_splitk import (
    TileScheduler as TileSchedulerSplitK,
)
from linalg.structuring import SMemPtr
from linalg.matmul.gpu.profiler import MatmulProfileWarp
from comm import MAX_GPUS, Signal
from comm.sync import _multi_gpu_barrier

# Import shared kernel components from kernel_common
from structured_kernels.kernel_common import (
    WarpRole,
    KernelContext,
    compute_clc_barrier_counts,
    compute_accum_barrier_counts,
    compute_input_consumer_count,
    init_core_barriers,
    init_clc_barriers,
)

# Import output pipeline from output_writer module
from ..structured_kernels.output_writer import TileWriter, StandardOutputWriter
from ..structured_kernels.output_writer_trait import OutputWriter


# =============================================================================
# B200MatmulSmem - Shared memory layout for SM100 matmul
# =============================================================================


struct B200MatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Shared memory layout for B200 SM100 matrix multiplication kernel.

    This struct manages the shared memory allocation for:
    - Input tiles (A and B matrices) with multi-stage pipelining
    - Output tile (C matrix) for accumulation
    - Synchronization barriers for producer-consumer coordination
    - CLC (Cluster Launch Control) barriers and response storage
    - TMEM (Tensor Memory) address and deallocation barrier

    The memory is organized to support asynchronous TMA loads and efficient
    bank-conflict-free access patterns for tensor core operations.

    Type aliases are provided for tile types (ATile, BTile, CTile) to enable
    cleaner function signatures.

    Parameters:
        a_type: Element type of the A input matrix tiles in shared memory.
        b_type: Element type of the B input matrix tiles in shared memory.
        c_type: Element type of the C output matrix tiles in shared memory.
        transpose_b: Whether B is stored transposed (K-major), selecting the
            B tile layout.
        config: `MatmulConfig` holding block tile shape, MMA shape, output
            tile shape, pipeline stage counts, and swizzle modes.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages: Int = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages: Int = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages

    # ========== Layout Definitions ==========
    comptime Layouts = SmemLayouts[
        Self.a_type,
        Self.b_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.OutputM,
        Self.OutputN,
        Self.config.a_swizzle,
        Self.config.b_swizzle,
        Self.transpose_b,
    ]

    # ========== Tile Storage (Single Source of Truth) ==========
    # Input tiles: A and B matrices
    # Tiles use TileTensor with swizzled layouts, passed directly to TMA/MMA.
    comptime InputTiles = StandardTileStorage[
        Self.a_type,
        Self.b_type,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        Self.num_pipeline_stages,
    ]
    # Output tiles: C matrix (different stage count)
    comptime OutputTiles = OutputTileStorage[
        Self.c_type,
        Self.OutputM,
        Self.OutputN,
        Self.num_output_stages,
    ]

    # Epilogue load tiles: allocated only when use_tma_epilogue_load=True (0 stages → zero-sized).
    # 1D bias: 1×MMA_N per stage, one load per output tile (like AB_swapped 2D).
    # non-AB_swapped 2D: each stage is BM×stageN (= OutputN); num_tma_epilogue_pipeline_stages stages.
    # AB_swapped 2D: transposed layout (MMA_N×BM per stage) requires all MMA_N rows
    #   simultaneously for ldmatrix.trans; keep full tile with num_accum_pipeline_stages.
    comptime num_epilogue_load_stages: Int = (
        Self.config.num_accum_pipeline_stages if (
            Self.config.AB_swapped or Self.config.epilogue_is_1d
        ) else Self.config.num_tma_epilogue_pipeline_stages
    ) if Self.config.use_tma_epilogue_load else 0
    comptime epilogue_load_tile_rows: Int = 1 if Self.config.epilogue_is_1d else (
        Self.MMA_N if Self.config.AB_swapped else Self.BM
    )
    comptime epilogue_load_tile_cols: Int = (
        Self.BM if Self.config.AB_swapped else Self.MMA_N
    ) if Self.config.epilogue_is_1d else (
        Self.BM if Self.config.AB_swapped else Self.OutputN
    )
    comptime EpilogueLoadTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.epilogue_load_tile_rows,
        Self.epilogue_load_tile_cols,
        Self.num_epilogue_load_stages,
        128,
    ]

    # Re-export tile array types for external use
    # Re-export tile array types
    comptime ATileArray = Self.InputTiles.ATileArray
    comptime BTileArray = Self.InputTiles.BTileArray
    comptime CTileArray = Self.OutputTiles.CTileArray

    # ========== Tile Storage Fields ==========
    var input_tiles: Self.InputTiles
    var output_tiles: Self.OutputTiles
    var epilogue_load_tiles_storage: Self.EpilogueLoadTileArray.Storage

    # ========== Tile Accessors (Delegated) ==========
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        return self.input_tiles.a_tiles()

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        return self.input_tiles.b_tiles()

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return self.output_tiles.c_tiles()

    @always_inline
    def epilogue_load_tiles(
        ref[AddressSpace.SHARED] self,
    ) -> Self.EpilogueLoadTileArray:
        return Self.EpilogueLoadTileArray(
            self.epilogue_load_tiles_storage.unsafe_ptr()
        )

    # ========== Pipeline Storage (Composed Bundle) ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.num_group_pipeline_stages,
        Self.num_accum_pipeline_stages,
        Self.num_clc_pipeline_stages,
        StandardTilePayload[
            Self.a_type,
            Self.b_type,
            IndexList[2](Self.BM, Self.BK),  # A tile shape
            IndexList[2](Self.BN, Self.BK),  # B tile shape
            Self.num_pipeline_stages,
        ],
        Self.num_epilogue_load_stages,
    ]
    var pipelines: Self.Pipelines

    # ========== Size Calculations ==========

    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.ATileArray.num_elements + Self.BTileArray.num_elements

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    def epilogue_load_tile_size() -> Int:
        """Size of epilogue load tiles for all stages (in elements). Zero when config.use_tma_epilogue_load=False.
        """
        return Self.EpilogueLoadTileArray.num_elements

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+C+epilogue load) in elements."""
        return (
            Self.ab_pipeline_size()
            + Self.c_output_size()
            + Self.epilogue_load_tile_size()
        )


# ===----------------------------------------------------------------------=== #
# BlackwellMatmulSM100Kernel - Structured kernel for SM100 matrix multiplication
# ===----------------------------------------------------------------------=== #


struct BlackwellMatmulSM100Kernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    # Configuration
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Cluster shape (must match config, needed for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional features
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
    # Injected output-writer policy (see structured_kernels/output_writer_trait).
    # Defaults to a local TMA store
    output_writer_type: OutputWriter = StandardOutputWriter,
    # Override C TMA descriptor box middle-dim (row count per TMA).
    # 0 means "use c_tile_dim0" (default, whole SMEM tile per TMA — prefill).
    # Decode wants this set to 1 (one row per TMA).
    c_desc_dim0_override: Int = 0,
]:
    """Blackwell SM100 GEMM kernel with warp specialization.

    This struct unifies all parameters and derived types for the SM100
    matmul kernel, providing:
    - Compile-time parameter validation
    - Centralized derived type computation
    - Factory methods for kernel components
    - Multiple kernel entry points (standard, split-k)

    The SM100 kernel uses:
    - Tensor Memory (TMEM) for MMA accumulators
    - Cluster Launch Control (CLC) for dynamic tile scheduling
    - Warp specialization: Scheduler, TMA Load, MMA, Epilogue warps
    - Software pipelining for overlapping compute and memory operations

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix; must be `bfloat16`,
            `float8_e4m3fn`, or `float32`.
        transpose_b: Whether B is stored transposed (K-major); must be `True`.
        config: `MatmulConfig` holding block tile shape, MMA shape, output
            tile shape, pipeline stage counts, and swizzle modes.
        cluster_shape: Thread block cluster dimensions, must match
            `config.cluster_shape` (defaults to `(1, 1, 1)`).
        elementwise_lambda_fn: Optional epilogue function applied to output
            elements after MMA (defaults to `None`).
        elementwise_compute_lambda_fn: Optional fused compute epilogue
            applied during the MMA accumulation (defaults to `None`).
        pdl_level: Programmatic Dependent Launch level for inter-grid
            dependency ordering (defaults to `PDLLevel()`, off).
        max_profiled_tiles_per_SM: Maximum number of tiles to profile per SM
            (defaults to 0, profiling disabled).
        output_writer_type: Output writer policy controlling how C tiles are
            written to global memory (defaults to `StandardOutputWriter`).
        c_desc_dim0_override: Override for the C TMA descriptor box row count
            per TMA store; 0 uses the full SMEM tile dim0 (defaults to 0).
    """

    # ========== Derived Constants (from config) ==========

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime accum_type = Self.config.accum_type
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # MMA tile counts
    comptime num_m_mmas = Self.BM // (Self.MMA_M // Self.cta_group)
    comptime num_n_mmas = Self.BN // (Self.MMA_N // Self.cta_group)
    comptime num_k_mmas = Self.BK // Self.MMA_K

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE
    comptime EPILOGUE_LOAD_THREADS = WARP_SIZE if Self.config.use_tma_epilogue_load else 0

    # Total threads per block
    comptime NUM_THREADS = (
        Self.SCHEDULER_THREADS
        + Self.TMA_LOAD_THREADS
        + Self.MMA_THREADS
        + Self.EPILOGUE_THREADS
        + Self.EPILOGUE_LOAD_THREADS
    )

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Self.config.k_group_size
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration — divide all 512 columns evenly among accum stages.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = Self.NUM_TMEM_COLS // Self.num_accum_pipeline_stages

    comptime register_based_epilogue = Self.config.register_based_epilogue

    # Output pipeline config (bundles accum stages, stride, and cta_group)
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Barrier Arrival Counts ==========

    comptime _clc_barrier_counts = compute_clc_barrier_counts[
        Self.SCHEDULER_THREADS,
        Self.TMA_LOAD_THREADS,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS + Self.EPILOGUE_LOAD_THREADS,
        Self.CLUSTER_SIZE,
        Self.cta_group,
    ]()
    comptime clc_producer_arv_count = Self._clc_barrier_counts[0]
    comptime clc_consumer_arv_count = Self._clc_barrier_counts[1]
    comptime clc_throttle_producer_arv_count = Self._clc_barrier_counts[2]
    comptime clc_throttle_consumer_arv_count = Self._clc_barrier_counts[3]

    comptime _accum_barrier_counts = compute_accum_barrier_counts[
        Self.EPILOGUE_THREADS, Self.cta_group
    ]()
    comptime accum_pipeline_producer_arv_count = Self._accum_barrier_counts[0]
    comptime accum_pipeline_consumer_arv_count = Self._accum_barrier_counts[1]

    comptime _bias_tile_elems = Self.BM if Self.config.AB_swapped else Self.MMA_N
    comptime epi_load_producer_arv_count: Int32 = (
        Int32(
            ceildiv(Self._bias_tile_elems, 8)
        ) if Self.config.epilogue_is_1d else 1
    )
    comptime epi_load_consumer_arv_count: Int32 = Int32(Self.EPILOGUE_THREADS)

    # ========== Shared Memory Layout Types ==========

    comptime a_smem_layout = tile_layout_k_major_typed[
        Self.a_type, Self.BM, Self.BK, swizzle_mode=Self.config.a_swizzle
    ].to_layout()

    comptime b_smem_layout = tile_layout_k_major_typed[
        Self.b_type, Self.BN, Self.BK, swizzle_mode=Self.config.b_swizzle
    ].to_layout() if Self.transpose_b else tile_layout_mn_major_typed[
        Self.b_type, Self.BN, Self.BK, swizzle_mode=Self.config.b_swizzle
    ].to_layout()

    comptime SmemType = B200MatmulSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== MMA Operation Type ==========

    comptime MmaOp = MmaOpSM100_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=Self.transpose_b,
    ]

    # ========== Tile Scheduler Type ==========

    comptime Scheduler = TileScheduler[
        num_stages=Self.num_clc_pipeline_stages,
        cluster_shape=Index[dtype=DType.uint32](
            Self.config.cluster_shape[0],
            Self.config.cluster_shape[1],
            Self.config.cluster_shape[2],
        ),
        block_swizzle_size=Self.config.block_swizzle_size,
        rasterize_order=Self.config.raster_order,
    ]

    # ========== Tile Pipeline Type ==========

    comptime TilePayload = StandardTilePayload[
        Self.a_type,
        Self.b_type,
        IndexList[2](Self.BM, Self.BK),  # A tile shape
        IndexList[2](Self.BN, Self.BK),  # B tile shape
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipeline = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== Tile Loader Types ==========
    # Loaders wrapping TMA operations. Orchestration is in kernel.
    # Origins inferred from constructor Pointer arguments.

    # TileLoader types are constructed at call sites with inferred tma_origin.
    # See load_input_tiles() and the run/run_splitk loader construction.

    # Constants for TMA expected_bytes calculation
    comptime a_expected_bytes = Self.SmemType.Layouts.a_tile_elems * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.SmemType.Layouts.b_tile_elems * size_of[
        Self.b_type
    ]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes + Self.b_expected_bytes
    ) * Self.config.k_group_size

    # TMA descriptor layout sizes for peer CTA slicing
    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime a_tile_dim0 = Self.BM // Self.CLUSTER_N
    comptime b_tile_dim0 = Self.BN // (Self.CLUSTER_M // Self.cta_group)
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()
    comptime c_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.c_type
    ]()
    comptime epi_load_swizzle = Self.config.epi_load_swizzle
    comptime epi_load_swizzle_elems = Self.epi_load_swizzle.bytes() // size_of[
        Self.c_type
    ]()
    # C tile shape depends on MMA shape, cta_group, and AB_swapped.
    # Must match host-side create_tensor_tile tile/desc dimensions exactly.
    # When AB_swapped, output_tile_shape is transposed, so OutputM is the
    # N-dimension and always used as dim0. When not AB_swapped and MMA_M=128
    # with cta_group=2, dim0 is forced to 64 (two 64-row halves).
    comptime c_tile_dim0 = Self.OutputM if (
        Self.MMA_M == 256 or Self.cta_group == 1 or Self.config.AB_swapped
    ) else 64
    # When AB_swapped, dim1 uses swizzle elements; otherwise OutputN.
    comptime c_tile_dim1 = Self.c_swizzle_elems if (
        Self.config.AB_swapped
    ) else Self.OutputN

    # 3D TMA layouts (primary, used by run())
    comptime ATileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.a_tile_dim0, Self.BK]
    ]
    comptime ADescLayout = tma_desc_layout_3d[
        Self.a_type, 1, Self.a_tile_dim0, Self.config.a_swizzle
    ]
    comptime BTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.b_tile_dim0, Self.BK]
    ]
    comptime BDescLayout = tma_desc_layout_3d[
        Self.b_type, 1, Self.b_tile_dim0, Self.config.b_swizzle
    ]
    comptime CTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.c_tile_dim0, Self.c_tile_dim1]
    ]
    # For swizzled modes the innermost descriptor dim is capped at the swizzle
    # atom size. For SWIZZLE_NONE there is no such cap (driver only requires
    # boxDim[0] * elem_size % 16 == 0), so use the actual SMEM tile inner
    # dim -- this lets callers (e.g. matmul_rs decode) transfer full SMEM
    # rows per TMA instead of being limited to 16 bytes per call.
    comptime c_desc_inner_elems = (
        Self.c_tile_dim1
    ) if Self.config.c_swizzle == TensorMapSwizzle.SWIZZLE_NONE else (
        Self.config.c_swizzle.bytes() // size_of[Self.c_type]()
    )
    comptime c_desc_dim0 = Self.c_desc_dim0_override if Self.c_desc_dim0_override > 0 else Self.c_tile_dim0
    comptime CDescLayout = tma_desc_layout_3d_explicit_inner[
        1, Self.c_desc_dim0, Self.c_desc_inner_elems
    ]

    # 2D TMA layouts (only for run_splitk)
    comptime ATileLayout_splitk = static_row_major[Self.a_tile_dim0, Self.BK]
    comptime ADescLayout_splitk = static_row_major[
        Self.a_tile_dim0, Self.a_swizzle_elems
    ]
    comptime BTileLayout_splitk = static_row_major[Self.b_tile_dim0, Self.BK]
    comptime BDescLayout_splitk = static_row_major[
        Self.b_tile_dim0, Self.b_swizzle_elems
    ]
    comptime CTileLayout_splitk = static_row_major[
        Self.c_tile_dim0, Self.c_tile_dim1
    ]
    comptime CDescLayout_splitk = static_row_major[
        Self.c_tile_dim0, Self.c_swizzle_elems
    ]
    # TMA descriptor layouts for epilogue load (2D only; 1D uses cp.async.bulk).
    # For 1D, use 2D-default dimensions so the placeholder TMA descriptor is valid.
    comptime _epi_tma_tile_rows: Int = Self.MMA_N if Self.config.AB_swapped else Self.BM
    comptime _epi_tma_tile_cols: Int = Self.BM if Self.config.AB_swapped else Self.OutputN
    comptime EpilogueLoadTileLayout = static_row_major[
        Self._epi_tma_tile_rows,
        Self._epi_tma_tile_cols,
    ]
    comptime EpilogueLoadDescLayout = static_row_major[
        Self._epi_tma_tile_rows,
        Self.epi_load_swizzle_elems if Self.epi_load_swizzle_elems
        > 0 else Self._epi_tma_tile_cols,
    ]

    # 3D TMA operation types (primary, used by run())
    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime CTmaOp = TmaOpType[Self.c_type, Self.CTileLayout, Self.CDescLayout]

    # Bias TMA: 2D (BM, MMA_N) with swizzle derived from c_type and MMA_N.
    comptime EpilogueLoadTmaOp = TmaOpType[
        Self.c_type, Self.EpilogueLoadTileLayout, Self.EpilogueLoadDescLayout
    ]

    # 2D TMA operation types (only for run_splitk)
    comptime ATmaOp_splitk = TmaOpType[
        Self.a_type, Self.ATileLayout_splitk, Self.ADescLayout_splitk
    ]
    comptime BTmaOp_splitk = TmaOpType[
        Self.b_type, Self.BTileLayout_splitk, Self.BDescLayout_splitk
    ]
    comptime CTmaOp_splitk = TmaOpType[
        Self.c_type, Self.CTileLayout_splitk, Self.CDescLayout_splitk
    ]

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Tensor Memory Type ==========
    # TMEM allocation and typed accumulator tensor

    comptime Tmem = TmemAllocation[Self.opc.cta_group]

    # Layout-parameterized TMEM tensor for type-safe accumulator access
    comptime accum_layout = Layout.row_major(Self.MMA_M, Self.MMA_N)
    comptime AccumTensor = TmemTensor[
        Self.accum_type, Self.accum_layout, cta_group=Self.cta_group
    ]

    # ========== Output Tile Pipeline Type ==========
    # Manages MMA→Epilogue pipeline for TMEM accumulator stages

    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    # MMA-Epilogue handoff barrier (barrier_id=1)
    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]

    # TMEM deallocation barrier for cluster synchronization
    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    # MMA warp context (TMEM + dealloc + OutputPipeline)
    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # Epilogue warp context (works in run_splitk, issues in run with k_group>1)
    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Output Tile Writer ==========
    # Instance-based TileWriter with explicit config parameters
    # tma_origin, c_type, c_layout, c_desc_layout inferred from constructor arg
    # batched=True: run() always uses 3D TMA; write_batched for epilogue.
    comptime TileWriterType = TileWriter[
        a_type=Self.a_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=Self.config.AB_swapped,
        c_smem_dim0=Self.SmemType.OutputM,
        c_smem_dim1=Self.SmemType.OutputN,
        num_output_stages=Self.SmemType.num_output_stages,
        num_output_warps=Self.num_output_warps,
        elementwise_lambda_fn=Self.elementwise_lambda_fn,
        elementwise_compute_lambda_fn=Self.elementwise_compute_lambda_fn,
        register_based_epilogue=Self.register_based_epilogue,
        batched=True,
    ]

    # TileWriter for run_splitk (uses 2D TMA, no batch coordinate)
    comptime TileWriterType_splitk = TileWriter[
        a_type=Self.a_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=Self.config.AB_swapped,
        c_smem_dim0=Self.SmemType.OutputM,
        c_smem_dim1=Self.SmemType.OutputN,
        num_output_stages=Self.SmemType.num_output_stages,
        num_output_warps=Self.num_output_warps,
        elementwise_lambda_fn=Self.elementwise_lambda_fn,
        elementwise_compute_lambda_fn=Self.elementwise_compute_lambda_fn,
        register_based_epilogue=Self.register_based_epilogue,
        batched=False,
    ]

    # Number of C TMA descriptors the kernel must receive (1 for the local
    # store; one per peer for reduce-scatter). Comes from the injected policy.
    comptime num_c_tma_descriptors = Self.output_writer_type.num_peers

    @staticmethod
    @always_inline
    def write_output_tile[
        tma_origin: ImmOrigin
    ](
        c_tma_ops: Pointer[
            Array[Self.CTmaOp, Self.num_c_tma_descriptors], tma_origin
        ],
        c_tiles: Self.SmemType.CTileArray,
        stage: Self.OutputPipeline.Stage,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
    ):
        """Write one batched output tile through the injected writer policy.

        Construction of the concrete tile writer happens inside
        `Self.output_writer_type.write_batched`

        Parameters:
            tma_origin: Origin type for the C TMA descriptor memory.

        Args:
            c_tma_ops: Pointer to the array of C TMA descriptors, one per
                peer for reduce-scatter or one for the local store.
            c_tiles: Shared memory C tile array staging output tiles.
            stage: Output pipeline stage holding the TMEM accumulator to
                read.
            tile_coord: `(m, n, k_start)` coordinates of the output tile.
            shape: `(M, N)` problem dimensions for bounds checking.
        """
        Self.output_writer_type.write_batched[
            tma_origin,
            Self.CTmaOp.dtype,
            Self.CTmaOp.rank,
            Self.CTmaOp.tile_shape,
            Self.CTmaOp.desc_shape,
            Self.a_type,
            Self.accum_type,
            Self.config.block_tile_shape,
            Self.config.mma_shape,
            Self.opc,
            Self.config.c_swizzle,
            Self.config.AB_swapped,
            Self.SmemType.OutputM,
            Self.SmemType.OutputN,
            Self.SmemType.num_output_stages,
            Self.num_output_warps,
            Self.elementwise_lambda_fn,
            Self.elementwise_compute_lambda_fn,
            Self.register_based_epilogue,
        ](c_tma_ops, c_tiles, stage, tile_coord, shape)

    # ========== Kernel Context Type ==========
    # Type comptime for KernelContext with this kernel's parameters

    comptime Context = KernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

    # ========== Compile-Time Validation ==========

    @staticmethod
    @always_inline
    def validate_constraints():
        """Validate parameter constraints at compile time."""
        comptime assert Self.c_type in (
            DType.bfloat16,
            DType.float8_e4m3fn,
            DType.float32,
        ), "c_type must be bfloat16, float8_e4m3fn, or float32"
        comptime assert Self.transpose_b, "Only support transposed B (K-major)"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "Only support cta_group == 1 or 2"

        comptime if Self.cta_group == 2:
            comptime assert Self.MMA_M in (
                128,
                256,
            ), "cta_group=2 requires MMA_M == 128 or 256"
        else:
            comptime assert Self.MMA_M in (
                64,
                128,
            ), "cta_group=1 requires MMA_M == 64 or 128"

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    def init_barriers[
        use_tma_epilogue_load: Bool = False
    ](
        ctx: Self.Context,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        clc_throttle: Self.SmemType.Pipelines.ClcThrottleBarriers,
        clc_full: Self.SmemType.Pipelines.ClcBarriers,
        clc_empty: Self.SmemType.Pipelines.ClcBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
        epi_load_barriers: Self.SmemType.Pipelines.EpiLoadBarriers = Self.SmemType.Pipelines.EpiLoadBarriers(
            Self.SmemType.Pipelines.EpiLoadBarriers.ptr_type.unsafe_dangling()
        ),
    ):
        """Initialize barriers. TMA descriptor prefetch is done by each kernel
        entry point before calling this method.

        Parameters:
            use_tma_epilogue_load: Whether to initialize epilogue load barriers
                (defaults to `False`).

        Args:
            ctx: Kernel context with election vars and CTA coordinates.
            input_barriers: Barriers for the input (A/B) tile producer-consumer
                pipeline.
            accum_barriers: Barriers for the MMA-to-epilogue accumulator
                pipeline.
            clc_throttle: Throttle barriers for CLC scheduling.
            clc_full: Full barriers signaling CLC pipeline stage occupancy.
            clc_empty: Empty barriers signaling CLC pipeline stage availability.
            tmem_dealloc: Barrier for TMEM deallocation synchronization across
                the cluster.
            epi_load_barriers: Barriers for the epilogue load producer-consumer
                pipeline (defaults to a dangling pointer when
                `use_tma_epilogue_load` is `False`).
        """
        if ctx.elect_one_warp and ctx.elect_one_thread:
            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M, Self.CLUSTER_N, Self.cta_group
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(Self.EPILOGUE_THREADS * Self.cta_group),
            )

            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Int32(Self.clc_throttle_producer_arv_count),
                Int32(Self.clc_throttle_consumer_arv_count),
            )

            init_clc_barriers[Self.num_clc_pipeline_stages](
                clc_full.ptr,
                clc_empty.ptr,
                Int32(Self.clc_producer_arv_count),
                Int32(Self.clc_consumer_arv_count),
            )

            comptime if use_tma_epilogue_load:
                ProducerConsumerPipeline[
                    Self.SmemType.num_epilogue_load_stages
                ](epi_load_barriers.ptr).init_mbars(
                    Self.epi_load_producer_arv_count,
                    Self.epi_load_consumer_arv_count,
                )

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tmem_stage: Self.OutputPipeline.Stage.Tmem,
        tiles: ConsumerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: MmaOpSM100_SS,
        elect_one_warp: Bool,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations for one pipeline stage.

        This is the core MMA function designed to be called within a consumer
        stage context:

            with consumer.acquire() as tiles:
                Self.mma(stage.tmem, tiles, mma_op, ...)

        Parameters:
            tiles_origin: Origin type for the mutable tile payload memory
                (inferred).

        Args:
            tmem_stage: TMEM stage for accumulators.
            tiles: ConsumerTiles context with encapsulated tile access.
            mma_op: The MMA operation instance.
            elect_one_warp: Whether this warp should execute.
            iter_idx: K iteration index.
            k_start: Starting K iteration (for init_c determination).
        """
        # Get typed accumulator tensor from TMEM stage
        var accum = tmem_stage.tensor[Self.accum_type, Self.accum_layout]()

        if elect_one_sync():
            # Build the A/B MMA SMEM descriptors once from the j=0 tile, then
            # advance the descriptor base by the comptime per-k-group byte
            # stride for j>0 — instead of rebuilding the full descriptor
            # (runtime base-pointer mask + bitfield inserts) for every k-group
            # tile. Consecutive k-group tiles are contiguous in the SMEM tile
            # array (stride = tile elems * dtype size), and the SM100
            # descriptor base address adds linearly, so the advanced descriptor
            # is bit-identical to one freshly built from the j-th tile pointer.
            comptime a_stride_bytes = Self.BM * Self.BK * size_of[Self.a_type]()
            comptime b_stride_bytes = Self.BN * Self.BK * size_of[Self.b_type]()

            var a_tile0, b_tile0 = tiles.payload().get_tile[
                Self.config.k_group_size
            ](tiles.stage(), 0)
            var a_desc_base = mma_op.make_a_desc(a_tile0)
            var b_desc_base = mma_op.make_b_desc(b_tile0)

            comptime for j in range(Self.config.k_group_size):
                var is_first_k = (iter_idx + UInt32(j)) == k_start
                mma_op.mma_from_desc(
                    a_desc_base + (a_stride_bytes * j),
                    b_desc_base + (b_stride_bytes * j),
                    UInt32(accum.offset()),
                    init_c=is_first_k,
                )
            mma_op.commit(tiles.mbar())

    @staticmethod
    @always_inline
    def load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A and B tiles using 3D TMA.

        Uses async_multicast_load_3d with batch coordinate from work_tile_coord[2].
        For non-batched calls, batch coord is 0 (grid_dim.z = 1).

        Parameters:
            tiles_origin: Origin type for the mutable tile payload memory
                (inferred).

        Args:
            a_tma_op: 3D TMA descriptor for A matrix.
            b_tma_op: 3D TMA descriptor for B matrix.
            tiles: ProducerStage context with encapsulated tile access.
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, batch) coordinates.
            a_multicast_mask: Multicast mask for A tiles.
            b_multicast_mask: Multicast mask for B tiles.
            iter_idx: K iteration index (base index for k_group).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # Global memory coordinates
        var a_gmem_m_coord = (
            peer_m_rank * Self.a_tma_rows + work_tile_coord[0] * Self.BM
        )
        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + work_tile_coord[1] * Self.MMA_N
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            # Set expected bytes ONCE for all k_group tiles
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            # Get barrier for TMA multicast loads
            var barrier = tiles.barrier()

            comptime for j in range(Self.config.k_group_size):
                # Get tiles using payload accessor
                var a_tile, b_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                # Peer CTA slice using pointer arithmetic
                var a_peer_tile = type_of(a_tile)(
                    a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                # 3D TMA loads with batch coordinate
                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, a_gmem_m_coord, batch_coord),
                    a_multicast_mask,
                )
                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, b_gmem_n_coord, batch_coord),
                    b_multicast_mask,
                )

    @staticmethod
    @always_inline
    def prefetch_a_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        a_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A tiles only; set full expected bytes (A+B) on the barrier.

        Called before wait_on_dependent_grids() to prefetch the static weight
        matrix (kernel-A in swapAB mode). The barrier will not fire until
        the matching complete_b_tiles() call delivers the remaining B bytes.

        Parameters:
            tiles_origin: Origin type for the mutable tile payload memory
                (inferred).

        Args:
            a_tma_op: 3D TMA descriptor for A matrix.
            tiles: ProducerStage context with encapsulated tile access.
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, batch) coordinates.
            a_multicast_mask: Multicast mask for A tiles.
            iter_idx: K iteration index (base index for k_group).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_m_rank = peer_cta_coord[2]

        var a_gmem_m_coord = (
            peer_m_rank * Self.a_tma_rows + work_tile_coord[0] * Self.BM
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            comptime for j in range(Self.config.k_group_size):
                var a_tile, _ = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                var a_peer_tile = type_of(a_tile)(
                    a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, a_gmem_m_coord, batch_coord),
                    a_multicast_mask,
                )

    @staticmethod
    @always_inline
    def complete_b_tiles(
        b_tma_op: Self.BTmaOp,
        stage: UInt32,
        barrier: MbarPtr,
        payload: Self.TilePayload,
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
    ):
        """Load B tiles into a previously prefetched stage.

        Delivers the remaining B bytes so that the stage barrier fires and
        the consumer can proceed. Pair with prefetch_a_tiles().

        Args:
            b_tma_op: 3D TMA descriptor for B matrix.
            stage: Stage index saved from the prefetch phase.
            barrier: Barrier pointer saved from the prefetch phase.
            payload: Tile payload from the pipeline (gives smem pointers).
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, batch) coordinates.
            b_multicast_mask: Multicast mask for B tiles.
            iter_idx: K iteration index (base index for k_group).
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]

        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + work_tile_coord[1] * Self.MMA_N
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            comptime for j in range(Self.config.k_group_size):
                var _, b_tile = payload.get_tile[Self.config.k_group_size](
                    stage, j
                )

                var b_peer_tile = type_of(b_tile)(
                    b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, b_gmem_n_coord, batch_coord),
                    b_multicast_mask,
                )

    @staticmethod
    @always_inline
    def prefetch_b_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        b_tma_op: Self.BTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load B tiles only; set full expected bytes (A+B) on the barrier.

        Called before wait_on_dependent_grids() to prefetch the static weight
        matrix (kernel-B in non-swapAB mode). The barrier will not fire until
        the matching complete_a_tiles() call delivers the remaining A bytes.

        Parameters:
            tiles_origin: Origin type for the mutable tile payload memory
                (inferred).

        Args:
            b_tma_op: 3D TMA descriptor for B matrix.
            tiles: ProducerStage context with encapsulated tile access.
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, batch) coordinates.
            b_multicast_mask: Multicast mask for B tiles.
            iter_idx: K iteration index (base index for k_group).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]

        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + work_tile_coord[1] * Self.MMA_N
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            comptime for j in range(Self.config.k_group_size):
                var _, b_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                var b_peer_tile = type_of(b_tile)(
                    b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, b_gmem_n_coord, batch_coord),
                    b_multicast_mask,
                )

    @staticmethod
    @always_inline
    def complete_a_tiles(
        a_tma_op: Self.ATmaOp,
        stage: UInt32,
        barrier: MbarPtr,
        payload: Self.TilePayload,
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        a_multicast_mask: UInt16,
        iter_idx: UInt32,
    ):
        """Load A tiles into a previously prefetched stage.

        Delivers the remaining A bytes so that the stage barrier fires and
        the consumer can proceed. Pair with prefetch_b_tiles().

        Args:
            a_tma_op: 3D TMA descriptor for A matrix.
            stage: Stage index saved from the prefetch phase.
            barrier: Barrier pointer saved from the prefetch phase.
            payload: Tile payload from the pipeline (gives smem pointers).
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, batch) coordinates.
            a_multicast_mask: Multicast mask for A tiles.
            iter_idx: K iteration index (base index for k_group).
        """
        var peer_m_rank = peer_cta_coord[2]

        var a_gmem_m_coord = (
            peer_m_rank * Self.a_tma_rows + work_tile_coord[0] * Self.BM
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            comptime for j in range(Self.config.k_group_size):
                var a_tile, _ = payload.get_tile[Self.config.k_group_size](
                    stage, j
                )

                var a_peer_tile = type_of(a_tile)(
                    a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, a_gmem_m_coord, batch_coord),
                    a_multicast_mask,
                )

    @staticmethod
    @always_inline
    def load_input_tiles_splitk[
        a_tma_origin: ImmOrigin,
        b_tma_origin: ImmOrigin,
        tiles_origin: MutOrigin,
        //,
    ](
        a_loader: TileLoader[
            a_tma_origin,
            Self.a_type,
            Self.ATileLayout_splitk,
            Self.ADescLayout_splitk,
            cta_group=Self.cta_group,
        ],
        b_loader: TileLoader[
            b_tma_origin,
            Self.b_type,
            Self.BTileLayout_splitk,
            Self.BDescLayout_splitk,
            cta_group=Self.cta_group,
        ],
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        iter_idx: UInt32,
        work_m_coord: Int,
        work_n_coord: Int,
        peer_cta_coord: Tuple[Int, Int, Int],
        elect_one_cta: Bool,
    ):
        """Load k_group_size A and B tiles using 2D TMA (for split-K only).

        Orchestrates the tile loading operation including:
        - expect_bytes signaling
        - k-group iteration
        - Peer CTA slicing for 2-SM MMA

        Parameters:
            a_tma_origin: Origin type for the A matrix TMA descriptor memory
                (inferred).
            b_tma_origin: Origin type for the B matrix TMA descriptor memory
                (inferred).
            tiles_origin: Origin type for the mutable tile payload memory
                (inferred).

        Args:
            a_loader: TileLoader for A matrix (2D).
            b_loader: TileLoader for B matrix (2D).
            tiles: ProducerTiles context with encapsulated tile access.
            iter_idx: K iteration index (base index).
            work_m_coord: M coordinate of the output tile.
            work_n_coord: N coordinate of the output tile.
            peer_cta_coord: Peer CTA coordinates (rank_n, rank_m, peer_m_rank).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # Global memory coordinates for A (M) and B (N)
        var a_gmem_m_coord = (
            peer_m_rank * Self.a_tma_rows + work_m_coord * Self.BM
        )
        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + work_n_coord * Self.MMA_N
        )

        if elect_one_sync():
            # Set expected bytes ONCE for all k_group tiles
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            # Get barrier for TMA multicast loads
            var barrier = tiles.barrier()

            comptime for j in range(Self.config.k_group_size):
                # Get tiles using payload accessor
                var a_tile, b_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                # Peer CTA slice using pointer arithmetic (not tile[]).
                # The tile[] method uses SMEM layout strides which can differ from
                # TMA descriptor layout. Pointer arithmetic with a_tma_load_size
                # preserves the original working behavior.
                var a_peer_tile = type_of(a_tile)(
                    a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                # TileTensor directly to loader (uses TileTensor TMA overload)
                a_loader.load(
                    a_peer_tile,
                    barrier[0],
                    k_coord,
                    a_gmem_m_coord,
                )
                b_loader.load(
                    b_peer_tile,
                    barrier[0],
                    k_coord,
                    b_gmem_n_coord,
                )

    comptime Bias1DTileLayout = row_major[1, Self.MMA_N]()
    comptime Bias1DTile = TileTensor[
        Self.c_type, type_of(Self.Bias1DTileLayout), ImmutAnyOrigin
    ]

    comptime WorkIter = WorkIterator[
        Self.num_clc_pipeline_stages,
        Index[dtype=DType.uint32](
            Self.config.cluster_shape[0],
            Self.config.cluster_shape[1],
            Self.config.cluster_shape[2],
        ),
        Self.config.raster_order,
        Self.config.block_swizzle_size,
    ]

    @staticmethod
    @always_inline
    def epilogue_load_producer[
        _epi_pipeline_stages: Int,
    ](
        epi_load_iter: Self.WorkIter,
        mut epilogue_load_pipeline: ProducerConsumerPipeline[
            _epi_pipeline_stages
        ],
        epilogue_load_tma_op: Self.EpilogueLoadTmaOp,
        bias_1d_tile: Self.Bias1DTile,
        epilogue_load_tiles: Self.SmemType.EpilogueLoadTileArray,
        mnk: StaticTuple[UInt32, 3],
    ):
        """Load epilogue tiles (bias) from GMEM to SMEM for each output tile.

        Handles three cases based on config:
        - 1D bias: warp-wide cp.async with zero-fill for OOB elements
        - AB_swapped: full MMA_N x BM TMA per output tile
        - non-AB_swapped: BM x stageN strips in stage-outer/col_wg-inner order

        Parameters:
            _epi_pipeline_stages: Number of stages in the epilogue load
                producer-consumer pipeline.

        Args:
            epi_load_iter: Work iterator yielding output tile coordinates for
                epilogue loads.
            epilogue_load_pipeline: Producer-consumer pipeline managing
                epilogue load stage synchronization.
            epilogue_load_tma_op: TMA descriptor for the epilogue load tensor,
                used in 2D load paths.
            bias_1d_tile: 1D bias tile in global memory, used only for the 1D
                bias load path.
            epilogue_load_tiles: Shared memory tile array staging epilogue load
                data.
            mnk: Problem dimensions `(M, N, K)`; `N` bounds the bias extent for
                OOB zero-fill.
        """

        comptime if Self.config.epilogue_is_1d:
            # 1D bias: warp-wide cp.async GMEM->SMEM with zero-fill
            # for OOB elements. Each active lane copies 8 elements (16B for
            # bf16, 32B for fp32). When AB_swapped, bias is along the kernel's
            # M dim (=original N).
            comptime bias_dim = Self._bias_tile_elems
            var bias_N = Int(mnk[1])
            var lane = Int(lane_id())
            comptime elems_per_lane = 8
            comptime bytes_per_lane = elems_per_lane * size_of[
                Scalar[Self.c_type]
            ]()
            # cp.async copies at most 16B per instruction, so wide element types
            # split the per-lane copy into 16B chunks (bf16: 1, fp32: 2).
            comptime copy_bytes = min(bytes_per_lane, 16)
            comptime num_copies = bytes_per_lane // copy_bytes
            comptime elems_per_copy = elems_per_lane // num_copies
            var lane_start = lane * elems_per_lane
            for current in epi_load_iter:
                epilogue_load_pipeline.wait_consumer()
                var stage = epilogue_load_pipeline.producer_stage()
                var smem_tile = epilogue_load_tiles[Int(stage)]
                var gmem_offset = (
                    Int(current.m)
                    * Self.BM if Self.config.AB_swapped else Int(current.n)
                    * Self.MMA_N
                )
                var valid_elems = min(bias_dim, bias_N - gmem_offset)

                if lane_start < bias_dim:
                    var src_bytes = Int32(
                        copy_bytes
                    ) if lane_start + elems_per_lane <= valid_elems else Int32(
                        0
                    )
                    var src_ptr = (
                        bias_1d_tile._storage + gmem_offset + lane_start
                    ).address_space_cast[.GLOBAL]()
                    var dst_ptr = smem_tile._storage + lane_start
                    comptime for chunk in range(num_copies):
                        async_copy[
                            copy_bytes,
                            fill=Scalar[Self.c_type](0),
                        ](
                            src_ptr + chunk * elems_per_copy,
                            dst_ptr + chunk * elems_per_copy,
                            src_size=src_bytes,
                        )
                var mbar = epilogue_load_pipeline.producer_mbar(stage)
                if lane_start < bias_dim:
                    async_copy_arrive(mbar[0].unsafe_ptr())
                    _ = mbar[0].arrive()
                epilogue_load_pipeline.producer_step()
        elif Self.config.AB_swapped:
            # AB_swapped: full MMA_N x BM tile per pipeline stage (one TMA per output tile).
            # ldmatrix.trans accesses non-contiguous N-rows so we keep the full tile.
            comptime epilogue_load_expected_bytes = Self.MMA_N * Self.BM * size_of[
                Self.c_type
            ]()
            for current in epi_load_iter:
                epilogue_load_pipeline.wait_consumer()
                var stage = epilogue_load_pipeline.producer_stage()
                if elect_one_sync():
                    var mbar = epilogue_load_pipeline.producer_mbar(stage)
                    mbar[0].expect_bytes(Int32(epilogue_load_expected_bytes))
                    epilogue_load_tma_op.async_copy[
                        cta_group=1,
                        eviction_policy=CacheEviction.NO_ALLOCATE,
                    ](
                        epilogue_load_tiles[Int(stage)],
                        mbar[0],
                        (
                            Int(current.m) * Self.BM,
                            Int(current.n) * Self.MMA_N,
                        ),
                    )
                epilogue_load_pipeline.producer_step()
        else:
            # non-AB_swapped: BM x stageN tile per pipeline stage.
            # Tiles sent in stage-outer / col_wg-inner order to match the consumer.
            comptime stageN = Self.OutputN
            comptime epilogue_load_expected_bytes = Self.BM * stageN * size_of[
                Self.c_type
            ]()
            # num_stages mirrors EpilogueConfig.create logic.
            comptime num_epi_stages_tw = (
                Self.MMA_N
                // stageN
                // 2 if (
                    Self.MMA_M == 128 and Self.cta_group == 2
                ) else Self.MMA_N
                // stageN
            )
            comptime num_col_wgs = Self.MMA_N // (num_epi_stages_tw * stageN)
            for current in epi_load_iter:
                var base_n_col = Int(current.n) * Self.MMA_N
                var base_m_row = Int(current.m) * Self.BM
                comptime for epi_stage in range(num_epi_stages_tw):
                    comptime for col_wg in range(num_col_wgs):
                        comptime col_pos = col_wg * num_epi_stages_tw * stageN + epi_stage * stageN
                        epilogue_load_pipeline.wait_consumer()
                        var prod_stage = epilogue_load_pipeline.producer_stage()
                        if elect_one_sync():
                            var mbar = epilogue_load_pipeline.producer_mbar(
                                prod_stage
                            )
                            mbar[0].expect_bytes(
                                Int32(epilogue_load_expected_bytes)
                            )
                            epilogue_load_tma_op.async_copy[
                                cta_group=1,
                                eviction_policy=CacheEviction.NO_ALLOCATE,
                            ](
                                epilogue_load_tiles[Int(prod_stage)],
                                mbar[0],
                                (base_n_col + col_pos, base_m_row),
                            )
                        epilogue_load_pipeline.producer_step()

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_ops, `nvvm.grid_constant`)
    @__llvm_arg_metadata(epilogue_load_tma_op, `nvvm.grid_constant`)
    @__name(
        StaticString(Self.config.get_kernel_name())
        + StaticString(
            "_fused_compute_epi" if Self.elementwise_compute_lambda_fn
            is not None else ""
        )
        + StaticString(
            "_fused_epi" if Self.elementwise_lambda_fn is not None else ""
        ),
    )
    def run(
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_ops: Array[Self.CTmaOp, Self.num_c_tma_descriptors],
        epilogue_load_tma_op: Self.EpilogueLoadTmaOp,
        bias_1d_tile: Self.Bias1DTile,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
        rank_sigs: Optional[
            Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS]
        ] = None,
        my_rank_dev: Int32 = 0,
    ):
        """Main kernel entry point for SM100 matrix multiplication.

        Always uses 3D TMA descriptors. For non-batched inputs, batch=1 and
        batch_coord=0 (from k_start = block_idx.z = 0 when grid_dim.z = 1).
        For batched inputs, grid_dim.z = batch_size and batch_coord from k_start.

        Args:
            a_tma_op: 3D TMA descriptor for the A input matrix.
            b_tma_op: 3D TMA descriptor for the B input matrix.
            c_tma_ops: Array of C TMA descriptors, one per peer for
                reduce-scatter or one for the local store.
            epilogue_load_tma_op: TMA descriptor for the epilogue load
                (bias) tensor.
            bias_1d_tile: 1D bias tile in global memory, used only for the
                1D bias epilogue path.
            cluster_dim: Thread block cluster dimensions for CLC scheduling.
            mnk: Problem dimensions `(M, N, K)` in elements.
            workspace: Workspace buffer for profiling and scheduling state.
            rank_sigs: Per-rank signal pointers for multi-GPU
                reduce-scatter synchronization (defaults to `None`).
            my_rank_dev: Rank index of this GPU for multi-GPU reduce-scatter
                (defaults to 0).
        """
        var my_rank = Int(my_rank_dev)
        Self.validate_constraints()

        # Access shared memory via bitcast
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create input pipeline for TMA→MMA synchronization (with payload)
        var tile_payload = Self.TilePayload(smem.a_tiles(), smem.b_tiles())
        var input_pipeline = Self.InputTilePipeline(
            smem.pipelines.input_barriers(), tile_payload
        )

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.pipelines.tmem_addr())

        # Prefetch TMA descriptors and initialize barriers
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            comptime for i in range(Self.num_c_tma_descriptors):
                c_tma_ops[i].prefetch_descriptor()
            comptime if Self.config.use_tma_epilogue_load and not Self.config.epilogue_is_1d:
                epilogue_load_tma_op.prefetch_descriptor()

        Self.init_barriers[
            use_tma_epilogue_load=Self.config.use_tma_epilogue_load
        ](
            ctx,
            smem.pipelines.input_barriers(),
            smem.pipelines.accum_barriers(),
            smem.pipelines.clc_throttle(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.tmem_dealloc(),
            smem.pipelines.epilogue_load_barriers(),
        )

        comptime _epi_pipeline_stages = Self.SmemType.num_epilogue_load_stages if Self.config.use_tma_epilogue_load else 1
        var epilogue_load_pipeline = ProducerConsumerPipeline[
            _epi_pipeline_stages
        ](smem.pipelines.epilogue_load_barrier_ptr())

        var mma_op = Self.MmaOp()

        # Scheduler owns CLC throttle pipeline internally
        var scheduler = Self.Scheduler(
            cluster_dim,
            smem.pipelines.clc_response(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.clc_throttle(),
        )

        var num_iters: UInt32 = ceildiv(mnk[2], UInt32(Self.BK))

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        if WarpRole.is_main_load():
            var load_iter = scheduler.work_iterator()

            with MatmulProfilerType[0](workspace, 0):
                comptime if (
                    Self.pdl_level > PDLLevel.OFF
                    and Self.config.prefetch_tiles_n > 0
                ):
                    comptime assert (
                        Self.config.prefetch_tiles_n
                        <= Self.num_group_pipeline_stages
                    ), (
                        "prefetch_tiles_n ("
                        + String(Self.config.prefetch_tiles_n)
                        + ") must not exceed num_group_pipeline_stages ("
                        + String(Self.num_group_pipeline_stages)
                        + "); Phase 1 would fill the ring before barriers can"
                        " fire."
                    )
                    with input_pipeline.producer() as producer:
                        for current in load_iter:
                            scheduler.throttle_signal(
                                ctx.is_first_cta_in_cluster
                            )

                            var work_coord = (
                                Int(current.m),
                                Int(current.n),
                                Int(current.k_start),
                            )

                            # Phase 1: prefetch A (weight) tiles before PDL wait
                            var prefetch_stages = StaticTuple[
                                UInt32, Self.config.prefetch_tiles_n
                            ]()
                            var prefetch_barriers = StaticTuple[
                                MbarPtr, Self.config.prefetch_tiles_n
                            ]()
                            var prefetch_payloads = StaticTuple[
                                Self.TilePayload,
                                Self.config.prefetch_tiles_n,
                            ]()

                            comptime for pf in range(
                                Self.config.prefetch_tiles_n
                            ):
                                if (
                                    UInt32(pf * Self.config.k_group_size)
                                    < num_iters
                                ):
                                    with producer.acquire() as tiles:
                                        prefetch_stages[pf] = tiles.stage()
                                        prefetch_barriers[pf] = tiles.barrier()
                                        prefetch_payloads[pf] = tiles.payload()
                                        comptime if Self.config.AB_swapped:
                                            Self.prefetch_a_tiles(
                                                a_tma_op,
                                                tiles,
                                                ctx.peer_cta_coord,
                                                work_coord,
                                                ctx.a_multicast_mask,
                                                UInt32(
                                                    pf
                                                    * Self.config.k_group_size
                                                ),
                                                ctx.elect_one_cta,
                                            )
                                        else:
                                            Self.prefetch_b_tiles(
                                                b_tma_op,
                                                tiles,
                                                ctx.peer_cta_coord,
                                                work_coord,
                                                ctx.b_multicast_mask,
                                                UInt32(
                                                    pf
                                                    * Self.config.k_group_size
                                                ),
                                                ctx.elect_one_cta,
                                            )

                            wait_on_dependent_grids()

                            # Phase 2: complete activation tiles
                            comptime for pf in range(
                                Self.config.prefetch_tiles_n
                            ):
                                if (
                                    UInt32(pf * Self.config.k_group_size)
                                    < num_iters
                                ):
                                    comptime if Self.config.AB_swapped:
                                        Self.complete_b_tiles(
                                            b_tma_op,
                                            prefetch_stages[pf],
                                            prefetch_barriers[pf],
                                            prefetch_payloads[pf],
                                            ctx.peer_cta_coord,
                                            work_coord,
                                            ctx.b_multicast_mask,
                                            UInt32(
                                                pf * Self.config.k_group_size
                                            ),
                                        )
                                    else:
                                        Self.complete_a_tiles(
                                            a_tma_op,
                                            prefetch_stages[pf],
                                            prefetch_barriers[pf],
                                            prefetch_payloads[pf],
                                            ctx.peer_cta_coord,
                                            work_coord,
                                            ctx.a_multicast_mask,
                                            UInt32(
                                                pf * Self.config.k_group_size
                                            ),
                                        )

                            # Phase 3: remaining K iterations (normal paired loads)
                            for i in range(
                                Self.config.prefetch_tiles_n
                                * Self.config.k_group_size,
                                Int(num_iters),
                                Self.config.k_group_size,
                            ):
                                with producer.acquire() as tiles:
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        tiles,
                                        ctx.peer_cta_coord,
                                        work_coord,
                                        ctx.a_multicast_mask,
                                        ctx.b_multicast_mask,
                                        UInt32(i),
                                        ctx.elect_one_cta,
                                    )

                            syncwarp()

                        producer.drain()  # wait for consumer before CTA exits
                else:
                    comptime if Self.pdl_level > PDLLevel.OFF:
                        wait_on_dependent_grids()

                    with input_pipeline.producer() as producer:
                        for current in load_iter:
                            scheduler.throttle_signal(
                                ctx.is_first_cta_in_cluster
                            )

                            for i in range(
                                0, Int(num_iters), Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        tiles,
                                        ctx.peer_cta_coord,
                                        (
                                            Int(current.m),
                                            Int(current.n),
                                            Int(current.k_start),
                                        ),
                                        ctx.a_multicast_mask,
                                        ctx.b_multicast_mask,
                                        UInt32(i),
                                        ctx.elect_one_cta,
                                    )

                            syncwarp()

                        producer.drain()  # wait for consumer before CTA exits

        # With no CLC pipeline stages each SM only processes its initial work, so
        # there is nothing to schedule. KERN-3311: fold that into the condition
        # rather than `return`ing -- a return would skip the cluster exit barrier
        # at the end of this function, leaving it divergent (undefined behavior).
        comptime has_clc_scheduling = Self.config.num_clc_pipeline_stages != 0
        if (
            has_clc_scheduling
            and WarpRole.is_scheduler()
            and ctx.is_first_cta_in_cluster
        ):
            # Scheduler warp uses its own iterator that manages both
            # producer and consumer state, plus throttle signaling
            var sched_iter = scheduler.scheduler_iterator()

            with MatmulProfilerType[1](workspace, 0):
                comptime if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                for _ in sched_iter:
                    sched_iter.signal_and_advance()

                # Drain all pending CLC requests before kernel exit
                sched_iter.drain()

        comptime if Self.config.use_tma_epilogue_load:
            if WarpRole.is_epilogue_load():
                var epi_load_iter = scheduler.work_iterator()
                Self.epilogue_load_producer[_epi_pipeline_stages](
                    epi_load_iter,
                    epilogue_load_pipeline,
                    epilogue_load_tma_op,
                    bias_1d_tile,
                    smem.epilogue_load_tiles(),
                    mnk,
                )

        if WarpRole.is_mma():
            var mma_iter = scheduler.work_iterator()

            with MatmulProfilerType[2](workspace, 0):
                var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers().ptr,
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                with mma_ctx:  # TMEM lifecycle
                    for _ in mma_iter:
                        if ctx.elect_one_cta:
                            with mma_ctx.output_pipeline.producer() as output_stage:  # waits for epilogue
                                with input_pipeline.consumer() as consumer:
                                    for i in range(
                                        0,
                                        Int(num_iters),
                                        Self.config.k_group_size,
                                    ):
                                        with consumer.acquire() as input_tiles:  # waits for TMA
                                            Self.mma(
                                                output_stage.tmem,
                                                input_tiles,
                                                mma_op,
                                                ctx.elect_one_warp,
                                                UInt32(i),
                                                0,
                                            )

                    # cta_group=2: the peer CTA (elect_one_cta == False) never
                    # issues MMA -- the leader's single MMA multicasts its
                    # commit into both CTAs' TMEM -- so it skips producer()
                    # above. Unlike this CTA's own epilogue, which waits on the
                    # multicast accumulator barrier before reading TMEM, the
                    # peer's MMA warp would otherwise reach the TMEM dealloc
                    # handshake having never observed a hardware-backed signal
                    # that the leader's MMA completed, relying only on that
                    # barrier's cross-CTA arrive bookkeeping. Wait on the same
                    # barrier the epilogue uses. Gated to
                    # num_clc_pipeline_stages == 0, where exactly one tile
                    # (accumulator stage 0) is produced per cluster launch.
                    comptime if (
                        Self.cta_group == 2
                        and Self.config.num_clc_pipeline_stages == 0
                    ):
                        if not ctx.elect_one_cta:
                            mma_ctx.output_pipeline.pipeline.wait_producer()

                comptime if Self.pdl_level > PDLLevel.OFF:
                    launch_dependent_grids()

        comptime if not Self.config.use_tma_epilogue_load:
            if WarpRole.is_epilogue():
                Self.EpilogueCtx.Sync.wait()  # wait for MMA to publish TMEM addr

                var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
                var epi_ctx = Self.EpilogueCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers().ptr,
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                var epi_iter = scheduler.work_iterator()

                # Pre-barrier: ensure all peers' output buffers are ready
                comptime if Self.output_writer_type.needs_sync:
                    _multi_gpu_barrier[
                        Self.num_c_tma_descriptors,
                        is_start=True,
                        named_barrier_threads=Self.EPILOGUE_THREADS,
                    ](rank_sigs.value(), rank_sigs.value()[my_rank], my_rank)

                with epi_ctx:  # signals TMEM dealloc on exit
                    var tile_idx = 0

                    for current in epi_iter:
                        with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                            with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                                # Uniform write through the injected writer policy
                                Self.write_output_tile(
                                    Pointer(to=c_tma_ops),
                                    smem.c_tiles(),
                                    output_stage,
                                    (
                                        current.m,
                                        current.n,
                                        current.k_start,
                                    ),
                                    (mnk[0], mnk[1]),
                                )
                        tile_idx += 1

                # Post-barrier: ensure all peers have finished reduce-add writes.
                # Use a named barrier scoped to the 4 epilogue warps (128 threads).
                # Note this is only safe because epilogue warps are 0-3...
                comptime if Self.output_writer_type.needs_sync:
                    _multi_gpu_barrier[
                        Self.num_c_tma_descriptors,
                        is_start=False,
                        need_fence=True,
                        named_barrier_threads=Self.EPILOGUE_THREADS,
                    ](rank_sigs.value(), rank_sigs.value()[my_rank], my_rank)

        else:
            if WarpRole.is_epilogue():
                Self.EpilogueCtx.Sync.wait()  # wait for MMA to publish TMEM addr

                var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
                var epi_ctx = Self.EpilogueCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers().ptr,
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                var tile_writer = Self.TileWriterType(Pointer(to=c_tma_ops[0]))
                var epi_iter = scheduler.work_iterator()

                with epi_ctx:  # signals TMEM dealloc on exit
                    var tile_idx = 0

                    for current in epi_iter:
                        with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                            with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                                # Use loaded epilogue tile using TMA + write back to GMEM (only enabled for bfloat16)
                                comptime if Self.config.epilogue_is_1d:
                                    # 1D bias: single 1×MMA_N tile, one pipeline stage per output tile.
                                    epilogue_load_pipeline.wait_producer()
                                    var epilogue_load_stage_idx = Int(
                                        epilogue_load_pipeline.consumer_stage()
                                    )
                                    tile_writer.write_batched_with_1d_bias(
                                        smem.c_tiles(),
                                        output_stage,
                                        smem.epilogue_load_tiles()[
                                            epilogue_load_stage_idx
                                        ],
                                        (current.m, current.n, current.k_start),
                                        (mnk[0], mnk[1]),
                                    )
                                    _ = epilogue_load_pipeline.consumer_mbar(
                                        UInt32(epilogue_load_stage_idx)
                                    )[0].arrive()
                                    epilogue_load_pipeline.consumer_step()
                                elif Self.config.AB_swapped:
                                    # AB_swapped: full MMA_N×BM tile, one pipeline stage per output tile.
                                    epilogue_load_pipeline.wait_producer()
                                    var epilogue_load_stage_idx = Int(
                                        epilogue_load_pipeline.consumer_stage()
                                    )
                                    tile_writer.write_batched_with_tma_epilogue_load[
                                        Self.config.epi_load_swizzle,
                                    ](
                                        smem.c_tiles(),
                                        output_stage,
                                        smem.epilogue_load_tiles()[
                                            epilogue_load_stage_idx
                                        ],
                                        (current.m, current.n, current.k_start),
                                        (mnk[0], mnk[1]),
                                    )
                                    _ = epilogue_load_pipeline.consumer_mbar(
                                        UInt32(epilogue_load_stage_idx)
                                    )[0].arrive()
                                    epilogue_load_pipeline.consumer_step()
                                else:
                                    # non-AB_swapped: BM×8 tiles; writer owns pipeline loop.
                                    tile_writer.write_batched_with_tma_epilogue_load_strips[
                                        Self.config.epi_load_swizzle,
                                        _epi_pipeline_stages,
                                    ](
                                        smem.c_tiles(),
                                        output_stage,
                                        epilogue_load_pipeline,
                                        smem.epilogue_load_tiles().ptr.as_unsafe_any_origin(),
                                        Self.SmemType.EpilogueLoadTileArray.tile_size,
                                        (current.m, current.n, current.k_start),
                                        (mnk[0], mnk[1]),
                                    )

                        tile_idx += 1

        # KERN-3311: hold the cluster together until every CTA is finished. The
        # epilogue signals its peer through cluster-mapped `arrive_cluster`
        # (structured_kernels/tmem.mojo), which requires that peer to still be
        # resident; if one CTA retires first the arrive targets a departed block
        # and TMEM is then freed for a pair that no longer jointly owns it. The
        # `cluster_sync()` during setup only orders mbarrier initialization.
        # Gated on cta_group == 2, not merely CLUSTER_SIZE > 1: the hazard is the
        # cluster-mapped arrive in `signal_peer()`, which only exists for
        # cta_group == 2. A 1-SM config with a multicast cluster has no
        # cross-CTA arrive and must not pay for this barrier. cta_group == 2
        # implies a 2-CTA cluster.
        comptime if Self.cta_group == 2:
            cluster_sync()

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    @__name(
        StaticString(Self.config.get_kernel_name())
        + StaticString(
            "_fused_compute_epi" if Self.elementwise_compute_lambda_fn
            is not None else ""
        )
        + StaticString(
            "_fused_epi" if Self.elementwise_lambda_fn is not None else ""
        ),
    )
    def run_splitk[
        reduction_layout: TensorLayout,
    ](
        a_tma_op: Self.ATmaOp_splitk,
        b_tma_op: Self.BTmaOp_splitk,
        c_tma_op: Self.CTmaOp_splitk,
        reduction_tensor: TileTensor[
            Self.config.accum_type, reduction_layout, MutAnyOrigin
        ],
        lock_ptr: UnsafePointer[UInt8, AnyOrigin[mut=True]],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Split-K kernel entry point for better parallelism on small problems.

        Split-K divides the K dimension across multiple CTAs, with each CTA
        computing a partial result that is then reduced.

        Parameters:
            reduction_layout: Memory layout of the reduction workspace tensor,
                must match the layout of `reduction_tensor`.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            reduction_tensor: Workspace for partial results from each split.
            lock_ptr: Synchronization locks for reduction coordination.
            cluster_dim: Cluster dimensions.
            mnk: Problem dimensions (M, N, K).
            workspace: Workspace buffer for profiling/scheduling.
        """
        Self.validate_constraints()

        # Access shared memory via bitcast
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create input pipeline for TMA→MMA synchronization (with payload)
        var tile_payload = Self.TilePayload(smem.a_tiles(), smem.b_tiles())
        var input_pipeline = Self.InputTilePipeline(
            smem.pipelines.input_barriers(), tile_payload
        )

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.pipelines.tmem_addr())

        # Prefetch TMA descriptors and initialize barriers
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
        Self.init_barriers(
            ctx,
            smem.pipelines.input_barriers(),
            smem.pipelines.accum_barriers(),
            smem.pipelines.clc_throttle(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = MmaOpSM100_SS[
            Self.c_type,
            Self.a_type,
            Self.b_type,
            Self.config.block_tile_shape,
            Self.config.mma_shape,
            accum_type=Self.config.accum_type,
            cta_group=Self.config.cta_group,
            cluster_shape=Self.config.cluster_shape,
            a_swizzle=Self.config.a_swizzle,
            b_swizzle=Self.config.b_swizzle,
            transpose_b=True,
        ]()

        # Scheduler owns CLC throttle pipeline internally
        var scheduler = TileSchedulerSplitK[
            num_stages=Self.config.num_clc_pipeline_stages,
            reduction_tile_shape=Index(Self.BM, Self.MMA_N, Self.BK),
            cluster_shape=Index[dtype=DType.uint32](
                Self.config.cluster_shape[0],
                Self.config.cluster_shape[1],
                Self.config.cluster_shape[2],
            ),
            block_swizzle_size=Self.config.block_swizzle_size,
            rasterize_order=Self.config.raster_order,
            num_split_k=Self.config.num_split_k,
        ](
            cluster_dim,
            mnk,
            smem.pipelines.clc_response(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.clc_throttle(),
            lock_ptr,
        )

        # Create tile loaders for A and B matrices (2D for split-K)
        var a_loader = TileLoader[
            _,
            Self.a_type,
            Self.ATileLayout_splitk,
            Self.ADescLayout_splitk,
            cta_group=Self.cta_group,
        ](Pointer(to=a_tma_op), ctx.a_multicast_mask)
        var b_loader = TileLoader[
            _,
            Self.b_type,
            Self.BTileLayout_splitk,
            Self.BDescLayout_splitk,
            cta_group=Self.cta_group,
        ](Pointer(to=b_tma_op), ctx.b_multicast_mask)

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        if WarpRole.is_main_load():
            var load_iter = scheduler.work_iterator()

            with MatmulProfilerType[0](workspace, 0):
                # Producer context: coordinates with MMA consumer via barriers
                with input_pipeline.producer() as producer:
                    for current in load_iter:
                        scheduler.throttle_signal(ctx.is_first_cta_in_cluster)

                        var k_start = current.k_start
                        var k_end = k_start + current.num_k_tiles
                        for i in range(
                            Int(k_start),
                            Int(k_end),
                            Self.config.k_group_size,
                        ):
                            with producer.acquire() as tiles:  # waits for consumer
                                Self.load_input_tiles_splitk(
                                    a_loader,
                                    b_loader,
                                    tiles,
                                    UInt32(i),
                                    Int(current.m),
                                    Int(current.n),
                                    ctx.peer_cta_coord,
                                    ctx.elect_one_cta,
                                )

                        syncwarp()

                    producer.drain()  # wait for consumer before CTA exits

        # See run(): fold rather than `return`, so the cluster exit barrier below
        # is reached by every thread.
        comptime has_clc_scheduling = Self.config.num_clc_pipeline_stages != 0
        if (
            has_clc_scheduling
            and WarpRole.is_scheduler()
            and ctx.is_first_cta_in_cluster
        ):
            var sched_iter = scheduler.scheduler_iterator()

            with MatmulProfilerType[1](workspace, 0):
                for _ in sched_iter:
                    sched_iter.signal_and_advance()

                sched_iter.drain()

        if WarpRole.is_mma():
            var mma_iter = scheduler.work_iterator()

            with MatmulProfilerType[2](workspace, 0):
                var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers().ptr,
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                with mma_ctx:  # TMEM lifecycle
                    for current in mma_iter:
                        if ctx.elect_one_cta:
                            with mma_ctx.output_pipeline.producer() as output_stage:  # waits for epilogue
                                var k_start = current.k_start
                                var k_end = k_start + current.num_k_tiles
                                with input_pipeline.consumer() as consumer:
                                    for i in range(
                                        Int(k_start),
                                        Int(k_end),
                                        Self.config.k_group_size,
                                    ):
                                        with consumer.acquire() as input_tiles:  # waits for TMA
                                            Self.mma(
                                                output_stage.tmem,
                                                input_tiles,
                                                mma_op,
                                                ctx.elect_one_warp,
                                                UInt32(i),
                                                k_start,
                                            )

                    # cta_group=2: the peer CTA (elect_one_cta == False) never
                    # issues MMA -- the leader's single MMA multicasts its
                    # commit into both CTAs' TMEM -- so it skips producer()
                    # above. Unlike this CTA's own epilogue, which waits on the
                    # multicast accumulator barrier before reading TMEM, the
                    # peer's MMA warp would otherwise reach the TMEM dealloc
                    # handshake having never observed a hardware-backed signal
                    # that the leader's MMA completed, relying only on that
                    # barrier's cross-CTA arrive bookkeeping. Wait on the same
                    # barrier the epilogue uses. Gated to
                    # num_clc_pipeline_stages == 0, where exactly one tile
                    # (accumulator stage 0) is produced per cluster launch.
                    comptime if (
                        Self.cta_group == 2
                        and Self.config.num_clc_pipeline_stages == 0
                    ):
                        if not ctx.elect_one_cta:
                            mma_ctx.output_pipeline.pipeline.wait_producer()

        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()  # wait for MMA to publish TMEM addr

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    smem.pipelines.accum_barriers().ptr,
                    tmem,
                    UInt16(ctx.mma_complete_mask),
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tile_writer = Self.TileWriterType_splitk(Pointer(to=c_tma_op))
            var epi_iter = scheduler.work_iterator()

            with epi_ctx:  # signals TMEM dealloc on exit
                var tile_idx = 0

                for current in epi_iter:
                    with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                        with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                            tile_writer.write_splitk(
                                smem.c_tiles(),
                                output_stage,
                                scheduler,
                                reduction_tensor,
                                current,
                                (mnk[0], mnk[1]),
                                ctx.elect_one_warp,
                            )

                    tile_idx += 1

        # KERN-3311: hold the cluster together until every CTA is finished. The
        # epilogue signals its peer through cluster-mapped `arrive_cluster`
        # (structured_kernels/tmem.mojo), which requires that peer to still be
        # resident; if one CTA retires first the arrive targets a departed block
        # and TMEM is then freed for a pair that no longer jointly owns it. The
        # `cluster_sync()` during setup only orders mbarrier initialization.
        # Gated on cta_group == 2, not merely CLUSTER_SIZE > 1: the hazard is the
        # cluster-mapped arrive in `signal_peer()`, which only exists for
        # cta_group == 2. A 1-SM config with a multicast cluster has no
        # cross-CTA arrive and must not pay for this barrier. cta_group == 2
        # implies a 2-CTA cluster.
        comptime if Self.cta_group == 2:
            cluster_sync()


# ============================================================================
# BlackwellMatmulSM100FallbackKernel - Simple non-warp-specialized kernel
# ============================================================================


struct BlackwellMatmulSM100FallbackKernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    c_layout: TensorLayout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Simple fallback matmul kernel for SM100 (B200).

    This kernel is used when the warp-specialized kernel is not applicable,
    such as for small problem sizes or unsupported configurations.

    Unlike the main BlackwellMatmulSM100Kernel, this uses:
    - Single warp approach (no warp specialization)
    - Basic barrier synchronization (no CLC scheduling)
    - Direct TileTensor output (no TMA for C)
    - Simpler pipeline with single buffer

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        c_layout: Memory layout of the C output tensor in global memory, used
            for output tiling and static stride computation.
        block_tile_shape: Block tile dimensions `(BM, BN, BK)` for
            CTA-level tiling of the output and reduction dimensions.
        mma_shape: MMA instruction dimensions `(MMA_M, MMA_N, MMA_K)` for
            the tensor core operation.
        transpose_b: Whether B is stored transposed (K-major) (defaults to
            `True`).
        cluster_shape: Thread block cluster dimensions used for LLVM cluster
            metadata (defaults to `(1, 1, 1)`).
        a_swizzle: Swizzle pattern for A shared memory tiles (defaults to
            `TensorMapSwizzle.SWIZZLE_128B`).
        b_swizzle: Swizzle pattern for B shared memory tiles (defaults to
            `TensorMapSwizzle.SWIZZLE_128B`).
        num_threads: Number of threads per CTA; must be 128 or 256 (defaults
            to 128).
        elementwise_lambda_fn: Optional epilogue function applied to output
            elements (defaults to `None`).
    """

    # ========== Derived Constants ==========
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime BK = Self.block_tile_shape[2]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K

    # TMA layouts for A and B (computed from config)
    comptime a_swizzle_elems = Self.a_swizzle.bytes() // size_of[Self.a_type]()
    comptime b_swizzle_elems = Self.b_swizzle.bytes() // size_of[Self.b_type]()

    comptime ATileLayout = static_row_major[Self.BM, Self.BK]
    comptime ADescLayout = static_row_major[Self.BM, Self.a_swizzle_elems]
    comptime BTileLayout = static_row_major[Self.BN, Self.BK]
    comptime BDescLayout = static_row_major[Self.BN, Self.b_swizzle_elems]

    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]

    # Static N dimension (columns) from C layout stride -- used for output tiling
    comptime static_N = Self.c_layout.static_stride[0]

    # Row-major stride layout [N, 1] for C global memory tiles.
    # Used as stride_layout in tile/tile_with_offset to override
    # the parent TileTensor's dynamic strides with static values.
    comptime CGmemStrideLayout = _NewLayout[
        Coord[ComptimeInt[Self.static_N], ComptimeInt[1]].element_types,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]

    # Typed layouts (new Layout from tile_layout.mojo).
    # MmaOpSM100_SS requires transpose_b=True, so B is always K-major.
    comptime a_smem_layout_typed = tile_layout_k_major_typed[
        Self.a_type, Self.BM, Self.BK, Self.a_swizzle
    ]
    comptime b_smem_layout_typed = tile_layout_k_major_typed[
        Self.b_type, Self.BN, Self.BK, Self.b_swizzle
    ]

    comptime a_size: Int = Self.BM * Self.BK
    comptime b_size: Int = Self.BN * Self.BK

    # ========== Tile Type Aliases (TileTensor-based) ==========
    comptime ATile = TTSMemTile[Self.a_type, Self.a_smem_layout_typed]
    comptime BTile = TTSMemTile[Self.b_type, Self.b_smem_layout_typed]

    comptime accum_type = get_accum_type[Self.a_type]()
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // Self.num_threads
    comptime max_tmem_cols = 512

    # ========== Validation ==========
    @staticmethod
    @always_inline
    def validate_constraints():
        """Validate compile-time constraints for this kernel configuration."""
        comptime assert Self.num_threads == 128 or Self.num_threads == 256
        comptime assert (
            (Self.a_size * size_of[Self.a_type]()) % 128
        ) == 0, "preserve alignment"
        comptime assert (
            (Self.b_size * size_of[Self.b_type]()) % 16
        ) == 0, "preserve alignment"

    # ========== Kernel Entry Point ==========
    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    def run(
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c: TileTensor[Self.c_type, Self.c_layout, MutAnyOrigin],
        num_iters: Int32,
    ):
        """Run the fallback matmul kernel.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c: Output tensor C (TileTensor, direct global memory writes).
            num_iters: Number of K-dimension iterations.
        """
        var _num_iters = Int(num_iters)
        Self.validate_constraints()

        # Setup shared memory for A and B tiles
        var a_smem = rebind[SMemPtr[Scalar[Self.a_type]]](
            external_memory[
                Scalar[Self.a_type],
                address_space=.SHARED,
                alignment=128,
                name="tmem_test_dynamic_shared_memory",
            ]()
        )

        var b_smem = (a_smem + Self.a_size).bitcast[Scalar[Self.b_type]]()

        var a_smem_tile = Self.ATile(
            a_smem.as_unsafe_any_origin(), Self.a_smem_layout_typed
        )
        var b_smem_tile = Self.BTile(
            b_smem.as_unsafe_any_origin(), Self.b_smem_layout_typed
        )

        # Shared memory pointer to hold tensor memory address
        var ptr_tmem_addr = (b_smem + Self.b_size).bitcast[UInt32]()

        var c_frag: Array[Scalar[Self.accum_type], Self.c_frag_size]

        comptime a_expected_bytes = Self.a_size * size_of[Self.a_type]()
        comptime b_expected_bytes = Self.b_size * size_of[Self.b_type]()
        comptime expected_bytes = a_expected_bytes + b_expected_bytes

        var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
        var mma_mbar = tma_mbar + 1

        var elect_one_warp = get_warp_id() == 0
        var elect_one_thread = elect_one_warp and elect_one_sync()
        var elect_one_cta = block_rank_in_cluster() % 2 == 0

        if elect_one_thread:
            tma_mbar[0].init()
            mma_mbar[0].init()

        var tma_phase: UInt32 = 0
        var mma_phase: UInt32 = 0

        # Allocate tensor memory
        if elect_one_warp:
            tcgen05_alloc[1](ptr_tmem_addr, Self.max_tmem_cols)

        # Ensure all threads see initialized mbarrier and tensor memory allocation
        barrier()

        var tmem_addr = ptr_tmem_addr[0]

        # Create MmaOpSM100_SS instance
        var mma_op = MmaOpSM100_SS[
            Self.c_type,
            Self.a_type,
            Self.b_type,
            Self.block_tile_shape,
            Self.mma_shape,
            accum_type=Self.accum_type,
            cta_group=1,
            a_swizzle=Self.a_swizzle,
            b_swizzle=Self.b_swizzle,
            transpose_b=Self.transpose_b,
        ]()

        # Main loop over K dimension
        for i in range(_num_iters):
            # Only one thread per CTA does the copy
            if elect_one_thread:
                tma_mbar[0].expect_bytes(Int32(expected_bytes))

                a_tma_op.async_copy(
                    a_smem_tile,
                    tma_mbar[0],
                    (i * Self.BK, block_idx.y * Self.BM),
                )
                b_tma_op.async_copy(
                    b_smem_tile,
                    tma_mbar[0],
                    (
                        i * Self.BK,
                        block_idx.x * Self.BN,
                    ) if Self.transpose_b else (
                        block_idx.x * Self.BN,
                        i * Self.BK,
                    ),
                )

            # Wait for the copy to finish
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            # Perform MMA operation
            if elect_one_thread:
                mma_op.mma(
                    a_smem_tile,
                    b_smem_tile,
                    tmem_addr,
                    init_c=(i == 0),  # Initialize C on first iteration
                )
                mma_op.commit(mma_mbar)

            mma_mbar[0].wait(mma_phase)
            mma_phase ^= 1

        # Load accumulated result from tensor memory
        from ..structured_kernels.tmem import TmemAddress

        var tmem = TmemAddress(tmem_addr)
        c_frag = tmem.load_upper[
            Self.accum_type, Self.c_frag_size, 16, 256, Self.BN // 8
        ]()
        TmemAddress.wait_load()

        if elect_one_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem_addr, Self.max_tmem_cols)

        # Write output to global memory using tile/vectorize/distribute.
        # stride_layout overrides the parent's dynamic strides with
        # explicit static strides, enabling vectorize/distribute (all_dims_known).
        comptime num_warps = Self.num_threads // WARP_SIZE
        comptime N = Self.static_N
        var warp_id = get_warp_id()

        var ctile, ctile_coords, _ = c.tile_with_offset[
            Self.BM, Self.BN, stride_layout=Self.CGmemStrideLayout
        ](Coord(block_idx.y, block_idx.x))

        var M = c.dim[0]()

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                var warp_tile, warp_coords, _ = ctile.tile_with_offset[
                    Self.MMA_M // num_warps,
                    Self.MMA_N,
                    stride_layout=Self.CGmemStrideLayout,
                ](
                    Coord(
                        4 * m_mma + warp_id,
                        n_mma,
                    )
                )
                var warp_m = ctile_coords[0] + warp_coords[0]
                var warp_n = ctile_coords[1] + warp_coords[1]

                var vectorized = warp_tile.vectorize[1, 2]()
                var dist_result = vectorized.distribute_with_offset[
                    row_major[8, 4]()
                ](lane_id())
                var frag = dist_result[0]
                var frag_coords = dist_result[1]
                var frag_m = warp_m + frag_coords[0]
                var frag_n = warp_n + frag_coords[1] * 2

                comptime num_vecs_m = type_of(frag).static_shape[0]
                comptime num_vecs_n = type_of(frag).static_shape[1]

                comptime for n_vec in range(num_vecs_n):
                    comptime for m_vec in range(num_vecs_m):
                        comptime i_vec = n_vec * num_vecs_m + m_vec
                        var dst_idx = Int(frag.layout(coord[m_vec, n_vec]))
                        var dst_m_offset, dst_n_offset = divmod(dst_idx, N)
                        var m = UInt32(frag_m + dst_m_offset)
                        var n = UInt32(frag_n + dst_n_offset)

                        if m < UInt32(M) and n < UInt32(N):
                            var c_mn = SIMD[Self.accum_type, 2](
                                c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                            ).cast[Self.c_type]()

                            comptime if Self.elementwise_lambda_fn:
                                comptime alignment = align_of[
                                    SIMD[Self.c_type, 2]
                                ]()
                                comptime epilogue = (
                                    Self.elementwise_lambda_fn.value()
                                )
                                epilogue[alignment=alignment](
                                    (Int(m), Int(n)), c_mn
                                )
                            else:
                                frag[m_vec, n_vec] = rebind[
                                    type_of(frag).ElementType
                                ](c_mn)
