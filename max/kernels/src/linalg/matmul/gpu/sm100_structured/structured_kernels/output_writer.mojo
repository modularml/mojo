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

"""TileWriter for SM100 matmul output pipeline.

Writes accumulated results from TMEM → Registers → SMEM → GMEM (via TMA).

Usage:
    var writer = TileWriter[config=..., ...](Pointer(to=c_tma_op))
    writer.write(smem.c_tiles(), stage, coord, shape, elect)
"""

from std.collections import Optional
from std.memory import Pointer, UnsafePointer
from std.sys import simd_width_of, size_of, align_of

from max.gpu import WARP_SIZE, thread_idx
from max.gpu import lane_id, warp_id as get_warp_id
from max.gpu.memory import fence_async_view_proxy
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.layout import zipped_divide
from layout.layout_tensor import upcast
from layout.runtime_tuple import crd2idx as rt_crd2idx
from layout.swizzle import make_swizzle
from layout.tma_async import TMATensorTile
from max.gpu.compute.mma import ld_matrix
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)

from std.utils.index import IndexList

# TileTensor-based types for C tiles
from structured_kernels.tile_types import SMemTileArray2DRowMajor

from structured_kernels.barriers import WarpGroupBarrier
from structured_kernels.kernel_common import WarpRole1D1D
from .config import OutputPipelineConfig
from .tile_pipeline import OutputStage
from .output_writer_trait import OutputWriter
from .tile_scheduler_splitk import TileScheduler, WorkInfo
from .epilogue_components import (
    AccumBarrier,
    AccumTile,
    EpilogueApplier,
    EpilogueConfig,
    SMemEpilogueWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    TMEMToSMemWriter,
    tma_wait_pipelined,
)
from structured_kernels.pipeline import ProducerConsumerPipeline
from .tmem import TmemArrayType


struct TileWriter[
    # Inferred from constructor arg
    tma_origin: ImmOrigin,
    c_type: DType,
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    //,
    # Explicit config parameters (works with any config type)
    a_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    opc: OutputPipelineConfig,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    # Kernel-level parameters - dimensions replace c_smem_layout
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    num_output_stages: Int,
    num_output_warps: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    batched: Bool = False,
    problem_n: Int = 0,
    num_peers: Int = 1,  # this is a local epilogue
](TrivialRegisterPassable):
    """Output tile writer for SM100 matmul epilogue.

    Stores pointer to TMA descriptor. SMEM tiles passed per-call.

    Parameters are passed explicitly to work with both MatmulConfig
    and BlockScaledMatmulConfig.

    The opc (OutputPipelineConfig) parameter must match the config used
    when constructing the OutputTilePipeline that provides OutputStage
    instances to the write() method.

    Parameters:
        tma_origin: Memory origin of the TMA descriptor pointer
            (inferred).
        c_type: Element dtype of the C output tensor (inferred).
        c_rank: Rank of the C output tensor (inferred).
        c_tile_shape: Per-tile shape of the C output (inferred).
        c_desc_shape: TMA descriptor shape for C (inferred).
        a_type: Element dtype of the A input matrix.
        accum_type: Accumulator dtype stored in TMEM.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        opc: Output pipeline config bundling accumulator stages, stage
            stride, and CTA group.
        c_swizzle: TMA swizzle pattern for the C SMEM layout.
        transpose_c: Whether C is stored transposed.
        c_smem_dim0: Row dimension of the C SMEM tile.
        c_smem_dim1: Column dimension of the C SMEM tile.
        num_output_stages: Number of C SMEM pipeline stages.
        num_output_warps: Number of warps driving the output pipeline.
        elementwise_lambda_fn: Optional elementwise epilogue applied to
            fragments before the store (defaults to None).
        elementwise_compute_lambda_fn: Optional compute epilogue fused
            into the register path (defaults to None).
        register_based_epilogue: Whether the compute epilogue runs in
            registers (true) or SMEM (false) (defaults to True).
        batched: Whether the output uses 3D batched coordinates with a
            batch index (defaults to False).
        problem_n: Logical N dimension used for row-major bounds checking
            in the slow path; 0 disables the N check (defaults to 0).
        num_peers: Number of TMA store descriptors in the array; 1 for a
            local epilogue (defaults to 1).
    """

    # Local aliases from OutputPipelineConfig
    comptime cta_group = Self.opc.cta_group
    comptime num_accum_pipeline_stages = Self.opc.num_stages
    comptime stage_stride_cols = Self.opc.stage_stride_cols

    # Create internal layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    # Type aliases
    comptime TmaOp = TMATensorTile[
        Self.c_type, Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]
    # Whole-array pointer accepted by the `TileWriterLike` ctor (one descriptor
    # for the standard store; the ctor uses element [0]).
    comptime TmaOpArray = Array[Self.TmaOp, Self.num_peers]
    comptime TmaOpArrayPtr = Pointer[Self.TmaOpArray, Self.tma_origin]

    # No cross-GPU synchronization for a local TMA store.
    comptime needs_sync = False
    # C tile array (output and source tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]
    comptime Stage = OutputStage[Self.opc]

    # Derived constants
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    # FP8 uses float32 epilogue (GEX-2630), bf16/fp4 uses native type.
    comptime epilogue_dtype = Self.get_epilogue_dtype()

    # Stage dimensions - now use direct dimension access
    comptime N_dim = 0 if Self.transpose_c else 1
    comptime stageN = Self.c_smem_dim0 if Self.transpose_c else Self.c_smem_dim1
    comptime stage_contiguous_size = Self.c_smem_dim1

    # Warp-role table used only for `epilogue_role_index()`, which depends
    # on `EPILOGUE_WARP_START` alone — identical under either `has_sfb`
    # setting, so the default instantiation is exact for every caller.
    comptime WarpRole = WarpRole1D1D[num_epi_warps=Self.num_output_warps]

    # EpilogueConfig bundles common epilogue parameters
    comptime epc = EpilogueConfig.create(
        MMA_M=Self.MMA_M,
        MMA_N=Self.MMA_N,
        stageN=Self.stageN,
        cta_group=Self.cta_group,
        transpose_c=Self.transpose_c,
        BM=Self.BM,
        BN=Self.BN,
    )

    # Fragment layout constants
    comptime data_paths = 16
    comptime bits = 256
    comptime rep = Self.stageN // (Self.bits // 32)
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // WARP_SIZE
    comptime rep_frag_size = Self.fragment_size * Self.rep

    # Aliases from EpilogueConfig
    comptime is_lower_frag_required = Self.epc.is_lower_frag_required
    comptime num_stages = Self.epc.num_stages

    # TMEM array type for accumulator tiles
    comptime accum_tile_layout = Layout.row_major(Self.BM, Self.stageN)
    comptime AccumTmemArray = TmemArrayType[
        Self.accum_type,
        Self.accum_tile_layout,
        Self.num_stages,
        cta_group=Self.cta_group,
    ]

    var c_tma_op: Self.TmaOpPtr

    @always_inline
    def __init__(out self, c_tma_op: Self.TmaOpPtr):
        """Initialize with pointer to TMA descriptor.

        Args:
            c_tma_op: Pointer to the TMA store descriptor for C.
        """
        comptime assert (
            Self.stage_stride_cols > 0
        ), "stage_stride_cols must be positive"
        self.c_tma_op = c_tma_op

    @always_inline
    def __init__(out self, c_tma_ops: Self.TmaOpArrayPtr):
        """Initialize from the `c_tma_ops` array pointer (`TileWriterLike`).

        The standard local store targets a single descriptor, so this uses
        element `[0]` of the array. Unifies construction with the
        reduce-scatter writer, which retains all `num_peers` descriptors.

        Args:
            c_tma_ops: Pointer to the array of TMA store descriptors for
                C; element `[0]` is used for the local store.
        """
        comptime assert (
            Self.stage_stride_cols > 0
        ), "stage_stride_cols must be positive"
        self.c_tma_op = Pointer(to=c_tma_ops[][0])

    @always_inline
    @staticmethod
    def get_epilogue_dtype() -> DType:
        if (Self.a_type == Self.c_type == .bfloat16) or (
            Self.a_type == .uint8 and Self.c_type == .bfloat16
        ):
            return DType.bfloat16
        else:
            return DType.float32

    # ========== Public Write Methods ==========

    @always_inline
    def write(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write accumulated results to global memory (2D coords).

        Args:
            c_tiles: SMEM tile array for the C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile) tile coordinates.
            shape: (M, N) problem dimensions.
            elect_one_warp: Whether this warp is elected for coordination.
        """
        self._copy_to_gmem(c_tiles, stage, tile_coord, shape)

    @always_inline
    def write_batched(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Write accumulated results to global memory (3D batched coords).

        Args:
            c_tiles: TileTensor-based SMEM tile array for C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile, batch) coordinates.
            shape: (M, N) problem dimensions.
            alpha: Tensor scale factor (scalar).
        """
        self._copy_to_gmem_batched(c_tiles, stage, tile_coord, shape, alpha)

    @always_inline
    def write_splitk[
        reduction_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        scheduler: TileScheduler,
        reduction_tensor: TileTensor[
            Self.accum_type, reduction_layout, MutAnyOrigin
        ],
        work_info: WorkInfo,
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with split-K reduction. Only last split writes to GMEM."""
        var epilogue_thread_idx = thread_idx.x

        # Perform reduction and check if this is the last split
        var is_last_split = scheduler.reduction(
            reduction_tensor,
            stage.tmem.address(),
            epilogue_thread_idx,
            work_info,
        )

        # If not last split, signal and exit early
        if not is_last_split:
            AccumBarrier[Self.cta_group].arrive(stage.pipeline, stage.index)
            return

        self._copy_to_gmem(c_tiles, stage, (work_info.m, work_info.n), shape)

    @always_inline
    def write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
    ):
        """Write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute.

        Parameters:
            c_tensor_layout: Layout of the C tensor in GMEM (inferred).

        Args:
            c_tiles: SMEM tile array for the C output.
            output_stage: OutputStage with pipeline, index, and TMEM
                handle.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM for bounds-checked stores.
        """
        self._write_absolute_with_bounds_check[c_tensor_layout](
            c_tiles,
            output_stage,
            m_abs,
            n_abs,
            m_end,
            expert_scale,
            c_tensor,
        )

    @always_inline
    def _copy_to_gmem(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → SMEM → GMEM pipeline (2D coords)."""
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles, output_stage, c_coord, c_shape
            )
        else:
            self._copy_to_gmem_impl(c_tiles, output_stage, c_coord, c_shape)

    @always_inline
    def _copy_to_gmem_batched(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32,
    ):
        """TMEM → Registers → GMEM (elementwise epilogue) pipeline (3D batched coords).
           TMEM → Registers → SMEM → GMEM (compute epilogue) pipeline (3D batched coords).

        If elementwise epilogue function is provided, it will be used to write the results to global memory.
        Otherwise, the results will be written to global memory using the standard TMA based pipeline.
        """
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )
        else:
            self._copy_to_gmem_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )

    @always_inline
    def _copy_to_gmem_with_elementwise_epilogue_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → GMEM (elementwise epilogue) pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.

        In contrast to compute epilogue, elementwise epilogue input is casted to c_type, not epilogue_dtype.
        This is because elementwise epilogue writes directly to global memory, not registers.
        Therefore, we need to cast the input to c_type to match the output type.
        """

        comptime assert (
            Self.elementwise_lambda_fn is not None
        ), "Elementwise epilogue function is not provided"

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform, computed once per tile: lets the direct-GMEM epilogue
        # skip per-position bounds checks (and their branches) for tiles fully
        # inside (M, N). transpose_c swaps the row/col → user-M/user-N mapping.
        # Mirrors _copy_to_gmem_impl's tile_in_bounds.
        var tile_in_bounds: Bool
        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to c_type in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions
            # (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
            var upper_simd = SIMD[Self.c_type, Self.rep_frag_size]()
            var lower_simd = SIMD[Self.c_type, Self.rep_frag_size]()

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.c_type]()
                comptime for _j in range(cast_width):
                    upper_simd[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        lower_simd[offset + _j] = dst[_j]

            if tile_in_bounds:
                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                    is_in_bounds=True,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )
            else:
                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                    is_in_bounds=False,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    # ========== Shared Output Helpers ==========

    @always_inline
    def _cast_frags_and_write_to_smem[
        c_tile_layout: TensorLayout,
    ](
        self,
        upper_frag_casted: Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ],
        lower_frag_casted: Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ],
        c_smem_tile: TileTensor[
            Self.c_type, c_tile_layout, MutAnyOrigin, address_space=.SHARED
        ],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Cast fragments from epilogue dtype to c_type and write to SMEM."""
        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(warp_id, lane)

        comptime expected_size = Self.epc.fragment_size * Self.rep
        var upper_c = Array[Scalar[Self.c_type], expected_size](
            uninitialized=True
        )
        var lower_c = Array[Scalar[Self.c_type], expected_size](
            uninitialized=True
        )

        comptime cast_width_c = (4 // size_of[Scalar[Self.c_type]]())
        comptime for _chunk in range(Self.rep_frag_size // cast_width_c):
            comptime offset = _chunk * cast_width_c
            var src_u = SIMD[Self.epilogue_dtype, cast_width_c]()
            var src_l = SIMD[Self.epilogue_dtype, cast_width_c]()
            comptime for _j in range(cast_width_c):
                src_u[_j] = upper_frag_casted[offset + _j]
                src_l[_j] = lower_frag_casted[offset + _j]
            var dst_u = src_u.cast[Self.c_type]()
            var dst_l = src_l.cast[Self.c_type]()
            comptime for _j in range(cast_width_c):
                upper_c[offset + _j] = dst_u[_j]
                lower_c[offset + _j] = dst_l[_j]
        smem_writer.write_fragments[Self.rep](
            rebind[Array[Scalar[Self.c_type], expected_size]](upper_c),
            rebind[Array[Scalar[Self.c_type], expected_size]](lower_c),
            c_smem_tile,
        )
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _tma_store_to_gmem[
        stage: Int,
        c_tile_layout: TensorLayout,
    ](
        self,
        c_smem_tile: TileTensor[
            Self.c_type, c_tile_layout, MutAnyOrigin, address_space=.SHARED
        ],
        c_coord: Tuple[UInt32, UInt32],
        batch_idx: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """TMA store from SMEM to GMEM with pipelined wait and barrier sync."""
        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime StoreCoords = TMAStoreCoords[
            Self.epc,
            Self.c_smem_dim0,
            stage,
            batched=Self.batched,
        ]

        var store_coords = StoreCoords(
            (c_coord[0], c_coord[1], batch_idx if Self.batched else 0),
            warp_id,
        )
        StoreExecutor.execute[
            Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
        ](
            c_smem_tile,
            store_coords,
            self.c_tma_op[],
            warp_id,
            lane,
        )

        tma_wait_pipelined[
            Self.c_type,
            Self.c_rank,
            Self.c_tile_shape,
            Self.c_desc_shape,
            stage == Self.num_stages - 1,
        ](self.c_tma_op[])

        comptime if stage > 0 or stage == Self.num_stages - 1:
            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _copy_to_gmem_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → SMEM → GMEM pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform: lets apply_to_both_fragments skip per-position
        # bounds checks for fully-in-bounds tiles. transpose_c swaps the
        # row/col → user-M/user-N mapping.
        var tile_in_bounds: Bool

        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            var frags = accum_tiles[stage].load_fragments[Self.rep]()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                Self.AccumTmemArray.Tile.wait_load()
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to epilogue dtype in SIMD chunks
            # of at least 4 bytes for efficient hardware cast
            # instructions (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Apply epilogue lambda if provided
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    if tile_in_bounds:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=True,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()
                    else:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=False,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()

            var c_smem_tile = c_tiles[stage % 2]

            comptime if (
                Self.register_based_epilogue
                or not Self.elementwise_compute_lambda_fn
            ):
                self._cast_frags_and_write_to_smem(
                    upper_frag_casted,
                    lower_frag_casted,
                    c_smem_tile,
                    UInt32(warp_id),
                    UInt32(lane),
                )
            else:
                var writer = SMemEpilogueWriter[
                    Self.c_smem_dim0,
                    Self.c_smem_dim1,
                    Self.epilogue_dtype,
                    Self.epc,
                    Self.num_output_warps,
                    Self.c_swizzle,
                    simd_size,
                    stage,
                    Self.rep_frag_size,
                    Self.elementwise_compute_lambda_fn.value(),
                ](UInt32(warp_id), c_tiles, c_shape, c_coord)
                writer.write_tile(
                    AccumTile(upper_frag_casted, lower_frag_casted)
                )

            self._tma_store_to_gmem[stage](
                c_smem_tile,
                (c_coord[0], c_coord[1]),
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def _write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
    ):
        """Internal implementation of write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute (not tile index).
        Handles partial tiles that cross expert boundaries by using element-by-element
        stores for rows that would exceed m_end.

        Args:
            c_tiles: SMEM tile array for C output (TileTensor-based).
            output_stage: OutputStage with pipeline, index, and TMEM handle.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM (for bounds-checked stores).
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())
        # Role-relative: this path's `warp_id == 0` single-writer election
        # (TMAStoreCoords, commit_group) means "first EPILOGUE warp", not
        # "first warp in the block". Identical today (the pool starts at
        # thread 0) and correct if the pool ever moves.
        var warp_id = Self.WarpRole.epilogue_role_index()
        var lane = lane_id()
        var scale = expert_scale.cast[Self.accum_type]()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutorLocal = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=False,  # Always 2D for absolute coords
        ]

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for loop_stage in range(Self.num_stages):
            # Phase 1: TMEM Load
            var frags = accum_tiles[loop_stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # rebind bridges symbolic size mismatch between
            # TmemTensor.frag_size*rep and Self.fragment_size*rep
            comptime PartialType2 = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType2](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType2](frags.lower).copy()

            # Phase 2: Barrier Arrive
            comptime if loop_stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale and cast to epilogue dtype in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions.
            comptime cast_width_e = (
                4 // size_of[Scalar[Self.epilogue_dtype]]()
            )

            comptime for _chunk in range(Self.rep_frag_size // cast_width_e):
                comptime offset = _chunk * cast_width_e
                var src = SIMD[Self.accum_type, cast_width_e]()
                comptime for _j in range(cast_width_e):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * scale).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width_e):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(
                    Self.rep_frag_size // cast_width_e
                ):
                    comptime offset = _chunk * cast_width_e
                    var src = SIMD[Self.accum_type, cast_width_e]()
                    comptime for _j in range(cast_width_e):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * scale).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width_e):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Phase 3: SMEM Write
            var c_smem_tile = c_tiles[loop_stage % 2]

            comptime expected_size = Self.epc.fragment_size * Self.rep
            # Cast from epilogue_dtype to c_type in SIMD chunks
            # of at least 4 bytes.
            var upper_c2 = Array[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )
            var lower_c2 = Array[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )

            comptime cast_width_c2 = (4 // size_of[Scalar[Self.c_type]]())
            comptime for _chunk in range(Self.rep_frag_size // cast_width_c2):
                comptime offset = _chunk * cast_width_c2
                var src_u = SIMD[Self.epilogue_dtype, cast_width_c2]()
                var src_l = SIMD[Self.epilogue_dtype, cast_width_c2]()
                comptime for _j in range(cast_width_c2):
                    src_u[_j] = upper_frag_casted[offset + _j]
                    src_l[_j] = lower_frag_casted[offset + _j]
                var dst_u = src_u.cast[Self.c_type]()
                var dst_l = src_l.cast[Self.c_type]()
                comptime for _j in range(cast_width_c2):
                    upper_c2[offset + _j] = dst_u[_j]
                    lower_c2[offset + _j] = dst_l[_j]
            smem_writer.write_fragments[Self.rep](
                rebind[Array[Scalar[Self.c_type], expected_size]](upper_c2),
                rebind[Array[Scalar[Self.c_type], expected_size]](lower_c2),
                c_smem_tile,
            )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

            # Phase 4: TMA Store with bounds checking
            comptime CG2_TMA_BM = Self.c_smem_dim0 if Self.MMA_M == 256 else Self.BM
            comptime CG1_TMA_BM = Self.c_smem_dim0
            comptime TMA_BM = CG2_TMA_BM if Self.cta_group == 2 else CG1_TMA_BM
            comptime StoreCoordsLocal = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                loop_stage,
                batched=False,
            ]

            # Transpose and non-transpose have different bounds checks
            # and coordinate conventions:
            # - Non-transpose: check m_abs (token dim), store_non_transpose
            #   swaps coords so coord_m→tc1(tokens), coord_n→tc0(weights)
            # - Transpose: check per-stage token dim (in n_abs position),
            #   store_transpose_lt passes coord_m→tc0(weights),
            #   coord_n→tc1(tokens)
            var tile_needs_bounds_check: Bool

            comptime if Self.transpose_c:
                # Per-stage bounds check on token dimension (passed as
                # n_abs). Each stage writes stageN token rows.
                var stage_token_start = m_abs + UInt32(loop_stage * Self.stageN)
                tile_needs_bounds_check = (
                    stage_token_start + UInt32(Self.stageN) > m_end
                )
            else:
                tile_needs_bounds_check = m_abs + UInt32(TMA_BM) > m_end

            if tile_needs_bounds_check:
                comptime if Self.transpose_c:
                    # CUDA core fallback for unaligned group boundaries
                    Self._store_with_bounds_check_transpose[c_tensor_layout](
                        c_smem_tile._storage,
                        c_tensor,
                        m_abs + UInt32(loop_stage * Self.stageN),
                        n_abs,
                        m_end,
                    )
                else:
                    # Slow path: element-by-element stores with bounds check
                    Self._store_with_bounds_check[c_tensor_layout](
                        c_smem_tile,
                        c_tensor,
                        m_abs,
                        n_abs + UInt32(loop_stage * Self.stageN),
                        m_end,
                        UInt32(warp_id),
                        UInt32(lane),
                    )

                # Advance TMA group counter so wait_group[1] properly
                # drains the previous stage's in-flight TMA store.
                # Without this, the group count stays stale and
                # wait_group[1] returns immediately, allowing the next
                # double-buffer stage to overwrite SMEM while TMA reads
                # it.
                if warp_id == 0 and lane == 0:
                    self.c_tma_op[].commit_group()
            else:
                # Fast path: TMA store for tiles fully within bounds
                var n_tile = n_abs / UInt32(Self.MMA_N)
                var dummy_m_tile = UInt32(0)
                var store_coords = StoreCoordsLocal(
                    (dummy_m_tile, n_tile), UInt32(warp_id)
                )

                comptime if Self.transpose_c:
                    # Transpose: coord_m→tc0→N(weights),
                    # coord_n→tc1→M(tokens)
                    store_coords.coord_m = Int(n_abs)
                    store_coords.coord_n = Int(m_abs) + loop_stage * Self.stageN
                else:
                    # Non-transpose: coord_m→tc1→M(tokens),
                    # coord_n→tc0→N(weights)
                    store_coords.coord_m = Int(m_abs)
                    store_coords.coord_n = Int(n_abs) + loop_stage * Self.stageN

                StoreExecutorLocal.execute[
                    Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
                ](
                    c_smem_tile,
                    store_coords,
                    self.c_tma_op[],
                    UInt32(warp_id),
                    UInt32(lane),
                )

            # Phase 5: TMA Wait — unconditional to drain outstanding
            # TMA stores from earlier stages when the slow path skips
            # TMA, preventing SMEM races with double-buffered tiles.
            tma_wait_pipelined[
                Self.c_type,
                Self.c_rank,
                Self.c_tile_shape,
                Self.c_desc_shape,
                loop_stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            comptime if loop_stage > 0 or loop_stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @staticmethod
    @always_inline
    def _store_with_bounds_check[
        c_tensor_layout: TensorLayout,
        c_smem_layout: TensorLayout,
    ](
        c_smem_tile: TileTensor[
            Self.c_type, c_smem_layout, MutAnyOrigin, address_space=.SHARED
        ],
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Store SMEM tile to GMEM with per-element bounds checking.

        Used when the tile crosses the expert boundary (m_abs + TMA_BM > m_end).
        Uses element-by-element stores to avoid writing past m_end.

        Args:
            c_smem_tile: SMEM tile to store (TileTensor).
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Absolute M coordinate (start of tile).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            warp_id: Current warp ID.
            lane: Current lane ID.
        """
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime c_smem_M = Self.c_smem_dim0
        comptime TMA_BM = 64 if Self.cta_group == 1 else 128
        # Ensure enough work for all threads: need thread_rows <= TMA_BM,
        # i.e., output_threads / thread_n <= TMA_BM,
        # i.e., simd_size <= stageN * TMA_BM / output_threads.
        comptime max_simd = simd_width_of[Self.c_type]()
        comptime max_allowed_simd = Self.stageN * TMA_BM // output_threads
        comptime simd_size = min(max_simd, max(1, max_allowed_simd))
        comptime alignment = align_of[SIMD[Self.c_type, simd_size]]()
        comptime thread_n = Self.stageN // simd_size
        comptime thread_layout = Layout.row_major(
            output_threads // thread_n, thread_n
        )

        # Precompute the split layout from known dimensions
        comptime split_layout = Layout(
            IntTuple(TMA_BM, Self.stageN), IntTuple(Self.c_smem_dim1, 1)
        )

        # Swizzle function
        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        # Ensure fence before reading from SMEM
        if warp_id == 0 and lane == 0:
            fence_async_view_proxy()

        # Synchronize all epilogue threads
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

        # Iterate over SMEM chunks
        comptime for i in range(c_smem_M // TMA_BM):
            var c_smem_split = c_smem_tile.tile[TMA_BM, Self.stageN](i, 0)
            comptime zipped = zipped_divide(
                upcast(split_layout, simd_size), thread_layout
            )
            # Use new Layout for idx2crd
            comptime split_layout_new = row_major[TMA_BM, Self.stageN]()

            comptime for j in range(zipped.shape[1][0].value()):
                var input_crd = RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE, j),
                    element_type=.uint32,
                ](thread_idx.x, j)
                var linear_idx = rt_crd2idx[
                    IntTuple(UNKNOWN_VALUE, j),
                    zipped.shape,
                    zipped.stride,
                    DType.uint32,
                ](
                    input_crd,
                    RuntimeTuple[zipped.shape](),
                    RuntimeTuple[zipped.stride](),
                ) * UInt32(
                    simd_size
                )
                var cmem_crd = split_layout_new.idx2crd[out_dtype=DType.uint32](
                    Int(linear_idx)
                )
                var local_i = cmem_crd[0].value()
                var local_j = cmem_crd[1].value()
                var coord_m = m_abs + UInt32(i * TMA_BM)
                var global_i = coord_m + UInt32(local_i)
                var global_j = n_abs + UInt32(local_j)

                # Bounds check: only store if within M and N boundaries.
                # The N check prevents row-major wrap-around when the
                # last N-tile extends past the logical output width.
                var in_bounds = global_i < m_end
                comptime if Self.problem_n > 0:
                    in_bounds = in_bounds and (
                        global_j + UInt32(simd_size) <= UInt32(Self.problem_n)
                    )
                if in_bounds:
                    comptime if size_of[Self.c_type]() == 2:
                        var src_ptr = c_smem_split._storage + swizzle(
                            linear_idx
                        )
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            Coord(Int(global_i), Int(global_j)),
                            src,
                        )
                    else:
                        var src_ptr = c_smem_split._storage + linear_idx
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            Coord(Int(global_i), Int(global_j)),
                            src,
                        )

    @staticmethod
    @always_inline
    def _store_with_bounds_check_transpose[
        c_tensor_layout: TensorLayout,
    ](
        c_smem_ptr: UnsafePointer[
            Scalar[Self.c_type], _, address_space=.SHARED
        ],
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
    ):
        """CUDA core fallback for unaligned group boundaries (transpose).

        With SWIZZLE_32B, each swizzle_width chunk (16 bf16 elements) in
        SMEM maps to one TMA row. We read flat SMEM with swizzle(simd_size
        * tidx) and decompose tidx into (vec_chunkM_idx, n_idx, chunk_idx)
        to compute global coordinates.

        This matches the reference implementation in grouped_matmul_sm100_1d1d.

        Args:
            c_smem_ptr: Raw pointer to SMEM tile.
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Token start for this stage.
            n_abs: Weight start (absolute N coordinate).
            m_end: Token boundary (exclusive).
        """
        comptime simd_size = simd_width_of[Self.c_type]()
        comptime swizzle_width = Self.c_swizzle.bytes() // size_of[
            Self.c_type
        ]()
        comptime chunkM = swizzle_width
        comptime vec_chunkM = chunkM // simd_size
        comptime chunk_num = Self.stage_contiguous_size // chunkM
        comptime logical_size = chunk_num * Self.stageN * vec_chunkM
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime assert (
            logical_size % output_threads == 0
        ), "logical_size must be divisible by output_threads"
        comptime value_shape = logical_size // output_threads
        comptime cN = c_tensor.static_shape[1]
        comptime smem_alignment = align_of[SIMD[Self.c_type, simd_size]]()

        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        var n_inbound = Int32(m_end) - Int32(m_abs)

        comptime for v in range(value_shape):
            comptime thread_offset = v * output_threads
            var tidx = UInt32(thread_idx.x) + UInt32(thread_offset)
            var rest, vec_chunkM_idx = divmod(tidx, UInt32(vec_chunkM))
            var n_idx = rest % UInt32(Self.stageN)
            if Int32(n_idx) >= min(n_inbound, Int32(Self.stageN)):
                continue
            var src_idx = UInt32(simd_size) * tidx
            var c_smem_idx = swizzle(src_idx)
            var val_vec = (c_smem_ptr + c_smem_idx).load[
                width=simd_size,
                alignment=smem_alignment,
            ]()
            var chunk_idx = rest // UInt32(Self.stageN)
            # m_abs = token index, n_abs = weight index
            var global_token = m_abs + n_idx
            var global_weight = n_abs + (
                chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
            ) * UInt32(simd_size)
            if global_token < m_end and global_weight < UInt32(cN):
                c_tensor.store[width=simd_size, alignment=smem_alignment](
                    Coord(Int(global_token), Int(global_weight)),
                    val_vec,
                )

    # ========== Residual Add Support ==========
    # Methods for D = lambda(accum) + beta * C residual operations

    @always_inline
    def write_with_residual[
        pipeline_origin: MutOrigin,
        //,
        num_src_stages: Int,
    ](
        self,
        out_tiles: Self.CTileArray,
        stage: Self.Stage,
        src_tile: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            num_src_stages,
            128,
        ],
        src_pipeline: Pointer[
            ProducerConsumerPipeline[num_src_stages], pipeline_origin
        ],
        beta: Scalar[Self.c_type],
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with residual: D = lambda(accum) + beta * C.

        Matches the CUTLASS `sm100_epilogue_tma_warpspecialized` lockstep
        pattern: the epilogue load warp pre-fetches one source sub-tile per
        inner epilogue stage into a `num_src_stages`-deep SMEM pipeline; this
        method drives one `wait_producer / use / consumer_release / step`
        cycle on `src_pipeline` per inner stage. The buffer index is read from
        the pipeline's `consumer_stage()` rather than computed offline, so
        producer and consumer stay synchronized exactly as in CUTLASS's
        `consumer_wait → copy(sC) → consumer_release` per epi sub-tile.

        Pipeline per inner stage:
        1. Load accum from TMEM to registers (epilogue dtype).
        2. Apply `elementwise_compute_lambda_fn` (pre-residual fusion).
        3. Wait for source[k] via `src_pipeline.consume()`; compute
           `D = accum + beta * C` reading from the SMEM buffer at the
           pipeline's current stage index; release source[k] on context exit.
        4. Apply `elementwise_lambda_fn` (post-residual, owns the GMEM store)
           OR stage to output SMEM and TMA-store to GMEM.

        Parameters:
            pipeline_origin: Mutability origin of the source pipeline ref.
            num_src_stages: Number of source SMEM buffers; must equal the
                epi-load pipeline's stage count in the kernel.

        Args:
            out_tiles: Output SMEM tile array (for D output).
            stage: OutputStage with pipeline, index, and TMEM handle.
            src_tile: Source C SMEM tile array (num_src_stages buffers).
            src_pipeline: Pointer to the source producer/consumer pipeline.
                One acquire/release cycle is driven per inner epilogue stage.
            beta: Residual scale factor.
            tile_coord: (m_tile, n_tile) coordinates.
            shape: (M, N) problem dimensions.
            elect_one_warp: Whether this warp is elected for coordination.
        """
        self._copy_to_gmem_with_residual[num_src_stages](
            out_tiles,
            stage,
            src_tile,
            src_pipeline,
            beta,
            tile_coord,
            shape,
        )

    @always_inline
    def _copy_to_gmem_with_residual[
        pipeline_origin: MutOrigin,
        //,
        num_src_stages: Int,
    ](
        self,
        out_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        src_tiles: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            num_src_stages,
            128,
        ],
        src_pipeline: Pointer[
            ProducerConsumerPipeline[num_src_stages], pipeline_origin
        ],
        beta: Scalar[Self.c_type],
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → (+ beta*C) → SMEM → GMEM pipeline with residual.

        Per-inner-stage CUTLASS lockstep: wait the source pipeline, read the
        SMEM tile at `consumer_stage()`, do the residual add, release on
        context exit. The buffer index is provided by the pipeline rather
        than computed locally.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform: lets apply_to_both_fragments skip per-position
        # bounds checks for fully-in-bounds tiles. transpose_c swaps the
        # row/col → user-M/user-N mapping.
        var tile_in_bounds: Bool

        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()
            var casted = frags.cast[Self.epilogue_dtype]()

            var upper_frag_casted = Array[_, Self.rep_frag_size](
                fill_with_unrolled=lambda [i: Int]() -> Scalar[
                    Self.epilogue_dtype
                ]: casted.upper[i]
            )
            var lower_frag_casted = Array[_, Self.rep_frag_size](
                fill_with_unrolled=lambda [i: Int]() -> Scalar[
                    Self.epilogue_dtype
                ]: casted.lower[i]
            )

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Apply epilogue lambda (if present)
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    if tile_in_bounds:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=True,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()
                    else:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=False,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()

            # 3. Apply residual: D = accum + beta * C in registers.
            # CUTLASS-style lockstep: wait, read SMEM at the buffer the
            # producer just filled, do the add, release the stage, advance.
            # Each per-stage SMEM buffer holds exactly one inner stage's
            # source (BM x OutputN), so we pass `stage=0` to the residual
            # helper — its internal `stage * stageN` column offset would
            # otherwise index past the buffer.
            comptime residual_swizzle = make_swizzle[
                Self.c_type, Self.c_swizzle
            ]()
            src_pipeline[].wait_producer()
            var _src_idx = src_pipeline[].consumer_stage()
            var src_smem_tile = src_tiles[Int(_src_idx)]
            var _residual_result = (
                epilogue_applier.add_residual_to_both_fragments[
                    Self.epilogue_dtype,
                    Self.rep_frag_size,
                    Self.is_lower_frag_required,
                    Self.c_type,
                    Self.c_smem_dim1,
                    residual_swizzle,
                ](
                    upper_frag_casted,
                    lower_frag_casted,
                    UInt32(0),
                    src_smem_tile._storage,
                    beta.cast[Self.epilogue_dtype](),
                )
            )
            upper_frag_casted = _residual_result[0].copy()
            lower_frag_casted = _residual_result[1].copy()
            # Release this source stage and advance — every epilogue thread
            # arrives on the consumer mbar (arv_count = 128).
            _ = src_pipeline[].consumer_mbar(_src_idx)[0].arrive()
            src_pipeline[].consumer_step()

            # 4. Final store. When a void `elementwise_lambda_fn` is set the
            # lambda owns the GMEM write (post-residual contract — see
            # `nn/conv/gpu/amd/amd_4wave_conv_residual.mojo` for the matching
            # AMD path); otherwise stage to SMEM and TMA-store.
            comptime if Self.elementwise_lambda_fn:
                # Cast fragments from epilogue_dtype to c_type. Chunked by
                # 4 bytes to match the hardware cvt instruction width.
                comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
                var upper_simd = SIMD[Self.c_type, Self.rep_frag_size]()
                var lower_simd = SIMD[Self.c_type, Self.rep_frag_size]()

                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src_u = SIMD[Self.epilogue_dtype, cast_width]()
                    var src_l = SIMD[Self.epilogue_dtype, cast_width]()
                    comptime for _j in range(cast_width):
                        src_u[_j] = upper_frag_casted[offset + _j]
                        src_l[_j] = lower_frag_casted[offset + _j]
                    var dst_u = src_u.cast[Self.c_type]()
                    var dst_l = src_l.cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        upper_simd[offset + _j] = dst_u[_j]
                        lower_simd[offset + _j] = dst_l[_j]

                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )

                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
            else:
                # Write to output SMEM
                var c_smem_tile = out_tiles[stage % 2]

                comptime if (
                    Self.register_based_epilogue
                    or not Self.elementwise_compute_lambda_fn
                ):
                    self._cast_frags_and_write_to_smem(
                        upper_frag_casted,
                        lower_frag_casted,
                        c_smem_tile,
                        UInt32(warp_id),
                        UInt32(lane),
                    )
                else:
                    var writer = SMemEpilogueWriter[
                        Self.c_smem_dim0,
                        Self.c_smem_dim1,
                        Self.epilogue_dtype,
                        Self.epc,
                        Self.num_output_warps,
                        Self.c_swizzle,
                        simd_size,
                        stage,
                        Self.rep_frag_size,
                        Self.elementwise_compute_lambda_fn.value(),
                    ](UInt32(warp_id), out_tiles, c_shape, c_coord)
                    writer.write_tile(
                        AccumTile(upper_frag_casted, lower_frag_casted)
                    )

                self._tma_store_to_gmem[stage](
                    c_smem_tile,
                    c_coord,
                    UInt32(0),
                    UInt32(warp_id),
                    UInt32(lane),
                )

    @always_inline
    def write_batched_with_tma_epilogue_load[
        epi_load_swizzle: TensorMapSwizzle,
        epilogue_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        epilogue_tile: TileTensor[
            Self.c_type, epilogue_layout, MutAnyOrigin, address_space=.SHARED
        ],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with epilogue tensor addition to global memory.

        Pipeline: TMEM → Registers → (+epilogue from SMEM) → SMEM → GMEM (TMA).
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )

        comptime sub_tile_n = epi_load_swizzle.bytes() // size_of[Self.c_type]()
        comptime epi_rows = Self.MMA_N if Self.transpose_c else Self.BM
        comptime sub_tile_elems = epi_rows * sub_tile_n
        var epi_load_sw = make_swizzle[Self.c_type, epi_load_swizzle]()

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast to epilogue dtype
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = src.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = src.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # 3. Add epilogue tensor from swizzled SMEM.
            # Epilogue SMEM is row-major [M, N] with swizzle. Fragment coords
            # from compute_staged_coords are in TMEM space:
            #   non-transpose: row→M, col→N
            #   transpose:     row→N, col→M
            # So for transpose we swap which TMEM dim maps to epilogue row/col.
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )
            comptime for rep in range(Self.rep):
                comptime inc = rep * 8
                comptime frag_offset = rep * 4

                comptime if Self.transpose_c:
                    # N position: lanes 0-7 → N, lanes 8-15 → N+8
                    var n_pos = local_row + UInt32(lane & 8)
                    var ldm_tidx = n_pos // UInt32(sub_tile_n)
                    var ldm_nloc = n_pos % UInt32(sub_tile_n)
                    var ldm_tile_base = ldm_tidx * UInt32(sub_tile_elems)

                    var ldm_m_row = local_col + UInt32(inc) + UInt32(lane & 7)

                    # Top + bottom via ldmatrix.trans .x2
                    var ldm_off = ldm_tile_base + epi_load_sw(
                        ldm_m_row * UInt32(sub_tile_n) + ldm_nloc
                    )
                    var epi_vals = ld_matrix[simd_width=4, transpose=True](
                        epilogue_tile._storage + Int(ldm_off)
                    )

                    upper_frag_casted[frag_offset] += epi_vals[0].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 1] += epi_vals[1].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += epi_vals[2].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 3] += epi_vals[3].cast[
                        Self.epilogue_dtype
                    ]()

                else:
                    var col_base = local_col + UInt32(inc)
                    var ldm_tidx = col_base // UInt32(sub_tile_n)
                    var ldm_nloc = col_base % UInt32(sub_tile_n)
                    var ldm_tile_base = ldm_tidx * UInt32(sub_tile_elems)

                    # Rows 0-15 via ldmatrix .x2
                    var ldm_row = local_row + UInt32(lane & 15)
                    var ldm_off = ldm_tile_base + epi_load_sw(
                        ldm_row * UInt32(sub_tile_n) + ldm_nloc
                    )
                    var epi_vals = ld_matrix[simd_width=4](
                        epilogue_tile._storage + Int(ldm_off)
                    )

                    upper_frag_casted[frag_offset] += epi_vals[0].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 1] += epi_vals[1].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += epi_vals[2].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 3] += epi_vals[3].cast[
                        Self.epilogue_dtype
                    ]()

            comptime if Self.is_lower_frag_required:
                comptime for rep in range(Self.rep):
                    comptime inc = rep * 8
                    comptime frag_offset = rep * 4

                    comptime if Self.transpose_c:
                        # N position: lanes 0-7 → N+16, lanes 8-15 → N+24
                        var n_pos_l = local_row + 16 + UInt32(lane & 8)
                        var ldm_tidx_l = n_pos_l // UInt32(sub_tile_n)
                        var ldm_nloc_l = n_pos_l % UInt32(sub_tile_n)
                        var ldm_tile_base_l = ldm_tidx_l * UInt32(
                            sub_tile_elems
                        )

                        var ldm_m_row_l = (
                            local_col + UInt32(inc) + UInt32(lane & 7)
                        )

                        # Rows 16-31 via ldmatrix.trans .x2
                        var ldm_off_l = ldm_tile_base_l + epi_load_sw(
                            ldm_m_row_l * UInt32(sub_tile_n) + ldm_nloc_l
                        )
                        var epi_vals_l = ld_matrix[
                            simd_width=4, transpose=True
                        ](epilogue_tile._storage + Int(ldm_off_l))

                        lower_frag_casted[frag_offset] += epi_vals_l[0].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 1] += epi_vals_l[
                            1
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 2] += epi_vals_l[
                            2
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 3] += epi_vals_l[
                            3
                        ].cast[Self.epilogue_dtype]()
                    else:
                        var col_base_l = local_col + UInt32(inc)
                        var ldm_tidx_l = col_base_l // UInt32(sub_tile_n)
                        var ldm_nloc_l = col_base_l % UInt32(sub_tile_n)
                        var ldm_tile_base_l = ldm_tidx_l * UInt32(
                            sub_tile_elems
                        )

                        # Rows 16-31 via ldmatrix .x2
                        var ldm_row_l = local_row + 16 + UInt32(lane & 15)
                        var ldm_off_l = ldm_tile_base_l + epi_load_sw(
                            ldm_row_l * UInt32(sub_tile_n) + ldm_nloc_l
                        )
                        var epi_vals_l = ld_matrix[simd_width=4](
                            epilogue_tile._storage + Int(ldm_off_l)
                        )

                        lower_frag_casted[frag_offset] += epi_vals_l[0].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 1] += epi_vals_l[
                            1
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 2] += epi_vals_l[
                            2
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 3] += epi_vals_l[
                            3
                        ].cast[Self.epilogue_dtype]()

            # 4. Cast to c_type, write to output SMEM, and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def write_batched_with_1d_bias[
        epilogue_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        epilogue_tile: TileTensor[
            Self.c_type, epilogue_layout, MutAnyOrigin, address_space=.SHARED
        ],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with 1D bias addition to global memory.

        Pipeline: TMEM -> Registers -> (+1D bias broadcast from SMEM) -> SMEM -> GMEM (TMA).

        The bias SMEM tile is 1×MMA_N loaded via cp.async (linear layout,
        no swizzle) and then broadcast across all M rows.
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast to epilogue dtype
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = src.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = src.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # 3. Add 1D bias from SMEM.
            # Non-transpose: vectorized 32-bit load (2 consecutive bf16).
            # Transpose: scalar load + broadcast (non-contiguous addresses).
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )
            var top_u = epilogue_applier.coords.top_upper
            var bot_u = epilogue_applier.coords.bottom_upper

            comptime for rep in range(Self.rep):
                comptime inc = rep * 8
                comptime frag_offset = rep * 4

                comptime if Self.transpose_c:
                    var n_top = local_row + top_u[0]
                    var n_bot = local_row + bot_u[0]
                    var b_top = epilogue_tile._storage[Int(n_top)].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset] += b_top
                    upper_frag_casted[frag_offset + 1] += b_top
                    var b_bot = epilogue_tile._storage[Int(n_bot)].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += b_bot
                    upper_frag_casted[frag_offset + 3] += b_bot
                else:
                    var n_col = local_col + top_u[1] + UInt32(inc)
                    var bias = (
                        (epilogue_tile._storage + Int(n_col))
                        .load[width=2]()
                        .cast[Self.epilogue_dtype]()
                    )
                    upper_frag_casted[frag_offset] += bias[0]
                    upper_frag_casted[frag_offset + 1] += bias[1]
                    upper_frag_casted[frag_offset + 2] += bias[0]
                    upper_frag_casted[frag_offset + 3] += bias[1]

            comptime if Self.is_lower_frag_required:
                var top_l = epilogue_applier.coords.top_lower
                var bot_l = epilogue_applier.coords.bottom_lower

                comptime for rep in range(Self.rep):
                    comptime inc = rep * 8
                    comptime frag_offset = rep * 4

                    comptime if Self.transpose_c:
                        var n_top_l = local_row + top_l[0]
                        var n_bot_l = local_row + bot_l[0]
                        var b_top_l = epilogue_tile._storage[Int(n_top_l)].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset] += b_top_l
                        lower_frag_casted[frag_offset + 1] += b_top_l
                        var b_bot_l = epilogue_tile._storage[Int(n_bot_l)].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 2] += b_bot_l
                        lower_frag_casted[frag_offset + 3] += b_bot_l
                    else:
                        var n_col_l = local_col + top_l[1] + UInt32(inc)
                        var bias = (
                            (epilogue_tile._storage + Int(n_col_l))
                            .load[width=2]()
                            .cast[Self.epilogue_dtype]()
                        )
                        lower_frag_casted[frag_offset] += bias[0]
                        lower_frag_casted[frag_offset + 1] += bias[1]
                        lower_frag_casted[frag_offset + 2] += bias[0]
                        lower_frag_casted[frag_offset + 3] += bias[1]

            # 4. Cast to c_type, write to output SMEM, and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def write_batched_with_tma_epilogue_load_strips[
        epi_load_swizzle: TensorMapSwizzle,
        num_epi_stages: Int,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        mut epilogue_pipeline: ProducerConsumerPipeline[num_epi_stages],
        epilogue_tiles_base: UnsafePointer[
            Scalar[Self.c_type], MutAnyOrigin, address_space=.SHARED
        ],
        epilogue_tile_elems: Int,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with BM×stageN pipelined epilogue addition.

        For non-AB_swapped configs. Each epilogue pipeline stage is one BM×stageN tile.
        Producer sends tiles in stage-outer / col_wg-inner order; consumer mirrors that
        structure so each TMEM stage is fully processed (load → add epilogue → write)
        before advancing to the next.
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id), UInt32(lane), c_shape
        )

        # Each epilogue SMEM stage is BM×stageN. sub_tile_n equals stageN in all modes:
        #   SWIZZLE_NONE  (stageN= 8): sub_tile_n= 8
        #   SWIZZLE_32B   (stageN=16): sub_tile_n=16  (32 bytes / 2 bytes per bf16)
        #   SWIZZLE_64B   (stageN=32): sub_tile_n=32
        #   SWIZZLE_128B  (stageN=64): sub_tile_n=64
        comptime sub_tile_n = Self.stageN
        var epi_load_sw = make_swizzle[Self.c_type, epi_load_swizzle]()

        comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())
        comptime num_col_warp_groups = Self.MMA_N // (
            Self.num_stages * Self.stageN
        )

        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            var upper_partial = rebind[PartialType](frags.upper).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast upper (and lower) accumulator fragments to epilogue dtype
            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime off = _chunk * cast_width
                var src_u = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src_u[_j] = upper_partial[off + _j]
                var dst_u = src_u.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[off + _j] = dst_u[_j]

            comptime if Self.is_lower_frag_required:
                var lower_partial = rebind[PartialType](frags.lower).copy()
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime off = _chunk * cast_width
                    var src_l = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src_l[_j] = lower_partial[off + _j]
                    var dst_l = src_l.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[off + _j] = dst_l[_j]

            # 3. Consume epilogue pipeline stages and add to register fragments.
            # Producer sends tiles in stage-outer / col_wg-inner order so the
            # (stage, col_wg)-th tile covers N columns
            #   [col_wg*num_stages*stageN + stage*stageN,
            #    col_wg*num_stages*stageN + (stage+1)*stageN).
            # For cta_group=2/MMA_M=128 (num_col_warp_groups=2): warps 0+1 own
            # col_wg=0 columns, warps 2+3 own col_wg=1 columns. All warps
            # participate in every pipeline step for correct barrier counts; only
            # the matching warp group adds the loaded values.
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )

            comptime for col_wg in range(num_col_warp_groups):
                epilogue_pipeline.wait_producer()
                var epi_stage_idx = Int(epilogue_pipeline.consumer_stage())
                var tile_ptr = (
                    epilogue_tiles_base + epi_stage_idx * epilogue_tile_elems
                )

                comptime expected_col = (
                    col_wg * Self.num_stages * Self.stageN + stage * Self.stageN
                )
                var is_my_col_group = local_col == UInt32(expected_col)

                comptime for rep in range(Self.rep):
                    # In-tile column = rep*8 (always < stageN, so ldm_tidx = 0)
                    comptime col_in_tile = rep * 8
                    var ldm_row = local_row + UInt32(lane & 15)
                    var ldm_off = epi_load_sw(
                        ldm_row * UInt32(sub_tile_n) + UInt32(col_in_tile)
                    )
                    var epi_vals = ld_matrix[simd_width=4](
                        tile_ptr + Int(ldm_off)
                    )

                    if is_my_col_group:
                        upper_frag_casted[rep * 4] += epi_vals[0].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 1] += epi_vals[1].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 2] += epi_vals[2].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 3] += epi_vals[3].cast[
                            Self.epilogue_dtype
                        ]()

                    comptime if Self.is_lower_frag_required:
                        var ldm_row_l = local_row + 16 + UInt32(lane & 15)
                        var ldm_off_l = epi_load_sw(
                            ldm_row_l * UInt32(sub_tile_n) + UInt32(col_in_tile)
                        )
                        var epi_vals_l = ld_matrix[simd_width=4](
                            tile_ptr + Int(ldm_off_l)
                        )

                        if is_my_col_group:
                            lower_frag_casted[rep * 4] += epi_vals_l[0].cast[
                                Self.epilogue_dtype
                            ]()
                            lower_frag_casted[rep * 4 + 1] += epi_vals_l[
                                1
                            ].cast[Self.epilogue_dtype]()
                            lower_frag_casted[rep * 4 + 2] += epi_vals_l[
                                2
                            ].cast[Self.epilogue_dtype]()
                            lower_frag_casted[rep * 4 + 3] += epi_vals_l[
                                3
                            ].cast[Self.epilogue_dtype]()

                _ = epilogue_pipeline.consumer_mbar(UInt32(epi_stage_idx))[
                    0
                ].arrive()
                epilogue_pipeline.consumer_step()

            # 4. Write to output SMEM and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )


# ===----------------------------------------------------------------------=== #
# StandardOutputWriter - default OutputWriter policy (local TMA store)
# ===----------------------------------------------------------------------=== #


struct StandardOutputWriter(OutputWriter):
    """Default `OutputWriter` policy: local TMA store via `TileWriter`.

    One peer, no cross-GPU synchronization. This is the writer policy
    `BlackwellMatmulSM100Kernel` uses unless a reduce-scatter policy is
    injected. Target hardware: SM100 (B200).
    """

    comptime needs_sync = False
    comptime num_peers = 1

    @staticmethod
    @always_inline
    def write_batched[
        tma_origin: ImmOrigin,
        c_type: DType,
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        a_type: DType,
        accum_type: DType,
        block_tile_shape: IndexList[3],
        mma_shape: IndexList[3],
        opc: OutputPipelineConfig,
        c_swizzle: TensorMapSwizzle,
        transpose_c: Bool,
        c_smem_dim0: Int,
        c_smem_dim1: Int,
        num_output_stages: Int,
        num_output_warps: Int,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type],
        elementwise_compute_lambda_fn: Optional[
            elementwise_compute_lambda_type
        ],
        register_based_epilogue: Bool,
    ](
        c_tma_ops: Pointer[
            Array[
                TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
                Self.num_peers,
            ],
            tma_origin,
        ],
        c_tiles: SMemTileArray2DRowMajor[
            c_type, c_smem_dim0, c_smem_dim1, num_output_stages
        ],
        stage: OutputStage[opc],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Local TMA store of one batched output tile (uses descriptor [0]).

        Parameters:
            tma_origin: Memory origin of the TMA descriptor pointer
                (inferred).
            c_type: Element dtype of the C output tensor (inferred).
            c_rank: Rank of the C output tensor (inferred).
            c_tile_shape: Per-tile shape of the C output (inferred).
            c_desc_shape: TMA descriptor shape for C (inferred).
            a_type: Element dtype of the A input matrix.
            accum_type: Accumulator dtype stored in TMEM.
            block_tile_shape: Block tile shape as (BM, BN, BK).
            mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
            opc: Output pipeline config bundling accumulator stages,
                stage stride, and CTA group.
            c_swizzle: TMA swizzle pattern for the C SMEM layout.
            transpose_c: Whether C is stored transposed.
            c_smem_dim0: Row dimension of the C SMEM tile.
            c_smem_dim1: Column dimension of the C SMEM tile.
            num_output_stages: Number of C SMEM pipeline stages.
            num_output_warps: Number of warps driving the output
                pipeline.
            elementwise_lambda_fn: Optional elementwise epilogue applied
                to fragments before the store.
            elementwise_compute_lambda_fn: Optional compute epilogue
                fused into the register path.
            register_based_epilogue: Whether the compute epilogue runs
                in registers (true) or SMEM (false).

        Args:
            c_tma_ops: Pointer to the array of TMA store descriptors for
                C.
            c_tiles: SMEM tile array for the C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile, batch) tile coordinates.
            shape: (M, N) problem dimensions.
            alpha: Scalar applied to fragments before the store
                (defaults to 1.0).
        """
        # The descriptor params (tma_origin, c_type, c_rank, c_tile_shape,
        # c_desc_shape) are inferred from the `c_tma_ops` ctor arg.
        var writer = TileWriter[
            a_type=a_type,
            accum_type=accum_type,
            block_tile_shape=block_tile_shape,
            mma_shape=mma_shape,
            opc=opc,
            c_swizzle=c_swizzle,
            transpose_c=transpose_c,
            c_smem_dim0=c_smem_dim0,
            c_smem_dim1=c_smem_dim1,
            num_output_stages=num_output_stages,
            num_output_warps=num_output_warps,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            batched=True,
            num_peers=1,
        ](c_tma_ops)
        writer.write_batched(c_tiles, stage, tile_coord, shape, alpha)
