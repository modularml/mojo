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
"""SM100 Matmul CPU entry points - TMA setup and kernel launch wrappers.

This module contains the CPU-side code for SM100 matrix multiplication:
- TMA descriptor creation
- Kernel instantiation and launch via ctx.enqueue_function

All GPU code (kernel structs, runtime functions) is in matmul_kernels.mojo.
"""

from std.math import align_up, ceildiv
from std.sys import size_of

from comm import MAX_GPUS, Signal
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from max.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    Coord,
    Idx,
    DefaultEngine,
    RowMajorLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major as tt_row_major,
)
from structured_kernels.tile_types import create_tma_tile
from structured_kernels.kernel_common import _to_batched_3d

from std.utils.index import Index, IndexList
from std.collections import OptionalReg
from std.utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.tile_scheduler_splitk import (
    get_required_locks_buffer_size_bytes,
    get_num_tiles,
)
from linalg.matmul.gpu.profiler import MatmulWarpSpecializationWorkSpaceManager

# Import kernel structs and GPU functions from matmul_kernels
from .matmul_kernels import (
    B200MatmulSmem,
    BlackwellMatmulSM100Kernel,
    BlackwellMatmulSM100FallbackKernel,
)


def _blackwell_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
    EpilogueLayoutType: TensorLayout = RowMajorLayout[Int64],
    EpilogueEngine: TensorEngine = DefaultEngine[element_width=1],
](
    c_device: TileTensor,
    a_device: TileTensor,
    b_device: TileTensor,
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[
            config.c_type,
            EpilogueLayoutType,
            ImmutAnyOrigin,
            Engine=EpilogueEngine,
        ]
    ] = None,
) raises:
    """Internal matmul launch for SM100. Always takes rank-3 TileTensors.

    Creates 3D TMA descriptors and launches kernel.run().
    grid_dim.z = batch_size (1 for non-batched).
    Callers must reshape rank-2 inputs to rank-3 before calling this function.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime assert transpose_b, "Only support transposed B"

    comptime register_based_epilogue = config.register_based_epilogue

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime assert (
        elementwise_compute_lambda_fn is None or elementwise_lambda_fn is None
    ), "Either the epilogue lambda or the compute lambda can be used"

    comptime if config.cta_group == 2:
        comptime assert (
            MMA_M == 256 or MMA_M == 128
        ), "Only support cta_group == 2 with MMA_M == 128 or 256"
        comptime assert (MMA_M != 256) or (
            MMA_N % 16 == 0
        ), "MMA_N must be a multiple of 16 when MMA_M is 256"
        comptime assert (
            config.AB_swapped
            or MMA_M != 128
            or register_based_epilogue
            or elementwise_compute_lambda_fn is None
        ) or (MMA_N % 16 == 0), (
            "SM100 doesn't support shared memory based epilogue when MMA_M =="
            " 128 and MMA_N is not a multiple of 16"
        )
    else:
        comptime assert (
            MMA_M == 128 or MMA_M == 64
        ), "Only support MMA_M == 128 or 64 when cta_group == 1"

    comptime if c_type == .float32:
        comptime assert (
            a_type == b_type == .float32
        ), "Only support float32 input types is tested for float32 output dtype"
        comptime assert (
            register_based_epilogue
        ), "only register-based epilogue is supported for float32 output dtype"

    # requirements for float8_e4m3fn output dtype
    comptime if c_type == .float8_e4m3fn:
        comptime assert a_type == b_type == .bfloat16, (
            "Only support bfloat16 input types is tested for float8_e4m3fn"
            " output dtype"
        )
        comptime assert (
            config.c_swizzle == TensorMapSwizzle.SWIZZLE_NONE
        ), "c_swizzle must be for float8_e4m3fn output dtype"
        comptime assert (
            (config.cta_group == 1 or MMA_M == 256) and MMA_N % 16 == 0
        ) or (
            (config.cta_group == 2 or MMA_M == 128) and MMA_N % 32 == 0
        ), "MMA_N must be a multiple of 16/32 for float8_e4m3fn output dtype"
        comptime assert register_based_epilogue, (
            "only register-based epilogue is supported for float8_e4m3fn output"
            " dtype"
        )

    comptime cluster_shape = config.cluster_shape

    var B = Int(c_device.dim[0]())
    var M = Int(c_device.dim[1]())
    var N = Int(c_device.dim[2]())
    var M_maybe_swapped = Int(a_device.dim[1]())
    var N_maybe_swapped = Int(b_device.dim[1]())
    comptime K = type_of(a_device).LayoutType.static_shape[2]

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime epilogue_is_1d = config.epilogue_is_1d

    comptime assert not (
        config.use_tma_epilogue_load
        and elementwise_compute_lambda_fn is not None
    ), (
        "use_tma_epilogue_load is mutually exclusive with"
        " elementwise_compute_lambda_fn"
    )

    comptime SmemType = B200MatmulSmem[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    # Instantiate kernel first -- TMA layouts are computed from config
    comptime matmul_kernel = BlackwellMatmulSM100Kernel[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    # Create 3D TMA descriptors using kernel's primary layout types
    comptime KernelType = type_of(matmul_kernel)

    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tma_tile[
        KernelType.ATileLayout,
        KernelType.ADescLayout,
        a_tma_tile_shape,
        swizzle_mode=config.a_swizzle,
    ](ctx, a_device)

    # fmt: off
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    ) if transpose_b else Index(
        1, BK, BN // (cluster_shape[0] // config.cta_group)
    )
    var b_tma_op = create_tma_tile[
        KernelType.BTileLayout,
        KernelType.BDescLayout,
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        1, 64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(1, config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    ) if (MMA_M == 256 or config.cta_group == 1) else c_tma_tile_shape_mma128

    comptime assert (not config.AB_swapped) or config.c_swizzle.bytes() in (128, 16), "Only support 128B or None swizzle mode when AB_swapped is True"
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    comptime c_tma_tile_shape_final = c_tma_tile_shape if not config.AB_swapped else Index(
        1, c_tma_tile_shape[1], c_tma_tile_shape_1
    )

    var c_tma_op = create_tma_tile[
        KernelType.CTileLayout,
        KernelType.CDescLayout,
        c_tma_tile_shape_final,
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)
    # fmt: on

    comptime assert (not config.use_tma_epilogue_load) or (
        c_type == .bfloat16 or (config.epilogue_is_1d and c_type == .float32)
    ), "TMA epilogue load is only supported for bfloat16 (2D) or float32 (1D)"

    # Epilogue tensor TMA descriptor (2D only; 1D uses cp.async.bulk).
    # 2D bias: epilogue tensor is 2D (M, N) in GMEM.
    #   When AB_swapped, the CTA's M/N swap, so the tile becomes (MMA_N, BM).
    # When no epilogue tensor or 1D, c_device._storage is used as a valid
    # placeholder;
    # the descriptor is never accessed by the kernel.
    comptime epi_load_tma_tile_rows = MMA_N if config.AB_swapped else BM
    comptime epi_load_tma_tile_cols = BM if config.AB_swapped else config.output_tile_shape[
        1
    ]
    comptime epi_load_tma_tile_shape = Index(
        epi_load_tma_tile_rows, epi_load_tma_tile_cols
    )
    comptime MutPtr = UnsafePointer[Scalar[c_type], MutAnyOrigin]
    var epi_load_tma_ptr: MutPtr
    var epi_load_tma_rows: Int
    var epi_load_tma_cols: Int
    comptime if config.use_tma_epilogue_load and not epilogue_is_1d:
        var epi_tt = epilogue_tensor.value()
        epi_load_tma_ptr = rebind[MutPtr](epi_tt._storage)
        epi_load_tma_rows = Int(epi_tt.dim(0))
        epi_load_tma_cols = Int(epi_tt.dim(1))
    else:
        epi_load_tma_ptr = rebind[MutPtr](c_device._storage)
        epi_load_tma_rows = epi_load_tma_tile_rows
        epi_load_tma_cols = epi_load_tma_tile_cols

    var epi_load_2d = TileTensor(
        epi_load_tma_ptr,
        tt_row_major(Coord(IndexList[2](epi_load_tma_rows, epi_load_tma_cols))),
    )
    var epi_load_tma_op = create_tma_tile[
        KernelType.EpilogueLoadTileLayout,
        KernelType.EpilogueLoadDescLayout,
        epi_load_tma_tile_shape,
        swizzle_mode=config.epi_load_swizzle,
    ](ctx, epi_load_2d)

    # 1D bias: pass raw TileTensor to kernel (cp.async.bulk, no TMA descriptor).
    # For non-1D, a placeholder dangling pointer is used (never accessed).
    comptime ImmutPtr = UnsafePointer[Scalar[c_type], ImmutAnyOrigin]
    var bias_1d_ptr: ImmutPtr
    comptime if epilogue_is_1d:
        bias_1d_ptr = rebind[ImmutPtr](epilogue_tensor.value()._storage)
    else:
        bias_1d_ptr = rebind[ImmutPtr](c_device._storage)
    var bias_1d_tile = KernelType.Bias1DTile(
        bias_1d_ptr,
        KernelType.Bias1DTileLayout,
    )

    # Get the kernel entry point from the struct
    comptime kernel = matmul_kernel.run

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        B,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace: Span[UInt64, MutAnyOrigin]

    comptime if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = {}

    # This is wrapped in an Array to match reduce-scatter friendly kernel interface
    var c_tma_ops: Array[type_of(c_tma_op), 1] = [c_tma_op]
    var rank_sigs: Optional[
        Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS]
    ] = None

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_ops,
        epi_load_tma_op,
        bias_1d_tile,
        cluster_dim,
        mnk,
        workspace,
        rank_sigs,
        Int32(0),
        grid_dim=grid_dim,
        block_dim=KernelType.NUM_THREADS,
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    comptime if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


def blackwell_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
    EpilogueLayoutType: TensorLayout = RowMajorLayout[Int64, Int64],
    EpilogueEngine: TensorEngine = DefaultEngine[element_width=1],
](
    c_device: TileTensor,
    a_device: TileTensor,
    b_device: TileTensor,
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[
            config.c_type,
            EpilogueLayoutType,
            ImmutAnyOrigin,
            Engine=EpilogueEngine,
        ]
    ] = None,
) raises:
    """Public entry point for SM100 matmul (non-batched, rank-2 inputs).

    Split-K uses separate 2D path. Non-split-K delegates to
    blackwell_batched_matmul_tma_umma_warp_specialized which handles
    _to_batched_3d wrapping and AB_swapped dispatch.

    Parameters:
        transpose_b: Whether B is stored transposed as (N, K). Must be True.
        config: Matmul configuration holding tile shapes, dtypes, swizzle
            modes, cluster shape, and pipeline stages.
        elementwise_lambda_fn: Optional epilogue lambda applied in the
            epilogue phase (defaults to None).
        elementwise_compute_lambda_fn: Optional compute lambda applied in the
            compute phase; mutually exclusive with elementwise_lambda_fn
            (defaults to None).
        pdl_level: Programmatic dependent launch level for the kernel
            (defaults to PDLLevel()).
        max_profiled_tiles_per_SM: Maximum number of tiles to profile per SM;
            when set, enables kernel profiling (defaults to None).
        EpilogueLayoutType: Layout type of the epilogue tensor (defaults to
            RowMajorLayout[Int64, Int64]).
        EpilogueEngine: Engine of the epilogue tensor (defaults to
            DefaultEngine[element_width=1]).
    Args:
        c_device: Output TileTensor of shape (M, N).
        a_device: LHS TileTensor of shape (M, K).
        b_device: RHS TileTensor of shape (N, K) (transposed).
        ctx: Device context used to create TMA descriptors and enqueue the
            kernel.
        epilogue_tensor: Optional epilogue tensor (for example, bias) consumed
            by the epilogue lambda (defaults to None).
    """
    comptime if config.num_split_k > 1:
        comptime if config.AB_swapped:
            comptime new_config = config.swap_AB_type()

            # When both A and B are K-major, then the matrix multiplication
            # math is C = A @ B'. If we swap A and B, we have D = B @ A'.
            # Note that D' = (B @ A')' = A'' @ B' = A @ B' which is the same
            # as the original math. Therefore, when we swap A and B, we need
            # to transpose the result for consistency and correctness.
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                transpose_b,
                config=new_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, b_device, a_device, ctx)
        else:
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)
    else:
        comptime if config.use_tma_epilogue_load:
            blackwell_batched_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](
                c_device,
                a_device,
                b_device,
                ctx,
                epilogue_tensor=epilogue_tensor,
            )
        else:
            blackwell_batched_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)


def _blackwell_matmul_tma_umma_warp_specialized_split_k[
    transpose_b: Bool,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: TileTensor,
    a_device: TileTensor,
    b_device: TileTensor,
    ctx: DeviceContext,
) raises:
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime assert transpose_b, "Only support transposed B"

    comptime register_based_epilogue = config.register_based_epilogue

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert (
        elementwise_lambda_fn is None
    ), "Split-K does not support elementwise epilogue function yet!"
    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime if config.cta_group == 2:
        comptime assert (
            MMA_M == 256 or MMA_M == 128
        ), "Only support cta_group == 2 with MMA_M == 128 or 256"
        comptime assert (MMA_M != 256) or (
            MMA_N % 16 == 0
        ), "MMA_N must be a multiple of 16 when MMA_M is 256"

        # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
        comptime assert (
            not config.AB_swapped
        ) or MMA_M == 256, "swapAB is only supported for MMA_M == 256"

    else:
        comptime assert (
            MMA_M == 128 or MMA_M == 64
        ), "Only support MMA_M == 128 or 64 when cta_group == 1"
        comptime assert (
            register_based_epilogue or elementwise_compute_lambda_fn is None
        ), "only register-based epilogue is supported for cta_group == 1"

    comptime cluster_shape = config.cluster_shape

    var M = Int(c_device.dim[0]())
    var N = Int(c_device.dim[1]())
    var M_maybe_swapped = Int(a_device.dim[0]())
    var N_maybe_swapped = Int(b_device.dim[0]())
    comptime K = type_of(a_device).LayoutType.static_shape[1]

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime smem_size = size_of[SmemType]()
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    # Instantiate kernel first -- TMA layouts are computed from config
    comptime matmul_kernel = BlackwellMatmulSM100Kernel[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    # Create 2D TMA descriptors using kernel's _splitk layout types
    comptime KernelType = type_of(matmul_kernel)

    var a_tma_op = create_tma_tile[
        KernelType.ATileLayout_splitk,
        KernelType.ADescLayout_splitk,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode=config.a_swizzle,
    ](ctx, a_device)

    var b_tma_op = create_tma_tile[
        KernelType.BTileLayout_splitk,
        KernelType.BDescLayout_splitk,
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    # c_swizzle is set to 32B mode when swapAB is enabled so we need to adjust
    # the tile shape with 128B swizzle mode, there should always be 64 elements
    # on the contiguous dim.
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        KernelType.CTileLayout_splitk,
        KernelType.CDescLayout_splitk,
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode=config.c_swizzle,
    ](ctx, c_device)

    # Get the split-K kernel entry point.
    # Reduction TileTensor layout: shape = (UNKNOWN, BM, MMA_N),
    # strides = (BM*MMA_N, MMA_N, 1) -- all strides are static.
    comptime ReductionTTLayout = type_of(reduction_tensor).LayoutType
    comptime kernel = matmul_kernel.run_splitk[ReductionTTLayout]

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        config.num_split_k,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace: Span[UInt64, MutAnyOrigin]

    var output_tiles = get_num_tiles(
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )
    var num_output_tiles = output_tiles[0] * output_tiles[1]
    var lock_buffer_size_bytes = get_required_locks_buffer_size_bytes[
        config.accum_type
    ](
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )

    var locks_buffer = ctx.enqueue_create_buffer[.uint8](lock_buffer_size_bytes)
    var reduction_workspace = ctx.enqueue_create_buffer[config.accum_type](
        num_output_tiles * BM * MMA_N
    )

    var reduction_tensor = TileTensor(
        reduction_workspace,
        tt_row_major(
            (
                Int64(num_output_tiles),
                Idx[BM],
                Idx[MMA_N],
            )
        ),
    )

    ctx.enqueue_memset(locks_buffer, 0)

    comptime if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin]()

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        reduction_tensor,
        locks_buffer,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )

    _ = reduction_workspace^
    _ = locks_buffer^

    comptime if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


# =============================================================================
# Batched matmul helpers and entry points
# =============================================================================


def blackwell_batched_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
    EpilogueLayoutType: TensorLayout = RowMajorLayout[Int64, Int64],
    EpilogueEngine: TensorEngine = DefaultEngine[element_width=1],
](
    c_device: TileTensor,
    a_device: TileTensor,
    b_device: TileTensor,
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[
            config.c_type,
            EpilogueLayoutType,
            ImmutAnyOrigin,
            Engine=EpilogueEngine,
        ]
    ] = None,
) raises:
    """Public entry point for batched SM100 BF16 matmul.

    Accepts rank-2 (non-batched, batch=1) or rank-3 (batched) TileTensors.
    Rank-2 inputs are reshaped to 3D before calling the internal function.
    Handles AB_swapped dispatch.

    Parameters:
        transpose_b: Whether B is stored transposed as (N, K). Must be True.
        config: Matmul configuration holding tile shapes, dtypes, swizzle
            modes, cluster shape, and pipeline stages.
        elementwise_lambda_fn: Optional epilogue lambda applied in the
            epilogue phase (defaults to None).
        elementwise_compute_lambda_fn: Optional compute lambda applied in the
            compute phase; mutually exclusive with elementwise_lambda_fn
            (defaults to None).
        pdl_level: Programmatic dependent launch level for the kernel
            (defaults to PDLLevel()).
        max_profiled_tiles_per_SM: Maximum number of tiles to profile per SM;
            when set, enables kernel profiling (defaults to None).
        EpilogueLayoutType: Layout type of the epilogue tensor (defaults to
            RowMajorLayout[Int64, Int64]).
        EpilogueEngine: Engine of the epilogue tensor (defaults to
            DefaultEngine[element_width=1]).
    Args:
        c_device: Output TileTensor of shape (M, N) or (B, M, N).
        a_device: LHS TileTensor of shape (M, K) or (B, M, K).
        b_device: RHS TileTensor of shape (N, K) or (B, N, K) (transposed).
        ctx: Device context used to create TMA descriptors and enqueue the
            kernel.
        epilogue_tensor: Optional epilogue tensor (for example, bias) consumed
            by the epilogue lambda (defaults to None).
    """
    comptime if type_of(c_device).rank == 2:
        comptime if config.AB_swapped:
            comptime new_config = config.swap_AB_type()
            comptime SwappedEpilogue = OptionalReg[
                TileTensor[
                    new_config.c_type,
                    EpilogueLayoutType,
                    ImmutAnyOrigin,
                    Engine=EpilogueEngine,
                ]
            ]
            _blackwell_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=new_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](
                _to_batched_3d(c_device),
                _to_batched_3d(b_device),
                _to_batched_3d(a_device),
                ctx,
                rebind[SwappedEpilogue](epilogue_tensor),
            )
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](
                _to_batched_3d(c_device),
                _to_batched_3d(a_device),
                _to_batched_3d(b_device),
                ctx,
                epilogue_tensor,
            )
    else:
        comptime if config.AB_swapped:
            comptime new_config = config.swap_AB_type()
            comptime SwappedEpilogue = OptionalReg[
                TileTensor[
                    new_config.c_type,
                    EpilogueLayoutType,
                    ImmutAnyOrigin,
                    Engine=EpilogueEngine,
                ]
            ]
            _blackwell_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=new_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](
                c_device,
                b_device,
                a_device,
                ctx,
                rebind[SwappedEpilogue](epilogue_tensor),
            )
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx, epilogue_tensor)


def matmul_sm100_fallback[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](c: TileTensor, a: TileTensor, b: TileTensor, ctx: DeviceContext,) raises:
    """Launches the SM100 fallback matmul kernel for unsupported shapes or dtypes.

    Uses a simple non-pipelined kernel without CLC/TMEM overhead, computing
    shared memory from the actual tile sizes rather than the hardware maximum.
    Only transposed-B, bfloat16, and float8_e4m3fn inputs are supported.

    Parameters:
        c_type: Output element dtype of the result matrix.
        a_type: Input element dtype of the LHS matrix; must be
            `bfloat16` or `float8_e4m3fn`.
        b_type: Input element dtype of the RHS matrix; must equal `a_type`.
        transpose_b: Whether B is stored transposed as (N, K). Must be True.
        umma_shape: Tensor core MMA instruction shape as a 3-element
            `IndexList` of (M, N, K).
        block_tile_shape: CTA block tile shape as a 3-element
            `IndexList` of (BM, BN, BK) giving the tile dimensions
            along M, N, and K.
        a_swizzle: TMA swizzle mode for A tensor loads (defaults to
            `SWIZZLE_128B`).
        b_swizzle: TMA swizzle mode for B tensor loads (defaults to
            `SWIZZLE_128B`).
        elementwise_lambda_fn: Optional epilogue lambda applied after
            the matmul (defaults to None).
    Args:
        c: Output TileTensor of shape (M, N).
        a: LHS TileTensor of shape (M, K).
        b: RHS TileTensor of shape (N, K) (transposed).
        ctx: Device context used to create TMA descriptors and enqueue the kernel.
    """
    comptime assert transpose_b, "Only support transposed B"

    comptime assert a_type == b_type and a_type in (
        DType.bfloat16,
        DType.float8_e4m3fn,
    ), "Only support bfloat16 and float8_e4m3fn"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    # Fallback kernel uses actual computed SMEM (not b200_smem hardware max)
    # because it's a simple non-pipelined kernel without CLC/TMEM overhead.
    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime block_dim = 128

    # Instantiate fallback kernel first (TMA layouts computed from config)
    comptime fallback_kernel = BlackwellMatmulSM100FallbackKernel[
        a_type,
        b_type,
        c_type,
        type_of(c).LayoutType,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    comptime FallbackKernelType = type_of(fallback_kernel)
    comptime kernel = fallback_kernel.run

    # Create TMA descriptors using kernel-derived layout types
    var a_tma_op = create_tma_tile[
        FallbackKernelType.ATileLayout,
        FallbackKernelType.ADescLayout,
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a)
    var b_tma_op = create_tma_tile[
        FallbackKernelType.BTileLayout,
        FallbackKernelType.BDescLayout,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        Int32(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
