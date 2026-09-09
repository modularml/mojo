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
"""Dispatch entry points for SM100 (B200+) structured matmul kernels.

Selects between GEMV, split-K GEMV, small-MN GEMMs, heuristic outlier configs,
and the warp-specialized TMA/UMMA tile GEMM based on problem shape and dtype,
falling back to vendor BLAS only for untuned or low-performance shapes.
"""
from std.math import align_up, ceildiv
from std.sys import (
    get_defined_bool,
    get_defined_int,
    simd_width_of,
    size_of,
    has_nvidia_gpu_accelerator,
)

from max.algorithm import elementwise
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from layout import (
    Coord,
    Idx,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
)
from layout.tile_tensor import NullableTileTensor
from std.logger import Logger

from std.utils.index import Index, IndexList
from std.collections import OptionalReg

from .....utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .....utils_gpu import MatmulKernels, _vendor_blas_fallback_disabled
from ..structured_kernels.config import (
    MatmulConfig,
    build_sm100_matmul_configs,
    build_sm100_batched_matmul_configs,
    choose_config,
    default_matmul_config_bf16_fp8,
    GEMMKind,
)
from ... import matmul_kernel_naive, gemv_gpu, multistage_gemm, gemm_mma_cpasync
from ....vendor.matmul import matmul as matmul_vendor
from ...tile_scheduler import RasterOrder
from linalg.gemv import gemv_split_k, gemv_gpu_dispatch, GEMVAlgorithm
from .matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
    blackwell_batched_matmul_tma_umma_warp_specialized,
)
from internal_utils import Table
from .tuning_configs import (
    _get_tuning_list_sm100_fp8,
    _get_tuning_list_sm100_fp32,
    TuningConfigSM100,
    TuningConfigSmallMNGemms,
    _get_tuning_list_sm100_bf16,
    _get_tuning_list_sm100_batched_bf16,
    _get_tuning_list_sm100_batched_fp8,
    _get_tuning_list_sm100_batched_fp32,
    _get_tuning_list_small_MN_gemms_bf16,
)

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1

comptime logger = Logger()


@always_inline
def small_MN_gemms[
    config: TuningConfigSmallMNGemms,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    ctx: DeviceContext,
) raises:
    """Launches a small-MN GEMM via the configured split-K GEMV or MMA-CPasync kernel.

    Selects between `gemm_mma_cpasync` (for `GEMM_MMA_CPASYNC` kernel kind) and
    `gemv_split_k` (otherwise) based on `config.kernel_kind`, then enqueues the
    chosen kernel with the runtime M, N, K derived from the input tiles.

    Parameters:
        config: Tuning config selecting the kernel kind
            (`GEMM_MMA_CPASYNC` or split-K GEMV) and the tile shapes, thread
            count, and unroll factor for the launched kernel.
        elementwise_lambda_fn: Optional epilogue applied to each output
            element (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`.
        ctx: Device context used to enqueue the selected kernel.
    """
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2

    comptime if config.kernel_kind == GEMVAlgorithm.GEMM_MMA_CPASYNC:
        var m = Int(c.dim[0]())
        comptime static_K = a.static_shape[1]
        comptime static_N = c.static_shape[1]
        gemm_mma_cpasync[
            pdl_level=pdl_level,
            tile_k=config.tile_k,
            elementwise_lambda_fn=elementwise_lambda_fn,
            swapAB=config.swapAB,
        ](
            c,
            a,
            b,
            m,
            static_K,
            static_N,
            1,
            ctx,
        )
    else:
        comptime c_type = c.dtype
        comptime a_type = a.dtype
        comptime b_type = b.dtype
        comptime simd_width = simd_width_of[a_type, target=get_gpu_target()]()
        comptime static_N = c.static_shape[1]
        # m is only known at runtime, so the grid can overshoot the final
        # rows whenever tile_m > 1 (m % tile_m != 0); the row guard must
        # then be on. The column guard is comptime-decidable from static N.
        comptime check_bounds_m = config.tile_m > 1
        comptime check_bounds_n = static_N % config.tile_n != 0

        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())

        comptime c_layout = type_of(c).LayoutType
        comptime a_layout = type_of(a).LayoutType
        comptime b_layout = type_of(b).LayoutType

        comptime kernel = gemv_split_k[
            c_type,
            a_type,
            b_type,
            c_layout,
            a_layout,
            b_layout,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(b).Engine,
            simd_width=simd_width,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_threads=config.num_threads,
            unroll_factor=config.unroll_factor,
            elementwise_lambda_fn=elementwise_lambda_fn,
            check_bounds_m=check_bounds_m,
            check_bounds_n=check_bounds_n,
        ]

        ctx.enqueue_function[kernel](
            c,
            a.as_immut(),
            b.as_immut(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, config.tile_m), ceildiv(n, config.tile_n)),
            block_dim=config.num_threads,
            attributes=pdl_launch_attributes(pdl_level),
        )


@always_inline
def dispatch_gemv[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    """Dispatch M=1 (or N=1) matmul to GEMV or SM100 GEMM based on (N, K).

    For most M=1 shapes GEMV is preferred, but for certain large (N, K)
    combinations the SM100 GEMM kernel achieves higher throughput. Add new
    (N, K) pairs to `SM100_GEMV_SHAPES` as they are identified through benchmarking.

    N=1 always routes to GEMV: SM100 TMA requires N * sizeof(c_type) % 16 == 0.

    Parameters:
        c_type: Output element type (inferred).
        a_type: Element type of the LHS operand `a` (inferred).
        b_type: Element type of the RHS operand `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to
            `False`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element, passed to the SM100 GEMM path (defaults to `None`).
        elementwise_lambda_wrapper: Optional epilogue lambda passed to
            the GEMV path, folding in the compute lambda
            (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale, passed to the SM100 GEMM path
            (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
    """
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    comptime static_NK = Index(static_N, static_K)

    # (N, K) shapes where SM100 GEMM outperforms GEMV kernel.
    comptime SM100_GEMV_SHAPES = [
        Index(12288, 1536),
        Index(7168, 8192),
        Index(7168, 21504),
        Index(7168, 18432),
    ]

    comptime if static_NK in SM100_GEMV_SHAPES:
        var status = sm100_heuristic_and_outliers_dispatch[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

        if status:
            logger.info("------ Executing SM100 GEMV kernel ------")
            return

    logger.info("------ Executing GEMV Matmul------")
    gemv_gpu[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_wrapper,
        pdl_level=pdl_level,
    ](c, a, b, ctx)


@always_inline
def matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    use_tf32: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    """Dispatches a 2D matmul to the appropriate SM100 (B200+) kernel.

    Routes the problem to GEMV for M=1 or N=1 shapes, to the IEEE-fp32 split-K
    GEMV for precise float32, or to the dtype-specific SM100 dispatcher (bf16,
    fp8, fp32) for general shapes, falling back to vendor BLAS when no Mojo
    SM100 config applies. In autotuning mode, launches a single
    compile-time-configured kernel from environment defines.

    Parameters:
        c_type: Output element type.
        a_type: Element type of the LHS operand `a`.
        b_type: Element type of the RHS operand `b`.
        transpose_b: Whether `b` is stored transposed (defaults to
            `False`).
        use_tf32: Whether to allow TF32 (truncated mantissa) multiplies
            for float32 instead of requiring IEEE-fp32 precision
            (defaults to `True`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element, passed to the SM100 GEMM path (defaults to `None`).
        elementwise_lambda_wrapper: Optional epilogue lambda for GEMV and
            vendor fallback paths, folding in the compute lambda
            (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale, passed to the SM100 GEMM path
            (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime assert a_type == b_type, "a_type and b_type must be the same"

    var m = Int(c.dim[0]())
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    # When N is dynamic, static_N will be set to -1.
    comptime has_static_N = static_N > -1

    comptime if get_defined_bool["AUTOTUNING_MODE", False]():
        comptime BM = get_defined_int["TUNE_BM", 128]()
        comptime BN = get_defined_int["TUNE_BN", 64]()
        comptime BK = (
            TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]()
        )
        comptime MMA_K = 32 // size_of[a_type]()
        comptime CLUSTER_DIM_X = get_defined_int["TUNE_CLUSTER_DIM_X", 2]()
        comptime CLUSTER_DIM_Y = get_defined_int["TUNE_CLUSTER_DIM_Y", 1]()
        comptime CLUSTER_DIM_Z = get_defined_int["TUNE_CLUSTER_DIM_Z", 1]()
        comptime CLUSTER_DIM = Index(
            CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z
        )
        comptime BLOCK_SWIZZLE_SIZE = get_defined_int[
            "TUNE_BLOCK_SWIZZLE_SIZE", 0
        ]()
        comptime RASTERIZE_ORDER = get_defined_int["TUNE_RASTER_ORDER", 1]()
        comptime CTA_GROUP = get_defined_int["TUNE_CTA_GROUP", 2]()
        comptime K_GROUP_SIZE = get_defined_int["TUNE_K_GROUP_SIZE", 1]()
        comptime AB_SWAPPED = get_defined_bool["TUNE_AB_SWAPPED", False]()

        comptime umma_shape = Index(BM * CTA_GROUP, BN * CTA_GROUP, MMA_K)

        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            mma_shape=umma_shape,
            cluster_shape=CLUSTER_DIM,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            raster_order=RasterOrder(Int32(RASTERIZE_ORDER)),
            cta_group=CTA_GROUP,
            AB_swapped=AB_SWAPPED,
            k_group_size=K_GROUP_SIZE,
            use_tma_epilogue_load=False,
        )

        return blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
        ](c, a, b, ctx)

    # M = 1(or N = 1) : dispatch to GEMV or SM100 based on(N, K).
    # For certain large(N, K) shapes SM100 GEMM outperforms GEMV even at M = 1.
    comptime if a_type in (DType.bfloat16, DType.float8_e4m3fn):
        if static_N == 1 or m == 1:
            dispatch_gemv[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_lambda_wrapper=elementwise_lambda_wrapper,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return

    # Tiny-/mid-M, small-N FP32 GEMM (e.g. the decode router/gate GEMM:
    # M<=64, N=128, K=6144, transpose_b). The SM100 tile GEMM launches only
    # ~2 CTAs for tiny M, leaving HBM and the MMA units almost idle; the
    # split-K GEMV (warps along K, all-K-in-block) instead streams the N*K
    # weight across many CTAs.
    #
    # M-adaptive tile_m: each GEMV block processes tile_m output rows, reusing
    # the weight tile across them. This cuts the redundant L2 weight re-reads
    # (~ceildiv(M, tile_m) * N*K) at the cost of a larger per-thread
    # [tile_m, tile_n] accumulator and warp-reduce tail, so the optimum is
    # M-dependent and non-monotone (grid quantization). The bucket boundaries
    # below are the swept winners on B200; crossover to the tile GEMM is M=64
    # (its tensor-core weight reuse wins for larger M). KERN-3076.
    #
    # Gate is conservative so FP32 shapes the tile GEMM serves better are not
    # diverted: static_N<=256 (weight dominates; wider N favors MMA) and
    # static_K>=2048 (enough K to hide the many-CTA launch). A fused epilogue
    # rides through as elementwise_lambda_wrapper (which already folds in any
    # compute lambda) and is applied per output element by the GEMV.
    comptime has_precise_f32_gemv = (
        a_type == .float32
        and c_type == .float32
        and transpose_b
        and has_static_N
        and static_N <= 256
        and static_K >= 2048
        and static_K % simd_width_of[a_type, target=get_gpu_target()]() == 0
    )

    # use_tf32=False promises IEEE-fp32 multiplies, which the SM100 tensor
    # core cannot deliver (tcgen05 has no fp32 UMMA kind) — the split-K GEMV
    # is the only fp32-precise path, so the shape must satisfy its gate.
    comptime assert use_tf32 or a_type != .float32 or has_precise_f32_gemv, (
        "use_tf32=False requires the IEEE-fp32 split-K GEMV: an fp32"
        " transpose_b matmul with static N <= 256 and static K >= 2048 (K a"
        " multiple of the fp32 simd width); this shape has no fp32-precise"
        " SM100 path"
    )

    comptime if has_precise_f32_gemv:
        # tile_m is a comptime kernel param, so each bucket instantiates a
        # distinct gemv_split_k; the runtime `m` selects the bucket.
        @__parameter
        def _dispatch_split_k[tile_m: Int]() raises:
            gemv_gpu_dispatch[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                pdl_level=pdl_level,
                tile_m=tile_m,
            ](GEMVAlgorithm.GEMV_SPLIT_K, c, a, b, ctx)

        if m <= 6:
            _dispatch_split_k[1]()
            return
        elif m <= 12:
            _dispatch_split_k[2]()
            return
        # m > 64 normally crosses over to the UMMA tile GEMM, which truncates
        # fp32 operands to TF32's 10-bit mantissa (accumulation stays fp32).
        # use_tf32=False keeps every M on this IEEE-fp32 GEMV instead
        # (KERN-3151), giving up tensor-core weight reuse at large M.
        elif m <= 64 or not use_tf32:
            _dispatch_split_k[4]()
            return

    # C's row stride is not 16-byte aligned, so no TMA descriptor can
    # describe C.
    comptime has_unaligned_n_alt_dispatch = (
        a_type in (DType.bfloat16, DType.float8_e4m3fn)
        and c_type == .bfloat16
        and transpose_b
        and has_static_N
        and static_N * size_of[c_type]() % 16 != 0
    )

    comptime if has_unaligned_n_alt_dispatch:
        comptime has_split_k_band = (
            static_N <= 642
            and static_K >= 128
            and static_K % simd_width_of[a_type, target=get_gpu_target()]() == 0
        )

        comptime if has_split_k_band:

            @__parameter
            def _dispatch_unaligned_n_split_k[tile_m: Int]() raises:
                logger.info(
                    (
                        "------ Dispatching to SM100 unaligned-N split-K GEMV"
                        " (tile_m="
                    ),
                    tile_m,
                    ") ------ Problem Shape: MNK=[",
                    m,
                    ", ",
                    static_N,
                    ", ",
                    static_K,
                    "]",
                )
                gemv_gpu_dispatch[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                    pdl_level=pdl_level,
                    tile_m=tile_m,
                ](GEMVAlgorithm.GEMV_SPLIT_K, c, a, b, ctx)

            if m <= 6:
                _dispatch_unaligned_n_split_k[1]()
                return
            elif m <= 12:
                _dispatch_unaligned_n_split_k[2]()
                return
            elif m <= 64:
                _dispatch_unaligned_n_split_k[4]()
                return

    comptime if _vendor_blas_fallback_disabled():
        comptime if (
            (
                (
                    a_type == .bfloat16
                    and c_type in (DType.bfloat16, DType.float8_e4m3fn)
                )
                or (a_type == .float8_e4m3fn and c_type in (DType.bfloat16,))
                or (a_type == .float32 and c_type in (DType.float32,))
            )
            and static_N * size_of[c_type]() % 16 == 0
            and static_K * size_of[a_type]() % 16 == 0
            and transpose_b
        ):
            var status = sm100_heuristic_and_outliers_dispatch[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            if status:
                return
            else:
                raise Error(
                    "Heuristic failed to find a config for this (N,K) or m"
                )

    var epilogue_type = String("None")

    comptime if elementwise_compute_lambda_fn:
        epilogue_type = String("Compute Epilogue")
    elif elementwise_lambda_fn:
        epilogue_type = String("Normal Epilogue")

    logger.info("------ Dispatching to SM100 (B200+) ------")
    logger.info(
        "Input Data Types: ",
        a_type,
        ", ",
        b_type,
        " Output Data Type: ",
        c_type,
        " Problem Shape: MNK=[",
        m,
        ", ",
        static_N,
        ", ",
        static_K,
        "]",
        " Epilogue Type: ",
        epilogue_type,
    )

    # Default matmul config for SM100.
    comptime MMA_K = 32 // size_of[a_type]()
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    # SM100 kernel requirements:
    # 1. `N * size_of(c_type) % 16B == 0` for output buffer (TMA requirement).
    # 2. Supported output dtypes: bfloat16, float8_e4m3fn, and float32.
    #    float32 input only supports float32 output.
    #    float8_e4m3fn input only supports bfloat16 output.
    comptime if (
        static_N * size_of[c_type]() % 16 == 0
        and static_K * size_of[a_type]() % 16 == 0
        and transpose_b
    ):
        var status = DISPATCH_MISS

        comptime if a_type == .bfloat16 and c_type in (
            DType.bfloat16,
            DType.float8_e4m3fn,
        ):
            status = matmul_dispatch_sm100_bf16[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_lambda_wrapper=elementwise_lambda_wrapper,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        elif a_type == .float8_e4m3fn and c_type in (DType.bfloat16,):
            status = matmul_dispatch_sm100_fp8[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        elif a_type == .float32 and c_type in (DType.float32,):
            status = matmul_dispatch_sm100_fp32[
                c_type=c_type,
                a_type=a_type,
                b_type=b_type,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        if status:
            logger.info("------ Executing MOJO SM100 Matmul------")
            return

    # Fallback to vendor matmul for untuned shapes.
    # We assume this is always a hit because the worst case is a naive matmul.
    return _vendor_blas_matmul_sm100[
        c_type,
        a_type,
        b_type,
        transpose_b,
        elementwise_lambda_wrapper=elementwise_lambda_wrapper,
    ](c, a, b, ctx)


@always_inline
# NOTE:
# 1. SM100 matmul supports compute lambdas, so we should use normal and
#    compute lambdas.
def matmul_dispatch_sm100_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an FP8-input SM100 matmul to a tuned or heuristic config.

    For M <= 128, tries the heuristic outlier dispatch first and falls back to
    the default SM100 config on a miss. For larger M, searches the FP8 tuning
    table by static M bucket, then falls through to the heuristic outlier
    dispatch for untuned (N, K) shapes. Only bfloat16 output is supported.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    comptime assert c_type in (DType.bfloat16,), "Only support bfloat16 output"

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
    var m = Int(c.dim[0]())

    if m <= 128:
        var status = heuristic_and_outliers_dispatch[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)
        if status:
            return status

        # Untuned small-M (N, K): unlike the fp8-OUTPUT case,
        # `select_and_launch_sm100_config` has no never-miss for bf16 output, so
        # it DISPATCH_MISSes when `choose_config` yields a config absent from the
        # sampled config set. That happens for the small-M decode band
        # (Nemotron c=32, m in {25..31}: `choose_config` picks mma_n=16/cta=1,
        # which no build_sm100_matmul_configs grid sample -- stepped by 8, with m
        # passed exact -- ever produces), which would fall back to vendor
        # cuBLASLt. Mirror the fp8-output never-miss above: launch the guaranteed
        # -valid default SM100 config on MAX's own tcgen05 Mojo FP8 kernel. The
        # static-scale compute epilogue rides through as
        # `elementwise_compute_lambda_fn`.
        comptime default_config = default_matmul_config_bf16_fp8[
            a_type, b_type, c_type, transpose_b
        ]()
        _matmul_dispatch_sm100[
            transpose_b=transpose_b,
            config=default_config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)
        return DISPATCH_HIT

    @__parameter
    @always_inline("nodebug")
    def _dispatch[entry: TuningConfigSM100]() raises:
        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            block_swizzle_size=entry.block_swizzle_size,
        )

        return _matmul_dispatch_sm100[
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    @__parameter
    @always_inline("nodebug")
    def _search[
        T: Table[TuningConfigSM100],
        domain: List[Int] = List[Int](),
    ]() raises -> Int:
        comptime m_values = T.query_values[Int, domain=domain](
            rule=lambda (x: TuningConfigSM100) -> Int: x.M
        )

        comptime for static_m in m_values:
            if m <= static_m:
                comptime idx_list = T.query_index[domain=domain](
                    rule=lambda (x: TuningConfigSM100) -> Bool: x.M == static_m
                )

                comptime if idx_list:
                    comptime entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # Dynamic m is in range but no corresponding config exists
                    # in the table.
                    break

        return DISPATCH_MISS

    comptime tuning_list = _get_tuning_list_sm100_fp8[mma_k=MMA_K, bk=BK]()
    comptime tuning_table = Table(tuning_list, "tuning_table_sm100_fp8")

    comptime nk_idx_list = tuning_table.query_index(
        rule=lambda (x: TuningConfigSM100) -> Bool: x.K == static_K
        and x.N == static_N
    )

    # TODO: Re-enable the following tuning dispatch.
    # Make sure `domain(nk_idx_list)` is not empty.
    if m > 128:
        comptime if nk_idx_list:
            if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
                return DISPATCH_HIT

    # TODO(KERN-2084): Enable default matmul for large shapes to increase
    # accuracy.
    # #fallback to default matmul for large shapes
    # alias block_tile_shape = Index(128, 128, BK)
    # alias umma_shape = Index(
    # block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
    # )
    # alias cluster_shape = Index(2, 1, 1)
    # alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
    # block_tile_shape = block_tile_shape,
    # mma_shape = umma_shape,
    # cluster_shape = cluster_shape,
    # )
    # _matmul_dispatch_sm100[
    # transpose_b = transpose_b,
    # config = config,
    # elementwise_lambda_fn = elementwise_lambda_fn,
    # elementwise_compute_lambda_fn = elementwise_compute_lambda_fn,
    # pdl_level = pdl_level,
    # block_swizzle_size = 0,
    # ](c, a, b, ctx)
    # return DISPATCH_HIT

    # Untuned (N, K): fall through to the existing heuristic config-set
    # dispatch (the same tail the bf16 dispatcher uses at
    # `matmul_dispatch_sm100_bf16`) instead of DISPATCH_MISSing to vendor
    # cuBLASLt. `choose_config` + `build_sm100_matmul_configs` cover every
    # prefill m on MAX's own tcgen05 Mojo FP8 kernel (verified host-side:
    # 0 miss over m in [129, 8192] for the served FP8 (N, K) shapes), so this
    # keeps FP8 prefill on the Mojo kernel rather than the closed vendor BLAS.
    # The static-scale compute epilogue rides through as
    # `elementwise_compute_lambda_fn`.
    return sm100_heuristic_and_outliers_dispatch[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](c, a, b, ctx)


def _sm100_outlier_configs[
    a_type: DType, mma_k: Int, bk: Int, static_N: Int, static_K: Int
]() -> List[TuningConfigSM100]:
    """Per-dtype heuristic outlier tuning list, filtered to this (N, K).

    Uses a comptime branch (not a ternary) so the fp8 list -- which bakes
    `mma_k` into its tile shapes -- is never instantiated for bf16/fp32.
    """

    @always_inline
    def rule(x: TuningConfigSM100) {} -> Bool:
        return x.K == static_K and x.N == static_N

    comptime if a_type == .bfloat16:
        return Table(
            _get_tuning_list_sm100_bf16(), "bf16_heuristic_outliers"
        ).find(rule=rule)
    elif a_type == .float32:
        return Table(
            _get_tuning_list_sm100_fp32(), "fp32_heuristic_outliers"
        ).find(rule=rule)
    else:
        return Table(
            _get_tuning_list_sm100_fp8[mma_k, bk](), "fp8_heuristic_outliers"
        ).find(rule=rule)


def select_and_launch_sm100_config[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    launch_type: def[config: MatmulConfig[...]](
        TileTensor[mut=True, c_type, ...],
        TileTensor[a_type, ...],
        TileTensor[b_type, ...],
        DeviceContext,
        OptionalReg[
            TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
        ],
    ) raises -> None,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
](
    launch: launch_type,
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
    ] = None,
) raises -> Int:
    """Selects and launches an SM100 matmul config for the given shape.

    Checks the per-dtype outlier tuning list for a matching (N, K, M) config,
    then falls back to the heuristic config set from `build_sm100_matmul_configs`
    chosen via `choose_config`. For float8_e4m3fn output, always launches the
    default config on a miss; other dtypes return `DISPATCH_MISS`.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    var m = Int(c.dim[0]())
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    comptime assert a_type == b_type and a_type in (
        DType.bfloat16,
        DType.float8_e4m3fn,
        DType.float32,
    ), "Only support bfloat16, float8_e4m3fn, and float32 input types"
    comptime assert (
        a_type != .float32 or c_type == .float32
    ), "float32 input only supports float32 output"

    comptime MMA_K = 32 // size_of[a_type]()
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime outlier_configs = _sm100_outlier_configs[
        a_type, MMA_K, BK, static_N, static_K
    ]()

    # do not use outliers list when c_type is FP8 as we don't support all tile shapes dude to TMA requirements
    comptime if c_type != .float8_e4m3fn:
        comptime for tuning_config in outlier_configs:
            if m >= tuning_config.M and m < tuning_config.M_end:
                comptime matmul_config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    mma_shape=tuning_config.mma_shape,
                    cta_group=tuning_config.cta_group,
                    cluster_shape=tuning_config.cluster_shape,
                    block_swizzle_size=tuning_config.block_swizzle_size,
                    raster_order=tuning_config.rasterize_order,
                    AB_swapped=tuning_config.swapAB,
                    num_accum_pipeline_stages=tuning_config.num_accum_pipeline_stages,
                    num_clc_pipeline_stages=tuning_config.num_clc_pipeline_stages,
                    k_group_size=tuning_config.k_group_size,
                    num_split_k=tuning_config.num_split_k,
                    use_tma_epilogue_load=has_epilogue_tensor,
                    epilogue_is_1d=epilogue_is_1d,
                )

                logger.info("dispatching to outlier config: ", matmul_config)

                launch[matmul_config](c, a, b, ctx, epilogue_tensor)
                return DISPATCH_HIT

    comptime configs = build_sm100_matmul_configs[
        a_type,
        b_type,
        c_type,
        static_N,
        static_K,
        transpose_b,
        has_epilogue_tensor=has_epilogue_tensor,
        epilogue_is_1d=epilogue_is_1d,
    ]()
    var aligned_m = align_up(m, 64) if m >= 256 else m
    var config_runtime = choose_config[
        a_type,
        b_type,
        c_type,
        transpose_b,
        has_epilogue_tensor=has_epilogue_tensor,
        epilogue_is_1d=epilogue_is_1d,
    ](aligned_m, static_N, static_K, 1)

    comptime for config in configs:
        if config_runtime == config:
            logger.info("dispatching to config: ", config)

            launch[config](c, a, b, ctx, epilogue_tensor)
            return DISPATCH_HIT

    # For float8_e4m3fn output, dispatch should never fail; use the
    # default config.
    comptime if c_type == .float8_e4m3fn:
        comptime default_config = default_matmul_config_bf16_fp8[
            a_type,
            b_type,
            c_type,
            transpose_b,
            has_epilogue_tensor=has_epilogue_tensor,
        ]()
        launch[default_config](c, a, b, ctx, epilogue_tensor)
        return DISPATCH_HIT

    return DISPATCH_MISS


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[c.dtype, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
    ] = None,
) raises -> Int:
    """Dispatches an SM100 matmul through the heuristic outlier config set.

    Wraps `select_and_launch_sm100_config` with a launch callback that invokes
    `_matmul_dispatch_sm100` (the epilogue-aware SM100 tile GEMM launcher) for
    each selected config.

    Parameters:
        c_type: Output element type (inferred).
        a_type: Element type of the LHS operand `a` (inferred).
        b_type: Element type of the RHS operand `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
        has_epilogue_tensor: Whether an epilogue tensor is supplied for
            the TMA epilogue load path (defaults to `False`).
        epilogue_is_1d: Whether the epilogue tensor is treated as
            1D rather than row-major 2D (defaults to `False`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
        epilogue_tensor: Optional row-major epilogue tensor of the
            same dtype as `c`, consumed by the TMA epilogue load path
            (defaults to `None`).
    """

    @always_inline
    def launch_callback[
        config: MatmulConfig[...]
    ](
        c_tensor: TileTensor[mut=True, c_type, ...],
        a_tensor: TileTensor[a_type, ...],
        b_tensor: TileTensor[b_type, ...],
        dispatch_ctx: DeviceContext,
        dispatch_epilogue_tensor: OptionalReg[
            TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
        ],
    ) raises:
        _matmul_dispatch_sm100[
            transpose_b,
            rebind[MatmulConfig[a_type, b_type, c_type, transpose_b]](config),
            elementwise_lambda_fn,
            elementwise_compute_lambda_fn,
            pdl_level,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            dispatch_ctx,
            epilogue_tensor=dispatch_epilogue_tensor,
        )

    return select_and_launch_sm100_config[
        transpose_b,
        elementwise_lambda_fn,
        elementwise_compute_lambda_fn,
        pdl_level,
        has_epilogue_tensor=has_epilogue_tensor,
        epilogue_is_1d=epilogue_is_1d,
    ](launch_callback, c, a, b, ctx, epilogue_tensor)


# NOTE:
# 1. SM100 matmul supports compute lambdas, so we should use normal and
#    compute lambdas.
def matmul_dispatch_sm100_bf16[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches a bfloat16-input SM100 matmul to a tuned or heuristic config.

    Routes known low-performance shapes to vendor BLAS, tries the small-MN GEMM
    tuning table for matched (N, K), then falls back to the heuristic outlier
    dispatch and finally the default SM100 config on a miss.

    Parameters:
        c_type: Output element type (inferred).
        a_type: Element type of the LHS operand `a` (inferred).
        b_type: Element type of the RHS operand `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element, passed to the SM100 GEMM path (defaults to `None`).
        elementwise_lambda_wrapper: Optional epilogue lambda for vendor
            BLAS and small-MN GEMM paths, folding in the compute lambda
            (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale, passed to the SM100 GEMM path
            (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"

    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    comptime MMA_K = 16
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime low_perf_shapes = [
        Index(2112, 14336),
    ]

    # fallback to vendor matmul for shapes that Mojo kernel is lagging behind
    comptime if (static_N, static_K) in low_perf_shapes and c_type in (
        DType.bfloat16,
    ):
        _vendor_blas_matmul_sm100[
            c_type,
            a_type,
            b_type,
            transpose_b,
            elementwise_lambda_wrapper=elementwise_lambda_wrapper,
        ](c, a, b, ctx)
        return DISPATCH_HIT

    comptime small_MN_gemms_table = Table(
        _get_tuning_list_small_MN_gemms_bf16(), "small_MN_gemms_configs"
    )

    comptime small_MN_gemms_configs = small_MN_gemms_table.find(
        rule=lambda (x: TuningConfigSmallMNGemms) -> Bool: x.K == static_K
        and x.N == static_N
    )

    comptime if small_MN_gemms_configs and c_type in (DType.bfloat16,):
        var m = Int(c.dim[0]())
        comptime for config in small_MN_gemms_configs:
            if m >= config.M and m < config.M_end:
                logger.info("Dispatching to small_MN_gemms: ", config)
                small_MN_gemms[
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return DISPATCH_HIT

    var status = sm100_heuristic_and_outliers_dispatch[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](c, a, b, ctx)
    if status:
        return status

    # Untuned small-M (N, K): `select_and_launch_sm100_config`'s never-miss is
    # fp8-OUTPUT-only (config.mojo, `c_type == float8_e4m3fn` guard), so for
    # bf16 output it DISPATCH_MISSes when `choose_config` yields a config absent
    # from the sampled set. That happens for the small-M decode band (m in
    # {25..31}: `choose_config` picks mma_n=16/cta=1, which no
    # build_sm100_matmul_configs grid sample -- stepped by 8, with m passed
    # exact -- ever produces), which would fall back to vendor cuBLASLt. Mirror
    # the fp8-band fix in `matmul_dispatch_sm100_fp8`: launch the guaranteed
    # -valid default SM100 config on MAX's own tcgen05 Mojo kernel.
    comptime default_config = default_matmul_config_bf16_fp8[
        a_type, b_type, c_type, transpose_b
    ]()
    _matmul_dispatch_sm100[
        transpose_b=transpose_b,
        config=default_config,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](c, a, b, ctx)
    return DISPATCH_HIT


def matmul_dispatch_sm100_fp32[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches a float32 SM100 matmul via the heuristic outlier dispatch.

    Delegates directly to `sm100_heuristic_and_outliers_dispatch`; only float32
    input and output are supported.

    Parameters:
        c_type: Output element type (inferred).
        a_type: Element type of the LHS operand `a` (inferred).
        b_type: Element type of the RHS operand `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime assert (
        a_type == b_type == .float32 and c_type == .float32
    ), "matmul_dispatch_sm100_fp32 only supports float32 input and output"

    return sm100_heuristic_and_outliers_dispatch[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](c, a, b, ctx)


# NOTE: Vendor BLAS, naive matmul, and multistage GEMM do not support compute
# lambdas, so we wrap them in a lambda function.
# If there is no compute lambda, this wrapper is a simple elementwise lambda.
@always_inline
def _vendor_blas_matmul_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_wrapper: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime K = a.static_shape[1]

    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    try:
        logger.info("Executing vendor BLAS (cuBLAS/cublasLt)")
        return matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](c, a, b, ctx)

    except:
        # Fallback to multistage/naive GEMMs if cuBLAS fails.
        # This is a temporary workaround for KERN-1812.
        logger.warning("Vendor BLAS failed")

        comptime if not a_type.is_float8() and K * size_of[a_type]() >= 8 * 16:
            logger.info("Executing Multistage matmul kernel")
            comptime kernels = MatmulKernels[
                a_type, b_type, c_type, transpose_b
            ]()
            comptime config = kernels.ampere_256x64_4
            multistage_gemm[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ](c, a, b, config, ctx)
        else:
            comptime BLOCK_DIM = 16
            logger.info("Executing Naive matmul kernel")

            comptime kernel = matmul_kernel_naive[
                c_type,
                a_type,
                b_type,
                type_of(c).LayoutType,
                type_of(a).LayoutType,
                type_of(b).LayoutType,
                BLOCK_DIM,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ]

            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )
        return


def _matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c_tensor: TileTensor[mut=True, c_type, ...],
    a_tensor: TileTensor[a_type, ...],
    b_tensor: TileTensor[b_type, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
    ] = None,
) raises:
    _matmul_dispatch_sm100[
        transpose_b=transpose_b,
        config=config,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](
        NullableTileTensor(c_tensor),
        a_tensor,
        b_tensor,
        ctx,
        epilogue_tensor=epilogue_tensor,
    )


def _matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c_tensor: NullableTileTensor[mut=True, c_type, ...],
    a_tensor: TileTensor[a_type, ...],
    b_tensor: TileTensor[b_type, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
    ] = None,
) raises:
    """Our sm100 matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations if there is any.
    """

    comptime assert (
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None
    ), "Either the epilogue lambda or the compute lambda can be used"

    comptime if not elementwise_lambda_fn:
        if not c_tensor.ptr:
            raise "c must be allocated!"

        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](
            c_tensor.value(),
            a_tensor,
            b_tensor,
            ctx,
            epilogue_tensor=epilogue_tensor,
        )
        return

    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load / store to global memory, see KERN - 2037.
        comptime use_32b_simd = (
            has_nvidia_gpu_accelerator()
            and ctx.default_device_info.compute >= B200.compute
        )
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        # If c is already allocated, we can just use the sm100 matmul and
        # apply the epilogue.
        if c_tensor.ptr:
            var m = Int(c_tensor.dim[0]())
            var n = Int(c_tensor.dim[1]())
            var c_tt = c_tensor.value()

            def epilogue_wrapper[
                simd_width: Int, alignment: Int = 1
            ](idx: Coord) {var}:
                comptime assert c_tt.flat_rank >= 2
                var c_val = c_tt.load[
                    width=simd_width,
                    # load_alignment is in bytes, lambda alignment is in elements
                    alignment=alignment * size_of[c_type](),
                ](idx)
                epilogue[c_type, simd_width, alignment=alignment](
                    IndexList[2](Int(idx[0].value()), Int(idx[1].value())),
                    c_val,
                )

            blackwell_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](
                c_tt,
                a_tensor,
                b_tensor,
                ctx,
                epilogue_tensor=epilogue_tensor,
            )

            elementwise[simd_size, target="gpu"](epilogue_wrapper, (m, n), ctx)
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](
            c_tensor.num_elements()
        )

        var c_tmp = TileTensor(tmp_device_buffer, c_tensor.layout)

        _matmul_dispatch_sm100[
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c_tmp, a_tensor, b_tensor, ctx)

        _ = tmp_device_buffer^


def _sm100_batched_outlier_configs[
    a_type: DType, static_N: Int, static_K: Int
]() -> List[TuningConfigSM100]:
    """Per-dtype batched heuristic outlier tuning list, filtered to this (N, K).

    Mirrors `_sm100_outlier_configs` for the batched matmul path so future
    hand-tuned fp32 batched configs added to `_get_tuning_list_sm100_batched_fp32`
    are picked up automatically. Uses a comptime branch (not a ternary) so each
    dtype's list is only instantiated for its own dtype.
    """

    @always_inline
    def rule(x: TuningConfigSM100) {} -> Bool:
        return x.K == static_K and x.N == static_N

    comptime if a_type == .bfloat16:
        return Table(
            _get_tuning_list_sm100_batched_bf16(),
            "batched_bf16_heuristic_outliers",
        ).find(rule=rule)
    elif a_type == .float32:
        return Table(
            _get_tuning_list_sm100_batched_fp32(),
            "batched_fp32_heuristic_outliers",
        ).find(rule=rule)
    else:
        return Table(
            _get_tuning_list_sm100_batched_fp8(),
            "batched_fp8_heuristic_outliers",
        ).find(rule=rule)


@always_inline
def dispatch_sm100_batched_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    pdl_level: PDLLevel = PDLLevel.OFF,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    ctx: DeviceContext,
) raises:
    """Dispatch batched matmul to SM100 kernel.

    First, try to dispatch to a batched matmul config from the tuning table. Then try to find a optimized config for the given shape.
    If not found, then dispatch to a default config.

    Parameters:
        c_type: Element type of the output tensor `c`.
        a_type: Element type of the LHS operand `a`.
        b_type: Element type of the RHS operand `b`.
        transpose_b: Whether `b` is stored transposed.
        pdl_level: Programmatic dependent launch level for the dispatched
            kernel (defaults to `PDLLevel.OFF`).
    Args:
        c: Output batched matrix as a rank-3 mutable `TileTensor` of
            shape `[batch, M, N]`.
        a: LHS input batched matrix as a rank-3 immutable `TileTensor`
            of shape `[batch, M, K]`.
        b: RHS input batched matrix as a rank-3 immutable `TileTensor`
            of shape `[batch, K, N]`, or `[batch, N, K]` when
            `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
    """

    comptime MMA_K = 32 // size_of[a_type]()
    comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]()

    var batch_size = Int(c.dim(0))
    var m = Int(c.dim(1))
    comptime static_K = a.LayoutType._shape_types[2].static_value
    comptime static_N = c.LayoutType._shape_types[2].static_value

    comptime static_NK = Index(static_N, static_K)

    logger.info(
        "Dispatching to SM100 Batched Matmul B= ",
        batch_size,
        " M= ",
        m,
        " N= ",
        static_N,
        " K= ",
        static_K,
    )

    comptime outlier_configs = _sm100_batched_outlier_configs[
        a_type, static_N, static_K
    ]()

    comptime if c_type in (DType.bfloat16, DType.float32):
        comptime for tuning_config in outlier_configs:
            if (
                batch_size == tuning_config.batch_size
                and m >= tuning_config.M
                and m < tuning_config.M_end
            ):
                comptime matmul_config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    mma_shape=tuning_config.mma_shape,
                    cta_group=tuning_config.cta_group,
                    cluster_shape=tuning_config.cluster_shape,
                    block_swizzle_size=tuning_config.block_swizzle_size,
                    raster_order=tuning_config.rasterize_order,
                    AB_swapped=tuning_config.swapAB,
                    num_accum_pipeline_stages=tuning_config.num_accum_pipeline_stages,
                    num_clc_pipeline_stages=tuning_config.num_clc_pipeline_stages,
                    k_group_size=tuning_config.k_group_size,
                    num_split_k=tuning_config.num_split_k,
                    gemm_kind=GEMMKind.BMM,
                )

                logger.info("Using batched tuning config: ", matmul_config)

                blackwell_batched_matmul_tma_umma_warp_specialized[
                    transpose_b=transpose_b,
                    config=matmul_config,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)

                return

    comptime configs = build_sm100_batched_matmul_configs[
        a_type, b_type, c_type, static_N, static_K, transpose_b
    ]()
    var aligned_m = align_up(m, 64) if m >= 256 else m
    var config_runtime = choose_config[
        a_type, b_type, c_type, transpose_b, gemm_kind=GEMMKind.BMM
    ](aligned_m, static_N, static_K, batch_size)

    comptime for config in configs:
        if config_runtime == config:
            logger.info("dispatching to batched matmul config: ", config)

            blackwell_batched_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                config=config,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return

    # fallback to default config
    comptime default_config = default_matmul_config_bf16_fp8[
        a_type,
        b_type,
        c_type,
        transpose_b,
        gemm_kind=GEMMKind.BMM,
    ]()

    blackwell_batched_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=default_config,
        pdl_level=pdl_level,
    ](c, a, b, ctx)


def sm100_heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[c.dtype, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
    ] = None,
) raises -> Int:
    """Dispatches an SM100 matmul through the heuristic outlier config set.

    Wraps `select_and_launch_sm100_config` with a launch callback that invokes
    `blackwell_matmul_tma_umaa_warp_specialized` directly, passing through the
    elementwise and compute epilogue lambdas.

    Parameters:
        c_type: Output element type (inferred).
        a_type: Element type of the LHS operand `a` (inferred).
        b_type: Element type of the RHS operand `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute epilogue lambda,
            for example a static scale (defaults to `None`).
        pdl_level: Programmatic dependent launch level for the
            dispatched kernel (defaults to `PDLLevel()`).
        has_epilogue_tensor: Whether an epilogue tensor is supplied for
            the TMA epilogue load path (defaults to `False`).
        epilogue_is_1d: Whether the epilogue tensor is treated as
            1D rather than row-major 2D (defaults to `False`).
    Args:
        c: Output matrix as a rank-2 mutable `TileTensor` of shape
            `[M, N]`.
        a: LHS input matrix as a rank-2 `TileTensor` of shape `[M, K]`.
        b: RHS input matrix as a rank-2 `TileTensor` of shape `[K, N]`,
            or `[N, K]` when `transpose_b` is set.
        ctx: Device context used to enqueue the selected kernel.
        epilogue_tensor: Optional row-major epilogue tensor of the
            same dtype as `c`, consumed by the TMA epilogue load path
            (defaults to `None`).
    """

    @always_inline
    def launch_callback[
        config: MatmulConfig[...]
    ](
        c_tensor: TileTensor[mut=True, c_type, ...],
        a_tensor: TileTensor[a_type, ...],
        b_tensor: TileTensor[b_type, ...],
        dispatch_ctx: DeviceContext,
        dispatch_epilogue_tensor: OptionalReg[
            TileTensor[c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin]
        ],
    ) raises:
        blackwell_matmul_tma_umma_warp_specialized[
            transpose_b,
            config=rebind[MatmulConfig[a_type, b_type, c_type, transpose_b]](
                config
            ),
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            dispatch_ctx,
            epilogue_tensor=dispatch_epilogue_tensor,
        )

    return select_and_launch_sm100_config[
        transpose_b,
        elementwise_lambda_fn,
        elementwise_compute_lambda_fn,
        pdl_level,
        has_epilogue_tensor=has_epilogue_tensor,
        epilogue_is_1d=epilogue_is_1d,
    ](launch_callback, c, a, b, ctx, epilogue_tensor)
