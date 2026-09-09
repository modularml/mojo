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

"""
Dispatches rank-2 matmuls to the SM90 (Hopper) warp-specialized kernel.

Routes BF16, FP32, and FP8 (e4m3fn) problems to tuned configurations based on
static N and K, falling back to a generic kernel for unsupported shapes.
"""

from std.math import ceildiv
from std.sys import get_defined_bool, get_defined_int, size_of

from layout import TileTensor
from max.gpu.primitives.grid_controls import PDLLevel
from max.gpu.host import DeviceContext
from max.gpu.host.info import H100
from internal_utils import Table
from std.logger import Logger

from std.utils.index import Index, IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig, _vendor_blas_fallback_disabled
from ..tile_scheduler import MatmulSchedule, RasterOrder
from .matmul import warp_specialize_gemm_with_multicasting
from .tuning_configs import (
    _get_tuning_list_bf16,
    TuningConfigSM90,
    TuningGroup,
)
from .config import (
    build_configs,
    build_configs_generic,
    swapAB_smallM,
    swapAB_smallM_ceildiv,
    swapAB_midM_linear,
    swapAB_largeM_clustered,
    MatmulConfig as MatmulConfigSM90,
)

comptime MAX_M = Int.MAX

# TODO: Move to a general location and use for all dispatch

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1

comptime logger = Logger()


def matmul_dispatch_sm90[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches a rank-2 matmul to the SM90 (Hopper) warp-specialized kernel.

    Checks static dtype and shape constraints (BF16, FP8, or FP32; transposed
    B; K alignment), then routes to `matmul_dispatch_sm90_fp8`,
    `matmul_dispatch_sm90_bf16_fp32`, or returns `DISPATCH_MISS` if the
    problem does not meet SM90 requirements.

    Parameters:
        c_type: Element type of the output tensor `c`.
        a_type: Element type of the input tensor `a`.
        b_type: Element type of the input tensor `b`.
        transpose_b: Whether `b` is stored transposed (defaults to `False`);
            the SM90 kernel requires this to be `True` to dispatch.
        elementwise_lambda_fn: Epilogue applied to each output tile after the
            matmul, consuming the tile and returning nothing (defaults to
            `None` for no epilogue).
        elementwise_compute_lambda_fn: Compute lambda transforming each output
            value before the epilogue, returning the transformed value
            (defaults to `None` for no transform).
        pdl_level: Programmatic dependent launch level for overlapping this
            kernel with prior work (defaults to `PDLLevel()`).

    Args:
        c: Rank-2 output tensor of shape `(M, N)`.
        a: Rank-2 input tensor of shape `(M, K)`.
        b: Rank-2 input tensor of shape `(K, N)` when `transpose_b` is `True`.
        ctx: Device context for the kernel launch.

    Returns:
        `DISPATCH_HIT` (1) if the kernel was launched, `DISPATCH_MISS` (0)
        otherwise.
    """
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime is_AB_fp8 = a_type == b_type == DType.float8_e4m3fn
    comptime is_AB_bf16 = a_type == b_type == DType.bfloat16
    comptime is_AB_fp32 = a_type == b_type == DType.float32

    comptime input_type_supported = is_AB_fp8 or is_AB_bf16 or is_AB_fp32

    # fmt: off
    comptime has_static_NK = (b.static_shape[0] > -1 and b.static_shape[1] > -1) \
                      and a.static_shape[1] > -1 \
                      and c.static_shape[1] > -1
    # fmt: on

    comptime N = c.static_shape[1]
    comptime N_multiple_of_8 = N % 8 == 0

    logger.info("------ Dispatching to sm90 ------")

    # Support K multiple of 16B for FP8 due to using TMA.
    # 4B and 8B alignments are supported for BF16/FP32 by using
    # cp.async.ca.
    comptime K = a.static_shape[1]
    comptime K_multiple_of_16B = K * size_of[a_type]() % 16 == 0
    comptime K_multiple_of_4B = K * size_of[a_type]() % 4 == 0
    comptime K_align_supported = (K_multiple_of_16B and is_AB_fp8) or (
        K_multiple_of_4B and (is_AB_bf16 or is_AB_fp32)
    )

    @always_inline
    @__parameter
    @__copy_capture(c, a, b)
    def _dispatch() raises -> Int:
        # General constraints for H100 matmul
        # fmt: off
        comptime if not (
            input_type_supported and \
            transpose_b and \
            has_static_NK and \
            K_align_supported
        ):
            return DISPATCH_MISS
        # fmt: on

        comptime if is_AB_fp8:
            logger.info("------ Dispatching to sm90 FP8 ------")
            return matmul_dispatch_sm90_fp8[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        elif is_AB_bf16 or is_AB_fp32:
            logger.info("------ Dispatching to sm90 BF16/FP32 ------")
            return matmul_dispatch_sm90_bf16_fp32[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, ctx)

        logger.info("SM90 dispatch miss - no matching path")
        return DISPATCH_MISS

    comptime if _vendor_blas_fallback_disabled():
        if _dispatch():
            return DISPATCH_HIT
        # On any miss (unsupported config or no tuning for this shape), return
        # DISPATCH_MISS so the caller can fall back to vendor BLAS or other paths.
        return DISPATCH_MISS

    return _dispatch()


# ===----------------------------------------------------------------------=== #

# FP8 (e4m3fn) Dispatch

# ===----------------------------------------------------------------------=== #

# llama-405B-FP8 gemm shapes

comptime llama_405b_fp8_list: List[TuningConfigSM90] = [
    ##############################
    # N=16384 and K=2048
    TuningConfigSM90(
        M=64,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        grid_shape=Index(128, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=128,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=256,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=512,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(H100.sm_count, 1),
        schedule=MatmulSchedule.DS_SCHEDULER,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=16384,
        K=2048,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=2304 and K=16384
    TuningConfigSM90(
        M=64,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 48, 32),
        block_tile_shape=Index(64, 48, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=128,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 48, 32),
        block_tile_shape=Index(64, 48, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=256,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 96, 32),
        block_tile_shape=Index(64, 96, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=512,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(H100.sm_count, 1),
    ),
    TuningConfigSM90(
        M=2048,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 144, 32),
        block_tile_shape=Index(128, 144, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(16, 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=2304,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=13312 and K=16384
    TuningConfigSM90(
        M=64,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(128, 1),
    ),
    TuningConfigSM90(
        M=128,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=256,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 208, 32),
        block_tile_shape=Index(128, 208, 128),
        cluster_shape=Index(1, 2, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=512,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=1024,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=13312,
        K=16384,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
    ##############################
    # N=16384 and K=6656
    TuningConfigSM90(
        M=64,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=False,
        schedule=MatmulSchedule.DS_SCHEDULER,
        grid_shape=Index(128, 1),
    ),
    TuningConfigSM90(
        M=1024,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        schedule=MatmulSchedule.NONE,
        grid_shape=None,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=16384,
        K=6656,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
]

comptime llama_405b_fp8_table = Table(llama_405b_fp8_list, "llama_405b_fp8")

# llama-8B-FP8 gemm shapes

comptime llama_8b_fp8_list: List[TuningConfigSM90] = [
    ##############################
    # ignore N and K for this table.
    TuningConfigSM90(
        M=128,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(64, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=8,
        num_consumer=1,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.NONE,
    ),
    TuningConfigSM90(
        M=1024,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=6,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=None,
        schedule=MatmulSchedule.NONE,
    ),
    TuningConfigSM90(
        M=MAX_M,
        N=-1,
        K=-1,
        mma_shape=IndexList[3](64, 128, 32),
        block_tile_shape=Index(128, 128, 128),
        cluster_shape=Index(2, 1, 1),
        num_pipeline_stages=6,
        num_consumer=2,
        partitioned_multicast=True,
        grid_shape=Index(8, H100.sm_count // 8),
        schedule=MatmulSchedule.TILE2D,
    ),
]

comptime llama_8b_fp8_table = Table(llama_8b_fp8_list, "llama_8b_fp8")


def matmul_dispatch_sm90_fp8[
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
    c: TileTensor[c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an FP8 (e4m3fn) rank-2 matmul to the SM90 warp-specialized kernel.

    Searches the llama-405B and llama-8B FP8 tuning tables for matching static
    N and K, then falls back to a generic kernel sized by
    `_find_largest_bn_for_sm90_matmul` for other shapes. Honors the
    `AUTOTUNING_MODE` compile-time flag to launch a single autotuning config.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the input tensor `a`; must be
            `float8_e4m3fn` (inferred).
        b_type: Element type of the input tensor `b`; must be
            `float8_e4m3fn` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to `True`);
            the SM90 kernel requires this to be `True` to dispatch.
        elementwise_lambda_fn: Epilogue applied to each output tile after the
            matmul, consuming the tile and returning nothing (defaults to
            `None` for no epilogue).
        elementwise_compute_lambda_fn: Compute lambda transforming each output
            value before the epilogue, returning the transformed value
            (defaults to `None` for no transform).
        pdl_level: Programmatic dependent launch level for overlapping this
            kernel with prior work (defaults to `PDLLevel()`).

    Args:
        c: Rank-2 output tensor of shape `(M, N)`.
        a: Rank-2 input tensor of shape `(M, K)`.
        b: Rank-2 input tensor of shape `(K, N)` when `transpose_b` is `True`.
        ctx: Device context for the kernel launch.

    Returns:
        `DISPATCH_HIT` (1) if the kernel was launched, `DISPATCH_MISS` (0)
        otherwise.
    """
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    var m = Int(c.dim[0]())

    comptime if get_defined_bool["AUTOTUNING_MODE", False]():
        comptime NUM_PIPELINE_STAGES = get_defined_int[
            "TUNE_NUM_PIPELINE_STAGES", 4
        ]()
        comptime NUM_CONSUMER = get_defined_int["TUNE_NUM_CONSUMER", 1]()
        comptime WGMMA_N = get_defined_int["TUNE_WGMMA_N", 128]()
        comptime CLUSTER_DIM_X = get_defined_int["TUNE_CLUSTER_DIM_X", 1]()
        comptime GRID_DIM_X = get_defined_int["TUNE_GRID_DIM_X", 1]()
        comptime GRID_DIM_Y = H100.sm_count // GRID_DIM_X
        comptime BLOCK_TILE_DIM_M = 64 * NUM_CONSUMER

        comptime SCHEDULE_TYPE = MatmulSchedule(
            Int32(get_defined_int["TUNE_SCHEDULE_TYPE", 1]())
        )

        comptime H100_FP8_TUNING_CONFIG = MatmulConfig[
            a_type,
            b_type,
            c_type,
            transpose_b,
        ](
            block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, 128),
            mma_shape=Index(64, WGMMA_N, 32),
            cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
            num_pipeline_stages=NUM_PIPELINE_STAGES,
            num_consumer=NUM_CONSUMER,
            partitioned_multicast=False,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=H100_FP8_TUNING_CONFIG,
            grid_shape=Index(128, 1),
            schedule=MatmulSchedule.DS_SCHEDULER,
        ](c, a, b, ctx)
        return DISPATCH_HIT

    @__parameter
    @always_inline("nodebug")
    def _dispatch[entry: TuningConfigSM90]() raises:
        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=entry.block_tile_shape,
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            num_pipeline_stages=entry.num_pipeline_stages,
            num_consumer=entry.num_consumer,
            partitioned_multicast=entry.partitioned_multicast,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=config,
            schedule=entry.schedule,
            grid_shape=entry.grid_shape,
        ](c, a, b, ctx)

    @__parameter
    @always_inline("nodebug")
    def _search[
        T: Table[TuningConfigSM90], domain: List[Int] = List[Int]()
    ]() raises -> Int:
        comptime m_values = T.query_values[Int, domain=domain](
            rule=lambda (x: TuningConfigSM90) -> Int: x.M
        )

        comptime for static_m in m_values:
            if m <= static_m:
                comptime idx_list = T.query_index[domain=domain](
                    rule=lambda (x: TuningConfigSM90) -> Bool: x.M == static_m
                )

                comptime if idx_list:
                    comptime entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # dynamic m is in the range but cannot find any corresponding config in the table.
                    break

        return DISPATCH_MISS

    # llama-405B-FP8 gemm shapes
    comptime if (
        (static_N == 16384 and static_K == 2048)
        or (static_N == 2304 and static_K == 16384)
        or (static_N == 13312 and static_K == 16384)
        or (static_N == 16384 and static_K == 6656)
    ):
        # First, filter by static params N and K
        comptime nk_idx_list = llama_405b_fp8_table.query_index(
            rule=lambda (x: TuningConfigSM90) -> Bool: x.K == static_K
            and x.N == static_N
        )
        # Search the table for matching values of M within domain
        if _search[llama_405b_fp8_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # llama-8B-FP8 gemm shapes
    elif (
        (static_N == 6144 and static_K == 4096)
        or (static_N == 4096 and static_K == 4096)
        or (static_N == 28672 and static_K == 4096)
        or (static_N == 4096 and static_K == 14336)
    ):
        # Search the table for matching values of M, no domain specified.
        if _search[llama_8b_fp8_table]() == DISPATCH_HIT:
            return DISPATCH_HIT

    else:
        # for gemms with small n and k we fall back the naive kernel
        comptime BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
        comptime BK = 128

        comptime if BN != -1 and static_K % BK == 0:
            # If the number of blocks is less than the number of SMs, it's probably better to not use any persistent kernel
            if ceildiv(m, 64) * ceildiv(static_N, BN) <= H100.sm_count:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, BN, BK),
                    mma_shape=Index(64, BN, 32),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m <= 1024:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, BN, BK),
                    mma_shape=Index(64, BN, 32),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.DS_SCHEDULER,
                    grid_shape=Index(H100.sm_count, 1),
                ](c, a, b, ctx)
                return DISPATCH_HIT
            else:
                comptime config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                ](
                    block_tile_shape=Index(128, BN, BK),
                    mma_shape=Index(64, BN, 32),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.DS_SCHEDULER,
                    grid_shape=Index(H100.sm_count, 1),
                ](c, a, b, ctx)
                return DISPATCH_HIT
    return DISPATCH_MISS


# ===----------------------------------------------------------------------=== #

# BF16 and FP32 Dispatch

# ===----------------------------------------------------------------------=== #


def matmul_dispatch_sm90_bf16_fp32[
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
    c: TileTensor[c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches a BF16 or FP32 rank-2 matmul to the SM90 warp-specialized kernel.

    Searches the BF16 tuning table plus InternVL, llama-3.3-70B, gemma-3-27B,
    and miscellaneous shape tables for matching static N and K, with special
    case configs for small M and known shapes. Skips M=1 in favor of fast
    GEMV. Honors the `AUTOTUNING_MODE` compile-time flag to launch a single
    autotuning config.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the input tensor `a` (inferred).
        b_type: Element type of the input tensor `b` (inferred).
        transpose_b: Whether `b` is stored transposed (defaults to `True`).
        elementwise_lambda_fn: Epilogue applied to each output tile after the
            matmul, consuming the tile and returning nothing (defaults to
            `None` for no epilogue).
        elementwise_compute_lambda_fn: Compute lambda transforming each output
            value before the epilogue, returning the transformed value
            (defaults to `None` for no transform).
        pdl_level: Programmatic dependent launch level for overlapping this
            kernel with prior work (defaults to `PDLLevel()`).

    Args:
        c: Rank-2 output tensor of shape `(M, N)`.
        a: Rank-2 input tensor of shape `(M, K)`.
        b: Rank-2 input tensor of shape `(K, N)` when `transpose_b` is `True`.
        ctx: Device context for the kernel launch.

    Returns:
        `DISPATCH_HIT` (1) if the kernel was launched, `DISPATCH_MISS` (0)
        otherwise.
    """
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime size_factor = 2 if a_type == DType.float32 else 1
    comptime mma_k = 16 // size_factor
    comptime BK = 64 // size_factor

    comptime if get_defined_bool["AUTOTUNING_MODE", False]():
        comptime static_N = c.static_shape[1]
        comptime static_K = a.static_shape[1]

        comptime IS_LARGE_GEMM_SHAPE = get_defined_bool[
            "TUNE_LARGE_GEMM_SHAPE", True
        ]()
        comptime CLUSTER_DIM_X = get_defined_int["TUNE_CLUSTER_DIM_X", 1]()
        comptime CLUSTER_DIM_Y = get_defined_int["TUNE_CLUSTER_DIM_Y", 1]()
        comptime NUM_PIPELINE_STAGES = get_defined_int[
            "TUNE_NUM_PIPELINE_STAGES", 4
        ]()
        comptime NUM_CONSUMER = get_defined_int["TUNE_NUM_CONSUMER", 1]()
        comptime WGMMA_N = get_defined_int["TUNE_WGMMA_N", 128]()
        comptime BLOCK_TILE_DIM_M = 64 * NUM_CONSUMER
        comptime PARTITIONED_MULTICAST = get_defined_bool[
            "TUNE_PARTITIONED_MULTICAST", False
        ]()
        comptime SCHEDULE_TYPE = MatmulSchedule(
            Int32(get_defined_int["TUNE_SCHEDULE_TYPE", 0]())
        )

        comptime if IS_LARGE_GEMM_SHAPE:
            # GRID_DIM_X = 2^n for n in range[0-7]
            comptime GRID_DIM_X = get_defined_int["TUNE_GRID_DIM_X", 1]()
            comptime GRID_DIM_Y = H100.sm_count // GRID_DIM_X

            comptime H100_TUNING_CONFIG = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
            ](
                block_tile_shape=Index(
                    BLOCK_TILE_DIM_M, WGMMA_N // size_factor, BK
                ),
                mma_shape=Index(64, WGMMA_N // size_factor, mma_k),
                cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                num_pipeline_stages=NUM_PIPELINE_STAGES,
                num_consumer=NUM_CONSUMER,
                partitioned_multicast=PARTITIONED_MULTICAST,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=H100_TUNING_CONFIG,
                grid_shape=Index(GRID_DIM_X, GRID_DIM_Y),
                schedule=SCHEDULE_TYPE,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        else:
            comptime IS_SPLITK = get_defined_bool["TUNE_IS_SPLITK", False]()

            comptime if not IS_SPLITK:
                comptime NUM_PIPELINE_STAGES = get_defined_int[
                    "TUNE_NUM_PIPELINE_STAGES", 4
                ]()
                comptime GRID_DIM_X = H100.sm_count
                comptime GRID_DIM_Y = 1

                comptime assert (
                    SCHEDULE_TYPE != MatmulSchedule.DS_SCHEDULER
                    or (
                        CLUSTER_DIM_X == 1
                        and CLUSTER_DIM_Y == 1
                        and (not PARTITIONED_MULTICAST)
                    )
                ), "Deepseek scheduler dose not support multicasting"

                comptime SMALL_SHAPE_H100_BF16_TUNING_CONFIG_NON_SPLITK = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                ](
                    block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, BK),
                    cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                    num_pipeline_stages=NUM_PIPELINE_STAGES,
                    num_consumer=NUM_CONSUMER,
                    partitioned_multicast=PARTITIONED_MULTICAST,
                    pdl_level=pdl_level,
                    mma_shape=Index(64, WGMMA_N, 16),
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=SMALL_SHAPE_H100_BF16_TUNING_CONFIG_NON_SPLITK,
                    grid_shape=Index(GRID_DIM_X, GRID_DIM_Y),
                    schedule=SCHEDULE_TYPE,
                ](c, a, b, ctx)
                return DISPATCH_HIT

            else:
                comptime SPLITS = get_defined_int["TUNE_SPLITS", 2]()

                comptime SMALL_SHAPE_H100_BF16_TUNING_CONFIG_SPLITK = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                ](
                    block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, BK),
                    cluster_shape=Index(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1),
                    num_pipeline_stages=NUM_PIPELINE_STAGES,
                    num_consumer=NUM_CONSUMER,
                    partitioned_multicast=PARTITIONED_MULTICAST,
                    pdl_level=pdl_level,
                    mma_shape=Index(64, WGMMA_N, 16),
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=SMALL_SHAPE_H100_BF16_TUNING_CONFIG_SPLITK,
                    splits=SPLITS,
                    raster_order=RasterOrder.AlongM,
                ](c, a, b, ctx)
                return DISPATCH_HIT

    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]
    comptime a_is_bfloat16_or_float32 = a_type in (
        DType.bfloat16,
        DType.float32,
    )

    var m = Int(c.dim[0]())

    # We have fast gemv for BF16 and FP32, skip H100 matmul here
    # and continue dispatching outside to reach the fast gemv.
    if m == 1:
        return DISPATCH_MISS

    # load custom tables
    comptime tuning_list = _get_tuning_list_bf16[size_factor, mma_k, BK]()
    comptime tuning_table = Table(tuning_list, "tuning_table_bf16")

    @__parameter
    @always_inline("nodebug")
    def _dispatch[entry: TuningConfigSM90]() raises:
        comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=entry.block_tile_shape,
            mma_shape=entry.mma_shape,
            cluster_shape=entry.cluster_shape,
            num_pipeline_stages=entry.num_pipeline_stages,
            num_consumer=entry.num_consumer,
            partitioned_multicast=entry.partitioned_multicast,
            pdl_level=pdl_level,
        )

        comptime if not entry.splits:
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=entry.schedule,
                grid_shape=entry.grid_shape,
            ](c, a, b, ctx)
        else:
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=entry.schedule,
                grid_shape=entry.grid_shape,
                splits=entry.splits.value(),
                raster_order=entry.raster_order.value(),
            ](c, a, b, ctx)

    @__parameter
    @always_inline("nodebug")
    def _search[
        T: Table[TuningConfigSM90],
        domain: List[Int] = List[Int](),
    ]() raises -> Int:
        comptime m_values = T.query_values[Int, domain=domain](
            rule=lambda (x: TuningConfigSM90) -> Int: x.M
        )

        comptime for static_m in m_values:
            if m <= static_m:
                comptime idx_list = T.query_index[domain=domain](
                    rule=lambda (x: TuningConfigSM90) -> Bool: x.M == static_m
                )

                comptime if idx_list:
                    comptime entry = T.configs[idx_list[0]]
                    _dispatch[entry]()
                    return DISPATCH_HIT
                else:
                    # dynamic m is in the range but cannot find any corresponding config in the table.
                    break

        return DISPATCH_MISS

    @always_inline
    def rule_eq_nk(x: TuningConfigSM90) {} -> Bool:
        return x.K == static_K and x.N == static_N

    @always_inline
    def rule_eq_nk_group[
        group: TuningGroup
    ](x: TuningConfigSM90,) {} -> Bool:
        return rule_eq_nk(x) and x.dispatch_group == group

    # First check the new tuning table before falling back on any old results
    comptime tuning_nk_idx_list = tuning_table.query_index(
        rule=rule_eq_nk_group[TuningGroup.CORE]
    )

    # make sure the domain (nk_idx_list) is not empty!
    comptime if tuning_nk_idx_list:
        # TODO(GENAI-326): Skip problematic configs
        # - N=27648, K=5120, M<=8: accuracy bugs
        # - N=5120 with m <=8 : causes hang (unknown root cause in tuning configs)
        if not (
            (static_N == 27648 and static_K == 5120 and m <= 8)
            or (static_N == 5120 and m <= 8)
        ):
            if (
                _search[tuning_table, domain=tuning_nk_idx_list]()
                == DISPATCH_HIT
            ):
                return DISPATCH_HIT

    comptime if a_is_bfloat16_or_float32 and (
        static_N == 4096 and static_K == 1536
    ):
        if m > 256:
            comptime nk_idx_list = tuning_table.query_index(
                rule=rule_eq_nk_group[TuningGroup.MISCELLANEOUS]
            )
            if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
                return DISPATCH_HIT

    comptime if a_is_bfloat16_or_float32 and (
        (static_N == 1536 and static_K == 4096)
        or (static_N == 1536 and static_K == 4608)
    ):
        comptime cond = (static_N == 1536 and static_K == 4096)

        comptime if cond:
            if m < 32:
                var runtime_config = swapAB_smallM[
                    a_type,
                    b_type,
                    c_type,
                    prioritize_compute_over_ctas=True,
                    transpose_b=transpose_b,
                ](
                    m,
                    static_N,
                    static_K,
                    Index(1, 1, 1),
                    1,
                    1,
                    False,
                    pdl_level,
                    4,
                    16,
                )

                def config_fn(
                    m: Int,
                ) {} -> MatmulConfigSM90[a_type, b_type, c_type, transpose_b]:
                    return swapAB_smallM[
                        a_type,
                        b_type,
                        c_type,
                        prioritize_compute_over_ctas=True,
                        transpose_b=transpose_b,
                    ](
                        m,
                        static_N,
                        static_K,
                        Index(1, 1, 1),
                        1,
                        1,
                        False,
                        pdl_level,
                        4,
                        16,
                    )

                comptime configs = build_configs_generic[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    1,
                    32,
                    config_fn=config_fn,
                ]()

                comptime for config in configs:
                    if runtime_config == config:
                        # Only convert to base config after match is found
                        comptime base_config = config.to_base_config()

                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=base_config,
                            schedule=MatmulSchedule.NONE,
                            swapAB=True,
                        ](c, a, b, ctx)
                        return DISPATCH_HIT

            elif m < 41:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 16, 64),
                    mma_shape=Index(64, 16, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 49:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 24, 64),
                    mma_shape=Index(64, 24, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=14,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 65:
                pass

            elif m < 97:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 24, 64),
                    mma_shape=Index(64, 24, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 120:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 32, 64),
                    mma_shape=Index(64, 32, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 129:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 32, 64),
                    mma_shape=Index(64, 32, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=16,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=4,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    # swapAB = True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 161:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 40, 64),
                    mma_shape=Index(64, 40, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 169:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 56, 64),
                    mma_shape=Index(64, 56, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)

                return DISPATCH_HIT

            elif m < 193:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 48, 64),
                    mma_shape=Index(64, 48, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)
                return DISPATCH_HIT

            elif m < 225:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 56, 64),
                    mma_shape=Index(64, 56, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=12,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)
                return DISPATCH_HIT

            elif m < 256:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 64, 64),
                    mma_shape=Index(64, 64, 16),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=10,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                    swapAB=True,
                ](c, a, b, ctx)
                return DISPATCH_HIT
            elif m == 256:
                comptime config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, 48, 64),
                    mma_shape=Index(64, 48, 16),
                    cluster_shape=Index(1, 2, 1),
                    num_pipeline_stages=14,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                    k_group_size=2,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule=MatmulSchedule.NONE,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        comptime nk_idx_list = tuning_table.query_index(
            rule=rule_eq_nk_group[TuningGroup.MISCELLANEOUS]
        )
        if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # Internvl 2xH100 shapes
    comptime if a_is_bfloat16_or_float32 and (
        (static_N == 2560 and static_K == 5120)
        or (static_N == 5120 and static_K == 3584)
        or (static_N == 5120 and static_K == 27648)
        or (static_N == 13824 and static_K == 5120)
        or (static_N == 3200 and static_K == 6400)
        or (static_N == 6400 and static_K == 3200)
        or (static_N == 3200 and static_K == 4992)
        or (static_N == 3200 and static_K == 4608)
        or (static_N == 1664 and static_K == 3200)
        or (static_N == 1536 and static_K == 3200)
        or (static_N == 5120 and static_K == 75837)
        or (static_N == 12800 and static_K == 2560)
    ):
        # First, filter by static params N and K
        comptime nk_idx_list = tuning_table.query_index(
            rule=rule_eq_nk_group[TuningGroup.INTERNVL]
        )
        # Search the table for matching values of M within domain
        if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
            return DISPATCH_HIT

    # matmul configs for llama_3_3_70b
    comptime if a_is_bfloat16_or_float32 and static_N == 2560 and static_K == 8192:
        comptime nk_idx_list = tuning_table.query_index(
            rule=rule_eq_nk_group[TuningGroup.LLAMA_3_3_70B]
        )

        # In this case for m>64 the ranges are not supported.
        # TODO: add ranges for <=256, 512, 1024, 2048
        if m <= 64 or m in [512, 4096, 8192]:
            if _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT:
                return DISPATCH_HIT

    # matmul configs for gemma_3_27b
    comptime if a_is_bfloat16_or_float32 and (
        (static_N == 5376 and static_K == 21504)
        or (static_N == 5376 and static_K == 4096)
        # or (static_N == 262208 and static_K == 5376)
        or (static_N == 43008 and static_K == 5376)
        or (static_N == 8192 and static_K == 5376)
    ):
        comptime nk_idx_list = tuning_table.query_index(
            rule=rule_eq_nk_group[TuningGroup.GEMMA_3_27B]
        )

        comptime if nk_idx_list:
            # TODO: add ranges for <=256, 512, 1024, 2048
            if (
                m >= 16
                and _search[tuning_table, domain=nk_idx_list]() == DISPATCH_HIT
            ):
                return DISPATCH_HIT

    comptime if a_is_bfloat16_or_float32 and static_N == 8192 and static_K == 2048:
        if m <= 16:
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=MatmulSchedule.DS_SCHEDULER,
                grid_shape=Index(128, 1),
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m <= 64:
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=MatmulSchedule.DS_SCHEDULER,
                grid_shape=Index(128, 1),
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime M8192_N8192_K2048_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N8192_K2048_config,
                grid_shape=Index(4, H100.sm_count // 4),
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif m == 4096:
            comptime M4096_N8192_K2048_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N8192_K2048_config,
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    comptime if a_is_bfloat16_or_float32 and static_N == 14336 and static_K == 8192:
        if m <= 64:
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 112 // size_factor, BK),
                mma_shape=Index(64, 112 // size_factor, mma_k),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=MatmulSchedule.DS_SCHEDULER,
                grid_shape=Index(128, 1),
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime M8192_N14336_K8192_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N14336_K8192_config,
                grid_shape=Index(8, H100.sm_count // 8),
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif m == 4096:
            comptime M4096_N14336_K8192_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N14336_K8192_config,
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    comptime if a_is_bfloat16_or_float32 and static_N == 8192 and static_K == 7168:
        if m <= 16:
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=12,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=MatmulSchedule.DS_SCHEDULER,
                grid_shape=Index(128, 1),
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m <= 64:
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(64, 64 // size_factor, BK),
                mma_shape=Index(64, 64 // size_factor, mma_k),
                cluster_shape=Index(1, 1, 1),
                num_pipeline_stages=8,
                num_consumer=1,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=config,
                schedule=MatmulSchedule.DS_SCHEDULER,
                grid_shape=Index(128, 1),
            ](c, a, b, ctx)
            return DISPATCH_HIT
        elif m == 8192:
            comptime M8192_N8192_K7168_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M8192_N8192_K7168_config,
                grid_shape=Index(8, H100.sm_count // 8),
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

        elif m == 4096:
            comptime M4096_N8192_K7168_config = MatmulConfig[
                a_type,
                b_type,
                c_type,
                transpose_b,
            ](
                block_tile_shape=Index(128, 256 // size_factor, BK),
                mma_shape=Index(64, 256 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M4096_N8192_K7168_config,
                schedule=MatmulSchedule.TILE2D,
            ](c, a, b, ctx)
            return DISPATCH_HIT

    comptime if (
        a_is_bfloat16_or_float32
        and static_N == 3840
        and static_K in (15360, 4096)
    ):
        if m <= 512:
            comptime M512_N3840_K15360_config = MatmulConfig[
                a_type, b_type, c_type, transpose_b
            ](
                block_tile_shape=Index(128, 128 // size_factor, BK),
                mma_shape=Index(64, 128 // size_factor, mma_k),
                cluster_shape=Index(2, 1, 1),
                num_pipeline_stages=4,
                num_consumer=2,
                partitioned_multicast=False,
                pdl_level=pdl_level,
            )
            warp_specialize_gemm_with_multicasting[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                config=M512_N3840_K15360_config,
                schedule=MatmulSchedule.NONE,
            ](c, a, b, ctx)

            return DISPATCH_HIT

    comptime BN = _find_largest_bn_for_sm90_matmul[
        a_type, static_N
    ]() // size_factor

    # `audio_decoder/test_residual_fsq.py::test_fsq` test fails if
    # we enable float32 here.
    # Fallback path with vectorized output and cp.async.ca load if K
    # is not multiple of 16B.
    comptime if a_type == .bfloat16 and BN != -1:
        comptime cond = static_N == 4096 and static_K == 1536

        comptime if not cond:
            if m <= 128:
                comptime default_bf16_config = MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ](
                    block_tile_shape=Index(64, BN, BK),
                    mma_shape=Index(64, BN, mma_k),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=1,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=default_bf16_config,
                    schedule=MatmulSchedule.NONE,
                ](c, a, b, ctx)
                return DISPATCH_HIT

        comptime if cond:
            # m < 41: BN = ceildiv(m, 8) * 8, stages=12
            if m < 41:
                var runtime_config = swapAB_smallM_ceildiv[
                    a_type, b_type, c_type, transpose_b
                ](m, pdl_level)

                def config_fn_small(
                    m_val: Int,
                ) {} -> MatmulConfigSM90[a_type, b_type, c_type, transpose_b]:
                    return swapAB_smallM_ceildiv[
                        a_type, b_type, c_type, transpose_b
                    ](m_val, pdl_level)

                comptime configs_small = build_configs_generic[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    1,
                    41,
                    config_fn=config_fn_small,
                ]()

                comptime for config in configs_small:
                    if runtime_config == config:
                        comptime base_config = config.to_base_config()
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=base_config,
                            schedule=MatmulSchedule.NONE,
                            swapAB=True,
                        ](c, a, b, ctx)
                        return DISPATCH_HIT

            elif m < 65:
                pass

            # m 65-128: BN = 40 + ((m-65)//16)*8, stages=8
            elif m < 129:
                var runtime_config = swapAB_midM_linear[
                    a_type, b_type, c_type, transpose_b
                ](m, pdl_level)

                def config_fn_mid(
                    m_val: Int,
                ) {} -> MatmulConfigSM90[a_type, b_type, c_type, transpose_b]:
                    return swapAB_midM_linear[
                        a_type, b_type, c_type, transpose_b
                    ](m_val, pdl_level)

                comptime configs_mid = build_configs_generic[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    65,
                    129,
                    config_fn=config_fn_mid,
                ]()

                comptime for config in configs_mid:
                    if runtime_config == config:
                        comptime base_config = config.to_base_config()
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=base_config,
                            schedule=MatmulSchedule.NONE,
                            swapAB=True,
                        ](c, a, b, ctx)
                        return DISPATCH_HIT

            # m 129-240: BN = 72 + ((m-129)//16)*8, cluster=(2,1,1), k_group=2
            elif m <= 240:
                var runtime_config = swapAB_largeM_clustered[
                    a_type, b_type, c_type, transpose_b
                ](m, pdl_level)

                def config_fn_large(
                    m_val: Int,
                ) {} -> MatmulConfigSM90[a_type, b_type, c_type, transpose_b]:
                    return swapAB_largeM_clustered[
                        a_type, b_type, c_type, transpose_b
                    ](m_val, pdl_level)

                comptime configs_large = build_configs_generic[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    129,
                    241,
                    config_fn=config_fn_large,
                ]()

                comptime for config in configs_large:
                    if runtime_config == config:
                        comptime base_config = config.to_base_config()
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=base_config,
                            schedule=MatmulSchedule.NONE,
                            swapAB=True,
                        ](c, a, b, ctx)
                        return DISPATCH_HIT

        @__parameter
        def get_k_groups[N: Int]() -> Optional[Int]:
            comptime if N == 1536:
                return None
            else:
                return 1

        @__parameter
        def get_consumer_groups[N: Int]() -> Optional[Int]:
            comptime if N == 1536:
                return 1
            else:
                return None

        comptime k_groups = get_k_groups[static_N]()
        comptime consumer_groups = get_consumer_groups[static_N]()

        var runtime_config = MatmulConfigSM90[
            a_type, b_type, c_type, transpose_b
        ](
            m,
            static_N,
            static_K,
            num_k_partitions=1,
            partitioned_multicast=False,
            pdl_level=pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
        )

        # Build compile-time configs (using same parameters)
        comptime configs = build_configs[
            a_type,
            b_type,
            c_type,
            static_N,
            static_K,
            transpose_b,
            1,
            False,
            pdl_level,
            k_groups=k_groups,
            consumer_groups=consumer_groups,
        ]()

        comptime for config in configs:
            # Compare SM90 configs directly
            if runtime_config == config:
                # Only convert to base config after match is found
                comptime base_config = config.to_base_config()

                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=base_config,
                    schedule=MatmulSchedule.NONE,
                ](c, a, b, ctx)
                return DISPATCH_HIT

    # Fallback path, will use scalar 2B output and lots of OOB check.
    comptime if a_type == .bfloat16:
        comptime BN = 256
        comptime default_bf16_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            block_tile_shape=Index(128, BN, 64),
            mma_shape=Index(64, BN, mma_k),
            num_pipeline_stages=4,
            num_consumer=2,
            pdl_level=pdl_level,
        )
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            config=default_bf16_config,
            schedule=MatmulSchedule.NONE,
        ](c, a, b, ctx)
        return DISPATCH_HIT

    return DISPATCH_MISS


def _find_largest_bn_for_sm90_matmul[dtype: DType, N: Int]() -> Int:
    comptime if N % 8 != 0:
        return -1

    def _get_max_bn() -> Int:
        # For float8_e4m3fn maximum BN that will not result in register spilling is 160
        var BN = 160 if dtype == DType.float8_e4m3fn else 256
        while BN >= 8:
            if N % BN == 0:
                return BN
            BN -= 8
        return 8

    return _get_max_bn()
