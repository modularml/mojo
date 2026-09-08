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

"""CPU entry points for blockwise FP8 SM100 matmul.

Creates TMA descriptors for A, B, C and A-scales, then launches the
warp-specialized blockwise FP8 kernel with register-based accumulation.
"""

from std.math import align_up, ceildiv
from std.sys import get_defined_bool, size_of

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from layout import TileTensor
from structured_kernels.tile_types import create_tma_tile

from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from ..structured_kernels.config import MatmulConfig, GEMMKind
from .blockwise_fp8_smem import BlockwiseFP8Smem
from .blockwise_fp8_matmul_kernel import BlackwellBlockwiseFP8MatmulKernel

# Legacy kernel for fallback via -D USE_LEGACY_BLOCKWISE_FP8=True
from ...sm100.warp_specialized_blockwise_fp8 import (
    sm100_warp_specialized_blockwise_fp8,
)


# =============================================================================
# Public API: blockwise_fp8_matmul
# =============================================================================


def blockwise_fp8_matmul[
    transpose_b: Bool,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
    n_scale_granularity: Int = 128,
](
    c: TileTensor,
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    a_scales: TileTensor[mut=False, ...],
    b_scales: TileTensor[mut=False, ...],
    ctx: DeviceContext,
) raises:
    """Launch blockwise FP8 matmul kernel.

    Parameters:
        transpose_b: Whether `b` is stored transposed as N x K rather than
            K x N. Only transposed B is currently supported.
        a_scales_type: `DType` of the `a_scales` scaling factors. Must be
            `float32` and must match `b_scales_type`.
        b_scales_type: `DType` of the `b_scales` scaling factors. Must be
            `float32` and must match `a_scales_type`.
        config: `MatmulConfig` controlling tile shapes, CTA grouping, cluster
            shape, swizzle modes, and pipeline stages for the kernel launch.
        n_scale_granularity: B-scale block size along the N dimension in
            elements (defaults to 128). Set smaller when the matmul's N-tile
            spans multiple finer-grained scale blocks.

    Args:
        c: Output matrix (M x N).
        a: Input matrix A (M x K), FP8.
        b: Input matrix B (K x N or N x K if transposed), FP8.
        a_scales: Scaling factors for A (M x ceil(K/128)), FP32.
        b_scales: Scaling factors for B (ceil(N/128) x ceil(K/128)), FP32.
        ctx: Device context for kernel launch.

    Environment:
        USE_LEGACY_BLOCKWISE_FP8: If True, use legacy kernel instead of structured.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type

    comptime if get_defined_bool["USE_LEGACY_BLOCKWISE_FP8", False]():
        sm100_warp_specialized_blockwise_fp8[
            transpose_b=transpose_b,
            a_scales_type=a_scales_type,
            b_scales_type=b_scales_type,
            config=config,
        ](c, a, b, a_scales, b_scales, ctx)
        return

    comptime assert transpose_b, "Only support transposed B"
    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type
    ), "Only support float32 for scales"

    if (Int(a_scales.dim[1]()) * size_of[a_scales_type]()) % 16 != 0:
        raise Error(
            "a_scales should be a multiple of 16 bytes on the M dimension"
        )

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
    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    # Re-create config with extra_smem_per_stage to account for A-scales SMEM.
    # The caller's config was computed without a_scales overhead, so pipeline
    # stages may be too high. _maximize_pipeline_stages handles the correction
    # when given extra_smem_per_stage = BM * size_of[a_scales_type].
    comptime a_scales_smem_per_stage = BM * size_of[a_scales_type]()
    comptime corrected_config = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        cta_group=config.cta_group,
        mma_shape=config.mma_shape,
        cluster_shape=config.cluster_shape,
        AB_swapped=config.AB_swapped,
        num_accum_pipeline_stages=config.num_accum_pipeline_stages,
        num_clc_pipeline_stages=config.num_clc_pipeline_stages,
        block_swizzle_size=config.block_swizzle_size,
        raster_order=config.raster_order,
        k_group_size=config.k_group_size,
        extra_smem_per_stage=a_scales_smem_per_stage,
        gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime SmemType = BlockwiseFP8Smem[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        transpose_b,
        config=corrected_config,
    ]
    comptime smem_size = size_of[SmemType]()

    # Instantiate kernel type - use corrected_config which has proper num_pipeline_stages
    comptime Kernel = BlackwellBlockwiseFP8MatmulKernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(b_scales).LayoutType,
        transpose_b=transpose_b,
        config=corrected_config,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(corrected_config.cluster_shape[0]),
            Int32(corrected_config.cluster_shape[1]),
            Int32(corrected_config.cluster_shape[2]),
        ),
        n_scale_granularity=n_scale_granularity,
        b_scales_engine=type_of(b_scales).Engine,
    ]

    # Create TMA descriptors using kernel's layout types
    var a_tma_op = create_tma_tile[
        Kernel.ATileLayout,
        Kernel.ADescLayout,
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode=config.a_swizzle,
    ](ctx, a)

    var b_tma_op = create_tma_tile[
        Kernel.BTileLayout,
        Kernel.BDescLayout,
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b)

    var a_scales_tma_op = create_tma_tile[
        Kernel.AScalesLayout,
        Kernel.AScalesLayout,
        Index(1, BM),
    ](ctx, a_scales)

    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    var c_tma_op = create_tma_tile[
        Kernel.CTileLayout,
        Kernel.CDescLayout,
        c_tma_tile_shape,
        swizzle_mode=config.c_swizzle,
    ](ctx, c)

    var grid_dim = (
        align_up(ceildiv(M, BM), config.cluster_shape[0]),
        align_up(ceildiv(N, MMA_N), config.cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], config.cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], config.cluster_shape[1])),
        1,
    )

    var problem_shape = StaticTuple[Int32, 3](Int32(M), Int32(N), Int32(K))

    ctx.enqueue_function[Kernel.run, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_scales_tma_op,
        cluster_dim,
        Int32(ceildiv(K, BK)),
        b_scales,
        problem_shape,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps = 7 warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )
