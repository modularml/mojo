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
"""CPU entrypoint for grouped 1D2D blockwise FP8 SM100 matmul.

This module provides the public API for launching the grouped 1D2D blockwise
FP8 matmul kernel for Mixture of Experts (MoE) layers.

Usage:
    grouped_matmul_1d2d_blockwise_fp8[transpose_b=True, config=config](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        ctx,
    )
"""

from std.sys import size_of

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from layout import TileTensor, flatten_leading
from structured_kernels.tile_types import create_tma_tile

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ..structured_kernels.config import MatmulConfig
from .blockwise_fp8_1d2d_smem import BlockwiseFP8_1D2DSmem
from .blockwise_fp8_1d2d_matmul_kernel import BlockwiseFP8_1D2DMatmulKernel


def grouped_matmul_1d2d_blockwise_fp8[
    a_scales_type: DType,
    b_scales_type: DType,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
](
    c_device: TileTensor,
    a_device: TileTensor,
    b_device: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch grouped 1D-2D blockwise FP8 matmul kernel for MoE layers.

    This function sets up TMA descriptors and launches the kernel with the
    proper configuration for 1D-2D tensor layout with blockwise FP8 scaling.

    Parameters:
        a_scales_type: `DType` of the A scaling factors (inferred).
        b_scales_type: `DType` of the B scaling factors (inferred).
        transpose_b: Whether B is transposed (inferred). Must be `True`.
        config: `MatmulConfig` controlling MMA shape, CTA group, cluster
            shape, and swizzle modes for the kernel.

    Args:
        c_device: Output tensor (total_tokens, N).
        a_device: Input A tensor (total_tokens, K).
        b_device: Weight tensor B (num_experts, N, K).
        a_scales: Scaling factors for A (K//128 x total_tokens), FP32.
        b_scales: Scaling factors for B (num_experts x N//128 x K//128), FP32.
        a_offsets: Per-expert offsets (num_active_experts + 1).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime assert transpose_b, "Only support transposed B"
    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type
    ), "a_scales_type and b_scales_type must match"
    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"
    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert BK in (64, 128), "Only support BK in (64, 128)"

    comptime num_experts = type_of(b_device).LayoutType.static_shape[0]
    comptime N = type_of(c_device).LayoutType.static_shape[1]
    comptime K = type_of(a_device).LayoutType.static_shape[1]
    comptime expert_n = N

    # Reshape B from (num_experts, N, K) to (num_experts * N, K)
    var b_2d = flatten_leading(b_device)

    # Reshape b_scales from 3D to 2D
    var b_scales_2d = flatten_leading(b_scales)

    # Shared memory size calculation
    comptime SmemType = BlockwiseFP8_1D2DSmem[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    # B200 SMEM limit
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Instantiate kernel type -- layout params derived from caller's TileTensors
    # so types match by construction in enqueue_function.
    comptime KernelType = BlockwiseFP8_1D2DMatmulKernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(b_scales_2d).LayoutType,
        type_of(c_device).LayoutType,
        transpose_b,
        config=config,
        static_N=expert_n,
        static_K=K,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        b_scales_engine=type_of(b_scales_2d).Engine,
        c_device_engine=type_of(c_device).Engine,
        offsets_engine=type_of(a_offsets).Engine,
        expert_ids_engine=type_of(expert_ids).Engine,
        expert_scales_engine=type_of(expert_scales).Engine,
    ]
    comptime kernel = KernelType.run

    # Create TMA descriptors using kernel's layout types
    var a_tma_op = create_tma_tile[
        KernelType.ATileLayout,
        KernelType.ADescLayout,
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode=config.a_swizzle,
    ](ctx, a_device)

    var b_tma_op = create_tma_tile[
        KernelType.BTileLayout,
        KernelType.BDescLayout,
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b_2d)

    var a_scales_tma_op = create_tma_tile[
        KernelType.AScalesLayout,
        KernelType.AScalesLayout,
        Index(1, BM),
    ](ctx, a_scales)

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    # Thread configuration: 1 Load + 1 MMA + 4 Epilogue = 6 warps = 192 threads
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime epilogue_warps = 4

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_scales_tma_op,
        b_scales_2d,
        a_offsets,
        expert_ids,
        expert_scales,
        c_device,
        Int32(num_active_experts),
        UInt32(K),
        grid_dim=grid_dim,
        block_dim=(32 * (load_warps + mma_warps + epilogue_warps)),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )


def grouped_matmul_dynamic_scaled_fp8_1d2d[
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    transpose_b: Bool = True,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Compatibility wrapper that matches the existing dispatch API.

    Creates the default config and calls the new structured kernel.

    Parameters:
        a_scales_type: `DType` of the A scaling factors (inferred).
        b_scales_type: `DType` of the B scaling factors (inferred).
        transpose_b: Whether B is transposed (defaults to `True`).

    Args:
        c: Output tensor of shape `(total_tokens, N)`.
        a: Input A tensor of shape `(total_tokens, K)`.
        b: Weight tensor B of shape `(num_experts, N, K)`.
        a_scales: Scaling factors for A of shape `(K//128, total_tokens)`,
            FP32.
        b_scales: Scaling factors for B of shape
            `(num_experts, N//128, K//128)`, FP32.
        a_offsets: Per-expert offsets of length `num_active_experts + 1`.
        expert_ids: Active expert IDs of length `num_active_experts`.
        expert_scales: Per-expert output scaling of length `num_experts`.
        num_active_experts: Number of active experts.
        ctx: Device context for kernel launch.
    """
    comptime umma_shape: IndexList[3] = Index(64, 64, 32)
    # A-scales: 1 x BM floats per pipeline stage
    comptime BM = umma_shape[0]  # cta_group=1
    comptime a_scales_smem_per_stage = BM * size_of[DType.float32]()

    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = type_of(c).dtype
    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
        extra_smem_per_stage=a_scales_smem_per_stage,
    )

    grouped_matmul_1d2d_blockwise_fp8[
        a_scales_type=a_scales_type,
        b_scales_type=b_scales_type,
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        ctx,
    )
