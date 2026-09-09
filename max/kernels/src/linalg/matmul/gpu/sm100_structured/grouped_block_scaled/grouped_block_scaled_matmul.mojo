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
"""CPU entry points for grouped block-scaled SM100 matmul.

Supports multiple GEMM operations with variable problem sizes per group.
Uses TMATensorTileArray for per-block updatable TMA descriptors.

This module implements grouped block-scaled GEMM following the architecture
of NVIDIA CuTe DSL grouped_blockscaled_gemm.py:
1. Creates template TMA descriptors from caller-provided TileTensors
2. Creates TMATensorTileArray with one tensormap per block
3. Launches GroupedBlockScaledMatmulKernel with per-group pointers

All tensor arguments use TileTensor with compile-time dtype constraints.
Callers must provide template tensors in the correct shapes:
- A/B/C templates: 3D (1, M/N, K/N) with batch=1
- Scale factor templates: 5D (1, groups_M/N, groups_K, SF_ATOM_M[0],
  SF_ATOM_M[1] * SF_ATOM_K)

Usage:
    # Per-group pointers as TileTensor[.uint64, ...]
    var a_ptrs = TileTensor(ptr, tile_row_major[num_groups, 1]())
    ...

    # Problem sizes as TileTensor[.int32, ...]
    var problem_sizes = TileTensor(ptr, tile_row_major[num_groups, 4]())

    # 3D template TileTensors for TMA descriptor creation
    var a_template = TileTensor(a_ptr, tile_row_major[1, M, K]())
    ...

    grouped_block_scaled_matmul[...](
        a_ptrs, b_ptrs, c_ptrs, sfa_ptrs, sfb_ptrs,
        problem_sizes, num_groups, total_tiles,
        a_template, b_template, c_template,
        sfa_template, sfb_template, ctx
    )
"""

from std.collections import Optional
from std.math import align_up
from std.sys import size_of

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from layout import TileTensor

from structured_kernels.tile_types import create_tma_tile

from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from linalg.utils import elementwise_compute_lambda_type
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from .grouped_block_scaled_matmul_kernel import GroupedBlockScaledMatmulKernel
from .grouped_block_scaled_smem import GroupedBlockScaledSmem


# =============================================================================
# Validation: Check constraints matching NVIDIA CuTe DSL
# =============================================================================


def validate_grouped_gemm_constraints[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]():
    """Validate grouped GEMM configuration constraints.

    Constraints from NVIDIA CuTe DSL grouped_blockscaled_gemm.py:
    - MMA tiler M: 128 or 256
    - MMA tiler N: 128 or 256
    - Cluster M/N: Power of 2, <=4 per axis (for SF multicast)
    - Total cluster size: <=16
    - 16-byte alignment on contiguous dimensions

    Parameters:
        a_type: Element type of the A operand matrices.
        b_type: Element type of the B operand matrices.
        c_type: Element type of the C output matrices.
        sfa_dtype: Element type of the A scaling factors.
        sfb_dtype: Element type of the B scaling factors.
        transpose_b: Whether B operands are stored transposed (must be True).
        config: Block-scaled matmul configuration carrying tile, cluster, and
            pipeline parameters.
    """
    # MMA tiler constraints
    comptime assert config.mma_shape[0] in (
        128,
        256,
    ), "MMA tiler M must be 128 or 256"
    comptime assert config.mma_shape[1] in (
        128,
        256,
    ), "MMA tiler N must be 128 or 256"

    # Cluster constraints
    comptime assert (
        config.cluster_shape[0] <= 4
    ), "Cluster M must be <=4 for SF multicast"
    comptime assert (
        config.cluster_shape[1] <= 4
    ), "Cluster N must be <=4 for SF multicast"
    comptime assert (
        config.cluster_shape[0] * config.cluster_shape[1] <= 16
    ), "Total cluster size must be <=16"

    # Must be transposed B
    comptime assert transpose_b, "Only support transposed B"

    # SF dtype must match
    comptime assert sfa_dtype == sfb_dtype, "sfa_dtype and sfb_dtype must match"


# =============================================================================
# Main Entry Point: Grouped Block-Scaled Matmul
# =============================================================================


def grouped_block_scaled_matmul[
    transpose_b: Bool,
    max_groups: Int,
    *,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
](
    # Per-group tensor pointers (max_groups, 1) TileTensors
    a_ptrs: TileTensor[.uint64, ...],
    b_ptrs: TileTensor[.uint64, ...],
    c_ptrs: TileTensor[.uint64, ...],
    sfa_ptrs: TileTensor[.uint64, ...],
    sfb_ptrs: TileTensor[.uint64, ...],
    # Per-group problem sizes: (max_groups, 4) with [M, N, K, L]
    problem_sizes: TileTensor[.int32, ...],
    # Number of active groups (runtime)
    num_groups: Int,
    # Total tiles across all groups (computed by caller on host)
    total_tiles: Int,
    # Template tensors for TMA descriptor creation
    # A/B/C: 3D (1, M/N, K/N) batched TileTensors
    a_template: TileTensor[config.a_type, ...],
    b_template: TileTensor[config.b_type, ...],
    c_template: TileTensor[config.c_type, ...],
    # Scale factor templates: 5D (1, M/N_groups, K_groups, SF_ATOM_M[0],
    # SF_ATOM_M[1] * SF_ATOM_K)
    sfa_template: TileTensor[config.sfa_dtype, ...],
    sfb_template: TileTensor[config.sfb_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Launch grouped block-scaled FP8 matmul kernel on SM100.

    Computes C[g] = scale(A[g]) @ scale(B[g]) for g in range(num_groups),
    where each group can have different M, N, K dimensions.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        max_groups: Maximum number of groups (compile-time bound).
        config: Block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional epilogue lambda for element-wise
            operations on output. Applied after matmul, before writing to global
            memory.

    Args:
        a_ptrs: Per-group A matrix pointers (max_groups, 1).
        b_ptrs: Per-group B matrix pointers (max_groups, 1).
        c_ptrs: Per-group C matrix pointers (max_groups, 1).
        sfa_ptrs: Per-group A scaling factor pointers (max_groups, 1).
        sfb_ptrs: Per-group B scaling factor pointers (max_groups, 1).
        problem_sizes: Per-group problem sizes (max_groups, 4) as [M, N, K, L].
        num_groups: Actual number of groups (runtime value <= max_groups).
        total_tiles: Total tiles across all groups (computed by caller).
        a_template: Template A tensor (1, M, K) for TMA descriptor creation.
        b_template: Template B tensor (1, N, K) for TMA descriptor creation.
        c_template: Template C tensor (1, M, N) for TMA descriptor creation.
        sfa_template: Template SFA tensor for TMA descriptor creation.
        sfb_template: Template SFB tensor for TMA descriptor creation.
        ctx: Device context for kernel launch.

    Raises:
        If configuration constraints are violated.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime sfa_dtype = config.sfa_dtype
    comptime sfb_dtype = config.sfb_dtype

    # ===== Validate constraints =====
    validate_grouped_gemm_constraints[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b, config
    ]()

    # ===== Compute tile dimensions =====
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]
    comptime cluster_shape = config.cluster_shape
    comptime CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]

    # ===== Instantiate Kernel First (TMA layouts computed from config) =====
    comptime matmul_kernel = GroupedBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
        max_groups=max_groups,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ]
    comptime KernelType = type_of(matmul_kernel)

    # ===== Create template TMA descriptors using kernel-derived layouts =====

    # A matrix TMA
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tma_tile[
        KernelType.ATileLayout,
        KernelType.ADescLayout,
        a_tma_tile_shape,
        swizzle_mode=config.a_swizzle,
    ](ctx, a_template)

    # B matrix TMA
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    )
    var b_tma_op = create_tma_tile[
        KernelType.BTileLayout,
        KernelType.BDescLayout,
        b_tma_tile_shape,
        swizzle_mode=config.b_swizzle,
    ](ctx, b_template)

    # C matrix TMA
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    )
    var c_tma_op = create_tma_tile[
        KernelType.CTileLayout,
        KernelType.CDescLayout,
        c_tma_tile_shape,
        swizzle_mode=config.c_swizzle,
    ](ctx, c_template)

    # Scaling factors TMA
    comptime sfa_tma_tile_shape = Index(
        1,
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfa_tma_op = create_tma_tile[
        KernelType.SFATileLayout,
        KernelType.SFADescLayout,
        sfa_tma_tile_shape,
    ](ctx, sfa_template)

    comptime sfb_tma_tile_shape = Index(
        1,
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfb_tma_op = create_tma_tile[
        KernelType.SFBTileLayout,
        KernelType.SFBDescLayout,
        sfb_tma_tile_shape,
    ](ctx, sfb_template)

    # ===== Create TMATensorTileArray for per-block tensormaps =====
    # Each block gets its own tensormap copy that can be updated at runtime.
    # We allocate based on actual grid size at runtime,
    # but use a compile-time constant for the type parameter.
    # The type parameter is not used for bounds checking - actual indexing
    # is by pointer arithmetic: ptr + index * 128.

    # Allocate device memory for tensormap arrays (128 bytes per descriptor)
    # For 2SM, grid size = total_tiles * cluster_shape[0] (each cluster = 1 tile)
    comptime TMA_DESC_SIZE = 128
    var num_blocks = total_tiles * cluster_shape[0]  # Actual grid size
    var device_tensormaps_a = ctx.enqueue_create_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_b = ctx.enqueue_create_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_sfa = ctx.enqueue_create_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_sfb = ctx.enqueue_create_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_c = ctx.enqueue_create_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )

    # Create TMATensorTileArray instances
    # Note: The compile-time num_blocks parameter is just for type signature.
    # Actual array access uses pointer math, so we can safely use any valid index
    # up to the allocated size. Using CLUSTER_SIZE as type param for now.
    var tma_array_a = KernelType.TMATensorTileArrayA(device_tensormaps_a)
    var tma_array_b = KernelType.TMATensorTileArrayB(device_tensormaps_b)
    var tma_array_sfa = KernelType.TMATensorTileArraySFA(device_tensormaps_sfa)
    var tma_array_sfb = KernelType.TMATensorTileArraySFB(device_tensormaps_sfb)
    var tma_array_c = KernelType.TMATensorTileArrayC(device_tensormaps_c)

    # ===== Initialize per-block tensormaps from templates =====
    # Each block's tensormap slot needs to be initialized with the template descriptor.
    # The kernel will update these at runtime when groups change.
    # We copy the template descriptor to all block slots, then copy to device.

    # Create host buffers for tensormap initialization
    var host_buf_a = ctx.enqueue_create_host_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_b = ctx.enqueue_create_host_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_sfa = ctx.enqueue_create_host_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_sfb = ctx.enqueue_create_host_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_c = ctx.enqueue_create_host_buffer[.uint8](
        TMA_DESC_SIZE * num_blocks
    )

    # Copy template descriptor bytes to each block's slot
    for blk in range(num_blocks):
        for j in range(TMA_DESC_SIZE):
            host_buf_a[blk * TMA_DESC_SIZE + j] = a_tma_op.descriptor.data[j]
            host_buf_b[blk * TMA_DESC_SIZE + j] = b_tma_op.descriptor.data[j]
            host_buf_sfa[blk * TMA_DESC_SIZE + j] = sfa_tma_op.descriptor.data[
                j
            ]
            host_buf_sfb[blk * TMA_DESC_SIZE + j] = sfb_tma_op.descriptor.data[
                j
            ]
            host_buf_c[blk * TMA_DESC_SIZE + j] = c_tma_op.descriptor.data[j]

    ctx.enqueue_copy(device_tensormaps_a, host_buf_a)
    ctx.enqueue_copy(device_tensormaps_b, host_buf_b)
    ctx.enqueue_copy(device_tensormaps_sfa, host_buf_sfa)
    ctx.enqueue_copy(device_tensormaps_sfb, host_buf_sfb)
    ctx.enqueue_copy(device_tensormaps_c, host_buf_c)
    ctx.synchronize()

    # ===== Shared Memory Size =====
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Use GroupedBlockScaledSmem which includes SMEM for TMA descriptors
    comptime SmemType = GroupedBlockScaledSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    # ===== Grid and Block Dimensions =====
    # For grouped GEMM, grid is based on total tiles
    # For 2SM (cta_group=2), each cluster handles 1 tile, so we need
    # total_tiles * cluster_shape[0] blocks = total_tiles clusters
    var grid_dim = (
        align_up(total_tiles * cluster_shape[0], cluster_shape[0]),
        1,
        1,
    )

    # Thread organization: 7 warps (224 threads)
    # 1 TMA + 1 MMA + 1 Scheduler + 4 Epilogue warps
    comptime num_threads = 32 * 7

    # ===== Kernel Launch =====
    # Dispatch to run_2sm() for 2SM mode (cta_group=2), else run() for 1SM

    comptime if config.cta_group == 2:
        # 2SM mode: use CLC-based run_2sm() for proper cluster synchronization
        ctx.enqueue_function[matmul_kernel.run_2sm](
            # Template TMA descriptors
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            # Per-block tensormap arrays
            tma_array_a,
            tma_array_b,
            tma_array_sfa,
            tma_array_sfb,
            tma_array_c,
            # Per-group pointer tensors
            a_ptrs,
            b_ptrs,
            c_ptrs,
            sfa_ptrs,
            sfb_ptrs,
            # Problem sizes and group count
            problem_sizes,
            Int32(num_groups),
            grid_dim=grid_dim,
            block_dim=num_threads,
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
        )
    else:
        # 1SM mode: use linear iteration run()
        ctx.enqueue_function[matmul_kernel.run](
            # Template TMA descriptors
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            # Per-block tensormap arrays
            tma_array_a,
            tma_array_b,
            tma_array_sfa,
            tma_array_sfb,
            tma_array_c,
            # Per-group pointer tensors
            a_ptrs,
            b_ptrs,
            c_ptrs,
            sfa_ptrs,
            sfb_ptrs,
            # Problem sizes and group count
            problem_sizes,
            Int32(num_groups),
            grid_dim=grid_dim,
            block_dim=num_threads,
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
        )


# =============================================================================
# Helper: Calculate SMEM size for grouped kernel
# =============================================================================


def grouped_smem_size[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]() -> Int:
    """Calculate shared memory size for grouped block-scaled kernel.

    Parameters:
        a_type: Element type of the A operand matrices.
        b_type: Element type of the B operand matrices.
        c_type: Element type of the C output matrices.
        sfa_dtype: Element type of the A scaling factors.
        sfb_dtype: Element type of the B scaling factors.
        transpose_b: Whether B operands are stored transposed (must be True).
        config: Block-scaled matmul configuration carrying tile, cluster, and
            pipeline parameters.

    Returns:
        SMEM size in bytes, including tensormap descriptor storage.
    """
    comptime SmemType = GroupedBlockScaledSmem[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b, config=config
    ]
    return size_of[SmemType]()
