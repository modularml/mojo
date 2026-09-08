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
"""Implements grouped (batched expert) GEMM for SM90 (Hopper) GPUs.

Provides the `grouped_matmul_sm90` entry point, which dispatches a
variable-length batched matmul across MoE expert weight matrices using
TMA-based warp-specialized pipelining via `HopperMatmulSM90Kernel`.
"""
from std.collections import Optional
from std.math import ceildiv
from std.sys import size_of

from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Layout,
    TileTensor,
    flatten_leading,
)
from layout.tma_async import create_tensor_tile

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from .matmul_kernels import HopperMatmulSM90Kernel
from .matmul import _get_c_smem_layout

from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig


@always_inline
def default_config_sm90[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    wgmma_shape: IndexList[3],
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    """Returns the default SM90 matmul config for the given WGMMA shape.

    Sets BM=128, BN from the WGMMA N dimension, BK to the maximum TMA-aligned
    value, 4 pipeline stages, 2 consumer warp groups, and no multicast.

    Parameters:
        a_type: A-matrix (activations) element type; drives BK via TMA
            alignment.
        b_type: B-matrix (expert weights) element type.
        c_type: Output element type.
        transpose_b: Whether B is stored transposed.
        wgmma_shape: WGMMA instruction shape `(M, N, K)`; `N` sets the
            block tile `BN`.

    Returns:
        A `MatmulConfig` suitable for use as the default grouped matmul config.
    """
    comptime BN = wgmma_shape[1]
    comptime BK = 128 // size_of[a_type]()
    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, BN, BK),
        mma_shape=wgmma_shape,
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )


def grouped_matmul_sm90[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool = True,
    wgmma_shape: IndexList[3] = Index(64, 256, 16),
    config: MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ] = default_config_sm90[a_type, b_type, c_type, transpose_b, wgmma_shape](),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[a_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    b: TileTensor[b_type, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Performs grouped GEMM for MoE routing on SM90 (Hopper) GPUs.

    Dispatches a batched expert matmul where each expert has a variable
    number of tokens, stored contiguously in `a` at offsets given by
    `a_offsets`. Expert weight matrices are stacked along axis 0 in `b`.
    Uses TMA-based warp-specialized pipelining via `HopperMatmulSM90Kernel`.

    Parameters:
        c_type: Output element type.
        a_type: A-matrix (activations) element type.
        b_type: B-matrix (expert weights) element type.
        transpose_b: Whether B is stored transposed (must be True).
        wgmma_shape: WGMMA instruction shape (M, N, K).
        config: Full SM90 kernel configuration.
        elementwise_lambda_fn: Optional epilogue applied to each output tile.

    Args:
        c: Output matrix `[total_tokens, N]`.
        a: Activation matrix `[total_tokens, K]`.
        a_offsets: Per-expert token start offsets into `a`.
        max_num_tokens_per_expert: Maximum tokens for any single expert.
        b: Expert weight tensor `[num_experts, N, K]`.
        expert_ids: Active expert indices `[num_active_experts]`.
        num_active_experts: Number of experts with non-zero token count.
        ctx: Device context for kernel launch.
    """
    # Early-exit for empty inputs to avoid creating invalid TMA descriptors.
    if num_active_experts == 0 or Int(a.dim[0]()) == 0 or Int(c.dim[0]()) == 0:
        return
    comptime num_experts = b.static_shape[0]
    comptime N = b.static_shape[1]
    comptime K = b.static_shape[2]

    comptime cluster_shape = StaticTuple[Int32, 3](
        Int32(config.cluster_shape[0]),
        Int32(config.cluster_shape[1]),
        Int32(config.cluster_shape[2]),
    )

    comptime c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
        config.k_group_size,
    ]()
    comptime c_smem_tile = Index(
        c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
    )

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]

    # Create TMA op for the entire A tensor including all tokens.
    var a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a
    )

    # Flatten B tensor into a 2D TileTensor for easier TMA support.
    var b_flat = flatten_leading(b)
    var b_tma_op = create_tensor_tile[Index(BN, BK), swizzle_mode=b_swizzle](
        ctx, b_flat
    )

    # Create a dummy TMA op for C, we don't support TMA store for output.
    var c_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=c_swizzle](
        ctx, c
    )

    comptime num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE
    comptime smem_size = config.num_pipeline_stages * (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + (size_of[Int64]() * 2)
    ) + c_smem_layout.size() * size_of[c_type]()

    comptime kernel = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        type_of(a).LayoutType,
        type_of(b_flat).LayoutType,
        type_of(c).LayoutType,
        c_smem_layout,
        config.block_tile_shape,
        wgmma_shape,
        cluster_shape,
        config.num_pipeline_stages,
        num_threads,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast=config.partitioned_multicast,
        use_tma_store=False,
        promotion_frequency=1,
        pdl_level=config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        a_engine=type_of(a).Engine,
        b_engine=type_of(b_flat).Engine,
        c_engine=type_of(c).Engine,
        a_offsets_engine=type_of(a_offsets).Engine,
        expert_ids_engine=type_of(expert_ids).Engine,
    ].run_grouped[
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(c_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        type_of(c_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).desc_shape,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        type_of(c).LayoutType,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_offsets,
        expert_ids,
        c.as_unsafe_any_origin(),
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
