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
"""CPU entry point for fused GEMM+SwiGLU on SM100.

Creates TMA descriptors for A, B, and C output, then launches the SwiGLU
kernel. Output uses a 3-phase epilogue: st_matrix to full SMEM, SwiGLU pass
to half SMEM, TMA store to GMEM.
"""

from std.math import align_up, ceildiv
from std.sys import size_of

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from max.gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    Coord,
    DefaultEngine,
    Idx,
    RowMajorLayout,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major as tt_row_major,
)
from std.collections import OptionalReg
from structured_kernels.tile_types import create_tma_tile
from structured_kernels.kernel_common import _to_batched_3d

from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from .config import FusedSwiGLUMatmulConfig
from .matmul_swiglu_kernels import (
    SwiGLUKernelConstants,
    blackwell_swiglu_warp_specialized_kernel,
)


def _blackwell_matmul_swiglu[
    transpose_b: Bool,
    *,
    config: FusedSwiGLUMatmulConfig[_, _, _, transpose_b],
    pdl_level: PDLLevel = PDLLevel(),
    BiasLayoutType: TensorLayout = RowMajorLayout[Int64],
    BiasEngine: TensorEngine = DefaultEngine[element_width=1],
](
    c_out: TileTensor[...],
    a_device: TileTensor,
    b_device: TileTensor,
    ctx: DeviceContext,
    bias_tensor: OptionalReg[
        TileTensor[
            config.c_type,
            BiasLayoutType,
            ImmutAnyOrigin,
            Engine=BiasEngine,
        ]
    ] = None,
) raises:
    """Launch fused GEMM+SwiGLU kernel on SM100.

    Takes rank-2 inputs in kernel-frame order. Callers must swap inputs
    upstream (in ``matmul_swiglu_dispatch_sm100``) when ``AB_swapped`` is
    True so that:

        AB_swapped=False:  a_device = X [M, K], b_device = W [2H, K]
                           (W pre-permuted on its N axis, gate/up adjacent)
        AB_swapped=True :  a_device = W [2H, K], b_device = X [M, K]
                           (W pre-permuted on its M axis with stride-8 row
                             blocks: see _swiglu_epilogue_smem_tma docs)

    ``c_out`` is always [M, H] in user frame (H = N/2). The kernel computes
    A @ B^T in kernel frame and writes the SwiGLU-reduced output to
    ``c_out`` directly.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime assert transpose_b, "Only support transposed B"
    comptime assert c_out.rank == 2, "non-batched only: c_out must be rank-2"
    comptime assert a_device.rank == 2, "non-batched only: a must be rank-2"
    comptime assert b_device.rank == 2, "non-batched only: b must be rank-2"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]
    comptime cluster_shape = config.cluster_shape

    # In kernel frame: M is the first input's first dim, N is the second
    # input's first dim. Caller swapped a/b when AB_swapped, so these are
    # always kernel-frame regardless.
    var M = Int(a_device.dim[0]())
    var K = Int(a_device.dim[1]())
    var N = Int(b_device.dim[0]())

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Compile-time constants: TMA layouts, smem sizes, thread count.
    comptime KernelType = SwiGLUKernelConstants[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
    ]

    # Reshape rank-2 inputs to rank-3 for TMA (batch=1)
    var a_3d = _to_batched_3d(a_device)
    var b_3d = _to_batched_3d(b_device)

    # Create TMA descriptors for A and B (same as default kernel)
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tma_tile[
        KernelType.ATileLayout,
        KernelType.ADescLayout,
        a_tma_tile_shape,
        swizzle_mode=config.a_swizzle,
    ](ctx, a_3d)

    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    )
    var b_tma_op = create_tma_tile[
        KernelType.BTileLayout,
        KernelType.BDescLayout,
        b_tma_tile_shape,
        swizzle_mode=config.b_swizzle,
    ](ctx, b_3d)

    # Create 2D TMA descriptor for C output: [M, H] with half-width tiles.
    # ``c_tma_inner`` may pad ``HalfN`` up to 8 when ``register_swiglu``
    # is True (descriptor is allocated but unused in that path).
    comptime c_tma_tile_shape = Index(
        KernelType.c_store_m, KernelType.c_tma_inner
    )
    var c_tma_op = create_tma_tile[
        KernelType.CTileLayout,
        KernelType.CDescLayout,
        c_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, c_out)

    # Grid dimensions: tile over matmul's (M, N=2H) space
    var grid_dim = (
        align_up(ceildiv(M, BM), cluster_shape[0]),
        align_up(ceildiv(N, MMA_N), cluster_shape[1]),
        1,  # no batch
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace = Span[UInt64, MutAnyOrigin]()

    # c_out is always user-frame [user_M, user_H]; derive H from c_out so
    # this is correct under both AB_swapped values.
    var c_gmem_ptr = c_out.ptr
    var c_gmem_stride = UInt32(Int(c_out.dim[1]()))

    # Build 1D bias tile (real ptr when use_bias, dummy c_out ptr otherwise).
    comptime ImmutPtr = UnsafePointer[Scalar[c_type], ImmutAnyOrigin]
    var bias_1d_ptr: ImmutPtr
    comptime if config.use_bias:
        bias_1d_ptr = rebind[ImmutPtr](bias_tensor.value().ptr)
    else:
        bias_1d_ptr = rebind[ImmutPtr](c_out.ptr)
    var bias_1d_tile = KernelType.Bias1DTile(
        bias_1d_ptr, KernelType.Bias1DTileLayout
    )

    comptime cluster_tuple = StaticTuple[Int32, 3](
        Int32(config.cluster_shape[0]),
        Int32(config.cluster_shape[1]),
        Int32(config.cluster_shape[2]),
    )

    # Fully specialize the kernel, extracting TMA type parameters from the
    # concrete TMA op objects via type_of().  This matches the pattern in
    # matmul_bf16fp8.mojo and avoids the "failed to infer a_rank" error
    # that occurs when using a partial specialization in enqueue_function.
    comptime kernel = blackwell_swiglu_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
        transpose_b,
        config=config,
        cluster_shape=cluster_tuple,
        pdl_level=pdl_level,
    ]

    ctx.enqueue_function[kernel, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c_gmem_ptr,
        c_gmem_stride,
        bias_1d_tile,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        block_dim=KernelType.NUM_THREADS,
        shared_mem_bytes=KernelType.total_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )
