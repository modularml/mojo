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
Provides SM90 warp-specialized GEMM dispatchers with TMA multicasting, split-K
reduction, and optional epilogue fusion for Hopper matmuls.
"""
from std.math import ceildiv
from std.sys import size_of

from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import pdl_launch_attributes
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import H100
from layout import (
    Layout,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
)
from layout.tma_async import create_tensor_tile, create_tma_tile_template
from std.logger import Logger
from std.bit import log2_floor

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig, get_hilbert_lut_with_cache
from ..tile_scheduler import MatmulSchedule, RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from .matmul_kernels import HopperMatmulSM90Kernel, find_K_alignment_upto_16B
from .matmul_kernel_persistent import HopperMatmulSM90Kernel
from std.collections import OptionalReg

comptime logger = Logger()


def _is_valid_cluster_shape[
    cluster_shape: IndexList[3]
](grid_shape: IndexList[2], num_tiles_n: Int) -> Bool:
    if num_tiles_n % cluster_shape[0] != 0:
        return False

    comptime for i in range(2):
        if (
            grid_shape[i] < cluster_shape[i]
            or grid_shape[i] % cluster_shape[i] != 0
        ):
            return False

    return True


def _get_grid_shape[
    cluster_shape: IndexList[3] = Index(1, 1, 1)
](num_tiles_n: Int) -> IndexList[2]:
    # Hardcode values on purpose until we move this inside tile scheduler
    # in a more robust way.
    comptime h100_num_SMs = H100.sm_count
    var num_blocks_n = min(num_tiles_n, h100_num_SMs)
    var adjusted_grid_shape = Index(
        num_blocks_n,
        h100_num_SMs // num_blocks_n,
    )

    # A Naive heuristic to select grid shape based on number of tile in N.
    if num_tiles_n % 8 == 0 or not _is_valid_cluster_shape[cluster_shape](
        adjusted_grid_shape, num_tiles_n
    ):
        return Index(8, 16)

    return adjusted_grid_shape


def _is_valid_grid_shape[
    grid_shape: IndexList[2], cluster_shape: IndexList[3]
](num_tiles_n: Int) -> Bool:
    comptime assert (
        grid_shape[0] * grid_shape[1] <= H100.sm_count
    ), "Total grid size exceed number of SMs in H100."

    if not _is_valid_cluster_shape[cluster_shape](grid_shape, num_tiles_n):
        return False

    if grid_shape[0] <= num_tiles_n:
        return num_tiles_n % grid_shape[0] == 0

    return grid_shape[0] % num_tiles_n == 0


def warp_specialize_gemm_with_multicasting[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
    splits: Int = 0,
    raster_order: RasterOrder = RasterOrder.AlongM,
    swapAB: Bool = False,
](
    c_device: TileTensor[c_type, ...],
    a_device: TileTensor[a_type, ...],
    b_device: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    """Dispatches an SM90 warp-specialized matmul to split-K or regular kernel paths.

    Routes to `warp_specialize_gemm_with_multicasting_splitk` when `splits > 0`
    (partitioning K across multiple thread blocks with a reduction pass), or to
    the persistent `_warp_specialize_gemm_with_multicasting_impl` otherwise.
    Supports optional TMA stores, epilogue functions, scheduling strategies, and
    A/B swapping for small-M problems.

    Parameters:
        c_type: Element type of the output `C` tensor.
        a_type: Element type of the input `A` tensor.
        b_type: Element type of the input `B` tensor.
        transpose_b: Whether `B` is stored transposed. Must be `True` on H100.
        config: Kernel configuration holding block tile shape, cluster shape,
            MMA shape, pipeline stages, and consumer count.
        grid_shape: Optional override for the launch grid shape
            `(blocks_m, blocks_n)`. Required when `schedule` is `DS_SCHEDULER`
            (defaults to `None`).
        use_tma_store: Whether to use TMA for storing the output tile to global
            memory (defaults to `False`).
        elementwise_lambda_fn: Optional epilogue function applied to the output
            tile after the matmul (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute lambda fused into the
            MMA loop (defaults to `None`). Mutually exclusive with
            `elementwise_lambda_fn`.
        schedule: Persistent kernel scheduling strategy (defaults to
            `MatmulSchedule.NONE`).
        hilbert_swizzle: Whether to reorder tile iteration via a Hilbert curve
            for improved L2 locality (defaults to `False`).
        splits: Number of partitions along `K` for split-K reduction. Routes to
            the split-K kernel when greater than zero (defaults to `0`).
        raster_order: Tile rasterization direction, `AlongM` or `AlongN`
            (defaults to `AlongM`).
        swapAB: Whether to swap the `A` and `B` operands for small-`M` problems
            (defaults to `False`).

    Args:
        c_device: Output `C` tensor on device with shape `(M, N)`.
        a_device: Input `A` tensor on device with shape `(M, K)`.
        b_device: Input `B` tensor on device with shape `(N, K)` when
            `transpose_b` is `True`.
        ctx: Device context used to enqueue the kernel.
    """

    comptime if splits > 0:
        # TODO: Remove if unnecessary otherwise add support
        comptime assert (
            swapAB == False
        ), "swapAB is not supported for split-k kernel"
        # Dispatch to split-k kernel
        warp_specialize_gemm_with_multicasting_splitk[
            transpose_b=transpose_b,
            config=config,
            splits=splits,
            raster_order=raster_order,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](c_device, a_device, b_device, ctx)
    else:
        # Dispatch to regular kernel
        _warp_specialize_gemm_with_multicasting_impl[
            transpose_b=transpose_b,
            config=config,
            grid_shape=grid_shape,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            schedule=schedule,
            hilbert_swizzle=hilbert_swizzle,
            swapAB=swapAB,
        ](c_device, a_device, b_device, ctx)


def _warp_specialize_gemm_with_multicasting_impl[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
    swapAB: Bool = False,
](
    c_device: TileTensor[c_type, ...],
    a_device: TileTensor[a_type, ...],
    b_device: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    comptime assert c_device.rank == 2, "c must be rank 2"
    comptime assert a_device.rank == 2, "a must be rank 2"
    comptime assert b_device.rank == 2, "b must be rank 2"

    # Static shape from the TileTensor type-level layout.
    comptime N_static = type_of(c_device).LayoutType.static_shape[1]
    comptime K_static = type_of(a_device).LayoutType.static_shape[1]

    comptime assert not swapAB or (
        schedule == MatmulSchedule.NONE
    ), "swapAB does not support persistent kernels yet"
    comptime assert not swapAB or (
        hilbert_swizzle == False
    ), "swapAB does not support hilbert swizzle yet"
    comptime assert not swapAB or (
        use_tma_store == False
    ), "swapAB does not support TMA store yet"
    comptime assert (
        transpose_b == True
    ), "H100 matmul only supports transposed B"

    # C is in reference to A and B not being swapped
    # so we derive M and N from A and B instead
    var M = Int(b_device.dim[0]()) if swapAB else Int(a_device.dim[0]())
    var N = Int(a_device.dim[0]()) if swapAB else Int(b_device.dim[0]())
    var K = Int(a_device.dim[1]())

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]

    comptime k_group_size = config.k_group_size

    comptime assert (a_type == b_type == .float8_e4m3fn) or (
        a_type == b_type and a_type in (DType.bfloat16, DType.float32)
    ), "Unsupported input dtype"

    comptime assert (
        a_type != .float8_e4m3fn or BK == 128
    ), "BK must be 128 for fp8 data type for numerical accuracy correctness"

    comptime assert (
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None
    ), "Either the epilogue lambda or the compute lambda can be used"

    comptime assert BM > 64 or (
        BM == 64 and config.num_consumer == 1
    ), "Only support 1 consumer for BM=64"

    comptime k_align = find_K_alignment_upto_16B(K_static * size_of[a_type]())
    comptime assert k_align in (
        4,
        8,
        16,
    ), "H100 matmul K dim must be multiple of 4B"

    logger.info("Executing Warp Specialized Gemm with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

    comptime if schedule == MatmulSchedule.NONE:
        pass
    elif schedule == MatmulSchedule.DS_SCHEDULER:
        comptime assert (
            grid_shape is not None
        ), "Grid shape must be provided for DS scheduler"
        comptime ds_grid_shape = grid_shape.value()
        comptime assert (
            ds_grid_shape[0] <= H100.sm_count and ds_grid_shape[1] == 1
        ), "Deepseek scheduler only accepts grid shape with 1 column"

    elif grid_shape:
        comptime assert _is_valid_grid_shape[
            grid_shape.value(), config.cluster_shape
        ](ceildiv(N_static, BN)), String(
            "grid shape:",
            grid_shape.value(),
            "is not compatible with cluster shape:",
            config.cluster_shape,
            "and static N:",
            N_static,
            sep=" ",
        )

    comptime grid_shape_adjusted = grid_shape.value() if grid_shape else _get_grid_shape[
        config.cluster_shape
    ](
        ceildiv(N_static, BN)
    )

    comptime cluster_shape = StaticTuple[Int32, 3](
        Int32(config.cluster_shape[0]),
        Int32(config.cluster_shape[1]),
        Int32(config.cluster_shape[2]),
    )

    comptime CLUSTER_N = Int(cluster_shape[0])
    comptime CLUSTER_M = Int(cluster_shape[1])

    comptime c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
        k_group_size,
        swapAB,
    ]()

    # Compute C shared memory tile dimensions
    # For normal: rows stay full, cols are divided by num_consumer
    # For swapAB: rows are divided by num_consumer, cols stay full
    comptime c_smem_rows_reg = c_smem_layout.shape[0].value()
    comptime c_smem_cols_reg = c_smem_layout.shape[
        1
    ].value() // config.num_consumer

    comptime c_smem_rows_swapAB = c_smem_layout.shape[
        0
    ].value() // config.num_consumer
    comptime c_smem_cols_swapAB = c_smem_layout.shape[1].value()

    comptime c_smem_rows = c_smem_rows_reg if not swapAB else c_smem_rows_swapAB
    comptime c_smem_cols = c_smem_cols_reg if not swapAB else c_smem_cols_swapAB
    comptime c_smem_tile = Index(c_smem_rows, c_smem_cols)

    # BK is always 64
    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    comptime c_swizzle = TensorMapSwizzle(
        Int32(min(log2_floor(c_smem_tile[1] // 8), 3))
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    var c_tma_op = create_tma_tile_template[
        c_type,
        2,
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_shape=Index(c_smem_tile[0], c_smem_tile[1]),
    ]()

    comptime if use_tma_store:
        c_tma_op = create_tensor_tile[
            c_smem_tile,
            swizzle_mode=c_swizzle,
            __desc_shape=Index(c_smem_tile[0], c_smem_tile[1]),
        ](ctx, c_device)

    var lut_ptr = ctx.enqueue_create_buffer[.uint32](0)

    comptime if hilbert_swizzle:
        var grid_x = ceildiv(N, BN)
        var grid_y = ceildiv(M, BM)
        lut_ptr = get_hilbert_lut_with_cache(ctx, grid_x, grid_y)

    # one producer and num_consumer consumers each one warpgroup in size
    comptime num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE

    comptime matmul_kernel[
        a_type: DType,
        b_type: DType,
        c_type: DType,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        a_swizzle: TensorMapSwizzle,
        b_swizzle: TensorMapSwizzle,
        c_swizzle: TensorMapSwizzle,
        a_engine: TensorEngine = DefaultEngine[element_width=1],
        b_engine: TensorEngine = DefaultEngine[element_width=1],
        c_engine: TensorEngine = DefaultEngine[element_width=1],
        swapAB: Bool = False,
        hilbert_swizzle: Bool = False,
    ] = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        a_layout,
        b_layout,
        c_layout,
        c_smem_layout,
        config.block_tile_shape,
        config.mma_shape,
        cluster_shape,
        config.num_pipeline_stages,
        num_threads,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast=config.partitioned_multicast,
        use_tma_store=use_tma_store,
        promotion_frequency=1,
        pdl_level=config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        hilbert_swizzle=hilbert_swizzle,
        k_group_size=k_group_size,
        swapAB=swapAB,
        a_engine=a_engine,
        b_engine=b_engine,
        c_engine=c_engine,
    ]

    comptime matmul_kernel_regular[
        hilbert_swizzle: Bool = False
    ] = matmul_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_device).LayoutType,
        type_of(b_device).LayoutType,
        type_of(c_device).LayoutType,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        a_engine=type_of(a_device).Engine,
        b_engine=type_of(b_device).Engine,
        c_engine=type_of(c_device).Engine,
        swapAB=False,
        hilbert_swizzle=hilbert_swizzle,
    ]

    comptime matmul_kernel_swapAB = matmul_kernel[
        b_type,
        a_type,
        c_type,
        type_of(b_device).LayoutType,
        type_of(a_device).LayoutType,
        type_of(c_device).LayoutType,
        a_swizzle=b_swizzle,
        b_swizzle=a_swizzle,
        c_swizzle=c_swizzle,
        a_engine=type_of(b_device).Engine,
        b_engine=type_of(a_device).Engine,
        c_engine=type_of(c_device).Engine,
        swapAB=True,
        hilbert_swizzle=False,
    ]

    comptime smem_size = matmul_kernel_regular[].SMem.storage_size() if not swapAB else matmul_kernel_swapAB.SMem.storage_size()

    comptime assert (
        smem_size <= H100.shared_memory_per_multiprocessor - 1024
    ), "requested SMEM size exceeds 227KB limit."

    # TMA requires stride (K) multiple of 16B. If not satisfied,
    # we need to use cp.async.ca for 4B and 8B access, and ld for
    # 2B or smaller access.
    # Note that K * size_of[a_type]() decides the 2nd row's alignment
    # and Nvidia requires access alignment by access size.
    # Dispatch kernel using TMA load when the stride is multiple of 16B.
    comptime if k_align == 16:
        comptime if not swapAB:
            var a_tma_op = create_tensor_tile[
                Index(
                    BM // CLUSTER_N, BK
                ) if config.partitioned_multicast else Index(BM, BK),
                swizzle_mode=a_swizzle,
            ](ctx, a_device)

            var b_tma_op = create_tensor_tile[
                Index(
                    BN // CLUSTER_M, BK
                ) if config.partitioned_multicast else Index(BN, BK),
                swizzle_mode=b_swizzle,
            ](ctx, b_device)

            comptime if schedule != MatmulSchedule.NONE:
                comptime kernel = matmul_kernel_regular[].run_persistent[
                    type_of(a_tma_op).rank,
                    type_of(b_tma_op).rank,
                    type_of(c_tma_op).rank,
                    type_of(a_tma_op).tile_shape,
                    type_of(b_tma_op).tile_shape,
                    type_of(c_tma_op).tile_shape,
                    type_of(a_tma_op).desc_shape,
                    type_of(b_tma_op).desc_shape,
                    type_of(c_tma_op).desc_shape,
                    grid_shape=grid_shape_adjusted,
                    schedule=schedule,
                    c_tensor_layout=type_of(c_device).LayoutType,
                ]

                ctx.enqueue_function[kernel](
                    a_tma_op,
                    b_tma_op,
                    c_tma_op,
                    c_device.as_unsafe_any_origin(),
                    Index(M, N, K),
                    grid_dim=(grid_shape_adjusted[0], grid_shape_adjusted[1]),
                    block_dim=(num_threads),
                    shared_mem_bytes=smem_size,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_size)
                    ),
                    attributes=pdl_launch_attributes(config.pdl_level()),
                )
            else:
                comptime kernel = matmul_kernel_regular[
                    hilbert_swizzle=hilbert_swizzle
                ].run[
                    type_of(a_tma_op).rank,
                    type_of(b_tma_op).rank,
                    type_of(c_tma_op).rank,
                    type_of(a_tma_op).tile_shape,
                    type_of(b_tma_op).tile_shape,
                    type_of(c_tma_op).tile_shape,
                    type_of(a_tma_op).desc_shape,
                    type_of(b_tma_op).desc_shape,
                    type_of(c_tma_op).desc_shape,
                    type_of(a_device).LayoutType,
                    type_of(b_device).LayoutType,
                    type_of(c_device).LayoutType,
                ]

                ctx.enqueue_function[kernel](
                    a_tma_op,
                    b_tma_op,
                    c_tma_op,
                    a_device.as_immut().as_unsafe_any_origin(),
                    b_device.as_immut().as_unsafe_any_origin(),
                    c_device.as_unsafe_any_origin(),
                    lut_ptr,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(num_threads),
                    shared_mem_bytes=smem_size,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_size)
                    ),
                    attributes=pdl_launch_attributes(config.pdl_level()),
                )
        else:
            var a_tma_op = create_tensor_tile[
                Index(
                    BM // CLUSTER_N, BK
                ) if config.partitioned_multicast else Index(BM, BK),
                swizzle_mode=a_swizzle,
            ](ctx, b_device)

            var b_tma_op = create_tensor_tile[
                Index(
                    BN // CLUSTER_M, BK
                ) if config.partitioned_multicast else Index(BN, BK),
                swizzle_mode=b_swizzle,
            ](ctx, a_device)

            comptime if schedule == MatmulSchedule.NONE:
                comptime kernel = matmul_kernel_swapAB.run[
                    type_of(a_tma_op).rank,
                    type_of(b_tma_op).rank,
                    type_of(c_tma_op).rank,
                    type_of(a_tma_op).tile_shape,
                    type_of(b_tma_op).tile_shape,
                    type_of(c_tma_op).tile_shape,
                    type_of(a_tma_op).desc_shape,
                    type_of(b_tma_op).desc_shape,
                    type_of(c_tma_op).desc_shape,
                    type_of(b_device).LayoutType,
                    type_of(a_device).LayoutType,
                    type_of(c_device).LayoutType,
                ]

                ctx.enqueue_function[kernel](
                    a_tma_op,
                    b_tma_op,
                    c_tma_op,
                    b_device.as_immut().as_unsafe_any_origin(),
                    a_device.as_immut().as_unsafe_any_origin(),
                    c_device.as_unsafe_any_origin(),
                    lut_ptr,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(num_threads),
                    shared_mem_bytes=smem_size,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(smem_size)
                    ),
                    attributes=pdl_launch_attributes(config.pdl_level()),
                )

    # Dispatch kernel using cp.async.ca when the stride is not multiple of 4B or 8B..
    else:
        # TODO add support for swapAB
        comptime assert (
            swapAB == False
        ), "swapAB is not supported for unaligned kernel"
        comptime kernel = matmul_kernel_regular[].run_unaligned[
            type_of(c_tma_op).rank,
            type_of(c_tma_op).tile_shape,
            type_of(c_tma_op).desc_shape,
        ]

        ctx.enqueue_function[kernel](
            c_tma_op,
            a_device.as_immut().as_unsafe_any_origin(),
            b_device.as_immut().as_unsafe_any_origin(),
            c_device.as_unsafe_any_origin(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(num_threads),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_size)
            ),
            attributes=pdl_launch_attributes(config.pdl_level()),
        )


def _get_c_smem_layout[
    block_tile_shape: IndexList[3],
    a_type: DType,
    b_type: DType,
    c_type: DType,
    num_pipeline_stages: Int,
    k_group_size: Int,
    swapAB: Bool = False,
]() -> Layout:
    comptime BM: Int = block_tile_shape[0]
    comptime BN: Int = block_tile_shape[1]
    comptime BK: Int = block_tile_shape[2]

    comptime WG_BM = BM
    comptime MAX_WG_BN = 128  # a cap on the shared memory size

    comptime available_smem_size: Int = H100.shared_memory_per_multiprocessor - 1024

    comptime assert not swapAB or (
        a_type == b_type == c_type == .bfloat16
    ), "swapAB is only supported for bfloat16 dtypes"

    comptime groups = num_pipeline_stages // k_group_size
    comptime barrier_size = size_of[Int64]() * 2 * groups
    comptime total_smem_tile_size = num_pipeline_stages * (
        BM * BK * size_of[a_type]() + BN * BK * size_of[b_type]()
    )

    comptime pipeline_smem_size = total_smem_tile_size + barrier_size

    comptime available_c_smem_size = available_smem_size - pipeline_smem_size

    # In the normal case Shared Memory M will be the same as BM which can be either 64 or 128
    # This value is derived from the MMA_M shape which is fixed to 64. Similarly BN is the same as
    # MMA_N (8 -> 256), however this poses a problem. At worst our shared memory size would be 128x256
    # this leaves little shared memory for other resources. To solve this we set the max shared memory N to 128, and
    # try to minimize it as much as possible.

    # We can't make Shared Memory N 1, since we would like to use stmatrix. stmatrix transports
    # matrices of sizes of 16bytes by 16bytes, and we need to also be able to use TMA. The lowest
    # TMA swizzle is 16 bytes. So we set the minimum shared memory N to 16.

    # In the SwapAB case Shared Memory N will be fixed to 64 or 128 based on the value of BN.
    # This meets the TMA requirements but we need m to be at least 16 to be able to use stmatrix.
    comptime min_wg_bn = 16
    comptime MIN_WG_BN = min_wg_bn if size_of[c_type]() == 2 else BN // 4

    comptime assert available_smem_size > (
        pipeline_smem_size + (WG_BM * MIN_WG_BN * size_of[c_type]())
    ), (
        "There is not enough SMEM to fit the pipeline yet alone the"
        " output tile!"
        + " available_smem_size: "
        + String(available_smem_size)
        + " pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type](): "
        + String(pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type]())
    )

    def _get_max_wg_bn() -> Int:
        var WG_BN = MAX_WG_BN
        while (
            available_c_smem_size < WG_BM * WG_BN * size_of[c_type]()
            or BN % WG_BN != 0
        ) and WG_BN > MIN_WG_BN:
            WG_BN //= 2
        return WG_BN

    comptime max_wg_bn = _get_max_wg_bn()
    return Layout.row_major(max_wg_bn, WG_BM) if swapAB else Layout.row_major(
        WG_BM, max_wg_bn
    )


def warp_specialize_gemm_with_multicasting_splitk[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    splits: Int,
    raster_order: RasterOrder,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
](
    c_device: TileTensor[c_type, ...],
    a_device: TileTensor[a_type, ...],
    b_device: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    """Dispatches an SM90 warp-specialized split-K matmul that partitions the K dimension across thread blocks and reduces partial sums via a workspace buffer and locks.

    Allocates a workspace accumulation buffer and a locks buffer, builds a
    `SplitKTileScheduler` to map tiles to thread blocks, and enqueues the
    split-K kernel variant of `HopperMatmulSM90Kernel`. Supports optional TMA
    stores and epilogue or compute lambda functions.

    Parameters:
        c_type: Element type of the output `C` tensor.
        a_type: Element type of the input `A` tensor.
        b_type: Element type of the input `B` tensor.
        transpose_b: Whether `B` is stored transposed. Must be `True` on H100.
        config: Kernel configuration holding block tile shape, cluster shape,
            MMA shape, pipeline stages, and consumer count.
        splits: Number of partitions along `K`. Each partition is computed by a
            separate thread block and reduced via a workspace buffer.
        raster_order: Tile rasterization direction, `AlongM` or `AlongN`.
        use_tma_store: Whether to use TMA for storing the output tile to global
            memory (defaults to `False`).
        elementwise_lambda_fn: Optional epilogue function applied to the output
            tile after the matmul (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute lambda fused into the
            MMA loop (defaults to `None`). Mutually exclusive with
            `elementwise_lambda_fn`.

    Args:
        c_device: Output `C` tensor on device with shape `(M, N)`.
        a_device: Input `A` tensor on device with shape `(M, K)`.
        b_device: Input `B` tensor on device with shape `(N, K)` when
            `transpose_b` is `True`.
        ctx: Device context used to enqueue the kernel.
    """

    comptime assert c_device.rank == 2, "c must be rank 2"
    comptime assert a_device.rank == 2, "a must be rank 2"
    comptime assert b_device.rank == 2, "b must be rank 2"

    var M = c_device.dim[0]()
    # Static shape from the TileTensor type-level layout.
    comptime N = type_of(c_device).LayoutType.static_shape[1]
    comptime K = type_of(a_device).LayoutType.static_shape[1]

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime k_group_size = config.k_group_size

    comptime assert k_group_size == 1, "Only support k_group_size == 1 for now"

    comptime assert (a_type == b_type == .float8_e4m3fn) or (
        a_type == b_type and a_type in (DType.bfloat16, DType.float32)
    ), "Unsupported input dtype"

    comptime assert (
        a_type != .float8_e4m3fn or BK == 128
    ), "BK must be 128 for fp8 data type for numerical accuracy correctness"

    comptime assert (
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None
    ), "Either the epilogue lambda or the compute lambda can be used"

    comptime assert BM > 64 or (
        BM == 64 and config.num_consumer == 1
    ), "Only support 1 consumer for BM=64"

    logger.info("Executing Split-K Warp Specialized GEMM with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

    comptime cluster_shape = StaticTuple[Int32, 3](
        Int32(config.cluster_shape[0]),
        Int32(config.cluster_shape[1]),
        Int32(config.cluster_shape[2]),
    )

    comptime CLUSTER_N = Int(cluster_shape[0])
    comptime CLUSTER_M = Int(cluster_shape[1])

    comptime c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
        k_group_size,
    ]()
    comptime c_smem_tile = Index(
        c_smem_layout.shape[0].value(),
        c_smem_layout.shape[1].value() // config.num_consumer,
    )

    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    comptime c_swizzle = TensorMapSwizzle(
        Int32(min(log2_floor(c_smem_tile[1] // 8), 3))
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    var a_tma_op = create_tensor_tile[
        Index(BM // CLUSTER_N, BK) if config.partitioned_multicast else Index(
            BM, BK
        ),
        swizzle_mode=a_swizzle,
    ](ctx, a_device)
    var b_tma_op = create_tensor_tile[
        Index(BN // CLUSTER_M, BK) if config.partitioned_multicast else Index(
            BN, BK
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b_device)

    var c_tma_op = create_tensor_tile[
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_shape=Index(c_smem_tile[0], c_smem_tile[1]),
    ](ctx, c_device)

    comptime scheduler = SplitKTileScheduler[
        locks_origin=MutAnyOrigin,
        Index(N, K),
        config.block_tile_shape,
        UInt32(splits),
        UInt32(config.num_consumer),
        UInt32(config.num_pipeline_stages),
        Index(config.cluster_shape[1], config.cluster_shape[0]),
        raster_order,
    ]

    var launch_grid_shape = scheduler.get_grid_shape(
        config.cluster_shape,
        raster_order,
    )

    comptime accum_type = DType.float32  # fix this

    var NUM_TILES = scheduler.get_num_tiles(
        Index(M, N, K),
        config.block_tile_shape,
        Index(config.cluster_shape[1], config.cluster_shape[0]),
    )

    var workspace_data = ctx.enqueue_create_buffer[accum_type](
        NUM_TILES * BM * BN
    )
    var workspace_ptr = workspace_data.unsafe_ptr()

    var locks_buffer_size_bytes = (
        scheduler.get_required_locks_buffer_size_bytes[
            accum_type, UInt32(config.num_consumer)
        ](
            Index(M, N, K),
            config.block_tile_shape,
            Index(CLUSTER_M, CLUSTER_N),
        )
    )

    var locks_ptr = ctx.enqueue_create_buffer[.uint8](locks_buffer_size_bytes)

    ctx.enqueue_memset(locks_ptr, 0)

    comptime num_threads = config.num_consumer * 128 + 128

    comptime matmul_kernel = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_device).LayoutType,
        type_of(b_device).LayoutType,
        type_of(c_device).LayoutType,
        c_smem_layout,
        config.block_tile_shape,
        config.mma_shape,
        cluster_shape,
        config.num_pipeline_stages,
        Int(num_threads),
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast=config.partitioned_multicast,
        use_tma_store=use_tma_store,
        promotion_frequency=1,
        pdl_level=config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        a_engine=type_of(a_device).Engine,
        b_engine=type_of(b_device).Engine,
        c_engine=type_of(c_device).Engine,
    ]

    comptime smem_size = matmul_kernel.SMem.storage_size()

    comptime assert (
        smem_size <= H100.shared_memory_per_multiprocessor - 1024
    ), "requested SMEM size exceeds 227KB limit."

    comptime kernel = matmul_kernel.run_splitk[
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(c_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        type_of(c_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).desc_shape,
        splits=splits,
        raster_order=raster_order,
        c_tensor_layout=type_of(c_device).LayoutType,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c_device.as_unsafe_any_origin(),
        workspace_ptr,
        locks_ptr,
        Index(M, N, K),
        grid_dim=(
            launch_grid_shape[0],
            launch_grid_shape[1],
            launch_grid_shape[2],
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
        attributes=pdl_launch_attributes(config.pdl_level()),
    )

    _ = workspace_data^
    _ = locks_ptr^
