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


"""Provides dispatch logic for SM100 block-scaled (NVFP4, MXFP4, MXFP8) matmul kernels with optional elementwise epilogue."""


from std.math import ceildiv
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives.grid_controls import PDLLevel
from layout import Coord, Idx, Layout, TileTensor, row_major
from layout.tile_tensor import NullableTileTensor
from std.logger import Logger
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    get_scaling_kind,
)
from max.gpu.host.info import _is_sm10x_gpu
from std.collections import Optional
from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)
from std.utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from std.memory import UnsafePointer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.sys import size_of, simd_width_of
from max.algorithm import elementwise
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_nvfp4,
    _get_tuning_list_sm100_mxfp4,
    _get_tuning_list_sm100_mxfp8,
)
from internal_utils import Table
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    build_block_scaled_configs,
    choose_block_scaled_config,
)

comptime logger = Logger()

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
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
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an SM100 block-scaled matmul by selecting a tuning config from
    per-format outlier tables for specific M ranges, falling back to a
    small-BN config for GEMVs (`m == 1`) and a heuristic config table for the
    remaining cases. Returns `DISPATCH_HIT` when a matching config is found and
    launched, or `DISPATCH_MISS` when no config matches.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the LHS input tensor `a` (inferred).
        b_type: Element type of the RHS input tensor `b` (inferred).
        scales_dtype: Element type of the per-block scale tensors
            `a_scales` and `b_scales` (inferred).
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
            Must match the format: 16 for NVFP4, 32 for MXFP4, or 32
            for MXFP8.
        transpose_b: Whether `b` is stored transposed. Must be `True`
            (defaults to `True`).
        elementwise_lambda_fn: Optional epilogue applied to the matmul
            result `c` in a separate kernel after the matmul completes
            (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute function fused
            into the matmul kernel epilogue (defaults to `None`).
        pdl_level: Programmatic Dependent Launch scheduling level for
            overlapping this kernel with prior GPU work (defaults to
            `PDLLevel()`).

    Args:
        c: Output TileTensor accumulating the matmul result.
        a: LHS input TileTensor.
        b: RHS input TileTensor (must be transposed).
        a_scales: Per-block scales for `a`.
        b_scales: Per-block scales for `b`.
        tensor_sf: Global tensor scaling factor applied as `alpha`.
        ctx: Device context used to launch the kernel.

    Returns:
        `DISPATCH_HIT` if a config was selected and the kernel launched, otherwise `DISPATCH_MISS`.
    """
    var m = Int(c.dim[0]())

    comptime scaling_kind = get_scaling_kind[
        a_type, scales_dtype, SF_VECTOR_SIZE, b_type
    ]()
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
    comptime num_k_iters = ceildiv(a.static_shape[1], BK)
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1] * 2 if is_fp4 else a.static_shape[1]

    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        (
            scaling_kind == UMMAKind.KIND_MXF4NVF4
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF4
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF8F6F4
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        )
    ), "Only support NVFP4, MXFP4, or MXFP8 scale/dtype combinations."

    comptime assert (
        a_scales.static_shape[1] == b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    comptime outliers = Table(
        _get_tuning_list_sm100_nvfp4(), "nvfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4NVF4 else Table(
        _get_tuning_list_sm100_mxfp4(), "mxfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4 else Table(
        _get_tuning_list_sm100_mxfp8(), "mxfp8_heuristic_outliers"
    )

    comptime outlier_configs = outliers.find(
        rule=lambda (x: TuningConfigSM100) -> Bool: x.K == static_K
        and x.N == static_N
    )

    comptime for tuning_config in outlier_configs:
        if m >= tuning_config.M and m < tuning_config.M_end:
            comptime matmul_config = BlockScaledMatmulConfig[
                a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
            ](
                scaling_kind=scaling_kind,
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
                num_pipeline_stages=Optional(
                    tuning_config.num_pipeline_stages
                ) if tuning_config.num_pipeline_stages
                > 0 else None,
                is_small_bn=tuning_config.is_small_bn,
            )

            logger.info("Using tuning config: ", matmul_config)

            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=matmul_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, a_scales, b_scales, tensor_sf, ctx)

            return DISPATCH_HIT

    # Dispatch to the small-BN kernel for the small-M decode regime (m <= 16:
    # m == 1 GEMV plus small-batch / speculative-decode m ~ 8, 16). It is
    # optimized for skinny GEMMs; MMA_N=8 tiles the M dim (m=16 -> 2 tiles).
    # Larger M keeps the cta_group=2 prefill heuristic below.
    if m <= 16:
        # Larger k-groups shorten the mainloop for this latency-bound skinny
        # decode GEMM (K=6144 -> 48 k-iters; kg=4 -> 12 groups vs 24), which an
        # isolated B200 sweep at the served QKV shapes (N in {2304,2560}, K=6144,
        # M<=16) showed is ~8% faster than kg=2 and ~40% faster than vendor
        # cuBLASLt, bit-exact. Needs num_pipeline_stages % kg == 0 (12 % 4 == 0).
        comptime k_group_size = (
            4 if num_k_iters % 4 == 0 else (2 if num_k_iters % 2 == 0 else 1)
        )
        comptime config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=scaling_kind,
            cta_group=1,
            mma_shape=Index(128, 8, 32),
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            num_accum_pipeline_stages=1,
            k_group_size=k_group_size,
            num_clc_pipeline_stages=0,
            AB_swapped=True,
            is_small_bn=True,
        )
        _block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, a_scales, b_scales, tensor_sf, ctx)

        logger.info("Using small-BN config: ", config)
        return DISPATCH_HIT

    comptime configs = build_block_scaled_configs[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        scales_dtype,
        static_N,
        static_K,
        transpose_b,
    ]()
    var config_runtime = choose_block_scaled_config[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](m, static_N, static_K)

    comptime for config in configs:
        if config_runtime == config:
            logger.info("Using heuristic config: ", config)
            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, a_scales, b_scales, tensor_sf, ctx)
            return DISPATCH_HIT

    return DISPATCH_MISS


########################################################
# SM100 Block Scaled matmul with normal epilogue kernel dispatch
########################################################


def _block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Launch the SM100 block-scaled matmul, fusing the elementwise epilogue
    in-kernel.

    When an `elementwise_lambda_fn` is supplied the matmul stores are redirected
    through it inside the kernel (TMEM -> registers -> lambda, no scratch
    round-trip): `blackwell_block_scaled_matmul_tma_umma_warp_specialized`
    threads the lambda into both its main and small-BN paths, where
    `TileWriter` fires the store-redirect epilogue. When it is `None` the kernel
    performs its default store. Callers must still allocate `c`; the
    default-store path writes it, and it defines the (row, col) coordinate space
    the lambda addresses.
    """

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    if m == 0 or n == 0:
        return

    comptime K_phys = a.static_shape[1]

    # The kernel fuses the elementwise epilogue in-kernel across all regimes
    # (`TileWriter` fires the store-redirect on both the main and small-BN
    # paths), so redirect the store through `elementwise_lambda_fn` directly
    # rather than falling back to a separate elementwise pass at small M.
    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        K=K_phys,
        config=config,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        pdl_level=pdl_level,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
        alpha=tensor_sf,
    )


def _vendor_blas_block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NullableTileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "Only support NVFP4_SF_DTYPE (float8_e4m3fn) for scales for now."

    comptime assert SF_VECTOR_SIZE in (
        NVFP4_SF_VECTOR_SIZE,
    ), "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4)"

    comptime assert (
        a_scales.static_shape[1] == b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    if m == 0 or n == 0:
        return

    comptime if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated!"

        matmul(
            ctx,
            c.value(),
            a,
            b,
            a_scales=a_scales,
            b_scales=b_scales,
            transpose_b=True,
            c_row_major=True,
            alpha=tensor_sf,
        )
    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # Nvidia GPUs >= sm_100 arch support 32B load/store to global memory.
        comptime use_32b_simd = True
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        # If c is already allocated, we can just use the sm100 blockwise scaled fp8 matmul and
        # apply the epilogue.
        if c.ptr:
            var c_tt = c.value()

            def epilogue_wrapper[
                simd_width: Int, alignment: Int = 1
            ](idx: Coord) {var}:
                var c_val = rebind[SIMD[c_type, simd_width]](
                    c_tt.load[width=simd_width](idx)
                )
                epilogue[c_type, simd_width, alignment=alignment](
                    Index(idx[0].value(), idx[1].value()), c_val
                )

            matmul(
                ctx,
                c_tt,
                a,
                b,
                a_scales=a_scales,
                b_scales=b_scales,
                alpha=tensor_sf,
                transpose_b=True,
                c_row_major=True,
            )
            elementwise[simd_size, target="gpu"](epilogue_wrapper, (m, n), ctx)
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](
            c.num_elements()
        )
        var c_tmp = TileTensor(tmp_device_buffer, c.layout)

        _vendor_blas_block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tmp,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
        )
