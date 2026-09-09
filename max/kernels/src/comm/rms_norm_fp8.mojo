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
"""Fused RMSNorm + FP8 quantization kernel.

Provides the fused RMSNorm + FP8 quantization primitive used by both the
standalone normalization layer and the fused allreduce + RMSNorm + FP8 kernel.
Lives in comm/ so that allreduce_residual_rmsnorm can depend on it without
introducing a comm → nn → comm circular dependency.
"""

from std.math import rsqrt
from std.sys import align_of, simd_width_of
from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from max.gpu import WARP_SIZE, block_idx, thread_idx
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from std.utils import IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from max.runtime.tracing import Trace, TraceLevel, trace_arg

from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize


@always_inline
def block_reduce_sum_and_max[
    dtype: DType, max_warps_per_block: Int
](sum_val: Scalar[dtype], max_val: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """Combined block reduction for sum and max in a single barrier pass.

    Performs both sum and max reductions across the block using only 2
    barriers (vs 4 for separate block.sum + block.max with broadcast).
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx == 0:
            return warp.sum(v)
        else:
            return warp.max(v)

    var results = block._block_reduce[
        max_warps_per_block * WARP_SIZE,
        warp_reduce_fn=_reduce_fn,
        broadcast=True,
    ](
        StaticTuple[Scalar[dtype], 2](sum_val, max_val),
        initial_vals=StaticTuple[Scalar[dtype], 2](0, Scalar[dtype].MIN_FINITE),
    )
    return (results[0], results[1])


@always_inline
def rms_norm_fused_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        in_dtype, width
    ],
    /,
    target: StaticString = "gpu",
    compile_only: Bool = False,
](
    shape: IndexList[rank],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
) raises:
    """Fused RMSNorm + FP8 quantization kernel (TileTensor overload).

    Computes RMSNorm normalization and quantizes the output to FP8 format in a
    single pass. This is the primary implementation that operates on TileTensor
    inputs.

    Parameters:
        in_dtype: Input data type (float32, float16, or bfloat16).
        out_dtype: Output FP8 data type (float8_e4m3fn or float8_e4m3fnuz).
        scales_dtype: Data type for scale factors (bfloat16, float16, or
            float32).
        rank: Tensor rank.
        input_fn: Function to load input values.
        target: Target device ("gpu" or "cpu").
        compile_only: If True, only compiles the kernel without executing it.
            Used to pre-compile kernels and avoid JIT compilation deadlocks
            in multi-GPU contexts.

    Args:
        shape: Input tensor shape.
        output: Output TileTensor to write FP8 quantized values.
        gamma: RMSNorm scale parameter (rank 1).
        epsilon: Small constant for numerical stability.
        weight_offset: Offset to add after normalization.
        ctx: Device context.
        scale_ub: Upper bound for dynamic scale factor to limit the scale
            value.
        scale_output: TileTensor to write per-row dynamic scales.
    """
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert (
        scale_output.rank == rank
    ), "scale_output.rank must be the same as rank"
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert in_dtype in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
    ), "input dtype should be float16, bfloat16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    # Tracing for performance profiling
    @always_inline
    def description_fn() {imm} -> String:
        return (
            trace_arg("input", shape, in_dtype)
            + " -> "
            + trace_arg("output", shape, out_dtype)
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_fp8",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        if target == "gpu":
            _rms_norm_fused_fp8_gpu[
                in_dtype,
                out_dtype,
                scales_dtype,
                rank,
                input_fn,
                compile_only=compile_only,
            ](
                shape,
                output,
                gamma,
                epsilon,
                weight_offset,
                scale_ub,
                scale_output,
                ctx,
            )
        else:
            raise Error("CPU implementation not yet supported")


@always_inline
def _rms_norm_fused_fp8_gpu[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        in_dtype, width
    ],
    compile_only: Bool = False,
](
    shape: IndexList[rank],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    ctx: DeviceContext,
) raises:
    """GPU dispatcher for fused RMSNorm + FP8 quantization."""

    if rank == 0:
        return

    var last_dim = shape[rank - 1]
    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    # Create 2D input function (following rms_norm_fused_residual_add pattern)
    @__parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[in_dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank](indices.canonicalize())

    # Create 2D output TileTensor view
    var output_2d = TileTensor(output._storage, row_major(Coord(rows, cols)))

    # Create 1D view of scale_output for internal kernel use
    var scale_output_1d = TileTensor(
        scale_output._storage, row_major(Coord(rows))
    )

    # Dispatch based on column count (following rms_norm_gpu pattern)
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime base_simd_width = simd_width_of[
        in_dtype, target=get_gpu_target()
    ]()

    # Dispatch: select SIMD width and kernel strategy based on column count
    @__parameter
    def launch[sw: Int, warp_tiling: Bool]() raises:
        _rms_norm_fused_fp8_gpu_launch[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            input_fn_2d,
            use_warp_tiling=warp_tiling,
            compile_only=compile_only,
        ](
            rows,
            cols,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            scale_ub,
            scale_output_1d,
            ctx,
        )

    if cols % base_simd_width == 0:
        if cols <= (WARP_SIZE * base_simd_width * max_warps_per_block):
            launch[base_simd_width, True]()
        elif (
            cols <= (WARP_SIZE * (base_simd_width * 2) * max_warps_per_block)
            and cols % (base_simd_width * 2) == 0
        ):
            launch[base_simd_width * 2, True]()
        else:
            launch[base_simd_width, False]()
    else:
        launch[1, False]()


@__name(t"rms_norm_fused_fp8_warp_tiling_{in_dtype}_{out_dtype}")
def _rms_norm_fused_fp8_kernel_warp_tiling[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ScaleEngine: TensorEngine,
    //,
    simd_width: Int,
    threads_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    output_fn: def[width: Int](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    gamma: TileTensor[in_dtype, LayoutType, origin, Engine=Engine],
    scale_buffer: TileTensor[
        mut=True,
        scales_dtype,
        ScaleLayoutType,
        scale_origin,
        Engine=ScaleEngine,
    ],
    epsilon: Float32,
    weight_offset: Float32,
    cols: Int32,
    scale_ub: Float32,
):
    """GPU kernel for fused RMSNorm + FP8 with warp-tiling - optimized like standalone RMS norm.

    This kernel always multiplies by gamma before quantizing to FP8.
    """
    var _cols = Int(cols)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var row = block_idx.x
    var tid = thread_idx.x
    var idx = tid * simd_width

    # Helper: Load gamma and apply to value (shared between both kernel variants)
    @always_inline
    @__copy_capture(gamma, _weight_offset)
    @__parameter
    def apply_gamma[
        width: Int
    ](val: SIMD[accum_type, width], col: Int) -> SIMD[accum_type, width]:
        var gamma_val = gamma.load[width=width, alignment=align](Coord(col))
        var gamma_accum = (
            gamma_val.cast[accum_type]() + _weight_offset.cast[accum_type]()
        )
        return val * gamma_accum

    var vec_data = SIMD[accum_type, simd_width](0)
    var is_valid = idx < _cols

    with PDL():
        # Phase 1: Load input ONCE, compute mean square AND max(|gamma*x|).
        # Key insight: max(|gamma*x*norm_factor|) = max(|gamma*x|) * norm_factor
        # since norm_factor is a positive scalar. Computing both sum(x^2) and
        # max(|gamma*x|) in the load phase allows both reductions to be issued
        # back-to-back, eliminating the compute gap between the two barriers.
        var thread_m2 = Scalar[accum_type](0)
        var thread_abs_max_gamma_x = Scalar[accum_type](0)

        if is_valid:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()
            thread_m2 = (vec_data**2).reduce_add()
            # Compute |gamma * x| for max finding (temporary, not stored)
            thread_abs_max_gamma_x = abs(
                apply_gamma[simd_width](vec_data, idx)
            ).reduce_max()

        # Combined reduction: sum for m2 and max for scaling in a single
        # 2-barrier pass (vs 4 barriers for separate block.sum + block.max).
        comptime max_warps = threads_per_block // WARP_SIZE
        var row_m2, row_abs_max_gamma_x = block_reduce_sum_and_max[
            max_warps_per_block=max_warps
        ](thread_m2, thread_abs_max_gamma_x)

        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](_cols)) + _epsilon.cast[accum_type]()
        )

        # Derive max of normalized values: max(|gamma*x|) * norm_factor
        var row_max = row_abs_max_gamma_x * norm_factor

        var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
            out_dtype
        ](row_max, _scale_ub)
        if tid == 0:
            scale_buffer[row] = scale_factor

        # Phase 2: Normalize (preserving original FP order), quantize, write
        if is_valid:
            var normalized = apply_gamma[simd_width](
                vec_data * norm_factor, idx
            )
            var output_fp8 = fp8_quantize[out_dtype](
                normalized, scale_factor_recip
            )
            output_fn[simd_width](row, idx, output_fp8)


def _rms_norm_fused_fp8_gpu_launch[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    use_warp_tiling: Bool,
    compile_only: Bool = False,
](
    rows: Int,
    cols: Int,
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Unified kernel launcher for fused RMSNorm + FP8 quantization.

    Selects between warp-tiling and block-tiling kernels at compile time.
    """

    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE

    var grid_dim = rows
    var block_dim = threads_per_block

    @always_inline
    @__parameter
    @__copy_capture(output)
    def output_fn[width: Int](row: Int, col: Int, val: SIMD[out_dtype, width]):
        """Write output to buffer."""
        output.store_linear[width=width](IndexList[2](row, col), val)

    comptime if use_warp_tiling:
        comptime kernel = _rms_norm_fused_fp8_kernel_warp_tiling[
            mut=gamma.mut,
            origin=gamma.origin,
            LayoutType=gamma.LayoutType,
            Engine=gamma.Engine,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            scales_dtype=scales_dtype,
            scale_origin=scale_output.origin,
            ScaleLayoutType=scale_output.LayoutType,
            ScaleEngine=scale_output.Engine,
            simd_width=simd_width,
            threads_per_block=threads_per_block,
            input_fn=input_fn,
            output_fn=output_fn,
        ]
        comptime if compile_only:
            _ = ctx.compile_function[kernel]()
        else:
            ctx.enqueue_function[kernel](
                gamma,
                scale_output,
                epsilon.cast[.float32](),
                weight_offset.cast[.float32](),
                Int32(cols),
                Float32(scale_ub),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
    else:
        comptime kernel = _rms_norm_fused_fp8_kernel_block[
            mut=gamma.mut,
            origin=gamma.origin,
            LayoutType=gamma.LayoutType,
            Engine=gamma.Engine,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            scales_dtype=scales_dtype,
            scale_origin=scale_output.origin,
            ScaleLayoutType=scale_output.LayoutType,
            ScaleEngine=scale_output.Engine,
            simd_width=simd_width,
            threads_per_block=threads_per_block,
            input_fn=input_fn,
            output_fn=output_fn,
        ]
        comptime if compile_only:
            _ = ctx.compile_function[kernel]()
        else:
            ctx.enqueue_function[kernel](
                gamma,
                scale_output,
                epsilon.cast[.float32](),
                weight_offset.cast[.float32](),
                Int32(cols),
                Float32(scale_ub),
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )


@__name(t"rms_norm_fused_fp8_block_{in_dtype}_{out_dtype}")
def _rms_norm_fused_fp8_kernel_block[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ScaleEngine: TensorEngine,
    //,
    simd_width: Int,
    threads_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    output_fn: def[width: Int](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    gamma: TileTensor[in_dtype, LayoutType, origin, Engine=Engine],
    scale_buffer: TileTensor[
        mut=True,
        scales_dtype,
        ScaleLayoutType,
        scale_origin,
        Engine=ScaleEngine,
    ],
    epsilon: Float32,
    weight_offset: Float32,
    cols: Int32,
    scale_ub: Float32,
):
    """GPU kernel for fused RMSNorm + FP8 with block-tiling - optimized version.

    This kernel always multiplies by gamma before quantizing to FP8.
    """
    var _cols = Int(cols)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var row = block_idx.x
    var tid = thread_idx.x

    # Helper: Load gamma and apply to value (same as warp-tiling variant)
    @always_inline
    @__copy_capture(gamma, _weight_offset)
    @__parameter
    def apply_gamma[
        width: Int
    ](val: SIMD[accum_type, width], col: Int) -> SIMD[accum_type, width]:
        var gamma_val = gamma.load[width=width, alignment=align](Coord(col))
        var gamma_accum = (
            gamma_val.cast[accum_type]() + _weight_offset.cast[accum_type]()
        )
        return val * gamma_accum

    with PDL():
        # Phase 1: Compute mean square AND max(|gamma*x|) in a single pass.
        # Key insight: max(|gamma*x*norm_factor|) = max(|gamma*x|) * norm_factor
        # since norm_factor is a positive scalar. This lets us derive both
        # norm_factor and the FP8 scale from a single pass over the input data.
        var thread_m2 = Scalar[accum_type](0)
        var thread_abs_max_gamma_x = Scalar[accum_type](0)

        for col_offset in range(0, _cols, threads_per_block * simd_width):
            var col = col_offset + tid * simd_width
            if col < _cols:
                var vec_data = input_fn[simd_width](row, col).cast[accum_type]()
                thread_m2 += (vec_data**2).reduce_add()
                # Compute |gamma * x| to find max for FP8 scaling
                var gamma_x = apply_gamma[simd_width](vec_data, col)
                thread_abs_max_gamma_x = max(
                    thread_abs_max_gamma_x, abs(gamma_x).reduce_max()
                )

        # Combined reduction: sum for m2 and max for scaling in a single
        # 2-barrier pass (vs 4 barriers for separate block.sum + block.max).
        comptime max_warps = threads_per_block // WARP_SIZE
        var row_m2, row_abs_max_gamma_x = block_reduce_sum_and_max[
            max_warps_per_block=max_warps
        ](thread_m2, thread_abs_max_gamma_x)

        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](_cols)) + _epsilon.cast[accum_type]()
        )

        # Derive max of normalized values: max(|gamma*x|) * norm_factor
        var row_max = row_abs_max_gamma_x * norm_factor

        # Compute scale factor
        var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
            out_dtype
        ](row_max, _scale_ub)

        # Write scale
        if tid == 0:
            scale_buffer[row] = scale_factor

        # Phase 2: Normalize, quantize and write output
        for col_offset in range(0, _cols, threads_per_block * simd_width):
            var col = col_offset + tid * simd_width
            if col < _cols:
                var vec_data = input_fn[simd_width](row, col).cast[accum_type]()
                var normalized = apply_gamma[simd_width](
                    vec_data * norm_factor, col
                )

                var output_fp8 = fp8_quantize[out_dtype](
                    normalized, scale_factor_recip
                )
                output_fn[simd_width](row, col, output_fp8)
