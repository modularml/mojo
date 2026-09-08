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

"""Provides FP8 quantization kernels supporting static, dynamic, and blockwise scaling."""

from std.collections.string.string_span import get_static_string
from std.math import ceildiv
from std.math.uutils import ufloordiv
from std.atomic import Atomic
from std.sys import simd_width_of, has_nvidia_gpu_accelerator
from std.sys import align_of, size_of, get_defined_bool
import max.gpu.primitives.block as block
from max.algorithm.functional import _elementwise_impl_gpu
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    thread_idx,
)
from max.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.info import B200, _is_sm10x_gpu
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    lt_to_tt,
    row_major,
)
from layout.tile_layout import TensorLayout
from layout.tensor_engine import TensorEngine
from std.logger import Logger
from std.memory import bitcast
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from max.algorithm import elementwise
from std.utils.coord import Coord, Idx, coord_to_index_list
from std.utils.index import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type, max_finite

from .matmul import matmul
from .matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)
from .utils import elementwise_epilogue_type
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
    GEMMKind,
)
from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize
from max.gpu.primitives.grid_controls import PDLLevel

comptime logger = Logger()


comptime _row_col_input_fn_trait[
    in_dtype: DType
] = ImplicitlyCopyable & RegisterPassable & (
    def[width: Int, alignment: Int](row: Int, col: Int) -> SIMD[in_dtype, width]
)

comptime _batched_input_fn_trait[
    in_dtype: DType
] = ImplicitlyCopyable & RegisterPassable & (
    def[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) -> SIMD[in_dtype, width]
)


########################################################
# static scaled fp8 quantization
########################################################


@always_inline
def quantize_static_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scale_is_inverted: Bool = True,
](
    out_tensor: TileTensor[mut=True, out_dtype, ...],
    in_tensor: TileTensor[mut=False, in_dtype, ...],
    scale: Float32,
    context: DeviceContext,
) raises:
    """TileTensor implementation of static scaled FP8 quantization."""
    comptime assert out_tensor.rank == 2, "expected rank-2 output"
    comptime assert in_tensor.rank == 2, "expected rank-2 input"
    comptime assert in_dtype in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
    ), "input dtype should be float16, bfloat16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    @always_inline
    @__parameter
    @__copy_capture(out_tensor, in_tensor, scale)
    def scaled_fp8_quant[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var in_vec_f32 = in_tensor.load_linear[width](idx).cast[.float32]()
        var inversed_scale: Float32 = 1.0 / scale
        out_tensor.store_linear(
            idx,
            fp8_quantize[out_dtype, use_clamp=True](in_vec_f32, inversed_scale),
        )

    comptime target_simd_width = simd_width_of[
        in_dtype, target=get_gpu_target()
    ]()

    def scaled_fp8_quant_unified[width: Int, alignment: Int = 1](idx: Coord) {}:
        scaled_fp8_quant[width, idx.rank, alignment](coord_to_index_list(idx))

    _elementwise_impl_gpu[
        simd_width=target_simd_width,
        trace_description="scaled_fp8_quant",
    ](
        scaled_fp8_quant_unified,
        shape=(Int(in_tensor.dim[0]()), Int(in_tensor.dim[1]())),
        ctx=context,
    )


def zero_scale_global_kernel(
    scale_global: UnsafePointer[Float32, MutAnyOrigin]
):
    """Zeros the global FP8 scale factor in a single thread.

    Writes ``0`` to ``scale_global[0]`` directly rather than using
    ``enqueue_fill``, which can deadlock under CUDA graph replay when the GPU
    is already spinning inside a collectives kernel.

    Args:
        scale_global: Pointer to the single FP32 global scale to zero.
    """
    # GENAI-512: Avoid using `enqueue_fill` for this operation as this can
    # deadlock when using CUDA graphs. The graph node for the async memset
    # could try to load a CUDA kernel, but if the GPU is spinning inside a
    # collectives kernel, then the graph replay will deadlock.
    scale_global[0] = 0


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
def max_reduction_scale_kernel[
    in_dtype: DType,
    out_dtype: DType,
    input_layout: TensorLayout,
    scale_layout: TensorLayout,
    num_threads: Int,
](
    scale_global: TileTensor[mut=True, .float32, scale_layout, MutAnyOrigin],
    input_tensor: TileTensor[in_dtype, input_layout, MutAnyOrigin],
):
    """Per-row strided max-|x| reduction into a global FP8 scale.

    One block scans one row: threads stride across the hidden dimension, reduce
    to a row-wise max absolute value, then thread 0 atomically updates
    ``scale_global`` with ``row_max / max_finite[out_dtype]``.

    Args:
        scale_global: Length-1 FP32 ``TileTensor``; must be zero before launch.
        input_tensor: Rank-2 input.
    """
    comptime fp8_max = Float32(max_finite[out_dtype]())
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var hidden_size = Int(input_tensor.dim[1]())

    var thread_max = Float32(0)
    with PDL():
        for e in range(tid, hidden_size, num_threads):
            var v = abs(
                input_tensor.load_linear(Index(row, e)).cast[.float32]()
            ).reduce_max()
            thread_max = max(thread_max, v)

        var row_max = block.max[block_size=num_threads, broadcast=True](
            thread_max
        )

        if tid == 0:
            _ = Atomic[Float32].max(scale_global.ptr, row_max / fp8_max)


@always_inline
def quantize_tensor_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    InputFnType: _row_col_input_fn_trait[in_dtype],
    //,
    group_size_or_per_token: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    input_fn: InputFnType,
    scaled_output: TileTensor[mut=True, out_dtype, ...],
    scales: TileTensor[mut=True, scales_dtype, ...],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
) raises:
    """TileTensor primary implementation of dynamic scaled FP8 quantization.

    Parameters:
        out_dtype: FP8 dtype of the quantized output; must be
            `float8_e4m3fn`.
        in_dtype: Dtype of the input values loaded by `input_fn`.
        scales_dtype: Dtype of the per-group scale factors written
            to `scales`; one of `bfloat16`, `float16`, or `float32`.
        InputFnType: Compile-time callable type that loads
            `SIMD[in_dtype, width]` tiles for a given `(row, col)`
            (inferred).
        group_size_or_per_token: Number of columns per quantization
            group; `-1` selects per-tensor scaling over all
            `num_cols`.
        num_cols: Number of columns in the input; the hidden
            dimension size.
        pdl_level: Programmatic dependent launch level for the
            kernels (defaults to `PDLLevel.ON`).

    Args:
        input_fn: Callable that loads input tiles for a given
            `(row, col)` pair.
        scaled_output: Rank-2 output `TileTensor` of FP8-quantized
            values.
        scales: Rank-2 output `TileTensor` of per-group scale
            factors.
        scale_ub: Upper bound for the dynamic scale factor.
        ctx: Device context used to enqueue the kernels.
        num_rows: Number of rows to quantize; `0` skips the launch.
    """
    comptime assert scaled_output.rank == 2, "expected rank-2 output"
    comptime assert scales.rank == 2, "expected rank-2 scales"

    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    comptime assert (scales_dtype != .float8_e8m0fnu) or (
        out_dtype == .float8_e4m3fn
    ), "float8_e8m0fnu is only supported for float8_e4m3fn output dtype"

    comptime group_size = num_cols if group_size_or_per_token == -1 else group_size_or_per_token
    comptime simd_width = 16 if group_size % 16 == 0 else 8 if group_size % 8 == 0 else 4
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )
    comptime num_threads = warps_per_block * WARP_SIZE

    comptime assert (
        group_size % simd_width == 0
    ), "group size must be multiple of simd size"

    # TODO: MOCO-4295
    @always_inline
    def wrap[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var input_fn} -> SIMD[in_dtype, width]:
        return input_fn.__call__[width=width, alignment=alignment](row, col)

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "quantize_tensor_dynamic_scaled_fp8",
        task_id=Int(ctx.id()),
    ):
        if num_rows == 0:
            return

        # Per-tensor mode: two kernel launches.
        # 1) Existing per-group kernel: computes per-group scales
        #    and writes them to the scales buffer.
        # 2) Per-tensor quantize kernel: loops over all per-group
        #    scales to find the tensor-wide max, then re-quantizes
        #    every element with that single scale.
        if num_rows > 1:
            var scales_kernel = _ComputeScalesFp8Kernel[
                out_type=out_dtype,
                num_threads=num_threads,
                group_size=group_size,
                simd_width=simd_width,
            ](
                wrap,
                scales.address_space_cast[.GENERIC](),
                scale_ub.cast[scales_dtype](),
            )

            ctx.enqueue_function(
                scales_kernel,
                grid_dim=(num_rows, num_cols // group_size, 1),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )

            var quant_kernel = _QuantizeFp8KernelPerTensor[
                num_threads=num_threads,
                group_size=group_size,
                simd_width=simd_width,
                num_groups=num_cols // group_size,
            ](
                wrap,
                scaled_output.address_space_cast[.GENERIC](),
                scales.address_space_cast[.GENERIC](),
                scale_ub.cast[scales_dtype](),
                Int32(num_rows),
            )

            ctx.enqueue_function(
                quant_kernel,
                grid_dim=(num_rows, num_cols // group_size, 1),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )
        else:
            var kernel = _QuantizeFp8Kernel[
                num_threads=num_threads,
                group_size=group_size,
                simd_width=simd_width,
            ](
                wrap,
                scaled_output.address_space_cast[.GENERIC](),
                scales.address_space_cast[.GENERIC](),
                scale_ub.cast[scales_dtype](),
            )

            ctx.enqueue_function(
                kernel,
                grid_dim=(num_rows, num_cols // group_size, 1),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )


########################################################
# dynamic scaled fp8 quantization
########################################################


@always_inline
def quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    InputFnType: _row_col_input_fn_trait[in_dtype],
    //,
    group_size_or_per_token: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    input_fn: InputFnType,
    scaled_output: TileTensor[mut=True, out_dtype, ...],
    scales: TileTensor[mut=True, scales_dtype, ...],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
) raises:
    """TileTensor primary implementation of dynamic scaled FP8 quantization."""
    comptime assert scaled_output.rank == 2, "expected rank-2 output"
    comptime assert scales.rank == 2, "expected rank-2 scales"

    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
        DType.float8_e8m0fnu,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    comptime assert (scales_dtype != .float8_e8m0fnu) or (
        out_dtype == .float8_e4m3fn
    ), "float8_e8m0fnu is only supported for float8_e4m3fn output dtype"

    comptime group_size = num_cols if group_size_or_per_token == -1 else group_size_or_per_token
    comptime simd_width = 16 if group_size % 16 == 0 else 8 if group_size % 8 == 0 else 4
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )
    comptime num_threads = warps_per_block * WARP_SIZE

    comptime assert (
        group_size % simd_width == 0
    ), "group size must be multiple of simd size"

    # TODO: MOCO-4295
    @always_inline
    def wrap[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var input_fn} -> SIMD[in_dtype, width]:
        return input_fn.__call__[width=width, alignment=alignment](row, col)

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "quantize_dynamic_scaled_fp8",
        task_id=Int(ctx.id()),
    ):
        if num_rows == 0:
            return

        comptime if get_defined_bool["ENABLE_PER_TENSOR_FP8_QUANTIZE", False]():
            quantize_tensor_dynamic_scaled_fp8[
                in_dtype=in_dtype,
                group_size_or_per_token=group_size_or_per_token,
                num_cols=num_cols,
                pdl_level=pdl_level,
            ](
                input_fn,
                scaled_output,
                scales,
                scale_ub,
                ctx,
                num_rows,
            )

        else:
            var kernel = _QuantizeFp8Kernel[
                num_threads=num_threads,
                group_size=group_size,
                simd_width=simd_width,
            ](
                wrap,
                scaled_output.address_space_cast[.GENERIC](),
                scales.address_space_cast[.GENERIC](),
                scale_ub.cast[scales_dtype](),
            )

            ctx.enqueue_function(
                kernel,
                grid_dim=(num_rows, num_cols // group_size, 1),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )


@fieldwise_init
struct _QuantizeFp8Kernel[
    out_type: DType,
    scales_type: DType,
    output_layout: TensorLayout,
    output_origin: MutOrigin,
    output_engine: TensorEngine,
    output_idx_type: DType,
    scales_layout: TensorLayout,
    scales_origin: MutOrigin,
    scales_engine: TensorEngine,
    scales_idx_type: DType,
    //,
    in_type: DType,
    InputFnType: _row_col_input_fn_trait[in_type],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var input_fn: Self.InputFnType
    var output: TileTensor[
        mut=True,
        Self.out_type,
        Self.output_layout,
        Self.output_origin,
        Engine=Self.output_engine,
        linear_idx_type=Self.output_idx_type,
    ]
    var scales: TileTensor[
        mut=True,
        Self.scales_type,
        Self.scales_layout,
        Self.scales_origin,
        Engine=Self.scales_engine,
        linear_idx_type=Self.scales_idx_type,
    ]
    var scale_ub: Scalar[Self.scales_type]

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    def __call__(self) capturing:
        var input_fn = self.input_fn
        var output = TileTensor(self.output.ptr, self.output.layout)
        var scales = TileTensor(self.scales.ptr, self.scales.layout)
        var scale_ub = self.scale_ub
        comptime use_warp_tiling = Self.group_size <= Self.num_threads * Self.simd_width
        comptime fp8_max = Scalar[Self.out_type].MAX_FINITE
        comptime accum_type = get_accum_type[Self.in_type]()

        comptime assert (Self.scales_type != .float8_e8m0fnu) or (
            accum_type == .float32
        ), (
            "float8_e8m0fnu quantization is only supported for float32 accum"
            " type"
        )

        var input_vec = SIMD[accum_type, Self.simd_width](0)
        var thread_max = Scalar[accum_type](0)

        var tid = thread_idx.x
        var row = block_idx.x
        var group_idx = block_idx.y

        with PDL():
            for i in range(
                tid, Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size
                input_vec = input_fn.__call__[
                    width=Self.simd_width,
                    alignment=Self.simd_width,
                ](row, idx).cast[accum_type]()
                thread_max = max(thread_max, abs(input_vec).reduce_max())

            var group_max = block.max[
                block_size=Self.num_threads, broadcast=True
            ](thread_max)

            var scale_factor: Scalar[Self.scales_type]
            var scale_factor_recip: Scalar[accum_type]

            comptime if Self.scales_type == .float8_e8m0fnu:
                scale_factor = max(
                    group_max / fp8_max.cast[accum_type](),
                    Scalar[accum_type](1e-10),
                ).cast[Self.scales_type]()
                scale_factor_recip = (
                    0.0 if group_max
                    == 0.0 else 1.0 / scale_factor.cast[accum_type]()
                )
            else:
                scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                    Self.out_type
                ](group_max, scale_ub)

            if tid == 0:
                scales.store_linear(Index(group_idx, row), scale_factor)

            for i in range(
                tid, Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size

                comptime if use_warp_tiling:
                    pass
                else:
                    input_vec = input_fn.__call__[
                        width=Self.simd_width,
                        alignment=Self.simd_width,
                    ](row, idx).cast[accum_type]()

                output.store_linear(
                    Index(row, idx),
                    fp8_quantize[Self.out_type](input_vec, scale_factor_recip),
                )


@fieldwise_init
struct _ComputeScalesFp8Kernel[
    out_type: DType,
    scales_type: DType,
    scales_layout: TensorLayout,
    scales_origin: MutOrigin,
    scales_engine: TensorEngine,
    scales_idx_type: DType,
    //,
    in_type: DType,
    InputFnType: _row_col_input_fn_trait[in_type],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var input_fn: Self.InputFnType
    var scales: TileTensor[
        mut=True,
        Self.scales_type,
        Self.scales_layout,
        Self.scales_origin,
        Engine=Self.scales_engine,
        linear_idx_type=Self.scales_idx_type,
    ]
    var scale_ub: Scalar[Self.scales_type]

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    def __call__(self) capturing:
        """Compute per-group FP8 scale factors without quantizing.

        Each block scans its (row, group) tile via ``input_fn``, computes the
        scale factor, and writes it to ``scales[group_idx, row]``. This is
        the first half of the per-tensor path, so `_QuantizeFp8KernelPerTensor`
        can find the tensor-wide max scale.
        """
        var input_fn = self.input_fn
        var scales = TileTensor(self.scales.ptr, self.scales.layout)
        comptime fp8_max = Scalar[Self.out_type].MAX_FINITE
        comptime accum_type = get_accum_type[Self.in_type]()

        comptime assert Self.scales_type in (
            DType.bfloat16,
            DType.float16,
            DType.float32,
        ), "scales dtype should be bfloat16, float16 or float32"
        comptime assert (Self.scales_type != .float8_e8m0fnu) or (
            accum_type == .float32
        ), (
            "float8_e8m0fnu quantization is only supported for float32 accum"
            " type"
        )

        var thread_max = Scalar[accum_type](0)

        var tid = thread_idx.x
        var row = block_idx.x
        var group_idx = block_idx.y

        with PDL():
            for i in range(
                Int(tid), Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size
                var input_vec = input_fn.__call__[
                    width=Self.simd_width,
                    alignment=Self.simd_width,
                ](row, idx).cast[accum_type]()
                thread_max = max(thread_max, abs(input_vec).reduce_max())

            var group_max = block.max[
                block_size=Self.num_threads, broadcast=True
            ](thread_max)

            if tid == 0:
                scales.store_linear(
                    Index(group_idx, row), group_max.cast[Self.scales_type]()
                )


@fieldwise_init
struct _QuantizeFp8KernelPerTensor[
    out_type: DType,
    scales_type: DType,
    output_layout: TensorLayout,
    output_origin: MutOrigin,
    output_engine: TensorEngine,
    output_idx_type: DType,
    scales_layout: TensorLayout,
    scales_origin: MutOrigin,
    scales_engine: TensorEngine,
    scales_idx_type: DType,
    //,
    in_type: DType,
    InputFnType: _row_col_input_fn_trait[in_type],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
    num_groups: Int,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var input_fn: Self.InputFnType
    var output: TileTensor[
        mut=True,
        Self.out_type,
        Self.output_layout,
        Self.output_origin,
        Engine=Self.output_engine,
        linear_idx_type=Self.output_idx_type,
    ]
    var scales: TileTensor[
        mut=True,
        Self.scales_type,
        Self.scales_layout,
        Self.scales_origin,
        Engine=Self.scales_engine,
        linear_idx_type=Self.scales_idx_type,
    ]
    var scale_ub: Scalar[Self.scales_type]
    var num_rows_dev: Int32

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    def __call__(self) capturing:
        """Per-tensor FP8 quantize kernel.

        Reads all per-group scales written by ``_ComputeScalesFp8Kernel``
        (stored as ``scales[group_idx, row]``), finds the tensor-wide maximum
        scale, and re-quantizes every element with that single scale.

        Block (0, 0) thread 0 overwrites ``scales[0, 0]`` with the final
        per-tensor scale factor so the caller can read it back.
        """
        var input_fn = self.input_fn
        var output = TileTensor(self.output.ptr, self.output.layout)
        var scales = TileTensor(self.scales.ptr, self.scales.layout)
        var scale_ub = self.scale_ub
        var num_rows = Int(self.num_rows_dev)
        comptime accum_type = get_accum_type[Self.in_type]()

        var tid = thread_idx.x
        var row = block_idx.x
        var group_idx = block_idx.y

        with PDL():
            # Find the max scale across all groups and rows.
            # The scales buffer is small (Self.num_groups * num_rows), so a serial
            # scan per block is cheap and avoids cross-block synchronisation.
            var max_scale = Scalar[Self.scales_type](0)
            comptime for g in range(Self.num_groups):
                for r in range(num_rows):
                    var s = rebind[Scalar[Self.scales_type]](
                        scales.load_linear(Index(g, r))
                    )
                    max_scale = max(max_scale, s)

            # `max_scale` is the tensor-wide max-abs: _ComputeScalesFp8Kernel writes
            # each group's RAW max, and the scan above takes the largest. Route the
            # scale and its reciprocal through the shared compute_dynamic_fp8_scale so
            # the per-tensor path gets the SAME finite-reciprocal guard as the
            # per-group kernels: a near-zero/denormal tensor max makes scale_factor
            # underflow to a nonzero f32 denormal and 1/scale_factor overflow to +Inf,
            # which NaNs the fp8 cast on a zero lane (0*Inf). The guard treats a
            # non-finite reciprocal as zero scale, so the group quantizes to fp8 zero.
            var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                Self.out_type
            ](max_scale.cast[accum_type](), scale_ub)

            # Write the per-tensor scale to every position so downstream
            # readers that index scales[group_idx, row] see the correct value.
            if tid == 0:
                scales.store_linear(Index(group_idx, row), scale_factor)

            for i in range(
                Int(tid), Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size
                var input_vec = input_fn.__call__[
                    width=Self.simd_width,
                    alignment=Self.simd_width,
                ](row, idx).cast[accum_type]()
                output.store_linear(
                    Index(row, idx),
                    fp8_quantize[Self.out_type](input_vec, scale_factor_recip),
                )


@always_inline
def batched_quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    InputFnType: _batched_input_fn_trait[in_dtype],
    //,
    group_size_or_per_token: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    input_fn: InputFnType,
    scaled_output: TileTensor[mut=True, out_dtype, ...],
    scales: TileTensor[mut=True, scales_dtype, ...],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
    batch_size: Int,
) raises:
    """TileTensor primary implementation of batched dynamic scaled FP8
    quantization."""
    comptime assert scaled_output.rank == 3, "expected rank-3 output"
    comptime assert scales.rank == 3, "expected rank-3 scales"

    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    comptime group_size = num_cols if group_size_or_per_token == -1 else group_size_or_per_token
    comptime simd_width = 16 if group_size % 16 == 0 else 8 if group_size % 8 == 0 else 4
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )
    comptime num_threads = warps_per_block * WARP_SIZE

    comptime assert (
        group_size % simd_width == 0
    ), "group size must be multiple of simd size"

    if batch_size == 0 or num_rows == 0:
        return

    # TODO: MOCO-4295
    @always_inline
    def wrap[
        width: Int, alignment: Int
    ](batch: Int, row: Int, col: Int) {var input_fn} -> SIMD[in_dtype, width]:
        return input_fn.__call__[width=width, alignment=alignment](
            batch, row, col
        )

    var kernel = _BatchedQuantizeFp8Kernel[
        num_threads=num_threads,
        group_size=group_size,
        simd_width=simd_width,
    ](
        wrap,
        scaled_output.address_space_cast[.GENERIC](),
        scales.address_space_cast[.GENERIC](),
        scale_ub.cast[scales_dtype](),
    )

    ctx.enqueue_function(
        kernel,
        grid_dim=(num_rows, num_cols // group_size, batch_size),
        block_dim=num_threads,
        attributes=pdl_launch_attributes(pdl_level),
    )


@fieldwise_init
struct _BatchedQuantizeFp8Kernel[
    out_type: DType,
    scales_type: DType,
    output_layout: TensorLayout,
    output_origin: MutOrigin,
    output_engine: TensorEngine,
    output_idx_type: DType,
    scales_layout: TensorLayout,
    scales_origin: MutOrigin,
    scales_engine: TensorEngine,
    scales_idx_type: DType,
    //,
    in_type: DType,
    InputFnType: _batched_input_fn_trait[in_type],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var input_fn: Self.InputFnType
    var output: TileTensor[
        mut=True,
        Self.out_type,
        Self.output_layout,
        Self.output_origin,
        Engine=Self.output_engine,
        linear_idx_type=Self.output_idx_type,
    ]
    var scales: TileTensor[
        mut=True,
        Self.scales_type,
        Self.scales_layout,
        Self.scales_origin,
        Engine=Self.scales_engine,
        linear_idx_type=Self.scales_idx_type,
    ]
    var scale_ub: Scalar[Self.scales_type]

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    def __call__(self) capturing:
        var input_fn = self.input_fn
        var output = TileTensor(self.output.ptr, self.output.layout)
        var scales = TileTensor(self.scales.ptr, self.scales.layout)
        var scale_ub = self.scale_ub
        comptime use_warp_tiling = Self.group_size <= Self.num_threads * Self.simd_width
        comptime accum_type = get_accum_type[Self.in_type]()

        var input_vec = SIMD[accum_type, Self.simd_width](0)
        var thread_max = Scalar[accum_type](0)

        var tid = thread_idx.x
        var row = block_idx.x
        var group_idx = block_idx.y
        var batch_idx = block_idx.z

        with PDL():
            for i in range(
                tid, Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size
                input_vec = input_fn.__call__[
                    width=Self.simd_width,
                    alignment=Self.simd_width,
                ](batch_idx, row, idx).cast[accum_type]()
                thread_max = max(thread_max, abs(input_vec).reduce_max())

            var group_max = block.max[
                block_size=Self.num_threads, broadcast=True
            ](thread_max)

            var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                Self.out_type
            ](group_max, scale_ub)

            if tid == 0:
                scales.store_linear(
                    Index(batch_idx, group_idx, row), scale_factor
                )

            for i in range(
                tid, Self.group_size // Self.simd_width, Self.num_threads
            ):
                var idx: Int = i * Self.simd_width + group_idx * Self.group_size

                comptime if use_warp_tiling:
                    pass
                else:
                    input_vec = input_fn.__call__[
                        width=Self.simd_width,
                        alignment=Self.simd_width,
                    ](batch_idx, row, idx).cast[accum_type]()

                output.store_linear(
                    Index(batch_idx, row, idx),
                    fp8_quantize[Self.out_type](input_vec, scale_factor_recip),
                )


########################################################
# scaled fp8 matmul
########################################################


@always_inline
def matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """TileTensor primary implementation of dynamic scaled FP8 matmul."""
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert a_scales.rank == 2
    comptime assert b_scales.rank == 2

    comptime assert a_type == b_type, "input A and B dtype should be the same"
    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be the same"

    comptime assert a_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input A dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert b_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input B dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert a_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input A scales dtype should be bfloat16, float16 or float32"
    comptime assert b_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input B scales dtype should be bfloat16, float16 or float32"

    _matmul_dynamic_scaled_fp8_impl[
        input_scale_granularity,
        weight_scale_granularity,
        m_scale_granularity,
        n_scale_granularity,
        k_scale_granularity,
        transpose_b,
        target,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )


def _matmul_dynamic_scaled_fp8_impl[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """TileTensor implementation of dynamic scaled FP8 matmul."""
    comptime assert a_type == b_type, "input A and B dtype should be the same"
    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be the same"

    comptime assert a_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input A dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert b_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input B dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert a_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input A scales dtype should be bfloat16, float16 or float32"
    comptime assert b_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input B scales dtype should be bfloat16, float16 or float32"

    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert a_scales.rank == 2
    comptime assert b_scales.rank == 2
    # Provide evidence that flat_rank >= 2 for Coord(..., ...)
    # loads/stores on a_scales, b_scales, and c below.
    comptime assert c.flat_rank >= 2
    comptime assert a_scales.flat_rank >= 2
    comptime assert b_scales.flat_rank >= 2

    comptime b_row_axis = 0 if transpose_b else 1
    var M = Int(a.dim[0]())

    if M == 0:
        return

    # get_static_string requires compile-time values, so we use
    # static_shape (which is -1 for dynamic dims). We map -1 to 0
    # in the trace string since trace_arg formats the shape as-is
    # and 0 is a clearer indicator of "unknown at compile time".
    comptime _trace_string = get_static_string[
        trace_arg(
            "A_scales",
            IndexList[2](
                a_scales.static_shape[0] if a_scales.static_shape[0]
                > -1 else 0,
                a_scales.static_shape[1] if a_scales.static_shape[1]
                > -1 else 0,
            ),
            a_scales_type,
        ),
        ";",
        trace_arg(
            "B_scales",
            IndexList[2](
                b_scales.static_shape[0] if b_scales.static_shape[0]
                > -1 else 0,
                b_scales.static_shape[1] if b_scales.static_shape[1]
                > -1 else 0,
            ),
            b_scales_type,
        ),
    ]()

    # Tensorwise and Channelwise scaling
    comptime if (
        input_scale_granularity == "colwise"
        and weight_scale_granularity == "rowwise"
    ) or (input_scale_granularity == weight_scale_granularity == "tensor"):
        logger.info(
            "Dispatching Matmul Dynamic Scaled FP8. Input Scale Granularity: ",
            input_scale_granularity,
            ", Weight Scale Granularity: ",
            weight_scale_granularity,
        )

        comptime if _is_sm10x_gpu(ctx.default_device_info):

            @__parameter
            @always_inline
            @__copy_capture(a_scales, b_scales)
            def scale_compute_lambda_fn[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var a_scale = a_scales.load[width=1](
                    Coord(Idx[0], idx[0])
                ).cast[.float32]()
                var b_scale: SIMD[.float32, width]

                comptime if transpose_b:
                    b_scale = b_scales.load[width=width](
                        Coord(idx[1], Idx[0])
                    ).cast[.float32]()
                else:
                    b_scale = b_scales.load[width=width](
                        Coord(Idx[0], idx[1])
                    ).cast[.float32]()

                var scaled_val = val.cast[.float32]() * a_scale * b_scale
                return scaled_val.cast[_dtype]()

            @__parameter
            @always_inline
            @__copy_capture(a_scales, b_scales)
            def scale_compute_lambda_fn_tensor[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var a_scale = a_scales.load[width=1](
                    Coord(Idx[0], Idx[0])
                ).cast[.float32]()
                var b_scale = b_scales.load[width=1](
                    Coord(Idx[0], Idx[0])
                ).cast[.float32]()
                var scaled_val = val.cast[.float32]() * a_scale * b_scale
                return scaled_val.cast[_dtype]()

            comptime if input_scale_granularity == "tensor":
                matmul[
                    target=target,
                    transpose_b=transpose_b,
                    elementwise_compute_lambda_fn=scale_compute_lambda_fn_tensor,
                    _trace_description=_trace_string,
                ](c, a, b, Optional[DeviceContext](ctx))
            else:
                matmul[
                    target=target,
                    transpose_b=transpose_b,
                    elementwise_compute_lambda_fn=scale_compute_lambda_fn,
                    _trace_description=_trace_string,
                ](c, a, b, Optional[DeviceContext](ctx))

        else:
            # create a dummy TileTensor to instruct the matmul kernel to
            # output values in the correct dtype.

            @__parameter
            @__copy_capture(c, a_scales, b_scales)
            @always_inline
            def scaled_output_fn[
                dtype: DType, width: SIMDLength, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                var a_scale = a_scales.load[width=1](
                    Coord(Idx[0], idx[0])
                ).cast[dtype]()
                var b_scale: SIMD[dtype, width]

                comptime if transpose_b:
                    b_scale = b_scales.load[width=width](
                        Coord(idx[1], Idx[0])
                    ).cast[dtype]()
                else:
                    b_scale = b_scales.load[width=width](
                        Coord(Idx[0], idx[1])
                    ).cast[dtype]()

                var scaled_val = val * a_scale * b_scale

                c.store[width=width, alignment=alignment](
                    Coord(idx[0], idx[1]),
                    scaled_val.cast[c_type](),
                )

            @__parameter
            @__copy_capture(c, a_scales, b_scales)
            @always_inline
            def scaled_output_fn_tensor[
                dtype: DType, width: SIMDLength, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                var a_scale = a_scales.load[width=1](
                    Coord(Idx[0], Idx[0])
                ).cast[dtype]()
                var b_scale = b_scales.load[width=1](
                    Coord(Idx[0], Idx[0])
                ).cast[dtype]()
                var scaled_val = val * a_scale * b_scale

                c.store[width=width, alignment=alignment](
                    Coord(idx[0], idx[1]),
                    scaled_val.cast[c_type](),
                )

            # Allocate an fp32 scratch buffer for the matmul accumulator;
            # the epilogue lambda reads from it, applies scaling, and
            # writes the quantized result into the real fp8 `c`.
            # Preserve the compile-time-static N dimension from b so the
            # SM90 dispatch sees c.static_shape[1] > -1.
            comptime b_N = b.static_shape[b_row_axis]
            comptime if b_N > -1:
                var scratch_buffer = ctx.enqueue_create_buffer[.float32](
                    M * b_N
                )
                var c_scratch = TileTensor(
                    scratch_buffer.unsafe_ptr(),
                    row_major(Coord(M, Idx[b_N])),
                )

                comptime if input_scale_granularity == "tensor":
                    matmul[
                        target=target,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=scaled_output_fn_tensor,
                        _trace_description=_trace_string,
                    ](c_scratch, a, b, Optional[DeviceContext](ctx))
                else:
                    matmul[
                        target=target,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=scaled_output_fn,
                        _trace_description=_trace_string,
                    ](c_scratch, a, b, Optional[DeviceContext](ctx))
            else:
                var N_rt = Int(b.dim[b_row_axis]())
                var scratch_buffer = ctx.enqueue_create_buffer[.float32](
                    M * N_rt
                )
                var c_scratch = TileTensor(
                    scratch_buffer.unsafe_ptr(),
                    row_major(Coord(M, N_rt)),
                )

                comptime if input_scale_granularity == "tensor":
                    matmul[
                        target=target,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=scaled_output_fn_tensor,
                        _trace_description=_trace_string,
                    ](c_scratch, a, b, Optional[DeviceContext](ctx))
                else:
                    matmul[
                        target=target,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=scaled_output_fn,
                        _trace_description=_trace_string,
                    ](c_scratch, a, b, Optional[DeviceContext](ctx))

    elif (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ):
        blockwise_scaled_fp8_with_epilogue[
            transpose_b=transpose_b,
            scales_granularity_mnk=IndexList[3](
                m_scale_granularity,
                n_scale_granularity,
                k_scale_granularity,
            ),
        ](c, a, b, a_scales, b_scales, ctx)

    else:
        comptime assert False, (
            "Unsupported scaling mode: input_scale_granularity="
            + input_scale_granularity
            + ", weight_scale_granularity="
            + weight_scale_granularity
        )


def naive_blockwise_scaled_fp8_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    BLOCK_DIM: Int = 16,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    scales_granularity_mnk: Optional[IndexList[3]] = None,
](
    c: LayoutTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: LayoutTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: LayoutTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: LayoutTensor[
        mut=False, a_scales_type, address_space=.GENERIC, ...
    ],
    b_scales: LayoutTensor[
        mut=False, b_scales_type, address_space=.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Dispatches the naive blockwise scaled FP8 matmul kernel on the GPU.

    Converts the ``LayoutTensor`` operands to ``TileTensor`` views and enqueues
    ``naive_blockwise_scaled_fp8_matmul_kernel`` with a 2D grid of
    ``BLOCK_DIM``-sized tiles covering the ``M`` x ``N`` output.

    Args:
        c: Rank-2 output accumulator tensor.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix; K-major when ``transpose_b`` is True, otherwise N-major.
        a_scales: Rank-2 per-block scales for ``a`` in M-major format.
        b_scales: Rank-2 per-block scales for ``b``; K-major when ``transpose_b`` is True, otherwise N-major.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert a_type == b_type == .float8_e4m3fn, (
        "Only float8_e4m3fn is supported for input dtype for blockwise"
        " scaled fp8 matmul"
    )

    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be same"

    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1)

    var a_scales_dim0 = a_scales.dim(0)
    var b_scales_dim0 = b_scales.dim(0)
    var b_scales_dim1 = b_scales.dim(1)

    if M == 0 or N == 0 or K == 0:
        return

    # these checks are only applicable when A_SCALES_SIZE and B_SCALES_SIZE are not provided
    comptime if not scales_granularity_mnk:
        if K % a_scales_dim0 != 0:
            raise Error(
                "K must be divisible by a_scales.dim(0) if A_SCALES_SIZE is not"
                " provided"
            )

        if transpose_b and (K % b_scales_dim1 != 0 or N % b_scales_dim0 != 0):
            raise Error(
                "K must be divisible by b_scales.dim(1) and N must be divisible"
                " by b_scales.dim(0) if B_SCALES_SIZE is not provided"
            )

        if not transpose_b and (
            K % b_scales_dim0 != 0 or N % b_scales_dim1 != 0
        ):
            raise Error(
                "K must be divisible by b_scales.dim(0) and N must be divisible"
                " by b_scales.dim(1) if B_SCALES_SIZE is not provided"
            )

    logger.info("Executing Naive Blockwise Scaled FP8 GEMM")
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]", sep="")
    logger.info(
        "A Scales Shape: [", a_scales.dim(0), ", ", a_scales.dim(1), "]", sep=""
    )
    logger.info(
        "B Scales Shape: [", b_scales.dim(0), ", ", b_scales.dim(1), "]", sep=""
    )

    var a_tt = lt_to_tt(a).as_immut()
    var b_tt = lt_to_tt(b).as_immut()
    var c_tt = lt_to_tt(c)
    var a_scales_tt = lt_to_tt(a_scales).as_immut()
    var b_scales_tt = lt_to_tt(b_scales).as_immut()

    comptime kernel = naive_blockwise_scaled_fp8_matmul_kernel[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        type_of(c_tt).LayoutType,
        type_of(a_scales_tt).LayoutType,
        type_of(b_scales_tt).LayoutType,
        BLOCK_DIM=BLOCK_DIM,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ]

    ctx.enqueue_function[kernel](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


def naive_blockwise_scaled_fp8_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: TileTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: TileTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    """Computes one output element per thread for the naive blockwise scaled FP8 matmul GPU kernel.

    Each thread accumulates the scaled dot product for a single ``(row, col)``
    output element by looping over the K dimension, loading the corresponding
    ``a`` and ``b`` values and their per-block scale factors, and applying the
    optional elementwise epilogue before storing the result.

    Supports two scaling modes: inferred per-group scale sizes from the input
    and scale tensor shapes (when ``scales_granularity_mnk`` is ``None``), or
    explicit ``(M, N, K)`` scale granularities provided via
    ``scales_granularity_mnk``.

    Args:
        c: Rank-2 output accumulator tensor.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-2 FP8 weight matrix; K-major when ``transpose_b`` is True, otherwise N-major.
        a_scales: Rank-2 per-block scales for ``a`` in M-major format.
        b_scales: Rank-2 per-block scales for ``b``; K-major when ``transpose_b`` is True, otherwise N-major.
    """
    # Note: This is a naive kernel that supports a generalized blockwise scaled
    # fp8 matmul.
    # Currently, it supports two modes:
    # 1. [1 x SCALE_SIZE_K] x [SCALE_SIZE_K x SCALE_SIZE_N] blockwise scaling if
    #    scales_granularity_mnk is not provided. In this mode, the kernel will infer
    #    the scale sizes from the input and scale tensor shapes. The input shapes
    #    must be divisible by the scale sizes otherwise it will raise an error.
    # 2. [SCALE_SIZE_M x SCALE_SIZE_K] x [SCALE_SIZE_K x SCALE_SIZE_N] blockwise scaling if
    #    scales_granularity_mnk is provided. In this mode, the kernel will use the
    #    provided scale sizes to compute the scaled matmul.
    #
    # Assumptions:
    # 1. a should be always in K-major format
    # 2. b should be in K-major format if transpose_b is True otherwise it is in N-major format
    # 3. a_scales should be always in M-major format
    # 4. b_scales should be in K-major format if transpose_b is True otherwise it is in N-major format

    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert a_scales.rank == 2
    comptime assert b_scales.rank == 2

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    var x = global_idx.x
    var y = global_idx.y

    if x >= M or y >= N:
        return

    var MAT_A_ROWS_SCALE_SIZE: Int
    var MAT_A_COLS_SCALE_SIZE: Int
    var MAT_B_ROWS_SCALE_SIZE: Int
    var MAT_B_COLS_SCALE_SIZE: Int

    comptime if scales_granularity_mnk:
        comptime scales_granularity = scales_granularity_mnk.value()
        MAT_A_ROWS_SCALE_SIZE = scales_granularity[2]
        MAT_A_COLS_SCALE_SIZE = scales_granularity[0]
        MAT_B_ROWS_SCALE_SIZE = scales_granularity[
            1
        ] if transpose_b else scales_granularity[2]
        MAT_B_COLS_SCALE_SIZE = scales_granularity[
            2
        ] if transpose_b else scales_granularity[1]

    else:
        var a_scale_0 = Int(a_scales.dim[0]())
        # var a_scale_1 = Int(a_scales.dim[1]())
        var b_scale_0 = Int(b_scales.dim[0]())
        var b_scale_1 = Int(b_scales.dim[1]())
        MAT_A_ROWS_SCALE_SIZE = K // a_scale_0
        # MAT_A_COLS_SCALE_SIZE = M // a_scale_1
        MAT_A_COLS_SCALE_SIZE = 1
        MAT_B_ROWS_SCALE_SIZE = (
            N // b_scale_0 if transpose_b else K // b_scale_0
        )
        MAT_B_COLS_SCALE_SIZE = (
            K // b_scale_1 if transpose_b else N // b_scale_1
        )

    var accum = Scalar[accum_type](0)
    for k in range(K):
        var a_val = rebind[Scalar[a_type]](a.load_linear(Index(x, k))).cast[
            accum_type
        ]()
        var a_scale_factor = rebind[Scalar[a_scales_type]](
            a_scales.load_linear(
                Index(
                    k // MAT_A_ROWS_SCALE_SIZE,
                    ufloordiv(x, MAT_A_COLS_SCALE_SIZE),
                )
            )
        ).cast[accum_type]()

        var b_val: Scalar[accum_type]
        var b_scale_factor: Scalar[accum_type]

        comptime if transpose_b:
            b_val = rebind[Scalar[b_type]](b.load_linear(Index(y, k))).cast[
                accum_type
            ]()
            b_scale_factor = rebind[Scalar[b_scales_type]](
                b_scales.load_linear(
                    Index(
                        ufloordiv(y, MAT_B_ROWS_SCALE_SIZE),
                        k // MAT_B_COLS_SCALE_SIZE,
                    )
                )
            ).cast[accum_type]()
        else:
            b_val = rebind[Scalar[b_type]](b.load_linear(Index(k, y))).cast[
                accum_type
            ]()
            b_scale_factor = rebind[Scalar[b_scales_type]](
                b_scales.load_linear(
                    Index(
                        k // MAT_B_ROWS_SCALE_SIZE,
                        ufloordiv(y, MAT_B_COLS_SCALE_SIZE),
                    )
                )
            ).cast[accum_type]()

        accum += a_val * b_val * a_scale_factor * b_scale_factor

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c.store_linear(Index(x, y), accum.cast[c_type]())


def naive_blockwise_scaled_fp8_grouped_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    //,
    BLOCK_DIM_N: Int = 32,
    BLOCK_DIM_M: Int = 16,
    transpose_b: Bool = True,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[mut=True, c_type, c_layout, ...],
    a: LayoutTensor[mut=False, a_type, a_layout, ...],
    b: LayoutTensor[mut=False, b_type, b_layout, ...],
    a_scales: LayoutTensor[mut=False, a_scales_type, a_scale_layout, ...],
    b_scales: LayoutTensor[mut=False, b_scales_type, b_scale_layout, ...],
    a_offsets: LayoutTensor[mut=False, a_offsets_type, a_offsets_layout, ...],
    expert_ids: LayoutTensor[
        mut=False, expert_ids_type, expert_ids_layout, ...
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Dispatches the naive blockwise scaled FP8 grouped matmul kernel on the GPU.

    Enqueues ``naive_blockwise_scaled_fp8_grouped_matmul_kernel`` with one
    expert per grid-Z slice, tiling the per-expert ``M_local`` x ``N`` output
    with ``BLOCK_DIM_M`` x ``BLOCK_DIM_N`` thread blocks.

    Args:
        c: Rank-2 output accumulator tensor holding all expert outputs.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-3 FP8 weight tensor indexed by expert, in K-major format.
        a_scales: Per-block scales for ``a`` in M-major format.
        b_scales: Per-block scales for ``b`` indexed by expert.
        a_offsets: Prefix-sum offsets delimiting each expert's rows in ``a``.
        expert_ids: Expert id (or ``-1`` to skip) for each grid-Z slice.
        max_num_tokens_per_expert: Maximum row count assigned to any single expert.
        num_active_experts: Number of active experts to dispatch.
        ctx: Device context used to enqueue the kernel.
    """
    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        transpose_b
    ), "Only support transposed B in grouped fp8 matmul."
    comptime assert a_type == b_type == .float8_e4m3fn, (
        "Only float8_e4m3fn is supported for inputs in grouped blockwise"
        " scaled fp8 matmul"
    )
    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    comptime assert a_offsets_type == .uint32, (
        "Only uint32 is supported for a_offsets in grouped blockwise scaled"
        " fp8 matmul"
    )
    comptime assert expert_ids_type == .int32, (
        "Only int32 is supported for expert_ids in grouped blockwise scaled"
        " fp8 matmul"
    )

    if max_num_tokens_per_expert == 0:
        return

    logger.info("Executing Naive Grouped Blockwise Scaled FP8 GEMM")

    comptime kernel = naive_blockwise_scaled_fp8_grouped_matmul_kernel[
        c_layout,
        a_layout,
        b_layout,
        a_scale_layout,
        b_scale_layout,
        a_offsets_layout,
        expert_ids_layout,
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        a_offsets_type,
        expert_ids_type,
        accum_type,
        transpose_b,
        scales_granularity_mnk,
        elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        a_scales,
        b_scales,
        grid_dim=(
            ceildiv(c.dim(1), BLOCK_DIM_N),
            ceildiv(max_num_tokens_per_expert, BLOCK_DIM_M),
            num_active_experts,
        ),
        block_dim=(BLOCK_DIM_N, BLOCK_DIM_M, 1),
    )


def naive_blockwise_scaled_fp8_grouped_matmul_kernel[
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    accum_type: DType,
    transpose_b: Bool = True,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    a_offsets: LayoutTensor[a_offsets_type, a_offsets_layout, ImmutAnyOrigin],
    expert_ids: LayoutTensor[
        expert_ids_type, expert_ids_layout, ImmutAnyOrigin
    ],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    """Computes one output element per thread for the naive blockwise scaled FP8 grouped matmul GPU kernel.

    Each thread handles a single ``(m_local, n)`` output element within one
    expert's tile, determined by ``block_idx.z`` and the ``a_offsets`` prefix
    sum. The thread loops over the K dimension, loading the expert's ``a`` row
    and ``b`` slice along with their per-block scale factors, accumulates the
    scaled product, and applies the optional elementwise epilogue before
    storing the result.

    Args:
        c: Rank-2 output accumulator tensor holding all expert outputs.
        a: Rank-2 FP8 input matrix in K-major format.
        b: Rank-3 FP8 weight tensor indexed by expert, in K-major format.
        a_offsets: Prefix-sum offsets delimiting each expert's rows in ``a``.
        expert_ids: Expert id (or ``-1`` to skip) for each grid-Z slice.
        a_scales: Per-block scales for ``a`` in M-major format.
        b_scales: Per-block scales for ``b`` indexed by expert.
    """
    comptime assert (
        accum_type == .float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var N = b.dim[1]()
    var K = b.dim[2]()

    # Indices in current expert's matmul tile
    var n = global_idx.x
    var m_local = global_idx.y

    var expert_idx = block_idx.z

    # Determine rows for this expert
    var M_local: Int = Int(a_offsets[expert_idx + 1] - a_offsets[expert_idx])
    if n >= N or m_local >= M_local:
        return

    var MAT_A_ROWS_SCALE_SIZE: Int
    var MAT_A_COLS_SCALE_SIZE: Int
    var MAT_B_ROWS_SCALE_SIZE: Int
    var MAT_B_COLS_SCALE_SIZE: Int

    comptime if scales_granularity_mnk:
        comptime scales_granularity = scales_granularity_mnk.value()
        MAT_A_ROWS_SCALE_SIZE = scales_granularity[2]
        MAT_A_COLS_SCALE_SIZE = scales_granularity[0]
        MAT_B_ROWS_SCALE_SIZE = scales_granularity[1]
        MAT_B_COLS_SCALE_SIZE = scales_granularity[2]

    else:
        var a_s0 = a_scales.dim(0)
        var a_s1 = a_scales.dim(1)
        var b_s0 = b_scales.dim(1)
        var b_s1 = b_scales.dim(2)
        MAT_A_ROWS_SCALE_SIZE = K // a_s0
        MAT_A_COLS_SCALE_SIZE = c.dim(0) // a_s1
        MAT_B_ROWS_SCALE_SIZE = N // b_s0
        MAT_B_COLS_SCALE_SIZE = K // b_s1

    var a_start_row = Int(a_offsets[expert_idx])
    var expert = Int(expert_ids[expert_idx])
    var skip = expert == -1
    var accum = Scalar[accum_type](0)

    if not skip:
        var m_global = a_start_row + m_local
        var a_row_ptr = a.ptr + m_global * K
        var b_expert_ptr = b.ptr + expert * N * K
        for k in range(K):
            var a_val = a_row_ptr[k].cast[accum_type]()
            var a_scale = rebind[Scalar[a_scales_type]](
                a_scales[
                    k // MAT_A_ROWS_SCALE_SIZE,
                    m_global // MAT_A_COLS_SCALE_SIZE,
                ]
            ).cast[accum_type]()
            var b_val = b_expert_ptr[n * K + k].cast[accum_type]()
            var b_scale = rebind[Scalar[b_scales_type]](
                b_scales[
                    expert,
                    n // MAT_B_ROWS_SCALE_SIZE,
                    k // MAT_B_COLS_SCALE_SIZE,
                ]
            ).cast[accum_type]()
            accum += a_val * b_val * a_scale * b_scale

    comptime if elementwise_lambda_fn:
        comptime ep = elementwise_lambda_fn.value()
        ep[c_type, 1](Index(a_start_row + m_local, n), accum.cast[c_type]())
    else:
        var c_ptr = c.ptr + a_start_row * N
        c_ptr[m_local * N + n] = accum.cast[c_type]()


########################################################
# FP8 E4M3FN to E4M3FNUZ Conversion for AMD GPUs
########################################################


@always_inline
def convert_e4m3fn_to_e4m3fnuz(
    input_buffer: TileTensor[mut=False, .float8_e4m3fn, ...],
    output_buffer: TileTensor[mut=True, .float8_e4m3fnuz, ...],
    context: DeviceContext,
) raises:
    """Convert E4M3FN weights to E4M3FNUZ format for AMD GPU compatibility.

    This conversion handles the key differences between E4M3FN and E4M3FNUZ:
    1. The bit pattern 10000000 (-128) represents zero in E4M3FN but NaN in E4M3FNUZ

    Args:
        input_buffer: Input tensor in E4M3FN format.
        output_buffer: Output tensor to store E4M3FNUZ format.
        context: Device context for kernel execution.
    """
    comptime assert input_buffer.rank == 2, "expected rank-2 input"
    comptime assert output_buffer.rank == 2, "expected rank-2 output"
    debug_assert(
        Int(input_buffer.dim[0]()) == Int(output_buffer.dim[0]())
        and Int(input_buffer.dim[1]()) == Int(output_buffer.dim[1]()),
        "Input and output shapes must match",
    )

    @always_inline
    @__parameter
    @__copy_capture(input_buffer, output_buffer)
    def convert_kernel[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var input_vec_e4m3fn = input_buffer.load_linear[width](idx)
        var input_vec_int8 = bitcast[.int8](input_vec_e4m3fn)

        comptime ROCM_FP8_NAN_AS_INT = -128

        input_vec_int8 = input_vec_int8.eq(ROCM_FP8_NAN_AS_INT).select(
            Int8(0), input_vec_int8
        )
        var output_vec = bitcast[.float8_e4m3fnuz](input_vec_int8)
        output_buffer.store_linear(idx, output_vec)

    comptime target_simd_width = simd_width_of[
        DType.float8_e4m3fn, target=get_gpu_target()
    ]()

    def convert_kernel_unified[width: Int, alignment: Int = 1](idx: Coord) {}:
        convert_kernel[width, idx.rank, alignment](coord_to_index_list(idx))

    _elementwise_impl_gpu[
        simd_width=target_simd_width,
        trace_description="fp8_e4m3fn_to_e4m3fnuz_convert",
    ](
        convert_kernel_unified,
        shape=(
            Int(input_buffer.dim[0]()),
            Int(input_buffer.dim[1]()),
        ),
        ctx=context,
    )


########################################################
# SM100 Blockwise Scaled FP8 + FP32 with normal epilogue kernel dispatch
########################################################


def blockwise_scaled_fp8_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    scales_granularity_mnk: IndexList[3],
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Our sm100 blockwise scaled fp8 matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 blockwise scaled fp8 matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations. For non B200 GPUs, we use the naive blockwise scaled fp8 matmul which support normal epilogue natively.
    Callers must allocate `c`; when an `elementwise_lambda_fn` is supplied
    the matmul result is written into `c` and then read back by the lambda.
    """

    # 1D/2D blockwise scaling with (m, n, k) granularity in
    # (1, {64,128}, 128).
    comptime if (
        _is_sm10x_gpu(ctx.default_device_info)
        and transpose_b
        and c_type == .bfloat16
        and scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[2] == 128
        and scales_granularity_mnk[1] in (64, 128)
    ):
        comptime N_G = scales_granularity_mnk[1]
        comptime MMA_K = 32
        comptime block_tile_shape = Index(64, 96, 128)
        # MMA_N must be <= 2 * n_scale_granularity because the accumulator
        # only loads 2 b_scales per MMA tile.
        comptime umma_shape = Index(128, 192, MMA_K) if N_G == 128 else Index(
            128, 128, MMA_K
        )
        comptime cluster_shape = Index(2, 1, 1)
        comptime matmul_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=umma_shape,
            cta_group=2,
            gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
        )

        comptime if not elementwise_lambda_fn:
            blockwise_fp8_matmul[
                transpose_b=transpose_b,
                a_scales_type=a_scales_type,
                b_scales_type=b_scales_type,
                config=matmul_config,
                n_scale_granularity=scales_granularity_mnk[1],
            ](c, a, b, a_scales, b_scales, ctx)
        else:
            comptime epilogue = elementwise_lambda_fn.value()
            # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
            # arch support 32B load/store to global memory, see KERN-2037.
            comptime use_32b_simd = (
                has_nvidia_gpu_accelerator()
                and ctx.default_device_info.compute >= B200.compute
            )
            comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
                simd_width_of[c_type, target=get_gpu_target()]()
            )

            var m = Int(c.dim[0]())
            var n = Int(c.dim[1]())

            def epilogue_wrapper[
                simd_width: Int, alignment: Int = 1
            ](idx: Coord) {var}:
                var c_val = c.load[simd_width](idx)
                epilogue[c_type, simd_width, alignment=alignment](
                    Index(idx[0].value(), idx[1].value()), c_val
                )

            blockwise_fp8_matmul[
                transpose_b=transpose_b,
                a_scales_type=a_scales_type,
                b_scales_type=b_scales_type,
                config=matmul_config,
                n_scale_granularity=scales_granularity_mnk[1],
            ](c, a, b, a_scales, b_scales, ctx)
            elementwise[simd_size, target="gpu"](
                epilogue_wrapper, Coord(m, n), ctx
            )

    else:
        # For non B200 GPUs, use the naive blockwise scaled fp8 matmul
        # which supports normal epilogue natively. Convert TileTensors to
        # LayoutTensors for the LayoutTensor overload.
        naive_blockwise_scaled_fp8_matmul[
            transpose_b=transpose_b,
            scales_granularity_mnk=scales_granularity_mnk,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c.to_layout_tensor(),
            a.to_layout_tensor(),
            b.to_layout_tensor(),
            a_scales.to_layout_tensor(),
            b_scales.to_layout_tensor(),
            ctx,
        )
        return
