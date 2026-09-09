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
"""RMSNorm with fused residual connection for state space models."""

from std.math import align_down, align_up, ceildiv, rsqrt
from std.sys.info import align_of, simd_width_of, size_of

from std.algorithm import vectorize
from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from max.gpu.host.info import is_gpu
from max.gpu.memory import external_memory
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import TensorLayout, TileTensor
from std.random import Random

from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import IndexList
from std.utils.numerics import get_accum_type

from nn.normalization import _rms_norm_gpu_block_subkernel, _sum_to_mean


# ===----------------------------------------------------------------------=== #
# CPU Implementations
# ===----------------------------------------------------------------------=== #


def _rms_norm_fused_residual_cpu_2d[
    dtype: DType,
    InputFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    ResidualReadFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    residual_read_fn: ResidualReadFnType,
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    out_shape: IndexList[2],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
):
    """Core 2D implementation of RMSNorm with fused residual.

    Uses simple (row, col) indexing to avoid compile-time evaluation issues.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    comptime simd_width = simd_width_of[dtype]()
    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # Calculate dropout scale if needed
    var dropout_scale = Scalar[dtype](1.0)
    var zero_scalar = Scalar[dtype](0.0)
    if dropout_p > zero_scalar:
        var one_scalar = Scalar[dtype](1.0)
        dropout_scale = one_scalar / (one_scalar - dropout_p)

    for var row in range(num_rows):
        # First compute sum of squared (input + residual) for RMSNorm
        var sum_simd = SIMD[intermediate_type, simd_width]()

        # SIMD loop
        for col in range(0, simd_loop_end, simd_width):
            var input_vals = input_fn[simd_width](row, col)
            var residual_vals = residual_input_fn[simd_width](row, col)

            # Apply dropout if enabled
            if dropout_p > zero_scalar:
                comptime for i in range(simd_width):
                    var element_offset = row * num_cols + col + i
                    var generator = Random(
                        seed=seed, offset=UInt64(element_offset)
                    )
                    var rng = generator.step_uniform()
                    var rng_val = rng[0].cast[dtype]()
                    if rng_val >= dropout_p:
                        input_vals[i] = input_vals[i] * dropout_scale
                    else:
                        input_vals[i] = zero_scalar

            var sum_vals = input_vals + residual_vals

            # Output pre-normalized value (x + residual)
            output_residual_fn[simd_width, 1](row, col, sum_vals)

            # Accumulate for RMSNorm
            sum_simd += sum_vals.cast[intermediate_type]() ** 2

        # Scalar loop for remainder
        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            var input_val = input_fn[1](row, col)[0]
            var residual_val = residual_input_fn[1](row, col)[0]

            # Apply dropout if enabled
            if dropout_p > zero_scalar:
                var element_offset = row * num_cols + col
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p:
                    input_val = input_val * dropout_scale
                else:
                    input_val = zero_scalar

            var sum_val_scalar = input_val + residual_val

            # Output pre-normalized value
            output_residual_fn[1, 1](row, col, sum_val_scalar)

            # Accumulate for RMSNorm
            sum_val += sum_val_scalar.cast[intermediate_type]() ** 2

        # Compute normalization factor
        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(mean_val + epsilon.cast[intermediate_type]())

        # Second pass: apply normalization
        def _normalize[
            sw: Int
        ](col: Int) {gamma, weight_offset, residual_read_fn, output_fn, mut}:
            # Read the pre-computed sum values (input + residual) from first pass
            var sum_vals = residual_read_fn[sw](row, col).cast[
                intermediate_type
            ]()

            var gamma_val = gamma.raw_load[width=sw](col)
            var norm_val: SIMD[dtype, sw]

            if multiply_before_cast:
                var gamma_offset = gamma_val + weight_offset
                norm_val = (sum_vals * norm_factor).cast[dtype]() * gamma_offset
            else:
                norm_val = (sum_vals * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset
                )

            output_fn[sw, 1](row, col, norm_val)

        vectorize[simd_width](num_cols, _normalize)


def rms_norm_fused_residual_cpu[
    dtype: DType,
    rank: Int,
    InputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    ResidualReadFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    residual_read_fn: ResidualReadFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Generic rank wrapper that delegates to the 2D core implementation.

    Creates 2D wrapper lambdas that translate (row, col) to IndexList[rank]
    at runtime, avoiding compile-time evaluation issues with _lambda_load.

    Parameters:
        dtype: Element data type of the tensors (inferred).
        rank: Tensor rank of `shape` (inferred).
        InputFnType: Type of the `input_fn` lambda (inferred).
        ResidualInputFnType: Type of the `residual_input_fn` lambda (inferred).
        OutputFnType: Type of the `output_fn` lambda (inferred).
        OutputResidualFnType: Type of the `output_residual_fn` lambda
            (inferred).
        ResidualReadFnType: Type of the `residual_read_fn` lambda (inferred).
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype (defaults to `True`).

    Args:
        input_fn: Lambda that loads the primary input `SIMD[dtype, width]`
            at a given `IndexList[rank]`.
        residual_input_fn: Lambda that loads the residual input at a given
            index.
        output_fn: Lambda that stores the normalized output at a given
            index.
        output_residual_fn: Lambda that stores the summed residual (input
            plus residual) before normalization.
        residual_read_fn: Lambda that re-reads the summed residual during
            the second normalization pass.
        shape: Shape of the tensors; the last dimension is normalized over.
        gamma: Scale (weight) vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        dropout_p: Dropout probability applied to the primary input before
            the residual add (defaults to 0.0, which disables dropout).
        seed: RNG seed for dropout.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    # Create 2D wrapper lambdas that translate indices at runtime
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {shape, input_fn} -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank](indices)

    @always_inline
    def residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {shape, residual_input_fn} -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width, rank](indices)

    @always_inline
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        shape, output_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices, val)

    @always_inline
    def output_residual_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        shape, output_residual_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices, val)

    @always_inline
    def residual_read_fn_2d[
        sw: Int
    ](row: Int, col: Int) {shape, residual_read_fn} -> SIMD[dtype, sw]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_read_fn[sw, rank](indices)

    # Call the 2D core implementation
    _rms_norm_fused_residual_cpu_2d[multiply_before_cast=multiply_before_cast](
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        residual_read_fn_2d,
        gamma,
        epsilon,
        weight_offset,
        out_shape=IndexList[2](prod_all_but_last_dim, last_dim),
        dropout_p=dropout_p,
        seed=seed,
    )


def _rms_norm_fused_residual_cpu_entry[
    dtype: DType,
    rank: Int,
    InputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """CPU entry point that builds the residual read closure and dispatches.

    The op registration calls this for the CPU target, passing unified closures
    that capture the underlying tensors. The CPU kernel reads back the
    `input + residual` values it wrote in its first pass; since we have no
    direct handle to that buffer here, we recompute them from the input
    closures, matching the first pass exactly. Keeping this on the CPU path lets
    the whole chain use runtime closures end to end, while the GPU path keeps
    its `capturing` comptime closures (see `_rms_norm_fused_residual_impl`).
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    # Note: we only support reduction along the last dimension
    if Int(gamma.dim[0]()) != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.dim[0]())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @always_inline
    def residual_read_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) {
        input_fn, residual_input_fn, dropout_p, seed, shape
    } -> SIMD[dtype, width]:
        var input_vals = input_fn[width, _rank](coords)
        var residual_vals = residual_input_fn[width, _rank](coords)

        # Apply dropout if enabled (matching first pass exactly)
        var zero_scalar = Scalar[dtype](0.0)
        if dropout_p > zero_scalar:
            var one_scalar = Scalar[dtype](1.0)
            var dropout_scale = one_scalar / (one_scalar - dropout_p)
            var last_dim = shape[_rank - 1]
            var row = coords.flattened_length() // last_dim

            comptime for i in range(width):
                var col_idx = coords[_rank - 1] + i
                var element_offset = row * last_dim + col_idx
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p:
                    input_vals[i] = input_vals[i] * dropout_scale
                else:
                    input_vals[i] = zero_scalar

        return input_vals + residual_vals

    rms_norm_fused_residual_cpu[multiply_before_cast=multiply_before_cast](
        input_fn,
        residual_input_fn,
        output_fn,
        output_residual_fn,
        residual_read_fn,
        shape,
        gamma,
        epsilon,
        weight_offset,
        dropout_p,
        seed,
    )


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


@__name(
    t"rms_norm_fused_residual_gpu_block_{dtype}_{multiply_before_cast}",
)
def rms_norm_fused_residual_gpu_block[
    dtype: DType,
    GammaLayout: TensorLayout,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDLength, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, GammaLayout, MutAnyOrigin],
    epsilon: Float32,
    weight_offset: Float32,
    num_cols: Int32,
    dropout_p: Float32 = Float32(0.0),
    seed: UInt64 = 0,
):
    var _num_cols = Int(num_cols)
    var _epsilon = Scalar[dtype](epsilon)
    var _weight_offset = Scalar[dtype](weight_offset)
    var _dropout_p = Scalar[dtype](dropout_p)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var shared_mem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=align_of[SIMD[dtype, simd_width]](),
        name="intermediate_shared_memory",
    ]()
    with PDL():
        # First stage: apply dropout, add residual to input and store in shared memory.
        # Loop to handle cases where _num_cols > block_dim * simd_width,
        # matching the loop structure in _rms_norm_gpu_block_subkernel.
        var tid = thread_idx.x
        var row = block_idx.x

        for x in range(ceildiv(_num_cols // simd_width, block_dim.x)):
            var idx = x * block_dim.x * simd_width + tid * simd_width

            if idx < _num_cols:
                var input_val = input_fn[simd_width](row, idx)

                # Apply dropout if enabled
                var zero_scalar = Scalar[dtype](0.0)
                if _dropout_p > zero_scalar:
                    var one_scalar = Scalar[dtype](1.0)
                    var dropout_scale = one_scalar / (one_scalar - _dropout_p)

                    for i in range(simd_width):
                        if idx + i < _num_cols:
                            # Use element position as offset for RNG to ensure different values per element
                            var element_offset = (
                                UInt64(row) * UInt64(_num_cols)
                                + UInt64(idx)
                                + UInt64(i)
                            )
                            var generator = Random(
                                seed=seed, offset=element_offset
                            )
                            var rng = generator.step_uniform()
                            var rng_val = rng[0].cast[dtype]()
                            if rng_val >= _dropout_p:
                                input_val[i] = input_val[i] * dropout_scale
                            else:
                                input_val[i] = zero_scalar

                var residual_val = residual_input_fn[simd_width](row, idx)
                var residual_add_val = input_val + residual_val

                # Output the pre-normalized value (x + residual) for prenorm mode
                output_residual_fn[
                    simd_width, align_of[SIMD[dtype, simd_width]]()
                ](row, idx, residual_add_val)

                # Store in shared memory for normalization
                shared_mem.store[
                    width=simd_width,
                    alignment=align_of[SIMD[dtype, simd_width]](),
                ](idx, residual_add_val)

        barrier()

        # Second stage: apply RMSNorm using shared memory as input
        @__parameter
        @always_inline
        @__copy_capture(shared_mem)
        def shared_mem_input_fn[
            width: Int
        ](row: Int, col: Int) -> SIMD[dtype, width]:
            return shared_mem.load[width=width](col)

        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            shared_mem_input_fn,
            output_fn,
            multiply_before_cast,
        ](gamma, epsilon, _weight_offset, _num_cols)


def rms_norm_fused_residual_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_residual_fn: def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_fn: def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Dispatches the fused RMS-norm-plus-residual kernel on GPU.

    Selects the optimal SIMD width and warp count for `shape`, then enqueues
    `rms_norm_fused_residual_gpu_block` on `ctx`. Input, residual, normalized
    output, and residual output are all accessed through caller-supplied
    lambdas, enabling fusion with arbitrary prologue/epilogue patterns.

    Parameters:
        dtype: Element data type.
        rank: Tensor rank of `shape`.
        input_fn: Lambda that loads a `SIMD[dtype, width]` for a given index.
        residual_input_fn: Lambda that loads the residual value for a given index.
        output_residual_fn: Lambda that stores the summed residual value.
        output_fn: Lambda that stores the normalized output value.
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype.

    Args:
        shape: Shape of the input tensor (the last dim is normalized over).
        gamma: Scale vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        ctx: Device context for GPU execution.
        dropout_p: Dropout probability; 0.0 disables dropout.
        seed: RNG seed for dropout.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    @__parameter
    @always_inline
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @__parameter
    @always_inline
    def output_residual_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices.canonicalize(), val)

    @__parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @__parameter
    @always_inline
    def residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width](indices.canonicalize())

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        align_up(ceildiv(cols, simd_width), WARP_SIZE),
        WARP_SIZE * max_warps_per_block,
    )

    var shared_mem_size = align_up(cols, simd_width) * size_of[dtype]()

    comptime kernel = rms_norm_fused_residual_gpu_block[
        GammaLayout=type_of(gamma).LayoutType,
        simd_width,
        max_warps_per_block,
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        multiply_before_cast=multiply_before_cast,
    ]
    ctx.enqueue_function[kernel](
        gamma,
        epsilon.cast[.float32](),
        weight_offset.cast[.float32](),
        Int32(cols),
        dropout_p.cast[.float32](),
        seed,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(PDLLevel.ON),
        shared_mem_bytes=shared_mem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(
                ctx.default_device_info.shared_memory_per_multiprocessor - 4096
            )
        ),
    )


def _rms_norm_fused_residual_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert is_gpu[target](), (
        "`_rms_norm_fused_residual_impl` only handles the GPU path; the CPU"
        " path is dispatched directly from the op registration via"
        " `_rms_norm_fused_residual_cpu_entry`, where the runtime closures can"
        " capture the underlying tensors."
    )

    # Note: we only support reduction along the last dimension
    if Int(gamma.dim[0]()) != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.dim[0]())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    rms_norm_fused_residual_gpu[
        input_0_fn,
        input_1_fn,
        output_residual_fn,
        output_fn,
        multiply_before_cast=multiply_before_cast,
    ](
        shape,
        gamma,
        epsilon,
        weight_offset,
        ctx,
        dropout_p,
        seed,
    )


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


@always_inline
def rms_norm_fused_residual[
    dtype: DType,
    rank: Int,
    //,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: def[width: SIMDLength, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDLength, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Applies fused residual add and RMS layer normalization.

    Computes `output = rms_norm(input + residual) * (gamma + weight_offset)`
    and writes the updated residual `input + residual` to `output_residual_fn`.
    Dispatches to a CPU or GPU implementation based on `target`.

    All tensor accesses go through caller-supplied lambdas, which lets the
    graph compiler fuse adjacent elementwise prologue/epilogue operations
    without materializing intermediate buffers.

    Parameters:
        dtype: Element data type.
        rank: Tensor rank of `shape`.
        input_0_fn: Lambda that loads the primary input.
        input_1_fn: Lambda that loads the residual input.
        output_0_fn: Lambda that stores the normalized output.
        output_residual_fn: Lambda that stores the summed residual before
            normalization.
        target: Compilation target, e.g. `"cpu"` or `"gpu"`.
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype.

    Args:
        shape: Shape of the tensors; the last dimension is normalized over.
        gamma: Scale (weight) vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        ctx: Device context for GPU execution; unused on CPU.
        dropout_p: Dropout probability applied to the primary input before the
            residual add; 0.0 disables dropout.
        seed: RNG seed for dropout.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    @always_inline
    @__parameter
    def output_fn_wrapper[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @__parameter
    def output_residual_fn_wrapper[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_residual_fn[width, rank, alignment](idx, val)

    @always_inline
    def description_fn() {imm} -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_residual",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        _rms_norm_fused_residual_impl[
            dtype,
            rank,
            input_0_fn,
            input_1_fn,
            output_fn_wrapper,
            output_residual_fn_wrapper,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx,
            dropout_p,
            seed,
        )
