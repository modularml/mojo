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
"""Provides numerically stable softmax kernels for CPU and GPU, including fused and online variants."""

from std.math import align_down, ceildiv, exp, exp2, log
from std.builtin.device_passable import DevicePassable
from std.math.uutils import umod, ufloordiv, udivmod
from std.collections import Optional, OptionalReg

from std.sys import align_of, is_amd_gpu, is_nvidia_gpu, simd_width_of, size_of
from std.sys._assembly import inlined_assembly

import max.gpu.primitives.warp as warp
from std.algorithm import vectorize

from max.algorithm import sync_parallelize
from std.algorithm.backend.unswitch import unswitch
from max.algorithm.backend.gpu.reduction import block_reduce, row_reduce
from max.algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _reduce_generator,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceAttribute, DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu, is_gpu
from max.gpu.primitives import block
from layout._utils import idx2crd
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    LayoutTensor,
    DefaultEngine,
    RowMajorLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from layout.tile_layout import Layout as InternalLayout
from layout.tensor_core import get_fragment_size
from std.memory import unsafe_stack_allocation
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils import IndexList, StaticTuple
from std.utils.coord import ComptimeInt, Coord, CoordLike
from std.utils.index import product
from std.utils.numerics import get_accum_type, min_or_neg_inf

# Free-form row-wise scaffolder (Row) + monoids.
from algorithm import rowwise
from algorithm.rowwise_types import RowCoord
from algorithm.reduce_op import ReduceMax, ReduceSum

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


def reduce_add_simd[
    simd_width: SIMDLength,
    step_simd_width: SIMDLength,
    dtype: DType,
](
    mut scalar: Scalar[dtype],
    mut vector: SIMD[dtype, simd_width],
    val: SIMD[dtype, step_simd_width],
):
    """This functions adds val to either the scalar value or the vector value
    depending on the step_simd_width. This is useful when the simd_width varies
    between iterations as in vectorize.

    Parameters:
        simd_width: The full SIMD width of the `vector` accumulator.
        step_simd_width: The width of the current step's `val`; when 1,
            `val` is accumulated into `scalar`, otherwise into `vector`.
        dtype: The element type of the accumulators and `val`.

    Args:
        scalar: Scalar accumulator for single-element steps; updated in
            place when `step_simd_width` is 1.
        vector: SIMD accumulator for full-width steps; updated in place
            when `step_simd_width` matches `simd_width`.
        val: The partial reduction value to accumulate into either
            `scalar` or `vector`.
    """

    comptime if step_simd_width == 1:
        # When the step_simd_width is 1, then we add to the scalar value.
        scalar += val[0]
    else:
        # When the step_simd_Width is the same as the simd_width, then we add to
        # the vector value.
        vector += rebind[SIMD[dtype, simd_width]](val)


@always_inline
def sub(x: SIMD, y: type_of(x)) -> type_of(x):
    """Returns the element-wise difference `x - y`.

    Args:
        x: The minuend SIMD vector.
        y: The subtrahend SIMD vector; must have the same type as `x`.

    Returns:
        A SIMD vector with each element equal to `x[i] - y[i]`.
    """
    return x - y


@always_inline
def mul(x: SIMD, y: type_of(x)) -> type_of(x):
    """Returns the element-wise product `x * y`.

    Args:
        x: The first SIMD vector multiplicand.
        y: The second SIMD vector multiplicand; must have the same type as `x`.

    Returns:
        A SIMD vector with each element equal to `x[i] * y[i]`.
    """
    return x * y


@always_inline
def identity(x: SIMD) -> type_of(x):
    """Returns the input SIMD vector unchanged.

    Args:
        x: The input SIMD vector.

    Returns:
        `x` unmodified.
    """
    return x


@always_inline
def reciprocal(x: SIMD) -> type_of(x):
    """Returns the element-wise reciprocal `1 / x`.

    Args:
        x: The input SIMD vector.

    Returns:
        A SIMD vector with each element equal to `1 / x[i]`.
    """
    return 1 / x


@always_inline
def _exp_concrete(x: SIMD) -> type_of(x):
    """The concrete implementation of the exp function.

    This is a helper function that is used to provide a concrete implementation
    of the exp function. This is necessary because exp uses the _Expable trait
    and mojo cannot disambiguate between the different exp functions otherwise.
    """
    comptime assert x.dtype.is_floating_point(), "dtype must be floating point"
    return exp(x)


@always_inline
def _exp2_concrete(x: SIMD) -> type_of(x):
    """The concrete implementation of the exp2 function."""
    comptime assert x.dtype.is_floating_point(), "dtype must be floating point"
    return exp2(x)


@always_inline
def _log_concrete(x: SIMD) -> type_of(x):
    """The concrete implementation of the log function."""
    comptime assert x.dtype.is_floating_point(), "dtype must be floating point"
    return log(x)


# Packed f32x2 FMA/add (`fma.rn.ftz.f32x2` / `add.ftz.f32x2`). Mojo does not
# fold a SIMD[f32,2] mul+add into one FFMA2, so the SM100 softmax folds the
# scale and pairs the row-sum via these explicit PTX ops -- same idiom the dense
# FA4 path uses (sm100/attention_utils.mojo). Gated comptime-OFF for the
# generic helpers below; only the MSA single-tile path opts in.
@always_inline
def _fma_f32x2(
    a: SIMD[.float32, 2],
    b: SIMD[.float32, 2],
    c: SIMD[.float32, 2],
) -> SIMD[.float32, 2]:
    return inlined_assembly[
        "fma.rn.ftz.f32x2 $0, $1, $2, $3;",
        SIMD[.float32, 2],
        constraints="=l,l,l,l",
        has_side_effect=False,
    ](a, b, c)


@always_inline
def _add_f32x2(a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    return inlined_assembly[
        "add.ftz.f32x2 $0, $1, $2;",
        SIMD[.float32, 2],
        constraints="=l,l,l",
        has_side_effect=False,
    ](a, b)


# ===-----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===-----------------------------------------------------------------------===#


def _softmax_2_pass_step1[
    simd_width: Int,
    dtype: DType,
](input: TileTensor[mut=False, dtype, ...]) -> StaticTuple[Scalar[dtype], 2]:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert input.rank == 1
    # STEP 1: find the runningMax and runningSum in each batch.
    #   runningMax = -∞
    #   runningSum = 0
    #   STAGE 1:
    #   for i = 0 to N do
    #     newMax = max(runningMax, Input[i])
    #     runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
    #     runningMax = newMax
    #   end for
    #   return runningMax, runningSum

    var running_max_vec = SIMD[dtype, simd_width](min_or_neg_inf[dtype]())
    var running_sum_vec = SIMD[dtype, simd_width](0)

    var length = input.num_elements()
    var vector_end = align_down(length, simd_width)

    for i in range(0, vector_end, simd_width):
        var simd_elem = input.load_linear[width=simd_width, alignment=1](
            IndexList[1](i)
        )
        var new_max_vec = SIMD[dtype, simd_width](
            max(running_max_vec, simd_elem).reduce_max()
        )
        running_sum_vec = running_sum_vec * exp(
            running_max_vec - new_max_vec
        ) + exp(simd_elem - new_max_vec)
        running_max_vec = new_max_vec

    var running_max = running_max_vec.reduce_max()
    var running_sum = running_sum_vec.reduce_add()

    for i in range(vector_end, length):
        var elem = input.load_linear[width=1, alignment=1](IndexList[1](i))
        var new_max = max(running_max, elem)
        running_sum = running_sum * exp(running_max - new_max) + exp(
            elem - new_max
        )
        running_max = new_max

    return StaticTuple[Scalar[dtype], 2](running_max[0], running_sum[0])


def _softmax_2_pass_step2[
    simd_width: Int,
    unroll_factor: Int,
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    running_max: Scalar[dtype],
    running_sum: Scalar[dtype],
):
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert input.rank == 1
    comptime assert output.rank == 1

    # Step 2:
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    @always_inline
    def _step_2[
        simd_width: Int
    ](idx: Int) {running_max, running_sum, input, output, mut}:
        var running_max_simd = SIMD[dtype, simd_width](running_max)
        var running_sum_simd = SIMD[dtype, simd_width](running_sum)
        var input_val = input.load_linear[width=simd_width, alignment=1](
            IndexList[1](idx)
        )
        output.store_linear[width=simd_width, alignment=1](
            IndexList[1](idx),
            exp(input_val - running_max_simd) / running_sum_simd,
        )

    vectorize[simd_width, unroll_factor=unroll_factor](
        output.num_elements(), _step_2
    )


def softmax_2_pass[
    simd_width: Int,
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
):
    """Performs an unbatched softmax on an input tensor using the two-pass
    online algorithm.

    The unbatched two-pass online softmax is described in "Online
    normalizer calculation for softmax" (https://arxiv.org/abs/1805.02867) and
    "A full-stack search technique for domain optimized deep learning
    accelerators" (https://dl.acm.org/doi/abs/10.1145/3503222.3507767) and is
    defined as:

        procedure SoftmaxUnbatched(InputInput)
          runningMax = -∞
          runningSum = 0
          STAGE 1:
          for i = 0 to N do
            newMax = max(runningMax, Input[i])
            runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
            runningMax = newMax
          end for
          for i = 0 to N do
            Output[i] = exp(Input[i] - runningMax) / runningSum
          end for

    Parameters:
        simd_width: The simd_width to use in vectorization.
        dtype: The dtype of the input and output buffers.

    Args:
        output: The output buffer in which to store the softmax values.
        input: The input buffer used to compute the softmax.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert input.rank == output.rank
    comptime assert input.rank == 1

    var running_info = _softmax_2_pass_step1[simd_width, dtype](input)

    var running_max = running_info[0]
    var running_sum = running_info[1]

    comptime unroll_factor = 8  # TODO: search
    _softmax_2_pass_step2[simd_width, unroll_factor, dtype](
        output, input, running_max, running_sum
    )


# ===-----------------------------------------------------------------------===#
# Softmax 3 Pass
# ===-----------------------------------------------------------------------===#


def _softmax_3_pass_step_2[
    simd_width: Int,
    unroll_factor: Int,
    dtype: DType,
    input_fn_1d: def[_simd_width: Int](Int) capturing[_] -> SIMD[
        dtype, _simd_width
    ],
    pre_update_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
    post_update_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
](
    output: TileTensor[mut=True, dtype, ...],
    max_val: Scalar[dtype],
) -> Scalar[
    dtype
]:
    comptime assert output.rank == 1
    # STEP 2: compute for each batch
    # for i = 0 to N do
    #   Output[i] = pre_update_func(Input[i] - max_val)
    #   accum += post_update_func(Output[i])
    # end for
    comptime outer_simd_width = simd_width

    var accum_scalar: Scalar[dtype] = 0
    var accum_simd: SIMD[dtype, outer_simd_width] = 0

    @always_inline
    def step_2[simd_width: Int](idx: Int) {max_val, output, mut}:
        var vin = input_fn_1d[simd_width](idx)
        var elem = vin - SIMD[dtype, simd_width](max_val)

        elem = pre_update_func[dtype, simd_width](elem)
        output.store_linear[width=simd_width, alignment=1](
            IndexList[1](idx), elem
        )
        elem = post_update_func[dtype, simd_width](elem)
        reduce_add_simd[outer_simd_width, simd_width, dtype](
            accum_scalar, accum_simd, elem
        )

    vectorize[simd_width, unroll_factor=unroll_factor](
        output.num_elements(), step_2
    )
    # Reduce the values from both the scalar and vector accum.
    return accum_scalar + accum_simd.reduce_add()


def _softmax_3_pass_step_3[
    simd_width: Int,
    unroll_factor: Int,
    dtype: DType,
    accum_proc_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
    accum_apply_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
](output: TileTensor[mut=True, dtype, ...], accum: Scalar[dtype],):
    comptime assert output.rank == 1
    # STEP 3: normalize each batch
    # accum = accum_proc_func(accum)
    # for i = 0 to N do
    #   accum_apply_func(Output[b, i], accum)
    # end for
    var accum_proc = accum_proc_func[dtype, 1](accum)

    @always_inline
    def step_3[simd_width: Int](idx: Int) {var accum_proc, output}:
        var accum_simd = SIMD[dtype, simd_width](accum_proc)
        var elem = output.load_linear[width=simd_width, alignment=1](
            IndexList[1](idx)
        )
        elem = accum_apply_func[dtype, simd_width](elem, accum_simd)
        output.store_linear[width=simd_width, alignment=1](
            IndexList[1](idx), elem
        )

    vectorize[simd_width, unroll_factor=unroll_factor](
        output.num_elements(), step_3
    )


def _softmax_3_pass_base[
    simd_width: Int,
    dtype: DType,
    input_fn_1d: def[_simd_width: Int](Int) capturing[_] -> SIMD[
        dtype, _simd_width
    ],
    step2_pre_update_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
    step2_post_update_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
    step3_accum_proc_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
    step3_accum_apply_func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
](output: TileTensor[mut=True, dtype, ...]) raises:
    """Performs an unbatched three-pass softmax. The actual behavior of each
    step can be different between the (regular) softmax and logsoftmax.

    Parameters:
        simd_width: The simd_width to use in vectorization.
        dtype: The dtype of the input and output buffers.
        input_fn_1d: The elementwise input lambda.
        step2_pre_update_func: Pre update function.
        step2_post_update_func: Post update function.
        step3_accum_proc_func: Pre accumulation function.
        step3_accum_apply_func: Post accumulation function.

    Args:
        output: The output buffer in which to store the softmax values.
    """
    comptime assert output.rank == 1
    # STEP 1 - Calculate max
    # Allocate buffer for max_val
    var max_buff = tt_stack_allocation[dtype=dtype](row_major[1]())

    # Use _reduce_generator to fuse input lambda with max-reduction
    # Reduce function
    @always_inline
    @__parameter
    def reduce_impl[
        ty: DType, width: SIMDLength
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return max(v1, v2)

    # Input function
    # Translate the given input lambda from 1D to n-D because _reduce_generator
    # needs n-D.
    @__parameter
    @always_inline
    def input_fn[
        _dtype: DType, _width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[_dtype, _width]:
        comptime assert _rank == 1
        return rebind[SIMD[_dtype, _width]](input_fn_1d[_width](coords[0]))

    # Output function
    @__parameter
    @always_inline
    def output_fn[
        _dtype: DType, _width: SIMDLength, _rank: Int
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        comptime assert _rank == 1
        max_buff[0] = val.reduce_max().cast[dtype]()

    # Generate fused input-reduction
    _reduce_generator[
        input_fn,
        output_fn,
        reduce_impl,
        reduce_dim=0,
    ](
        Coord((output.num_elements(),)),
        init=Scalar[dtype].MIN,
    )

    var max_val = max_buff[0]

    # STEP 2
    comptime unroll_factor = 8  # TODO: search
    var accum = _softmax_3_pass_step_2[
        simd_width,
        unroll_factor,
        dtype,
        input_fn_1d,
        step2_pre_update_func,
        step2_post_update_func,
    ](output, max_val)

    # STEP 3
    _softmax_3_pass_step_3[
        simd_width,
        unroll_factor,
        dtype,
        step3_accum_proc_func,
        step3_accum_apply_func,
    ](output, accum)


def softmax_3_pass[
    simd_width: Int,
    dtype: DType,
    origins: OriginSet,
    input_fn_1d: def[_simd_width: Int](Int) capturing[origins] -> SIMD[
        dtype, _simd_width
    ],
    logsoftmax: Bool = False,
](output: TileTensor[mut=True, dtype, ...]) raises:
    """Performs an unbatched softmax on an input tensor using the three-pass
    algorithm.

    The unbatched three-pass softmax is defined as:

        procedure SoftmaxUnbatched(InputInput)
          maxVal = -∞
          denom = 0
          STEP 1: find the max value in each batch
          for i = 0 to N do
            maxVal = max(maxVal, Input[b, i])
          end for
          STEP 2: compute the exponential for each batch
          for i = 0 to N do
            Output[b, i] = exp(Input[b, i] - maxVal)
            denom += Output[b, i]
          end for
          STEP 3: normalize each batch
          for i = 0 to N do
            Output[b, i] /= denom
          end for

    Parameters:
        simd_width: The simd_width to use in vectorization.
        dtype: The dtype of the input and output buffers.
        origins: The OriginSet of captured arguments by the input_fn_1d.
        input_fn_1d: The elementwise input lambda.
        logsoftmax: Enable to apply elementwise log() to outputs after softmax.

    Args:
        output: The output buffer in which to store the softmax values.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert output.rank == 1

    comptime if logsoftmax:
        _softmax_3_pass_base[
            simd_width,
            dtype,
            input_fn_1d,
            identity,
            _exp_concrete,
            _log_concrete,
            sub,
        ](output)
    else:
        _softmax_3_pass_base[
            simd_width,
            dtype,
            input_fn_1d,
            _exp_concrete,
            identity,
            reciprocal,
            mul,
        ](output)


# ===-----------------------------------------------------------------------===#
# LogSoftmax
# ===-----------------------------------------------------------------------===#


def logsoftmax_inline[
    dtype: DType,
    simd_width: Int,
    rank: Int,
    input_fn: def[_simd_width: Int](Coord) capturing[_] -> SIMD[
        dtype, _simd_width
    ],
    target: StaticString = "cpu",
    has_prologue_fusion: Bool = True,
](
    shape: Coord,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    context: Optional[DeviceContext] = None,
) raises:
    """Computes log-softmax over the given axis using a caller-supplied input lambda.

    Delegates to `softmax_inline` with `logsoftmax=True`, which applies an
    elementwise `log` to the normalized outputs.

    Parameters:
        dtype: The dtype of the input and output buffers.
        simd_width: The simd_width to use in vectorization.
        rank: The rank of the input and output tensors.
        input_fn: The elementwise input lambda.
        target: The target device ("cpu" or "gpu").
        has_prologue_fusion: Whether the input lambda supports prologue fusion.

    Args:
        shape: The shape of the output tensor.
        output: The output buffer in which to store the log-softmax values.
        axis: The axis along which to compute the log-softmax.
        context: Optional device context for GPU execution.
    """
    softmax_inline[
        dtype,
        simd_width,
        rank,
        input_fn,
        target,
        logsoftmax=True,
        has_prologue_fusion=has_prologue_fusion,
    ](shape, output, axis, context)


def logsoftmax_inline[
    dtype: DType,
    simd_width: Int,
    rank: Int,
    target: StaticString = "cpu",
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    context: Optional[DeviceContext] = None,
) raises:
    """Computes log-softmax over the given axis of `input` and stores the result in `output`.

    Wraps `input` with a load lambda and delegates to `softmax_inline` with
    `logsoftmax=True`.

    Parameters:
        dtype: The dtype of the input and output buffers.
        simd_width: The simd_width to use in vectorization.
        rank: The rank of the input and output tensors.
        target: The target device ("cpu" or "gpu").

    Args:
        input: The input buffer used to compute the log-softmax.
        output: The output buffer in which to store the log-softmax values.
        axis: The axis along which to compute the log-softmax.
        context: Optional device context for GPU execution.
    """

    @__parameter
    @always_inline
    def input_fn[_simd_width: Int](coords: Coord) -> SIMD[dtype, _simd_width]:
        return input.load[width=_simd_width, alignment=1](coords)

    softmax_inline[dtype, simd_width, rank, input_fn, target, logsoftmax=True](
        input.layout.shape_coord(),
        output,
        axis,
        context,
    )


# ===-----------------------------------------------------------------------===#
# Softmax
# ===-----------------------------------------------------------------------===#


def _softmax_cpu[
    dtype: DType,
    simd_width: Int,
    rank: Int,
    origins: OriginSet,
    input_fn: def[_simd_width: Int](Coord) capturing[origins] -> SIMD[
        dtype, _simd_width
    ],
    logsoftmax: Bool = False,
](
    shape: Coord,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    # TODO: Add rowwise generator to de-duplicate partitioning logic between
    # softmax and logsoftmax
    if axis != rank - 1:
        raise Error("softmax not supported on non-inner axis yet")

    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))

    if shape_il.flattened_length() == 0:
        return

    var inner_dim = Int(output.dim[rank - 1]())
    var outer_dim = product[rank](shape_il, rank - 1)
    var num_workers = min(parallelism_level(ctx), outer_dim)
    var chunk_size = ceildiv(outer_dim, num_workers)

    @always_inline
    def task_func(
        task_id: Int,
    ) raises {var chunk_size, var inner_dim, var outer_dim, imm}:
        var start_offset = task_id * chunk_size
        var end_offset = min((task_id + 1) * chunk_size, outer_dim)
        for i in range(start_offset, end_offset):
            var buffer_offset = i * inner_dim
            var output_buffer_view = TileTensor(
                output.ptr + buffer_offset,
                row_major(Coord(inner_dim)),
            )
            var indices = _get_nd_indices_from_flat_index(i, shape_il, rank - 1)

            @__parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lambda with some 1D-to-n-D translation logic.
            def input_fn_1d[_width: Int](idx: Int) -> SIMD[dtype, _width]:
                indices[rank - 1] = idx
                return input_fn[_width](Coord(indices))

            softmax_3_pass[
                simd_width,
                dtype,
                origin_of()._mlir_origin,
                input_fn_1d,
                logsoftmax=logsoftmax,
            ](output_buffer_view)
            _ = indices

    sync_parallelize(task_func, num_workers, ctx)


# Softmax (no input lambda)
def softmax_inline[
    dtype: DType,
    simd_width: Int,
    rank: Int,
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
) raises:
    """Computes softmax over the given axis of `input` and stores the result in `output`.

    Wraps `input` with a load lambda and delegates to the main `softmax_inline`
    entry point.

    Parameters:
        dtype: The dtype of the input and output buffers.
        simd_width: The simd_width to use in vectorization.
        rank: The rank of the input and output tensors.

    Args:
        input: The input buffer used to compute the softmax.
        output: The output buffer in which to store the softmax values.
        axis: The axis along which to compute the softmax.
    """

    @__parameter
    @always_inline
    def input_fn[_simd_width: Int](coords: Coord) -> SIMD[dtype, _simd_width]:
        return input.load[width=_simd_width, alignment=1](coords)

    softmax_inline[dtype, simd_width, rank, input_fn](
        input.layout.shape_coord(),
        output,
        axis,
    )


@__name(t"softmax_kernel_{dtype}_{sink}_{logsoftmax}")
def softmax_kernel[
    BLOCK_SIZE: Int,
    input_fn: def[_dtype: DType, _simd_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing[_] -> SIMD[_dtype, _simd_width],
    dtype: DType,
    sink_type: DType,
    rank: Int,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    SinkWeightsLayoutType: TensorLayout,
    accum_type: DType = get_accum_type[dtype](),
    *,
    sink: Bool = False,
    logsoftmax: Bool = False,
](
    shape: IndexList[rank],
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    sink_weights: TileTensor[sink_type, SinkWeightsLayoutType, ImmutAnyOrigin],
):
    """GPU kernel implementing the three-pass softmax with optional sink-attention and logsoftmax variants.

    Each block reduces one row: step 1 finds the row max (optionally clamped
    with a per-head sink weight), step 2 computes `exp(x - max)` and the row
    sum, and step 3 normalizes (and applies `log` when `logsoftmax` is set).

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        input_fn: The elementwise input lambda.
        dtype: The dtype of the input and output buffers.
        sink_type: The dtype of the sink weights.
        rank: The rank of the input and output tensors.
        OutputLayoutType: The layout type of the output tensor.
        output_origin: The origin of the output tensor.
        OutputEngine: Engine of the output tensor.
        SinkWeightsLayoutType: The layout type of the sink weights tensor.
        accum_type: The accumulation dtype (defaults to the accumulation type for `dtype`).
        sink: Whether to apply sink-attention bias to the row max.
        logsoftmax: Enable to apply elementwise log() to outputs after softmax.

    Args:
        shape: The shape of the tensor as an IndexList.
        output: The output buffer in which to store the softmax values.
        sink_weights: Per-head sink weights used when `sink` is True.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime axis = rank - 1

    var row_size: Int = shape[axis]
    var num_rows = ufloordiv(shape.flattened_length(), row_size)

    var max_buf = tt_stack_allocation[dtype=accum_type, address_space=.SHARED](
        row_major[1]()
    )
    var exp_sum_buf = tt_stack_allocation[
        dtype=accum_type, address_space=.SHARED
    ](row_major[1]())

    @__parameter
    @always_inline
    def _max[
        dtype: DType, width: SIMDLength
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(x, y)

    @__parameter
    @always_inline
    def _sum[
        dtype: DType, width: SIMDLength
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    var tid = thread_idx.x

    with PDL():
        # grid stride loop over rows
        # each block reduces a row, which is convenient because it requires no partial
        # reductions across blocks
        for row_idx in range(block_idx.x, num_rows, grid_dim.x):
            var sink_val = Scalar[accum_type].MIN

            # Step 1: compute max in row
            var row_coords = _get_nd_indices_from_flat_index(
                row_idx, shape, axis
            )

            comptime if sink:
                # Sinks are per-head, and the head lives in the OUTERMOST
                # row dim (e.g. attention lays the softmax rows out as
                # `(batch*num_heads, prompt_len, num_keys)`, head-major).
                # Indexing by the flat `row_idx` only recovers the head when
                # `prompt_len == 1` (decode); for prefill (`prompt_len > 1`)
                # it mis-maps the sink to a position instead of a head. Index
                # by the outermost coordinate so `coord % num_sinks == head`
                # holds for any `prompt_len`. For rank-2 inputs
                # `row_coords[0] == row_idx`, so this is a no-op there.
                sink_val = sink_weights.load_linear[width=1](
                    IndexList[1](
                        umod(Int(row_coords[0]), Int(sink_weights.dim[0]()))
                    )
                ).cast[accum_type]()

            var row_max = row_reduce[
                BLOCK_SIZE,
                input_fn,
                _max,
                dtype,
                1,
                rank,
                accum_type=accum_type,
                axis=axis,
            ](row_coords, Scalar[dtype].MIN, row_size)

            comptime if sink:
                row_max = max(row_max, sink_val)

            if tid == 0:
                max_buf[0] = row_max
            barrier()

            row_max = max_buf[0][0]

            # Step 2: out[i] = exp(in[i] - max) and compute sum of out[i]
            var exp_sum = Scalar[accum_type](0)

            for row_offset in range(tid, row_size, BLOCK_SIZE):
                row_coords[axis] = row_offset

                # loads from input_fn twice
                var val = exp(
                    input_fn[dtype, 1, rank](row_coords).cast[accum_type]()
                    - row_max
                )

                # TODO we're writing to and reading from global memory twice
                # we can reduce the amount of reads by keeping values local here.
                output.store_linear(row_coords, val.cast[dtype]())
                exp_sum += val

            var block_exp_sum = block_reduce[BLOCK_SIZE, _sum](exp_sum, 0)

            comptime if sink:
                block_exp_sum += exp(sink_val - row_max)

            if tid == 0:
                exp_sum_buf[0] = block_exp_sum
            barrier()

            # Step 3: Normalize output (and apply log for logsoftmax)
            var block_exp_sum_recip = 1 / exp_sum_buf[0]
            for row_offset in range(tid, row_size, BLOCK_SIZE):
                row_coords[axis] = row_offset
                var normalized = (
                    output.load_linear[width=1](row_coords)
                    * block_exp_sum_recip.cast[dtype]()
                )

                comptime if logsoftmax:
                    normalized = log(normalized)

                output.store_linear(row_coords, normalized)


# TileTensor layout type for 1D row-major tensors with dynamic size,
# used for sink_weights parameters.
comptime _SinkWeightsTTLayout = InternalLayout[
    shape_types=Coord[Int64].element_types,
    stride_types=Coord[ComptimeInt[1]].element_types,
]


@__name(t"softmax_warp_{dtype}_{WARP_ROWS}_fused_{has_prologue_fusion}")
def _softmax_warp_kernel[
    WARP_ROWS: Int,
    has_prologue_fusion: Bool,
    input_fn: def[_dtype: DType, _simd_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing[_] -> SIMD[_dtype, _simd_width],
    dtype: DType,
    rank: Int,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    accum_type: DType = get_accum_type[dtype](),
](
    output: TileTensor[
        mut=True, dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
):
    """Warp-local softmax for short inner axes (no shared memory).

    One warp owns one row; each lane handles one inner-axis element
    (`row_size <= WARP_SIZE`), reduced with warp shuffles (no barriers).
    Assumes a contiguous inner-axis tensor (the only layout MAX produces).
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert accum_type.is_floating_point()
    comptime axis = rank - 1

    var row_size = Int(output.dim[axis]())
    var num_rows = ufloordiv(output.num_elements(), row_size)

    var warp_idx = thread_idx.x // WARP_SIZE
    var lane = Int(lane_id())
    var row_stride = grid_dim.x * WARP_ROWS

    with PDL():
        for row_idx in range(
            block_idx.x * WARP_ROWS + warp_idx, num_rows, row_stride
        ):
            # Flat coords for the (always contiguous) store, and for the load
            # when there is no real input fusion.
            var coords = IndexList[rank](0)
            comptime if rank >= 2:
                coords[axis - 1] = row_idx

            var val = min_or_neg_inf[accum_type]()
            var has_data = lane < row_size
            if has_data:
                comptime if has_prologue_fusion:
                    var load_coords = IndexList[rank](0)
                    var rem = row_idx

                    comptime for di in range(axis):
                        comptime d = axis - 1 - di
                        var dim_d = Int(output.dim[d]())
                        load_coords[d] = umod(rem, dim_d)
                        rem = ufloordiv(rem, dim_d)
                    load_coords[axis] = lane
                    val = input_fn[dtype, 1, rank](load_coords).cast[
                        accum_type
                    ]()[0]
                else:
                    coords[axis] = lane
                    val = input_fn[dtype, 1, rank](coords).cast[accum_type]()[0]

            var row_max = warp.max(SIMD[accum_type, 1](val))[0]

            var local_sum = Scalar[accum_type](0)
            if has_data:
                local_sum = exp(val - row_max)
            var exp_sum = warp.sum(SIMD[accum_type, 1](local_sum))[0]
            var recip = Scalar[accum_type](1) / exp_sum

            if has_data:
                coords[axis] = lane
                # Do not reuse `local_sum` to avoid perf regressions.
                var out_val = (exp(val - row_max) * recip).cast[dtype]()
                output.store_linear[width=1](coords, SIMD[dtype, 1](out_val))


def _softmax_gpu[
    dtype: DType,
    simd_width: Int,
    rank: Int,
    input_fn: def[_simd_width: Int](Coord) capturing[_] -> SIMD[
        dtype, _simd_width
    ],
    *,
    sink: Bool = False,
    sink_type: DType = dtype,
    logsoftmax: Bool = False,
    has_prologue_fusion: Bool = True,
](
    shape: Coord,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        TileTensor[sink_type, _SinkWeightsTTLayout, ImmutAnyOrigin]
    ] = None,
) raises:
    if axis != rank - 1:
        raise Error("softmax not supported on non-inner axis yet")

    @always_inline
    @__parameter
    def input_fn_wrapper[
        _dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_dtype, width]:
        return rebind[SIMD[_dtype, width]](input_fn[width](Coord(idx)))

    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))
    var num_rows = shape_il.flattened_length() // shape_il[axis]
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    comptime sm_overprovision_factor = 32  # tunable

    # Single-pass online softmax. The sink and logsoftmax variants stay on
    # the legacy 3-pass kernel below.
    comptime if not sink and not logsoftmax:
        comptime BLOCK_SIZE = 256
        comptime WARP_ROWS = 4

        # Short inner axes (<32) use a warp-local kernel with one
        # element per lane for coalesced loads. Longer rows stay on the
        # block/online path below.
        def dispatch_warp_or_block[
            use_warp: Bool
        ]() raises {var num_rows, var shape_il, var output, var sm_count, imm}:
            comptime if use_warp:
                comptime WARP_BLOCK_SIZE = WARP_SIZE * WARP_ROWS
                var warp_num_blocks = min(
                    ceildiv(num_rows, WARP_ROWS),
                    sm_overprovision_factor * sm_count,
                )
                comptime warp_kernel = _softmax_warp_kernel[
                    WARP_ROWS,
                    has_prologue_fusion,
                    input_fn_wrapper,
                    dtype,
                    rank,
                    output.LayoutType,
                    output.origin,
                    output.Engine,
                ]
                ctx.enqueue_function[warp_kernel](
                    output,
                    grid_dim=warp_num_blocks,
                    block_dim=WARP_BLOCK_SIZE,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
            else:
                var num_blocks = min(
                    num_rows, sm_overprovision_factor * sm_count
                )

                # Split-K: when `num_rows` blocks under-fill the GPU, split
                # each row's columns across `num_splits` blocks. The per-row
                # reduction serializes, so only more blocks per row helps.
                # `num_splits == 1` keeps the single-block path below.
                comptime blocks_per_sm = 6
                comptime max_splits = 32
                # Below this many tiles per split the two-launch overhead outweighs
                # the parallelism (short rows like 256x4096 regress ~2x).
                comptime min_tiles_per_split = 4
                comptime accum_type = get_accum_type[dtype]()
                var target_blocks = blocks_per_sm * sm_count
                # Widest-tile count so no split ends up empty in either variant.
                var total_tiles = ceildiv(
                    shape_il[axis], BLOCK_SIZE * simd_width
                )
                var num_splits = min(
                    min(max(target_blocks // num_rows, 1), max_splits),
                    max(total_tiles // min_tiles_per_split, 1),
                )

                # Vectorised loads need each row to start on a `simd_width`-element
                # boundary so per-row strides stay aligned, and need enough work
                # per row to amortise the wider tile dispatch. Otherwise downgrade
                # to scalar; `unswitch` lifts the predicate so each kernel variant
                # has one inner-loop shape.
                def dispatch[
                    use_vectorized: Bool
                ]() raises {
                    var num_blocks,
                    var shape_il,
                    var output,
                    var num_splits,
                    imm,
                }:
                    comptime kernel_simd_width = (
                        simd_width if use_vectorized else 1
                    )
                    var null_temp_arr = Optional[
                        UnsafePointer[Float32, ImmutAnyOrigin]
                    ]()

                    if num_splits > 1:
                        var num_pairs = num_rows * num_splits
                        var partial_max = ctx.enqueue_create_buffer[accum_type](
                            num_pairs
                        )
                        var partial_sum = ctx.enqueue_create_buffer[accum_type](
                            num_pairs
                        )

                        comptime partial_kernel = _softmax_split_partial_kernel[
                            BLOCK_SIZE,
                            kernel_simd_width,
                            input_fn_wrapper,
                            dtype,
                            DType.float32,
                            rank,
                        ]
                        ctx.enqueue_function[partial_kernel](
                            shape_il,
                            Float32(1),
                            null_temp_arr,
                            Int32(num_splits),
                            partial_max,
                            partial_sum,
                            grid_dim=num_pairs,
                            block_dim=BLOCK_SIZE,
                        )

                        comptime combine_kernel = _softmax_split_combine_kernel[
                            BLOCK_SIZE,
                            kernel_simd_width,
                            input_fn_wrapper,
                            dtype,
                            DType.float32,
                            rank,
                            output.LayoutType,
                            output.origin,
                            output.Engine,
                        ]
                        ctx.enqueue_function[combine_kernel](
                            shape_il,
                            output,
                            Float32(1),
                            null_temp_arr,
                            Int32(num_splits),
                            partial_max,
                            partial_sum,
                            grid_dim=num_pairs,
                            block_dim=BLOCK_SIZE,
                        )
                        _ = partial_max^
                        _ = partial_sum^
                        return

                    comptime kernel = _softmax_temperature_kernel[
                        BLOCK_SIZE,
                        kernel_simd_width,
                        input_fn_wrapper,
                        dtype,
                        DType.float32,
                        rank,
                        output.LayoutType,
                        output.origin,
                        output.Engine,
                    ]
                    ctx.enqueue_function[kernel](
                        shape_il,
                        output,
                        Float32(1),
                        null_temp_arr,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(PDLLevel.ON),
                    )

                unswitch(
                    simd_width > 1
                    and shape_il[axis] % simd_width == 0
                    and shape_il[axis] >= BLOCK_SIZE * simd_width,
                    dispatch,
                )

        unswitch(shape_il[axis] <= WARP_SIZE, dispatch_warp_or_block)
    else:
        # Fallback: sink-attention or logsoftmax variants stay on the legacy
        # 3-pass kernel until those variants are added to the online path.
        comptime BLOCK_SIZE = 128
        var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)
        comptime kernel = softmax_kernel[
            BLOCK_SIZE,
            input_fn_wrapper,
            dtype,
            sink_type,
            rank,
            output.LayoutType,
            output.origin,
            output.Engine,
            _SinkWeightsTTLayout,
            sink=sink,
            logsoftmax=logsoftmax,
        ]
        ctx.enqueue_function[kernel](
            shape_il,
            output,
            sink_weights.unsafe_value(),
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


def softmax_inline[
    dtype: DType,
    simd_width: Int,
    rank: Int,
    input_fn: def[_simd_width: Int](Coord) capturing[_] -> SIMD[
        dtype, _simd_width
    ],
    target: StaticString = "cpu",
    logsoftmax: Bool = False,
    has_prologue_fusion: Bool = True,
](
    shape: Coord,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    context: Optional[DeviceContext] = None,
) raises:
    """Dispatches softmax (or logsoftmax) to the CPU or GPU target.

    Selects the appropriate CPU or GPU implementation based on `target` and
    traces the operation. Exits early when the tensor is empty.

    Parameters:
        dtype: The dtype of the input and output buffers.
        simd_width: The simd_width to use in vectorization.
        rank: The rank of the input and output tensors.
        input_fn: The elementwise input lambda.
        target: The target device ("cpu" or "gpu").
        logsoftmax: Enable to apply elementwise log() to outputs after softmax.
        has_prologue_fusion: Whether the input lambda supports prologue fusion.

    Args:
        shape: The shape of the output tensor.
        output: The output buffer in which to store the softmax values.
        axis: The axis along which to compute the softmax.
        context: Optional device context for GPU execution.
    """
    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))

    def trace_information() {imm} -> String:
        return trace_arg("input", shape_il, dtype)

    with Trace[TraceLevel.OP, target=target](
        "softmax",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
    ):
        # Exit early if the tensors are empty.
        if shape_il.flattened_length() == 0:
            return
        comptime if is_cpu[target]():
            _softmax_cpu[
                dtype,
                simd_width,
                rank,
                origin_of()._mlir_origin,
                input_fn,
                logsoftmax=logsoftmax,
            ](shape, output, axis, context)
        elif is_gpu[target]():
            _softmax_gpu[
                dtype,
                simd_width,
                rank,
                input_fn,
                logsoftmax=logsoftmax,
                has_prologue_fusion=has_prologue_fusion,
            ](
                shape,
                output,
                axis,
                context.value(),
            )
        else:
            comptime assert False, String("unsupported target ", target)


# ===----------------------------------------------------------------------=== #
# Softmax with temperature scaling (GPU only).
# ===----------------------------------------------------------------------=== #


@__name(t"softmax_temperature_{dtype}_{temp_dtype}_{simd_width}")
def _softmax_temperature_kernel[
    BLOCK_SIZE: Int,
    simd_width: Int,
    input_fn: def[_dtype: DType, _simd_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing[_] -> SIMD[_dtype, _simd_width],
    dtype: DType,
    temp_dtype: DType,
    rank: Int,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    accum_type: DType = get_accum_type[dtype](),
](
    shape: IndexList[rank],
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    temperature: Float32,
    temperature_arr: Optional[
        UnsafePointer[Scalar[temp_dtype], ImmutAnyOrigin]
    ],
):
    """Computes `softmax(logits / T)` over the inner axis. T is resolved per
    row from `temperature_arr` (if non-null) or the scalar `temperature`
    fallback; pass `temperature=1` and `temperature_arr=None` for plain
    softmax.

    Recomputes `exp` during the normalize pass to avoid materializing the
    intermediate `exp(x - max)` tensor in HBM. The caller picks `simd_width`
    based on what the input lambda's loads can handle: `1` is always safe;
    higher values require the input/output layout to support an aligned
    `simd_width`-wide load/store at every row offset.
    """
    var _temperature = Scalar[temp_dtype](temperature)
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime axis = rank - 1
    comptime BLOCK_SPAN = BLOCK_SIZE * simd_width

    var row_size = shape[axis]
    var num_rows = ufloordiv(shape.flattened_length(), row_size)
    var tid = thread_idx.x

    # Masked positions arrive as `-inf` (e.g. attention scores after a
    # causal mask). When the running `row_max` is also `-inf`,
    # `exp(v - row_max)` evaluates to `exp(-inf - (-inf)) = exp(NaN) =
    # NaN`, which then poisons `exp_sum` for the rest of the row.
    # Skip lanes whose value equals the negative-infinity sentinel so
    # the recurrence only advances on finite inputs.
    comptime NEG_INF = Scalar[accum_type].MIN

    with PDL():
        for row_idx in range(block_idx.x, num_rows, grid_dim.x):
            var row_coords = _get_nd_indices_from_flat_index(
                row_idx, shape, axis
            )

            var temp = _temperature.cast[accum_type]()
            if temperature_arr:
                temp = temperature_arr.unsafe_value()[row_idx].cast[
                    accum_type
                ]()
            # Clamp to prevent division by zero on greedy (T=0) rows.
            temp = max(temp, Scalar[accum_type](1e-6))
            var inv_temp = Scalar[accum_type](1) / temp

            # Step 1: online max + exp_sum in a single pass over the input.
            var row_max = Scalar[accum_type].MIN
            var exp_sum = Scalar[accum_type](0)

            for tile_base in range(0, row_size, BLOCK_SPAN):
                var lane_base = tile_base + tid * simd_width
                if lane_base < row_size:
                    var lane_count = min(row_size - lane_base, simd_width)

                    @always_inline
                    def online_max_sum[
                        width: Int
                    ](offset: Int) {row_coords, lane_base, mut}:
                        var coords = row_coords
                        coords[axis] = Int(lane_base) + offset
                        var v = input_fn[dtype, width, rank](coords).cast[
                            accum_type
                        ]()
                        var lane_max = v.reduce_max()
                        if lane_max > NEG_INF:
                            var new_max = max(row_max, lane_max)
                            exp_sum = (
                                exp_sum * exp((row_max - new_max) * inv_temp)
                                + exp(
                                    (v - SIMD[accum_type, width](new_max))
                                    * SIMD[accum_type, width](inv_temp)
                                ).reduce_add()
                            )
                            row_max = new_max

                    vectorize[simd_width](lane_count, online_max_sum)

            # Block-wide reduction of (max, sum) pair. Reduce max first, then
            # correct each thread's partial sum before summing.
            var global_max = block.max[block_size=BLOCK_SIZE](row_max)
            if global_max > NEG_INF:
                exp_sum *= exp((row_max - global_max) * inv_temp)
            var global_sum = block.sum[block_size=BLOCK_SIZE](exp_sum)

            # Step 2: normalize. Recompute exp to avoid round-tripping the
            # intermediate exp(x - max) tensor through HBM.
            var recip = Scalar[accum_type](1) / global_sum

            for tile_base in range(0, row_size, BLOCK_SPAN):
                var lane_base = tile_base + tid * simd_width
                if lane_base < row_size:
                    var lane_count = min(row_size - lane_base, simd_width)

                    @always_inline
                    def normalize[
                        width: Int
                    ](offset: Int) {row_coords, lane_base, output, mut}:
                        var coords = row_coords
                        coords[axis] = Int(lane_base) + offset
                        var logit = input_fn[dtype, width, rank](coords).cast[
                            accum_type
                        ]()
                        var diff = (
                            logit - SIMD[accum_type, width](global_max)
                        ) * SIMD[accum_type, width](inv_temp)
                        var val = exp(diff) * SIMD[accum_type, width](recip)
                        output.store_linear[width=width](
                            coords, val.cast[dtype]()
                        )

                    vectorize[simd_width](lane_count, normalize)


# ===----------------------------------------------------------------------=== #
# Split-K (column/vocab-parallel) softmax (GPU only, any family).
# ===----------------------------------------------------------------------=== #
#
# For a few long rows the single-block-per-row kernel above leaves most SMs
# idle. The split path partitions each row across `num_splits` blocks in two
# passes: stage 1 reduces each column chunk to a partial `(max, sum)` in a
# `[num_rows, num_splits]` scratch; stage 2 combines the partials per row (same
# online rescale used within a block) and normalizes its own chunk. Two passes
# avoid the cross-block atomics a single-kernel combine would need.


@__name(t"softmax_split_partial_{dtype}_{temp_dtype}_{simd_width}")
def _softmax_split_partial_kernel[
    BLOCK_SIZE: Int,
    simd_width: Int,
    input_fn: def[_dtype: DType, _simd_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing[_] -> SIMD[_dtype, _simd_width],
    dtype: DType,
    temp_dtype: DType,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    shape: IndexList[rank],
    temperature: Scalar[temp_dtype],
    temperature_arr: Optional[
        UnsafePointer[Scalar[temp_dtype], ImmutAnyOrigin]
    ],
    num_splits_dev: Int32,
    partial_max: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
    partial_sum: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
):
    """Stage 1 of split-K softmax: reduce one contiguous column chunk of one
    row to a partial `(max, sum)`.

    Grid is `num_rows * num_splits` blocks; block `b` owns `(row, split) =
    (b // num_splits, b % num_splits)`. The chunk math and online recurrence
    mirror step 1 of `_softmax_temperature_kernel`; a fully-masked chunk yields
    `(NEG_INF, 0)`, which stage 2 skips.
    """
    var num_splits = Int(num_splits_dev)
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime axis = rank - 1
    comptime BLOCK_SPAN = BLOCK_SIZE * simd_width
    comptime NEG_INF = Scalar[accum_type].MIN

    var row_size = shape[axis]
    var num_rows = ufloordiv(shape.flattened_length(), row_size)
    var num_pairs = num_rows * num_splits
    var tid = thread_idx.x
    var tiles_per_split = ceildiv(ceildiv(row_size, BLOCK_SPAN), num_splits)

    for pair_idx in range(block_idx.x, num_pairs, grid_dim.x):
        var row_idx = pair_idx // num_splits
        var split = pair_idx % num_splits
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

        # Every block owning this row must read the same temperature.
        var temp = temperature.cast[accum_type]()
        if temperature_arr:
            temp = temperature_arr.unsafe_value()[row_idx].cast[accum_type]()
        temp = max(temp, Scalar[accum_type](1e-6))
        var inv_temp = Scalar[accum_type](1) / temp

        var row_max = Scalar[accum_type].MIN
        var exp_sum = Scalar[accum_type](0)

        # BLOCK_SPAN-aligned chunk so `lane_base` stays simd_width-aligned and
        # the caller's vectorized-load predicate still holds.
        var col_lo = split * tiles_per_split * BLOCK_SPAN
        var col_hi = min(col_lo + tiles_per_split * BLOCK_SPAN, row_size)

        for tile_base in range(col_lo, col_hi, BLOCK_SPAN):
            var lane_base = tile_base + tid * simd_width
            if lane_base < row_size:
                var lane_count = min(row_size - lane_base, simd_width)

                @always_inline
                def online_max_sum[
                    width: Int
                ](offset: Int) {row_coords, lane_base, mut}:
                    var coords = row_coords
                    coords[axis] = Int(lane_base) + offset
                    var v = input_fn[dtype, width, rank](coords).cast[
                        accum_type
                    ]()
                    var lane_max = v.reduce_max()
                    if lane_max > NEG_INF:
                        var new_max = max(row_max, lane_max)
                        exp_sum = (
                            exp_sum * exp((row_max - new_max) * inv_temp)
                            + exp(
                                (v - SIMD[accum_type, width](new_max))
                                * SIMD[accum_type, width](inv_temp)
                            ).reduce_add()
                        )
                        row_max = new_max

                vectorize[simd_width](lane_count, online_max_sum)

        var chunk_max = block.max[block_size=BLOCK_SIZE](row_max)
        if chunk_max > NEG_INF:
            exp_sum *= exp((row_max - chunk_max) * inv_temp)
        var chunk_sum = block.sum[block_size=BLOCK_SIZE](exp_sum)

        if tid == 0:
            partial_max[pair_idx] = chunk_max
            partial_sum[pair_idx] = chunk_sum


@__name(t"softmax_split_combine_{dtype}_{temp_dtype}_{simd_width}")
def _softmax_split_combine_kernel[
    BLOCK_SIZE: Int,
    simd_width: Int,
    input_fn: def[_dtype: DType, _simd_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing[_] -> SIMD[_dtype, _simd_width],
    dtype: DType,
    temp_dtype: DType,
    rank: Int,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    OutputEngine: TensorEngine,
    accum_type: DType = get_accum_type[dtype](),
](
    shape: IndexList[rank],
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    temperature: Scalar[temp_dtype],
    temperature_arr: Optional[
        UnsafePointer[Scalar[temp_dtype], ImmutAnyOrigin]
    ],
    num_splits_dev: Int32,
    partial_max: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
    partial_sum: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
):
    """Stage 2 of split-K softmax: combine the row's partials into
    `(global_max, global_sum)`, then normalize this block's column chunk.

    A fully-masked chunk contributes `0` (not NaN) via the `my_max > NEG_INF`
    guard; a fully-masked row keeps the single-block kernel's `1/0` NaN result.
    """
    var num_splits = Int(num_splits_dev)
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime axis = rank - 1
    comptime BLOCK_SPAN = BLOCK_SIZE * simd_width
    comptime NEG_INF = Scalar[accum_type].MIN

    var row_size = shape[axis]
    var num_rows = ufloordiv(shape.flattened_length(), row_size)
    var num_pairs = num_rows * num_splits
    var tid = thread_idx.x
    var tiles_per_split = ceildiv(ceildiv(row_size, BLOCK_SPAN), num_splits)

    for pair_idx in range(block_idx.x, num_pairs, grid_dim.x):
        var row_idx = pair_idx // num_splits
        var split = pair_idx % num_splits
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

        var temp = temperature.cast[accum_type]()
        if temperature_arr:
            temp = temperature_arr.unsafe_value()[row_idx].cast[accum_type]()
        temp = max(temp, Scalar[accum_type](1e-6))
        var inv_temp = Scalar[accum_type](1) / temp

        # Threads outside `[0, num_splits)` contribute the neutral `(NEG_INF, 0)`.
        var my_max = Scalar[accum_type].MIN
        var my_sum = Scalar[accum_type](0)
        if tid < num_splits:
            my_max = partial_max[row_idx * num_splits + tid]
            my_sum = partial_sum[row_idx * num_splits + tid]

        var global_max = block.max[block_size=BLOCK_SIZE](my_max)
        if my_max > NEG_INF:
            my_sum *= exp((my_max - global_max) * inv_temp)
        var global_sum = block.sum[block_size=BLOCK_SIZE](my_sum)
        var recip = Scalar[accum_type](1) / global_sum

        var col_lo = split * tiles_per_split * BLOCK_SPAN
        var col_hi = min(col_lo + tiles_per_split * BLOCK_SPAN, row_size)

        for tile_base in range(col_lo, col_hi, BLOCK_SPAN):
            var lane_base = tile_base + tid * simd_width
            if lane_base < row_size:
                var lane_count = min(row_size - lane_base, simd_width)

                @always_inline
                def normalize[
                    width: Int
                ](offset: Int) {row_coords, lane_base, output, mut}:
                    var coords = row_coords
                    coords[axis] = Int(lane_base) + offset
                    var logit = input_fn[dtype, width, rank](coords).cast[
                        accum_type
                    ]()
                    var diff = (
                        logit - SIMD[accum_type, width](global_max)
                    ) * SIMD[accum_type, width](inv_temp)
                    var val = exp(diff) * SIMD[accum_type, width](recip)
                    output.store_linear[width=width](coords, val.cast[dtype]())

                vectorize[simd_width](lane_count, normalize)


def softmax_with_temperature[
    dtype: DType,
    temp_dtype: DType = .float32,
    TempLayoutType: TensorLayout = RowMajorLayout[Int64],
    TempEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    temperature: Scalar[temp_dtype] = Float32(1.0),
    temperature_arr: Optional[
        TileTensor[
            temp_dtype, TempLayoutType, ImmutAnyOrigin, Engine=TempEngine
        ]
    ] = None,
) raises:
    """GPU softmax with per-row temperature scaling.

    Computes `softmax(logits / T)` where T can be a scalar or a per-row array.
    When `temperature_arr` is provided, each row uses its own temperature value.
    Falls back to the scalar `temperature` for rows without an array entry.

    Parameters:
        dtype: The data type of the input and output tensors.
        temp_dtype: The data type for temperature values (default float32).
        TempLayoutType: The layout type for the optional temperature array.
        TempEngine: Engine for the optional temperature array.

    Args:
        ctx: Device context for kernel execution.
        input: Input logits tensor [batch_size, vocab_size].
        output: Output probability tensor (same shape as input).
        temperature: Scalar temperature fallback (default 1.0).
        temperature_arr: Optional per-row temperature values [batch_size].
    """
    comptime assert input.rank == 2, "input must be rank 2"
    comptime assert output.rank == 2, "output must be rank 2"

    var shape = coord_to_index_list(input.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    # CUDA rejects grid_dim=0; skip empty launches.
    if batch_size == 0 or d == 0:
        return

    # Extract raw pointer for the kernel (null if not provided).
    var temp_ptr = Optional[UnsafePointer[Scalar[temp_dtype], ImmutAnyOrigin]]()
    if temperature_arr:
        temp_ptr = temperature_arr.value().ptr

    comptime BLOCK_SIZE = 256
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    comptime sm_overprovision_factor = 32
    var num_blocks = min(batch_size, sm_overprovision_factor * sm_count)

    var input_immut = input.as_immut()

    @always_inline
    @__parameter
    @__copy_capture(input_immut)
    def input_load_fn[
        _dtype: DType, width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[_dtype, width]:
        return rebind[SIMD[_dtype, width]](
            input_immut.load_linear[width=width](rebind[IndexList[2]](idx))
        )

    comptime kernel = _softmax_temperature_kernel[
        BLOCK_SIZE,
        simd_width,
        input_load_fn,
        dtype,
        temp_dtype,
        2,
        output.LayoutType,
        output.origin,
        output.Engine,
    ]
    ctx.enqueue_function[kernel](
        IndexList[2](batch_size, d),
        output,
        temperature.cast[.float32](),
        temp_ptr,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ===----------------------------------------------------------------------=== #
# Online softmax in flash attention.
# ===----------------------------------------------------------------------=== #


def _online_softmax_kernel[
    WM: Int,
    WN: Int,
    dtype: DType,
    layout: Layout,
    fragment_transpose: Bool = False,
](
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    """This is only for online softmax validation, NOT a general kernel."""

    comptime assert not fragment_transpose or (
        fragment_transpose and is_amd_gpu()
    ), "fragment_transpose must be False on NVIDIA"

    comptime mma_shape = IndexList[3](
        16, 8, 8
    ) if is_nvidia_gpu() else IndexList[3](16, 16, 16)
    comptime num_seqs = input.shape[0]()
    comptime seqlen = input.shape[1]()

    comptime assert (
        WM == num_seqs
    ), "Only consider WM equal to number of rows in test."

    comptime num_m_mmas = WM // mma_shape[0]
    comptime num_n_mmas = WN // mma_shape[1]

    # TODO: This is a temporary hack, hopefully we can come up with a better way.
    comptime mma_fragment_groups = 2 if is_nvidia_gpu() else 1

    # Each 16x8 mma tile has two 8x8 units and corresponds to 8x4 thread layout
    # in a single warp.
    comptime num_mma_units = num_m_mmas * num_n_mmas * mma_fragment_groups
    comptime score_layout_by_mma_unit = Layout.row_major(
        num_m_mmas * mma_fragment_groups, num_n_mmas
    )
    comptime warp_layout = Layout.row_major(8, 4) if is_nvidia_gpu() else (
        Layout.col_major(16, 4) if fragment_transpose else Layout.row_major(
            4, 16
        )
    )

    # Only consider 2 iterations in this test. The number of warps is based on
    # half sequence length.
    comptime num_rowwise_warps = seqlen // 2 // WN
    comptime block_layout_by_warp = Layout.row_major(1, num_rowwise_warps)

    comptime frag_size = get_fragment_size[mma_shape]()[2]

    var warp_id = warp_id()
    var lane_id = lane_id()

    # If we do more than 2 iterations, the first N - 2 iterations won't be
    # corrected with the right rowmax.
    var input_warp_tile0 = input.tile[WM, WN](0, warp_id)
    var input_warp_tile1 = input.tile[WM, WN](0, warp_id + num_rowwise_warps)

    var output_warp_tile0 = output.tile[WM, WN](0, warp_id)
    var output_warp_tile1 = output.tile[WM, WN](0, warp_id + num_rowwise_warps)

    var p = LayoutTensor[
        dtype,
        Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    comptime fragment_layout = Layout.row_major(1, 2) if is_nvidia_gpu() else (
        Layout.row_major(1, 4) if fragment_transpose else Layout.row_major(4, 1)
    )
    comptime simdwidth_row = fragment_layout.shape[0].value()
    comptime simdwidth_col = fragment_layout.shape[1].value()

    comptime if is_nvidia_gpu():
        p.vectorize[1, 2]().transpose().copy_from(
            input_warp_tile0.vectorize[1, 2]().distribute[warp_layout](lane_id)
        )
    else:
        p.vectorize[1, 4]().copy_from(
            input_warp_tile0.vectorize[
                simdwidth_row, simdwidth_col
            ]().distribute[warp_layout](lane_id)
        )

    var p_vecs = p.reshape[
        Layout.row_major(num_mma_units, frag_size // mma_fragment_groups)
    ]().vectorize[1, frag_size // mma_fragment_groups]()

    var o = (
        LayoutTensor[
            dtype,
            Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )
    var o_vecs = o.reshape[
        Layout.row_major(num_mma_units, frag_size // mma_fragment_groups)
    ]().vectorize[1, frag_size // mma_fragment_groups]()

    comptime frag_num_rows = 2 if is_nvidia_gpu() else (
        1 if fragment_transpose else 4
    )
    comptime row_alignment = align_of[SIMD[dtype, simd_width_of[dtype]()]]()
    var rowmax = unsafe_stack_allocation[
        num_m_mmas * frag_num_rows, dtype, alignment=row_alignment
    ]()
    var rowsum = unsafe_stack_allocation[
        num_m_mmas * frag_num_rows, dtype, alignment=row_alignment
    ]()

    var warp_scratch = LayoutTensor[
        dtype,
        Layout.row_major(2 * num_rowwise_warps, WM),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    comptime for i in range(0, frag_num_rows * num_m_mmas, frag_num_rows):
        rowmax.store(i, SIMD[dtype, frag_num_rows](min_or_neg_inf[dtype]()))
        rowsum.store(i, SIMD[dtype, frag_num_rows](0))

    _online_softmax_iter_for_mma_output[
        dtype,
        score_layout_by_mma_unit,
        block_layout_by_warp,
        warp_layout,
        fragment_layout=fragment_layout,
    ](o_vecs, p_vecs, warp_scratch, rowmax, rowsum)

    # P has the softmax numerator for the first half, save it in q.
    o.copy_from(p)

    comptime if is_nvidia_gpu():
        p.vectorize[1, 2]().transpose().copy_from(
            input_warp_tile1.vectorize[1, 2]().distribute[warp_layout](lane_id)
        )
    else:
        p.vectorize[1, 4]().copy_from(
            input_warp_tile1.vectorize[
                simdwidth_row, simdwidth_col
            ]().distribute[warp_layout](lane_id)
        )

    _online_softmax_iter_for_mma_output[
        dtype,
        score_layout_by_mma_unit,
        block_layout_by_warp,
        warp_layout,
        fragment_layout=fragment_layout,
    ](o_vecs, p_vecs, warp_scratch, rowmax, rowsum)

    # o, p has the correct softmax numerator for the 1st and 2nd half.
    # rowsum has the correct sum. Ready for correction.

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime for i in range(frag_size // mma_fragment_groups):
                comptime if is_nvidia_gpu():
                    p[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                    p[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                        2 * m_mma + 1
                    ]
                    o[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                    o[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                        2 * m_mma + 1
                    ]
                else:
                    var rowsum_tensor = LayoutTensor[
                        dtype, Layout.row_major(num_m_mmas, frag_num_rows)
                    ](rowsum)
                    p[n_mma * num_m_mmas + m_mma, i] /= rowsum_tensor[
                        m_mma, 0 if fragment_transpose else i
                    ]
                    o[n_mma * num_m_mmas + m_mma, i] /= rowsum_tensor[
                        m_mma, 0 if fragment_transpose else i
                    ]

    comptime if is_nvidia_gpu():
        output_warp_tile0.vectorize[1, 2]().distribute[warp_layout](
            lane_id
        ).copy_from(o.vectorize[1, 2]().transpose())
        output_warp_tile1.vectorize[1, 2]().distribute[warp_layout](
            lane_id
        ).copy_from(p.vectorize[1, 2]().transpose())
    else:
        output_warp_tile0.vectorize[simdwidth_row, simdwidth_col]().distribute[
            warp_layout
        ](lane_id).copy_from(o.vectorize[1, 4]())
        output_warp_tile1.vectorize[simdwidth_row, simdwidth_col]().distribute[
            warp_layout
        ](lane_id).copy_from(p.vectorize[1, 4]())


@always_inline
def _online_softmax_iter_for_mma_output[
    dtype: DType,
    score_layout_by_mma_unit: Layout,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    use_exp2: Bool = False,
    warp_split_k: Bool = False,
    fragment_layout: Layout = Layout.row_major(
        1, 2
    ) if is_nvidia_gpu() else Layout.row_major(4, 1),
](
    output_reg_tile: LayoutTensor[mut=True, dtype, ...],
    score_reg_tile: LayoutTensor[mut=True, dtype, ...],
    warp_scratch: LayoutTensor[mut=True, dtype, ...],
    rowmax: UnsafePointer[mut=True, Scalar[dtype], _],
    rowsum: UnsafePointer[mut=True, Scalar[dtype], _],
):
    comptime num_colwise_warps = block_layout_by_warp.shape[0].value()
    comptime num_rowwise_warps = block_layout_by_warp.shape[1].value()

    var lane_id = lane_id()
    var warp_x = umod(warp_id[broadcast=True](), num_rowwise_warps)

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    comptime frag_type = score_reg_tile.element_type
    comptime frag_num_rows = fragment_layout.shape[0].value()
    comptime frag_num_cols = fragment_layout.shape[1].value()

    comptime frag_is_row_vector = frag_num_rows == 1

    # Number of mma unit tiles in the score matrix.
    # 2*num_m_mmas
    comptime num_colwise_tiles = score_layout_by_mma_unit.shape[0].value()
    # num_n_mmas
    comptime num_rowwise_tiles = score_layout_by_mma_unit.shape[1].value()
    # The online softmax attributes for each thread's elements (fragments).
    comptime num_rows_per_thread = num_colwise_tiles * frag_num_rows

    var score_frag_rowmax = LayoutTensor[
        dtype,
        Layout.row_major(num_colwise_tiles, frag_num_rows),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    var score_frag_rowsum = LayoutTensor[
        dtype,
        Layout.row_major(num_colwise_tiles, frag_num_rows),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    var correction = LayoutTensor[
        dtype,
        Layout.row_major(num_colwise_tiles, frag_num_rows),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    var rowmax_tensor = LayoutTensor[
        dtype,
        Layout.row_major(num_colwise_tiles, frag_num_rows),
        address_space=rowmax.address_space,
    ](rowmax)
    var rowsum_tensor = LayoutTensor[
        dtype,
        Layout.row_major(num_colwise_tiles, frag_num_rows),
        address_space=rowsum.address_space,
    ](rowsum)

    # Initialize local max with the running max, and local sum with zero.
    comptime for col_tile in range(num_colwise_tiles):
        comptime for row in range(frag_num_rows):
            score_frag_rowmax[col_tile, row] = rowmax_tensor[col_tile, row]
            score_frag_rowsum[col_tile, row] = 0

    comptime num_shuffles_per_row = log2_floor(warp_layout.shape[1].value())

    comptime num_rowwise_lanes = UInt32(warp_layout.shape[1].value())
    comptime num_colwise_lanes = UInt32(warp_layout.shape[0].value())
    comptime rowwise_lanes_stride = UInt32(warp_layout.stride[1].value())

    comptime exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    # Online softmax
    comptime for col_tile in range(num_colwise_tiles):
        comptime for row_tile in range(num_rowwise_tiles):
            comptime tile_id = col_tile + row_tile * num_colwise_tiles

            # Assume this is a rowwise vector for now see above constraint.
            var frag = score_reg_tile[tile_id, 0]

            comptime for row in range(frag_num_rows):
                comptime for col in range(frag_num_cols):
                    score_frag_rowmax[col_tile, row] = max(
                        score_frag_rowmax[col_tile, row],
                        frag[col if frag_is_row_vector else row],
                    )

        comptime if warp_split_k:
            # HACK: this makes a test failure go away for some reason
            barrier()

        # Every four threads have elements on the same row.
        # Reduce max for T0-T3, T4-T7, etc for nvidia
        #                T0-T15, T16-T31, etc for amd
        comptime for row in range(frag_num_rows):
            score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                Int(num_rowwise_lanes), stride=Int(rowwise_lanes_stride)
            ](score_frag_rowmax[col_tile, row])

    var coords = idx2crd[warp_layout](lane_id)
    var lane_contains_first_column = coords[1] == 0
    var lane_row = coords[0]

    # If a row is split across multiple warps, communicate via shared memory
    # to achieve the rowwise max.
    comptime if num_rowwise_warps > 1 and not warp_split_k:
        # Write per warp rowmax to shared memory.
        if lane_contains_first_column:
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    var score_row_idx = (
                        UInt32(col_tile)
                        * num_colwise_lanes
                        * UInt32(frag_num_rows)
                        + UInt32(lane_row * frag_num_rows)
                        + UInt32(row)
                    )

                    # warp scratch has layout row_major(num_warps, num_rows). The
                    # "score_row_idx" is the idx-th row in the score matrix.
                    warp_scratch[
                        warp_x, Int(score_row_idx)
                    ] = score_frag_rowmax[col_tile, row][0]

        barrier()

        # Reduce the warpwise rowmax.
        if lane_contains_first_column:
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    var score_row_idx = (
                        UInt32(col_tile)
                        * num_colwise_lanes
                        * UInt32(frag_num_rows)
                        + UInt32(lane_row * frag_num_rows)
                        + UInt32(row)
                    )

                    comptime for row_warp in range(num_rowwise_warps):
                        score_frag_rowmax[col_tile, row] = max(
                            rebind[Scalar[dtype]](
                                score_frag_rowmax[col_tile, row]
                            ),
                            rebind[Scalar[dtype]](
                                warp_scratch[row_warp, Int(score_row_idx)]
                            ),
                        )

    # TODO: We can let all threads read shared memory in the above so that
    # we don't need to use warp shuffling.
    comptime for col_tile in range(num_colwise_tiles):
        # Broadcast to 4 threads in the same row.
        comptime if num_rowwise_warps > 1 and not warp_split_k:
            comptime for row in range(frag_num_rows):
                score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                    Int(num_rowwise_lanes), stride=Int(rowwise_lanes_stride)
                ](score_frag_rowmax[col_tile, row])

        # Corrention since previous max may be updated.
        comptime for row in range(frag_num_rows):
            correction[col_tile, row] = exp_function(
                rowmax_tensor[col_tile, row] - score_frag_rowmax[col_tile, row]
            )

        # Softmax numerator based on mma results.
        comptime for row_tile in range(num_rowwise_tiles):
            comptime tile_id = col_tile + num_colwise_tiles * row_tile

            comptime if frag_is_row_vector:
                score_reg_tile[tile_id, 0] = exp_function(
                    score_reg_tile[tile_id, 0]
                    - rebind[frag_type](
                        SIMD[dtype, frag_num_cols](
                            score_frag_rowmax[col_tile, 0][0]
                        )
                    )
                )
            else:
                comptime for row in range(frag_num_rows):
                    score_reg_tile[tile_id, 0][row] = exp_function(
                        score_reg_tile[tile_id, 0][row]
                        - score_frag_rowmax[col_tile, row][0]
                    )

        # Sum softmax numerator from a thread's fragments.
        comptime for row_tile in range(num_rowwise_tiles):
            comptime tile_id = col_tile + num_colwise_tiles * row_tile
            var frag = score_reg_tile[tile_id, 0]

            comptime for row in range(frag_num_rows):
                comptime for col in range(frag_num_cols):
                    score_frag_rowsum[col_tile, row] += frag[
                        col if frag_is_row_vector else row
                    ]

        comptime for row in range(frag_num_rows):
            score_frag_rowsum[col_tile, row] = warp.lane_group_sum[
                Int(num_rowwise_lanes), stride=Int(rowwise_lanes_stride)
            ](score_frag_rowsum[col_tile, row])

    # Reduce rowsum via shared memory.

    comptime if num_rowwise_warps > 1 and not warp_split_k:
        # Write per warp rowmax to shared memory.
        if lane_contains_first_column:
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    # Each thread handle two rows in the mma output.
                    var score_row_idx = (
                        UInt32(col_tile)
                        * num_colwise_lanes
                        * UInt32(frag_num_rows)
                        + UInt32(lane_row * frag_num_rows)
                        + UInt32(row)
                    )

                    warp_scratch[
                        warp_x + num_rowwise_warps, Int(score_row_idx)
                    ] = score_frag_rowsum[col_tile, row][0]

        # Guard writing warp_scratch
        barrier()

        # Reduce the warpwise rowsum.
        if lane_contains_first_column:
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    var score_row_idx = (
                        UInt32(col_tile)
                        * num_colwise_lanes
                        * UInt32(frag_num_rows)
                        + UInt32(lane_row * frag_num_rows)
                        + UInt32(row)
                    )

                    score_frag_rowsum[col_tile, row] = 0

                    # Reduce rowmax. Warps in the same row do the same reduction.
                    comptime for row_warp in range(num_rowwise_warps):
                        score_frag_rowsum[col_tile, row] += rebind[
                            Scalar[dtype]
                        ](
                            warp_scratch[
                                row_warp + num_rowwise_warps, Int(score_row_idx)
                            ]
                        )

            # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.

        comptime for col_tile in range(num_colwise_tiles):
            comptime for row in range(frag_num_rows):
                # Broadcast to 4 threads in the same row.
                score_frag_rowsum[col_tile, row] = warp.lane_group_max[
                    Int(num_rowwise_lanes), stride=Int(rowwise_lanes_stride)
                ](score_frag_rowsum[col_tile, row])

    comptime num_output_replications = output_reg_tile.layout.shape[
        0
    ].value() // (num_colwise_tiles * num_rowwise_tiles)
    # if num_output_replications != 1, then `warp_split_k` and it must equal `num_warps_n`.
    # FIXME: require `warp_split_k` when delaying inter-warp communication.
    comptime assert (
        num_output_replications == 1
        or num_output_replications % num_rowwise_warps == 0
    )

    # if num_output_replications
    comptime for k in range(num_output_replications):
        # Correct previous result
        comptime for col_tile in range(num_colwise_tiles):
            comptime for row_tile in range(num_rowwise_tiles):
                comptime tile_id = col_tile + row_tile * num_colwise_tiles + k * num_colwise_tiles * num_rowwise_tiles

                comptime output_frag_type = type_of(
                    output_reg_tile
                ).element_type

                comptime if frag_is_row_vector:
                    output_reg_tile[tile_id, 0] = output_reg_tile[
                        tile_id, 0
                    ] * output_frag_type(correction[col_tile, 0][0])
                else:
                    comptime for row in range(frag_num_rows):
                        output_reg_tile[tile_id, 0][row] = (
                            output_reg_tile[tile_id, 0][row]
                            * correction[col_tile, row][0]
                        )

    # Save current rowmax and rowsum
    comptime for col_tile in range(num_colwise_tiles):
        comptime for row in range(frag_num_rows):
            rowmax_tensor[col_tile, row] = score_frag_rowmax[col_tile, row]
            rowsum_tensor[col_tile, row] = (
                rowsum_tensor[col_tile, row] * correction[col_tile, row]
                + score_frag_rowsum[col_tile, row]
            )


# This performs a reduction after warp-level split-K for mha
# See `_online_softmax_iter_for_mma_output_split_warp` for
# the implementation of the online component that
# accumulates into separate tiles.
# `output_reg_tile` is `num_warps_n * num_m_mmas * num_n_mmas` rows.
# This performs the reduction, accumulating the `num_warps_n`
# row blocks of size `num_m_mmas * num_n_mmas` into the first row.
#
# This performns:
# m_i_x = -Inf
# for k in range(0, K): # across warps
#   m_i_x = max(m_i_x, m_i_k_{T_c-1})
# O_i_x = 0
# l_i_x_x_x 0
# for k in range(0, K): # across warps
#   c_k_x = exp(m_i_k_{T_c-1} - m_i_x)
#   O_i_x += O_i_k_{T_c-1} * c_k_x
#   l_i_x += l_i_k_{T_c-1} * c_k_x
#
# O_i = diag(l_i_x)^(-1) @ O_i_x
#
# Note that the `for k` loops are across warps (k is the index into
# the `num_warps_n` rowwise warps).
@always_inline
def _online_softmax_iter_for_mma_output_split_warp_reduce[
    output_layout: Layout,
    //,
    dtype: DType,
    score_layout_by_mma_unit: Layout,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    WM: Int,
    WN: Int,
    /,
    use_exp2: Bool = False,
](
    output_reg_tile: LayoutTensor[
        mut=True,
        dtype,
        output_layout,
        address_space=.LOCAL,
        ...,
    ],
    warp_scratch: LayoutTensor[mut=True, dtype, address_space=.SHARED, ...],
    o_smem_ptr_base: UnsafePointer[
        mut=True, Scalar[dtype], address_space=.SHARED, _
    ],
    rowmax: UnsafePointer[mut=True, Scalar[dtype], _],
    rowsum: UnsafePointer[mut=True, Scalar[dtype], _],
):
    # Here, we use naming conventions aligning with MHA's
    comptime num_m_mmas = score_layout_by_mma_unit.shape[0].value()
    comptime num_n_mmas = score_layout_by_mma_unit.shape[1].value()
    comptime num_warps_m = block_layout_by_warp.shape[0].value()
    comptime num_warps_n = block_layout_by_warp.shape[1].value()
    comptime num_lanes_m = UInt32(warp_layout.shape[0].value())
    comptime num_lanes_n = UInt32(warp_layout.shape[1].value())

    comptime if num_warps_n == 1:
        return
    # Note that MHA cut the frag size in half:
    # var output_reg_vecs = output_reg_tile.tile[
    #     num_warps_n * num_m_mmas * num_n_mmas, p_frag_size // 2
    # ](0, 0).vectorize[1, p_frag_size // 2]()
    comptime frag_size = output_reg_tile.element_layout.size()
    comptime assert WM * WN == (
        (2 * frag_size) * WARP_SIZE * num_m_mmas * num_n_mmas
    )
    # alias num_m_mmas = WM // MMA_M
    # alias num_n_mmas = WN // MMA_N
    # alias frag_size = MMA_M * MMA_N // WARP_SIZE
    #

    var tid = thread_idx.x
    var lane = UInt32(lane_id())
    var warp_y, warp_x = udivmod(ufloordiv(tid, WARP_SIZE), num_warps_n)

    comptime fragment_layout = Layout.row_major(
        1, 2
    ) if is_nvidia_gpu() else Layout.row_major(4, 1)
    comptime frag_num_rows = fragment_layout.shape[0].value()

    # Write output reg to smem
    # Each warp has `num_warps_n` output register tiles
    # P(A @ B) @ C
    # `P(A @ B)` is a a `num_warps_m` x `num_warps_n` grid of warp tiles.
    # `C` is partitioned into a `num_warps_n` x `num_warps_n` grid of warp tiles
    #
    # When we don't `split_k_warp`, `P(A @ B)` is copied to smem, so that a warp tile
    # for `D = P(A @ B) @ C` can iterate across all columns of `P(A @ B)`.
    #
    # However, with `split_k_warp`, we skip this copy to smem.
    # Instead, for each `num_warps_n`, they calculate a row of `D`,
    # corresponding to their local columns `P(A @ B)`/rows `C`.
    # We must then perform the reduction afterwards.
    # First, each warp writes the parts other warps need to smem.
    #
    # o_smem is implicitly partitioned into a 5d array:
    # num_warps_m x num_warps_n x (num_warps_n - 1) x
    #    (num_m_mmas * num_n_mmas) x frag_size
    # The axis are:
    # 0. warp_m: No communication across `warps_m` is needed, so we offset the
    #    smem ptr immediately rather than representing this explicitly.
    # 1. warp_n: currently local to a warp, corresponding to axis 0 of
    #    `output_reg_tile`. We iterate across this when writing, and keep it
    #    constant when reducing.
    # 2. warp_n - 1: the other warp_n - 1 column tiles of the answer. We keep it
    #    constant when writing, and iterate across it when reducing.
    # 3-4. ((WM*WN)//frag_size) x frag_size: the two trailing dimensions of
    #    output_reg_tile
    comptime warp_tile_size = WM * WN  # ((WM*WN)//frag_size) x frag_size
    comptime row_warp_tile_size = (num_warps_n - 1) * warp_tile_size
    # Makes sure arithmetic is optimized away when `num_warps_m == 1`.
    var o_smem_ptr = (
        o_smem_ptr_base
        + warp_y * (num_warps_n - 1) * row_warp_tile_size if num_warps_m
        > 1 else o_smem_ptr_base
    )

    # NOTE: we must ensure that `output_reg_tile` is only ever indexed by constants.
    var out_reg_tile = output_reg_tile.tile[num_m_mmas * num_n_mmas, 1](0, 0)

    comptime o_smem_layout = Layout.row_major(
        WM * WN // (2 * frag_size), frag_size
    )

    comptime exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    comptime layout = Layout.row_major(num_m_mmas, frag_num_rows)
    comptime TensorType = LayoutTensor[
        dtype, layout, MutAnyOrigin, address_space=.LOCAL
    ]
    var interwarp_frag_rowmax = TensorType.stack_allocation()
    var interwarp_frag_rowsum = TensorType.stack_allocation()
    var correction = TensorType.stack_allocation()
    var rowmax_tensor = TensorType.stack_allocation()
    var rowsum_tensor = TensorType.stack_allocation()
    # corrections across warps
    # Write per warp rowmax to shared memory.
    if lane % num_lanes_n == 0:
        comptime for col_tile in range(num_m_mmas):
            comptime for row in range(frag_num_rows):
                var score_row_idx = (
                    UInt32(col_tile) * num_lanes_m
                    + (lane // num_lanes_n) * UInt32(frag_num_rows)
                    + UInt32(row)
                )
                # warp scratch has layout row_major(num_warps, num_rows). The
                # "score_row_idx" is the idx-th row in the score matrix.
                warp_scratch[
                    warp_x + num_warps_n, Int(score_row_idx)
                ] = rowmax_tensor[col_tile, row][0]

    barrier()

    # Reduce the warpwise rowmax.
    if lane % num_lanes_n == 0:
        comptime for col_tile in range(num_m_mmas):
            comptime for row in range(frag_num_rows):
                var score_row_idx = (
                    UInt32(col_tile) * num_lanes_m
                    + (lane // num_lanes_n) * UInt32(frag_num_rows)
                    + UInt32(row)
                )

                interwarp_frag_rowmax[col_tile, row] = rebind[Scalar[dtype]](
                    warp_scratch[num_warps_n, Int(score_row_idx)]
                )

                comptime for row_warp in range(1, num_warps_n):
                    interwarp_frag_rowmax[col_tile, row] = max(
                        rebind[Scalar[dtype]](
                            interwarp_frag_rowmax[col_tile, row]
                        ),
                        rebind[Scalar[dtype]](
                            warp_scratch[
                                row_warp + num_warps_n, Int(score_row_idx)
                            ]
                        ),
                    )

    comptime for col_tile in range(num_m_mmas):
        # Broadcast to 4 threads in the same row.
        comptime if num_warps_n > 1:
            comptime for row in range(frag_num_rows):
                interwarp_frag_rowmax[col_tile, row] = warp.lane_group_max[
                    Int(num_lanes_n)
                ](interwarp_frag_rowmax[col_tile, row])

        # Corrention since previous max may be updated.
        comptime for row in range(frag_num_rows):
            correction[col_tile, row] = exp_function(
                rowmax_tensor[col_tile, row]
                - interwarp_frag_rowmax[col_tile, row]
            )

    if lane % num_lanes_n == 0:
        comptime for col_tile in range(num_m_mmas):
            comptime for row in range(frag_num_rows):
                var score_row_idx = (
                    UInt32(col_tile) * num_lanes_m
                    + (lane // num_lanes_n) * UInt32(frag_num_rows)
                    + UInt32(row)
                )
                var c = rebind[Scalar[dtype]](correction[col_tile, row])
                warp_scratch[warp_x, Int(score_row_idx)] = (
                    0.0 if c == 0.0 else rowsum_tensor[col_tile, row][0] * c
                )

    barrier()

    # Reduce the warpwise rowsum.
    if lane % num_lanes_n == 0:
        comptime for col_tile in range(num_m_mmas):
            comptime for row in range(frag_num_rows):
                var score_row_idx = (
                    UInt32(col_tile) * num_lanes_m
                    + (lane // num_lanes_n) * UInt32(frag_num_rows)
                    + UInt32(row)
                )
                interwarp_frag_rowsum[col_tile, row] = rebind[Scalar[dtype]](
                    warp_scratch[0, Int(score_row_idx)]
                )

                # Reduce rowmax. Warps in the same row do the same reduction.
                comptime for row_warp in range(1, num_warps_n):
                    interwarp_frag_rowsum[col_tile, row] += rebind[
                        Scalar[dtype]
                    ](warp_scratch[row_warp, Int(score_row_idx)])

        # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.

    comptime for col_tile in range(num_m_mmas):
        comptime for row in range(frag_num_rows):
            # Broadcast to 4 threads in the same row.
            interwarp_frag_rowsum[col_tile, row] = warp.lane_group_max[
                # interwarp_frag_rowsum[col_tile, row] = lane_group_sum[
                Int(num_lanes_n)
            ](interwarp_frag_rowsum[col_tile, row])

    var output = output_reg_tile.split[num_warps_n, axis=0]()

    comptime for col_tile in range(num_m_mmas):
        comptime for row in range(frag_num_rows):
            # correction[col_tile, row] /= interwarp_frag_rowsum[col_tile, row]
            rowsum_tensor[col_tile, row] = interwarp_frag_rowsum[col_tile, row]

    # var ort00 = output_reg_tile[0,0]
    # scale output reg
    comptime for col_tile in range(num_m_mmas):
        comptime for row_tile in range(num_n_mmas):
            comptime tile_id = col_tile + row_tile * num_m_mmas
            comptime output_frag_type = type_of(output_reg_tile).element_type

            comptime for row in range(frag_num_rows):
                var c = correction[col_tile, row][0]

                comptime for warp_tile in range(num_warps_n):
                    output[warp_tile][tile_id, 0] = (
                        0.0 if c == 0.0 else output[warp_tile][tile_id, 0] * c
                    )

    # reduce
    comptime for warp_n in range(num_warps_n):
        var reg_tile = output_reg_tile.tile[num_m_mmas * num_n_mmas, 1](
            warp_n, 0
        )
        if warp_n == warp_x:
            comptime if warp_n > 0:
                # we want `output_reg_tile[0,:,:]` to be the real output reg tile.
                out_reg_tile.copy_from(
                    reg_tile.as_unsafe_any_origin()
                )  # hack aliasing.
        else:
            # copy output reg tile to smem
            # Example smem row, col when `num_warps_n = 4`:
            # -----------------------------------
            # | N\X |   0  |   1  |   2  |   3  |
            # |  0  |      | 0, 0 | 0, 1 | 0, 2 |
            # |  1  | 1, 0 |      | 1, 1 | 1, 2 |
            # |  2  | 2, 0 | 2, 1 |      | 2, 2 |
            # |  3  | 3, 0 | 3, 1 | 3, 2 |      |
            # -----------------------------------
            # `N\X` refer to `warp_n`, `warp_x`
            comptime row = warp_n
            var col = warp_x - (1 if warp_x > warp_n else 0)
            var o_smem_ptr_write = (
                o_smem_ptr + (row * (num_warps_n - 1) + col) * warp_tile_size
            )
            var o_smem_write = (
                LayoutTensor[
                    dtype,
                    o_smem_layout,
                    address_space=.SHARED,
                ](o_smem_ptr_write)
                .vectorize[1, frag_size]()
                .distribute[Layout.row_major(WARP_SIZE, 1)](Int(lane))
            )
            # after distribute and vectorize, the shape should be
            # WM * WN // (2*frag_size * WARP_SIZE), 1
            # Note that we have
            # frag_size = MMA_M * MMA_N // (2*WARP_SIZE)
            # num_m_mmas = WM // MMA_M
            # num_n_mmas = WN // MMA_N
            # so (because 2*WARP_SIZE*frag_size == MMA_M * MMA_N):
            # WM * WN // (2*frag_size * WARP_SIZE) = WM * WN // (MMA_M * MMA_N)
            #   = num_m_mmas * num_n_mmas
            # thus the shape of `o_smem_write` matches that of `reg_tile`.
            o_smem_write.copy_from(reg_tile)

    barrier()

    # Perform the reduction.
    comptime for warp_n in range(num_warps_n - 1):
        var row = warp_x
        comptime col = warp_n
        var o_smem_ptr_reduce = (
            o_smem_ptr + (row * (num_warps_n - 1) + col) * warp_tile_size
        )
        var o_smem_reduce = (
            LayoutTensor[
                dtype,
                o_smem_layout,
                address_space=.SHARED,
            ](o_smem_ptr_reduce)
            .vectorize[1, frag_size]()
            .distribute[Layout.row_major(WARP_SIZE, 1)](Int(lane))
        )

        comptime for i in range(o_smem_reduce.layout.size()):
            out_reg_tile[i] += rebind[SIMD[dtype, frag_size]](o_smem_reduce[i])


@always_inline
def _rowmax_online_softmax[
    dtype: DType,
    reg_tile_layout: Layout,
    row_accum_layout: Layout,
    fragment_layout: Layout,
    accum_frag_layout: Layout,
    //,
    num_rowwise_warps: Int,
    warp_layout: Layout,
    use_exp2: Bool,
    fold_scale_fma: Bool = False,
](
    out score_frag_rowmax: LayoutTensor[
        dtype,
        row_accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=accum_frag_layout,
    ],
    score_reg_tile: LayoutTensor[
        dtype,
        reg_tile_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=fragment_layout,
    ],
    rowmax_tensor: LayoutTensor[
        dtype,
        row_accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=accum_frag_layout,
    ],
    init_rowmax: Bool = False,
    scale_log2e: Scalar[dtype] = 1.0,
):
    comptime assert (
        num_rowwise_warps == 1
    ), "FIXME: add support for num_rowwise_warps>1, required by deepseek"

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    comptime frag_size = fragment_layout.size()
    # alias frag_num_rows = fragment_layout.shape[0].value() # sm90 1
    comptime frag_num_cols = fragment_layout.shape[1].value()  # sm90 2
    comptime frag_num_rows = accum_frag_layout.size()
    comptime assert frag_num_rows == fragment_layout.shape[0].value()

    comptime num_colwise_tiles = reg_tile_layout[0].size()
    comptime num_rowwise_tiles = reg_tile_layout[1].size()
    # The online softmax attributes for each thread's elements (fragments).
    score_frag_rowmax = type_of(rowmax_tensor).stack_allocation()

    comptime num_rowwise_lanes = UInt32(warp_layout.shape[1].value())

    comptime exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    # Online softmax
    comptime for col_tile in range(num_colwise_tiles):
        # Initialize local max with the running max.
        score_frag_rowmax[col_tile] = score_reg_tile[col_tile, 0].reduce_max[
            frag_num_rows
        ]()

        comptime for row_tile in range(1, num_rowwise_tiles):
            score_frag_rowmax[col_tile] = max(
                score_frag_rowmax[col_tile],
                score_reg_tile[col_tile, row_tile].reduce_max[frag_num_rows](),
            )
    if not init_rowmax:
        comptime for col_tile in range(num_colwise_tiles):
            score_frag_rowmax[col_tile] = max(
                score_frag_rowmax[col_tile],
                rowmax_tensor[col_tile],
            )

    comptime for col_tile in range(num_colwise_tiles):
        # Every four threads have elements on the same row.
        # Reduce max for  T0-T3,  T4-T7, etc for nvidia
        #                T0-T15, T16-T31, etc for amd
        score_frag_rowmax[col_tile] = warp.lane_group_max[
            Int(num_rowwise_lanes)
        ](score_frag_rowmax[col_tile])

        # Softmax numerator based on mma results. fold_scale_fma folds the
        # `*scale_log2e` (dropped from the mask) and the `-m*scale_log2e`
        # subtract into one FFMA2 per pair, matching the MM-Sparse ref
        # scale_subtract_rowmax; the score is RAW S, the rowmax is over raw S.
        comptime if fold_scale_fma:
            comptime assert (
                dtype == .float32 and frag_size == 2
            ), "fold_scale_fma needs the f32x2 score pair"
            var vscale = SIMD[.float32, 2](rebind[Float32](scale_log2e))
            var neg_m_scaled = SIMD[.float32, 2](
                rebind[Float32](score_frag_rowmax[col_tile])
                * rebind[Float32](scale_log2e)
                * -1.0
            )
            comptime for row_tile in range(num_rowwise_tiles):
                var s = rebind[SIMD[.float32, 2]](
                    score_reg_tile[col_tile, row_tile]
                )
                score_reg_tile[col_tile, row_tile] = rebind[
                    SIMD[dtype, frag_size]
                ](exp2(_fma_f32x2(s, vscale, neg_m_scaled)))
        else:
            comptime for row_tile in range(num_rowwise_tiles):
                var sfm: SIMD[dtype, frag_size]

                comptime if accum_frag_layout.size() == 1:
                    sfm = {rebind[Scalar[dtype]](score_frag_rowmax[col_tile])}
                else:
                    sfm = rebind[SIMD[dtype, frag_size]](
                        score_frag_rowmax[col_tile]
                    )
                score_reg_tile[col_tile, row_tile] = exp_function(
                    score_reg_tile[col_tile, row_tile] - sfm
                )


@always_inline
def _rowsum[
    dtype: DType,
    reg_tile_layout: Layout,
    fragment_layout: Layout,
    //,
    warp_layout: Layout,
    packed_reduce: Bool = False,
](
    score_reg_tile: LayoutTensor[
        dtype,
        reg_tile_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=fragment_layout,
    ],
    out score_frag_rowsum: LayoutTensor[
        dtype,
        Layout.row_major(reg_tile_layout[0].size()),
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=Layout.row_major(fragment_layout.shape[0].value()),
    ],
):
    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    comptime frag_num_rows = score_frag_rowsum.element_layout.size()
    comptime frag_size = fragment_layout.size()

    comptime num_colwise_tiles = reg_tile_layout[0].size()
    comptime num_rowwise_tiles = reg_tile_layout[1].size()
    # The online softmax attributes for each thread's elements (fragments).
    comptime num_rows_per_thread = num_colwise_tiles * frag_num_rows

    score_frag_rowsum = type_of(score_frag_rowsum).stack_allocation()

    comptime num_rowwise_lanes = UInt32(warp_layout.shape[1].value())

    # packed_reduce keeps one f32x2 partial per row, chaining the row_tiles via
    # `add.ftz.f32x2` (one FADD2 each), then folds the pair once -- matching the
    # MM-Sparse ref fadd_reduce. Halves the in-fragment FADD count.
    comptime if packed_reduce:
        comptime assert (
            dtype == .float32 and frag_size == 2 and frag_num_rows == 1
        ), "packed_reduce needs the f32x2 score pair (one row per fragment)"
        comptime for col_tile in range(num_colwise_tiles):
            var acc = rebind[SIMD[.float32, 2]](score_reg_tile[col_tile, 0])
            comptime for row_tile in range(1, num_rowwise_tiles):
                acc = _add_f32x2(
                    acc,
                    rebind[SIMD[.float32, 2]](
                        score_reg_tile[col_tile, row_tile]
                    ),
                )
            score_frag_rowsum[col_tile] = rebind[Scalar[dtype]](acc[0] + acc[1])
    else:
        # Initialize sum with first column
        comptime for col_tile in range(num_colwise_tiles):
            score_frag_rowsum[col_tile] = score_reg_tile[
                col_tile, 0
            ].reduce_add[frag_num_rows]()

        comptime for row_tile in range(1, num_rowwise_tiles):
            comptime for col_tile in range(num_colwise_tiles):
                score_frag_rowsum[col_tile] = (
                    score_frag_rowsum[col_tile]
                    + score_reg_tile[col_tile, row_tile].reduce_add[
                        frag_num_rows
                    ]()
                )

    comptime for col_tile in range(num_colwise_tiles):
        score_frag_rowsum[col_tile] = warp.lane_group_sum[
            Int(num_rowwise_lanes)
        ](score_frag_rowsum[col_tile])


@always_inline
def _online_softmax_correction[
    dtype: DType,
    row_accum_layout: Layout,
    accum_frag_layout: Layout,
    //,
    use_exp2: Bool,
](
    rowmax_tensor: LayoutTensor[
        dtype,
        row_accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=accum_frag_layout,
    ],
    score_frag_rowmax: LayoutTensor[
        dtype,
        row_accum_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=accum_frag_layout,
    ],
):
    comptime num_colwise_tiles = row_accum_layout.size()
    comptime exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    comptime for col_tile in range(num_colwise_tiles):
        # Corrention since previous max may be updated.
        var sfr = score_frag_rowmax[col_tile]
        score_frag_rowmax[col_tile] = exp_function(
            rowmax_tensor[col_tile] - sfr
        )
        rowmax_tensor[col_tile] = sfr


# ===----------------------------------------------------------------------=== #
# Row-based softmax (free-form body layer).
#
# Three phases over a Row: ReduceMax, ReduceSum-of-exp (reads the max via a
# captured scalar), then a per-element `map`. On the cache-eligible tier the row
# is materialized once and replayed from registers across both reductions + the
# map (one load per element); otherwise it streams. A sink term — which the
# declarative pipeline could not carry — would fold into `l` as one extra line.
# ===----------------------------------------------------------------------=== #


def softmax[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    logsoftmax: Bool = False,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    context: Optional[DeviceContext] = None,
) raises:
    # softmax's two dependent reduces (row max, then sum of `exp(x - max)`)
    # plus its final normalizing write are passed to the scaffolder as
    # `num_phases=3`, which lets it split each row across many blocks on the
    # GPU when a few rows leave the device under-occupied (e.g. Kimi
    # 8x163840); CPU ignores it. No real caller has ever needed a different
    # value, so it's hardcoded at the `launch` call below rather than
    # exposed here.
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert 0 <= reduce_dim < rank, "reduce_dim must be in [0, rank)"
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "softmax requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, dtype, accum
    ]()

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size, var output, var input_fn
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce (phase 1): row max.
        @always_inline
        def vmax[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            return tile.cast[accum]()

        # Row stats held W-wide (one value broadcast to every lane, or one
        # per output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        var row_max = row.reduce[ReduceMax[accum, params.simd_width]](
            vmax, load
        ).acc

        # Reduce (phase 2): sum of exp(tile - row_max).
        @always_inline
        def vexp[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var row_max
        } -> SIMD[accum, width]:
            return exp(tile.cast[accum]() - row_max.slice[width]())

        var denom = row.reduce[ReduceSum[accum, params.simd_width]](
            vexp, load
        ).acc
        var recip = SIMD[accum, params.simd_width](1) / denom
        var log_denom = log(denom)

        # Emit: per-element normalize + store.
        @always_inline
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var row_max,
            var recip,
            var log_denom,
            var output,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var tile_accum = tile.cast[accum]()

            comptime if logsoftmax:
                var shifted = (
                    tile_accum
                    - row_max.slice[width]()
                    - log_denom.slice[width]()
                )
                var result = shifted.cast[dtype]()
                output.store[width=width, alignment=alignment](
                    idx.coord, result
                )
            else:
                var numerator = exp(tile_accum - row_max.slice[width]())
                var result = (numerator * recip.slice[width]()).cast[dtype]()
                output.store[width=width, alignment=alignment](
                    idx.coord, result
                )

        # `row_max`/`recip`/`log_denom`/`output` ride `write`'s capture list
        # into `elementwise`.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        computationally_expensive=True,
        num_phases=3,
        dtype_size=size_of[dtype](),
    ](body, shape, context)
