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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers reduction graph ops (sum, mean, argmax, and related) over tensor axes."""

import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from algorithm.reductions import (
    reduce_argmax,
    reduce_argmin,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_min_and_max,
    reduce_product,
    reduce_sum,
)
from algorithm.rowwise_types import RowCoord

from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_gpu
from nn import arg_nonzero
from nn.argsort import argsort
from nn.cumsum import cumsum
from nn.gather_scatter import _unsafe_normalize_neg_index
from nn.normalization import (
    apply_qk_rms_norm,
    group_norm,
    layer_norm,
    layer_norm_rope_ragged,
    rms_norm_fused_residual_add,
    rms_norm_rope,
    rms_norm,
    row_mean_of_squares_qk,
    row_mean_of_squares,
)
from nn.softmax import softmax
from nn.topk import top_k, top_k_shape_impl
from state_space.rms_norm_fused_residual import (
    _rms_norm_fused_residual_cpu_entry,
    rms_norm_fused_residual,
)
from extensibility import (
    InputTensor,
    OutputTensor,
    Tensor,
    TileTensorable,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from std.logger import Logger

comptime logger = Logger()
"""Logger for the reductions module."""

from std.utils import IndexList
from std.utils.coord import ComptimeInt, Coord, coord_to_index_list
from layout import UNKNOWN_VALUE
from std.utils.index import Index

# ===-----------------------------------------------------------------------===#
from .kernels import *


@extensibility.register("mo.reduce.arg_max")
struct ArgMax:
    """Registers the `mo.reduce.arg_max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[rank=rank, ...],
        input: FusedInputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.arg_max` graph op.

        Parameters:
            target: Compilation target string.
            rank: Tensor rank of the input and output tensors.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        # `axis` is normalized at comptime (Row `reduce_dim` is comptime);
        # the new impl handles CPU + GPU and any axis.
        comptime reduce_dim = axis if axis >= 0 else axis + rank

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[.int64, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_argmax[
            input.dtype,
            target=target,
            reduce_dim=reduce_dim,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register("mo.reduce.arg_min")
struct ArgMin:
    """Registers the `mo.reduce.arg_min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[rank=rank, ...],
        input: FusedInputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.arg_min` graph op.

        Parameters:
            target: Compilation target string.
            rank: Tensor rank of the input and output tensors.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        # `axis` is normalized at comptime (Row `reduce_dim` is comptime);
        # the new impl handles CPU + GPU and any axis.
        comptime reduce_dim = axis if axis >= 0 else axis + rank

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[.int64, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_argmin[
            input.dtype,
            target=target,
            reduce_dim=reduce_dim,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register("mo.arg_nonzero")
struct ArgNonZero:
    """Registers the `mo.arg_nonzero` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output_buffer: OutputTensor[rank=2, ...],
        input_buffer: InputTensor,
    ) raises:
        """Executes the `mo.arg_nonzero` graph op.

        Args:
            output_buffer: Output tensor receiving the nonzero indices.
            input_buffer: Input tensor whose nonzero indices are found.
        """
        arg_nonzero.arg_nonzero(
            input_buffer.to_tile_tensor[.int64](),
            output_buffer.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("mo.arg_nonzero")
def arg_nonzero_shape(input_buffer: Some[TileTensorable]) -> IndexList[2]:
    """Computes the output shape for the `mo.arg_nonzero` graph op.

    Args:
        input_buffer: Input tensor whose nonzero element indices are returned.

    Returns:
        The output shape as a rank-2 `IndexList` of nonzero element indices.
    """
    return arg_nonzero.arg_nonzero_shape(input_buffer.to_tile_tensor())


@extensibility.register("mo.reduce.mean")
struct Mean:
    """Registers the `mo.reduce.mean` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.mean` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_mean[
            output.dtype,
            target=target,
            reduce_dim=axis,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.mean")
def reduce_mean_shape[
    axis: Int,
](input: Some[Tensor]) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.mean` graph op.

    Parameters:
        axis: Dimension along which to average the elements of `input`.

    Args:
        input: Input tensor reduced along `axis` by averaging.

    Returns:
        The output shape after reducing `input` along `axis` by averaging.
    """
    return reduce_shape(input, axis)


@extensibility.register("mo.reduce.row_mean_of_squares")
struct RowMeanOfSquares:
    """Per-row mean of squares over the last axis, accumulated in float32.

    For input `x` of shape `[M, N]` computes `out[m, 0] = sum_n(x[m,n]^2) / N`
    and writes a `[M, 1]` `output.dtype` result (typically float32). The square
    and accumulation always run in the input's accumulation type (float32 for
    bfloat16/float16/float32 inputs), independent of the output dtype.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        input: FusedInputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.row_mean_of_squares` graph op.

        Parameters:
            target: Compilation target string.

        Args:
            output: Output tensor of shape `[M, 1]` receiving the per-row
                mean of squares.
            input: Input tensor of shape `[M, N]` to compute over.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If `output` does not have shape `[input_rows, 1]`.
        """
        if output.shape()[0] != input.shape()[0] or output.shape()[1] != 1:
            raise Error("output must have shape [input_rows, 1]")

        @always_inline
        def input_fn[
            width: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](coords)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            # `output` is `[M, 1]`, so `width` adjacent rows (tiled tier) at
            # column 0 sit contiguously -- one vector store, no lane splitting.
            output.store[width=width](Index(Int(coords[0].value()), 0), val)

        row_mean_of_squares[input.dtype, output.dtype, 2, target=target](
            input_fn, output_fn, input.shape_coord(), ctx
        )


@extensibility.register_shape_function("mo.reduce.row_mean_of_squares")
def reduce_row_mean_of_squares_shape(input: Some[Tensor]) -> IndexList[2]:
    """Computes the output shape for the `mo.reduce.row_mean_of_squares` graph op.

    Args:
        input: Two-dimensional input tensor of shape `[M, N]` whose
            per-row mean of squares produces a `[M, 1]` output.

    Returns:
        The output shape `[M, 1]` after computing the per-row
        mean of squares.
    """
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    var input_shape = coord_to_index_list(input.shape().tuple())
    return Index(input_shape[0], 1)


@extensibility.register("mo.reduce.row_mean_of_squares_qk")
struct RowMeanOfSquaresQK:
    """Fused per-row mean of squares for two operands Q and K.

    For `q` of shape `[M, Nq]` and `k` of shape `[M, Nk]` (sharing rows but with
    possibly different column counts), computes `out[m, 0] = mean_n(q[m,n]^2)`
    and `out[m, 1] = mean_n(k[m,n]^2)` into a `[M, 2]` output. The square and
    accumulation always run in float32. This is a single-launch fusion of two
    `mo.reduce.row_mean_of_squares` ops plus a concat, used for cross-head
    QK-RMSNorm statistics under tensor parallelism.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        q: InputTensor[rank=2, ...],
        k: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.row_mean_of_squares_qk` graph op.

        Parameters:
            target: Compilation target string.

        Args:
            output: Output tensor of shape `[M, 2]` receiving the per-row
                mean of squares for `q` and `k`.
            q: Query tensor of shape `[M, Nq]`.
            k: Key tensor of shape `[M, Nk]`.
            ctx: Device context used to enqueue the kernel.
        """
        comptime assert q.dtype == k.dtype, "q and k must share a dtype"
        if (
            output.shape()[0] != q.shape()[0]
            or output.shape()[0] != k.shape()[0]
            or output.shape()[1] != 2
        ):
            raise Error("output must have shape [rows, 2] matching q/k rows")

        # `k` is bitcast to `q.dtype` to unify the single `in_dtype` kernel
        # parameter (q and k share a dtype, asserted above).
        row_mean_of_squares_qk[target=target](
            output.to_tile_tensor[.int64](),
            q.to_tile_tensor[.int64](),
            k.to_tile_tensor[.int64]().bitcast[q.dtype](),
            q.shape()[0],
            q.shape()[1],
            k.shape()[1],
            ctx,
        )


@extensibility.register_shape_function("mo.reduce.row_mean_of_squares_qk")
def reduce_row_mean_of_squares_qk_shape(
    q: Some[Tensor], k: Some[Tensor]
) -> IndexList[2]:
    """Computes the output shape for the `mo.reduce.row_mean_of_squares_qk` graph op.

    Args:
        q: Query tensor of shape `[M, Nq]` whose per-row mean of squares
            forms output column 0.
        k: Key tensor of shape `[M, Nk]` whose per-row mean of squares
            forms output column 1; must share row count with `q`.

    Returns:
        The output shape `[M, 2]` with per-row mean of squares
        for `q` and `k`.
    """
    comptime assert (
        type_of(q).rank == 2 and type_of(k).rank == 2
    ), "q and k must be rank 2"
    var q_shape = coord_to_index_list(q.shape().tuple())
    return Index(q_shape[0], 2)


@extensibility.register("mo.norm.apply_qk_rms_norm")
struct ApplyQKRMSNorm:
    """Fused per-element QK-RMSNorm apply for two operands Q and K.

    Given the already cross-rank-reduced per-row statistics `qk_var [M, 2]`
    (col 0 = mean(q^2), col 1 = mean(k^2), float32) and per-column float32
    scales `gamma_q [Nq]` / `gamma_k [Nk]`, applies in a single launch:

    `q_out[m,c] = cast((cast(q[m,c], f32) * rsqrt(qk_var[m,0] + eps)) * gamma_q[c], q.dtype)`
    and likewise for K with column 1. The grouping `((x * rs) * gamma)` then
    cast matches the unfused graph this replaces for bit-accuracy. This fuses
    the QK-RMSNorm apply chain (~7 tiny elementwise/View kernels) into one
    launch, used for cross-head QK-RMSNorm under tensor parallelism.

    Outputs (in order): `q_out [M, Nq]`, `k_out [M, Nk]` (both q/k dtype).
    Inputs (in order): `q [M, Nq]`, `k [M, Nk]` (activation dtype),
    `qk_var [M, 2]` (float32), `gamma_q [Nq]` (float32),
    `gamma_k [Nk]` (float32). Attribute: `epsilon` (float32 host scalar).
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        q_out: OutputTensor[rank=2, ...],
        k_out: OutputTensor[rank=2, ...],
        q: InputTensor[rank=2, ...],
        k: InputTensor[rank=2, ...],
        qk_var: InputTensor[dtype=.float32, rank=2, ...],
        gamma_q: InputTensor[dtype=.float32, rank=1, ...],
        gamma_k: InputTensor[dtype=.float32, rank=1, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.norm.apply_qk_rms_norm` graph op.

        Parameters:
            target: Compilation target string.

        Args:
            q_out: Output tensor receiving the normalized `q`.
            k_out: Output tensor receiving the normalized `k`.
            q: Query input tensor.
            k: Key input tensor.
            qk_var: Per-row variance tensor of shape `[M, 2]`.
            gamma_q: Per-column scale weights for `q`.
            gamma_k: Per-column scale weights for `k`.
            epsilon: Small constant for numerical stability.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If output shapes do not match the corresponding inputs.
        """
        comptime assert q.dtype == k.dtype, "q and k must share a dtype"
        comptime assert (
            q_out.dtype == q.dtype and k_out.dtype == q.dtype
        ), "outputs must match the q/k dtype"
        if q_out.shape()[0] != q.shape()[0] or q_out.shape()[1] != q.shape()[1]:
            raise Error("q_out must have shape [rows, Nq] matching q")
        if k_out.shape()[0] != k.shape()[0] or k_out.shape()[1] != k.shape()[1]:
            raise Error("k_out must have shape [rows, Nk] matching k")
        if (
            qk_var.shape()[0] != q.shape()[0]
            or qk_var.shape()[0] != k.shape()[0]
            or qk_var.shape()[1] != 2
        ):
            raise Error("qk_var must have shape [rows, 2] matching q/k rows")
        if gamma_q.shape()[0] != q.shape()[1]:
            raise Error("gamma_q must have shape [Nq] matching q cols")
        if gamma_k.shape()[0] != k.shape()[1]:
            raise Error("gamma_k must have shape [Nk] matching k cols")

        # `out_dtype` is inferred from `q_out`; `k_out` shares the same dtype
        # (asserted above), so bitcast its tile tensor to `q_out.dtype` to
        # unify the single `out_dtype` parameter. Likewise `in_dtype` is
        # inferred from `q`, so bitcast `k` to `q.dtype`.
        apply_qk_rms_norm[target=target,](
            q_out.to_tile_tensor[.int64](),
            k_out.to_tile_tensor[.int64]().bitcast[q_out.dtype](),
            gamma_q.to_tile_tensor[.int64](),
            gamma_k.to_tile_tensor[.int64](),
            qk_var.to_tile_tensor[.int64](),
            q.to_tile_tensor[.int64](),
            k.to_tile_tensor[.int64]().bitcast[q.dtype](),
            epsilon,
            q.shape()[0],
            q.shape()[1],
            k.shape()[1],
            ctx,
        )


@extensibility.register("mo.reduce.add")
struct ReduceAdd:
    """Registers the `mo.reduce.add` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.add` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_sum[
            output.dtype,
            target=target,
            reduce_dim=axis,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.add")
def reduce_add_shape[
    axis: Int,
](input: Some[Tensor]) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.add` graph op.

    Parameters:
        axis: Dimension along which to sum the elements of `input`.

    Args:
        input: Input tensor reduced along `axis` by summation.

    Returns:
        The output shape after reducing `input` along `axis` by summation.
    """
    return reduce_shape(input, axis)


@extensibility.register("mo.reduce.mul")
struct ReduceMul:
    """Registers the `mo.reduce.mul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.mul` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_product[
            output.dtype,
            target=target,
            reduce_dim=axis,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.mul")
def reduce_mul_shape[
    axis: Int,
](input: Some[Tensor]) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.mul` graph op.

    Parameters:
        axis: Dimension along which to multiply the elements of `input`.

    Args:
        input: Input tensor reduced along `axis` by multiplication.

    Returns:
        The output shape after reducing `input` along `axis`
        by multiplication.
    """
    return reduce_shape(input, axis)


@extensibility.register("mo.reduce.max")
struct ReduceMax:
    """Registers the `mo.reduce.max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.max` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_max[
            output.dtype,
            target=target,
            reduce_dim=axis,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.max")
def reduce_max_shape[
    axis: Int,
](input: Some[Tensor]) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.max` graph op.

    Parameters:
        axis: Dimension along which to take the maximum of `input`.

    Args:
        input: Input tensor reduced along `axis` by maximum.

    Returns:
        The output shape after reducing `input` along `axis` by maximum.
    """
    return reduce_shape(input, axis)


@extensibility.register("mo.reduce.min")
struct ReduceMin:
    """Registers the `mo.reduce.min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.min` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            _trace_name: Name used for tracing and debugging.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[output.dtype, width]) {var output}:
            output._lambda_store[width=width](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_min[
            output.dtype,
            target=target,
            reduce_dim=axis,
        ](input_fn, output_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.min")
def reduce_min_shape[
    axis: Int,
](input: Some[Tensor]) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.min` graph op.

    Parameters:
        axis: Dimension along which to take the minimum of `input`.

    Args:
        input: Input tensor reduced along `axis` by minimum.

    Returns:
        The output shape after reducing `input` along `axis` by minimum.
    """
    return reduce_shape(input, axis)


@extensibility.register("mo.reduce.layer_norm")
struct LayerNorm:
    """Registers the `mo.reduce.layer_norm` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.layer_norm` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            gamma: Per-column scale weights.
            beta: Per-column shift weights.
            epsilon: Small constant for numerical stability.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")
        if Int(gamma.shape()[0]) != Int(input.shape()[rank - 1]):
            raise Error("Gamma size does not match dimension of reduction.")
        if Int(beta.shape()[0]) != Int(input.shape()[rank - 1]):
            raise Error("Beta size does not match dimension of reduction.")

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        # gamma / beta are passed as tile tensors (the Row body loads them
        # via a strided load); gamma's producer fusion, if any, is not used.
        # Pass the static row width when known so the Row register-caches the
        # input strip (`_fuse`), avoiding a second global read of the input in
        # the map phase; a dynamic width falls back to streaming inside the
        # kernel.
        comptime ln_cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if ln_cols != UNKNOWN_VALUE:
            layer_norm[dtype, rank, target=target](
                input_fn,
                output_fn,
                input.shape_coord(),
                ComptimeInt[ln_cols](),
                gamma.to_tile_tensor[.int64](),
                beta.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                ctx,
            )
        else:
            layer_norm[dtype, rank, target=target](
                input_fn,
                output_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                gamma.to_tile_tensor[.int64](),
                beta.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                ctx,
            )


@extensibility.register_shape_function("mo.reduce.layer_norm")
def reduce_layer_norm_shape(
    input: Some[Tensor],
    gamma: Some[Tensor],
    beta: Some[Tensor],
    epsilon: Float32,
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.layer_norm` graph op.

    Args:
        input: Input tensor normalized across the last dimension.
        gamma: Per-column scale weights applied after normalization.
        beta: Per-column shift weights applied after scaling.
        epsilon: Small constant added inside the normalization variance
            for numerical stability.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert (
        type_of(beta).dtype == type_of(input).dtype
    ), "beta dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert type_of(beta).rank == 1, "beta must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.reduce.rms_norm")
struct ReduceRMSNorm:
    """Registers the `mo.reduce.rms_norm` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.rms_norm` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.
            multiply_before_cast: See the graph op signature.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            gamma: Per-column scale weights.
            epsilon: Small constant for numerical stability.
            weight_offset: Scalar offset added to the weight.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        # Shape passed via `input.shape_coord()` to preserve statically-known
        # dims in the `Coord` type.
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        # Pass the static row width when known so the Row register-caches the
        # input strip (`_fuse`), avoiding a second global read of the input in
        # the map phase; a dynamic width falls back to streaming.
        comptime rms_cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if rms_cols != UNKNOWN_VALUE:
            rms_norm[
                dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                output_fn,
                input.shape_coord(),
                ComptimeInt[rms_cols](),
                gamma.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                weight_offset,
                ctx,
            )
        else:
            rms_norm[
                dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                output_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                gamma.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                weight_offset,
                ctx,
            )


@extensibility.register_shape_function("mo.reduce.rms_norm")
def reduce_rms_norm_shape[
    dtype: DType,
](
    input: Some[Tensor],
    gamma: Some[Tensor],
    epsilon: Float32,
    weight_offset: Scalar[dtype=dtype],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.rms_norm` graph op.

    Args:
        input: Input tensor normalized across the last dimension.
        gamma: Per-column scale weights applied after normalization.
        epsilon: Small constant added inside the RMS normalization square
            root for numerical stability.
        weight_offset: Scalar offset added to `gamma` before scaling.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert (
        dtype == type_of(input).dtype
    ), "weight_offset dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.composite.rms_norm_rope")
struct ReduceRMSNormRoPE:
    """Fuses RMS normalization and Rotary Position Embedding (RoPE) into one operation.

    Computes per-row RMS normalization scaled by `weight`, then applies RoPE to
    the normalized values using the provided cosine and sine tables.  The last
    dimension of the input must be an even number.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_dtype: DType,
        cos_sin_dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=output_dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        cos_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        sin_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def cos_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var cos_vals} -> SIMD[cos_sin_dtype, width]:
            return cos_vals._fused_load[
                width=width, element_alignment=alignment
            ](coords)

        @always_inline
        def sin_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var sin_vals} -> SIMD[cos_sin_dtype, width]:
            return sin_vals._fused_load[
                width=width, element_alignment=alignment
            ](coords)

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[output_dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        # Pass the row width when statically known so the kernel can use a SIMD
        # rotate-half (needs `half` divisible by the SIMD width); a dynamic or
        # non-divisible width falls back to scalar inside the kernel.
        comptime rope_cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if rope_cols != UNKNOWN_VALUE:
            rms_norm_rope[
                dtype,
                output_dtype,
                cos_sin_dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                cos_fn,
                sin_fn,
                output_fn,
                input.shape_coord(),
                ComptimeInt[rope_cols](),
                weight.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                weight_offset,
                ctx,
            )
        else:
            rms_norm_rope[
                dtype,
                output_dtype,
                cos_sin_dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                cos_fn,
                sin_fn,
                output_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                weight.to_tile_tensor[.int64](),
                epsilon.cast[dtype](),
                weight_offset,
                ctx,
            )


@extensibility.register_shape_function("mo.composite.rms_norm_rope")
def composite_rms_norm_rope_shape[
    dtype: DType,
](
    input: Some[Tensor],
    weight: Some[Tensor],
    epsilon: Float32,
    weight_offset: Scalar[dtype=dtype],
    cos_vals: Some[Tensor],
    sin_vals: Some[Tensor],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.composite.rms_norm_rope` graph op.

    Args:
        input: Activation tensor normalized by RMS then rotated by RoPE;
            the last dimension must be even.
        weight: Per-column scale weights applied after RMS normalization.
        epsilon: Small constant added inside the RMS normalization square
            root for numerical stability.
        weight_offset: Scalar offset added to `weight` before scaling.
        cos_vals: Cosine table used by the RoPE rotation, matching
            `input` rank.
        sin_vals: Sine table used by the RoPE rotation, matching `input`
            rank.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(weight).dtype == type_of(input).dtype
    ), "weight dtype must match input dtype"
    comptime assert type_of(weight).rank == 1, "weight must be rank 1"
    comptime assert (
        dtype == type_of(input).dtype
    ), "weight_offset dtype must match input dtype"
    comptime assert (
        type_of(cos_vals).dtype == type_of(sin_vals).dtype
    ), "cos_vals and sin_vals must share a dtype"
    comptime assert (
        type_of(cos_vals).rank == type_of(input).rank
    ), "cos_vals rank must match input rank"
    comptime assert (
        type_of(sin_vals).rank == type_of(input).rank
    ), "sin_vals rank must match input rank"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.composite.layer_norm_rope_ragged")
struct LayerNormRopeRagged:
    """Registers the `mo.composite.layer_norm_rope_ragged` graph op with the graph compiler.

    Fuses LayerNorm with ragged RoPE applied to the leading `freqs_cis`-width
    columns of the normalized output; the remaining columns pass through
    unrotated.
    """

    @staticmethod
    def execute[
        input_dtype: DType,
        output_dtype: DType,
        freq_dtype: DType,
        rank: Int,
        target: StaticString,
        interleaved: Bool,
    ](
        output: FusedOutputTensor[dtype=output_dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=input_dtype, rank=rank, ...],
        gamma: InputTensor[dtype=input_dtype, rank=1, ...],
        beta: InputTensor[dtype=input_dtype, rank=1, ...],
        epsilon: Float32,
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")
        if Int(gamma.shape()[0]) != Int(input.shape()[rank - 1]):
            raise Error("Gamma size does not match dimension of reduction.")
        if Int(beta.shape()[0]) != Int(input.shape()[rank - 1]):
            raise Error("Beta size does not match dimension of reduction.")

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[input_dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[output_dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        comptime ln_cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if ln_cols != UNKNOWN_VALUE:
            layer_norm_rope_ragged[
                input_dtype,
                output_dtype,
                freq_dtype,
                rank,
                target=target,
                interleaved=interleaved,
            ](
                input_fn,
                output_fn,
                input.shape_coord(),
                ComptimeInt[ln_cols](),
                gamma.to_tile_tensor[.int64](),
                beta.to_tile_tensor[.int64](),
                epsilon.cast[input_dtype](),
                input_row_offsets.to_tile_tensor[.int64](),
                start_pos.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                ctx,
            )
        else:
            layer_norm_rope_ragged[
                input_dtype,
                output_dtype,
                freq_dtype,
                rank,
                target=target,
                interleaved=interleaved,
            ](
                input_fn,
                output_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                gamma.to_tile_tensor[.int64](),
                beta.to_tile_tensor[.int64](),
                epsilon.cast[input_dtype](),
                input_row_offsets.to_tile_tensor[.int64](),
                start_pos.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                ctx,
            )


@extensibility.register_shape_function("mo.composite.layer_norm_rope_ragged")
def composite_layer_norm_rope_ragged_shape(
    input: Some[Tensor],
    gamma: Some[Tensor],
    beta: Some[Tensor],
    epsilon: Float32,
    input_row_offsets: Some[Tensor],
    start_pos: Some[Tensor],
    freqs_cis: Some[Tensor],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.composite.layer_norm_rope_ragged` graph op.

    Args:
        input: Activation tensor normalized by LayerNorm then partially
            rotated by ragged RoPE.
        gamma: Per-column scale weights applied after normalization.
        beta: Per-column shift weights applied after scaling.
        epsilon: Small constant added inside the normalization variance for
            numerical stability.
        input_row_offsets: Ragged batch boundaries.
        start_pos: Per-sequence cache length used for the RoPE position
            lookup.
        freqs_cis: RoPE frequency table; its width sets the rotated prefix.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert (
        type_of(beta).dtype == type_of(input).dtype
    ), "beta dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert type_of(beta).rank == 1, "beta must be rank 1"
    comptime assert (
        type_of(input_row_offsets).dtype == .uint32
    ), "input_row_offsets must be uint32"
    comptime assert (
        type_of(input_row_offsets).rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        type_of(start_pos).dtype == .uint32
    ), "start_pos must be uint32"
    comptime assert type_of(start_pos).rank == 1, "start_pos must be rank 1"
    comptime assert type_of(freqs_cis).rank == 2, "freqs_cis must be rank 2"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.reduce.group_norm")
struct ReduceGroupNorm:
    """Registers the `mo.reduce.group_norm` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: FusedInputTensor[dtype=dtype, rank=1, ...],
        beta: FusedInputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        num_groups: Int32,
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.group_norm` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            gamma: Per-column scale weights.
            beta: Per-column shift weights.
            epsilon: Small constant for numerical stability.
            num_groups: Number of groups for group normalization.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @__parameter
        @always_inline
        def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](coords)

        @__parameter
        @always_inline
        def gamma_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width](coords)

        @__parameter
        @always_inline
        def beta_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
            return beta._lambda_load[width=width](coords)

        group_norm[dtype, rank, input_fn, gamma_fn, beta_fn, target](
            shape=input.shape_coord(),
            epsilon=epsilon,
            groups=num_groups,
            output=output.to_tile_tensor[.int64](),
            ctx=ctx,
        )


@extensibility.register_shape_function("mo.reduce.group_norm")
def reduce_group_norm_shape(
    input: Some[Tensor],
    gamma: Some[Tensor],
    beta: Some[Tensor],
    epsilon: Float32,
    num_groups: Int32,
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.group_norm` graph op.

    Args:
        input: Input tensor normalized across grouped channels.
        gamma: Per-channel scale weights applied after normalization.
        beta: Per-channel shift weights applied after scaling.
        epsilon: Small constant added inside the normalization variance
            for numerical stability.
        num_groups: Number of groups the channel dimension is split
            into for computing mean and variance.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert (
        type_of(beta).dtype == type_of(input).dtype
    ), "beta dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert type_of(beta).rank == 1, "beta must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.reduce.reduce_min_and_max")
struct ReduceMinAndMax:
    """Registers the `mo.reduce.reduce_min_and_max` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        rank: Int,
        axis: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        comptime norm_axis = axis + rank if axis < 0 else axis
        comptime assert (
            0 <= norm_axis < rank
        ), "axis must be between [0, <input rank>)"

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[dtype, width]:
            return input._fused_load[width=width, element_alignment=alignment](
                coords
            )

        # The op packs both reductions into one `[..., 2, ...]` output tensor:
        # min at slot 0, max at slot 1 of the reduced axis. A `Coord`'s
        # elements are immutable and typed per position, so the retarget goes
        # through `RowCoord`, whose axis element is dynamic and writable.
        @always_inline
        def output_min_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var output}:
            output._fused_store[width=width](
                RowCoord[output.rank](coords).at_axis[norm_axis](0).coord, val
            )

        @always_inline
        def output_max_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[dtype, width]) {var output}:
            output._fused_store[width=width](
                RowCoord[output.rank](coords).at_axis[norm_axis](1).coord, val
            )

        reduce_min_and_max[
            dtype,
            target=target,
            reduce_dim=norm_axis,
        ](input_fn, output_min_fn, output_max_fn, input.shape_coord(), ctx)


@extensibility.register_shape_function("mo.reduce.reduce_min_and_max")
def reduce_reduce_min_and_max_shape[
    axis: Int,
](input: Some[Tensor]) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.reduce.reduce_min_and_max` graph op.

    Parameters:
        axis: Dimension along which to compute the min and max reduction.

    Args:
        input: Input tensor reduced along `axis` by minimum and maximum.

    Returns:
        The output shape with `axis` replaced by 2 (for min and max).
    """
    var new_shape = rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )
    new_shape[_unsafe_normalize_neg_index(axis, type_of(input).rank)] = 2

    return new_shape


@extensibility.register("mo.composite.rms_norm_fused_residual_add")
struct ReduceRMSNormFusedResidualAdd:
    """Registers the `mo.composite.rms_norm_fused_residual_add` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        residual_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        residual_input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma1: InputTensor[dtype=dtype, rank=1, ...],
        gamma2: InputTensor[dtype=dtype, rank=1, ...],
        epsilon1: Float32,
        epsilon2: Float32,
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.composite.rms_norm_fused_residual_add` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.
            multiply_before_cast: See the graph op signature.

        Args:
            output: Output tensor receiving the result.
            residual_output: See the graph op signature.
            input: Input tensor to reduce.
            residual_input: Residual tensor added to the normalized input.
            gamma1: Scale weights for the first normalization.
            gamma2: Scale weights for the second normalization.
            epsilon1: Stability constant for the first normalization.
            epsilon2: Stability constant for the second normalization.
            weight_offset1: Scalar offset for the first weight.
            weight_offset2: Scalar offset for the second weight.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        @always_inline
        def input_fn[
            width: Int
        ](coords: Coord) {var input} -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                coords
            )

        @always_inline
        def residual_input_fn[
            width: Int
        ](coords: Coord) {var residual_input} -> SIMD[dtype, width]:
            return residual_input._lambda_load[width=width](coords)

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var output}:
            output._fused_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](val),
            )

        @always_inline
        def residual_output_fn[
            width: SIMDLength, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var residual_output}:
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                coords,
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        # Preserve a statically-known reduced-axis length (enables the Row
        # register cache); dynamic dims fall back to streaming.
        comptime cols = Int(input.static_spec.shape_tuple[rank - 1])

        comptime if cols != UNKNOWN_VALUE:
            rms_norm_fused_residual_add[
                dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                residual_input_fn,
                output_fn,
                residual_output_fn,
                input.shape_coord(),
                ComptimeInt[cols](),
                gamma1.to_tile_tensor[.int64](),
                epsilon1.cast[dtype](),
                weight_offset1,
                gamma2.to_tile_tensor[.int64](),
                epsilon2.cast[dtype](),
                weight_offset2,
                ctx,
            )
        else:
            rms_norm_fused_residual_add[
                dtype,
                rank,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input_fn,
                residual_input_fn,
                output_fn,
                residual_output_fn,
                input.shape_coord(),
                Int(Int(input.shape()[rank - 1])),
                gamma1.to_tile_tensor[.int64](),
                epsilon1.cast[dtype](),
                weight_offset1,
                gamma2.to_tile_tensor[.int64](),
                epsilon2.cast[dtype](),
                weight_offset2,
                ctx,
            )


@extensibility.register_shape_function(
    "mo.composite.rms_norm_fused_residual_add"
)
def composite_rms_norm_fused_residual_add_shape[
    dtype: DType,
](
    input: Some[Tensor],
    residual_input: Some[Tensor],
    gamma1: Some[Tensor],
    gamma2: Some[Tensor],
    epsilon1: Float32,
    epsilon2: Float32,
    weight_offset1: Scalar[dtype=dtype],
    weight_offset2: Scalar[dtype=dtype],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.composite.rms_norm_fused_residual_add` graph op.

    Args:
        input: Primary input tensor whose shape the output mirrors.
        residual_input: Residual tensor added to the normalized `input`.
        gamma1: Per-column scale weights applied to the first RMS
            normalization.
        gamma2: Per-column scale weights applied to the second RMS
            normalization.
        epsilon1: Small constant added inside the first RMS normalization
            square root for numerical stability.
        epsilon2: Small constant added inside the second RMS normalization
            square root for numerical stability.
        weight_offset1: Scalar offset added to `gamma1` before scaling the
            first normalization.
        weight_offset2: Scalar offset added to `gamma2` before scaling the
            second normalization.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(residual_input).dtype == type_of(input).dtype
    ), "residual_input dtype must match input dtype"
    comptime assert (
        type_of(residual_input).rank == type_of(input).rank
    ), "residual_input rank must match input rank"
    comptime assert (
        type_of(gamma1).dtype == type_of(input).dtype
    ), "gamma1 dtype must match input dtype"
    comptime assert (
        type_of(gamma2).dtype == type_of(input).dtype
    ), "gamma2 dtype must match input dtype"
    comptime assert type_of(gamma1).rank == 1, "gamma1 must be rank 1"
    comptime assert type_of(gamma2).rank == 1, "gamma2 must be rank 1"
    comptime assert (
        dtype == type_of(input).dtype
    ), "weight_offset1/weight_offset2 dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.composite.rms_norm_residual_add")
struct RMSNormResidualAdd:
    """Fused single-norm residual-add + RMSNorm.

    Computes ``intermediate = input + residual_input`` and
    ``output = rms_norm(intermediate, gamma, epsilon, weight_offset)`` in a
    single launch, returning both ``output`` and ``intermediate``. This is the
    canonical transformer/mamba pre-norm boundary ``rms_norm(residual + out)``
    where the pre-add value is carried forward as the next block's residual.
    Reuses the ``state_space`` fused-residual kernel; numerically identical to
    the unfused ``mo.add`` + ``mo.reduce.rms_norm`` pair.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        residual_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        residual_input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        comptime if is_gpu[target]():
            # GPU path: the device kernel bakes the callbacks in as `capturing`
            # comptime closures, so build them as comptime parameters. Reads go
            # through `_lambda_load` so a fused producer op folds into the load.
            @__parameter
            @always_inline
            def input_fn[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
                return input._lambda_load[width=width, element_alignment=width](
                    rebind[IndexList[input.rank]](coords)
                )

            @__parameter
            @always_inline
            def residual_input_fn[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
                return residual_input._lambda_load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )

            @__parameter
            @always_inline
            def output_fn[
                width: SIMDLength, _rank: Int, alignment: Int
            ](coords: IndexList[_rank], val: SIMD[dtype, width]):
                output._fused_store[width=width, element_alignment=alignment](
                    rebind[IndexList[output.rank]](coords),
                    rebind[SIMD[output.dtype, width]](val),
                )

            @__parameter
            @always_inline
            def residual_output_fn[
                width: SIMDLength, _rank: Int, alignment: Int
            ](coords: IndexList[_rank], val: SIMD[dtype, width]):
                residual_output._fused_store[
                    width=width, element_alignment=alignment
                ](
                    rebind[IndexList[residual_output.rank]](coords),
                    rebind[SIMD[residual_output.dtype, width]](val),
                )

            rms_norm_fused_residual[
                input_fn,
                residual_input_fn,
                output_fn,
                residual_output_fn,
                target=target,
                multiply_before_cast=multiply_before_cast,
            ](
                input.shape(),
                gamma.to_tile_tensor[.int64](),
                epsilon,
                weight_offset,
                ctx,
            )
        else:
            # CPU path: the migrated CPU kernel takes unified closures that
            # capture the tensors directly, so it can run the callbacks end to
            # end at runtime (see `RMSNormFusedResidual`). `_fused_load` works
            # whether or not a producer fused in, unlike the GPU-only comptime
            # `_lambda_load`.
            @always_inline
            def input_fn_cpu[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) {var input} -> SIMD[dtype, width]:
                return input._fused_load[width=width](
                    rebind[IndexList[input.rank]](coords)
                )

            @always_inline
            def residual_input_fn_cpu[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) {var residual_input} -> SIMD[
                dtype, width
            ]:
                return residual_input._fused_load[width=width](
                    rebind[IndexList[residual_input.rank]](coords)
                )

            @always_inline
            def output_fn_cpu[
                width: SIMDLength, alignment: Int
            ](coords: IndexList[rank], val: SIMD[dtype, width]) {
                var output
            } -> None:
                output._fused_store[width=width, element_alignment=alignment](
                    rebind[IndexList[output.rank]](coords),
                    rebind[SIMD[output.dtype, width]](val),
                )

            @always_inline
            def residual_output_fn_cpu[
                width: SIMDLength, alignment: Int
            ](coords: IndexList[rank], val: SIMD[dtype, width]) {
                var residual_output
            } -> None:
                residual_output._fused_store[
                    width=width, element_alignment=alignment
                ](
                    rebind[IndexList[residual_output.rank]](coords),
                    rebind[SIMD[residual_output.dtype, width]](val),
                )

            _rms_norm_fused_residual_cpu_entry[
                multiply_before_cast=multiply_before_cast
            ](
                input_fn_cpu,
                residual_input_fn_cpu,
                output_fn_cpu,
                residual_output_fn_cpu,
                input.shape(),
                gamma.to_tile_tensor[.int64](),
                epsilon,
                weight_offset,
            )


@extensibility.register_shape_function("mo.composite.rms_norm_residual_add")
def composite_rms_norm_residual_add_shape[
    dtype: DType,
](
    input: Some[Tensor],
    residual_input: Some[Tensor],
    gamma: Some[Tensor],
    epsilon: Float32,
    weight_offset: Scalar[dtype=dtype],
) -> IndexList[type_of(input).rank]:
    comptime assert (
        type_of(residual_input).dtype == type_of(input).dtype
    ), "residual_input dtype must match input dtype"
    comptime assert (
        type_of(residual_input).rank == type_of(input).rank
    ), "residual_input rank must match input rank"
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma dtype must match input dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert (
        dtype == type_of(input).dtype
    ), "weight_offset dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.bottom_k")
struct BottomK:
    """Registers the `mo.bottom_k` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[.bool],
        ctx: DeviceContext,
    ) raises:
        """Executes the `mo.bottom_k` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.

        Args:
            values: See the graph op signature.
            indices: See the graph op signature.
            input: Input tensor to reduce.
            k: Number of elements to select.
            axis: See the graph op signature.
            sorted: Whether to sort the selected elements.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        top_k[largest=False, target=target](
            input.to_tile_tensor[.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            sorted,
            ctx,
        )


@extensibility.register_shape_function("mo.bottom_k")
def bottom_k_shape(
    input: Some[TileTensorable],
    k: Scalar,
    axis: Scalar,
    sorted: Scalar[.bool],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.bottom_k` graph op.

    Args:
        input: Input tensor to select the bottom-k values from.
        k: Number of smallest elements to select along `axis`.
        axis: Dimension along which to select the bottom-k elements.
        sorted: Whether to sort the selected elements in ascending order.

    Returns:
        The output shape after selecting the bottom-k values along `axis`.
    """
    return rebind[IndexList[type_of(input).rank]](
        top_k_shape_impl(
            input.to_tile_tensor(),
            Int(k),
            Int(axis),
        )
    )


@extensibility.register("mo.top_k")
struct TopK:
    """Registers the `mo.top_k` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[.bool],
        ctx: DeviceContext,
    ) raises:
        """Executes the `mo.top_k` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: Compilation target string.
            _trace_name: Name used for tracing and debugging.

        Args:
            values: See the graph op signature.
            indices: See the graph op signature.
            input: Input tensor to reduce.
            k: Number of elements to select.
            axis: See the graph op signature.
            sorted: Whether to sort the selected elements.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        top_k[largest=True, target=target](
            input.to_tile_tensor[.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            sorted,
            ctx,
        )


@extensibility.register_shape_function("mo.top_k")
def top_k_shape(
    input: Some[TileTensorable],
    k: Scalar,
    axis: Scalar,
    sorted: Scalar[.bool],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.top_k` graph op.

    Args:
        input: Input tensor to select the top-k values from.
        k: Number of largest elements to select along `axis`.
        axis: Dimension along which to select the top-k elements.
        sorted: Whether to sort the selected elements in descending order.

    Returns:
        The output shape after selecting the top-k values along `axis`.
    """
    return rebind[IndexList[type_of(input).rank]](
        top_k_shape_impl(
            input.to_tile_tensor(),
            Int(k),
            Int(axis),
        )
    )


@extensibility.register("mo.reduce.softmax")
struct Softmax:
    """Registers the `mo.reduce.softmax` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        has_prologue_fusion: Bool,
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.softmax` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            has_prologue_fusion: See the graph op signature.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        # Pass the static reduced-axis width when known so the Row
        # register-caches the input strip (`_fuse`), avoiding a second global
        # read of the input in the map phase; a dynamic width falls back to
        # streaming.
        comptime sm_cols = Int(input.static_spec.shape_tuple[axis])

        comptime if sm_cols != UNKNOWN_VALUE:
            softmax[output.dtype, output.rank, target=target, reduce_dim=axis](
                input_fn,
                Coord(output.shape()),
                ComptimeInt[sm_cols](),
                output.to_tile_tensor[.int64](),
                axis,
                context=ctx,
            )
        else:
            softmax[output.dtype, output.rank, target=target, reduce_dim=axis](
                input_fn,
                Coord(output.shape()),
                Int(Int(input.shape()[axis])),
                output.to_tile_tensor[.int64](),
                axis,
                context=ctx,
            )


@extensibility.register("mo.reduce.logsoftmax")
struct LogSoftmax:
    """Registers the `mo.reduce.logsoftmax` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        has_prologue_fusion: Bool,
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Executes the `mo.reduce.logsoftmax` graph op.

        Parameters:
            target: Compilation target string.
            axis: Dimension along which to reduce.
            has_prologue_fusion: See the graph op signature.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """

        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                coords
            )

        # Pass the static reduced-axis width when known so the Row
        # register-caches the input strip (`_fuse`), avoiding a second global
        # read of the input in the map phase; a dynamic width falls back to
        # streaming.
        comptime lsm_cols = Int(input.static_spec.shape_tuple[axis])

        comptime if lsm_cols != UNKNOWN_VALUE:
            softmax[
                output.dtype,
                output.rank,
                target=target,
                logsoftmax=True,
                reduce_dim=axis,
            ](
                input_fn,
                Coord(output.shape()),
                ComptimeInt[lsm_cols](),
                output.to_tile_tensor[.int64](),
                axis,
                context=ctx,
            )
        else:
            softmax[
                output.dtype,
                output.rank,
                target=target,
                logsoftmax=True,
                reduce_dim=axis,
            ](
                input_fn,
                Coord(output.shape()),
                Int(Int(input.shape()[axis])),
                output.to_tile_tensor[.int64](),
                axis,
                context=ctx,
            )


@extensibility.register("mo.cumsum")
struct CumSum:
    """Registers the `mo.cumsum` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
        axis: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        """Executes the `mo.cumsum` graph op.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Tensor rank of the input and output tensors.
            exclusive: See the graph op signature.
            reverse: See the graph op signature.
            axis: Dimension along which to reduce.

        Args:
            output: Output tensor receiving the result.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.
        """
        cumsum[dtype, Bool(exclusive), Bool(reverse), axis=axis](
            output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
        )


@extensibility.register("mx.argsort")
struct ArgSort[*, ascending: Bool]:
    """Registers the `mx.argsort` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        indices: OutputTensor[rank=1, ...],
        input: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        """Executes the `mx.argsort` graph op.

        Parameters:
            target: Compilation target string.

        Args:
            indices: See the graph op signature.
            input: Input tensor to reduce.
            ctx: Device context used to enqueue the kernel.

        Raises:
            Error: If the operation parameters are invalid.
        """
        var indices_tensor = indices.to_tile_tensor[.int64]()
        var input_tensor = input.to_tile_tensor[.int64]()

        comptime if target == "cpu":
            argsort[ascending=Self.ascending](indices_tensor, input_tensor)
        else:
            argsort[ascending=Self.ascending, target=target](
                indices_tensor, input_tensor, ctx
            )
