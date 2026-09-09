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

"""Registers gather, scatter, and indexed-access graph ops backed by the `nn.gather_scatter` kernels."""

from std.math import gcd
from std.sys import align_of
from std.sys.info import simd_width_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext, DeviceContextArray
from max.gpu.host.info import is_cpu
from layout import IntTuple, TileTensor, UNKNOWN_VALUE, coord_to_index_list
from layout.int_tuple import _IntTupleToCoordLike
from layout.coord import DynamicCoord
from layout.tile_layout import Layout as TileLayout
from nn.concat import fused_concat, _fused_dual_concat_gpu
from nn.gather_scatter import (
    Axis,
    ScatterOobIndexStrategy,
    apply_packed_bitmask,
    gather,
    gather_nd,
    gather_nd_shape,
    gather_reduce,
    gather_shape,
    _unsafe_normalize_neg_index as normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
    scatter_nd_shape,
    scatter_set_constant,
)
from nn.index_tensor import (
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor,
)
from nn.reshape import reshape, reshape_shape
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_shape,
    sliced_add,
)
from nn.shard_and_stack import shard_and_stack
from nn.split import split
from nn.tile import tile, tile_shape
from extensibility import (
    ElementwiseUnaryOp,
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
)
from builtin_primitives.primitives import (
    foreach,
    view_copy_impl,
)
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _SliceStrideTypes,
    _TransposeStrideTypes,
)


@always_inline
def check_axis_in_range[idx: Int, dim_size: Int]() raises:
    """Indices passed to gather and scatter ops may be negative. This performs
    a check to see if the axis is valid.

    Raises:
        If the index is out of range [-dim_size, dim_size).
    """
    comptime if -dim_size <= idx < dim_size:
        return

    raise Error("indices must be in range [-dim_size, dim_size)")


@always_inline
def check_axis_in_range[dim_size: Int](idx: Int) raises:
    """Indices passed to gather and scatter ops may be negative. This performs
    a check to see if the axis is valid.

    Raises:
        If the index is out of range [-dim_size, dim_size).
    """
    if -dim_size <= idx < dim_size:
        return

    raise Error("indices must be in range [-dim_size, dim_size)")


@extensibility.register("mo.squeeze_shape")
struct SqueezeShape:
    """Registers the `mo.squeeze_shape` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        indices_type: DType,
    ](
        output_shape: OutputTensor[dtype=dtype, rank=1, ...],
        input_shape: InputTensor[dtype=dtype, rank=1, ...],
        remove_indices: InputTensor[dtype=indices_type, rank=1, ...],
    ) capturing:
        # remove_indices may not be sorted so our strategy is to use -1 to
        # represent removed dimensions in a copied version of our input shape buffer
        var num_input_dims = input_shape.dim_size[0]()
        var num_remove_indices = remove_indices.dim_size[0]()
        var final_rank = num_input_dims - num_remove_indices

        assert (
            final_rank == output_shape.dim_size[0]()
        ), "Incorrect output shape."

        comptime MAX_VECTOR_LIMIT = 12
        assert (
            num_input_dims <= MAX_VECTOR_LIMIT
        ), "Only support shape vectors up to rank-12."
        var input_shape_copy = IndexList[MAX_VECTOR_LIMIT]()
        for i in range(num_input_dims):
            input_shape_copy[i] = Int(input_shape[i])

        # Mark every squeezed dimension as -1 in our copy of the shape tensor
        for remove_index_index in range(num_remove_indices):
            var remove_index = Int(remove_indices[remove_index_index])
            var remove_index_normalize = remove_index + num_input_dims * Int(
                remove_indices[remove_index_index] < 0
            )
            input_shape_copy[remove_index_normalize] = -1

        # # Copy over the non -1 dimensions
        var output_shape_index = 0
        for input_shape_index in range(num_input_dims):
            if input_shape_copy[input_shape_index] == -1:
                continue
            output_shape[output_shape_index] = Scalar[dtype](
                input_shape_copy[input_shape_index]
            )
            output_shape_index += 1


@extensibility.register_shape_function("mo.squeeze_shape")
def squeeze_shape_fn[
    dtype: DType, indices_type: DType
](
    input_shape: InputTensor[dtype=dtype, rank=1, ...],
    remove_indices: InputTensor[dtype=indices_type, rank=1, ...],
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.squeeze_shape` graph op."""
    var out_dim = input_shape.dim_size[0]() - remove_indices.dim_size[0]()

    if out_dim < 0:
        raise Error(
            "[squeeze_shape] cannot remove more dimensions than there exists"
        )

    return IndexList[1](out_dim)


@extensibility.register("mo.unsqueeze_shape")
struct UnsqueezeShape:
    """Registers the `mo.unsqueeze_shape` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        indices_type: DType,
    ](
        output_shape: OutputTensor[dtype=dtype, rank=1, ...],
        input_shape: InputTensor[dtype=dtype, rank=1, ...],
        padding_indices: InputTensor[dtype=indices_type, rank=1, ...],
    ) capturing:
        # represent uninitialized dimensions, add the padding dimensions, and copy
        # over the remaining dimensions later.
        var num_input_dims = input_shape.dim_size[0]()
        var num_padding_indices = padding_indices.dim_size[0]()
        var final_rank = num_input_dims + num_padding_indices
        assert (
            final_rank == output_shape.dim_size[0]()
        ), "Incorrect output shape."
        for output_index in range(final_rank):
            output_shape[output_index] = -1

        for padding_index_index in range(num_padding_indices):
            var padding_index = Int(padding_indices[padding_index_index])
            var padding_index_normalize = padding_index + final_rank * Int(
                padding_indices[padding_index_index] < 0
            )

            assert (
                padding_index_normalize >= 0
                and padding_index_normalize < final_rank
            ), (
                "Padding indices must be between [-r, r-1] where r is the"
                " final output rank."
            )
            assert output_shape[padding_index_normalize] == -1, (
                "Duplicate padding indices point to the same dimension in"
                " the final output shape."
            )
            output_shape[padding_index_normalize] = 1

        # Copy over the remaining shapes
        var orig_shape_index = 0
        for output_shape_index in range(final_rank):
            if output_shape[output_shape_index] != -1:
                continue
            output_shape[output_shape_index] = input_shape[orig_shape_index]
            orig_shape_index += 1


@extensibility.register_shape_function("mo.unsqueeze_shape")
def unsqueeze_shape_fn[
    dtype: DType, indices_type: DType
](
    input_shape: InputTensor[dtype=dtype, rank=1, ...],
    remove_indices: InputTensor[dtype=indices_type, rank=1, ...],
) -> IndexList[1]:
    """Computes the output shape for the `mo.unsqueeze_shape` graph op."""
    var out_dim = input_shape.dim_size[0]() + remove_indices.dim_size[0]()
    return IndexList[1](out_dim)


@extensibility.register("mo.scatter_nd")
struct ScatterND:
    """Registers the `mo.scatter_nd` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        # Existing implementations do not require static shape information
        scatter_nd[target=target](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register_shape_function("mo.scatter_nd")
def scatter_nd_shape_fn[](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, ...],
    indices: InputTensor,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter_nd` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_nd_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.scatter_nd.skip_neg_indices")
struct ScatterNDSkipNegIndices:
    """Registers the `mo.scatter_nd.skip_neg_indices` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        # This is identical to mo.scatter_nd except in how we handle negative indices.
        # In mo.scatter_nd, it is well defined to pass indices between [-dim_size, dim_size).
        # Negative indices will be normalized by incrementing them by dim_size.
        # This allows the kernel to support negative relative indexing.
        # eg: x[-1] == x[dim_size - 1]
        #
        # In mo.scatter_nd.skip_neg_indices, we handle negative indices by skipping
        # the update for that index instead.
        scatter_nd_generator[
            oob_index_strategy=ScatterOobIndexStrategy.SKIP,
            target=target,
            reduce_fn=None,
            _trace_description="scatter_nd.skip_neg_indices",
        ](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    """Registers the `mo.scatter_nd.add` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs + rhs

        scatter_nd_generator[
            target=target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.add",
        ](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register_shape_function("mo.scatter_nd.add")
def scatter_nd_add_shape[](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, ...],
    indices: InputTensor,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter_nd.add` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_nd_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    """Registers the `mo.scatter_nd.mul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_nd_generator[
            target=target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.mul",
        ](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register_shape_function("mo.scatter_nd.mul")
def scatter_nd_mul_shape[](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, ...],
    indices: InputTensor,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter_nd.mul` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_nd_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.scatter_nd.min")
struct ScatterNDMin:
    """Registers the `mo.scatter_nd.min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_nd_generator[
            target=target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.min",
        ](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register_shape_function("mo.scatter_nd.min")
def scatter_nd_min_shape[](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, ...],
    indices: InputTensor,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter_nd.min` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_nd_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.scatter_nd.max")
struct ScatterNDMax:
    """Registers the `mo.scatter_nd.max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        @always_inline
        def reduce_fn[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_nd_generator[
            target=target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.max",
        ](
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context=ctx,
        )


@extensibility.register_shape_function("mo.scatter_nd.max")
def scatter_nd_max_shape[](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, ...],
    indices: InputTensor,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter_nd.max` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_nd_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.apply_packed_bitmask")
struct ApplyPackedBitmask:
    """Registers the `mo.apply_packed_bitmask` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        logits: InputTensor[dtype=dtype, rank=2, ...],
        packed: InputTensor[dtype=.int32, rank=2, ...],
        fill_value: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        apply_packed_bitmask[target](
            output.to_tile_tensor(),
            logits.to_tile_tensor(),
            packed.to_tile_tensor(),
            fill_value,
            ctx,
        )


@extensibility.register("mo.scatter_set_constant")
struct ScatterSetConstant:
    """Registers the `mo.scatter_set_constant` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        data_type: DType,
        index_type: DType,
        //,
        target: StaticString,
    ](
        data: MutableInputTensor[dtype=data_type, rank=2, ...],
        indices: InputTensor[dtype=index_type, rank=2, ...],
        fill_value: Scalar[data_type],
        ctx: DeviceContext,
    ) raises:
        scatter_set_constant[target](
            data.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            fill_value,
            ctx,
        )


@extensibility.register("mo.scatter")
struct Scatter:
    """Registers the `mo.scatter` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](axis)

        scatter_elements(
            input,
            indices,
            updates,
            normalize_neg_index(axis, output.rank),
            output,
            ctx,
        )


@extensibility.register_shape_function("mo.scatter")
def scatter_shape_fn[
    axis: Int,
](
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, rank=input.rank, ...],
    indices: InputTensor[rank=input.rank, ...],
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_elements_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            axis,
        )
    )


@extensibility.register("mo.scatter.add")
struct ScatterAdd:
    """Registers the `mo.scatter.add` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @always_inline
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs + rhs

        scatter_elements[reduce_fn=reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
            ctx,
        )


@extensibility.register_shape_function("mo.scatter.add")
def scatter_add_shape_fn(
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, rank=input.rank, ...],
    indices: InputTensor[rank=input.rank, ...],
    axis: Scalar,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter.add` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_elements_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.max")
struct ScatterMax:
    """Registers the `mo.scatter.max` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @always_inline
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_elements[reduce_fn=reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
            ctx,
        )


@extensibility.register_shape_function("mo.scatter.max")
def scatter_max_shape_fn(
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, rank=input.rank, ...],
    indices: InputTensor[rank=input.rank, ...],
    axis: Scalar,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter.max` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_elements_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.min")
struct ScatterMin:
    """Registers the `mo.scatter.min` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @always_inline
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_elements[reduce_fn=reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
            ctx,
        )


@extensibility.register_shape_function("mo.scatter.min")
def scatter_min_shape_fn(
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, rank=input.rank, ...],
    indices: InputTensor[rank=input.rank, ...],
    axis: Scalar,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter.min` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_elements_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            Int(axis),
        )
    )


@extensibility.register("mo.scatter.mul")
struct ScatterMul:
    """Registers the `mo.scatter.mul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        updates: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        indices: InputTensor[rank=output.rank, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        check_axis_in_range[output.rank](Int(axis))

        @always_inline
        def reduce_func[
            dtype: DType, width: SIMDLength
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_elements[reduce_fn=reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
            ctx,
        )


@extensibility.register_shape_function("mo.scatter.mul")
def scatter_mul_shape_fn(
    input: InputTensor,
    updates: InputTensor[dtype=input.dtype, rank=input.rank, ...],
    indices: InputTensor[rank=input.rank, ...],
    axis: Scalar,
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.scatter.mul` graph op."""
    return rebind[IndexList[input.rank]](
        scatter_elements_shape(
            input.to_tile_tensor[.int64](),
            updates.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            Int(axis),
        )
    )


@extensibility.register("mo.broadcast_to")
struct BroadcastTo:
    """Registers the `mo.broadcast_to` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(input: InputTensor, shape: InputTensor) raises:
        raise Error("Should never be called!")

    @staticmethod
    def shape_impl[
        input_rank: Int, output_rank: Int
    ](
        input: InputTensor[rank=input_rank, ...],
        shape: InputTensor[rank=1, ...],
    ) raises -> IndexList[output_rank]:
        if output_rank != shape.dim_size[0]():
            raise Error(
                "[broadcast_to] requires (len(target_shape) == output_rank)"
            )
        if input_rank > output_rank:
            raise Error("[broadcast_to] requires (input_rank <= output_rank)")

        # move the output shape from buffer into a static int tuple
        var output_shape = IndexList[output_rank]()

        for axis in range(output_rank):
            output_shape[axis] = Int(shape[axis])

        # Validate the compatibility between input and output shapes
        # NOTE we don't need to check the padded dims
        for i in range(input_rank):
            var input_axis = input_rank - i - 1
            var output_axis = output_rank - i - 1
            var input_dim = input.dim_size(input_axis)
            var output_dim = output_shape[output_axis]
            if input_dim != 1 and input_dim != output_dim:
                raise Error(
                    "[broadcast_to] input dimension at index ",
                    input_axis,
                    " (",
                    input_dim,
                    ") must be either 1 or equal to output dimension at index ",
                    output_axis,
                    " (",
                    output_dim,
                    ")",
                )
        return output_shape


@extensibility.register_shape_function("mo.broadcast_to")
def broadcast_to_shape_fn[
    input_rank: Int, output_rank: Int
](
    input: InputTensor[rank=input_rank, ...],
    shape: InputTensor[rank=1, ...],
) raises -> IndexList[output_rank]:
    """Computes the output shape for the `mo.broadcast_to` graph op."""
    return BroadcastTo.shape_impl[output_rank=output_rank](input, shape)


@extensibility.register("mo.broadcast_shape")
struct BroadcastShape:
    """Registers the `mo.broadcast_shape` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def broadcast_shape_impl(
        out_buf: ManagedTensorSlice[rank=1, ...],
        lhs_buf: ManagedTensorSlice[rank=1, ...],
        rhs_buf: ManagedTensorSlice[rank=1, ...],
    ):
        # Ensure lhs is always the smaller shape
        var lhs_rank = lhs_buf.size()
        var rhs_rank = rhs_buf.size()
        assert lhs_rank <= rhs_rank, "lhs shape must be the smaller one"

        # lhs_buf =      [l0, l1, ...]
        # rhs_buf = [..., r0, r1, ...]
        # out_buf = [..., o0, o1, ...]
        var size_diff = rhs_rank - lhs_rank
        for i in range(size_diff):
            out_buf[i] = rhs_buf[i].cast[out_buf.dtype]()

        for lhs_idx in range(lhs_rank):
            var rhs_idx = lhs_idx + size_diff
            var lhs_dim = Int(lhs_buf[lhs_idx])
            var rhs_dim = Int(rhs_buf[rhs_idx])
            if lhs_dim == rhs_dim:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

            elif lhs_dim != 1 and rhs_dim != 1:
                assert rhs_dim == 1, "one of the differing dimensions must be 1"

            elif lhs_dim != 1:
                out_buf[rhs_idx] = lhs_buf[lhs_idx].cast[out_buf.dtype]()

            elif rhs_dim != 1:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(
        out_buf: OutputTensor[rank=1, ...],
        lhs_buf: InputTensor[rank=1, ...],
        rhs_buf: InputTensor[rank=1, ...],
    ):
        var lhs_size = lhs_buf.size()
        var rhs_size = rhs_buf.size()
        if lhs_size > rhs_size:
            return BroadcastShape.broadcast_shape_impl(
                out_buf, rhs_buf, lhs_buf
            )
        return BroadcastShape.broadcast_shape_impl(out_buf, lhs_buf, rhs_buf)


@extensibility.register_shape_function("mo.broadcast_shape")
def broadcast_shape_fn(
    lhs_buf: InputTensor[rank=1, ...], rhs_buf: InputTensor[rank=1, ...]
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.broadcast_shape` graph op."""
    var lhs_dim = lhs_buf.dim_size[0]()
    var rhs_dim = rhs_buf.dim_size[0]()
    return IndexList[1](max(lhs_dim, rhs_dim))


@extensibility.register("mo.static.broadcast_to")
@extensibility.view_kernel
struct StaticBroadcastTo:
    """Registers the `mo.static.broadcast_to` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def build_view[
        out_rank: Int,
    ](x: InputTensor) -> IndexList[out_rank]:
        var new_strides = IndexList[out_rank]()
        comptime delta = out_rank - x.rank

        comptime for i in range(out_rank):
            comptime if i < delta:
                new_strides[i] = 0
            else:
                if x.dim_size[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = x.stride_length[i - delta]()

        return new_strides

    @staticmethod
    def get_view_strides_list[
        out_rank: Int,
        in_rank: Int,
        input_shape: IntTuple,
        input_strides: IntTuple,
    ]() -> IndexList[out_rank]:
        var new_strides = IndexList[out_rank]()
        comptime delta = out_rank - in_rank

        comptime for i in range(out_rank):
            comptime if i < delta:
                new_strides[i] = 0
            else:
                if Int(input_shape[i - delta]) == UNKNOWN_VALUE:
                    new_strides[i] = -1
                elif Int(input_shape[i - delta]) <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = Int(input_strides[i - delta])
        return new_strides

    @staticmethod
    def update_input_view[
        dtype: DType,
        in_rank: Int,
        out_rank: Int,
        //,
        output_static_shape: IntTuple,
    ](
        x: InputTensor[dtype=dtype, rank=in_rank, ...],
        output_shape: IndexList[out_rank],
        out result: InputTensor[
            static_spec=x.static_spec.with_int_tuple_layout[
                out_rank,
                output_static_shape,
                Self.get_view_strides_list[
                    out_rank,
                    x.rank,
                    x._static_shape_tuple,
                    x._static_strides_tuple,
                ](),
            ]()
        ],
    ):
        var x_runtime_strides = Self.build_view[out_rank](x)
        return {x.unsafe_ptr(), output_shape, x_runtime_strides}

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        in_rank: Int,
        out_rank: Int,
        _trace_name: StaticString,
    ](
        z: OutputTensor[dtype=dtype, rank=out_rank, ...],
        x: InputTensor[dtype=dtype, rank=in_rank, ...],
        output_shape: IndexList[out_rank],
        ctx: DeviceContext,
    ) raises:
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.

        var x_view = Self.update_input_view[z._static_shape_tuple](
            x, output_shape
        )

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](z, x_view, ctx)


@extensibility.register("mo.static.reshape")
@extensibility.view_kernel
struct StaticReshape:
    """Registers the `mo.static.reshape` graph op with the graph compiler."""

    @staticmethod
    def update_input_view[
        dtype: DType,
        output_rank: Int,
        //,
        output_static_shape: IntTuple,
    ](
        input: InputTensor[dtype=dtype, ...],
        shape: IndexList[output_rank],
        out result: InputTensor[
            static_spec=input.static_spec.with_row_major_int_tuple_layout[
                output_rank, output_static_shape
            ]()
        ],
    ):
        var view_buffer = reshape(
            input.to_tile_tensor[.int64](),
            shape,
        )

        return {
            view_buffer._storage,
            rebind[IndexList[output_rank]](
                coord_to_index_list(view_buffer.layout.shape_coord())
            ),
            rebind[IndexList[output_rank]](
                coord_to_index_list(view_buffer.layout.stride_coord())
            ),
        }

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        output_rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=output_rank, ...],
        input: InputTensor[dtype=dtype, ...],
        shape: IndexList[output_rank],
        ctx: DeviceContext,
    ) raises:
        var view_tensor = Self.update_input_view[output._static_shape_tuple](
            input, shape
        )

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


@extensibility.register("mo.reshape")
struct Reshape:
    """Registers the `mo.reshape` graph op with the graph compiler."""

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.reshape` to always simplify to `mo.static.reshape`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    def execute(input: InputTensor, shape: InputTensor) raises:
        raise Error("Should never be called!")


@extensibility.register_shape_function("mo.reshape")
def reshape_shape_fn[
    output_rank: Int
](input: InputTensor, shape: InputTensor[rank=1, ...]) raises -> IndexList[
    output_rank
]:
    """Computes the output shape for the `mo.reshape` graph op."""
    return reshape_shape[output_rank=output_rank](
        input.to_tile_tensor[.int64](),
        shape.to_tile_tensor[.int64](),
    )


@extensibility.register("mo.transpose")
@extensibility.view_kernel
struct Transpose:
    """Registers the `mo.transpose` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def transpose_in_place(
        input: InputTensor,
        permutations: InputTensor[rank=1, ...],
        out result: Tuple[IndexList[input.rank], IndexList[input.rank]],
    ):
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        comptime for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return {new_shape, new_stride}

    @staticmethod
    def update_input_view[
        dtype: DType,
        rank: Int,
        //,
        output_static_shape: IntTuple,
        static_permutations: IntTuple,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        permutations: InputTensor[rank=1, ...],
        out result: InputTensor[
            static_spec=input.static_spec.with_tile_layout[
                rank,
                TileLayout[
                    shape_types=_IntTupleToCoordLike[
                        DType.int, output_static_shape
                    ],
                    stride_types=_TransposeStrideTypes[
                        static_permutations,
                        rank,
                        input.static_spec.static_layout._stride_types,
                    ],
                ],
            ]()
        ],
    ):
        var shape, strides = Self.transpose_in_place(input, permutations)
        return {input.unsafe_ptr(), shape, strides}

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        static_permutations: IntTuple,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        permutations: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var view = Self.update_input_view[
            output._static_shape_tuple, static_permutations
        ](input, permutations)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view, ctx)

    # TODO(GEX-1033) Make it possible to have multiple raises.
    @no_inline
    @staticmethod
    def shape_impl(
        input: InputTensor,
        permutations: InputTensor[rank=1, ...],
    ) raises -> IndexList[input.rank]:
        if permutations.dim_size[0]() != input.rank:
            raise Error("[transpose] permutation size must match input rank")

        comptime for i in range(input.rank):
            var perm = Int(permutations[i])
            if perm < 0 or input.rank <= perm:
                raise Error(
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        var shape, _ = Self.transpose_in_place(input, permutations)
        var out = IndexList[input.rank]()

        comptime for i in range(input.rank):
            out[i] = shape[i]

        return out


@extensibility.register_shape_function("mo.transpose")
def transpose_shape_fn(
    input: InputTensor,
    permutations: InputTensor[rank=1, ...],
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.transpose` graph op."""
    return Transpose.shape_impl(input, permutations)


@extensibility.register("mo.slice")
@extensibility.view_kernel
struct Slice:
    """Registers the `mo.slice` graph op with the graph compiler."""

    @staticmethod
    def get_view_alignment[
        rank: Int,
        dtype: DType,
        input_strides: IntTuple,
        static_starts: IntTuple,
        static_steps: IntTuple,
    ](input_alignment: Int) -> Int:
        # Convert IntTuples to CoordLike types at the MLIR level, then
        # use type-level access (no interpreter heap allocation).
        comptime stride_types = _IntTupleToCoordLike[.int, input_strides]
        comptime start_types = _IntTupleToCoordLike[.int, static_starts]
        comptime step_types = _IntTupleToCoordLike[.int, static_steps]

        var alignment = input_alignment
        comptime for i in range(rank):
            # Bail if step is unknown or negative.
            comptime if not step_types[i].is_static_value or step_types[
                i
            ].static_value < 0:
                return 1

            # The offset for dimension `i` is `start[i] * strides[i]`. If the
            # start is not statically known the offset is unknown.
            comptime if not start_types[i].is_static_value:
                return 1

            comptime if start_types[i].static_value != 0:
                comptime if not stride_types[i].is_static_value:
                    return 1
                alignment = gcd(
                    alignment,
                    start_types[i].static_value
                    * stride_types[i].static_value
                    * align_of[dtype](),
                )

            # Stepping along a non-innermost dimension moves the pointer by
            # `step[i] * strides[i]` elements, so that stride bounds the
            # alignment.
            comptime if i != rank - 1:
                comptime if not stride_types[i].is_static_value:
                    return 1
                alignment = gcd(
                    alignment,
                    step_types[i].static_value
                    * stride_types[i].static_value
                    * align_of[dtype](),
                )

        return alignment

    @staticmethod
    def update_input_view[
        dtype: DType,
        rank: Int,
        //,
        output_static_shape: IntTuple,
        static_starts: IntTuple,
        static_steps: IntTuple,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        starts: InputTensor[rank=1, ...],
        stops: InputTensor[rank=1, ...],
        steps: InputTensor[rank=1, ...],
        out result: InputTensor[
            static_spec=input.static_spec.with_tile_layout_and_alignment[
                rank,
                TileLayout[
                    shape_types=_IntTupleToCoordLike[
                        DType.int, output_static_shape
                    ],
                    stride_types=_SliceStrideTypes[
                        rank,
                        input.static_spec.static_layout._stride_types,
                        _IntTupleToCoordLike[.int, static_steps],
                    ],
                ],
            ](
                Self.get_view_alignment[
                    rank,
                    dtype,
                    input._static_strides_tuple,
                    static_starts,
                    static_steps,
                ](input.alignment),
            )
        ],
    ):
        var view_buffer = slice_as_view(
            input.to_tile_tensor[.int64](),
            starts.to_tile_tensor[.int64](),
            stops.to_tile_tensor[.int64](),
            steps.to_tile_tensor[.int64](),
        )

        result = {
            view_buffer._storage,
            rebind[IndexList[rank]](
                coord_to_index_list(view_buffer.layout.shape_coord())
            ),
            rebind[IndexList[rank]](
                coord_to_index_list(view_buffer.layout.stride_coord())
            ),
        }

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        static_starts: IntTuple,
        static_steps: IntTuple,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        starts: InputTensor[rank=1, ...],
        stops: InputTensor[rank=1, ...],
        steps: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var view_tensor = Self.update_input_view[
            output._static_shape_tuple, static_starts, static_steps
        ](input, starts, stops, steps)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


@extensibility.register_shape_function("mo.slice")
def slice_shape_fn(
    input: InputTensor,
    starts: InputTensor[rank=1, ...],
    stops: InputTensor[rank=1, ...],
    steps: InputTensor[rank=1, ...],
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.slice` graph op."""
    return rebind[IndexList[input.rank]](
        slice_shape(
            input.to_tile_tensor[.int64](),
            starts.to_tile_tensor[.int64](),
            stops.to_tile_tensor[.int64](),
            steps.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.mutable.store")
struct MutableStore(ElementwiseUnaryOp):
    """Registers the `mo.mutable.store` graph op with the graph compiler."""

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return val

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        buffer: MutableInputTensor,
        tensor: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        # TODO: Remove the execute method (GEX-2453).
        raise Error("exec should never be called !")


@extensibility.register("mo.mutable.store.slice")
struct MutableStoreSlice:
    """Registers the `mo.mutable.store.slice` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
    ](
        to_buffer: MutableInputTensor[dtype=dtype, rank=rank, ...],
        in_slice: InputTensor[dtype=dtype, rank=rank, ...],
        starts: InputTensor[rank=1, ...],
        stops: InputTensor[rank=1, ...],
        steps: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        copy_to_slice[target=target](
            to_buffer.to_tile_tensor[.int64](),
            in_slice.to_tile_tensor[.int64](),
            starts.to_tile_tensor[.int64](),
            stops.to_tile_tensor[.int64](),
            steps.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.gather_nd")
struct GatherND:
    """Registers the `mo.gather_nd` graph op with the graph compiler."""

    @staticmethod
    def execute[
        batchDims: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor,
        data: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor,
        ctx: DeviceContext,
    ) raises:
        gather_nd[batch_dims=batchDims, target=target](
            data.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("mo.gather_nd")
def gather_nd_shape_fn[
    batch_dims: Int, output_rank: Int
](data: InputTensor, indices: InputTensor) raises -> IndexList[output_rank]:
    """Computes the output shape for the `mo.gather_nd` graph op."""
    return gather_nd_shape[
        batch_dims=batch_dims,
        output_rank=output_rank,
    ](
        data.to_tile_tensor[.int64](),
        indices.to_tile_tensor[.int64](),
    )


@extensibility.register("mo.gather")
struct Gather:
    """Registers the `mo.gather` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, ...],
        indices: FusedInputTensor,
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def input_fn[
            width: Int, _rank: Int, element_alignment: Int
        ](coords: IndexList[_rank]) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[
                width=width, element_alignment=element_alignment
            ](rebind[IndexList[input.rank]](coords))

        @always_inline
        def indices_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) {var indices} -> SIMD[indices.dtype, width]:
            return indices._fused_load[width=width](
                rebind[IndexList[indices.rank]](coords)
            )

        @always_inline
        def output_fn[
            width: SIMDLength, _rank: Int, element_alignment: Int
        ](coords: IndexList[_rank], val: SIMD[output.dtype, width]) {
            var output
        }:
            output._lambda_store[
                width=width, element_alignment=element_alignment
            ](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        gather[
            output.dtype,
            indices.dtype,
            target=target,
        ](
            Axis(axis, input.rank),
            input.shape(),
            indices.shape(),
            output.shape(),
            input_fn=input_fn,
            indices_fn=indices_fn,
            output_fn=output_fn,
            context=ctx,
        )


@extensibility.register_shape_function("mo.gather")
def gather_shape_fn[
    output_rank: Int,
    axis: Int,
](input: InputTensor, indices: InputTensor) raises -> IndexList[output_rank]:
    """Computes the output shape for the `mo.gather` graph op."""
    return gather_shape[output_rank=output_rank](
        input.to_tile_tensor[.int64](),
        indices.to_tile_tensor[.int64](),
        axis,
    )


@extensibility.register("mo.gather_sum")
struct GatherSum:
    """Registers the `mo.gather_sum` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, ...],
        indices: InputTensor[dtype=.int32, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_cpu[target](), "only valid on CPUs"

        def add[
            dtype: DType, simd_width: SIMDLength
        ](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
            dtype, simd_width
        ]:
            return x + y

        gather_reduce[output.dtype, 0, 1, simd_width_of[output.dtype](), add](
            output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            0,
            Optional[DeviceContext](ctx),
        )


@extensibility.register("mo.tile")
struct Tile:
    """Registers the `mo.tile` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        repeats: InputTensor,
    ) raises:
        tile(
            input.to_tile_tensor[.int64](),
            repeats.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
        )


@extensibility.register_shape_function("mo.tile")
def tile_shape_fn(
    input: InputTensor,
    repeats: InputTensor[rank=1, ...],
) raises -> IndexList[input.rank]:
    """Computes the output shape for the `mo.tile` graph op."""
    return rebind[IndexList[input.rank]](
        tile_shape(
            input.to_tile_tensor[.int64](),
            repeats.to_tile_tensor[.int64](),
        )
    )


@extensibility.register("mo.shard_and_stack")
struct ShardWeights:
    """Registers the `mo.shard_and_stack` graph op with the graph compiler."""

    @staticmethod
    def execute[
        axis: Int,
    ](
        outputs: OutputVariadicTensors,
        inputs: InputVariadicTensors[
            dtype=outputs.dtype,
            rank=outputs.rank - 1,
            ...,
        ],
        dev_ctxs_input: DeviceContextArray,
    ) raises:
        shard_and_stack[axis](outputs, inputs, dev_ctxs_input)


@extensibility.register("mo.concat")
struct Concat:
    """Registers the `mo.concat` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()

        check_axis_in_range[axis, output.rank]()

        comptime for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        # Copy-capture `inputs` into the device-kernel closure via
        # `@__copy_capture`. Without it the closure captures `inputs` by
        # reference, so on Metal the GPU kernel holds a host-side pointer to the
        # capture and reads garbage (concat with >=3 inputs silently returned
        # all-zeros). `inputs` (`FusedInputVariadicTensors`) stores its tensors
        # by value, so the copy brings their device pointers into the kernel.
        @always_inline
        @__parameter
        @__copy_capture(inputs)
        def inputs_lambda[
            input_index: Int, width: Int, _rank: Int, alignment: Int = 1
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            comptime assert (
                input_index < inputs.size
            ), "tensor index out of bounds"
            return inputs[input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[rank]](indices))

        # Copy-capture `output` for the same reason as `inputs` above: a
        # by-reference capture leaves the device kernel holding a host pointer
        # to the output tensor on Metal.
        @always_inline
        @__parameter
        @__copy_capture(output)
        def epilogue_wrapper[
            _dtype: DType, _rank: Int, width: SIMDLength, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](indices),
                rebind[SIMD[output.dtype, width]](value),
            )

        fused_concat[
            dtype,
            rank,
            inputs_lambda,
            epilogue_wrapper,
            axis=normalize_neg_index(axis, rank),
            target=target,
        ](
            input_shapes,
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("mo.concat")
def concat_shape_fn[
    dtype: DType,
    rank: Int,
    axis: Int,
](
    inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...]
) raises -> IndexList[rank]:
    """Computes the output shape for the `mo.concat` graph op."""
    return concat_shape_impl(axis, inputs)


@extensibility.register("mo.composite.concat_slice")
struct FusedConcatSlice:
    """Registers the `mo.composite.concat_slice` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        static_starts: IntTuple,
        static_steps: IntTuple,
        axis: Int,
    ](
        concat_output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        slice_output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()

        check_axis_in_range[axis, rank]()

        comptime for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        # Copy-capture the captured tensors into the device-kernel closures.
        # A by-reference capture leaves the GPU kernel holding host-side
        # pointers on Metal, so it reads garbage/zeros (the concat>=4-input
        # all-zeros bug). Copy-capture brings the device pointers into the
        # closure.
        @always_inline
        @__parameter
        @__copy_capture(inputs)
        def inputs_lambda[
            input_index: Int,
            width: Int,
            _rank: Int,
            alignment: Int = 1,
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            comptime assert (
                input_index < inputs.size
            ), "tensor index out of bounds"
            return inputs[input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[rank]](indices))

        @always_inline
        @__parameter
        @__copy_capture(concat_output, slice_output)
        def epilogue_wrapper[
            _dtype: DType, _rank: Int, width: SIMDLength, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_dtype, width]):
            var concat_indices = rebind[IndexList[rank]](indices)

            # Write to the full concat output.
            concat_output._lambda_store[
                width=width, element_alignment=alignment
            ](
                concat_indices,
                rebind[SIMD[concat_output.dtype, width]](value),
            )

            # Check if the current position falls within the slice range.
            # The inner dimension is guaranteed not to be sliced by the pattern,
            # so we only check the outer rank-1 dimensions.
            var slice_indices = IndexList[rank]()

            comptime slice_in_shape = concat_output._static_shape_tuple
            comptime slice_out_shape = slice_output._static_shape_tuple

            comptime for i in range(rank):
                comptime start = Int(static_starts[i])
                comptime step = Int(static_steps[i])

                comptime dim_not_sliced = i == rank - 1 or (
                    start == 0
                    and step == 1
                    and Int(slice_in_shape[i]) != UNKNOWN_VALUE
                    and Int(slice_in_shape[i]) == Int(slice_out_shape[i])
                )

                comptime if dim_not_sliced:
                    slice_indices[i] = concat_indices[i]
                else:
                    var start_norm = (
                        start if start
                        >= 0 else start + concat_output.dim_size[i]()
                    )
                    if (indices[i] - start_norm) % step != 0:
                        return
                    var slice_idx = (indices[i] - start_norm) // step
                    if slice_idx < 0 or slice_idx >= slice_output.dim_size[i]():
                        return
                    slice_indices[i] = slice_idx

            slice_output._lambda_store[
                width=width, element_alignment=alignment
            ](
                slice_indices,
                rebind[SIMD[slice_output.dtype, width]](value),
            )

        fused_concat[
            dtype,
            rank,
            inputs_lambda,
            epilogue_wrapper,
            axis=normalize_neg_index(axis, rank),
            target=target,
        ](
            input_shapes,
            concat_output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.dual_fused_concat_slice")
struct DualFusedConcatSlice:
    """Registers the `mo.dual_fused_concat_slice` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        num_inputs_0: Int,
        static_starts_0: IntTuple,
        static_steps_0: IntTuple,
        static_starts_1: IntTuple,
        static_steps_1: IntTuple,
        axis: Int,
    ](
        concat_output_0: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        slice_output_0: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        concat_output_1: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        slice_output_1: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime num_inputs_1 = inputs.size - num_inputs_0
        check_axis_in_range[axis, rank]()

        var input_shapes_0 = StaticTuple[IndexList[rank], num_inputs_0]()
        comptime for i in range(num_inputs_0):
            input_shapes_0[i] = inputs[i].shape()

        var input_shapes_1 = StaticTuple[IndexList[rank], num_inputs_1]()
        comptime for i in range(num_inputs_1):
            input_shapes_1[i] = inputs[num_inputs_0 + i].shape()

        # Copy-capture the captured tensors into the device-kernel closures.
        # A by-reference capture leaves the GPU kernel holding host-side
        # pointers on Metal, so it reads garbage/zeros. Copy-capture brings
        # the device pointers into the closure.
        @always_inline
        @__parameter
        @__copy_capture(inputs)
        def inputs_lambda_0[
            input_index: Int,
            width: Int,
            _rank: Int,
            alignment: Int = 1,
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            comptime assert (
                input_index < num_inputs_0
            ), "concat 0: tensor index out of bounds"
            return inputs[input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[rank]](indices))

        @always_inline
        @__parameter
        @__copy_capture(inputs)
        def inputs_lambda_1[
            input_index: Int,
            width: Int,
            _rank: Int,
            alignment: Int = 1,
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            comptime assert (
                input_index < num_inputs_1
            ), "concat 1: tensor index out of bounds"
            return inputs[num_inputs_0 + input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[rank]](indices))

        @always_inline
        @__parameter
        @__copy_capture(concat_output_0, slice_output_0)
        def epilogue_0[
            _dtype: DType, _rank: Int, width: SIMDLength, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_dtype, width]):
            var concat_indices = rebind[IndexList[rank]](indices)

            concat_output_0._lambda_store[
                width=width, element_alignment=alignment
            ](
                concat_indices,
                rebind[SIMD[concat_output_0.dtype, width]](value),
            )

            var slice_indices = IndexList[rank]()
            comptime slice_in_shape = concat_output_0._static_shape_tuple
            comptime slice_out_shape = slice_output_0._static_shape_tuple

            comptime for i in range(rank):
                comptime start = Int(static_starts_0[i])
                comptime step = Int(static_steps_0[i])

                comptime dim_not_sliced = i == rank - 1 or (
                    start == 0
                    and step == 1
                    and Int(slice_in_shape[i]) != UNKNOWN_VALUE
                    and Int(slice_in_shape[i]) == Int(slice_out_shape[i])
                )

                comptime if dim_not_sliced:
                    slice_indices[i] = concat_indices[i]
                else:
                    var start_norm = (
                        start if start
                        >= 0 else start + concat_output_0.dim_size[i]()
                    )
                    if (indices[i] - start_norm) % step != 0:
                        return
                    var slice_idx = (indices[i] - start_norm) // step
                    if (
                        slice_idx < 0
                        or slice_idx >= slice_output_0.dim_size[i]()
                    ):
                        return
                    slice_indices[i] = slice_idx

            slice_output_0._lambda_store[
                width=width, element_alignment=alignment
            ](
                slice_indices,
                rebind[SIMD[slice_output_0.dtype, width]](value),
            )

        @always_inline
        @__parameter
        @__copy_capture(concat_output_1, slice_output_1)
        def epilogue_1[
            _dtype: DType, _rank: Int, width: SIMDLength, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_dtype, width]):
            var concat_indices = rebind[IndexList[rank]](indices)

            concat_output_1._lambda_store[
                width=width, element_alignment=alignment
            ](
                concat_indices,
                rebind[SIMD[concat_output_1.dtype, width]](value),
            )

            var slice_indices = IndexList[rank]()
            comptime slice_in_shape = concat_output_1._static_shape_tuple
            comptime slice_out_shape = slice_output_1._static_shape_tuple

            comptime for i in range(rank):
                comptime start = Int(static_starts_1[i])
                comptime step = Int(static_steps_1[i])

                comptime dim_not_sliced = i == rank - 1 or (
                    start == 0
                    and step == 1
                    and Int(slice_in_shape[i]) != UNKNOWN_VALUE
                    and Int(slice_in_shape[i]) == Int(slice_out_shape[i])
                )

                comptime if dim_not_sliced:
                    slice_indices[i] = concat_indices[i]
                else:
                    var start_norm = (
                        start if start
                        >= 0 else start + concat_output_1.dim_size[i]()
                    )
                    if (indices[i] - start_norm) % step != 0:
                        return
                    var slice_idx = (indices[i] - start_norm) // step
                    if (
                        slice_idx < 0
                        or slice_idx >= slice_output_1.dim_size[i]()
                    ):
                        return
                    slice_indices[i] = slice_idx

            slice_output_1._lambda_store[
                width=width, element_alignment=alignment
            ](
                slice_indices,
                rebind[SIMD[slice_output_1.dtype, width]](value),
            )

        _fused_dual_concat_gpu[
            rank,
            dtype,
            inputs_lambda_0,
            epilogue_0,
            num_inputs_0,
            inputs_lambda_1,
            epilogue_1,
            num_inputs_1,
        ](
            normalize_neg_index(axis, rank),
            input_shapes_0,
            concat_output_0.to_tile_tensor[.int64](),
            input_shapes_1,
            concat_output_1.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.split")
struct Split:
    """Registers the `mo.split` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
        axis: Int,
    ](
        output: OutputVariadicTensors[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        split_sizes: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime shape_types = DynamicCoord[.int64, rank].element_types
        # Use Scalar for strides as well since make_dynamic produces all
        # runtime strides.
        comptime stride_types = DynamicCoord[.int64, rank].element_types

        check_axis_in_range[output.rank](axis)

        comptime normalized_axis = axis + rank if axis < 0 else axis

        var output_bufs = StaticTuple[
            TileTensor[
                output.dtype,
                TileLayout[shape_types=shape_types, stride_types=stride_types],
                MutAnyOrigin,
            ],
            output.size,
        ]()

        comptime for i in range(output.size):
            output_bufs[i] = rebind[output_bufs.element_type](
                TileTensor(
                    output[i].unsafe_ptr().as_unsafe_any_origin(),
                    output[i]
                    .to_tile_tensor[.int64]()
                    .layout.make_dynamic[.int64](),
                ),
            )

        split[
            dtype,
            target=target,
            trace_description=_trace_name,
            axis=normalized_axis,
        ](
            input.to_tile_tensor[.int64](),
            output_bufs,
            ctx,
        )


@extensibility.register("split_ith_output_shape")
struct SplitOutputShapeHelper:
    """Registers the `split_ith_output_shape` graph op with the graph compiler.
    """

    @staticmethod
    def execute(
        input_buf: InputTensor,
        split_sizes_buf: InputTensor,
        split_axis: Scalar,
        output_idx: Scalar,
    ) raises:
        raise Error("Should not be called directly.")


@extensibility.register_shape_function("split_ith_output_shape")
def split_ith_output_shape_fn[
    rank: Int,
    input_type: DType,
    split_size_type: DType,
](
    input_buf: InputTensor[dtype=input_type, rank=rank, ...],
    split_sizes_buf: InputTensor[dtype=split_size_type, rank=1, ...],
    split_axis: Scalar,
    output_idx: Scalar,
) raises -> IndexList[rank]:
    """Computes the output shape for the `split_ith_output_shape` graph op."""
    if not (0 <= Int(output_idx) < split_sizes_buf.size()):
        raise Error(
            "[split] output index must be within range [0, len(split_sizes))"
        )

    check_axis_in_range[rank](Int(split_axis))

    var output_split_size = Int(split_sizes_buf[Int(output_idx)])

    var normalized_split_axis = normalize_neg_index(Int(split_axis), rank)

    var split_sizes_sum = 0

    for i in range(split_sizes_buf.dim_size[0]()):
        split_sizes_sum += Int(split_sizes_buf[i])
    if split_sizes_sum != input_buf.dim_size(normalized_split_axis):
        raise Error(
            "[split] sum of split sizes must match input dimension at split"
            " axis"
        )

    var output_shape = input_buf.shape()
    output_shape[normalized_split_axis] = output_split_size
    return output_shape


@extensibility.register("index_tensor")
struct IndexTensor:
    """Registers the `index_tensor` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        indices_type: DType,
        data_rank: Int,
        indices_rank: Int,
        output_rank: Int,
        batch_dims: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=output_rank, ...],
        data: InputTensor[dtype=dtype, rank=data_rank, ...],
        indices: InputTensor[dtype=indices_type, rank=indices_rank, ...],
        ctx: DeviceContext,
    ) raises:
        index_tensor[dtype, indices_type, batch_dims, target=target](
            data.to_tile_tensor[.int64](),
            indices.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("advanced_indexing_getitem")
struct AdvancedIndexingGetItem:
    """Registers the `advanced_indexing_getitem` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_rank: Int,
        index_rank: Int,
        output_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int,
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tensor: OutputTensor[dtype=input_type, rank=output_rank, ...],
        input_tensor: FusedInputTensor[dtype=input_type, rank=input_rank, ...],
        indices: FusedInputVariadicTensors[
            dtype=index_type,
            rank=index_rank,
            size=num_index_tensors,
            ...,
        ],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            output_rank == input_rank + index_rank - num_index_tensors
        )
        # Do not support boolean masks at this time.
        comptime assert index_type != .bool

        @always_inline
        def input_tensor_fn[
            dtype: DType, width: Int
        ](idx: IndexList[input_rank]) {var input_tensor} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](
                input_tensor._fused_load[width](idx)
            )

        @always_inline
        def indices_fn[
            indices_index: Int,
        ](coordinates: IndexList[index_rank]) {var indices} -> Int:
            comptime assert (
                indices_index < num_index_tensors
            ), "tensor index out of bounds"
            return Int(indices[indices_index]._fused_load[width=1](coordinates))

        advanced_indexing_getitem[
            input_rank=input_rank,
            start_axis=start_axis,
            num_index_tensors=num_index_tensors,
            target=target,
            trace_description=_trace_name,
        ](
            out_tensor.to_tile_tensor[.int64](),
            input_tensor.strides(),
            ctx,
            input_tensor_fn,
            indices_fn,
        )


@extensibility.register_shape_function("advanced_indexing_getitem")
def advanced_indexing_getitem_shape_fn[
    input_rank: Int,
    index_rank: Int,
    input_type: DType,
    index_type: DType,
    num_index_tensors: Int,
    //,
    start_axis: Int,
](
    input_tensor: InputTensor[dtype=input_type, rank=input_rank, ...],
    indices: InputVariadicTensors[
        dtype=index_type, rank=index_rank, size=num_index_tensors, ...
    ],
) -> IndexList[input_rank + index_rank - num_index_tensors]:
    """Computes the output shape for the `advanced_indexing_getitem` graph op.
    """
    return advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=num_index_tensors
    ](input_tensor.shape(), indices[0].shape())


@extensibility.register("advanced_indexing_setitem_inplace")
struct AdvancedIndexingSetItemInplace:
    """Registers the `advanced_indexing_setitem_inplace` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_rank: Int,
        index_rank: Int,
        updates_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int,
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        input_tensor: MutableInputTensor[
            dtype=input_type, rank=input_rank, ...
        ],
        updates: FusedInputTensor[dtype=input_type, rank=updates_rank, ...],
        indices: FusedInputVariadicTensors[
            dtype=index_type,
            rank=index_rank,
            size=num_index_tensors,
            ...,
        ],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def updates_tensor_fn[
            dtype: DType, width: Int
        ](idx: IndexList[updates_rank]) {var updates} -> SIMD[dtype, width]:
            return rebind[SIMD[dtype, width]](updates._fused_load[width](idx))

        @always_inline
        def indices_fn[
            indices_index: Int,
        ](coordinates: IndexList[index_rank]) {var indices} -> Int:
            comptime assert (
                indices_index < num_index_tensors
            ), "tensor index out of bounds"
            return Int(indices[indices_index]._fused_load[width=1](coordinates))

        advanced_indexing_setitem_inplace[
            start_axis=start_axis,
            num_index_tensors=num_index_tensors,
            target=target,
            trace_description=_trace_name,
        ](
            input_tensor.to_tile_tensor[.int64](),
            indices[0].shape(),
            updates.strides(),
            ctx,
            updates_tensor_fn,
            indices_fn,
        )


@extensibility.register("advanced_indexing_setitem")
struct AdvancedIndexingSetItem:
    """Registers the `advanced_indexing_setitem` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_rank: Int,
        index_rank: Int,
        updates_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int,
        //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tensor: OutputTensor[dtype=input_type, rank=input_rank, ...],
        input_tensor: FusedInputTensor[dtype=input_type, rank=input_rank, ...],
        updates: FusedInputTensor[dtype=input_type, rank=updates_rank, ...],
        indices: FusedInputVariadicTensors[
            dtype=index_type,
            rank=index_rank,
            size=num_index_tensors,
            ...,
        ],
        ctx: DeviceContext,
    ) capturing raises:
        """Implement basic numpy-style advanced indexing with assignment but returns a copy.
        """

        # First copy over input tensor into the output
        @__parameter
        @always_inline
        def func[
            width: Int, element_alignment: Int
        ](idx: IndexList[output_tensor.rank]) -> SIMD[
            output_tensor.dtype, width
        ]:
            return input_tensor._fused_load[
                width, element_alignment=element_alignment
            ](idx)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name + "_p1/2_copy",
        ](output_tensor, ctx)

        # Then run the updates in-place.
        # For type checking
        var tensor = MutableInputTensor[
            dtype=input_type,
            rank=input_rank,
            static_spec=output_tensor.static_spec,
        ](
            output_tensor._ptr,
            output_tensor.shape(),
            output_tensor.strides(),
        )
        AdvancedIndexingSetItemInplace.execute[
            target=target,
            start_axis=start_axis,
            _trace_name=_trace_name + "_p2/2_update",
        ](tensor, updates, indices, ctx)


@extensibility.register("mo.sliced.add.ragged")
struct Struct_sliced_add_ragged:
    """Registers the `mo.sliced.add.ragged` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=dtype, rank=2, ...],
        a: InputTensor[dtype=dtype, rank=2, ...],
        b: InputTensor[dtype=dtype, rank=2, ...],
        lora_end_idx: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var c_tile_tensor = c.to_tile_tensor[.int64]()
        var a_tile_tensor = a.to_tile_tensor[.int64]()
        var b_tile_tensor = b.to_tile_tensor[.int64]()

        sliced_add[target=target](
            c_tile_tensor,
            a_tile_tensor,
            b_tile_tensor,
            lora_end_idx.to_tile_tensor[.int64](),
            context,
        )
