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
"""Implements the reshape operation that reinterprets a contiguous tensor's data under a new shape."""

from layout import Coord, TileTensor
from layout.coord import DynamicCoord
from layout.tile_layout import Layout

from std.utils.index import IndexList


# Reshape assumes inputs are contiguous. It should always be fused last and
# a non-contiguous tensor cannot be fused *into* this as input.
@always_inline
def reshape[
    dtype: DType,
    //,
    output_rank: Int,
](
    input: TileTensor[dtype, ...],
    new_shape: IndexList[output_rank],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=DynamicCoord[.int64, output_rank].element_types,
        stride_types=DynamicCoord[.int64, output_rank].element_types,
    ],
    input.origin,
    address_space=input.address_space,
    Engine=input.Engine,
]:
    """Returns a view of the contiguous `input` tensor reinterpreted under `new_shape` with matching element count.

    Computes row-major (contiguous) strides for the requested shape and
    constructs a new `TileTensor` that shares the input's underlying buffer,
    origin, and address space. The caller is responsible for ensuring the
    input is contiguous.

    Parameters:
        dtype: Element type of the input tensor.
        output_rank: Rank of the requested output shape.

    Args:
        input: Contiguous source tensor whose data is reinterpreted.
        new_shape: Desired output shape, whose element product must match
            the number of elements in `input`.

    Returns:
        A `TileTensor` view over the same buffer with the new shape and
        contiguous strides.
    """
    var stride_tuple = type_of(new_shape)()
    var stride: Int = 1

    # Create contiguous strides.
    comptime for i in reversed(range(output_rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= new_shape[i]

    # Return a view with the new shape, preserving the input's engine
    # (for example a device tile stays `DevicePointerEngine`) by rebinding the
    # storage handle rather than erasing it through a raw pointer.
    return {
        input._storage,
        Layout(Coord(new_shape), Coord(stride_tuple)),
    }


@always_inline
def reshape_shape[
    output_rank: Int,
    input_type: DType,
    target_shape_type: DType,
](
    input_buf: TileTensor[input_type, ...],
    target_shape_buf: TileTensor[target_shape_type, ...],
) raises -> IndexList[output_rank]:
    """Reads a target shape from a buffer and returns it as a static `IndexList`, inferring any single `-1` dimension.

    Copies the rank-1 `target_shape_buf` into a static `IndexList[output_rank]`,
    allowing at most one dimension to be specified as `-1`, which is then inferred
    from the input's element count. Raises an error if the constraints are
    violated or the inferred shape does not match the input's number of elements.

    Parameters:
        output_rank: Expected rank of the target shape.
        input_type: Element type of the input buffer.
        target_shape_type: Element type of the target shape buffer.

    Args:
        input_buf: Source tensor whose element count constrains the inferred shape.
        target_shape_buf: Rank-1 tensor holding the desired output dimensions,
            where at most one entry may be `-1`.

    Returns:
        A static `IndexList[output_rank]` with any `-1` dimension resolved.

    Raises:
        Error: If `target_shape_buf` is not rank 1, the rank does not match
            `output_rank`, more than one `-1` is present, a negative value
            other than `-1` appears, or the resulting element count does not
            match the input.
    """
    comptime assert (
        target_shape_buf.flat_rank == 1
    ), "target_shape_buf must be rank 1"
    if output_rank != Int(target_shape_buf.dim(0)):
        raise Error("[reshape] requires (len(target_shape) == output_rank)")

    # move the target shape from buffer into a static int tuple; also check and
    # record if there's any to-be-inferred dimension (-1).
    var target_shape = IndexList[output_rank]()
    var to_be_inferred_axis = -1
    var non_negative_dim_product = 1
    for axis in range(output_rank):
        var target_dim = Int(target_shape_buf[axis])
        target_shape[axis] = target_dim
        if target_dim < 0:
            if target_dim != -1:
                raise Error(
                    "[reshape] only -1 is allowed as a negative value in target"
                    " shape"
                )
            if to_be_inferred_axis != -1:
                raise Error("[reshape] only one -1 is allowed in target shape")
            to_be_inferred_axis = axis
        else:
            non_negative_dim_product *= target_dim

    var input_num_elems = input_buf.num_elements()
    var output_num_elems = non_negative_dim_product
    # Infer a dimension as the remaining elements, if needed.
    if to_be_inferred_axis != -1:
        target_shape[to_be_inferred_axis] = (
            input_num_elems // non_negative_dim_product
        )
        output_num_elems *= target_shape[to_be_inferred_axis]

    if output_num_elems != input_num_elems:
        raise Error("[reshape] input and output number of elements must match")

    return target_shape
