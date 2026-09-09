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
"""Provides average- and max-pooling kernels with configurable padding, dilation, and stride for CPU and GPU."""

from std.sys.info import simd_width_of

from max.algorithm import stencil, stencil_gpu
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from layout import Coord, TileTensor, coord_to_index_list

from std.utils.index import IndexList
from std.utils.numerics import min_or_neg_inf

from .shapes import get_sliding_window_out_dim


@fieldwise_init
struct PoolMethod(TrivialRegisterPassable):
    """Represents the pooling method, selecting between max and average pooling.
    """

    var value: Int
    comptime MAX = PoolMethod(0)  # Max pooling.
    comptime AVG = PoolMethod(1)  # Average pooling not counting padded regions.

    @always_inline("nodebug")
    def __eq__(self, rhs: PoolMethod) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    def __ne__(self, rhs: PoolMethod) -> Bool:
        return self.value != rhs.value


@always_inline
def pool_shape_ceil[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    filter_buf: TileTensor[mut=False, filter_type, ...],
    strides_buf: TileTensor[mut=False, strides_type, ...],
    dilations_buf: TileTensor[mut=False, dilations_type, ...],
    paddings_buf: TileTensor[mut=False, paddings_type, ...],
) raises -> IndexList[input_buf.rank]:
    """Computes the output shape of a pooling operation using ceil rounding for the spatial dimensions.

    Parameters:
        input_type: Data type of the input tensor.
        filter_type: Data type of the filter tensor.
        strides_type: Data type of the strides tensor.
        dilations_type: Data type of the dilations tensor.
        paddings_type: Data type of the paddings tensor.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter size buffer.
        strides_buf: The strides size buffer.
        dilations_buf: The dilations size buffer.
        paddings_buf: The paddings size buffer.

    Returns:
        The output shape with ceil-mode rounding applied.
    """
    return pool_shape_impl[
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        ceil_mode=True,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@always_inline
def pool_shape[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    filter_buf: TileTensor[mut=False, filter_type, ...],
    strides_buf: TileTensor[mut=False, strides_type, ...],
    dilations_buf: TileTensor[mut=False, dilations_type, ...],
    paddings_buf: TileTensor[mut=False, paddings_type, ...],
) raises -> IndexList[input_buf.rank]:
    """Computes the output shape of a pooling operation using floor rounding for the spatial dimensions.

    Parameters:
        input_type: Data type of the input tensor.
        filter_type: Data type of the filter tensor.
        strides_type: Data type of the strides tensor.
        dilations_type: Data type of the dilations tensor.
        paddings_type: Data type of the paddings tensor.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter size buffer.
        strides_buf: The strides size buffer.
        dilations_buf: The dilations size buffer.
        paddings_buf: The paddings size buffer.

    Returns:
        The output shape with floor-mode rounding applied.
    """
    return pool_shape_impl[
        input_type,
        filter_type,
        strides_type,
        dilations_type,
        paddings_type,
        ceil_mode=False,
    ](input_buf, filter_buf, strides_buf, dilations_buf, paddings_buf)


@always_inline
def pool_shape_impl[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    ceil_mode: Bool,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    filter_buf: TileTensor[mut=False, filter_type, ...],
    strides_buf: TileTensor[mut=False, strides_type, ...],
    dilations_buf: TileTensor[mut=False, dilations_type, ...],
    paddings_buf: TileTensor[mut=False, paddings_type, ...],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a pooling operation, and assert the inputs are
    compatible. Works for 2D pool operations only in the NHWC format.

    Parameters:
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.
        ceil_mode: Define rounding mode for shape calculation.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter size buffer.
        strides_buf: The strides size buffer.
        dilations_buf: The dilations size buffer.
        paddings_buf: The paddings size buffer.

    Returns:
        The output shape.
    """
    comptime assert input_buf.rank == 4, "[pooling] requires (input_rank == 4)"
    comptime assert filter_buf.flat_rank == 1
    comptime assert strides_buf.flat_rank == 1
    comptime assert dilations_buf.flat_rank == 1
    comptime assert paddings_buf.flat_rank == 1

    if (
        filter_buf.dim(0)
        != Scalar[filter_buf.linear_idx_type](input_buf.rank - 2)
        or strides_buf.dim(0)
        != Scalar[strides_buf.linear_idx_type](input_buf.rank - 2)
        or dilations_buf.dim(0)
        != Scalar[dilations_buf.linear_idx_type](input_buf.rank - 2)
    ):
        raise Error(
            "[pooling] requires (len(filter) == len(strides) == len(dilations)"
            " == input rank - 2)"
        )

    if paddings_buf.dim(0) != Scalar[paddings_buf.linear_idx_type](
        2 * (input_buf.rank - 2)
    ):
        raise Error(
            "[pooling] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has layout NHWC
    var batch_size = Int(input_buf.dim(0))
    var input_channels = Int(input_buf.dim(3))
    var output_shape = IndexList[input_buf.rank]()
    output_shape[0] = batch_size
    output_shape[input_buf.rank - 1] = input_channels

    comptime for i in range(0, input_buf.rank - 2):
        var input_spatial_dim = Int(input_buf.dim(i + 1))
        var filter = Int(filter_buf[i])
        var stride = Int(strides_buf[i])
        var dilation = Int(dilations_buf[i])
        var pad = Int(paddings_buf[2 * i] + paddings_buf[2 * i + 1])
        var output_spatial_dim = get_sliding_window_out_dim[ceil_mode](
            input_spatial_dim, filter, dilation, stride, pad
        )
        if output_spatial_dim <= 0:
            raise Error("[pooling] output spatial dim must be positive")
        output_shape[i + 1] = output_spatial_dim

    return output_shape


@always_inline
def max_pool_cpu[
    dtype: DType, int_type: DType
](
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
):
    """Computes fp32 pooling.

    Parameters:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            (pad_h_before, pad_h_after, pad_w_before, pad_w_after).
        output: Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """
    comptime assert filter.flat_rank == 1
    comptime assert strides.flat_rank == 1
    comptime assert dilations.flat_rank == 1
    comptime assert paddings.flat_rank == 1

    var empty_padding = True
    for i in range(paddings.num_elements()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    # var padding_w_high = 0 if empty_padding else Int(paddings[3])

    comptime simd_width = simd_width_of[dtype]()

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])

    comptime stencil_rank = 2
    comptime stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) {
        var stride_h,
        var padding_h_low,
        var padding_w_low,
        var stride_w,
        var dilation_h,
        var dilation_w,
        var pool_window_h,
        var pool_window_w,
    } -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[output.rank, ...]) {input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            input.load[width=simd_width](Coord(point))
        )

    @always_inline
    def max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    def max_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        output, mut
    }:
        var i = output.layout(Coord(point))
        output.raw_store(i, val)

    @always_inline
    def dilation_fn(dim: Int) {dilations, mut} -> Int:
        return Int(dilations[dim])

    comptime stencil_with_padding = stencil[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(max_pool_compute_init),
        type_of(max_pool_compute),
        type_of(max_pool_compute_finalize),
    ]

    comptime stencil_empty_padding = stencil[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(max_pool_compute_init),
        type_of(max_pool_compute),
        type_of(max_pool_compute_finalize),
    ]
    # ceil_mode = True implies padding to the right/bottom with neginfinity
    # value, so in that case we use stencil_with_padding
    if empty_padding and not ceil_mode:
        return stencil_empty_padding(
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord())
            ),
            rebind[IndexList[output.rank]](
                coord_to_index_list(input.layout.shape_coord())
            ),
            map_fn,
            dilation_fn,
            load_fn,
            max_pool_compute_init,
            max_pool_compute,
            max_pool_compute_finalize,
        )
    else:
        return stencil_with_padding(
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord()),
            ),
            rebind[IndexList[output.rank]](
                coord_to_index_list(input.layout.shape_coord()),
            ),
            map_fn,
            dilation_fn,
            load_fn,
            max_pool_compute_init,
            max_pool_compute,
            max_pool_compute_finalize,
        )


@always_inline
def max_pool_gpu[
    dtype: DType, int_type: DType
](
    ctx: DeviceContext,
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
) raises:
    """Computes max pooling on GPU.

    Parameters:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        input: (On device) Batched image input to the pool2d operator.
        filter: (On host) Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: (On host) Strides on height and width dimensions with assumed
            tuple (stride_h, stride_w).
        dilations: (On host) Dilations on height and width dimensions with assumed
            tuple (dilation_h, dilation_w).
        paddings: (On host) Paddings on height and width dimensions with assumed
            tuple (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: (On device) Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    comptime assert filter.flat_rank == 1
    comptime assert strides.flat_rank == 1
    comptime assert dilations.flat_rank == 1
    comptime assert paddings.flat_rank == 1

    var empty_padding = True
    for i in range(paddings.num_elements()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    # var padding_w_high = 0 if empty_padding else Int(paddings[3])

    comptime simd_width = 1

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])
    if dilations.layout.product() > 2:
        raise Error("Dilation not supported for size > 2")

    comptime stencil_rank = 2
    comptime stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) {
        var stride_h,
        var padding_h_low,
        var padding_w_low,
        var stride_w,
        var dilation_h,
        var dilation_w,
        var pool_window_h,
        var pool_window_w,
    } -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[output.rank, ...]) {var input} -> SIMD[
        dtype, simd_width
    ]:
        var i = input.layout(Coord(point))
        return rebind[SIMD[dtype, simd_width]](
            input.raw_load[width=simd_width](i)
        )

    @always_inline
    def max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    def max_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        var output
    }:
        var i = output.layout(Coord(point))
        output.raw_store(i, val)

    @always_inline
    def dilation_fn(
        dim: Int,
    ) {var dilation_h, var dilation_w,} -> Int:
        if dim == 0:
            return dilation_h
        else:
            return dilation_w

    comptime stencil_gpu_fn = stencil_gpu[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(max_pool_compute_init),
        type_of(max_pool_compute),
        type_of(max_pool_compute_finalize),
    ]
    return stencil_gpu_fn(
        ctx,
        rebind[IndexList[output.rank]](
            coord_to_index_list(output.layout.shape_coord())
        ),
        rebind[IndexList[output.rank]](
            coord_to_index_list(input.layout.shape_coord())
        ),
        map_fn,
        dilation_fn,
        load_fn,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    )


@always_inline
def avg_pool_cpu[
    dtype: DType,
    int_type: DType,
    rank: Int = 4,
    count_boundary: Bool = False,
](
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
):
    """Computes the average pool.

    Params:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.
        rank: Rank of the input and output tensors (defaults to 4).
        count_boundary: Whether to count the boundary in the average computation.

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            tuple (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            tuple (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            tuple (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    comptime assert filter.flat_rank == 1
    comptime assert strides.flat_rank == 1
    comptime assert dilations.flat_rank == 1
    comptime assert paddings.flat_rank == 1

    var empty_padding = True
    for i in range(paddings.num_elements()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_h_high = 0 if empty_padding else Int(paddings[1])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    var padding_w_high = 0 if empty_padding else Int(paddings[3])

    # If ceil_mode = True, there can be an implicit padding to the right
    # and bottom, so this needs to be added (to later be ignored in
    # avg_pool_compute_finalize_exclude_boundary).
    # Implicit padding equals SAME_UPPER calculations as shown at:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#averagepool
    if ceil_mode and not count_boundary:
        var implicit_pad0 = (
            (Int(output.dim(1)) - 1) * Int(strides[0])
            + ((Int(filter[0]) - 1) * Int(dilations[0]) + 1)
            - Int(input.dim(1))
        )
        var implicit_pad1 = (
            (Int(output.dim(2)) - 1) * Int(strides[1])
            + ((Int(filter[1]) - 1) * Int(dilations[1]) + 1)
            - Int(input.dim(2))
        )
        # Add implicit padding to any specified explicit padding.
        padding_h_high = padding_h_high + implicit_pad0
        padding_w_high = padding_w_high + implicit_pad1

    comptime simd_width = simd_width_of[dtype]()

    var output_height = Int(output.dim[1]())
    var output_width = Int(output.dim[2]())

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])

    comptime stencil_rank = 2
    comptime stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) {
        var stride_h,
        var stride_w,
        var padding_h_low,
        var padding_w_low,
        var dilation_h,
        var dilation_w,
        var pool_window_h,
        var pool_window_w,
    } -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[output.rank, ...]) {var input} -> SIMD[
        dtype, simd_width
    ]:
        var i = input.layout(Coord(point))
        return rebind[SIMD[dtype, simd_width]](
            input.raw_load[width=simd_width](i)
        )

    @always_inline
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    def avg_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def pool_dim_size(
        dim: Int, size: Int, pad_low: Int, pad_high: Int, pool_window_size: Int
    ) -> Int:
        if dim < pad_low:
            return pool_window_size - dim - 1
        elif dim >= size - pad_high:
            return pool_window_size - size + dim
        else:
            return pool_window_size

    @always_inline
    def avg_pool_compute_finalize_exclude_boundary[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        var output,
        var output_height,
        var padding_h_low,
        var padding_h_high,
        var pool_window_h,
        var output_width,
        var padding_w_low,
        var padding_w_high,
        var pool_window_w,
    }:
        var window_h = pool_dim_size(
            point[1],
            output_height,
            padding_h_low,
            padding_h_high,
            pool_window_h,
        )
        var window_w = pool_dim_size(
            point[2], output_width, padding_w_low, padding_w_high, pool_window_w
        )
        var res = val / Scalar[dtype](window_h * window_w)

        var coord = Coord(point)
        var i = output.layout(coord)

        output.raw_store(i, res)

    @always_inline
    def avg_pool_compute_finalize[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        var output,
        var pool_window_h,
        var pool_window_w,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)
        var i = output.layout(Coord(point))
        output.raw_store(i, res)

    def dilation_fn(dim: Int) {dilations, mut} -> Int:
        return Int(dilations[dim])

    comptime stencil_with_padding = stencil[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(avg_pool_compute_init),
        type_of(avg_pool_compute),
        type_of(avg_pool_compute_finalize),
    ]

    comptime stencil_with_padding_count_exclude_boundary = stencil[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(avg_pool_compute_init),
        type_of(avg_pool_compute),
        type_of(avg_pool_compute_finalize_exclude_boundary),
    ]

    comptime stencil_empty_padding = stencil[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(avg_pool_compute_init),
        type_of(avg_pool_compute),
        type_of(avg_pool_compute_finalize),
    ]

    if empty_padding and not ceil_mode:
        return stencil_empty_padding(
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord())
            ),
            rebind[IndexList[output.rank]](
                coord_to_index_list(input.layout.shape_coord())
            ),
            map_fn,
            dilation_fn,
            load_fn,
            avg_pool_compute_init,
            avg_pool_compute,
            avg_pool_compute_finalize,
        )
    else:
        comptime if count_boundary:
            return stencil_with_padding(
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(input.layout.shape_coord())
                ),
                map_fn,
                dilation_fn,
                load_fn,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize,
            )
        else:
            return stencil_with_padding_count_exclude_boundary(
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(input.layout.shape_coord())
                ),
                map_fn,
                dilation_fn,
                load_fn,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_exclude_boundary,
            )


@always_inline
def avg_pool_gpu[
    dtype: DType,
    int_type: DType,
    count_boundary: Bool = False,
](
    ctx: DeviceContext,
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
) raises:
    """Computes the average pool on GPU.

    Params:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.
        count_boundary: Whether to count the boundary in the average computation.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        input: (On device) Batched image input to the pool2d operator.
        filter: (On host) Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: (On host) Strides on height and width dimensions with assumed
            tuple (stride_h, stride_w).
        dilations: (On host) Dilations on height and width dimensions with assumed
            tuple (dilation_h, dilation_w).
        paddings: (On host) Paddings on height and width dimensions with assumed
            tuple (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: (On device) Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
    """

    comptime assert paddings.flat_rank == 1
    comptime assert filter.flat_rank == 1
    comptime assert dilations.flat_rank == 1
    comptime assert strides.flat_rank == 1

    var empty_padding = True
    for i in range(paddings.num_elements()):
        if paddings[i] != 0:
            empty_padding = False
            break

    var padding_h_low = 0 if empty_padding else Int(paddings[0])
    var padding_h_high = 0 if empty_padding else Int(paddings[1])
    var padding_w_low = 0 if empty_padding else Int(paddings[2])
    var padding_w_high = 0 if empty_padding else Int(paddings[3])

    # If ceil_mode = True, there can be an implicit padding to the right
    # and bottom, so this needs to be added (to later be ignored in
    # avg_pool_compute_finalize_exclude_boundary).
    # Implicit padding equals SAME_UPPER calculations as shown at:
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#averagepool
    if ceil_mode and not count_boundary:
        var implicit_pad0 = (
            (Int(output.dim(1)) - 1) * Int(strides[0])
            + ((Int(filter[0]) - 1) * Int(dilations[0]) + 1)
            - Int(input.dim(1))
        )
        var implicit_pad1 = (
            (Int(output.dim(2)) - 1) * Int(strides[1])
            + ((Int(filter[1]) - 1) * Int(dilations[1]) + 1)
            - Int(input.dim(2))
        )
        # Add implicit padding to any specified explicit padding.
        padding_h_high = padding_h_high + implicit_pad0
        padding_w_high = padding_w_high + implicit_pad1

    comptime simd_width = 1  # Must be 1 for GPU

    var output_height = Int(output.dim(1))
    var output_width = Int(output.dim(2))

    var pool_window_h = Int(filter[0])
    var pool_window_w = Int(filter[1])

    var stride_h = Int(strides[0])
    var stride_w = Int(strides[1])

    var dilation_h = Int(dilations[0])
    var dilation_w = Int(dilations[1])
    if dilations.layout.product() > 2:
        raise Error("Dilation not supported for size > 2")

    comptime stencil_rank = 2
    comptime stencil_axis = IndexList[stencil_rank](1, 2)

    @always_inline
    def map_fn(
        point: IndexList[stencil_rank, ...],
    ) {
        var stride_h,
        var stride_w,
        var padding_h_low,
        var padding_w_low,
        var dilation_h,
        var dilation_w,
        var pool_window_h,
        var pool_window_w,
    } -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride_h - padding_h_low,
            point[1] * stride_w - padding_w_low,
        )
        var upper_bound = IndexList[stencil_rank](
            lower_bound[0] + pool_window_h * dilation_h,
            lower_bound[1] + pool_window_w * dilation_w,
        )
        return lower_bound, upper_bound

    @always_inline
    def load_fn[
        simd_width: Int, dtype: DType
    ](point: IndexList[output.rank, ...]) {var input} -> SIMD[
        dtype, simd_width
    ]:
        var i = input.layout(Coord(point))
        return rebind[SIMD[dtype, simd_width]](
            input.raw_load[width=simd_width](i)
        )

    @always_inline
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    def avg_pool_compute[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def pool_dim_size(
        dim: Int, size: Int, pad_low: Int, pad_high: Int, pool_window_size: Int
    ) -> Int:
        if dim < pad_low:
            return pool_window_size - dim - 1
        elif dim >= size - pad_high:
            return pool_window_size - size + dim
        else:
            return pool_window_size

    @always_inline
    def avg_pool_compute_finalize_exclude_boundary[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        var output,
        var output_height,
        var padding_h_low,
        var padding_h_high,
        var pool_window_h,
        var output_width,
        var padding_w_low,
        var padding_w_high,
        var pool_window_w,
    }:
        var window_h = pool_dim_size(
            point[1],
            output_height,
            padding_h_low,
            padding_h_high,
            pool_window_h,
        )
        var window_w = pool_dim_size(
            point[2], output_width, padding_w_low, padding_w_high, pool_window_w
        )
        var res = val / Scalar[dtype](window_h * window_w)

        var i = output.layout(Coord(point))
        output.raw_store(i, res)

    @always_inline
    def avg_pool_compute_finalize[
        simd_width: SIMDLength
    ](
        point: IndexList[output.rank, ...],
        val: SIMD[dtype, simd_width],
    ) {
        var output,
        var pool_window_h,
        var pool_window_w,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)

        var i = output.layout(Coord(point))
        output.raw_store(i, res)

    @always_inline
    def dilation_fn(
        dim: Int,
    ) {var dilation_h, var dilation_w,} -> Int:
        if dim == 0:
            return dilation_h
        else:
            return dilation_w

    comptime stencil_gpu_fn = stencil_gpu[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(avg_pool_compute_init),
        type_of(avg_pool_compute),
        type_of(avg_pool_compute_finalize),
    ]

    comptime stencil_gpu_count_exclude_boundary = stencil_gpu[
        output.rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        type_of(map_fn),
        type_of(dilation_fn),
        type_of(load_fn),
        type_of(avg_pool_compute_init),
        type_of(avg_pool_compute),
        type_of(avg_pool_compute_finalize_exclude_boundary),
    ]

    if empty_padding and not ceil_mode:
        return stencil_gpu_fn(
            ctx,
            rebind[IndexList[output.rank]](
                coord_to_index_list(output.layout.shape_coord())
            ),
            rebind[IndexList[output.rank]](
                coord_to_index_list(input.layout.shape_coord())
            ),
            map_fn,
            dilation_fn,
            load_fn,
            avg_pool_compute_init,
            avg_pool_compute,
            avg_pool_compute_finalize,
        )
    else:
        comptime if count_boundary:
            return stencil_gpu_fn(
                ctx,
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(input.layout.shape_coord())
                ),
                map_fn,
                dilation_fn,
                load_fn,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize,
            )
        else:
            return stencil_gpu_count_exclude_boundary(
                ctx,
                rebind[IndexList[output.rank]](
                    coord_to_index_list(output.layout.shape_coord())
                ),
                rebind[IndexList[output.rank]](
                    coord_to_index_list(input.layout.shape_coord())
                ),
                map_fn,
                dilation_fn,
                load_fn,
                avg_pool_compute_init,
                avg_pool_compute,
                avg_pool_compute_finalize_exclude_boundary,
            )


@always_inline
def avg_pool[
    dtype: DType,
    int_type: DType,
    count_boundary: Bool = False,
    target: StaticString = "cpu",
](
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Dispatches the average pooling operation to the CPU or GPU backend based on the target.

    Parameters:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.
        count_boundary: Whether to count the boundary in the average computation.
        target: Execution target, either "cpu" or "gpu".

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            tuple (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            tuple (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            tuple (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
        ctx: The DeviceContext to use for GPU execution.
    """
    comptime if is_cpu[target]():
        avg_pool_cpu[count_boundary=count_boundary](
            input, filter, strides, dilations, paddings, output, ceil_mode
        )
    elif is_gpu[target]():
        avg_pool_gpu[count_boundary=count_boundary](
            ctx.value(),
            input,
            filter,
            strides,
            dilations,
            paddings,
            output,
            ceil_mode,
        )
    else:
        comptime assert False, "Unknown target " + target


@always_inline
def max_pool[
    dtype: DType,
    int_type: DType,
    target: StaticString = "cpu",
](
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, int_type, ...],
    strides: TileTensor[mut=False, int_type, ...],
    dilations: TileTensor[mut=False, int_type, ...],
    paddings: TileTensor[mut=False, int_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ceil_mode: Bool = False,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Dispatches the max pooling operation to the CPU or GPU backend based on the target.

    Parameters:
        dtype: Data type of the input and output tensors.
        int_type: Data type of the filter, strides, dilations, and paddings tensors.
        target: Execution target, either "cpu" or "gpu".

    Args:
        input: Batched image input to the pool2d operator.
        filter: Filter size on height and width dimensions with assumed tuple
            (filter_h, filter_w).
        strides: Strides on height and width dimensions with assumed
            tuple (stride_h, stride_w).
        dilations: Dilations on height and width dimensions with assumed
            tuple (dilation_h, dilation_w).
        paddings: Paddings on height and width dimensions with assumed
            tuple (pad_h_before, pad_h_after, pad_w_before, pad_w_after)).
        output: Pre-allocated output tensor space.
        ceil_mode: Ceiling mode defines the output shape and implicit padding.
        ctx: The DeviceContext to use for GPU execution.
    """
    comptime if is_cpu[target]():
        max_pool_cpu(
            input, filter, strides, dilations, paddings, output, ceil_mode
        )
    elif is_gpu[target]():
        max_pool_gpu(
            ctx.value(),
            input,
            filter,
            strides,
            dilations,
            paddings,
            output,
            ceil_mode,
        )
    else:
        comptime assert False, "Unknown target " + target
