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
"""Implements tensor resize (upsample/downsample) with nearest, bilinear, and other interpolation modes."""

from std.math import ceil, floor


from max.algorithm.functional import elementwise
from max.algorithm.reduction import _get_nd_indices_from_flat_index
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from std.memory import unsafe_memcpy

from std.utils import IndexList, StaticTuple


struct CoordinateTransformationMode(ImplicitlyCopyable):
    """Specifies how output coordinates map to input coordinates during resize.
    """

    var value: Int
    comptime HalfPixel = CoordinateTransformationMode(0)
    comptime AlignCorners = CoordinateTransformationMode(1)
    comptime Asymmetric = CoordinateTransformationMode(2)
    comptime HalfPixel1D = CoordinateTransformationMode(3)

    @always_inline
    def __init__(out self, value: Int):
        self.value = value

    @always_inline
    def __eq__(self, other: CoordinateTransformationMode) -> Bool:
        return self.value == other.value


@__parameter
@always_inline
def coord_transform[
    mode: CoordinateTransformationMode
](out_coord: Int, in_dim: Int, out_dim: Int, scale: Float32) -> Float32:
    """Maps an output coordinate to an input coordinate according to the given transformation mode.

    Parameters:
        mode: The coordinate transformation mode governing the mapping.

    Args:
        out_coord: The output coordinate to map.
        in_dim: The size of the input dimension.
        out_dim: The size of the output dimension.
        scale: The ratio of output dimension size to input dimension size.

    Returns:
        The corresponding input coordinate as a floating-point value.
    """
    var out_coord_f32 = Float32(out_coord)

    comptime if mode == CoordinateTransformationMode.HalfPixel:
        # note: coordinates are for the CENTER of the pixel
        # - 0.5 term at the end is so that when we round to the nearest integer
        # coordinate, we get the coordinate whose center is closest
        return (out_coord_f32 + Float32(0.5)) / scale - 0.5
    elif mode == CoordinateTransformationMode.HalfPixel1D:
        # Same as HalfPixel except for 1D output. Described here:
        # https://onnx.ai/onnx/operators/onnx__Resize.html
        if out_dim == 1:
            return 0
        return (out_coord_f32 + Float32(0.5)) / scale - 0.5
    elif mode == CoordinateTransformationMode.AlignCorners:
        # aligning "corners" when output is 1D isn't well defined
        # this matches pytorch
        if out_dim == 1:
            return 0
        # note: resized image will have same corners as original image
        return (
            out_coord_f32
            * (Float64(in_dim - 1) / Float64(out_dim - 1)).cast[.float32]()
        )
    elif mode == CoordinateTransformationMode.Asymmetric:
        return out_coord_f32 / scale
    else:
        comptime assert False, "coordinate_transformation_mode not implemented"


struct RoundMode(ImplicitlyCopyable):
    """Specifies how fractional coordinates are rounded to integer indices during nearest-neighbor resize.
    """

    var value: Int
    comptime HalfDown = RoundMode(0)
    comptime HalfUp = RoundMode(1)
    comptime Floor = RoundMode(2)
    comptime Ceil = RoundMode(3)

    @always_inline
    def __init__(out self, value: Int):
        self.value = value

    @always_inline
    def __eq__(self, other: RoundMode) -> Bool:
        return self.value == other.value


@fieldwise_init
struct InterpolationMode(ImplicitlyCopyable):
    """Specifies the interpolation method used during resize."""

    var value: Int
    comptime Linear = InterpolationMode(0)

    @always_inline
    def __eq__(self, other: InterpolationMode) -> Bool:
        return self.value == other.value


struct Interpolator[mode: InterpolationMode](
    Defaultable, TrivialRegisterPassable
):
    """Holds interpolation filter state and applies the filter for a given interpolation mode.
    """

    var cubic_coeff: Float32

    @always_inline
    def __init__(out self, cubic_coeff: Float32):
        self.cubic_coeff = cubic_coeff

    @always_inline
    def __init__(out self):
        self.cubic_coeff = 0

    @staticmethod
    @always_inline
    def filter_length() -> Int:
        comptime assert (
            Self.mode == InterpolationMode.Linear
        ), "InterpolationMode not supported"
        return 1

    @always_inline
    def filter(self, x: Float32) -> Float32:
        comptime assert (
            Self.mode == InterpolationMode.Linear
        ), "InterpolationMode not supported"
        return linear_filter(x)


def resize_nearest_neighbor[
    coordinate_transformation_mode: CoordinateTransformationMode,
    round_mode: RoundMode,
    dtype: DType,
](
    input: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Resizes input to output shape using nearest-neighbor interpolation.

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        round_mode: How to round fractional input coordinates to integer indices.
        dtype: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.
        ctx: The device context used to launch the kernel.
    """
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"
    var scales = StaticTuple[Float32, input.rank]()
    for i in range(input.rank):
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()

    @__parameter
    @always_inline
    def round[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
        comptime if round_mode == RoundMode.HalfDown:
            return ceil(val - 0.5)
        elif round_mode == RoundMode.HalfUp:
            return floor(val + 0.5)
        elif round_mode == RoundMode.Floor:
            return floor(val)
        elif round_mode == RoundMode.Ceil:
            return ceil(val)
        else:
            comptime assert False, "round_mode not implemented"

    def nn_interpolate[
        simd_width: Int, alignment: Int = 1
    ](out_coords: Coord) {var}:
        var in_coords = IndexList[input.rank](0)

        comptime for i in range(input.rank):
            in_coords[i] = min(
                Int(
                    round(
                        coord_transform[coordinate_transformation_mode](
                            Int(out_coords[i].value()),
                            Int(input.dim(i)),
                            Int(output.dim(i)),
                            scales[i],
                        )
                    )
                ),
                Int(input.dim(i)) - 1,
            )

        var in_idx = input.layout(Coord(in_coords))
        var out_idx = output.layout(out_coords)

        output.raw_store(out_idx, input.ptr[in_idx])

    # TODO (#21439): can use unsafe_memcpy when scale on inner dimension is 1
    elementwise[1](nn_interpolate, output.layout.shape_coord(), ctx)


@always_inline
def linear_filter(x: Float32) -> Float32:
    """This is a tent filter.

    f(x) = 1 + x, x < 0
    f(x) = 1 - x, 0 <= x < 1
    f(x) = 0, x >= 1

    """
    var coeff = x
    if x < 0:
        coeff = -x
    if x < 1:
        return 1 - coeff
    return 0


@__parameter
@always_inline
def interpolate_point_1d[
    InputLayoutType: TensorLayout,
    //,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    dtype: DType,
    interpolation_mode: InterpolationMode,
](
    interpolator: Interpolator[interpolation_mode],
    dim: Int,
    out_coords: IndexList[InputLayoutType.rank],
    scale: Float32,
    input: TileTensor[
        mut=False, dtype, InputLayoutType, address_space=.GENERIC, ...
    ],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
):
    """Computes one-dimensional interpolation for a single output point along a given dimension.

    Parameters:
        InputLayoutType: The layout type of the input tensor.
        coordinate_transformation_mode: The coordinate transformation mode to apply.
        antialias: Whether to stretch the filter to antialias when downsampling.
        dtype: The element type of the input and output tensors.
        interpolation_mode: The interpolation mode to use.

    Args:
        interpolator: The interpolator providing the filter function.
        dim: The dimension along which to interpolate.
        out_coords: The multi-dimensional coordinates of the output point.
        scale: The ratio of output dimension size to input dimension size.
        input: The input tensor to read from.
        output: The output tensor to write the interpolated value to.
    """
    var center = (
        coord_transform[coordinate_transformation_mode](
            out_coords[dim], Int(input.dim(dim)), Int(output.dim(dim)), scale
        )
        + 0.5
    )
    var filter_scale = 1 / scale if antialias and scale < 1 else 1
    var support = Float32(interpolator.filter_length()) * filter_scale
    var xmin = max(Int(center - support + 0.5), 0)
    var xmax = min(Int(input.dim(dim)), Int(center + support + 0.5))
    var in_coords = out_coords
    var sum = Scalar[dtype](0)
    var acc = Scalar[dtype](0)
    var ss = 1 / filter_scale
    for k in range(xmax - xmin):
        in_coords[dim] = k + xmin
        var dist_from_center = (
            (Float32(k + xmin) + Float32(0.5)) - center
        ) * ss
        var filter_coeff = interpolator.filter(dist_from_center).cast[dtype]()
        var in_idx = input.layout(Coord(in_coords))
        acc += input.raw_load(in_idx) * filter_coeff
        sum += filter_coeff

    # normalize to handle cases near image boundary where only 1 point is used
    # for interpolation
    var out_idx = output.layout(Coord(out_coords))
    output.raw_store(out_idx, acc / sum)


def resize_linear[
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    dtype: DType,
](
    input: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
):
    """Resizes input to output shape using linear interpolation.

    Parameters:
        coordinate_transformation_mode: How to map a coordinate in output to a coordinate in input.
        antialias: Whether or not to use an antialiasing linear/cubic filter, which when downsampling, uses
            more points to avoid aliasing artifacts. Effectively stretches the filter by a factor of 1 / scale.
        dtype: Type of input and output.

    Args:
        input: The input to be resized.
        output: The output containing the resized input.


    """
    _resize[
        InterpolationMode.Linear, coordinate_transformation_mode, antialias
    ](input, output)


def _resize[
    interpolation_mode: InterpolationMode,
    coordinate_transformation_mode: CoordinateTransformationMode,
    antialias: Bool,
    dtype: DType,
](
    input: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
):
    comptime assert (
        input.rank == output.rank
    ), "input rank must match output rank"

    if rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    ) == rebind[IndexList[input.rank]](
        coord_to_index_list(output.layout.shape_coord())
    ):
        return unsafe_memcpy(
            dest=output.ptr, src=input.ptr, count=input.num_elements()
        )
    var scales = StaticTuple[Float32, input.rank]()
    var resize_dims = List[Int](capacity=input.rank)
    var tmp_dims = IndexList[input.rank](0)
    for i in range(input.rank):
        # need to consider output dims when upsampling and input dims when downsampling
        tmp_dims[i] = max(Int(input.dim(i)), Int(output.dim(i)))
        scales[i] = (Float64(output.dim(i)) / Float64(input.dim(i))).cast[
            DType.float32
        ]()
        if Int(input.dim(i)) != Int(output.dim(i)):
            resize_dims.append(i)
    var interpolator = Interpolator[interpolation_mode]()

    var in_ptr = input.ptr.unsafe_origin_cast[MutUntrackedOrigin]()
    # SAFETY: Placeholder; always overwritten below.
    var out_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()

    var using_tmp1 = False
    var tmp_buffer1 = List[Scalar[dtype]]()
    var tmp_buffer2 = List[Scalar[dtype]]()

    # ping pong between using tmp_buffer1 and tmp_buffer2 to store outputs
    # of 1d interpolation pass across one of the dimensions
    if len(resize_dims) == 1:  # avoid allocating tmp_buffer
        out_ptr = output.ptr.unsafe_origin_cast[MutAnyOrigin]()
    if len(resize_dims) > 1:  # avoid allocating second tmp_buffer
        tmp_buffer1 = List[Scalar[dtype]](
            unsafe_uninit_length=tmp_dims.flattened_length()
        )
        out_ptr = tmp_buffer1.unsafe_ptr().as_unsafe_any_origin()
        using_tmp1 = True
    if len(resize_dims) > 2:  # need a second tmp_buffer
        # TODO: if you are upsampling all dims, you can use the output in place of tmp_buffer2
        # as long as you make sure that the last iteration uses tmp1_buffer as the input
        # and tmp_buffer2 (output) as the output
        tmp_buffer2 = List[Scalar[dtype]](
            unsafe_uninit_length=tmp_dims.flattened_length()
        )
    var in_shape = coord_to_index_list(input.layout.shape_coord())
    var out_shape = coord_to_index_list(input.layout.shape_coord())
    # interpolation is separable, so perform 1d interpolation across each
    # interpolated dimension
    for dim_idx in range(len(resize_dims)):
        if dim_idx == len(resize_dims) - 1:
            out_ptr = output.ptr.unsafe_origin_cast[MutAnyOrigin]()
        var resize_dim = resize_dims[dim_idx]
        out_shape[resize_dim] = Int(output.dim(resize_dim))

        var in_buf = TileTensor(in_ptr, row_major(Coord(in_shape)))
        var out_buf = TileTensor(out_ptr, row_major(Coord(out_shape)))

        var num_rows = out_buf.num_elements() // out_shape[resize_dim]
        for row_idx in range(num_rows):
            var coords = _get_nd_indices_from_flat_index(
                row_idx, out_shape, resize_dim
            )
            for i in range(out_shape[resize_dim]):
                coords[resize_dim] = i
                interpolate_point_1d[
                    InputLayoutType=in_buf.LayoutType,
                    coordinate_transformation_mode,
                    antialias,
                ](
                    interpolator,
                    resize_dim,
                    rebind[IndexList[in_buf.rank]](coords),
                    scales[resize_dim],
                    in_buf,
                    out_buf,
                )

        in_shape = out_shape
        in_ptr = out_ptr.unsafe_origin_cast[MutUntrackedOrigin]()

        out_ptr = (
            tmp_buffer2.unsafe_ptr() if using_tmp1 else tmp_buffer1.unsafe_ptr()
        ).as_unsafe_any_origin()
        using_tmp1 = not using_tmp1

    _ = tmp_buffer1^
    _ = tmp_buffer2^
