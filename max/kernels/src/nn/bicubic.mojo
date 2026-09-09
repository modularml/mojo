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
"""This module provides CPU and GPU implementations for bicubic interpolation.

Bicubic interpolation is a 2D extension of cubic interpolation for resampling
digital images. It uses the weighted average of the 4x4 neighborhood of pixels
around the target location to compute the interpolated value.
"""
from std.math import clamp, floor

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from max.gpu import block_dim, block_idx, thread_idx
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord,
    coord_to_index_list,
)
from std.itertools import product
from std.utils.coord import dyn_coord


@always_inline
def map_output_to_input_coord(output_coord: Int, scale: Float32) -> Float32:
    """Map output pixel coordinate to input coordinate using center alignment.
    This implements the standard coordinate mapping for image resizing:
    input_coord = (output_coord + 0.5) * scale - 0.5
    The +0.5 and -0.5 terms ensure pixel centers are aligned properly.
    Args:
        output_coord: Output pixel coordinate.
        scale: Scale factor (input_size / output_size).
    Returns:
        Corresponding input coordinate as a float.
    """
    return (Float32(output_coord) + 0.5) * scale - 0.5


def cubic_kernel(x: Float32) -> Float32:
    """Cubic interpolation kernel matching PyTorch/torchvision's BICUBIC
    filter.

    This uses the Catmull-Rom variant (Robidoux cubic) with a = -0.75,
    which is what PyTorch uses in get_cubic_upsample_coefficients.
    ([Source](https://github.com/pytorch/pytorch/blob/59eb61b2d1e4b64debbefa036acd0d8c7d55f0a3/aten/src/ATen/native/UpSample.h#L410-L423)).
    This also matches OpenCV's [interpolateCubic](https://github.com/opencv/opencv/blob/cf2a3c8e7430cc92569dd7f114609f9377b12d9e/modules/imgproc/src/resize.cpp#L907-L915).

    Args:
        x: Distance from the center point.

    Returns:
        Weight contribution based on the distance.
    """
    # Use a = -0.75 to match the PyTorch bicubic filter.
    var a: Float32 = -0.75
    var abs_x = abs(x)
    var abs_x_squared = abs_x * abs_x
    var abs_x_cubed = abs_x_squared * abs_x

    if abs_x <= 1:
        return (a + 2) * abs_x_cubed - (a + 3) * abs_x_squared + 1
    elif abs_x < 2:
        return a * abs_x_cubed - 5 * a * abs_x_squared + 8 * a * abs_x - 4 * a
    else:
        return 0


@always_inline
def cubic_kernel(x: SIMD) -> type_of(x):
    """Cubic interpolation kernel matching PyTorch/torchvision's BICUBIC
    filter.

    This uses the Catmull-Rom variant (Robidoux cubic) with a = -0.75,
    which is what PyTorch uses in get_cubic_upsample_coefficients.
    ([Source](https://github.com/pytorch/pytorch/blob/59eb61b2d1e4b64debbefa036acd0d8c7d55f0a3/aten/src/ATen/native/UpSample.h#L410-L423)).
    This also matches OpenCV's [interpolateCubic](https://github.com/opencv/opencv/blob/cf2a3c8e7430cc92569dd7f114609f9377b12d9e/modules/imgproc/src/resize.cpp#L907-L915).

    Args:
        x: Distance from the center point.

    Returns:
        Weight contribution based on the distance.
    """
    # Use a = -0.75 to match the PyTorch bicubic filter.
    comptime a = type_of(x)(-0.75)
    var abs_x = abs(x)
    var abs_x_squared = abs_x * abs_x
    var abs_x_cubed = abs_x_squared * abs_x

    # The cubic kernel is defined piecewise:
    # - For |x| <= 1: f(x) = (a + 2) * |x|^3 - (a + 3) * |x|^2 + 1
    # - For 1 < |x| < 2: f(x) = a * |x|^3 - 5 * a * |x|^2 + 8 * a * |x| - 4 * a
    # - For |x| >= 2: f(x) = 0

    var case_1 = (a + 2) * abs_x_cubed - (a + 3) * abs_x_squared + 1
    var case_2 = a * abs_x_cubed - 5 * a * abs_x_squared + 8 * a * abs_x - 4 * a
    var case_3 = type_of(x)(0)

    return abs_x.le(1).select(case_1, abs_x.lt(2).select(case_2, case_3))


def cpu_bicubic_kernel(
    output_host: TileTensor[mut=True, ...],
    input_host: TileTensor,
) -> None:
    """Perform bicubic interpolation on a TileTensor of form NCHW.

    Args:
        output_host: Output tensor with desired dimensions.
        input_host: Input tensor of shape [B, C, H, W].
    """
    comptime assert (
        output_host.flat_rank == 4 and input_host.flat_rank == 4
    ), "bicubic resize only supports rank 4 tensors"
    comptime assert output_host.dtype == input_host.dtype
    # Provide evidence that flat_rank >= 4 for the dyn_coord[DType.int64](...) loads/stores below.
    comptime assert input_host.flat_rank >= 4
    comptime assert output_host.flat_rank >= 4

    # get dimensions
    var input_shape = coord_to_index_list(input_host.layout.shape_coord())
    var output_shape = coord_to_index_list(output_host.layout.shape_coord())
    var batch_size = input_shape[0]
    var channels = input_shape[1]
    var in_height = input_shape[2]
    var in_width = input_shape[3]
    var out_height = output_shape[2]
    var out_width = output_shape[3]

    var scale_h = Float32(in_height) / Float32(out_height)
    var scale_w = Float32(in_width) / Float32(out_width)

    # each output pixel
    for b, c, y_out, x_out in product(
        range(batch_size), range(channels), range(out_height), range(out_width)
    ):
        # get the mapping of where the output pixel lies on input (so if 1d, and scaling from 1-5 to 1-10, x=7 output is mapped to x=3.5 input)
        var in_y = map_output_to_input_coord(y_out, scale_h)
        var in_x = map_output_to_input_coord(x_out, scale_w)

        # get the pixel righhtttt above and to left of the output pixel
        var in_y_floor = Int(floor(in_y))
        var in_x_floor = Int(floor(in_x))

        # (how far away from the pixel above and to the left is the output pixel?)
        var dy = in_y - Float32(in_y_floor)
        var dx = in_x - Float32(in_x_floor)

        # i want to look at surrounding 4x4 pixels, assign weights to them (closer pixels have more weight) and get final value, for each channel
        var sum_value: Float32 = 0.0
        var sum_weights: Float32 = 0.0

        # get the 4x4 surrounding pixels, and assign weights to them
        comptime for i, j in product(range(4), range(4)):
            # don't be <0 or >frame bounds
            var y_pos = clamp(in_y_floor + i - 1, 0, in_height - 1)
            var x_pos = clamp(in_x_floor + j - 1, 0, in_width - 1)

            # This implementation uses the convolution-based bicubic interpolation method,
            # matching PyTorch's implementation with a = -0.75 (Robidoux cubic).
            # This is what most image processing libraries (e.g., PyTorch, torchvision) use.
            var weight_y = cubic_kernel(Float32(i) - 1.0 - dy)
            var weight_x = cubic_kernel(Float32(j) - 1.0 - dx)
            var weight: Float32 = weight_y * weight_x

            # now that i have the weight y and x of said pixel, i multiply it by its weight and add it to the sum
            var pixel_value = Float32(
                input_host.load[width=1](
                    dyn_coord[.int64]((b, c, y_pos, x_pos))
                )
            )
            sum_value += pixel_value * weight
            sum_weights += weight

        # store the result in the output tensor
        output_host.store[width=1](
            dyn_coord[.int64]((b, c, y_out, x_out)),
            sum_value.cast[output_host.dtype](),
        )


@__name(t"gpu_bicubic_{dtype}")
def gpu_bicubic_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    InputLayoutType: TensorLayout,
    input_origin: ImmOrigin,
    OutputEngine: TensorEngine,
    InputEngine: TensorEngine,
](
    output: TileTensor[
        dtype, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    input: TileTensor[dtype, InputLayoutType, input_origin, Engine=InputEngine],
) -> None:
    """Perform bicubic interpolation using GPU.

    Parameters:
        dtype: Element type of the input and output tensors.
        OutputLayoutType: `TensorLayout` of the output tensor.
        output_origin: Mutable `Origin` of the output tensor.
        InputLayoutType: `TensorLayout` of the input tensor.
        input_origin: Immutable `Origin` of the input tensor.
        OutputEngine: Engine of the output tensor.
        InputEngine: Engine of the input tensor.

    Args:
        output: Output tensor with desired dimensions on the device.
        input: Input tensor of shape [B, C, H, W] on the device.
    """

    comptime assert input.flat_rank == 4
    comptime assert output.flat_rank == 4
    # Provide evidence that flat_rank >= 4 for the Coord(..., ...) loads/stores below.
    comptime assert input.flat_rank >= 4
    comptime assert output.flat_rank >= 4
    comptime assert output.element_size == 1
    comptime assert input.element_size == 1

    var b = block_idx.x
    var c = block_idx.y
    var tid = thread_idx.x

    var input_shape = coord_to_index_list(input.layout.shape_coord())
    var output_shape = coord_to_index_list(output.layout.shape_coord())
    var in_height = input_shape[2]
    var in_width = input_shape[3]
    var out_height = output_shape[2]
    var out_width = output_shape[3]

    var scale_h = Float32(in_height) / Float32(out_height)
    var scale_w = Float32(in_width) / Float32(out_width)

    # Each thread processes multiple output pixels
    var total_pixels = out_height * out_width
    var threads_per_block = block_dim.x

    for pixel_idx in range(tid, total_pixels, threads_per_block):
        var y_out, x_out = divmod(pixel_idx, out_width)

        var in_y = map_output_to_input_coord(y_out, scale_h)
        var in_x = map_output_to_input_coord(x_out, scale_w)

        # get the pixel right above and to left of the output pixel
        var in_y_floor = Int(floor(in_y))
        var in_x_floor = Int(floor(in_x))

        # (how far away from the pixel above and to the left is the output pixel?)
        var dy = in_y - Float32(in_y_floor)
        var dx = in_x - Float32(in_x_floor)

        # Look at surrounding 4x4 pixels, assign weights to them (closer pixels
        # have more weight) and get the final value for each channel.
        var sum_value: Float32 = 0.0
        var sum_weights: Float32 = 0.0

        # Pre-compute cubic weights for better performance
        var weights_y = cubic_kernel(SIMD[.float32, 4](-1, 0, 1, 2) - dy)
        var weights_x = cubic_kernel(SIMD[.float32, 4](-1, 0, 1, 2) - dx)

        # get the 4x4 surrounding pixels, and assign weights to them
        comptime for i, j in product(range(4), range(4)):
            # don't be <0 or >frame bounds
            var y_pos = clamp(in_y_floor + i - 1, 0, in_height - 1)
            var x_pos = clamp(in_x_floor + j - 1, 0, in_width - 1)

            # get the weight of the surrounding pixel
            var weight = weights_y[i] * weights_x[j]

            # now that i have the weight y and x of said pixel, i multiply it by its weight and add it to the sum
            var pixel_value = input.load[width=1](
                Coord(b, c, y_pos, x_pos)
            ).cast[.float32]()
            sum_value += pixel_value * weight
            sum_weights += weight

        output.store[width=1](
            Coord(b, c, y_out, x_out),
            sum_value.cast[dtype](),
        )


def resize_bicubic[
    dtype: DType,
    //,
    target: StaticString,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Perform bicubic interpolation.

    Parameters:
        dtype: Element type of the input and output tensors (inferred).
        target: `StaticString` identifying the execution platform, used to
            select between the GPU and CPU code paths.

    Args:
        output: Output tensor with desired dimensions on host or device.
        input: Input tensor of shape [B, C, H, W] on host or device.
        ctx: Device context to enqueue GPU kernels on.
    """
    comptime assert (
        output.flat_rank == 4 and input.flat_rank == 4
    ), "bicubic resize only supports rank 4 tensors"

    comptime if is_gpu[target]():
        var input_shape = coord_to_index_list(input.layout.shape_coord())
        var N = input_shape[0]
        var C = input_shape[1]

        # Use a fixed block size to avoid exceeding CUDA thread limits.
        var block_size = 256
        comptime kernel = gpu_bicubic_kernel[
            output.dtype,
            output_origin=output.origin,
            OutputLayoutType=output.LayoutType,
            OutputEngine=output.Engine,
            input_origin=ImmOrigin(input.origin),
            InputLayoutType=input.LayoutType,
            InputEngine=input.Engine,
        ]
        ctx.enqueue_function[kernel](
            output,
            input.as_immut(),
            grid_dim=(N, C),
            block_dim=(block_size,),
        )
    else:
        cpu_bicubic_kernel(output, input)
