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
"""Implements the fold operation."""


from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from layout import TileTensor

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


def fold[
    dtype: DType,
    stride: Tuple[Int, Int],
    dilation: Tuple[Int, Int],
    padding: Tuple[Int, Int],
    target: StaticString,
](
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    output_size: IndexList[2],
    kernel_size: IndexList[2],
    ctx: DeviceContext,
) raises:
    """Folds array of sliding local blocks into a single output tensor.

    Parameters:
        dtype: The data type for the input and output.
        stride: Stride of the sliding blocks.
        dilation: Dilation of the sliding blocks.
        padding: 0-paddings to be added on both sides of the inputs.
        target: The target architecture to compile for.

    Args:
        input: Input tensor to fold, shape [N, C x kernel size, num_blocks].
        output: Output tensor to write to, shape [N, C, H, W].
        output_size: Spatial shape of the output tensor (H, W).
        kernel_size: Size of the sliding blocks.
        ctx: The device context.
    """

    comptime assert stride[0] > 0 and stride[1] > 0, "Stride must be positive"
    comptime assert (
        dilation[0] > 0 and dilation[1] > 0
    ), "Dilation must be positive"
    comptime assert (
        padding[0] >= 0 and padding[1] >= 0
    ), "Padding must be non-negative"
    comptime assert output.flat_rank == 4
    comptime assert input.flat_rank == 3

    var N = Int(output.dim(0))
    var C = Int(output.dim(1))
    var H = Int(output.dim(2))
    var W = Int(output.dim(3))

    if output_size[0] != H or output_size[1] != W:
        raise Error("Output tensor size[2:] must be equal to output_size.")

    var channels_col = C * kernel_size[0] * kernel_size[1]

    if input.dim(1) != Scalar[input.linear_idx_type](channels_col):
        raise Error(
            "Input tensor channels must be equal to C * prod(kernel_size)."
        )
    var height_col = (
        H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1

    var width_col = (
        W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    var num_blocks = input.dim(2)

    var expected_blocks = height_col * width_col

    if num_blocks != Scalar[input.linear_idx_type](expected_blocks):
        raise Error(
            "Input tensor must have the same number of blocks ("
            + String(num_blocks)
            + ") as the expected number of blocks ("
            + String(expected_blocks)
            + ")."
        )

    var kernel_w = kernel_size[1]
    var kernel_h = kernel_size[0]
    comptime dilation_w = dilation[1]
    comptime dilation_h = dilation[0]
    comptime stride_w = stride[1]
    comptime stride_h = stride[0]

    @always_inline
    def fold_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 4, "fold_fn: rank must be 4"

        var batch = Int(idx[0].value())
        var channel = Int(idx[1].value())
        var h_out = Int(idx[2].value())
        var w_out = Int(idx[3].value())

        var output_val = Scalar[dtype](0)

        # The span of the kernel in the output tensor.
        var kernel_span_w = (kernel_w - 1) * dilation_w + 1
        var kernel_span_h = (kernel_h - 1) * dilation_h + 1

        # Given the position in the output tensor (h_out, w_out), compute the
        # start and end of the kernel patches that might overlap with this position.
        var h_start = max(((h_out - kernel_span_h) // stride_h + 1), 0)
        var w_start = max(((w_out - kernel_span_w) // stride_w + 1), 0)
        var h_end = min(h_out // stride_h + 1, height_col)
        var w_end = min(w_out // stride_w + 1, width_col)

        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                # compute the relative position of current position in the
                # kernel patch.
                var h_offset = h_out - h * stride_h
                var w_offset = w_out - w * stride_w

                # Check if the current position is covered by the patch.
                if h_offset % dilation_h == 0 and w_offset % dilation_w == 0:
                    h_offset = h_offset // dilation_h
                    w_offset = w_offset // dilation_w

                    var channel_offset = channel * kernel_h * kernel_w
                    var kernel_offset = h_offset * kernel_w + w_offset
                    var patch_offset = h * width_col + w

                    # Load and accumulate
                    comptime assert input.element_size == 1
                    output_val += input[
                        batch, channel_offset + kernel_offset, patch_offset
                    ][0]

        output[idx] = output_val

    var dispatch_shape = (N, C, H, W)
    elementwise[
        simd_width=1,
        target=target,
        _trace_description="fold_fn",
    ](fold_fn, dispatch_shape, ctx)


def fold_shape[
    dtype: DType
](
    input: TileTensor[dtype, ...],
    output_size: IndexList[2],
    kernel_size: IndexList[2],
) raises -> IndexList[4]:
    """Returns the shape of the output tensor of the fold operation."""
    var output_shape = IndexList[4]()
    output_shape[0] = Int(input.dim(0))
    output_shape[1] = Int(input.dim(1)) // (kernel_size[0] * kernel_size[1])
    output_shape[2] = output_size[0]
    output_shape[3] = output_size[1]
    return output_shape
