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
"""Op implementation for conv2d."""

from max._core.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import ConvInputLayout, FilterLayout, Shape
from ..value import TensorValue, TensorValueLike
from .elementwise import add
from .permute import permute
from .validation import assert_same_device


def conv2d_transpose(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    output_paddings: tuple[int, int] = (0, 0),
    bias: TensorValueLike | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.RSCF,
) -> TensorValue:
    """Computes the 2-D deconvolution of the input with the given filter, strides, dilations, and paddings.

    This computes the transpose (gradient) of convolution, with the following
    layout assumptions (where ``out_channels`` is with respect to the original
    convolution):

    - The input ``x`` has channels-last (NHWC) layout, meaning
      ``(batch_size, height, width, in_channels)``.
    - The filter has RSCF layout, meaning
      ``(kernel_height, kernel_width, out_channels, in_channels)``.
    - The bias has shape ``(out_channels,)``.

    This op effectively computes the gradient of a convolution with respect to
    its input, as if the original convolution had the same filter and
    hyperparameters as this op. For a visualization of the computation, see
    `Transposed Convolution
    <https://d2l.ai/chapter_computer-vision/transposed-conv.html>`_.

    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The ``padding`` values apply to the *output* spatial dimensions and take
    the form ``(pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after)``, where dim1 is H_out and dim2 is W_out. Each pair crops
    that many rows or columns of zeros from the corresponding edge of the
    output, so padding shrinks the transposed-convolution output rather than
    enlarging it as in a forward convolution.

    Args:
        x: An NHWC input tensor to perform the deconvolution upon.
        filter: The convolution filter in RSCF layout,
            ``(height, width, out_channels, in_channels)``.
        stride: A tuple ``(stride_h, stride_w)``. Defaults to ``(1, 1)``.
        dilation: The spacing between the kernel points.
        padding: The number of rows and columns cropped from the borders of the
            output spatial dimensions.
        output_paddings: The number of zeros added at the end of each output
            spatial axis. This resolves the ambiguity between multiple output
            shapes when a stride is greater than 1. Only ``0`` is supported.
        bias: An optional tensor of shape ``(out_channels,)``.
        input_layout: The layout of the input tensor. Defaults to NHWC.
        filter_layout: The layout of the filter tensor. Defaults to RSCF.

    Returns:
        A ``TensorValue`` representing the result of the deconvolution, with
        shape ``(batch_size, out_channels, height_out, width_out)`` in
        channels-first (NCHW) layout. This differs from the channels-last
        (NHWC) layout of the input.

    Raises:
        ValueError: If ``x`` isn't rank 4, ``filter`` isn't rank 4, ``bias`` is
            given and isn't rank 1, an output padding isn't smaller than its
            stride, or ``x`` and ``filter`` aren't on the same device.
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D deconvolution must be rank 1 with shape"
                " (out_channels,)"
            )

    if x.rank != 4:
        raise ValueError(
            "input to a 2-D deconvolution must be rank 4 with shape"
            " (batch_size, height, width, in_channels)"
        )

    if filter.rank != 4:
        raise ValueError(
            "filter for a 2-D deconvolution must be rank 4 with shape (height,"
            " width, out_channels, in_channels)"
        )
    if output_paddings[0] >= stride[0] or output_paddings[1] >= stride[1]:
        raise ValueError(
            "output padding must be smaller than either stride or dilation,"
            f" but got output_padding = {output_paddings}"
        )

    # TODO(GEX-2043): Add support for GPU kernel for conv_transpose and remove manual transfers
    # original_device = x.type.device
    # x = x.to(DeviceRef.CPU())
    # filter = filter.to(DeviceRef.CPU())

    assert_same_device(x=x, filter=filter)

    out = Graph.current._add_op_generated(
        rmo.ConvTransposeOp,
        input=x,
        filter=filter._with_layout(filter_layout),
        strides=Shape(stride),
        dilations=Shape(dilation),
        paddings=Shape(padding),
        output_paddings=Shape(output_paddings),
        input_layout=input_layout,
    )[0].tensor

    # out = out.to(original_device)

    if bias is not None:
        # Convert from NCHW to NHWC for bias broadcasting.
        # TODO: There should be a better way without transpose.
        out = permute(out, [0, 2, 3, 1])
        out = add(out, bias)
        # Convert back from NHWC to NCHW.
        return permute(out, [0, 3, 1, 2])
    return out
