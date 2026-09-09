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
"""Op implementation for pooling (max, avg, roi_align, etc)."""

from __future__ import annotations

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import DimLike
from ..graph import Graph
from ..shape import Shape
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def avg_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_boundary: bool = True,
) -> TensorValue:
    """Applies 2D average pooling.

    Slides a window of size ``kernel_size`` over the spatial dimensions
    and replaces each window with its average value.

    Args:
        input: The input tensor in channels-last (NHWC) layout,
            ``(batch_size, height, width, channels)``.
        kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height
            and width of the sliding window.
        stride: The stride of the sliding window. Either a single ``int``
            applied to both spatial dimensions, or a tuple
            ``(stride_h, stride_w)``. Defaults to ``1``.
        dilation: The spacing between kernel elements. Either a single
            ``int`` applied to both spatial dimensions, or a tuple
            ``(dilation_h, dilation_w)``. Defaults to ``1``.
        padding: Zero-padding added to both sides of each spatial
            dimension. Either a single ``int`` applied to both spatial
            dimensions, or a tuple ``(pad_h, pad_w)``. Defaults to ``0``.
        ceil_mode: When ``True``, uses ceil instead of floor when
            computing the output spatial shape. Defaults to ``False``.
        count_boundary: When ``True``, includes padding elements in the
            divisor when computing the average. Defaults to ``True``.

    Returns:
        A ``TensorValue`` representing the averaged values, with shape
        ``(batch_size, height_out, width_out, channels)``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op_generated(
        rmo.AvgPoolOp,
        input=input,
        filter_shape=Shape(kernel_size),
        strides=Shape(stride),
        dilations=Shape(dilation),
        paddings=Shape(_padding),
        ceil_mode=ceil_mode,
        count_boundary=count_boundary,
    )[0].tensor


def max_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
) -> TensorValue:
    """Applies 2D max pooling to a tensor.

    Slides a window of size ``kernel_size`` over the spatial dimensions
    and replaces each window with its maximum value.

    Args:
        input: The input tensor in channels-last (NHWC) layout,
            ``(batch_size, height, width, channels)``.
        kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height
            and width of the sliding window.
        stride: The stride of the sliding window. Either a single ``int``
            applied to both spatial dimensions, or a tuple
            ``(stride_h, stride_w)``. Defaults to ``1``.
        dilation: The spacing between kernel elements. Either a single
            ``int`` applied to both spatial dimensions, or a tuple
            ``(dilation_h, dilation_w)``. Defaults to ``1``.
        padding: Padding added to both sides of each spatial dimension.
            Out-of-bounds positions are excluded from the maximum
            (equivalently, they use the dtype's minimum value or negative
            infinity), so padding cannot win over negative input values.
            Either a single ``int`` applied to both spatial dimensions, or
            a tuple ``(pad_h, pad_w)``. Defaults to ``0``.
        ceil_mode: When ``True``, uses ceil instead of floor when
            computing the output spatial shape. Defaults to ``False``.

    Returns:
        A ``TensorValue`` representing the max-pooled values, with shape
        ``(batch_size, height_out, width_out, channels)``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op_generated(
        rmo.MaxPoolOp,
        input=input,
        filter_shape=Shape(kernel_size),
        strides=Shape(stride),
        dilations=Shape(dilation),
        paddings=Shape(_padding),
        ceil_mode=ceil_mode,
    )[0].tensor


def roi_align(
    input: TensorValueLike,
    rois: TensorValueLike,
    output_height: int,
    output_width: int,
    spatial_scale: float = 1.0,
    sampling_ratio: float = 0.0,
    aligned: bool = False,
    mode: str = "AVG",
) -> TensorValue:
    """Applies ROI-align pooling.

    Extracts fixed-size feature maps from regions of interest (ROIs) using
    bilinear interpolation.

    Args:
        input: The input tensor in channels-last (NHWC) layout,
            ``(batch_size, height, width, channels)``.
        rois: The regions of interest with shape ``(num_rois, 5)``, where each
            row is ``(batch_index, x1, y1, x2, y2)``.
        output_height: The height of each output feature map.
        output_width: The width of each output feature map.
        spatial_scale: The multiplicative factor mapping ROI coordinates to
            input spatial coordinates. Defaults to ``1.0``.
        sampling_ratio: The number of sampling points per bin in each
            direction. ``0`` means adaptive (``ceil(bin_size)``). Defaults to
            ``0.0``.
        aligned: When ``True``, applies a half-pixel offset to ROI
            coordinates for more precise alignment. Defaults to ``False``.
        mode: The pooling mode, either ``"AVG"`` or ``"MAX"``. Defaults to
            ``"AVG"``.

    Returns:
        A ``TensorValue`` representing the pooled values, with shape
        ``(num_rois, output_height, output_width, channels)``.

    Raises:
        ValueError: If ``input`` isn't rank 4, ``rois`` isn't rank 2 with
            5 columns, or ``mode`` is invalid.
    """
    input = TensorValue(input)
    rois = TensorValue(rois)

    if input.rank != 4:
        raise ValueError(
            f"roi_align expects rank-4 NHWC input, got rank {input.rank}"
        )
    if rois.rank != 2:
        raise ValueError(
            f"roi_align expects rank-2 rois [M, 5], got rank {rois.rank}"
        )
    if mode not in ("AVG", "MAX"):
        raise ValueError(f"roi_align mode must be 'AVG' or 'MAX', got '{mode}'")

    device = input.type.device
    num_rois = rois.type.shape[0]
    channels = input.type.shape[3]

    result_type = TensorType(
        input.dtype,
        [num_rois, output_height, output_width, channels],
        device,
    )

    oh_val = constant(output_height, DType.int64, device)
    ow_val = constant(output_width, DType.int64, device)
    scale_val = constant(spatial_scale, DType.float32, device)
    ratio_val = constant(sampling_ratio, DType.float32, device)

    return Graph.current._add_op_generated(
        rmo.MoRoiAlignOp,
        result_type,
        input,
        rois,
        oh_val,
        ow_val,
        scale_val,
        ratio_val,
        aligned,
        mode,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
