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
"""Op implementation for resize operations."""

from enum import Enum

from max._core.dialects import builtin, kgen, mo, rmo

from ..graph import Graph
from ..type import Shape, ShapeLike, TensorType
from ..value import TensorValue, TensorValueLike
from .shape_to_tensor import shape_to_tensor


class InterpolationMode(Enum):
    """Interpolation modes for image resize operations.

    This enum defines the available interpolation methods that can be used
    when resizing tensors.
    """

    NEAREST = "nearest"
    """Nearest neighbor interpolation."""
    BILINEAR = "bilinear"
    """Bilinear (linear) interpolation."""
    BICUBIC = "bicubic"
    """Bicubic interpolation."""

    def __str__(self) -> str:
        """Returns the string representation of the interpolation mode."""
        return self.value


def resize_linear(
    input: TensorValueLike,
    size: ShapeLike,
    coordinate_transform_mode: int = 0,
    antialias: bool = False,
) -> TensorValue:
    """Resizes a tensor using linear (bilinear) interpolation.

    Produces an output tensor whose shape is given by ``size`` using separable
    1-D linear filters. It resizes every dimension whose size changes,
    including batch and channel dimensions. The operation maps output
    coordinates back to input coordinates according to
    ``coordinate_transform_mode``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("resize_linear") as graph:
            # NCHW input: batch 1, 1 channel, 2x2 spatial.
            x = ops.constant(
                [[[[1.0, 2.0], [3.0, 4.0]]]], DType.float32, device=device
            )
            # Upscale the spatial dimensions to 4x4, shape (1, 1, 4, 4).
            graph.output(ops.resize_linear(x, [1, 1, 4, 4]))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to resize.
        size: The desired output shape. Must have the same rank as ``input``.
        coordinate_transform_mode: How to map an output coordinate to an input
            coordinate. Allowed values:

            - ``0`` (``half_pixel``): Default. Shifts by 0.5 before
              scaling, consistent with most deep learning frameworks.
            - ``1`` (``align_corners``): Aligns the corner pixels of the input
              and output so that the first and last coordinates are preserved
              exactly.
            - ``2`` (``asymmetric``): Applies no shift, mapping an output
              coordinate to ``out_coord / scale``.
            - ``3`` (``half_pixel_1D``): Like ``half_pixel`` on every axis,
              except any axis whose output size is 1 maps to coordinate 0.
        antialias: When ``True``, applies an antialiasing filter when
            downscaling, which reduces aliasing artifacts by widening the tent
            filter support by ``1 / scale``. Has no effect when upscaling.
            Defaults to ``False``.

    Returns:
        A ``TensorValue`` representing the resized tensor, with shape ``size``
        and the same dtype as ``input``.

    Raises:
        ValueError: If ``coordinate_transform_mode`` isn't 0-3, or if ``size``
            has a different rank than ``input``.
    """
    if coordinate_transform_mode not in (0, 1, 2, 3):
        raise ValueError(
            f"coordinate_transform_mode must be 0-3, got"
            f" {coordinate_transform_mode}"
        )

    input = TensorValue(input)
    size = Shape(size)

    if len(size) != input.rank:
        raise ValueError(
            f"size must have the same rank as input ({input.rank}), got"
            f" {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeLinearOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        mo.CoordinateTransformModeAttr(
            mo.CoordinateTransformMode(coordinate_transform_mode)
        ),
        builtin.BoolAttr(antialias),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize_nearest(
    input: TensorValueLike,
    size: ShapeLike,
    coordinate_transform_mode: int = 0,
    round_mode: int = 0,
) -> TensorValue:
    """Resizes a tensor using nearest-neighbor interpolation.

    Produces an output tensor whose dimensions are given by ``size`` by
    selecting the nearest input sample for each output coordinate.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("resize_nearest") as graph:
            # NCHW input: batch 1, 1 channel, 2x2 spatial.
            x = ops.constant(
                [[[[1.0, 2.0], [3.0, 4.0]]]], DType.float32, device=device
            )
            # Upscale the spatial dimensions to 4x4, shape (1, 1, 4, 4).
            graph.output(ops.resize_nearest(x, [1, 1, 4, 4]))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to resize.
        size: The desired output shape. Must have the same rank as ``input``.
        coordinate_transform_mode: How to map an output coordinate to an input
            coordinate. Allowed values:

            - ``0`` (``half_pixel``). Default.
            - ``1`` (``align_corners``).
            - ``2`` (``asymmetric``).
            - ``3`` (``half_pixel_1D``).

            See :func:`resize_linear` for a description of each mode.
        round_mode: How to round the mapped coordinate to select the nearest
            input sample. Allowed values:

            - ``0`` (``HalfDown``, the default): ``ceil(x - 0.5)``.
            - ``1`` (``HalfUp``): ``floor(x + 0.5)``.
            - ``2`` (``Floor``): ``floor(x)``.
            - ``3`` (``Ceil``): ``ceil(x)``.

    Returns:
        A ``TensorValue`` representing the resized tensor, with shape ``size``
        and the same dtype as ``input``.

    Raises:
        ValueError: If ``coordinate_transform_mode`` isn't 0-3, ``round_mode``
            isn't 0-3, or ``size`` has a different rank than ``input``.
    """
    if coordinate_transform_mode not in (0, 1, 2, 3):
        raise ValueError(
            f"coordinate_transform_mode must be 0-3, got"
            f" {coordinate_transform_mode}"
        )
    if round_mode not in (0, 1, 2, 3):
        raise ValueError(f"round_mode must be 0-3, got {round_mode}")

    input = TensorValue(input)
    size = Shape(size)

    if len(size) != input.rank:
        raise ValueError(
            f"size must have the same rank as input ({input.rank}), got"
            f" {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeNearestOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        mo.CoordinateTransformModeAttr(
            mo.CoordinateTransformMode(coordinate_transform_mode)
        ),
        builtin.IntegerAttr(builtin.IntegerType(64), round_mode),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize_bicubic(
    input: TensorValueLike,
    size: ShapeLike,
) -> TensorValue:
    """Resizes a tensor using bicubic interpolation.

    Produces an output tensor whose dimensions are given by ``size`` using a
    4x4-pixel Keys/PyTorch (``a = -0.75``) cubic convolution filter with
    half-pixel coordinate mapping.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("resize_bicubic") as graph:
            # NCHW input: batch 1, 1 channel, 2x2 spatial.
            x = ops.constant(
                [[[[1.0, 2.0], [3.0, 4.0]]]], DType.float32, device=device
            )
            # Upscale the spatial dimensions to 4x4, shape (1, 1, 4, 4).
            graph.output(ops.resize_bicubic(x, [1, 1, 4, 4]))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to resize. Must be rank 4 in
            channels-first (NCHW) layout,
            ``(batch_size, channels, height, width)``.
        size: The desired output shape, of length 4,
            ``(batch_size, channels, height, width)``.

    Returns:
        A ``TensorValue`` representing the resized tensor, with shape ``size``
        and the same dtype as ``input``.

    Raises:
        ValueError: If ``input`` doesn't have rank 4, or if ``size`` has a
            different length.
    """
    input = TensorValue(input)
    size = Shape(size)

    if input.rank != 4:
        raise ValueError(
            f"resize_bicubic requires rank-4 (NCHW) input, got rank"
            f" {input.rank}"
        )
    if len(size) != 4:
        raise ValueError(
            f"size must have 4 elements for NCHW format, got {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeBicubicOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize(
    input: TensorValueLike,
    shape: ShapeLike,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> TensorValue:
    """Resizes a tensor to a given shape using a specified interpolation method.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("resize") as graph:
            # NCHW input: batch 1, 1 channel, 2x2 spatial.
            x = ops.constant(
                [[[[1.0, 2.0], [3.0, 4.0]]]], DType.float32, device=device
            )
            # Upscale the spatial dimensions to 4x4, shape (1, 1, 4, 4).
            graph.output(
                ops.resize(x, [1, 1, 4, 4], ops.InterpolationMode.BILINEAR)
            )

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input tensor to resize. Must be rank 4 in channels-first
            (NCHW) layout, ``(batch_size, channels, height, width)``.
        shape: The desired output shape, of length 4, layout
            ``(batch_size, channels, height, width)``.
        interpolation: The interpolation method, given as an
            :class:`InterpolationMode`. Defaults to
            :attr:`InterpolationMode.BILINEAR`.

    Returns:
        A ``TensorValue`` representing the resized tensor with the given
        ``shape``.

    Raises:
        ValueError: If ``input`` doesn't have rank 4, or if ``shape`` has the
            wrong number of elements.
    """
    input = TensorValue(input)
    shape = Shape(shape)

    if input.rank != 4:
        raise ValueError(
            f"Input tensor must have rank 4 (NCHW format) for resize"
            f" operation, but got rank {input.rank}"
        )

    if len(shape) != 4:
        raise ValueError(
            f"shape must have 4 elements for NCHW format"
            f" (batch, channels, height, width), but got {len(shape)} elements"
        )

    if interpolation == InterpolationMode.NEAREST:
        return resize_nearest(input, shape)

    if interpolation == InterpolationMode.BILINEAR:
        return resize_linear(input, shape)

    # BICUBIC (only remaining option after NEAREST and BILINEAR).
    return resize_bicubic(input, shape)
