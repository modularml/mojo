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
"""Op implementation for fold."""

from __future__ import annotations

from ..dim import DimLike, StaticDim
from ..shape import Shape
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .custom import custom
from .shape_to_tensor import shape_to_tensor


def fold(
    input: TensorValueLike,
    output_size: tuple[DimLike, DimLike],
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> TensorValue:
    """Combines an array of sliding local blocks into a larger tensor.

    ``L``, the number of blocks, must equal ``prod((output_size[d] + 2 *
    padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) // stride[d] + 1)``,
    where ``d`` ranges over all spatial dimensions.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        # Shape (N, C * kernel_h * kernel_w, L) = (1, 1 * 2 * 2, 9).
        input_type = TensorType(DType.float32, [1, 4, 9], device=device)
        with Graph("fold", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            # Fold nine 2x2 blocks into a 4x4 image, shape (1, 1, 4, 4).
            graph.output(
                ops.fold(x, output_size=(4, 4), kernel_size=(2, 2))
            )

    Args:
        input: The 3-D tensor to fold, with shape
            ``(N, C * kernel_sizes, L)``, where ``N`` is the batch dimension,
            ``C`` is the number of channels, ``kernel_sizes`` is the product of
            the kernel sizes, and ``L`` is the number of local blocks.
        output_size: The spatial dimensions of the output tensor, as a tuple
            of two ints.
        kernel_size: The size of the sliding blocks, as a tuple of two ints.
        stride: The stride of the sliding blocks. Either an int or a tuple of
            two ints. Defaults to ``1``.
        dilation: The spacing between kernel elements. Either an int or a
            tuple of two ints. Defaults to ``1``.
        padding: The zero-padding added on both sides of the input. Either an
            int or a tuple of two ints. Defaults to ``0``.

    Returns:
        A ``TensorValue`` representing the folded 4-D tensor, with shape
        ``(N, C, output_size[0], output_size[1])``.

    Raises:
        ValueError: If the input's channel dimension isn't a multiple of the
            total kernel size, or if the number of blocks ``L`` doesn't match
            the value computed from the other arguments.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        padding = (padding, padding)

    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        channels = input.shape[1] // (kernel_size[0] * kernel_size[1])
        output_shape = Shape(
            [input.shape[0], channels, output_size[0], output_size[1]]
        )
    else:
        output_shape = Shape(
            [input.shape[0], "channels", output_size[0], output_size[1]]
        )

    # Run early shape checks if the shapes are statically known.
    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        if (
            isinstance(input.shape[1], StaticDim)
            and int(input.shape[1]) % (kernel_size[0] * kernel_size[1]) != 0
        ):
            raise ValueError(
                f"Dim 1 of the input tensor ({input.shape[1]}) must be a multiple "
                "of the product of the total kernel size"
                f" ({kernel_size[0]} * {kernel_size[1]})."
            )

        if (
            isinstance(input.shape[2], StaticDim)
            and isinstance(output_size[0], int)
            and isinstance(output_size[1], int)
        ):
            L = 1
            for n, (o, k) in enumerate(
                zip(output_size, kernel_size, strict=True)
            ):
                L_d = int(
                    (int(o) + 2 * padding[n] - dilation[n] * (int(k) - 1) - 1)  # type: ignore
                    // stride[n]
                    + 1
                )
                L *= L_d
            if int(input.shape[2]) != L:
                raise ValueError(
                    f"Last dimension of input tensor ({input.shape[2]}) must match "
                    f"the calculated number of blocks ({L})."
                )

    parameters: dict[str, int] = {
        "stride_h": stride[0],
        "stride_w": stride[1],
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
        "padding_h": padding[0],
        "padding_w": padding[1],
    }

    return custom(
        "fold",
        input.device,
        [
            input,
            shape_to_tensor(output_size),
            shape_to_tensor(kernel_size),
        ],
        [TensorType(input.dtype, output_shape, input.device)],
        parameters=parameters,
    )[0].tensor
