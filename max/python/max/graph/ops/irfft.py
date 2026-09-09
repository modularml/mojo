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

"""Op implementation for irfft."""

from __future__ import annotations

from enum import Enum

from max.dtype import DType

from ..dim import Dim, StaticDim
from ..type import DeviceKind, TensorType
from ..value import StrongTensorValueLike, TensorValue
from .concat import concat
from .constant import constant
from .custom import custom
from .elementwise import sqrt
from .pad import pad
from .transpose import transpose
from .unsqueeze import unsqueeze


class Normalization(Enum):
    """The normalization mode for inverse real FFT operations."""

    BACKWARD = "backward"
    ORTHO = "ortho"
    FORWARD = "forward"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, Normalization):
            return self.value == other.value
        return False


def _process_input_signal(
    input_tensor: TensorValue, n: int, axis: int
) -> TensorValue:
    """Resizes input tensor to the required signal size."""
    axis = axis % input_tensor.rank
    axis_dim = input_tensor.shape[axis]
    if not isinstance(axis_dim, StaticDim):
        raise ValueError(f"Axis dimension must be static, got {axis_dim}.")
    output_shape = list(input_tensor.shape)
    output_shape[axis] = Dim(n)

    required_axis_dim = n // 2 + 1
    if axis_dim > required_axis_dim:
        # Slice the input tensor to the new size.
        index = [slice(None)] * input_tensor.rank
        index[axis] = slice(0, required_axis_dim)
        input_tensor = input_tensor[tuple(index)]
    elif axis_dim < required_axis_dim:
        # Pad the input tensor with zeros to the new size.
        paddings = [0] * 2 * input_tensor.rank
        paddings[axis * 2 + 1] = required_axis_dim - int(axis_dim)
        input_tensor = pad(input_tensor, paddings=paddings)
    return input_tensor


def irfft(  # noqa: ANN201
    input_tensor: StrongTensorValueLike,
    n: int | None = None,
    axis: int = -1,
    normalization: Normalization | str = Normalization.BACKWARD,
    input_is_complex: bool = False,
    buffer_size_mb: int = 512,
):
    """Compute the inverse real FFT of the input tensor.

    Args:
        input_tensor: The input tensor to compute the inverse real FFT of.
        n: The size of the output tensor. Must be an int, and cannot be a
            symbolic Buffer. The input tensor will be padded or truncated to
            `n // 2 + 1` along the specified axis.
        axis: The axis to compute the inverse real FFT of.
        normalization: The normalization to apply to the output tensor.
            Can be "backward", "ortho", or "forward". When "backward", the
            output is divided by `n`. When "ortho", the output is divided by
            `sqrt(n)`. When "forward", no normalization is applied.
        input_is_complex: Whether the input tensor is already interleaved
            complex. The last dimension of the input tensor must be 2, and is
            excluded from the dimension referred to by `axis`.
        buffer_size_mb: The estimated size of a persistent buffer to use for
            storage of intermediate results. Needs to be the same across multiple
            calls to `irfft` within the same graph. Otherwise, multiple buffers
            will be allocated.

    Returns:
        The inverse real FFT of the input tensor. The shape of the output tensor
        is the same as the shape of the input tensor, except for the axis that
        the inverse real FFT is computed over, which is replaced by `n`.
    """
    input_tensor = TensorValue(input_tensor)

    if not input_tensor.dtype == DType.float32:
        raise ValueError(
            f"Input tensor must be of type float32, got {input_tensor.dtype}."
        )
    if input_tensor.device.device_type != DeviceKind.GPU:
        raise ValueError("IRFFT is currently only supported on GPU.")
    if input_is_complex and input_tensor.shape[-1] != 2:
        raise ValueError(
            "Input tensor is marked as complex, but last dimension is not 2. "
            f"Got input shape {input_tensor.shape}."
        )

    # Exclude last dimension from the rank if input tensor is complex.
    rank = input_tensor.rank - 1 if input_is_complex else input_tensor.rank

    # Transpose the input tensor so that the axis that the inverse real FFT is
    # computed over is the last axis.
    orig_axis = axis % rank
    axis = rank - 1
    if orig_axis != axis:
        input_tensor = transpose(input_tensor, orig_axis, axis)

    # Store the new input shape, which is used for computations later.
    input_shape = list(
        input_tensor.shape[:-1] if input_is_complex else input_tensor.shape
    )

    if not n:
        n = 2 * (int(input_shape[-1]) - 1)
    input_tensor = _process_input_signal(input_tensor, n, axis=axis)

    output_shape = input_shape.copy()
    output_shape[-1] = Dim(n)

    if not input_is_complex:
        # Convert input tensor to a complex tensor.
        # Since we don't support complex dtypes, we represent the complex values
        # as interleaved real and imaginary parts.
        x_re = unsqueeze(input_tensor, -1)
        x_im = constant(
            0, input_tensor.dtype, input_tensor.device
        ).broadcast_to(x_re.shape)
        x = concat([x_re, x_im], -1)
    else:
        x = input_tensor

    x = x.reshape((*x.shape[:axis], x.shape[axis] * 2))
    irfft_out = custom(
        "irfft",
        input_tensor.device,
        [x],
        [
            TensorType(
                dtype=input_tensor.dtype,
                shape=output_shape,
                device=input_tensor.device,
            )
        ],
        {"n": n, "buffer_size_mb": buffer_size_mb},
    )[0].tensor

    if normalization == Normalization.BACKWARD:
        irfft_out /= n
    elif normalization == Normalization.ORTHO:
        irfft_out /= sqrt(constant(n, input_tensor.dtype, input_tensor.device))
    elif normalization == Normalization.FORWARD:
        pass
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    # Transpose the output tensor back to the original axis.
    if orig_axis != axis:
        irfft_out = transpose(irfft_out, axis, orig_axis)
    return irfft_out
