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
from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops


def hadamard(n: int, dtype: type[Any] = int) -> npt.NDArray[Any]:
    """
    Source: https://github.com/scipy/scipy/blob/v1.17.0/scipy/linalg/_special_matrices.py#L230-L284
    Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction. `n` must be a power of 2.

    Parameters
    ----------
    n : int
        The order of the matrix. `n` must be a power of 2.
    dtype : dtype, optional
        The data type of the array to be constructed.

    Returns
    -------
    H : (n, n) ndarray
        The Hadamard matrix.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    >>> from scipy.linalg import hadamard
    >>> hadamard(2, dtype=complex)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.+0.j, -1.-0.j]])
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])

    """

    # This function is a slightly modified version of the
    # function contributed by Ivo in ticket #675.

    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log2(n))
    if 2**lg2 != n:
        raise ValueError(
            "n must be an positive integer, and n must be a power of 2"
        )

    H = np.array([[1]], dtype=dtype)

    # Sylvester's construction
    for _i in range(lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


class HadamardTransform:
    """Hadamard transform with cached weight and scaling."""

    hadamard_weight: TensorValue | None = None

    def __init__(
        self,
        scale: float = 1.0,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ):
        self.scale = scale
        self.dtype = dtype
        self.device = device

    def _create_hadamard_weight(self, dim_padded: int) -> None:
        # Create Hadamard matrix as constant
        self.hadamard_weight = ops.constant(
            hadamard(dim_padded, dtype=float),
            device=self.device,
        ).cast(self.dtype)

    def _pad_input_dim(
        self, inputs: TensorValue, dim: int
    ) -> tuple[TensorValue, int]:
        log_dim = math.ceil(math.log2(dim))
        dim_padded = 2**log_dim
        if dim != dim_padded:
            inputs = ops.pad(
                inputs, paddings=[0, 0, 0, dim_padded - dim], value=0
            )
        return inputs, dim_padded

    def __call__(self, inputs: TensorValue) -> TensorValue:
        """Applies Hadamard transform to the input tensor.

        Applies a Hadamard transform with optional padding to the next power of 2
        if the dimension is not already a power of 2.

        Args:
            inputs: Input tensor of shape (..., dim). The last dimension will be
                used for the Hadamard transform.

        Returns:
            Transformed tensor with the same shape as the input.

        Raises:
            TypeError: If the last dimension of the input shape is not statically known.
        """
        input_shape = inputs.shape
        dim = int(
            input_shape[-1]
        )  # This raises TypeError if last dimension is not StaticDim
        # Flatten leading dimensions
        inputs = inputs.reshape((-1, dim))
        inputs, dim_padded = self._pad_input_dim(inputs, dim)

        if self.hadamard_weight is None:
            self._create_hadamard_weight(dim_padded)

        assert self.hadamard_weight is not None
        # Apply linear transformation
        result = inputs @ self.hadamard_weight.T

        # Apply scale
        scale_tensor = ops.constant(
            self.scale,
            self.dtype,
            device=self.device,
        )
        result = result * scale_tensor

        # Slice back to original dimension
        if dim != dim_padded:
            result = result[..., :dim]

        # Reshape back to original shape
        result = ops.reshape(result, input_shape)
        return result
