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
"""Op implementation for squeeze."""

from ..type import Shape
from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def squeeze(x: TensorValueLike, axis: int) -> TensorValue:
    """Removes a dimension of size ``1`` from a symbolic tensor.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("squeeze") as graph:
            # x has shape (2, 1, 3).
            x = ops.constant(
                [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
                DType.float32,
                device=device,
            )
            # Remove the size-1 dimension at axis 1, producing shape (2, 3).
            graph.output(ops.squeeze(x, axis=1))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input symbolic tensor to squeeze.
        axis: The dimension to remove from the input's shape. If negative,
            this indexes from the end of the tensor. For example, a value of
            ``-1`` removes the last dimension.

    Returns:
        A ``TensorValue`` representing ``x`` with the dimension at ``axis``
        removed. That dimension size must equal ``1``, so the result holds the
        same elements as ``x`` with one fewer dimension.

    Raises:
        ValueError: If the dimension at ``axis`` does not have size ``1``.
        IndexError: If ``axis`` is out of range, including for a rank-zero
            input.
    """
    v = TensorValue(x)
    # TODO (MSDK-655): Probably want to add rmo.mo_squeeze_shape here
    shape = Shape(v.shape)
    if shape[axis] != 1:
        raise ValueError(f"Squeeze dim must be 1, got {axis=}, {shape=}")
    shape.pop(axis)
    return reshape(v, shape)
