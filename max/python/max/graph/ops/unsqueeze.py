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

from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def unsqueeze(x: TensorValueLike, axis: int) -> TensorValue:
    """Inserts a dimension of size ``1`` into a symbolic tensor.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("unsqueeze") as graph:
            # x has shape (3,).
            x = ops.constant(
                [1.0, 2.0, 3.0],
                DType.float32,
                device=device,
            )
            # Insert a size-1 dimension at axis 0, producing shape (1, 3).
            graph.output(ops.unsqueeze(x, axis=0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input symbolic tensor to unsqueeze.
        axis: The index at which to insert a new dimension into the input's
            shape. Elements at that index or higher are shifted back. If
            negative, it indexes relative to ``1`` plus the rank of the tensor.
            For example, a value of ``-1`` adds a new dimension at the end, and
            ``-2`` inserts the dimension immediately before the last dimension.

    Returns:
        A ``TensorValue`` representing ``x`` with a new dimension inserted at
        ``axis``. That dimension has a size of ``1``, so the result holds the
        same elements as ``x`` with one more dimension.

    Raises:
        ValueError: If ``axis`` is out of bounds.
    """
    x = TensorValue(x)
    rank = x.rank
    if axis < 0:
        axis += rank + 1
    if not 0 <= axis <= rank:
        raise ValueError(f"unsqueeze axis out of bounds: {axis=}, {rank=}")

    shape = x.shape
    new_shape = shape[:axis] + [1] + shape[axis:]
    return reshape(x, new_shape)
