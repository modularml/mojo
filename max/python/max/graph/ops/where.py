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
"""Op implementation for where."""

from max._core.dialects import rmo
from max.dtype import DType

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def where(
    condition: TensorValueLike, x: TensorValueLike, y: TensorValueLike
) -> TensorValue:
    """Selects elements from ``x`` or ``y`` element-wise based on a condition.

    At each position, takes the element from ``x`` where ``condition`` is true
    and the element from ``y`` where it's false. The inputs are broadcast to a
    common shape. Either ``x`` or ``y`` can be a Python scalar, which is
    promoted to a tensor using the other operand's dtype and device. At least
    one of ``x`` and ``y`` must be a tensor.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("where") as graph:
            condition = ops.constant(
                [True, False, True], DType.bool, device=device
            )
            x = ops.constant([1, 2, 3], DType.int32, device=device)
            y = ops.constant([10, 20, 30], DType.int32, device=device)
            # Take x where True and y where False, producing [1, 20, 3].
            graph.output(ops.where(condition, x, y))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        condition: The boolean tensor selecting which input to take at each
            position. Must have a boolean dtype.
        x: The tensor to select from where ``condition`` is true.
        y: The tensor to select from where ``condition`` is false.

    Returns:
        A ``TensorValue`` representing the element-wise selection from ``x`` and
        ``y`` according to ``condition``.

    Raises:
        ValueError: If ``condition`` doesn't have a boolean dtype, if
            ``condition``, ``x``, and ``y`` aren't all on the same device, or
            if a Python scalar operand can't be safely promoted to the other
            operand's dtype.
        TypeError: If neither ``x`` nor ``y`` is a tensor.
        Error: If the shapes of ``condition``, ``x``, and ``y`` can't be
            broadcast to a common shape.
    """
    condition = TensorValue(condition)
    if condition.dtype != DType.bool:
        raise ValueError(
            f"Expected condition to be a boolean tensor, but got a tensor with dtype {condition.dtype}"
        )

    x, y = dtype_promotion._promote_weak_dtypes(x, y)
    assert_same_device(condition=condition, x=x, y=y)
    return Graph.current._add_op_generated(
        rmo.SelectOp, cond=condition, input_x=x, input_y=y
    )[0].tensor
