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
"""Op implementation for stack."""

from collections.abc import Iterable

from ..value import TensorValue, TensorValueLike
from .concat import concat
from .unsqueeze import unsqueeze
from .utils import check_axis_in_bounds
from .validation import assert_same_device


def _axis_bounds(rank: int) -> tuple[int, int]:
    # For stack, valid axis range is [-rank+1, rank] because we're inserting a new dimension
    return -(rank + 1), rank


def _axis_out_of_range_error(
    axis: int, lower_bound: int, upper_bound: int
) -> str:
    return f"Axis out of range (expected to be in range of [{lower_bound}, {upper_bound}], but got {axis})"


def _stack_axis_bounds(rank: int) -> tuple[int, int]:
    # For stack, valid axis range is [-rank-1, rank] because we're inserting a new dimension
    return -(rank + 1), rank


def _check_stack_axis_in_bounds(axis: int, rank: int) -> None:
    lower_bound, upper_bound = _stack_axis_bounds(rank)
    if axis < lower_bound or axis > upper_bound:
        raise IndexError(
            _axis_out_of_range_error(axis, lower_bound, upper_bound)
        )


def stack(values: Iterable[TensorValueLike], axis: int = 0) -> TensorValue:
    """Stacks tensors along a new axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("stack_example") as graph:
            a = ops.constant([[1, 2], [3, 4]], DType.int32, device=device)
            b = ops.constant([[5, 6], [7, 8]], DType.int32, device=device)

            # Stack the two (2, 2) tensors into one (2, 2, 2) tensor
            graph.output(ops.stack([a, b], axis=0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        values: The tensors to stack. Each must have the same dtype, rank,
            and shape.
        axis: The position of the new axis. Negative values count from the end,
            where ``-1`` inserts the new axis as the last dimension. Defaults
            to ``0``.

    Returns:
        A ``TensorValue`` representing the stacked inputs. It has one more dimension than
        the inputs, and the new dimension has size ``len(values)``.

    Raises:
        ValueError: If no tensors are provided, if the inputs don't all have
            the same rank, if they don't all have the same dtype, or if they
            aren't all on the same device.
        IndexError: If ``axis`` is out of range for the new tensor's rank.
    """
    values_coerced = [TensorValue(v) for v in values]
    if len(values_coerced) == 0:
        raise ValueError("Expected at least one value to stack")

    rank = len(values_coerced[0].shape)
    if any(len(v.shape) != rank for v in values_coerced):
        raise ValueError("All inputs to stack must be the same rank")

    if any(v.dtype != values_coerced[0].dtype for v in values_coerced):
        raise ValueError("All inputs to stack must have the same dtype")

    assert_same_device(*values_coerced)

    # Check if axis is within bounds
    check_axis_in_bounds(axis, rank, _axis_bounds)

    unsqueezed = [unsqueeze(v, axis) for v in values_coerced]

    # Short circuit to avoid bloating graph with unneeded op.
    if len(unsqueezed) == 1:
        return unsqueezed[0]

    return concat(unsqueezed, axis=axis)
