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
"""Op implementation for concat."""

from collections.abc import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def concat(
    original_vals: Iterable[TensorValueLike],
    axis: int = 0,
) -> TensorValue:
    """Concatenates tensors along an axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("concat_example") as graph:
            a = ops.constant([[1, 2], [3, 4]], DType.int32, device=device)
            b = ops.constant([[5, 6], [7, 8]], DType.int32, device=device)
            graph.output(
                ops.concat([a, b], axis=0),  # [[1, 2], [3, 4], [5, 6], [7, 8]]
                ops.concat([a, b], axis=1),  # [[1, 2, 5, 6], [3, 4, 7, 8]]
            )

        model = InferenceSession().load(graph)
        vertical, horizontal = model.execute()

    .. invisible-code-block: python

        import numpy as np

        assert np.array_equal(
            vertical.to_numpy(), [[1, 2], [3, 4], [5, 6], [7, 8]]
        )
        assert np.array_equal(
            horizontal.to_numpy(), [[1, 2, 5, 6], [3, 4, 7, 8]]
        )

    Args:
        original_vals: The symbolic tensors to concatenate. They must have the same rank
            and size on every dimension except ``axis``.
        axis: The axis to concatenate along. Negative values count from the end.
            Defaults to ``0``.

    Returns:
        A ``TensorValue`` representing the concatenated inputs. Its size along ``axis`` is
        the sum of the inputs' sizes and every other axis is unchanged.

    Raises:
        ValueError: If no tensors are provided, if the inputs don't all have
            the same rank, if they differ in size on any dimension other
            than ``axis``, or if they aren't all on the same device.
        IndexError: If ``axis`` is out of range.
    """
    vals = [TensorValue(v) for v in original_vals]

    if not vals:
        raise ValueError("Must provide at least one value to concat.")
    if not all(val.rank == vals[0].rank for val in vals):
        raise ValueError(f"Concat inputs must all have the same rank. {vals=}")
    if not -vals[0].rank <= axis < vals[0].rank:
        raise IndexError(f"Axis out of range {axis=}, {vals=}")
    for i, dim in enumerate(vals[0].shape):
        if i in (axis, axis + vals[0].rank):
            continue
        if not all(val.shape[i] == dim for val in vals):
            raise ValueError(
                f"Concat inputs differ on non-concat axis {i}: {vals=}"
            )
    assert_same_device(*vals)

    axis_attr = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    result = Graph.current._add_op(rmo.concat, vals, axis=axis_attr)[0].tensor

    return result
