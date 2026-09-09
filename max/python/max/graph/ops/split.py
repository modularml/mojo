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
"""Op implementation for split."""

from __future__ import annotations

from collections.abc import Sequence

from max._core.dialects import builtin, kgen, mo
from max.dtype import DType

from ..dim import Dim, DimLike
from ..graph import Graph
from ..shape import Shape
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant
from .validation import assert_valid_axis


def split(
    x: TensorValueLike, split_sizes: Sequence[DimLike], axis: int = 0
) -> list[TensorValue]:
    """Splits a tensor into several tensors along an axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("split_example") as graph:
            x = ops.constant([1, 2, 3, 4, 5], DType.int32, device=device)

            # Split into a size-2 tensor and a size-3 tensor
            parts = ops.split(x, [2, 3], axis=0)  # Splits into [1, 2] and [3, 4, 5]
            graph.output(*parts)

        model = InferenceSession().load(graph)
        first, second = model.execute()

    Args:
        x: The tensor to split.
        split_sizes: The size of each output tensor along ``axis``. Unless
            empty, must sum to the size of ``x`` along ``axis``.
        axis: The axis to split along. Must have a known size.

    Returns:
        A list of ``TensorValue`` objects, one per entry in ``split_sizes``. Each has the same
        shape as ``x`` except along ``axis``, where its size is the matching
        entry in ``split_sizes``. If ``split_sizes`` is empty, returns an empty
        list.

    Raises:
        IndexError: If ``axis`` is out of range for the rank of ``x``.
        ValueError: If ``split_sizes`` doesn't sum to the size of ``x`` along
            ``axis``, or if any size is negative.
    """
    if not split_sizes:
        return []  # op will assert on empty splits

    x = TensorValue(x)
    sizes = [int(Dim(size)) for size in split_sizes]

    assert_valid_axis(x, axis)

    if axis < 0:
        axis += x.rank

    if sum(sizes) != x.shape[axis]:
        raise ValueError(
            "Split sizes must sum to dimension value; "
            f"{x.shape[axis]=} != sum({sizes=})"
        )

    if any(size < 0 for size in sizes):
        raise ValueError(f"Split sizes must be positive: {sizes=}")

    def split_type(dim: int):  # noqa: ANN202
        shape = Shape(x.shape)
        shape[axis] = Dim(dim)
        return TensorType(x.dtype, shape, x.device)

    result_types = [split_type(size) for size in sizes]

    outputs = Graph.current._add_op_generated(
        mo.SplitOp,
        results=result_types,
        input=x,
        split_sizes=constant(sizes, DType.int64, DeviceRef.CPU()),
        axis=builtin.IntegerAttr(builtin.IndexType(), axis),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )
    return [out.tensor for out in outputs]
