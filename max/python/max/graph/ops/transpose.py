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
"""Op implementation for transpose."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant
from .utils import check_axis_in_bounds


def _axis_bounds(rank: int) -> tuple[int, int]:
    if rank == 0:
        return -1, 0
    return -rank, rank - 1


def transpose(x: TensorValueLike, axis_1: int, axis_2: int) -> TensorValue:
    """Transposes two axes of a symbolic tensor.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("transpose") as graph:
            # x has shape (2, 3).
            x = ops.constant(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                DType.float32,
                device=device,
            )
            # Swap axes 0 and 1, producing shape (3, 2).
            graph.output(ops.transpose(x, 0, 1))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input symbolic tensor to transpose.
        axis_1: One of the two axes to transpose. If negative, this indexes from the
            end of the tensor. For example, a value of ``-1`` refers to the last
            axis.
        axis_2: The other axis to transpose. If negative, this indexes from the end
            of the tensor.

    Returns:
        A ``TensorValue`` representing the input tensor with ``axis_1`` and
        ``axis_2`` transposed. It has the same elements and dtype as ``x``,
        with the order of the elements changed according to the transposition.
        For a rank-zero tensor, axes ``-1`` and ``0`` are accepted and the
        scalar is returned unchanged.

    Raises:
        IndexError: If ``axis_1`` or ``axis_2`` is out of range.
    """
    v = TensorValue(x)

    rank = len(v.shape)

    check_axis_in_bounds(axis_1, rank, _axis_bounds, "axis_1")
    check_axis_in_bounds(axis_2, rank, _axis_bounds, "axis_2")

    if axis_1 < 0:
        axis_1 += rank
    if axis_2 < 0:
        axis_2 += rank

    new_shape = v.shape
    indices = list(range(len(new_shape)))

    # Only change the shape for non-zero rank tensors.
    if rank > 0:
        new_shape[axis_1], new_shape[axis_2] = (
            new_shape[axis_2],
            new_shape[axis_1],
        )
        indices[axis_1], indices[axis_2] = axis_2, axis_1

    return Graph.current._add_op_generated(
        rmo.MoTransposeOp,
        TensorType(dtype=v.dtype, shape=new_shape, device=v.device),
        v,
        constant(indices, DType.int64, DeviceRef.CPU()),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
