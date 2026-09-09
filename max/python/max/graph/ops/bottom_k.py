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
"""Op implementation for bottom_k."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import Dim
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def bottom_k(
    input: TensorValueLike, k: int, axis: int = -1
) -> tuple[TensorValue, TensorValue]:
    """Returns the ``k`` smallest values along an axis with their indices.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("bottom_k") as graph:
            x = ops.constant(
                [1.0, 3.0, 2.0, 5.0, 4.0], DType.float32, device=device
            )
            # The two smallest values are 1 and 2 at indices 0 and 2.
            values, indices = ops.bottom_k(x, k=2, axis=-1)
            graph.output(values, indices)

        model = InferenceSession().load(graph)
        values, indices = model.execute()

    .. note::

        On GPU, only the last axis is supported.

    Args:
        input: The input tensor from which to select the bottom ``k``.
        k: The number of values to select from ``input``. Must be in the range
            ``[0, input.shape[axis]]``.
        axis: The axis along which to select the bottom ``k``. Defaults to
            ``-1``.

    Returns:
        A tuple of two ``TensorValue`` objects. The first holds the bottom ``k``
        values along ``axis`` in ascending order, and the second holds their
        ``int64`` indices in ``input``. Both have the shape of ``input`` with
        the ``axis`` dimension set to ``k``.
    """
    input_tv = TensorValue(input)
    ndim = len(input_tv.shape)
    norm_axis = axis % ndim

    out_shape = list(input_tv.shape)
    out_shape[norm_axis] = Dim(k)

    values_type = TensorType(
        dtype=input_tv.dtype, shape=out_shape, device=input_tv.device
    )
    indices_type = TensorType(
        dtype=DType.int64, shape=out_shape, device=input_tv.device
    )

    k_val = constant(k, dtype=DType.int64, device=DeviceRef.CPU())
    axis_val = constant(norm_axis, dtype=DType.int64, device=DeviceRef.CPU())
    sorted_val = constant(True, dtype=DType.bool, device=DeviceRef.CPU())

    vals, idxs = Graph.current._add_op_generated(
        rmo.MoBottomKOp,
        values_type,
        indices_type,
        input_tv,
        k_val,
        axis_val,
        sorted_val,
        kgen.ParamDeclArrayAttr([]),
    )

    return vals.tensor, idxs.tensor
