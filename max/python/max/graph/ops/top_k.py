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
"""Op implementation for top_k."""

from max._core.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def top_k(
    input: TensorValueLike, k: int, axis: int = -1
) -> tuple[TensorValue, TensorValue]:
    """Returns the ``k`` largest values along an axis with their indices.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("top_k") as graph:
            x = ops.constant(
                [1.0, 3.0, 2.0, 5.0, 4.0], DType.float32, device=device
            )
            # The two largest values are 5 and 4 at indices 3 and 4.
            values, indices = ops.top_k(x, k=2, axis=-1)
            graph.output(values, indices)

        model = InferenceSession().load(graph)
        values, indices = model.execute()

    .. note::

        On GPU, only the last axis is supported.

    Args:
        input: The input tensor from which to select the top ``k``.
        k: The number of values to select from ``input``. Must be in the range
            ``[0, input.shape[axis]]``.
        axis: The axis along which to select the top ``k``. Defaults to ``-1``.

    Returns:
        A tuple of two ``TensorValue`` objects. The first holds the top ``k``
        values along ``axis``, and the second holds their ``int64`` indices in
        ``input``. Both have the shape of ``input`` with the ``axis`` dimension
        set to ``k``.
    """
    topk_weight, topk_idx = Graph.current._add_op_generated(
        rmo.TopKOp, input=TensorValue(input), k=k, axis=axis
    )

    return topk_weight.tensor, topk_idx.tensor
