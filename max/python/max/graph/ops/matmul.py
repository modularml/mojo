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
"""Op implementation for matmul."""

from max._core.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def matmul(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Computes the matrix product of two tensors.

    Use matrix multiplication to implement key building blocks like linear
    transformations, attention mechanisms, and fully connected layers. You can
    call ``matmul()`` directly or use the ``@`` operator, which calls
    ``matmul()`` implicitly.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("matmul_example") as graph:
            lhs = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], DType.float32, device=device
            )
            rhs = ops.constant(
                [[5.0, 6.0], [7.0, 8.0]], DType.float32, device=device
            )
            graph.output(ops.matmul(lhs, rhs))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(result.to_numpy(), [[19.0, 22.0], [43.0, 50.0]])

    The innermost two dimensions of each input are treated as a matrix.
    In the example above ``lhs`` has shape ``(M, K) = (2, 2)`` and ``rhs``
    has shape ``(K, N) = (2, 2)``, producing an output of shape
    ``(M, N) = (2, 2)``. The ``K`` dimensions must match. Any remaining
    outer (batch) dimensions are broadcast.

    If ``lhs`` is 1-D it is reshaped to ``1xD``, and if ``rhs`` is 1-D it is
    reshaped to ``Dx1``. In both cases, the added size-1 dimensions are
    removed from the output shape.

    Args:
        lhs: The left-hand side input tensor.
        rhs: The right-hand side input tensor.

    Returns:
        A tensor value representing the matrix product of ``lhs`` and ``rhs``.
        For 2-D inputs, the output shape is ``(M, N)`` where ``lhs`` is
        ``(M, K)`` and ``rhs`` is ``(K, N)``. For higher-dimensional inputs,
        batch dimensions are preserved and the operation is applied to the
        last two dimensions of each input.
    """
    lhs = TensorValue(lhs)
    rhs = TensorValue(rhs)
    assert_same_device(lhs=lhs, rhs=rhs)
    return Graph.current._add_op_generated(
        rmo.MatmulOp, input_x=lhs, input_y=rhs
    )[0].tensor
