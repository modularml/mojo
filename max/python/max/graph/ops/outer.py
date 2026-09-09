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
"""Op implementation for outer."""

from .. import dtype_promotion
from ..value import TensorValue, TensorValueLike
from .reshape import reshape
from .validation import assert_same_device


def outer(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Computes the outer product of two symbolic vectors.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("outer") as graph:
            lhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
            rhs = ops.constant([4.0, 5.0], DType.float32, device=device)
            # Outer product, producing [[4, 5], [8, 10], [12, 15]].
            graph.output(ops.outer(lhs, rhs))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        lhs: The left side of the product. Must be rank 1.
        rhs: The right side of the product. Must be rank 1.

    Returns:
        A ``TensorValue`` representing the
        `outer product <https://en.wikipedia.org/wiki/Outer_product>`_ of the
        two input vectors. It has rank 2, with dimension sizes equal to the
        number of elements of ``lhs`` and ``rhs`` respectively.

    Raises:
        ValueError: If ``lhs`` or ``rhs`` is not rank 1.
    """
    lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)
    if lhs.rank != 1 or rhs.rank != 1:
        raise ValueError("outer expected 1d-tensors as inputs")
    assert_same_device(lhs=lhs, rhs=rhs)
    return reshape(lhs, [-1, 1]) * reshape(rhs, [1, -1])
