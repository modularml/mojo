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
"""Op implementation for reshape."""

import operator
from collections.abc import Iterable
from functools import reduce

from max._core.dialects import rmo

from ..dim import Dim
from ..graph import Graph
from ..type import Shape, ShapeLike
from ..value import TensorValue, TensorValueLike


def _product(dims: Iterable[Dim]) -> Dim:
    # 1 is the multiplicative identity.
    return reduce(operator.mul, dims, Dim(1))


def reshape(x: TensorValueLike, shape: ShapeLike) -> TensorValue:
    """Reshapes a symbolic tensor.

    If a value of ``-1`` is present in ``shape``, that dimension becomes an
    automatically calculated dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor divided by
    the product of the other dimensions of ``shape``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("reshape") as graph:
            # x has shape (2, 3).
            x = ops.constant(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                DType.float32,
                device=device,
            )
            # Reshape the same 6 elements into shape (3, 2).
            graph.output(ops.reshape(x, [3, 2]))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input symbolic tensor to reshape.
        shape: The new shape as an iterable of dimensions (for example, a
            list, tuple, or ``Dim`` objects). A single dimension may be
            ``-1``.

    Returns:
        A ``TensorValue`` representing ``x`` with a new ``shape``. The order and
        total number of elements stays the same as the input.

    Raises:
        ValueError: If ``shape`` contains more than one ``-1`` dimension, if a
            ``-1`` dimension is requested while another dimension is ``0``, or
            if the input and target shapes have a different number of elements.
    """
    x = TensorValue(x)
    shape = Shape(shape)

    # Find the single -1 dimension (if any).
    if (has_negative := shape.count(Dim(-1))) > 1:
        raise ValueError("reshape(): at most one -1 dimension is allowed")

    if has_negative:
        # Disallow inferring -1 if another requested dim is 0.
        if 0 in shape:
            raise ValueError(
                "reshape(): cannot infer -1 dimension when another dimension is 0"
            )

        total = _product(x.shape)
        known = _product(d for d in shape if d != -1)
        # missing = total // known  (symbolic; folds when possible)
        shape[shape.index(Dim(-1))] = total // known

    return Graph.current._add_op_generated(
        rmo.ReshapeOp, input=x, new_shape=shape
    )[0].tensor
