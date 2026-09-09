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
"""Op implementation for broadcast_to."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import kgen, rmo

from ..dim import DimLike
from ..graph import Graph
from ..shape import Shape, ShapeLike
from ..type import TensorType
from ..value import StrongTensorValueLike, TensorValue


def broadcast_to(
    x: StrongTensorValueLike,
    shape: TensorValue | ShapeLike,
    out_dims: Iterable[DimLike] | None = None,
) -> TensorValue:
    """Broadcasts a tensor to a target shape.

    Each input dimension must either equal the corresponding target
    dimension or be ``1`` (which is then stretched to match). This
    follows NumPy broadcasting semantics and is equivalent to PyTorch's
    :func:`torch.broadcast_to`.

    .. code-block:: python

        import numpy as np
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("broadcast_to", input_types=[]) as graph:
            x = ops.constant(
                np.ones((3, 1), dtype=np.float32), DType.float32, device=device
            )
            result = ops.broadcast_to(x, [3, 4])
            # result has shape (3, 4)

            # Add a new leading dimension
            result = ops.broadcast_to(x, [2, 3, 4])
            # result has shape (2, 3, 4)

    .. invisible-code-block: python

            assert result.type.shape == [2, 3, 4]

    Args:
        x: The input symbolic tensor to broadcast. Must not contain any
            dynamic dimensions.
        shape: The target shape. Either a static shape (no dynamic
            dimensions) or a :class:`~max.graph.TensorValue` giving the
            shape at runtime.
        out_dims: The explicit output dimensions. Required when ``shape``
            is a :class:`~max.graph.TensorValue` (used to declare the
            symbolic output type); ignored otherwise.

    Returns:
        A ``TensorValue`` with the same elements as the input but with the
        target shape.

    Raises:
        ValueError: If ``shape`` is a :class:`~max.graph.TensorValue` and
            ``out_dims`` is :obj:`None`.
    """
    x = TensorValue(x)

    if isinstance(shape, TensorValue):
        # For tensor-valued shapes, dims need to be declared in the graph.
        # Push the onus of doing so onto the caller.
        if out_dims is None:
            message = f"must pass out_dims with tensor value shape {shape}"
            raise ValueError(message)

        return Graph.current._add_op_generated(
            rmo.MoBroadcastToOp,
            TensorType(x.dtype, shape=out_dims, device=x.device),
            x,
            shape._mlir_value,
            kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    new_shape = Shape(shape)
    return Graph.current._add_op_generated(
        rmo.BroadcastToOp, input=x, new_shape=new_shape
    )[0].tensor
