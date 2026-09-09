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
"""Op implementation for nonzero."""

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import DimLike
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .transfer_to import transfer_to
from .validation import _check_device_placement


def nonzero(x: TensorValueLike, out_dim: DimLike) -> TensorValue:
    """Returns the indices of all nonzero elements of a tensor.

    Each row is the multi-index of one nonzero element,
    and the rows are generated in row-major order.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("nonzero") as graph:
            x = ops.constant([[0, 1], [2, 0]], DType.int32, device=device)
            # Nonzero elements at (0, 1) and (1, 0) produce [[0, 1], [1, 0]]
            graph.output(ops.nonzero(x, out_dim="nonzero"))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input symbolic tensor.
        out_dim: The new data-dependent dimension for the number of nonzero
            elements.

    Returns:
        A ``TensorValue`` representing the indices of the nonzero elements of
        ``x``, with shape ``[out_dim, x.rank]`` and ``int64`` dtype.

    Raises:
        ValueError: If ``x`` is scalar, or if ``x`` is on a non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
    """
    x = TensorValue(x)

    if x.rank == 0:
        raise ValueError("Scalar inputs not supported")

    old_device = x.device if not x.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.nonzero", "TODO(GEX-2041).")
        x = transfer_to(x, DeviceRef.CPU())
    # TODO(GEX-2041): Add GPU kernel support for nonzero.
    result = Graph.current._add_op_generated(
        rmo.MoArgNonzeroOp,
        TensorType(dtype=DType.int64, shape=[out_dim, x.rank], device=x.device),
        TensorValue(x),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result
