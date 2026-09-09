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
"""Op implementation for tile."""

from collections.abc import Iterable

from max._core.dialects import kgen, rmo

from .. import dtype_promotion
from ..dim import Dim, DimLike, StaticDim
from ..graph import Graph
from ..shape import Shape
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .transfer_to import transfer_to
from .validation import _check_device_placement


def tile(x: TensorValueLike, repeats: Iterable[DimLike]) -> TensorValue:
    """Repeats a tensor along each of its dimensions.

    Each dimension ``i`` is copied ``repeats[i]`` times, so its output size is
    ``x.shape[i] * repeats[i]``.

    This op runs on CPU. By default, an input on another device is copied to
    CPU for the operation (emitting a warning) and the result is copied back.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("tile_example") as graph:
            x = ops.constant([[1, 2], [3, 4]], DType.int32, device=device)

            # Repeat the columns twice, leaving the rows unchanged.
            graph.output(ops.tile(x, [1, 2]))  # [[1, 2, 1, 2], [3, 4, 3, 4]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The tensor to tile.
        repeats: The number of copies for each dimension, one positive value per
            dimension of ``x``.

    Returns:
        A ``TensorValue`` representing the tiled input.

    Raises:
        ValueError: If ``repeats`` doesn't have one value per dimension, if a
            statically-known repeat isn't positive, or if ``x`` is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
    """
    x = dtype_promotion._restrict_to_strong_dtypes(x)
    shape = x.shape

    repeats = [Dim(d) for d in repeats]
    if len(shape) != len(repeats):
        raise ValueError(
            "Input rank and number of elements in repeats must match:"
            f" {shape=}, {repeats=}"
        )

    if any(count.dim <= 0 for count in repeats if isinstance(count, StaticDim)):
        raise ValueError(f"Repeats must all be positive: {repeats=}")

    output_dims = [
        dim * count for dim, count in zip(shape, repeats, strict=True)
    ]

    old_device = x.device if not x.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.tile", "TODO(GEX-2056).")
        x = transfer_to(x, DeviceRef.CPU())
    # TODO(GEX-2056): Add GPU kernel support for tile.
    result = Graph.current._add_op_generated(
        rmo.MoTileOp,
        TensorType(dtype=x.dtype, shape=output_dims, device=x.device),
        x,
        TensorValue(Shape(repeats)),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result
