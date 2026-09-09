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
"""Op implementation for range."""

from __future__ import annotations

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ...driver import Device
from .. import dtype_promotion
from ..dim import Dim, DimLike
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike, _is_scalar
from .cast import cast


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike = 1,
    out_dim: DimLike | None = None,
    *,
    dtype: DType,
    device: Device | DeviceRef,
) -> TensorValue:
    """Creates a sequence of evenly spaced values from ``start`` to ``stop``.

    The sequence begins at ``start`` and increments by ``step``, stopping
    before ``stop`` (the upper bound is exclusive).

    ``stop - start`` must be zero or have the same sign as ``step``.
    Also, graph compilation fails when ``stop - start`` isn't evenly
    divisible by ``step``. For example, ``range(0, 5, 2)`` should produce
    three values, ``[0, 2, 4]``, but shape inference declares an output length
    of 2. The generated values therefore don't fit the declared output shape.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("range_example") as graph:
            graph.output(ops.range(0, 5, 1, dtype=DType.float32, device=device))

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result holds [0.0, 1.0, 2.0, 3.0, 4.0].

    Args:
        start: The first value in the sequence. Must be a scalar value.
        stop: The exclusive upper bound. The sequence stops before this value. Must be a scalar value.
        step: The spacing between consecutive values. Must be non-zero. Defaults to ``1``.
        out_dim: The expected length of the output. Required when dynamic
            scalar :class:`~max.graph.TensorValue` inputs prevent static length
            inference. When omitted, it's computed from scalar literals.
        dtype: The element type of the result tensor.
        device: The device the result tensor lives on.

    Returns:
        A ``TensorValue`` representing the generated sequence.

    Raises:
        ValueError: If ``out_dim`` is omitted for dynamic scalar inputs, if
            any input isn't scalar, or if any input isn't on the CPU.
        RuntimeError: During graph compilation if a statically known interval
            isn't evenly divisible by ``step``, causing the inferred output
            length to disagree with the number of generated values.
    """
    device = DeviceRef.from_device(device)

    if out_dim is None:
        if not (_is_scalar(start) and _is_scalar(stop) and _is_scalar(step)):
            raise ValueError("Dynamic ranges must provide an explicit out_dim")
        # - Most combinations of scalars will work fine
        # - Specifically mixing float and Dim doesn't work today (but could)
        # - Telling mypy about this case specifically is hard, hence the ignore
        out_dim = (stop - start) // step if step != 0 else 0  # type: ignore
        if isinstance(out_dim, float):
            out_dim = int(out_dim)
        assert out_dim is not None
        out_dim = Dim(out_dim)

    def to_dtype(value: TensorValueLike) -> TensorValue:
        value = dtype_promotion._promote_to_strong(
            value, dtype, DeviceRef.CPU()
        )
        if value.dtype != dtype:
            value = cast(value, dtype)
        return value

    start = to_dtype(start)
    stop = to_dtype(stop)
    step = to_dtype(step)
    assert start.dtype == stop.dtype == step.dtype

    if not start.rank == stop.rank == step.rank == 0:
        raise ValueError("range expected scalar values as inputs!")
    if not start.device == stop.device == step.device == DeviceRef.CPU():
        raise ValueError("Range input values must be on CPU")

    return Graph.current._add_op_generated(
        rmo.MoRangeOp,
        TensorType(dtype, shape=[out_dim], device=device).to_mlir(),
        start,
        stop,
        step,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
