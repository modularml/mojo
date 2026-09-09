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
"""Op implementation for repeat_interleave."""

import numpy as np
from max.dtype import DType

from ..dim import Dim, DimLike
from ..shape import Shape
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .broadcast_to import broadcast_to
from .constant import constant
from .custom import custom


def _promote_repeats(
    repeats: int | TensorValue,
    input_dim: Dim,
    out_dim: DimLike | None,
) -> tuple[TensorValue, Dim | None]:
    if out_dim is not None:
        out_dim = Dim(out_dim)

    if isinstance(repeats, TensorValue):
        if repeats.rank == 0:
            repeats = broadcast_to(repeats, [1])
        return repeats, out_dim

    if repeats <= 0:
        raise ValueError(
            f"repeats_inteleave: repeat value must be positive, given {repeats=}"
        )

    return constant(
        np.array([repeats]), DType.int64, DeviceRef.CPU()
    ), input_dim * repeats


def repeat_interleave(
    x: TensorValueLike,
    repeats: int | TensorValue,
    axis: int | None = None,
    out_dim: DimLike | None = None,
) -> TensorValue:
    """Repeats each element of a tensor along an axis.

    Modeled after :obj:`torch.repeat_interleave`.

    Given the input ``[[1.0, 2.0], [3.0, 4.0]]`` (shape ``(2, 2)``), each
    element repeats ``repeats`` times along ``axis``. Use ``axis=0`` to repeat
    rows, ``axis=1`` to repeat columns, ``axis=None`` (the default) to flatten
    first, or pass a per-element ``repeats`` tensor to repeat each index a
    different number of times:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        input_type = TensorType(DType.float32, [2, 2], device=DeviceRef.CPU())
        with Graph("repeat_interleave", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            per_element = ops.constant(
                np.array([2, 3]), DType.int64, device=DeviceRef.CPU()
            )
            graph.output(
                ops.repeat_interleave(x, repeats=2, axis=0),  # shape (4, 2)
                ops.repeat_interleave(x, repeats=2, axis=1),  # shape (2, 4)
                ops.repeat_interleave(x, repeats=2),  # flattened, shape (8,)
                # Per-element repeats: row 0 twice, row 1 three times.
                ops.repeat_interleave(x, repeats=per_element, axis=0, out_dim=5),
            )

        model = InferenceSession(devices=[CPU()]).load(graph)
        rows, cols, flat, per_row = model.execute(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        # rows:    [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]
        # cols:    [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
        # flat:    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        # per_row: [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            rows.to_numpy(), [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]
        )
        np.testing.assert_allclose(
            cols.to_numpy(), [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
        )
        np.testing.assert_allclose(
            flat.to_numpy(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        )
        np.testing.assert_allclose(
            per_row.to_numpy(),
            [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]],
        )

    Args:
        x: The input tensor.
        repeats: The number of times to repeat each element. Pass either a
            positive integer or a rank-0/rank-1 integer ``TensorValue``.
        axis: The axis to repeat along. If ``None`` (the default), the input is
            flattened first.
        out_dim: The output size along ``axis``. Required when ``repeats`` is a
            ``TensorValue``.

    Returns:
        A ``TensorValue`` representing the input with its elements interleaved.

    Raises:
        ValueError: If ``repeats`` is non-positive, if ``axis`` is out of range,
            or if the input is on a GPU device.
    """
    x = TensorValue(x)

    if x.device == DeviceRef.GPU():
        raise ValueError("repeat_interleave is not supported on GPU")

    if axis is not None and not -x.rank <= axis < x.rank:
        raise ValueError(
            f"repeat_interleave: {axis=} out of bounds for {x.rank=}"
        )

    # For compatibility with Torch, if `axis` is not passed, flatten the input array and return a flat array.
    if axis is None:
        x = x.reshape([-1])
        axis = 0

    if axis < 0:
        axis += x.rank

    repeats, inferred_size = _promote_repeats(repeats, x.shape[axis], out_dim)

    result_shape = Shape(x.shape)

    if inferred_size is None:
        raise ValueError("out_dim must be provided for TensorValue repeats")

    # Try to infer the output shape if the multiplier along the axis
    # is statically known, otherwise use the provided out_dim.
    result_shape[axis] = inferred_size

    axis_val = constant(axis, DType.int64, DeviceRef.CPU())

    output = custom(
        "repeat_interleave",
        device=x.device,
        values=[x, repeats, axis_val],
        out_types=[TensorType(x.dtype, result_shape, device=x.device)],
    )

    return output[0].tensor
