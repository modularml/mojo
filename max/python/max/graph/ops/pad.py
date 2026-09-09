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

"""Op implementation for pad."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
from max._core.dialects import kgen, rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, DType, Shape, TensorType
from ..value import TensorValue, TensorValueLike
from .concat import concat


def _compute_result_shape(input_shape: Shape, paddings: list[int]) -> Shape:
    assert len(paddings) == 2 * len(input_shape)

    new_shape = Shape(input_shape)
    for i, s in enumerate(new_shape):
        new_shape[i] = s + paddings[2 * i] + paddings[2 * i + 1]

    return new_shape


def pad(
    input: TensorValueLike,
    paddings: Iterable[int],
    mode: Literal["constant", "reflect", "edge"] = "constant",
    value: TensorValueLike = 0,
) -> TensorValue:
    """Pads a tensor along every dimension.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("pad_example") as graph:
            x = ops.constant([[1, 2], [3, 4]], DType.int32, device=device)

            # Add a dim before and after dim `0` and a dim before and after dim ``1``
            graph.output(ops.pad(x, [1, 1, 1, 1]))

            # [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The tensor to pad.
        paddings: The amount to pad. For a tensor of rank ``N``, pass ``2*N``
            non-negative integers in the order ``[before_dim0, after_dim0,
            before_dim1, after_dim1, ...]``.
        mode: How to fill the padded cells. Supported values:

            * ``"constant"``: fill using ``value``.
            * ``"reflect"``: reflect the content across each edge, excluding
              the boundary element (like ``numpy.pad`` with ``mode='reflect'``).
            * ``"edge"``: repeat the nearest boundary element (like
              ``numpy.pad`` with ``mode='edge'``).

            Defaults to ``"constant"``.
        value: The fill value for ``mode="constant"``. Defaults to ``0``.

    Returns:
        A ``TensorValue`` representing the padded input, with the same dtype as
        ``input``.

    Raises:
        ValueError: If ``mode`` is unsupported, or any padding value is
            negative.
        AssertionError: If the number of padding values isn't twice the input
            rank.
    """
    input = TensorValue(input)
    paddings = list(paddings)

    if mode not in ("constant", "reflect", "edge"):
        raise ValueError(
            f"unsupported padding mode {mode!r}; "
            "expected 'constant', 'reflect', or 'edge'"
        )

    if any(x < 0 for x in paddings):
        raise ValueError(
            f"padding values must be non-negative but given {paddings}"
        )

    result_type = TensorType(
        input.dtype, _compute_result_shape(input.shape, paddings), input.device
    )

    promoted_paddings = [
        dtype_promotion._promote_to_strong(
            np.array([x]), DType.int64, DeviceRef.CPU()
        )
        for x in paddings
    ]
    padding_tensor = concat(promoted_paddings, axis=0)

    if mode == "constant":
        return Graph.current._add_op_generated(
            rmo.MoPadConstantOp,
            result=result_type,
            input=input,
            paddings=padding_tensor,
            constant=dtype_promotion._promote_to_strong(
                value, input.dtype, DeviceRef.CPU()
            ),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    if mode == "reflect":
        return Graph.current._add_op_generated(
            rmo.MoPadReflectOp,
            result=result_type,
            input=input,
            paddings=padding_tensor,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    # mode == "edge"
    return Graph.current._add_op_generated(
        rmo.MoPadRepeatOp,
        result=result_type,
        input=input,
        paddings=padding_tensor,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
