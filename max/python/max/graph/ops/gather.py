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
"""Op implementation for gather."""

from max._core.dialects import builtin, kgen, rmo

from ..dim import StaticDim
from ..graph import Graph
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def gather(
    input: TensorValueLike, indices: TensorValueLike, axis: int
) -> TensorValue:
    """Selects elements out of an input tensor by index.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("gather") as graph:
            x = ops.constant(
                [[1, 2], [3, 4], [5, 6]], DType.int32, device=device
            )
            indices = ops.constant([0, 2], DType.int64, device=device)
            # Select rows 0 and 2, producing [[1, 2], [5, 6]].
            graph.output(ops.gather(x, indices, axis=0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to select elements from.
        indices: A symbolic tensor of ``int32`` or ``int64`` index values
            on the same device as ``input``.
        axis: The dimension that ``indices`` indexes into ``input``. If
            negative, indexes relative to the end of the input tensor. For
            example, ``gather(input, indices, axis=-1)`` indexes against the
            last dimension of ``input``.

    Returns:
        A ``TensorValue`` representing the selected elements. Its shape is
        ``input.shape`` with the dimension at ``axis`` replaced by
        ``indices.shape``.

    Raises:
        IndexError: If ``axis`` is out of range for ``input``.
        ValueError: If ``indices`` isn't integral or isn't on the same
            device as ``input``.
    """
    input, indices = TensorValue(input), TensorValue(indices)
    shape = input.shape

    if not -input.rank <= axis < input.rank:
        raise IndexError(f"{axis=} out of range for {input=}")
    if axis < 0:
        axis += input.rank

    output_shape = [*shape[:axis], *indices.shape, *shape[axis + 1 :]]
    assert_same_device(input=input, indices=indices)
    return Graph.current._add_op_generated(
        rmo.MoGatherOp,
        result=TensorType(input.dtype, output_shape, input.device),
        input=input,
        indices=indices,
        axis=builtin.IntegerAttr(builtin.IndexType(), axis),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def gather_nd(
    input: TensorValueLike,
    indices: TensorValueLike,
    batch_dims: int = 0,
) -> TensorValue:
    """Selects elements from a tensor by N-dimensional index.

    Unlike :func:`gather()`, which indexes along a single axis,
    ``gather_nd()`` indexes along multiple dimensions at once. The last
    dimension of ``indices`` is the index vector: its values select
    elements from ``input`` immediately after any ``batch_dims`` leading
    dimensions. Any remaining trailing dimensions of ``input`` are sliced
    into the output as features.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_shape = ["a", "b", "c", "d", "e"]
        indices_shape = ["a", "f", 3]
        input_type = TensorType(DType.bfloat16, input_shape, device=device)
        indices_type = TensorType(DType.int32, indices_shape, device=device)
        with Graph("gather_nd", input_types=[input_type, indices_type]) as graph:
            input, indices = graph.inputs
            gathered = ops.gather_nd(input, indices, batch_dims=1)
            # gathered.type has shape ["a", "f", "e"]

    .. invisible-code-block: python

        assert gathered.type.shape == ["a", "f", "e"]
        assert gathered.type.dtype == DType.bfloat16

    In this example:

    - ``batch_dims`` is 1, so ``input`` and ``indices`` share the leading
      "a" dimension.
    - ``indices`` has an additional dimension "f" which becomes part of
      the output.
    - The last dimension of ``indices`` (size 3) is the index vector;
      each value selects into "b", "c", and "d" of ``input``.
    - Since ``batch_dims (1) + index size (3) < input.rank (5)``, the
      remaining dimension "e" is sliced into the output.

    Args:
        input: The input symbolic tensor to gather from.
        indices: An integer tensor of multi-dimensional indices. Its last
            dimension must be static and gives the size of the index
            vector.
        batch_dims: The number of leading batch dimensions shared by
            ``input`` and ``indices``. The shapes must match exactly along
            these leading dimensions. This function does not broadcast.
            Defaults to ``0``.

    Returns:
        A ``TensorValue`` representing the gathered elements, with the same
        dtype as ``input``. Its shape is the concatenation of:

        - ``input.shape[:batch_dims]`` — the leading batch dimensions.
        - ``indices.shape[batch_dims:-1]`` — the gather dimensions.
        - ``input.shape[batch_dims + indices.shape[-1]:]`` — the
          trailing sliced dimensions.

    Raises:
        ValueError: If any input is invalid. This includes when ``indices``'s
            last dimension is not static, ``indices`` is not an integer tensor,
            ``batch_dims`` is negative or greater than ``indices.rank - 1``,
            ``batch_dims + indices.shape[-1]`` exceeds ``input.rank``, or the
            leading ``batch_dims`` of ``input`` and ``indices`` don't match,
            or the inputs are on different devices.
    """
    input, indices = TensorValue(input), TensorValue(indices)
    assert_same_device(input=input, indices=indices)

    if not isinstance(indices.shape[-1], StaticDim):
        raise ValueError(f"index last dimension must be static: {indices=}")
    index_size = int(indices.shape[-1])

    if batch_dims < 0:
        raise ValueError(f"batch_dims must be non-negative: {batch_dims=}")
    if batch_dims > indices.rank - 1:
        raise ValueError(f"Not enough dims in {indices=} for {batch_dims=}")
    if batch_dims + index_size > input.rank:
        raise ValueError(
            f"Not enough dims in {input=}: {batch_dims=},"
            f" {index_size=} ({indices=})"
        )

    if input.shape[:batch_dims] != indices.shape[:batch_dims]:
        raise ValueError(
            f"{input=} and {indices=} must match up to {batch_dims=}"
        )

    if indices.dtype.is_float():
        raise ValueError(f"{indices.dtype=} must be an integer type.")

    output_shape = [
        *input.shape[:batch_dims],
        *indices.shape[batch_dims:-1],
        *input.shape[batch_dims + index_size :],
    ]

    return Graph.current._add_op_generated(
        rmo.MoGatherNdOp,
        result=TensorType(input.dtype, output_shape, input.device),
        input=input,
        indices=indices,
        batch_dims=builtin.IntegerAttr(builtin.IndexType(), batch_dims),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
