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
"""Op implementation for scatter."""

from max._core.dialects import builtin, kgen, rmo
from max.dtype import DType

from .. import dtype_promotion
from ..dim import DimLike
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant
from .nonzero import nonzero
from .transfer_to import transfer_to
from .validation import _check_device_placement, assert_same_device


def scatter(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Writes ``updates`` into a copy of ``input`` at positions given by ``indices``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("scatter") as graph:
            x = ops.constant([1, 2, 3, 4, 5], DType.int32, device=device)
            updates = ops.constant([10, 20], DType.int32, device=device)
            indices = ops.constant([0, 3], DType.int64, device=device)
            # Overwrite positions 0 and 3, producing [10, 2, 3, 20, 5].
            graph.output(ops.scatter(x, updates, indices, axis=0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to ``input``.
        indices: The positions in ``input`` to update.
        axis: The axis along which ``indices`` indexes. Defaults to ``-1``.

    Returns:
        A ``TensorValue`` representing ``input`` with ``updates`` written at
        ``indices``. It has the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, if ``input``, ``updates``,
            and ``indices`` aren't on the same device, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
        Error: If ``input``, ``updates``, and ``indices`` don't have equal
            rank, if ``updates.shape`` and ``indices.shape`` differ, or if any
            ``indices`` dimension exceeds the corresponding ``input``
            dimension.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range [-{input.rank},"
            f" {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type"
            " int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    # TODO(GEX-2197): Add GPU kernel support for scatter.
    result = Graph.current._add_op_generated(
        rmo.MoScatterOp,
        result=input.type,
        input=input,
        updates=updates,
        indices=indices,
        axis=builtin.IntegerAttr(builtin.IndexType(), axis),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def scatter_nd(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Scatters slices from ``updates`` into a copy of ``input`` at N-dimensional indices.

    The last dimension of ``indices`` is the index vector. Its values select a
    slice (or scalar) in ``input``. When the index vector length ``k`` is less
    than ``input.rank``, each update writes a whole slice of the trailing
    ``input.rank - k`` dimensions.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("scatter_nd") as graph:
            x = ops.constant(
                [[1, 2], [3, 4], [5, 6]], DType.int32, device=device
            )
            updates = ops.constant(
                [[10, 20], [50, 60]], DType.int32, device=device
            )
            indices = ops.constant([[0], [2]], DType.int64, device=device)
            # Overwrite rows 0 and 2, producing [[10, 20], [3, 4], [50, 60]].
            graph.output(ops.scatter_nd(x, updates, indices))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to ``input``. Its shape
            is ``[*Q, *input.shape[K:]]``, matching the batch dimensions of
            ``indices`` and the trailing ``input`` dimensions that ``K``
            doesn't index.
        indices: A symbolic tensor of indices specifying where to write
            ``updates``. Its shape is ``[*Q, K]`` for any number of leading
            batch dimensions ``Q`` and an index-vector length ``K`` in the last
            dimension, where ``K <= input.rank``.

    Returns:
        A ``TensorValue`` representing ``input`` with ``updates`` scattered in.
        It has the same shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, if
            ``indices`` dtype is not int32/int64, or if ``input``, ``updates``,
            and ``indices`` aren't on the same device.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type"
            " int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_nd_add(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor by accumulating updates into input at N-D indices.

    Produces an output tensor by scattering slices from updates into a copy
    of input according to N-dimensional index vectors, summing values at
    duplicate index positions. Each index vector is the last dimension of
    ``indices`` and selects a slice (or scalar) in input, so for an
    ``input`` of shape ``(4, 2)`` and ``indices`` of shape ``(3, 1)`` each
    row ``i`` adds ``updates[i, :]`` to ``output[indices[i, 0], :]``. When
    two index vectors point at the same row, both updates accumulate there.

    The following example writes three row updates, two of which target
    row ``0``, so that row receives the sum of both updates:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [4, 2], device=device)
        indices_type = TensorType(DType.int64, [3, 1], device=device)
        with Graph(
            "scatter_nd_add", input_types=[input_type, indices_type]
        ) as graph:
            x = graph.inputs[0].tensor
            indices = graph.inputs[1].tensor
            updates = ops.constant(
                np.array([[10.0, 10.0], [20.0, 20.0], [100.0, 100.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            graph.output(ops.scatter_nd_add(x, updates, indices))

        model = InferenceSession(devices=[CPU()]).load(graph)
        result = model.execute(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                     dtype=np.float32),
            np.array([[0], [2], [0]], dtype=np.int64),
        )[0]
        # result: [[111.0, 111.0], [2.0, 2.0], [23.0, 23.0], [4.0, 4.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            result.to_numpy(),
            [[111.0, 111.0], [2.0, 2.0], [23.0, 23.0], [4.0, 4.0]],
        )

    Args:
        input: The input symbolic tensor to accumulate into.
        updates: A symbolic tensor of values to add.
        indices: An index tensor whose last dimension is the index vector
            length ``k`` (``k <= input.rank``).

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, if
            ``indices`` dtype is not int32/int64, or if ``input``, ``updates``,
            and ``indices`` aren't on the same device.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdAddOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_nd_max(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the maximum of updates into input at N-D indices.

    Produces an output tensor by scattering slices from updates into a copy
    of input according to N-dimensional index vectors, keeping the maximum
    at duplicate index positions. Each index vector is the last dimension of
    ``indices`` and selects a slice (or scalar) in input, so for an
    ``input`` of shape ``(4, 2)`` and ``indices`` of shape ``(3, 1)`` each
    row ``i`` sets ``output[indices[i, 0], :]`` to the element-wise maximum
    of its current value and ``updates[i, :]``. When two index vectors point
    at the same row, that row keeps the maximum over both updates.

    The following example writes three row updates, two of which target
    row ``0``, so that row keeps the larger of the two updates:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [4, 2], device=device)
        indices_type = TensorType(DType.int64, [3, 1], device=device)
        with Graph(
            "scatter_nd_max", input_types=[input_type, indices_type]
        ) as graph:
            x = graph.inputs[0].tensor
            indices = graph.inputs[1].tensor
            updates = ops.constant(
                np.array([[10.0, 10.0], [20.0, 20.0], [100.0, 100.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            graph.output(ops.scatter_nd_max(x, updates, indices))

        model = InferenceSession(devices=[CPU()]).load(graph)
        result = model.execute(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                     dtype=np.float32),
            np.array([[0], [2], [0]], dtype=np.int64),
        )[0]
        # result: [[100.0, 100.0], [2.0, 2.0], [20.0, 20.0], [4.0, 4.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            result.to_numpy(),
            [[100.0, 100.0], [2.0, 2.0], [20.0, 20.0], [4.0, 4.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to compare.
        indices: An index tensor whose last dimension is the index vector
            length ``k`` (``k <= input.rank``).

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, if
            ``indices`` dtype is not int32/int64, or if ``input``, ``updates``,
            and ``indices`` aren't on the same device.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdMaxOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_nd_min(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the minimum of updates into input at N-D indices.

    Produces an output tensor by scattering slices from updates into a copy
    of input according to N-dimensional index vectors, keeping the minimum
    at duplicate index positions. Each index vector is the last dimension of
    ``indices`` and selects a slice (or scalar) in input, so for an
    ``input`` of shape ``(4, 2)`` and ``indices`` of shape ``(3, 1)`` each
    row ``i`` sets ``output[indices[i, 0], :]`` to the element-wise minimum
    of its current value and ``updates[i, :]``. When two index vectors point
    at the same row, that row keeps the minimum over both updates.

    The following example writes updates whose values all exceed the input,
    so every targeted row keeps its original smaller value:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [4, 2], device=device)
        indices_type = TensorType(DType.int64, [3, 1], device=device)
        with Graph(
            "scatter_nd_min", input_types=[input_type, indices_type]
        ) as graph:
            x = graph.inputs[0].tensor
            indices = graph.inputs[1].tensor
            updates = ops.constant(
                np.array([[10.0, 10.0], [20.0, 20.0], [100.0, 100.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            graph.output(ops.scatter_nd_min(x, updates, indices))

        model = InferenceSession(devices=[CPU()]).load(graph)
        result = model.execute(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                     dtype=np.float32),
            np.array([[0], [2], [0]], dtype=np.int64),
        )[0]
        # result: [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            result.to_numpy(),
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to compare.
        indices: An index tensor whose last dimension is the index vector
            length ``k`` (``k <= input.rank``).

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, if
            ``indices`` dtype is not int32/int64, or if ``input``, ``updates``,
            and ``indices`` aren't on the same device.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdMinOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_nd_mul(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the product of updates into input at N-D indices.

    Produces an output tensor by scattering slices from updates into a copy
    of input according to N-dimensional index vectors, multiplying values
    at duplicate index positions. Each index vector is the last dimension of
    ``indices`` and selects a slice (or scalar) in input, so for an
    ``input`` of shape ``(4, 2)`` and ``indices`` of shape ``(3, 1)`` each
    row ``i`` multiplies ``output[indices[i, 0], :]`` by ``updates[i, :]``.
    When two index vectors point at the same row, both factors multiply into
    that row.

    The following example writes three row updates, two of which target
    row ``0``, so that row is multiplied by both factors:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [4, 2], device=device)
        indices_type = TensorType(DType.int64, [3, 1], device=device)
        with Graph(
            "scatter_nd_mul", input_types=[input_type, indices_type]
        ) as graph:
            x = graph.inputs[0].tensor
            indices = graph.inputs[1].tensor
            updates = ops.constant(
                np.array([[10.0, 10.0], [20.0, 20.0], [100.0, 100.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            graph.output(ops.scatter_nd_mul(x, updates, indices))

        model = InferenceSession(devices=[CPU()]).load(graph)
        result = model.execute(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                     dtype=np.float32),
            np.array([[0], [2], [0]], dtype=np.int64),
        )[0]
        # result: [[1000.0, 1000.0], [2.0, 2.0], [60.0, 60.0], [4.0, 4.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            result.to_numpy(),
            [[1000.0, 1000.0], [2.0, 2.0], [60.0, 60.0], [4.0, 4.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to multiply.
        indices: An index tensor whose last dimension is the index vector
            length ``k`` (``k <= input.rank``).

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, if
            ``indices`` dtype is not int32/int64, or if ``input``, ``updates``,
            and ``indices`` aren't on the same device.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return Graph.current._add_op_generated(
        rmo.MoScatterNdMulOp,
        input.type,
        input,
        updates,
        indices,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def scatter_add(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor by accumulating updates into input at indices.

    Produces an output tensor by scattering elements from updates into input
    according to indices, summing values at duplicate indices. For a 2-D
    input with ``axis=0``, each element adds as ``output[indices[i][j]][j]
    += updates[i][j]``; with ``axis=1`` the index selects the column, so
    ``output[i][indices[i][j]] += updates[i][j]``.

    The following example adds the same updates along each axis of a
    zero-filled input, where index collisions accumulate:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [3, 3], device=device)
        with Graph("scatter_add", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            updates = ops.constant(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            indices = ops.constant(
                np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int64),
                DType.int64,
                device=device,
            )
            graph.output(
                ops.scatter_add(x, updates, indices, axis=0),
                ops.scatter_add(x, updates, indices, axis=1),
            )

        model = InferenceSession(devices=[CPU()]).load(graph)
        along_rows, along_cols = model.execute(np.zeros((3, 3), np.float32))
        # along_rows: [[5.0, 0.0, 6.0], [0.0, 7.0, 0.0], [0.0, 0.0, 3.0]]
        # along_cols: [[1.0, 2.0, 3.0], [10.0, 5.0, 0.0], [0.0, 0.0, 0.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            along_rows.to_numpy(),
            [[5.0, 0.0, 6.0], [0.0, 7.0, 0.0], [0.0, 0.0, 3.0]],
        )
        np.testing.assert_allclose(
            along_cols.to_numpy(),
            [[1.0, 2.0, 3.0], [10.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        )

    Args:
        input: The input symbolic tensor to accumulate into.
        updates: A symbolic tensor of values to add.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, if ``input``, ``updates``,
            and ``indices`` aren't on the same device, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
        Error: If ``input``, ``updates``, and ``indices`` don't have equal
            rank, if ``updates.shape`` and ``indices.shape`` differ, or if any
            ``indices`` dimension exceeds the corresponding ``input``
            dimension.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range"
            f" [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter_add", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    # TODO(GEX-2197): Add GPU kernel support for scatter_add.
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterAddOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def scatter_max(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the maximum of updates into input.

    Produces an output tensor by scattering elements from updates into input
    according to indices, keeping the maximum at duplicate indices. For a 2-D
    input with ``axis=0``, each element updates as ``output[indices[i][j]][j]
    = max(output[indices[i][j]][j], updates[i][j])``; with ``axis=1`` the
    index selects the column, so ``output[i][indices[i][j]] =
    max(output[i][indices[i][j]], updates[i][j])``.

    The following example scatters the maximum along each axis of a
    one-filled input, so a cell changes only where an update exceeds ``1``:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [3, 3], device=device)
        with Graph("scatter_max", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            updates = ops.constant(
                np.array([[5.0, 2.0, 3.0], [4.0, 7.0, 6.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            indices = ops.constant(
                np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int64),
                DType.int64,
                device=device,
            )
            graph.output(
                ops.scatter_max(x, updates, indices, axis=0),
                ops.scatter_max(x, updates, indices, axis=1),
            )

        model = InferenceSession(devices=[CPU()]).load(graph)
        along_rows, along_cols = model.execute(np.ones((3, 3), np.float32))
        # along_rows: [[5.0, 1.0, 6.0], [1.0, 7.0, 1.0], [1.0, 1.0, 3.0]]
        # along_cols: [[5.0, 2.0, 3.0], [6.0, 7.0, 1.0], [1.0, 1.0, 1.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            along_rows.to_numpy(),
            [[5.0, 1.0, 6.0], [1.0, 7.0, 1.0], [1.0, 1.0, 3.0]],
        )
        np.testing.assert_allclose(
            along_cols.to_numpy(),
            [[5.0, 2.0, 3.0], [6.0, 7.0, 1.0], [1.0, 1.0, 1.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to compare.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, if ``input``, ``updates``,
            and ``indices`` aren't on the same device, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
        Error: If ``input``, ``updates``, and ``indices`` don't have equal
            rank, if ``updates.shape`` and ``indices.shape`` differ, or if any
            ``indices`` dimension exceeds the corresponding ``input``
            dimension.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range"
            f" [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter_max", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterMaxOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def scatter_min(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the minimum of updates into input.

    Produces an output tensor by scattering elements from updates into input
    according to indices, keeping the minimum at duplicate indices. For a 2-D
    input with ``axis=0``, each element updates as ``output[indices[i][j]][j]
    = min(output[indices[i][j]][j], updates[i][j])``; with ``axis=1`` the
    index selects the column, so ``output[i][indices[i][j]] =
    min(output[i][indices[i][j]], updates[i][j])``.

    The following example scatters the minimum along each axis of an input
    filled with ``9``, so a cell changes only where an update is smaller:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [3, 3], device=device)
        with Graph("scatter_min", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            updates = ops.constant(
                np.array([[5.0, 2.0, 3.0], [4.0, 7.0, 6.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            indices = ops.constant(
                np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int64),
                DType.int64,
                device=device,
            )
            graph.output(
                ops.scatter_min(x, updates, indices, axis=0),
                ops.scatter_min(x, updates, indices, axis=1),
            )

        model = InferenceSession(devices=[CPU()]).load(graph)
        along_rows, along_cols = model.execute(
            np.full((3, 3), 9.0, np.float32)
        )
        # along_rows: [[4.0, 9.0, 6.0], [9.0, 2.0, 9.0], [9.0, 9.0, 3.0]]
        # along_cols: [[5.0, 2.0, 3.0], [4.0, 7.0, 9.0], [9.0, 9.0, 9.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            along_rows.to_numpy(),
            [[4.0, 9.0, 6.0], [9.0, 2.0, 9.0], [9.0, 9.0, 3.0]],
        )
        np.testing.assert_allclose(
            along_cols.to_numpy(),
            [[5.0, 2.0, 3.0], [4.0, 7.0, 9.0], [9.0, 9.0, 9.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to compare.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, if ``input``, ``updates``,
            and ``indices`` aren't on the same device, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
        Error: If ``input``, ``updates``, and ``indices`` don't have equal
            rank, if ``updates.shape`` and ``indices.shape`` differ, or if any
            ``indices`` dimension exceeds the corresponding ``input``
            dimension.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range"
            f" [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter_min", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterMinOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def scatter_mul(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
    axis: int = -1,
) -> TensorValue:
    """Creates a new symbolic tensor by scattering the product of updates into input.

    Produces an output tensor by scattering elements from updates into input
    according to indices, multiplying values at duplicate indices. For a 2-D
    input with ``axis=0``, each element updates as ``output[indices[i][j]][j]
    *= updates[i][j]``; with ``axis=1`` the index selects the column, so
    ``output[i][indices[i][j]] *= updates[i][j]``.

    The following example multiplies into each axis of a one-filled input,
    where cells hit twice along an axis accumulate both factors:

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [3, 3], device=device)
        with Graph("scatter_mul", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            updates = ops.constant(
                np.array([[5.0, 2.0, 3.0], [4.0, 7.0, 6.0]],
                         dtype=np.float32),
                DType.float32,
                device=device,
            )
            indices = ops.constant(
                np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int64),
                DType.int64,
                device=device,
            )
            graph.output(
                ops.scatter_mul(x, updates, indices, axis=0),
                ops.scatter_mul(x, updates, indices, axis=1),
            )

        model = InferenceSession(devices=[CPU()]).load(graph)
        along_rows, along_cols = model.execute(np.ones((3, 3), np.float32))
        # along_rows: [[20.0, 1.0, 6.0], [1.0, 14.0, 1.0], [1.0, 1.0, 3.0]]
        # along_cols: [[5.0, 2.0, 3.0], [24.0, 7.0, 1.0], [1.0, 1.0, 1.0]]

    .. invisible-code-block: python

        np.testing.assert_allclose(
            along_rows.to_numpy(),
            [[20.0, 1.0, 6.0], [1.0, 14.0, 1.0], [1.0, 1.0, 3.0]],
        )
        np.testing.assert_allclose(
            along_cols.to_numpy(),
            [[5.0, 2.0, 3.0], [24.0, 7.0, 1.0], [1.0, 1.0, 1.0]],
        )

    Args:
        input: The input symbolic tensor to scatter into.
        updates: A symbolic tensor of values to multiply.
        indices: The positions in input to update.
        axis: The axis along which indices indexes into.

    Returns:
        A ``TensorValue`` representing the updated tensor. It has the same
        shape and dtype as ``input``.

    Raises:
        ValueError: If ``axis`` is out of range, if dtypes mismatch, if
            ``indices`` dtype is not int32/int64, if ``input``, ``updates``,
            and ``indices`` aren't on the same device, or if any input is on a
            non-CPU device and
            ``strict_device_placement=DevicePlacementPolicy.Error``.
        Error: If ``input``, ``updates``, and ``indices`` don't have equal
            rank, if ``updates.shape`` and ``indices.shape`` differ, or if any
            ``indices`` dimension exceeds the corresponding ``input``
            dimension.
    """
    input = TensorValue(input)

    if not (-input.rank <= axis < input.rank):
        raise ValueError(
            f"Invalid axis value {axis}. Axis must be in range"
            f" [-{input.rank}, {input.rank - 1}]"
        )

    updates = TensorValue(updates)
    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype '{input.dtype}' and updates dtype"
            f" '{updates.dtype}' must match."
        )

    indices = TensorValue(indices)
    if indices.dtype not in [DType.int32, DType.int64]:
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'."
            " Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)
    old_device = input.device if not input.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.scatter_mul", "TODO(GEX-2197).")
        input = transfer_to(input, DeviceRef.CPU())
        updates = transfer_to(updates, DeviceRef.CPU())
        indices = transfer_to(indices, DeviceRef.CPU())
    axis_constant = constant(axis, DType.int64, DeviceRef.CPU())

    result = Graph.current._add_op_generated(
        rmo.MoScatterMulOp,
        input.type,
        input,
        updates,
        indices,
        axis_constant,
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result


def masked_scatter(
    input: TensorValueLike,
    mask: TensorValueLike,
    updates: TensorValueLike,
    out_dim: DimLike,
) -> TensorValue:
    """Writes ``updates`` into a copy of ``input`` at positions where ``mask`` is true.

    Positions are filled in row-major order, so the first ``True`` position in
    ``mask`` takes the first element of ``updates``, and so on.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("masked_scatter") as graph:
            x = ops.constant(
                [[1, 2], [3, 4]], DType.int32, device=device
            )
            mask = ops.constant(
                [[True, False], [False, True]], DType.bool, device=device
            )
            updates = ops.constant([10, 20], DType.int32, device=device)
            # Write into the True positions, producing [[10, 2], [3, 20]].
            graph.output(ops.masked_scatter(x, mask, updates, out_dim="num_updates"))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        input: The input symbolic tensor to write elements to.
        mask: A symbolic tensor selecting the positions to write. A Python list
            or scalar is coerced to a boolean tensor; a tensor of any dtype is
            accepted, with its nonzero elements marking the positions to write.
            It's broadcast to the shape of ``input``.
        updates: A symbolic tensor of elements to write to ``input``.
        out_dim: The new data-dependent dimension for the number of ``True``
            positions in ``mask``.

    Returns:
        A ``TensorValue`` representing ``input`` with ``updates`` written where
        ``mask`` is true. It has the same shape and dtype as ``input``.

    Raises:
        ValueError: If the dtypes of ``input`` and ``updates`` mismatch, or if
            ``input`` and ``updates`` are on different devices.
    """
    input, updates = TensorValue(input), TensorValue(updates)
    mask = dtype_promotion._promote_to_strong(
        mask, DType.bool, input.type.device or DeviceRef.CPU()
    )

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    mask = mask.broadcast_to(input.shape)
    indices = nonzero(mask, out_dim)

    updates = updates.flatten()

    return scatter_nd(input, updates, indices)
