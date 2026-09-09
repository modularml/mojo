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
"""Elementwise ops."""

from collections.abc import Callable

from max._core import Operation
from max._core.dialects import builtin, kgen, rmo
from max.dtype import DType

from .. import dtype_promotion
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike
from .cast import cast
from .custom import custom
from .validation import assert_same_device

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


# This implementation needs to be in sync with the mojo implementation found in
# stdlib/utils/numerics.mojo
def _accum_type(
    x: TensorValue | TensorType, preferred_type: DType = DType.float32
) -> DType:
    dtype = x.dtype
    if dtype.is_float8():
        return (
            DType.float32 if preferred_type == DType.float32 else DType.bfloat16
        )
    if dtype == DType.float16:
        return (
            DType.float32 if preferred_type == DType.float32 else DType.float16
        )
    if dtype == DType.bfloat16:
        return DType.float32
    return dtype


# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_binary(op_type: type[Operation], name: str):  # noqa: ANN202
    def elementwise_op(
        lhs: TensorValueLike, rhs: TensorValueLike
    ) -> TensorValue:
        lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)
        assert_same_device(lhs=lhs, rhs=rhs)
        return Graph.current._add_op_generated(
            op_type, input_x=lhs, input_y=rhs
        )[0].tensor

    elementwise_op.__name__ = name
    return elementwise_op


add = _elementwise_binary(rmo.AddOp, "add")
add.__doc__ = """Adds two tensors element-wise.


.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("add_example") as graph:
        lhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([4.0, 5.0, 6.0], DType.float32, device=device)
        graph.output(ops.add(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [5.0, 7.0, 9.0])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` representing the element-wise sums.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""


def div(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Divides two tensors element-wise using true division (Python ``/``).

    For integer operands, this performs true division by promoting to float,
    matching Python's ``/`` operator behavior. For floating-point operands,
    this performs standard floating-point division.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("div_example") as graph:
            lhs = ops.constant(
                [6.0, 10.0, 18.0], DType.float32, device=device
            )
            rhs = ops.constant([2.0, 5.0, 6.0], DType.float32, device=device)
            graph.output(ops.div(lhs, rhs))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.array_equal(result.to_numpy(), [3.0, 2.0, 3.0])

    Args:
        lhs: The numerator input.
        rhs: The denominator input.

    Returns:
        A ``TensorValue`` with the broadcast shape representing ``lhs / rhs``
        element-wise. The result has a floating-point dtype for integer
        operands and the promoted dtype for mixed types.

    Raises:
        Error: If the input shapes are not compatible for broadcasting.
        Error: If one of the inputs has an unsupported dtype.
        Error: If the two symbols are parts of different graphs.
    """
    lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)

    if lhs.dtype.is_integral() and rhs.dtype.is_integral():
        float_dtype = DType.float64  # Use double precision for accuracy
        lhs = cast(lhs, float_dtype)
        rhs = cast(rhs, float_dtype)

    assert_same_device(lhs, rhs)
    return Graph.current._add_op_generated(rmo.DivOp, input_x=lhs, input_y=rhs)[
        0
    ].tensor


def floor_div(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Divides two tensors element-wise using floor division (Python ``//``).

    The result is rounded toward negative infinity for all operands, matching
    Python's ``//``. Integer operands stay in the integer domain: the divide
    truncates toward zero, then a floor correction is applied for signed
    integers (a no-op for unsigned or non-negative operands). Floating-point
    operands compute ``floor(lhs / rhs)``.

    Unlike :obj:`div`, integer operands are never promoted to ``float64``. This
    matters on backends without native 64-bit floating-point support (for
    example, Apple/Metal GPUs), where an ``f64`` intermediate fails to compile.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("floor_div_example") as graph:
            lhs = ops.constant([7, 10, 18], DType.int32, device=device)
            rhs = ops.constant([2, 5, 6], DType.int32, device=device)
            graph.output(ops.floor_div(lhs, rhs))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.array_equal(result.to_numpy(), [3, 2, 3])

    Args:
        lhs: The numerator input.
        rhs: The denominator input.

    Returns:
        A ``TensorValue`` with the broadcast shape representing the element-wise
        floor division of ``lhs`` by ``rhs``.

    Raises:
        Error: If the input shapes are not compatible for broadcasting.
        Error: If one of the inputs has an unsupported dtype.
        Error: If the two symbols are parts of different graphs.
    """
    lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)
    assert_same_device(lhs, rhs)
    if lhs.dtype.is_integral() and rhs.dtype.is_integral():
        # Integer division stays in the integer domain, mirroring `mod`
        # (`rmo.ModOp`), so there is no `float64` promotion like `div` does.
        # `rmo.DivOp` truncates toward zero.
        quotient = Graph.current._add_op_generated(
            rmo.DivOp, input_x=lhs, input_y=rhs
        )[0].tensor
        if lhs.dtype.is_signed_integral():
            # Truncation toward zero and floor division differ by one when the
            # exact quotient is negative (operand signs differ) and the divide
            # leaves a nonzero remainder. Correct so the result matches `//`.
            remainder = mod(lhs, rhs)
            quotient = quotient - (
                (remainder != 0) & ((lhs < 0) ^ (rhs < 0))
            ).cast(quotient.dtype)
        return quotient
    return floor(div(lhs, rhs))


max = _elementwise_binary(rmo.MaxOp, "max")
max.__doc__ = """
Computes the element-wise maximum of two tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("max_example") as graph:
        lhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([4.0, 2.0, 6.0], DType.float32, device=device)
        graph.output(ops.max(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [4.0, 5.0, 6.0])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` representing the maximum value at each position.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

min = _elementwise_binary(rmo.MinOp, "min")
min.__doc__ = """
Computes the element-wise minimum of two tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("min_example") as graph:
        lhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([4.0, 2.0, 6.0], DType.float32, device=device)
        graph.output(ops.min(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` representing the minimum value at each position.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

mod = _elementwise_binary(rmo.ModOp, "mod")
mod.__doc__ = """
Computes the element-wise modulus of two tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("mod_example") as graph:
        lhs = ops.constant([10.0, 7.0, 5.0], DType.float32, device=device)
        rhs = ops.constant([3.0, 2.0, 4.0], DType.float32, device=device)
        graph.output(ops.mod(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [1.0, 1.0, 1.0])

Args:
    lhs: The dividend.
    rhs: The divisor.

Returns:
    A ``TensorValue`` representing ``lhs % rhs`` element-wise.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

mul = _elementwise_binary(rmo.MulOp, "mul")
mul.__doc__ = """
Multiplies two tensors element-wise.


.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("mul_example") as graph:
        lhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([4.0, 5.0, 6.0], DType.float32, device=device)
        graph.output(ops.mul(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [4.0, 10.0, 18.0])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` representing the element-wise products.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

pow = _elementwise_binary(rmo.PowOp, "pow")
pow.__doc__ = """
Raises elements of one tensor to the power of another element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("pow_example") as graph:
        lhs = ops.constant([2.0, 3.0, 4.0], DType.float32, device=device)
        rhs = ops.constant([3.0, 2.0, 0.5], DType.float32, device=device)
        graph.output(ops.pow(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [8.0, 9.0, 2.0], atol=1e-3)

Args:
    lhs: The base tensor.
    rhs: The exponent tensor.

Returns:
    A ``TensorValue`` with the broadcast shape representing ``lhs ** rhs`` element-wise.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

sub = _elementwise_binary(rmo.SubOp, "sub")
sub.__doc__ = """
Subtracts two tensors element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("sub_example") as graph:
        lhs = ops.constant([5.0, 7.0, 9.0], DType.float32, device=device)
        rhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        graph.output(ops.sub(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [4.0, 5.0, 6.0])

Args:
    lhs: The minuend (left-hand side).
    rhs: The subtrahend (right-hand side).

Returns:
    A ``TensorValue`` representing the result of ``lhs - rhs`` element-wise.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

equal = _elementwise_binary(rmo.EqualOp, "equal")
equal.__doc__ = """Tests element-wise equality between two tensors.


.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("equal_example") as graph:
        lhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        graph.output(ops.equal(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [True, False, True])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise result
    of ``lhs == rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

greater = _elementwise_binary(rmo.GreaterOp, "greater")
greater.__doc__ = """Tests element-wise whether one tensor is greater than another.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("greater_example") as graph:
        lhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([1.0, 2.0, 4.0], DType.float32, device=device)
        graph.output(ops.greater(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, False])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise result
    of ``lhs > rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

greater_equal = _elementwise_binary(rmo.GreaterEqualOp, "greater_equal")
greater_equal.__doc__ = """Tests element-wise whether one tensor is greater than or equal to another.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("greater_equal_example") as graph:
        lhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([1.0, 2.0, 4.0], DType.float32, device=device)
        graph.output(ops.greater_equal(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [True, True, False])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise result
    of ``lhs >= rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

not_equal = _elementwise_binary(rmo.NotEqualOp, "not_equal")
not_equal.__doc__ = """Tests element-wise inequality between two tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("not_equal_example") as graph:
        lhs = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        rhs = ops.constant([1.0, 5.0, 3.0], DType.float32, device=device)
        graph.output(ops.not_equal(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, False])

Args:
    lhs: The left-hand side input.
    rhs: The right-hand side input.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise result
    of ``lhs != rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_and = _elementwise_binary(rmo.AndOp, "logical_and")
logical_and.__doc__ = """Computes the element-wise logical AND of two boolean tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("logical_and_example") as graph:
        lhs = ops.constant([True, True, False], DType.bool, device=device)
        rhs = ops.constant([True, False, True], DType.bool, device=device)
        graph.output(ops.logical_and(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [True, False, False])

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise logical
    AND of ``lhs`` and ``rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_or = _elementwise_binary(rmo.OrOp, "logical_or")
logical_or.__doc__ = """Computes the element-wise logical OR of two boolean tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("logical_or_example") as graph:
        lhs = ops.constant([True, False, False], DType.bool, device=device)
        rhs = ops.constant([False, True, False], DType.bool, device=device)
        graph.output(ops.logical_or(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [True, True, False])

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise logical
    OR of ``lhs`` and ``rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_xor = _elementwise_binary(rmo.XorOp, "logical_xor")
logical_xor.__doc__ = """Computes the element-wise logical XOR of two boolean tensors.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("logical_xor_example") as graph:
        lhs = ops.constant([True, False, True], DType.bool, device=device)
        rhs = ops.constant([True, True, False], DType.bool, device=device)
        graph.output(ops.logical_xor(lhs, rhs))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, True])

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype representing the element-wise logical
    XOR of ``lhs`` and ``rhs``.

Raises:
    Error: If the input shapes are not compatible for broadcasting.
    Error: If one of the inputs has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""


# ===----------------------------------------------------------------------=== #
# Unary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_unary(op_type: type[Operation], name: str):  # noqa: ANN202
    def elementwise_op(x: TensorValueLike) -> TensorValue:
        x = dtype_promotion._restrict_to_strong_dtypes(x)
        return Graph.current._add_op_generated(
            op_type,
            result=x.type,
            input=x,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    elementwise_op.__name__ = name
    return elementwise_op


def _elementwise_unary_predicate(
    op_type: type[Operation], name: str
) -> Callable[[TensorValueLike], TensorValue]:
    def elementwise_op(x: TensorValueLike) -> TensorValue:
        x = dtype_promotion._restrict_to_strong_dtypes(x)
        return Graph.current._add_op_generated(
            op_type,
            result=TensorType(dtype=DType.bool, shape=x.shape, device=x.device),
            input_x=x,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    elementwise_op.__name__ = name
    return elementwise_op


def _activation(x: TensorValueLike, op_type: type[Operation]) -> TensorValue:
    """Builds a single fused activation op of the given type.

    Each elementwise activation function (``relu``, ``gelu`` and its
    approximations, ``sigmoid``, ``silu``) has its own dedicated op, backed by a
    hardware-optimized fused Mojo kernel, rather than a Python-level composition
    of ``exp``/``erf``/etc.
    """
    x = dtype_promotion._restrict_to_strong_dtypes(x)
    return Graph.current._add_op_generated(
        op_type,
        result=x.type,
        input=x,
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor


abs = _elementwise_unary(rmo.MoAbsOp, "abs")
abs.__doc__ = """Computes the absolute value of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("abs_example") as graph:
        x = ops.constant([-1.0, 2.0, -3.0], DType.float32, device=device)
        graph.output(ops.abs(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [1.0, 2.0, 3.0])

Args:
    x: The input tensor.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the
    absolute value of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor.
"""

exp = _elementwise_unary(rmo.MoExpOp, "exp")
exp.__doc__ = """Computes the exponential of a tensor element-wise.

This applies ``exp(x) = e^x``, where ``e`` is Euler's number.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("exp_example") as graph:
        x = ops.constant([0.0, 1.0, 2.0], DType.float32, device=device)
        graph.output(ops.exp(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [1.0, 2.718, 7.389], atol=1e-3)

Args:
    x: The input to the exponential function. Must have a floating-point
        dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing ``e``
    raised to the power of each element of ``x``.

Raises:
    Error: If the input does not represent a tensor or has a non-floating-point dtype.
"""

erf = _elementwise_unary(rmo.MoErfOp, "erf")
erf.__doc__ = """Computes the error function of a tensor element-wise.

The error function ``erf`` is the probability that a randomly sampled
normal distribution falls within a given range.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("erf_example") as graph:
        x = ops.constant([-1.0, 0.0, 1.0], DType.float32, device=device)
        graph.output(ops.erf(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [-0.842, 0.0, 0.842], atol=1e-3)

Args:
    x: The input to the error function. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the error
    function applied to each element of ``x``.

Raises:
    Error: If the input is not a tensor or has a non-floating-point dtype.
"""


def gelu(x: TensorValue, approximate: str = "none"):  # noqa: ANN201
    """Applies the GELU (Gaussian Error Linear Unit) activation element-wise.

    For ``approximate == "none"``, MAX computes the exact GELU function.

    For ``approximate == "tanh"``, MAX uses the approximation:

    .. code:: text

        gelu(x) = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    For ``approximate == "quick"``, MAX uses the approximation:

    .. code:: text

        gelu(x) = sigmoid(1.702 * x) * x

    Args:
        x: The input to the GELU computation. Must have a floating-point
            dtype.
        approximate: One of ``"none"``, ``"tanh"``, or ``"quick"``. Defaults
            to ``"none"``.

    Returns:
        A ``TensorValue`` of the same shape and dtype as ``x`` representing the
        GELU activation applied to each element of ``x``.

    Raises:
        Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
        ValueError: If the approximation method is invalid.
    """
    if approximate == "none":
        return _activation(x, rmo.MoGeluOp)
    if approximate == "tanh":
        return _activation(x, rmo.MoGeluTanhOp)
    if approximate == "quick":
        return _activation(x, rmo.MoGeluQuickOp)

    raise ValueError(f"Invalid approximation method: {approximate}")


log = _elementwise_unary(rmo.MoLogOp, "log")
log.__doc__ = """
Computes the natural logarithm of a tensor element-wise.

This applies ``log(x)``. It is the inverse of the exponential
function ``x = e^y``, where ``e`` is Euler's number.
Note that ``log(x)`` is undefined for ``x <= 0`` and complex numbers
are not currently supported.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("log_example") as graph:
        x = ops.constant(
            [1.0, 2.718, 7.389, 20.0], DType.float32, device=device
        )
        graph.output(ops.log(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [0.0, 1.0, 2.0, 2.996], atol=1e-3)


Args:
    x: The input to the log computation. Must contain positive values only.
        Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the
    natural logarithm of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

log1p = _elementwise_unary(rmo.MoLog1pOp, "log1p")
log1p.__doc__ = """Computes ``log(1 + x)`` element-wise.

Note that ``log(1 + x)`` is undefined for ``x <= -1`` and complex
numbers are not currently supported.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("log1p_example") as graph:
        x = ops.constant([0.0, 1.0, 9.0], DType.float32, device=device)
        graph.output(ops.log1p(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [0.0, 0.693, 2.302], atol=1e-3)


Args:
    x: The input to the log computation. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing
    ``log(1 + x)`` for each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""


def _softmax_like(op_type: type[Operation], name: str):  # noqa: ANN202
    def softmax_like_op(value: TensorValueLike, axis: int = -1) -> TensorValue:
        value = TensorValue(value)

        axis = value.rank - 1 if axis == -1 else axis
        value = dtype_promotion._restrict_to_strong_dtypes(value)
        return Graph.current._add_op_generated(
            op_type,
            result=value.type,
            input=value,
            axis=builtin.IntegerAttr(builtin.IndexType(), axis),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    softmax_like_op.__name__ = name
    return softmax_like_op


logsoftmax = _softmax_like(rmo.MoReduceLogsoftmaxOp, "logsoftmax")
logsoftmax.__doc__ = """Computes the log-softmax of a tensor along an axis.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("logsoftmax_example") as graph:
        x = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        graph.output(ops.logsoftmax(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(
        result.to_numpy(), [-2.407, -1.407, -0.407], atol=1e-3
    )

Args:
    value: The input to the log-softmax computation. Must have a
        floating-point dtype.
    axis: The axis along which to compute the log-softmax. Defaults to the
        final axis (``-1``).

Returns:
    A ``TensorValue`` of the same shape and dtype as ``value`` representing the
    log-softmax of ``value`` computed along ``axis``.

Raises:
    Error: If the input is not a tensor or has a non-floating-point dtype.
"""

relu = _elementwise_unary(rmo.MoReluOp, "relu")
relu.__doc__ = """Applies the ReLU (Rectified Linear Unit) activation element-wise.

ReLU is defined as ``relu(x) = max(0, x)``, meaning negative values are set to zero
while positive values are unchanged.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("relu_example") as graph:
        x = ops.constant(
            [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]],
            DType.float32,
            device=device,
        )
        graph.output(ops.relu(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(
        result.to_numpy(), [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    )

Args:
    x: The input to the ReLU computation.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing ``x`` with
    its negative elements replaced by ``0``.

Raises:
    Error: If the input doesn't represent a tensor.
"""


def sigmoid(x: TensorValue) -> TensorValue:
    """Applies the sigmoid activation function element-wise.

    Computes ``sigmoid(x) = 1 / (1 + exp(-x))``, mapping all values to the
    range ``(0, 1)``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("sigmoid_example") as graph:
            x = ops.constant(
                [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]],
                DType.float32,
                device=device,
            )
            graph.output(ops.sigmoid(x))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(
            result.to_numpy(),
            [[0.119, 0.269, 0.5], [0.731, 0.881, 0.953]],
            atol=1e-3,
        )

    Args:
        x: The input to the sigmoid computation. Must have a floating-point
            dtype.

    Returns:
        A ``TensorValue`` of the same shape and dtype as ``x`` representing each
        element of ``x`` mapped to the range ``(0, 1)``.

    Raises:
        Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
    """
    return _activation(x, rmo.MoSigmoidOp)


def silu(x: TensorValue):  # noqa: ANN201
    """Applies the SiLU (Swish) activation function element-wise.

    Computes ``silu(x) = x * sigmoid(x)``.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("silu_example") as graph:
            x = ops.constant(
                [-2.0, 0.0, 1.0, 3.0], DType.float32, device=device
            )
            graph.output(ops.silu(x))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(
            result.to_numpy(), [-0.238, 0.0, 0.731, 2.857], atol=1e-3
        )

    Args:
        x: The input to the SiLU computation. Must have a floating-point
            dtype.

    Returns:
        A ``TensorValue`` of the same shape and dtype as ``x`` representing the
        SiLU activation applied to each element of ``x``.

    Raises:
        Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
    """
    return _activation(x, rmo.MoSiluOp)


softmax = _softmax_like(rmo.MoReduceSoftmaxOp, "softmax")
softmax.__doc__ = """Computes the softmax of a tensor along an axis.

Normalizes the values along ``axis`` so that they sum to ``1``, with each
output element representing the exponentiated input divided by the sum of
exponentiated values along that axis.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("softmax_example") as graph:
        x = ops.constant([1.0, 2.0, 3.0], DType.float32, device=device)
        graph.output(ops.softmax(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [0.090, 0.244, 0.665], atol=1e-3)

Args:
    value: The input to the softmax computation. Must have a floating-point
        dtype.
    axis: The axis along which to compute the softmax. Defaults to the
        final axis (``-1``).

Returns:
    A ``TensorValue`` of the same shape and dtype as ``value`` representing the
    softmax of ``value`` computed along ``axis``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

cos = _elementwise_unary(rmo.MoCosOp, "cos")
cos.__doc__ = """Computes the cosine of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("cos_example") as graph:
        x = ops.constant([0.0, 1.5707, 3.1415], DType.float32, device=device)
        graph.output(ops.cos(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [1.0, 0.0, -1.0], atol=1e-3)

Args:
    x: The input interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the cosine
    of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

ceil = _elementwise_unary(rmo.MoCeilOp, "ceil")
ceil.__doc__ = """Computes the ceiling of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("ceil_example") as graph:
        x = ops.constant([1.5, -1.5, 2.7, -2.7], DType.float32, device=device)
        graph.output(ops.ceil(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [2.0, -1.0, 3.0, -2.0])

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing each
    element of ``x`` rounded up toward positive infinity.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

floor = _elementwise_unary(rmo.MoFloorOp, "floor")
floor.__doc__ = """Computes the floor of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("floor_example") as graph:
        x = ops.constant([1.5, -1.5, 2.7, -2.7], DType.float32, device=device)
        graph.output(ops.floor(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [1.0, -2.0, 2.0, -3.0])

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing each
    element of ``x`` rounded down toward negative infinity.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

round = _elementwise_unary(rmo.MoRoundOp, "round")
round.__doc__ = """Rounds a tensor to the nearest integer element-wise.

Values exactly halfway between two integers round to the nearest even integer
(for example, ``2.5`` rounds to ``2.0`` and ``3.5`` rounds to ``4.0``). All
other values follow normal rounding to the nearest integer.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("round_example") as graph:
        x = ops.constant([1.5, 2.5, 3.5, -1.5], DType.float32, device=device)
        graph.output(ops.round(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [2.0, 2.0, 4.0, -2.0])

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing each
    element of ``x`` rounded to the nearest integer.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

rsqrt = _elementwise_unary(rmo.MoRsqrtOp, "rsqrt")
rsqrt.__doc__ = """Computes the reciprocal square root of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("rsqrt_example") as graph:
        x = ops.constant([1.0, 4.0, 9.0, 16.0], DType.float32, device=device)
        graph.output(ops.rsqrt(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [1.0, 0.5, 0.333, 0.25], atol=1e-3)

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the
    reciprocal square root of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

sqrt = _elementwise_unary(rmo.MoSqrtOp, "sqrt")
sqrt.__doc__ = """Computes the square root of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("sqrt_example") as graph:
        x = ops.constant([1.0, 4.0, 9.0, 16.0], DType.float32, device=device)
        graph.output(ops.sqrt(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [1.0, 2.0, 3.0, 4.0], atol=1e-3)

Args:
    x: The input tensor. Must have a floating-point dtype. Negative values
    produce ``NaN`` since MAX doesn't support complex numbers.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the square
    root of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

sin = _elementwise_unary(rmo.MoSinOp, "sin")
sin.__doc__ = """Computes the sine of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("sin_example") as graph:
        x = ops.constant([0.0, 1.5707, 3.1415], DType.float32, device=device)
        graph.output(ops.sin(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [0.0, 1.0, 0.0], atol=1e-3)

Args:
    x: The input interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the sine
    of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

tanh = _elementwise_unary(rmo.MoTanhOp, "tanh")
tanh.__doc__ = """Computes the hyperbolic tangent of a tensor element-wise.

This applies ``tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))``, which maps
all values to the range ``(-1, 1)``.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("tanh_example") as graph:
        x = ops.constant(
            [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]],
            DType.float32,
            device=device,
        )
        graph.output(ops.tanh(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(
        result.to_numpy(),
        [[-0.964, -0.762, 0.0], [0.762, 0.964, 0.995]],
        atol=1e-3,
    )

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing each
    element of ``x`` mapped to the range ``(-1, 1)``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

atanh = _elementwise_unary(rmo.MoAtanhOp, "atanh")
atanh.__doc__ = """Computes the inverse hyperbolic tangent of a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("atanh_example") as graph:
        x = ops.constant([-0.5, 0.0, 0.5], DType.float32, device=device)
        graph.output(ops.atanh(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [-0.549, 0.0, 0.549], atol=1e-3)

Args:
    x: The input tensor, with values in the range ``(-1, 1)``. Must have a
        floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the
    inverse hyperbolic tangent of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
"""

trunc = _elementwise_unary(rmo.MoTruncOp, "trunc")
trunc.__doc__ = """Truncates a tensor toward zero element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("trunc_example") as graph:
        x = ops.constant([1.5, -1.5, 2.7, -2.7], DType.float32, device=device)
        graph.output(ops.trunc(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [1.0, -1.0, 2.0, -2.0])

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing each
    element of ``x`` truncated toward zero.

Raises:
    Error: If the input doesn't represent tensor or has a non-floating-point dtype.
"""

is_nan = _elementwise_unary_predicate(rmo.MoIsNanOp, "is_nan")
is_nan.__doc__ = """Tests element-wise whether a tensor contains NaN values.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("is_nan_example") as graph:
        x = ops.constant(
            [1.0, float("nan"), 3.0], DType.float32, device=device
        )
        graph.output(ops.is_nan(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, False])

Args:
    x: The input tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype and the same shape as ``x``,
    representing an element-wise NaN test. An element is ``True`` where ``x`` is
    NaN.

Raises:
    Error: If the input doesn't represent a tensor.
"""


is_inf = _elementwise_unary_predicate(rmo.MoIsInfOp, "is_inf")
is_inf.__doc__ = """Tests element-wise whether a tensor contains infinite values.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("is_inf_example") as graph:
        x = ops.constant(
            [1.0, float("inf"), 3.0], DType.float32, device=device
        )
        graph.output(ops.is_inf(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, False])

Args:
    x: The input tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype and the same shape as ``x``,
    representing an element-wise infinity test. An element is ``True`` where
    ``x`` is positive or negative infinity.

Raises:
    Error: If the input doesn't represent a tensor.
"""

logical_not = _elementwise_unary(rmo.MoNotOp, "logical_not")
logical_not.__doc__ = """Computes the element-wise logical NOT of a boolean tensor.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("logical_not_example") as graph:
        x = ops.constant([True, False, True], DType.bool, device=device)
        graph.output(ops.logical_not(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [False, True, False])

Args:
    x: The input boolean tensor.

Returns:
    A ``TensorValue`` with ``bool`` dtype and the same shape as ``x``,
    representing the element-wise logical NOT of ``x``.

Raises:
    Error: If the symbol doesn't represent a tensor.
"""

negate = _elementwise_unary(rmo.MoNegativeOp, "negate")
negate.__doc__ = """Negates a tensor element-wise.

.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("negate_example") as graph:
        x = ops.constant([1.0, -2.0, 3.0], DType.float32, device=device)
        graph.output(ops.negate(x))

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.array_equal(result.to_numpy(), [-1.0, 2.0, -3.0])

Args:
    x: The input tensor.

Returns:
    A ``TensorValue`` of the same shape and dtype as ``x`` representing the
    negation of each element of ``x``.

Raises:
    Error: If the input doesn't represent a tensor.
"""


def acos(x: TensorValue) -> TensorValue:
    """Computes the arccosine of a tensor element-wise.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("acos_example") as graph:
            x = ops.constant(
                [-1.0, 0.0, 0.5, 1.0], DType.float32, device=device
            )
            graph.output(ops.acos(x))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(
            result.to_numpy(), [3.141, 1.570, 1.047, 0.0], atol=1e-3
        )

    Args:
        x: The input tensor with values in ``[-1, 1]``. For the ``float16``,
            ``bfloat16``, and ``float32`` dtypes, values outside this domain
            are clamped to the valid range. For ``float64``, they yield
            ``NaN``. Must have a floating-point dtype.

    Returns:
        A ``TensorValue`` of the same shape and dtype as ``x`` representing the
        arccosine of each element of ``x``. Values range from ``[0, π]`` (radians).

    Raises:
        Error: If the input doesn't represent a tensor or has a non-floating-point dtype.
    """
    x = dtype_promotion._restrict_to_strong_dtypes(x)
    device = x.device
    return custom(
        "mo.acos",
        x.device,
        [x],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=x.tensor.shape,
                device=DeviceRef.from_device(device),
            )
        ],
    )[0].tensor
