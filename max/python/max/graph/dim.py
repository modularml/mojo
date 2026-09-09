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
"""Library for graph dimension types."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from max._core.dialects import builtin, kgen


class Dim:
    """A tensor dimension.

    Dims describe the shape of tensors in a :class:`Graph`. In most cases, you don't
    need to construct a ``Dim`` directly. Instead, you pass dimension values
    directly to :class:`TensorType` or :class:`BufferType` constructors:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import Dim, TensorType, DeviceRef

        # Create a TensorType with a symbolic "batch" dimension and a static dimension of size 10
        tensor_type = TensorType(DType.int64, ("batch", 10), device=DeviceRef.CPU())


    A tensor dimension can be one of three types:

    - **Static**: A known size. See :class:`StaticDim`.
    - **Symbolic**: An unknown size identified by name. See :class:`SymbolicDim`.
    - **Algebraic**: An expression derived from symbolic dimensions. See :class:`AlgebraicDim`.

    Static dimensions let the graph compiler resolve shapes at compile time.
    This enables more aggressive optimizations than symbolic or algebraic
    dimensions allow. That said, when tensors share a named symbolic dimension,
    the compiler can leverage the implied shape equality to optimize some
    operations.
    """

    def __new__(cls, value: DimLike):
        """Converts valid input values to Dim."""
        if cls is not Dim:
            # For subclass constructors, preserve identity only for same subclass.
            if isinstance(value, cls):
                return value
            return super().__new__(cls)
        if isinstance(value, Dim):
            # For base Dim constructor, pass through any existing Dim.
            return value
        if isinstance(
            value, int | np.integer | builtin.IntegerAttr | kgen.SIMDAttr
        ):
            return super().__new__(StaticDim)
        if isinstance(value, str | kgen.ParamDeclRefAttr):
            return super().__new__(SymbolicDim)
        if isinstance(value, kgen.ParamOperatorAttr):
            return super().__new__(AlgebraicDim)

        raise TypeError(f"Unsupported dimension type {value} ({type(value)})")

    def __index__(self) -> int:
        """Converts this dim to an index as used by indexing and slicing.

        This raises and suggests explicitly converting to int, so that we only
        support implicit slicing operations on TensorValues.
        Types such as list and np.ndarray call __index__ on inputs to their
        __getitem__ special methods to convert those inputs to int.

        This also prevents a MyPy false positive error: Slice index must be an
        integer or None.
        Related MyPy discussion: https://github.com/python/mypy/issues/2410
        """
        raise TypeError(
            "when using dims to index into a list or NumPy array, explicitly "
            "convert to int with int(dim)"
        )

    def __int__(self) -> int:
        """Conversion to an int only supported for static dims."""
        raise TypeError(
            f"int({self!r}): Int conversions only supported for static dims"
        )

    def __eq__(self, other: object) -> bool:
        """Checks whether two dimensions are equal.

        Dimensions are equal if they are the same dimension type
        (symbolic, static). Additionally, static dimensions
        are only equal if their dimension is the same size, and symbolic
        dimensions are only equal if they have the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions are equal, false otherwise.
        """
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        """Checks whether two dimensions are not equal.

        The inverse of __eq__.

        Args:
            other: The other dimension to check inequality against.

        Returns:
            False if the dimensions are equal, true otherwise.
        """
        return not self == other

    def __add__(self, rhs: DimLike) -> Dim:
        if not isinstance(rhs, DimLike):
            return NotImplemented
        return AlgebraicDim.apply(kgen.POC.add, self, rhs)

    def __radd__(self, lhs: DimLike) -> Dim:
        if not isinstance(lhs, DimLike):
            return NotImplemented
        return Dim(lhs) + self

    def __mul__(self, rhs: DimLike) -> Dim:
        if not isinstance(rhs, DimLike):
            return NotImplemented
        return AlgebraicDim.apply(kgen.POC.mul_no_wrap, self, rhs)

    def __rmul__(self, lhs: DimLike) -> Dim:
        if not isinstance(lhs, DimLike):
            return NotImplemented
        return Dim(lhs) * self

    def __neg__(self) -> Dim:
        return -1 * self

    def __sub__(self, rhs: DimLike) -> Dim:
        if not isinstance(rhs, DimLike):
            return NotImplemented
        return self + -Dim(rhs)

    def __rsub__(self, lhs: DimLike) -> Dim:
        if not isinstance(lhs, DimLike):
            return NotImplemented
        return lhs + -self

    def __floordiv__(self, rhs: DimLike) -> Dim:
        if not isinstance(rhs, DimLike):
            return NotImplemented
        if isinstance(rhs, int | StaticDim) and int(rhs) == 0:
            raise ZeroDivisionError
        return AlgebraicDim.apply(kgen.POC.div, self, rhs)

    def __rfloordiv__(self, lhs: DimLike) -> Dim:
        if not isinstance(lhs, DimLike):
            return NotImplemented
        return lhs // self

    def to_mlir(self) -> builtin.TypedAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            attr: The MLIR Attribute to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if isinstance(attr, builtin.IntegerAttr | kgen.SIMDAttr):
            return StaticDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamDeclRefAttr):
            return SymbolicDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamOperatorAttr):
            return AlgebraicDim.from_mlir(attr)
        else:
            raise ValueError("graph api does not support unknown dimensions")

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        raise NotImplementedError

    def substitute(self, mapping: Mapping[str, DimLike]) -> Dim:
        """Returns this dim with named symbols replaced per ``mapping``.

        Substituting static values folds the result through the compiler's
        own attribute evaluation; substituting symbols renames. Unmapped
        symbols are left intact.

        Args:
            mapping: A mapping from symbolic dimension name to the
                replacement dim (or dim-like value).

        Returns:
            The dim with substitutions applied.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SymbolicDim(Dim):
    """A symbolic tensor dimension with an unknown size identified by name.

    When you don't know a dimension value at compile time, you can use a
    symbolic dimension. This helps you identify dimensions by name and lets the
    compiler optimize operations when two or more dimensions share the same name.

    The following example creates a symbolic dimension implicitly passing the
    strings ``"batch"`` and ``"x"`` to :class:`TensorType`:

    .. code-block:: python

       from max.dtype import DType
       from max.graph import DeviceRef, TensorType

       tensor_type = TensorType(DType.float32, ("batch", "x", 10), device=DeviceRef.CPU())
    """

    name: str
    """The name of the dimension."""

    def __init__(self, name: str | builtin.TypedAttr | SymbolicDim) -> None:
        if isinstance(name, kgen.ParamDeclRefAttr):
            name = name.name.value
        elif isinstance(name, SymbolicDim):
            name = name.name
        elif not isinstance(name, str):
            raise TypeError(
                f"SymbolicDim.__init__ only accepts str, kgen.ParamDeclRefAttr, or SymbolicDim, got {type(name).__name__}"
            )
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("name", name)
        # TODO(MSDK-695): less restrictive names
        if not re.match(r"^[a-zA-Z_]\w*$", self.name):
            raise ValueError("Invalid name for symbolic dimension")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Dim({self.name!r})"

    def __eq__(self, other: object) -> bool:
        """Whether the dimension is the same as another symbolic dimension.

        Symbolic dimensions with the same name are interpreted as the same
        dimensionality! If you use Symbolic dimensions, make sure you're naming
        them consistently, your model will likely fail to compile if you name
        two actually different dimensions the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions have the same name, false otherwise.
        """
        return self.name == other or (
            isinstance(other, SymbolicDim) and self.name == other.name
        )

    def to_mlir(self) -> kgen.ParamDeclRefAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
        return kgen.ParamDeclRefAttr(self.name, si64)

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> SymbolicDim:
        """Constructs a ``SymbolicDim`` from a ``kgen.ParamDeclRefAttr``.

        Args:
            attr: The ``kgen.ParamDeclRefAttr`` to parse into a ``SymbolicDim``.

        Returns:
            SymbolicDim: The ``SymbolicDim`` represented by the ``kgen.ParamDeclRefAttr``.
        """
        if not isinstance(attr, kgen.ParamDeclRefAttr):
            raise TypeError(
                f"SymbolicDim.from_mlir only accepts kgen.ParamDeclRefAttr, got {type(attr).__name__}"
            )
        return SymbolicDim(attr)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        yield self

    def substitute(self, mapping: Mapping[str, DimLike]) -> Dim:
        """Returns this dim with named symbols replaced per ``mapping``.

        Args:
            mapping: A mapping from symbolic dimension name to the
                replacement dim (or dim-like value).

        Returns:
            ``Dim(mapping[self.name])`` if ``self.name`` is in ``mapping``,
            otherwise this dim unchanged.
        """
        replacement = mapping.get(self.name)
        return self if replacement is None else Dim(replacement)


@dataclass(frozen=True)
class AlgebraicDim(Dim):
    """A dimension defined by an expression over symbolic dimensions.

    Arithmetic on symbolic dimensions produces an ``AlgebraicDim``:

    .. code-block:: python

        from max.graph import AlgebraicDim, Dim

        isinstance(Dim("batch") * 5, AlgebraicDim)  # True

    Equivalent expressions simplify to the same form:

    .. code-block:: python

        from max.graph import Dim

        Dim("x") + 1 + 1 == Dim("x") + 2  # True

    .. note::

        Algebraic dimensions are valid inside a graph. However, they can't
        appear in graph input or output types because their underlying values
        can be ambiguous. For example, a dimension of ``Dim("foo") *
        Dim("bar")`` could be satisfied by multiple combinations of ``foo`` and
        ``bar``.

    """

    attr: kgen.ParamOperatorAttr

    def __init__(self, attr: builtin.TypedAttr | AlgebraicDim) -> None:
        if isinstance(attr, AlgebraicDim):
            attr = attr.attr
        elif not isinstance(attr, kgen.ParamOperatorAttr):
            raise TypeError(
                f"AlgebraicDim.__init__ only accepts kgen.ParamOperatorAttr or AlgebraicDim, got {type(attr).__name__}"
            )
        super().__setattr__("attr", attr)

    @classmethod
    def apply(cls, op: kgen.POC, *operands: DimLike):  # noqa: ANN206
        """Applies a parametric operator to operands and returns the resulting dimension."""
        # kgen.ParamOperatorAttr eagerly folds on construction!
        #  - this can return static or symbolic dims
        #  - let Dim decide what type to return
        attr = kgen.ParamOperatorAttr(
            op, [Dim(operand).to_mlir() for operand in operands]
        )
        return Dim(attr)

    def substitute(self, mapping: Mapping[str, DimLike]) -> Dim:
        """Returns this dim with named symbols replaced per ``mapping``.

        Substitutes into each operand, then reapplies the operator, so
        substitution to static values folds through the compiler's own
        attribute evaluation rather than being recomputed in Python.

        Args:
            mapping: A mapping from symbolic dimension name to the
                replacement dim (or dim-like value).

        Returns:
            The dim with substitutions applied, re-folded by the compiler.

        Raises:
            ZeroDivisionError: If substitution produces a zero divisor.
        """
        operands = [
            Dim(operand).substitute(mapping) for operand in self.attr.operands
        ]
        # __floordiv__'s zero guard does not run here, and the compiler
        # neither folds nor rejects a literal zero divisor.
        if self.attr.opcode == kgen.POC.div and operands[1] == 0:
            raise ZeroDivisionError(
                f"substituting into {self} produced a zero divisor"
            )
        return AlgebraicDim.apply(self.attr.opcode, *operands)

    def __format__(self, format_spec: str) -> str:
        formatters: Mapping[str, Callable[[Any], str]] = {
            "str": str,
            "repr": repr,
        }
        formatter = formatters[format_spec or "str"]

        def format(dim: Dim):  # noqa: ANN202
            formatted = formatter(dim)
            return (
                f"({formatted})" if isinstance(dim, AlgebraicDim) else formatted
            )

        # For the opcodes we support in the graph api, print with python math.
        opcodes = {
            kgen.POC.add: "+",
            kgen.POC.mul_no_wrap: "*",
            kgen.POC.div: "//",
        }
        opcode = self.attr.opcode
        dims = [Dim(operand) for operand in self.attr.operands]
        if opcode in opcodes:
            # Wrap algebraic sub-expressions in parens
            return f" {opcodes[opcode]} ".join(map(format, dims))
        return formatter(self.attr)

    def __str__(self) -> str:
        return f"{self:str}"

    def __repr__(self) -> str:
        return f"{self:repr}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AlgebraicDim) and self.attr == other.attr

    def to_mlir(self) -> kgen.ParamOperatorAttr:
        """Creates an mlir.Attribute representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        return self.attr

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> AlgebraicDim:
        """Constructs an ``AlgebraicDim`` from a ``kgen.ParamOperatorAttr``.

        Args:
            attr: The ``kgen.ParamOperatorAttr`` to parse into an ``AlgebraicDim``.

        Returns:
            AlgebraicDim: The ``AlgebraicDim`` represented by the ``kgen.ParamOperatorAttr``.
        """
        if not isinstance(attr, kgen.ParamOperatorAttr):
            raise TypeError(
                f"AlgebraicDim.from_mlir only accepts kgen.ParamOperatorAttr, got {type(attr).__name__}"
            )
        return AlgebraicDim(attr)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        for operand in self.attr.operands:
            yield from Dim(operand).parameters


@functools.total_ordering
@dataclass(frozen=True)
class StaticDim(Dim):
    """A static tensor dimension with a fixed size.

    Because a static dimension's size is fixed, related computation can be
    optimized at compile time. This is key to good model performance.

    The following example creates static dimensions implicitly by passing
    integer values to :class:`TensorType`:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, TensorType
        tensor = TensorType(DType.int64, (4, 5), device=DeviceRef.CPU())
        # This creates a tensor with 2 static dimensions: 4 and 5 respectively
    """

    dim: int
    """The size of the static dimension."""

    def __init__(self, dim: int | builtin.TypedAttr | StaticDim) -> None:
        # Use CastToBuiltinAttr to fold back to integer.
        if isinstance(dim, kgen.SIMDAttr):
            dim = kgen.CastToBuiltinAttr(dim)

        if isinstance(dim, builtin.IntegerAttr):
            dim = dim.value
        elif isinstance(dim, StaticDim):
            dim = dim.dim
        elif not isinstance(dim, int):
            raise TypeError(
                f"StaticDim.__init__ only accepts int, builtin.IntegerAttr, or StaticDim, got {type(dim).__name__}"
            )
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("dim", dim)
        if not -(2**63) <= self.dim < 2**63:
            raise ValueError("Dim value must be -2**63 <= dim < 2**63")

    def substitute(self, mapping: Mapping[str, DimLike]) -> Dim:
        """Returns this dim unchanged: a static dim has no symbols.

        Args:
            mapping: A mapping from symbolic dimension name to the
                replacement dim (or dim-like value). Ignored.

        Returns:
            This dim, unchanged.
        """
        return self

    def __str__(self) -> str:
        return str(self.dim)

    def __repr__(self) -> str:
        return f"Dim({repr(self.dim)})"

    def __int__(self) -> int:
        return self.dim

    def __eq__(self, other: object) -> bool:
        """Whether the dimension has the same size as another dimension.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if both dimensions have the same static size, false otherwise.
        """
        return self.dim == other or (
            isinstance(other, StaticDim) and self.dim == other.dim
        )

    def __lt__(self, other: int | StaticDim):
        return self.dim < (other.dim if isinstance(other, StaticDim) else other)

    def __hash__(self):
        return hash(self.dim)

    def to_mlir(self) -> builtin.TypedAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        return kgen.CastFromBuiltinAttr(builtin.IntegerAttr(si64, self.dim))

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> StaticDim:
        """Constructs a ``StaticDim`` from a ``builtin.IntegerAttr``.

        Args:
            attr: The ``builtin.IntegerAttr`` to parse into a ``StaticDim``.

        Returns:
            StaticDim: The ``StaticDim`` represented by the ``builtin.IntegerAttr``.
        """
        if not isinstance(attr, builtin.IntegerAttr | kgen.SIMDAttr):
            raise TypeError(
                f"StaticDim.from_mlir only accepts builtin.IntegerAttr, got {type(attr).__name__}"
            )
        return StaticDim(attr)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this dim depends."""
        return ()


DimLike = int | str | Dim | np.integer | builtin.TypedAttr
