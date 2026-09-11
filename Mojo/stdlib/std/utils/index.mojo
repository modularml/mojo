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
"""Implements `IndexList` which is commonly used to represent N-D
indices.

You can import these APIs from the `utils` package. For example:

```mojo
from std.utils import IndexList
```
"""

from std.hashlib.hasher import Hasher
from std.sys import bit_width_of

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.dtype import _int_type_of_width, _uint_type_of_width
import std.format._utils as fmt

from .static_tuple import StaticTuple

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _reduce_and_fn(a: Bool, b: Bool) -> Bool:
    """Performs AND operation on two boolean inputs.

    Args:
        a: The first boolean input.
        b: The second boolean input.

    Returns:
        The result of AND operation on the inputs.
    """
    return a and b


# ===-----------------------------------------------------------------------===#
# Integer and Bool Tuple Utilities:
#   Utilities to operate on tuples of integers or tuples of bools.
# ===-----------------------------------------------------------------------===#


@always_inline
def _int_tuple_binary_apply[
    binary_fn: def[dtype: DType](Scalar[dtype], Scalar[dtype]) thin -> Scalar[
        dtype
    ],
](a: IndexList, b: type_of(a), out c: type_of(a)):
    """Applies a given element binary function to each pair of corresponding
    elements in two tuples.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_binary_apply[size, Int.add](a, b)

    Args:
        a: Tuple containing lhs operands of the elementwise binary function.
        b: Tuple containing rhs operands of the elementwise binary function.

    Returns:
        Tuple containing the result.
    """

    c = {}

    comptime for i in range(a.size):
        c[i] = Int(
            binary_fn(
                Scalar[a.element_type](a.get[i]()),
                Scalar[a.element_type](b.get[i]()),
            )
        )


@always_inline
def _int_tuple_compare[
    comp_fn: def[dtype: DType](Scalar[dtype], Scalar[dtype]) thin -> Bool,
](a: IndexList, b: type_of(a)) -> StaticTuple[Bool, a.size]:
    """Applies a given element compare function to each pair of corresponding
    elements in two tuples and produces a tuple of Bools containing result.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_compare[size, Int.less_than](a, b)

    Args:
        a: Tuple containing lhs operands of the elementwise compare function.
        b: Tuple containing rhs operands of the elementwise compare function.

    Returns:
        Tuple containing the result.
    """

    var c = StaticTuple[Bool, a.size]()

    comptime for i in range(a.size):
        c[i] = comp_fn[a.element_type](
            Scalar[a.element_type](a.get[i]()),
            Scalar[a.element_type](b.get[i]()),
        )

    return c


@always_inline
def _bool_tuple_reduce[
    reduce_fn: def(Bool, Bool) thin -> Bool,
](a: StaticTuple[Bool, _], init: Bool) -> Bool:
    """Reduces the tuple argument with the given reduce function and initial
    value.

    Example Usage:
        var a: StaticTuple[Bool, size]
        var c = _bool_tuple_reduce[size, _reduce_and_fn](a, True)

    Parameters:
        reduce_fn: Reduce function to accumulate tuple elements.

    Args:
        a: Tuple containing elements to reduce.
        init: Value to initialize the reduction with.

    Returns:
        The result of the reduction.
    """

    var c: Bool = init

    comptime for i in range(a.size):
        c = reduce_fn(c, a.get[i]())

    return c


# ===-----------------------------------------------------------------------===#
# IndexList:
# ===-----------------------------------------------------------------------===#


def _type_of_width[bitwidth: Int, unsigned: Bool]() -> DType:
    comptime if unsigned:
        return _uint_type_of_width[bitwidth]()
    else:
        return _int_type_of_width[bitwidth]()


struct IndexList[size: Int, *, element_type: DType = .int64](
    Comparable,
    Defaultable,
    DevicePassable,
    Hashable,
    ImplicitlyCopyable,
    Sized,
    TrivialRegisterPassable,
    Writable,
):
    """A base struct that implements size agnostic index functions.

    Parameters:
        size: The size of the tuple.
        element_type: The underlying dtype of the integer element value.
    """

    comptime device_type = Self
    """Indicate the type being used on accelerator devices."""

    comptime _int_type = Scalar[Self.element_type]
    """The underlying storage of the integer element value."""

    var data: StaticTuple[Self._int_type, Self.size]
    """The underlying storage of the tuple value."""

    @always_inline
    def __init__(out self):
        """Constructs a static int tuple of the given size."""
        self.data = StaticTuple[_, Self.size](fill=Self._int_type(0))

    @always_inline
    @implicit
    def __init__(out self, data: StaticTuple[Self._int_type, Self.size]):
        """Constructs a static int tuple of the given size.

        Args:
            data: The StaticTuple to construct the IndexList from.
        """
        comptime assert (
            Self.element_type.is_integral()
        ), "Element type must be of integral type."
        self.data = data

    @always_inline
    @implicit
    def __init__[*Ts: Movable & Intable](out self, elems: Tuple[*Ts]):
        """Constructs a static int tuple given a tuple of integers.

        Parameters:
            Ts: The element types of the input tuple (must be `Intable`).

        Args:
            elems: The tuple to copy from.
        """
        comptime assert (
            Self.element_type.is_integral()
        ), "Element type must be of integral type."
        comptime num_elements = type_of(elems).__len__()
        comptime assert (
            Self.size == num_elements
        ), "[IndexList] mismatch in the number of elements"

        var tup = Self()

        comptime for idx in range(num_elements):
            tup[idx] = Int(elems[idx])

        self = tup

    @always_inline
    def __init__(out self, *elems: Int, __list_literal__: NoneType = None):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elems: The elements to construct the tuple.
            __list_literal__: Specifies that this constructor can be used for
               list literals.
        """
        comptime assert (
            Self.element_type.is_integral()
        ), "Element type must be of integral type."

        comptime assert (
            Self.element_type.is_integral()
        ), "Element type must be of integral type."
        var num_elements = len(elems)

        assert (
            Self.size == num_elements
        ), "[IndexList] mismatch in the number of elements"

        self = Self()
        comptime for idx in range(Self.size):
            self[idx] = elems[idx]

    @always_inline
    def __init__(out self, fill: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            fill: The elem to splat into the tuple.
        """
        comptime assert (
            Self.element_type.is_integral()
        ), "Element type must be of integral type."
        self.data = StaticTuple[_, Self.size](fill=Self._int_type(fill))

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Returns the size of the tuple.

        Returns:
            The tuple size.
        """
        return Self.size

    @always_inline
    def get[idx: Int](self) -> Int:
        """Gets an element from the tuple by index parameter.

        Parameters:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return Int(self.data.get[idx]())

    @always_inline("nodebug")
    def __getitem__[I: Indexer](self, idx: I) -> Int:
        """Gets an element from the tuple by index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return Int(self.data[idx])

    @always_inline("nodebug")
    def __setitem__(mut self, idx: Int, val: Int):
        """Sets an element in the tuple at the given index.

        Args:
            idx: The element index.
            val: The value to store.
        """
        self.data[idx] = Scalar[Self.element_type](val)

    @always_inline("nodebug")
    def as_tuple(self) -> StaticTuple[Int, Self.size]:
        """Converts this IndexList to StaticTuple.

        Returns:
            The corresponding StaticTuple object.
        """
        var res = StaticTuple[Int, Self.size]()

        comptime for i in range(Self.size):
            res[i] = self.get[i]()
        return res

    def as_index_tuple(self) -> StaticTuple[SIMDLength, Self.size]:
        """Converts this IndexList to a static tuple of mlir indexes.

        Returns:
            The corresponding StaticTuple object.
        """
        var res = StaticTuple[SIMDLength, Self.size]()

        comptime for i in range(Self.size):
            res[i] = self.get[i]()

        return res

    @always_inline("nodebug")
    def canonicalize(
        self,
        out result: IndexList[Self.size, element_type=.int64],
    ):
        """Canonicalizes the IndexList.

        Returns:
            Canonicalizes the object.
        """
        return self.cast[.int64]()

    @always_inline
    def reverse(self) -> Self:
        """Reverses the IndexList.

        Returns:
            A new IndexList with the elements in reverse order.
        """
        var result = Self(0)

        comptime for i in range(Self.size):
            result[i] = self[Self.size - i - 1]
        return result

    @always_inline
    def flattened_length(self) -> Int:
        """Returns the flattened length of the tuple.

        Returns:
            The flattened length of the tuple.
        """
        var length: Int = 1

        comptime for i in range(Self.size):
            length *= self[i]

        return length

    @always_inline
    def get_row_major_strides(self) -> Self:
        """Interpret the current index list as a shape, and return the strides
        to traverse such a shape in row-major order.

        Returns:
            The strides to traverse the index list in row-major order.
        """
        var strides = Self()
        var offset = 1
        comptime for i in reversed(range(Self.size)):
            strides[i] = offset
            offset *= self[i]
        return strides

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        """Performs element-wise integer add.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        def apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a + b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        """Performs element-wise integer subtract.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        def apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a - b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        """Performs element-wise integer multiply.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        def apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a * b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        """Performs element-wise integer floor division.

        Args:
            rhs: The elementwise divisor.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        def apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a // b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    def __rfloordiv__(self, rhs: Self) -> Self:
        """Floor divides rhs by this object.

        Args:
            rhs: The value to elementwise divide by self.

        Returns:
            The resulting index tuple.
        """
        return rhs // self

    @always_inline
    def remu(self, rhs: Self) -> Self:
        """Performs element-wise integer unsigned modulo.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        def apply_fn[
            dtype: DType
        ](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
            return a % b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    def __eq__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple for equality.

        The tuples are equal if all corresponding elements are equal.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        def apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a == b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    def __lt__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using LT comparison.

        A tuple is less-than another tuple if all corresponding elements of lhs
        is less than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        def apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a < b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    def __le__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using LE comparison.

        A tuple is less-or-equal than another tuple if all corresponding
        elements of lhs is less-or-equal than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        def apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a <= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    def __gt__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using GT comparison.

        A tuple is greater-than than another tuple if all corresponding
        elements of lhs is greater-than than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        def apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a > b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    def __ge__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple using GE comparison.

        A tuple is greater-or-equal than another tuple if all corresponding
        elements of lhs is greater-or-equal than rhs.

        Note: This is **not** a lexical comparison.

        Args:
            rhs: Right hand side tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        def apply_fn[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Bool:
            return a >= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this IndexList value to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("(")

        for i in range(Self.size):
            if i != 0:
                writer.write(", ")

            var element = self[i]

            comptime if bit_width_of[Self.element_type]() == 32:
                writer.write(Int32(element))
            else:
                writer.write(Int64(element))

        # Single element tuples should be printed with a trailing comma.
        comptime if Self.size == 1:
            writer.write(",")

        writer.write(")")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `IndexList` to a writer.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[].write_to(w)

        fmt.FormatStruct(writer, "IndexList").params(
            Self.size,
            Self.element_type,
        ).fields(write_fields)

    @always_inline
    def cast[
        dtype: DType
    ](self, out result: IndexList[Self.size, element_type=dtype]):
        """Casts to the target DType.

        Parameters:
            dtype: The dtype to cast towards.

        Returns:
            The list casted to the target type.
        """
        comptime assert dtype.is_integral(), "the target type must be integral"
        result = {}

        comptime for i in range(Self.size):
            result.data[i] = self.data.get[i]().cast[result.element_type]()

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """

        comptime for i in range(Self.size):
            self.data[i].__hash__(hasher)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """
        Convert the host type object to a device_type and store it at the
        target address.

        NOTE: This should only be called by `DeviceContext` during invocation
        of accelerator kernels.
        """
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).
        For example, Int would return "Int", DeviceBuffer[.float32] would
        return "DeviceBuffer[DType.float32]". This is used for error messages
        when passing types to the device.
        TODO: This method will be retired soon when better kernel call error
        messages arrive.

        Returns:
            The host type's name.
        """
        return String(t"IndexList[{Self.size},{Self.element_type}]")


# ===-----------------------------------------------------------------------===#
# Factory functions for creating index.
# ===-----------------------------------------------------------------------===#


@always_inline
def Index[
    *Ts: Intable,
    dtype: DType = .int64,
](*args: *Ts, out result: IndexList[args.__len__(), element_type=dtype]):
    """Constructs an N-D Index from the given values.

    Parameters:
        Ts: The types of the arguments (must be `Intable`).
        dtype: The integer type of the underlying element of the resulting list.

    Args:
        args: The values to construct the index from.

    Returns:
        The constructed IndexList.
    """
    result = {}
    comptime for i in range(args.__len__()):
        result[i] = Int(args[i])


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline
def product[
    size: Int
](tuple: IndexList[size, element_type=_], end_idx: Int = size) -> Int:
    """Computes a product of values in the tuple up to the given index.

    Parameters:
        size: The tuple size.

    Args:
        tuple: The tuple to get a product of.
        end_idx: The end index.

    Returns:
        The product of all tuple elements in the given range.
    """
    return product[size](tuple, 0, end_idx)


@always_inline
def product[
    size: Int
](tuple: IndexList[size, element_type=_], start_idx: Int, end_idx: Int) -> Int:
    """Computes a product of values in the tuple in the given index range.

    Parameters:
        size: The tuple size.

    Args:
        tuple: The tuple to get a product of.
        start_idx: The start index of the range.
        end_idx: The end index of the range.

    Returns:
        The product of all tuple elements in the given range.
    """
    var product: Int = 1
    for i in range(start_idx, end_idx):
        var dim = tuple[i]
        assert (
            dim == 0 or product <= Int.MAX // dim
        ), "IndexList.product() overflow: dimension product exceeds Int.MAX"
        product *= dim
    return product
