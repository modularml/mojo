# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements `StaticIntTuple` which is commonly used to represent N-D
indices.

You can import these APIs from the `utils` package. For example:

```mojo
from utils import StaticIntTuple
```
"""

from builtin.io import _get_dtype_printf_format, _snprintf
from builtin.string import _calc_initial_buffer_size

from . import unroll
from .static_tuple import StaticTuple

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


alias mlir_bool = __mlir_type.`!pop.scalar<bool>`


@always_inline
fn _reduce_and_fn(a: Bool, b: Bool) -> Bool:
    """Performs AND operation on two boolean inputs.

    Args:
        a: The first boolean input.
        b: The second boolean input.

    Returns:
        The result of AND operation on the inputs.
    """
    return a and b


# ===----------------------------------------------------------------------===#
# Integer and Bool Tuple Utilities:
#   Utilities to operate on tuples of integers or tuples of bools.
# ===----------------------------------------------------------------------===#


@always_inline
fn _int_tuple_binary_apply[
    size: Int,
    binary_fn: fn (Int, Int) -> Int,
](a: StaticTuple[Int, size], b: StaticTuple[Int, size]) -> StaticTuple[
    Int, size
]:
    """Applies a given element binary function to each pair of corresponding
    elements in two tuples.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_binary_apply[size, Int.add](a, b)

    Parameters:
        size: Static size of the operand and result tuples.
        binary_fn: Binary function to apply to tuple elements.

    Args:
        a: Tuple containing lhs operands of the elementwise binary function.
        b: Tuple containing rhs operands of the elementwise binary function.

    Returns:
        Tuple containing the result.
    """

    var c = StaticTuple[Int, size]()

    @always_inline
    @parameter
    fn do_apply[idx: Int]():
        var a_elem: Int = a.__getitem__[idx]()
        var b_elem: Int = b.__getitem__[idx]()
        c.__setitem__[idx](binary_fn(a_elem, b_elem))

    unroll[do_apply, size]()

    return c


@always_inline
fn _int_tuple_compare[
    size: Int,
    comp_fn: fn (Int, Int) -> Bool,
](a: StaticTuple[Int, size], b: StaticTuple[Int, size]) -> StaticTuple[
    mlir_bool,
    size,
]:
    """Applies a given element compare function to each pair of corresponding
    elements in two tuples and produces a tuple of Bools containing result.

    Example Usage:
        var a: StaticTuple[Int, size]
        var b: StaticTuple[Int, size]
        var c = _int_tuple_compare[size, Int.less_than](a, b)

    Parameters:
        size: Static size of the operand and result tuples.
        comp_fn: Compare function to apply to tuple elements.

    Args:
        a: Tuple containing lhs operands of the elementwise compare function.
        b: Tuple containing rhs operands of the elementwise compare function.

    Returns:
        Tuple containing the result.
    """

    var c = StaticTuple[mlir_bool, size]()

    @always_inline
    @parameter
    fn do_compare[idx: Int]():
        var a_elem: Int = a.__getitem__[idx]()
        var b_elem: Int = b.__getitem__[idx]()
        c.__setitem__[idx](comp_fn(a_elem, b_elem)._as_scalar_bool())

    unroll[do_compare, size]()

    return c


@always_inline
fn _bool_tuple_reduce[
    size: Int,
    reduce_fn: fn (Bool, Bool) -> Bool,
](a: StaticTuple[mlir_bool, size], init: Bool) -> Bool:
    """Reduces the tuple argument with the given reduce function and initial
    value.

    Example Usage:
        var a: StaticTuple[mlir_bool, size]
        var c = _bool_tuple_reduce[size, _reduce_and_fn](a, True)

    Parameters:
        size: Static size of the operand and result tuples.
        reduce_fn: Reduce function to accumulate tuple elements.

    Args:
        a: Tuple containing elements to reduce.
        init: Value to initialize the reduction with.

    Returns:
        The result of the reduction.
    """

    var c: Bool = init

    @always_inline
    @parameter
    fn do_reduce[idx: Int]():
        c = reduce_fn(c, a.__getitem__[idx]())

    unroll[do_reduce, size]()

    return c


# ===----------------------------------------------------------------------===#
# StaticIntTuple:
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct StaticIntTuple[size: Int](Sized, Stringable, Comparable):
    """A base struct that implements size agnostic index functions.

    Parameters:
        size: The size of the tuple.
    """

    var data: StaticTuple[Int, size]
    """The underlying storage of the tuple value."""

    @always_inline
    fn __init__(inout self):
        """Constructs a static int tuple of the given size."""
        self = 0

    @always_inline
    fn __init__(inout self, value: __mlir_type.index):
        """Constructs a sized 1 static int tuple of given the element value.

        Args:
            value: The initial value.
        """
        constrained[size == 1]()
        self = Int(value)

    @always_inline
    fn __init__(inout self, elems: (Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[StaticIntTuple] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Reference[Int, __lifetime_of(elems)]](
                Reference(elems[idx])
            )[]

        unroll[fill, 2]()

        self = tup

    @always_inline
    fn __init__(inout self, elems: (Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[StaticIntTuple] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Reference[Int, __lifetime_of(elems)]](
                Reference(elems[idx])
            )[]

        unroll[fill, 3]()

        self = tup

    @always_inline
    fn __init__(inout self, elems: (Int, Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[StaticIntTuple] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Reference[Int, __lifetime_of(elems)]](
                Reference(elems[idx])
            )[]

        unroll[fill, 4]()

        self = tup

    @always_inline
    fn __init__(inout self, *elems: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elems: The elements to construct the tuple.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[StaticIntTuple] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(size):
            tup[idx] = elems[idx]

        self = tup

    @always_inline
    fn __init__(inout self, elem: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elem: The elem to splat into the tuple.
        """

        self.data = __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[`!pop.array<`, size.value, `, `, Int, `>`]
        ](elem)

    @always_inline
    fn __init__(inout self, values: VariadicList[Int]):
        """Creates a tuple constant using the specified values.

        Args:
            values: The list of values.
        """
        constrained[size > 0]()
        self.data = values

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the size of the tuple.

        Returns:
            The tuple size.
        """
        return size

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Gets an element from the tuple by index.

        Args:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return self.data[idx]

    @always_inline("nodebug")
    fn __setitem__[index: Int](inout self, val: Int):
        """Sets an element in the tuple at the given static index.

        Parameters:
            index: The element index.

        Args:
            val: The value to store.
        """
        self.data.__setitem__[index](val)

    @always_inline("nodebug")
    fn __setitem__(inout self, idx: Int, val: Int):
        """Sets an element in the tuple at the given index.

        Args:
            idx: The element index.
            val: The value to store.
        """
        self.data[idx] = val

    @always_inline("nodebug")
    fn as_tuple(self) -> StaticTuple[Int, size]:
        """Converts this StaticIntTuple to StaticTuple.

        Returns:
            The corresponding StaticTuple object.
        """
        return self.data

    @always_inline
    fn flattened_length(self) -> Int:
        """Returns the flattened length of the tuple.

        Returns:
            The flattened length of the tuple.
        """
        var length: Int = 1

        @parameter
        for i in range(size):
            length *= self[i]

        return length

    @always_inline
    fn __add__(self, rhs: StaticIntTuple[size]) -> StaticIntTuple[size]:
        """Performs element-wise integer add.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Int:
            return a + b

        return Self {
            data: _int_tuple_binary_apply[size, apply_fn](self.data, rhs.data)
        }

    @always_inline
    fn __sub__(self, rhs: StaticIntTuple[size]) -> StaticIntTuple[size]:
        """Performs element-wise integer subtract.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Int:
            return a - b

        return Self {
            data: _int_tuple_binary_apply[size, apply_fn](self.data, rhs.data)
        }

    @always_inline
    fn __mul__(self, rhs: StaticIntTuple[size]) -> StaticIntTuple[size]:
        """Performs element-wise integer multiply.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Int:
            return a * b

        return Self {
            data: _int_tuple_binary_apply[size, apply_fn](self.data, rhs.data)
        }

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        """Performs element-wise integer floor division.

        Args:
            rhs: The elementwise divisor.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Int:
            return a // b

        return Self {
            data: _int_tuple_binary_apply[size, apply_fn](self.data, rhs.data)
        }

    @always_inline
    fn __rfloordiv__(self, rhs: Self) -> Self:
        """Floor divides rhs by this object.

        Args:
            rhs: The value to elementwise divide by self.

        Returns:
            The resulting index tuple.
        """
        return rhs // self

    @always_inline
    fn remu(self, rhs: StaticIntTuple[size]) -> StaticIntTuple[size]:
        """Performs element-wise integer unsigned modulo.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Int:
            return a % b

        return Self {
            data: _int_tuple_binary_apply[size, apply_fn](self.data, rhs.data)
        }

    @always_inline
    fn __eq__(self, rhs: StaticIntTuple[size]) -> Bool:
        """Compares this tuple to another tuple for equality.

        The tuples are equal if all corresponding elements are equal.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn(a: Int, b: Int) -> Bool:
            return a == b

        return _bool_tuple_reduce[size, _reduce_and_fn](
            _int_tuple_compare[size, apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ne__(self, rhs: StaticIntTuple[size]) -> Bool:
        """Compares this tuple to another tuple for non-equality.

        The tuples are non-equal if at least one element of LHS isn't equal to
        the corresponding element from RHS.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """
        return not (self == rhs)

    @always_inline
    fn __lt__(self, rhs: StaticIntTuple[size]) -> Bool:
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
        fn apply_fn(a: Int, b: Int) -> Bool:
            return a < b

        return _bool_tuple_reduce[size, _reduce_and_fn](
            _int_tuple_compare[size, apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __le__(self, rhs: StaticIntTuple[size]) -> Bool:
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
        fn apply_fn(a: Int, b: Int) -> Bool:
            return a <= b

        return _bool_tuple_reduce[size, _reduce_and_fn](
            _int_tuple_compare[size, apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __gt__(self, rhs: StaticIntTuple[size]) -> Bool:
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
        fn apply_fn(a: Int, b: Int) -> Bool:
            return a > b

        return _bool_tuple_reduce[size, _reduce_and_fn](
            _int_tuple_compare[size, apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ge__(self, rhs: StaticIntTuple[size]) -> Bool:
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
        fn apply_fn(a: Int, b: Int) -> Bool:
            return a >= b

        return _bool_tuple_reduce[size, _reduce_and_fn](
            _int_tuple_compare[size, apply_fn](self.data, rhs.data), True
        )

    fn __str__(self) -> String:
        """Get the tuple as a string.

        Returns:
            A string representation.
        """
        # Reserve space for opening and closing parentheses, plus each element
        # and its trailing commas.
        var buf = String._buffer_type()
        var initial_buffer_size = 2
        for i in range(size):
            initial_buffer_size += _calc_initial_buffer_size(self[i]) + 2
        buf.reserve(initial_buffer_size)

        # Print an opening `(`.
        buf.size += _snprintf["("](buf.data, 2)
        for i in range(size):
            # Print separators between each element.
            if i != 0:
                buf.size += _snprintf[", "](buf.data + buf.size, 3)
            buf.size += _snprintf[_get_dtype_printf_format[DType.index]()](
                buf.data + buf.size, _calc_initial_buffer_size(self[i]), self[i]
            )
        # Single element tuples should be printed with a trailing comma.
        if size == 1:
            buf.size += _snprintf[","](buf.data + buf.size, 2)
        # Print a closing `)`.
        buf.size += _snprintf[")"](buf.data + buf.size, 2)

        buf.size += 1  # for the null terminator.
        return buf^


# ===----------------------------------------------------------------------===#
# Factory functions for creating index.
# ===----------------------------------------------------------------------===#
@always_inline
fn Index[T0: Intable](x: T0) -> StaticIntTuple[1]:
    """Constructs a 1-D Index from the given value.

    Parameters:
        T0: The type of the 1st argument.

    Args:
        x: The initial value.

    Returns:
        The constructed StaticIntTuple.
    """
    return StaticIntTuple[1](int(x))


@always_inline
fn Index[T0: Intable, T1: Intable](x: T0, y: T1) -> StaticIntTuple[2]:
    """Constructs a 2-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.

    Returns:
        The constructed StaticIntTuple.
    """
    return StaticIntTuple[2](int(x), int(y))


@always_inline
fn Index[
    T0: Intable, T1: Intable, T2: Intable
](x: T0, y: T1, z: T2) -> StaticIntTuple[3]:
    """Constructs a 3-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.

    Returns:
        The constructed StaticIntTuple.
    """
    return StaticIntTuple[3](int(x), int(y), int(z))


@always_inline
fn Index[
    T0: Intable, T1: Intable, T2: Intable, T3: Intable
](x: T0, y: T1, z: T2, w: T3) -> StaticIntTuple[4]:
    """Constructs a 4-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.

    Returns:
        The constructed StaticIntTuple.
    """
    return StaticIntTuple[4](int(x), int(y), int(z), int(w))


@always_inline
fn Index[
    T0: Intable, T1: Intable, T2: Intable, T3: Intable, T4: Intable
](x: T0, y: T1, z: T2, w: T3, v: T4) -> StaticIntTuple[5]:
    """Constructs a 5-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.
        T4: The type of the 5th argument.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.
        v: The 5th initial value.

    Returns:
        The constructed StaticIntTuple.
    """
    return StaticIntTuple[5](int(x), int(y), int(z), int(w), int(v))


# ===----------------------------------------------------------------------===#
# Utils
# ===----------------------------------------------------------------------===#


@always_inline
fn product[size: Int](tuple: StaticIntTuple[size], end_idx: Int) -> Int:
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
fn product[
    size: Int
](tuple: StaticIntTuple[size], start_idx: Int, end_idx: Int) -> Int:
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
        product *= tuple[i]
    return product
