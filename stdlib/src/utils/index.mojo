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
"""Implements `IndexList` which is commonly used to represent N-D
indices.

You can import these APIs from the `utils` package. For example:

```mojo
from utils import IndexList
```
"""

from collections.string import _calc_initial_buffer_size
from sys import bitwidthof

from builtin.dtype import _int_type_of_width, _uint_type_of_width
from builtin.io import _get_dtype_printf_format, _snprintf

from . import unroll
from .static_tuple import StaticTuple

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Integer and Bool Tuple Utilities:
#   Utilities to operate on tuples of integers or tuples of bools.
# ===-----------------------------------------------------------------------===#


@always_inline
fn _int_tuple_binary_apply[
    binary_fn: fn[type: DType] (Scalar[type], Scalar[type]) -> Scalar[type],
](a: IndexList, b: __type_of(a)) -> __type_of(a):
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

    var c = __type_of(a)()

    @parameter
    for i in range(a.size):
        var a_elem = a.__getitem__[i]()
        var b_elem = b.__getitem__[i]()
        c.__setitem__[i](binary_fn[a.element_type](a_elem, b_elem))

    return c


@always_inline
fn _int_tuple_compare[
    comp_fn: fn[type: DType] (Scalar[type], Scalar[type]) -> Bool,
](a: IndexList, b: __type_of(a)) -> StaticTuple[Bool, a.size]:
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

    @parameter
    for i in range(a.size):
        var a_elem = a.__getitem__[i]()
        var b_elem = b.__getitem__[i]()
        c.__setitem__[i](comp_fn[a.element_type](a_elem, b_elem))

    return c


@always_inline
fn _bool_tuple_reduce[
    reduce_fn: fn (Bool, Bool) -> Bool,
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

    @parameter
    for i in range(a.size):
        c = reduce_fn(c, a.__getitem__[i]())

    return c


# ===-----------------------------------------------------------------------===#
# IndexList:
# ===-----------------------------------------------------------------------===#


fn _type_of_width[bitwidth: Int, unsigned: Bool]() -> DType:
    @parameter
    if unsigned:
        return _uint_type_of_width[bitwidth]()
    else:
        return _int_type_of_width[bitwidth]()


fn _is_unsigned[type: DType]() -> Bool:
    return type in (DType.uint8, DType.uint16, DType.uint32, DType.uint64)


@value
@register_passable("trivial")
struct IndexList[
    size: Int,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    Sized,
    Stringable,
    Writable,
    Comparable,
):
    """A base struct that implements size agnostic index functions.

    Parameters:
        size: The size of the tuple.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.
    """

    alias element_type = _type_of_width[element_bitwidth, unsigned]()
    """The underlying dtype of the integer element value."""

    alias _int_type = Scalar[Self.element_type]
    """The underlying storage of the integer element value."""

    var data: StaticTuple[Self._int_type, size]
    """The underlying storage of the tuple value."""

    @always_inline
    fn __init__(out self):
        """Constructs a static int tuple of the given size."""
        self = 0

    @always_inline
    @implicit
    fn __init__(out self, data: StaticTuple[Self._int_type, size]):
        """Constructs a static int tuple of the given size.

        Args:
            data: The StaticTuple to construct the IndexList from.
        """
        self.data = data

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        """Constructs a sized 1 static int tuple of given the element value.

        Args:
            value: The initial value.
        """
        constrained[size == 1]()
        self = Int(value)

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Int](elems[idx])

        unroll[fill, 2]()

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Int](elems[idx])

        unroll[fill, 3]()

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elems: (Int, Int, Int, Int)):
        """Constructs a static int tuple given a tuple of integers.

        Args:
            elems: The tuple to copy from.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        fn fill[idx: Int]():
            tup[idx] = rebind[Int](elems[idx])

        unroll[fill, 4]()

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, *elems: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elems: The elements to construct the tuple.
        """

        var num_elements = len(elems)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(size):
            tup[idx] = elems[idx]

        self = tup

    @always_inline
    @implicit
    fn __init__(out self, elem: Int):
        """Constructs a static int tuple given a set of arguments.

        Args:
            elem: The elem to splat into the tuple.
        """

        self.data = __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[
                `!pop.array<`, size.value, `, `, Self._int_type, `>`
            ]
        ](Self._int_type(elem))

    fn __init__(out self, *, other: Self):
        """Copy constructor.

        Args:
            other: The other tuple to copy from.
        """
        self.data = other.data

    @always_inline
    @implicit
    fn __init__(out self, values: VariadicList[Int]):
        """Creates a tuple constant using the specified values.

        Args:
            values: The list of values.
        """

        var num_elements = len(values)

        debug_assert(
            size == num_elements,
            "[IndexList] mismatch in the number of elements",
        )

        var tup = Self()

        @parameter
        for idx in range(size):
            tup[idx] = values[idx]

        self = tup

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the size of the tuple.

        Returns:
            The tuple size.
        """
        return size

    @always_inline
    fn __getitem__[idx: Int](self) -> Int:
        """Gets an element from the tuple by index.

        Parameters:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return int(self.data.__getitem__[idx]())

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Gets an element from the tuple by index.

        Args:
            idx: The element index.

        Returns:
            The tuple element value.
        """
        return int(self.data[idx])

    @always_inline("nodebug")
    fn __setitem__[index: Int](mut self, val: Int):
        """Sets an element in the tuple at the given static index.

        Parameters:
            index: The element index.

        Args:
            val: The value to store.
        """
        self.data.__setitem__[index](val)

    @always_inline("nodebug")
    fn __setitem__[index: Int](mut self, val: Self._int_type):
        """Sets an element in the tuple at the given static index.

        Parameters:
            index: The element index.

        Args:
            val: The value to store.
        """
        self.data.__setitem__[index](val)

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, val: Int):
        """Sets an element in the tuple at the given index.

        Args:
            idx: The element index.
            val: The value to store.
        """
        self.data[idx] = val

    @always_inline("nodebug")
    fn as_tuple(self) -> StaticTuple[Int, size]:
        """Converts this IndexList to StaticTuple.

        Returns:
            The corresponding StaticTuple object.
        """
        var res = StaticTuple[Int, size]()

        @parameter
        for i in range(size):
            res[i] = int(self.__getitem__[i]())
        return res

    @always_inline("nodebug")
    fn canonicalize(
        self,
        out result: IndexList[
            size, element_bitwidth = bitwidthof[Int](), unsigned=False
        ],
    ):
        """Canonicalizes the IndexList.

        Returns:
            Canonicalizes the object.
        """
        return self.cast[
            element_bitwidth = result.element_bitwidth,
            unsigned = result.unsigned,
        ]()

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
    fn __add__(self, rhs: Self) -> Self:
        """Performs element-wise integer add.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            return a + b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Performs element-wise integer subtract.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            return a - b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        """Performs element-wise integer multiply.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            return a * b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        """Performs element-wise integer floor division.

        Args:
            rhs: The elementwise divisor.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            return a // b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

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
    fn remu(self, rhs: Self) -> Self:
        """Performs element-wise integer unsigned modulo.

        Args:
            rhs: Right hand side operand.

        Returns:
            The resulting index tuple.
        """

        @always_inline
        fn apply_fn[
            type: DType
        ](a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
            return a % b

        return _int_tuple_binary_apply[apply_fn](self, rhs)

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares this tuple to another tuple for equality.

        The tuples are equal if all corresponding elements are equal.

        Args:
            rhs: The other tuple.

        Returns:
            The comparison result.
        """

        @always_inline
        fn apply_fn[type: DType](a: Scalar[type], b: Scalar[type]) -> Bool:
            return a == b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
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
    fn __lt__(self, rhs: Self) -> Bool:
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
        fn apply_fn[type: DType](a: Scalar[type], b: Scalar[type]) -> Bool:
            return a < b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __le__(self, rhs: Self) -> Bool:
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
        fn apply_fn[type: DType](a: Scalar[type], b: Scalar[type]) -> Bool:
            return a <= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __gt__(self, rhs: Self) -> Bool:
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
        fn apply_fn[type: DType](a: Scalar[type], b: Scalar[type]) -> Bool:
            return a > b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @always_inline
    fn __ge__(self, rhs: Self) -> Bool:
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
        fn apply_fn[type: DType](a: Scalar[type], b: Scalar[type]) -> Bool:
            return a >= b

        return _bool_tuple_reduce[_reduce_and_fn](
            _int_tuple_compare[apply_fn](self.data, rhs.data), True
        )

    @no_inline
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

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this int tuple to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        # TODO: Optimize this to avoid the intermediate String allocation.
        writer.write(str(self))

    @always_inline
    fn cast[
        type: DType
    ](
        self,
        out result: IndexList[
            size,
            element_bitwidth = bitwidthof[type](),
            unsigned = _is_unsigned[type](),
        ],
    ):
        """Casts to the target DType.

        Parameters:
            type: The type to cast towards.

        Returns:
            The list casted to the target type.
        """
        constrained[type.is_integral(), "the target type must be integral"]()

        var res = __type_of(result)()

        @parameter
        for i in range(size):
            res.data[i] = rebind[__type_of(result.data).element_type](
                rebind[Scalar[Self.element_type]](
                    self.data.__getitem__[i]()
                ).cast[result.element_type]()
            )
        return res

    @always_inline
    fn cast[
        *,
        element_bitwidth: Int = Self.element_bitwidth,
        unsigned: Bool = Self.unsigned,
    ](
        self,
        out result: IndexList[
            size, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
    ):
        """Casts to the target DType.

        Parameters:
            element_bitwidth: The bitwidth to cast towards.
            unsigned: The signess of the list.

        Returns:
            The list casted to the target type.
        """

        @parameter
        if (
            element_bitwidth == Self.element_bitwidth
            and unsigned == Self.unsigned
        ):
            return rebind[__type_of(result)](self)

        return rebind[__type_of(result)](
            self.cast[_type_of_width[element_bitwidth, unsigned]()]()
        )


# ===-----------------------------------------------------------------------===#
# Factory functions for creating index.
# ===-----------------------------------------------------------------------===#
@always_inline
fn Index[
    T0: Intable, //,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    x: T0,
    out result: IndexList[
        1, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 1-D Index from the given value.

    Parameters:
        T0: The type of the 1st argument.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x))


@always_inline
fn Index[
    *, element_bitwidth: Int = bitwidthof[Int](), unsigned: Bool = False
](
    x: UInt,
    out result: IndexList[
        1, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 1-D Index from the given value.

    Parameters:
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable, //,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    x: T0,
    y: T1,
    out result: IndexList[
        2, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 2-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x), int(y))


@always_inline
fn Index[
    *, element_bitwidth: Int = bitwidthof[Int](), unsigned: Bool = False
](
    x: UInt,
    y: UInt,
    out result: IndexList[
        2, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 2-D Index from the given values.

    Parameters:
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x), int(y))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable, //,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    x: T0,
    y: T1,
    z: T2,
    out result: IndexList[
        3, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 3-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x), int(y), int(z))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable,
    T3: Intable, //,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    x: T0,
    y: T1,
    z: T2,
    w: T3,
    out result: IndexList[
        4, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 4-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x), int(y), int(z), int(w))


@always_inline
fn Index[
    T0: Intable,
    T1: Intable,
    T2: Intable,
    T3: Intable,
    T4: Intable, //,
    *,
    element_bitwidth: Int = bitwidthof[Int](),
    unsigned: Bool = False,
](
    x: T0,
    y: T1,
    z: T2,
    w: T3,
    v: T4,
    out result: IndexList[
        5, element_bitwidth=element_bitwidth, unsigned=unsigned
    ],
):
    """Constructs a 5-D Index from the given values.

    Parameters:
        T0: The type of the 1st argument.
        T1: The type of the 2nd argument.
        T2: The type of the 3rd argument.
        T3: The type of the 4th argument.
        T4: The type of the 5th argument.
        element_bitwidth: The bitwidth of the underlying integer element type.
        unsigned: Whether the integer is signed or unsigned.

    Args:
        x: The 1st initial value.
        y: The 2nd initial value.
        z: The 3rd initial value.
        w: The 4th initial value.
        v: The 5th initial value.

    Returns:
        The constructed IndexList.
    """
    return __type_of(result)(int(x), int(y), int(z), int(w), int(v))


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn product[size: Int](tuple: IndexList[size, **_], end_idx: Int) -> Int:
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
](tuple: IndexList[size, **_], start_idx: Int, end_idx: Int) -> Int:
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
