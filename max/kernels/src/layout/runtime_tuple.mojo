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
"""
Provides the `RuntimeTuple` data structure and related utility functions
for handling tuple-like data with both compile-time and runtime elements.
`RuntimeTuple` is designed for high-performance tensor operations, supporting
efficient manipulation of multi-dimensional data structures like shapes, indices,
and coordinates.

Key features:
- Hybrid compile-time/runtime value handling
- Optimized for parallel execution and hardware acceleration
- Support for nested tuple structures
- Efficient conversion between linear indices and multi-dimensional coordinates
- Specialized operations for tensor shape calculations

The module includes functions for tuple manipulation (concatenation, flattening),
coordinate transformations (`idx2crd`, `crd2idx`), and specialized tensor operations
like shape division and prefix products.
"""

from std.os import abort

from std.builtin.dtype import _int_type_of_width, _uint_type_of_width
from layout.int_tuple import UNKNOWN_VALUE, IntTuple, flatten
from layout.int_tuple import idx2crd as idx2crd_int_tuple
from layout.int_tuple import prefix_product as prefix_product_int_tuple
from layout.int_tuple import shape_div as shape_div_int_tuple
from layout.int_tuple import product as product_int_tuple
from std.utils import IndexList


def concat(var lhs: IntTuple, rhs: IntTuple) -> IntTuple:
    """Concatenates two `IntTuple` instances into a single `IntTuple`.

    This function appends all elements from the right-hand side tuple to the
    left-hand side tuple, creating a new combined tuple. The operation preserves
    the hierarchical structure of both tuples.

    Args:
        lhs: The left-hand side `IntTuple` that will be modified (var parameter).
        rhs: The right-hand side `IntTuple` whose elements will be appended.

    Returns:
        A new `IntTuple` containing all elements from both tuples in sequence.
    """
    for i in range(len(rhs)):
        lhs.append(rhs[i])
    return lhs.owned_copy()


def _get_returned_type[bitwidth: Int, unsigned: Bool]() -> DType:
    comptime if unsigned:
        return _uint_type_of_width[bitwidth]()

    return _int_type_of_width[bitwidth]()


struct RuntimeTuple[
    S: IntTuple = UNKNOWN_VALUE, /, *, element_type: DType = .int64
](Defaultable, Intable, Sized, TrivialRegisterPassable, Writable):
    """A struct representing tuple-like data with compile-time and runtime elements.
    RuntimeTuple combines static (compile-time) and dynamic (runtime) handling of
    tuple-like data structures, typically used for tensor shapes, indices, and coordinates
    in high-performance computing contexts.
    This struct is optimized for parallel execution and hardware acceleration, allowing
    efficient manipulation of multi-dimensional data. It supports both known compile-time
    values and runtime-determined values.

    Parameters:
        S: `IntTuple` with compile-time known values (or `UNKNOWN_VALUE` for runtime values).
        element_type: Integer type of the underlying elements.
    """

    comptime scalar_length = len(flatten(Self.S))
    """The total number of scalar elements in this RuntimeTuple after flattening nested tuples."""

    var value: IndexList[Self.scalar_length, element_type=Self.element_type]
    """Storage for the actual tuple values, implemented as an IndexList with the appropriate size and element type."""

    @always_inline
    def __init__(out self):
        """Initialize a `RuntimeTuple` with default values.

        For dimensions with known compile-time values in S, uses those values.
        For unknown dimensions, initializes them to UNKNOWN_VALUE.
        """
        self.value = {}

        comptime f = flatten(Self.S)

        comptime for i in range(Self.scalar_length):
            comptime v = f[i].value()

            comptime if v != UNKNOWN_VALUE:
                self.value[i] = v
            else:
                self.value[i] = UNKNOWN_VALUE

    @always_inline
    def __init__(out self, *values: Int):
        """Initialize a `RuntimeTuple` with the provided values.

        Args:
            values: Variadic number of integer values to initialize the tuple with.
        """
        self.value = type_of(self.value)(*values)

    @always_inline
    @implicit
    def __init__[l: Int](out self, values: IndexList[l, ...]):
        """Initialize a `RuntimeTuple` from an `IndexList`.

        Parameters:
            l: Compile-time length of the input `IndexList`.

        Args:
            values: `IndexList` to initialize from. Must have same length as the `RuntimeTuple`.
                    The values will be cast to the appropriate element type if needed.
        """
        comptime assert Self.scalar_length == l, String(
            "Must use same tuple length, expected ",
            Self.scalar_length,
            " but got ",
            l,
        )
        self.value = rebind[type_of(self.value)](
            values.cast[type_of(values).element_type]()
        )

    @staticmethod
    @always_inline
    def offset_until[i: Int]() -> Int:
        """Calculates the offset in the flattened value array for a given tuple index.

        This method computes the sum of lengths of all flattened subtuple elements
        that come before the specified index, which is used for indexing into the
        internal storage.

        Parameters:
            i: The tuple index to calculate the offset for.

        Returns:
            The offset in the flattened array where the i-th element begins.
        """
        var result = 0

        comptime for j in range(i):
            result += len(flatten(Self.S[j]))
        return result

    @always_inline
    def get_int(self) -> Scalar[Self.element_type]:
        """Returns the integer value of this RuntimeTuple.

        For tuples with a known compile-time value, returns that value.
        For tuples with a runtime value, returns the first element of the
        internal storage array.

        Returns:
            The integer value of this RuntimeTuple.
        """
        comptime comptime_value = Scalar[Self.element_type](Self.S.value())

        comptime if comptime_value != UNKNOWN_VALUE:
            return comptime_value
        else:
            return Scalar[Self.element_type](self.value[0])

    @always_inline
    def __getitem_param__[
        i: Int
    ](self, out res: RuntimeTuple[Self.S[i], element_type=Self.element_type]):
        """Retrieves the element at the specified index in the tuple.

        This method provides array-like indexing for RuntimeTuple, allowing access
        to individual elements or sub-tuples. It handles the internal offset calculation
        to access the correct elements in the flattened storage array.

        Parameters:
            i: The index of the element to retrieve.

        Returns:
            A new `RuntimeTuple` containing the element or sub-tuple at the specified index.
        """
        comptime offset = Self.offset_until[i]()
        res = {}

        comptime for i in range(res.scalar_length):
            res.value[i] = self.value[i + offset]

    @always_inline
    def concat[
        R: IntTuple
    ](
        self,
        rhs: RuntimeTuple[R, element_type=Self.element_type],
        out result: RuntimeTuple[
            concat(Self.S, R), element_type=Self.element_type
        ],
    ):
        """Concatenates two `RuntimeTuple`s together.

        This method combines the current `RuntimeTuple` with another one, preserving
        both compile-time and runtime values. It handles the complexity of merging
        the underlying storage arrays while maintaining the proper semantic structure.

        Parameters:
            R: The `IntTuple` type parameter of the right-hand side RuntimeTuple.

        Args:
            rhs: The `RuntimeTuple` to concatenate to the end of this one.

        Returns:
            A new `RuntimeTuple` containing all elements from both tuples in sequence.
        """
        result = {}

        comptime S_flat = flatten(Self.S)

        comptime for i in range(Self.scalar_length):
            comptime if S_flat[i] == UNKNOWN_VALUE:
                result.value[i] = self.value[i]

        comptime R_flat = flatten(R)

        comptime for i in range(rhs.scalar_length):
            comptime if R_flat[i] == UNKNOWN_VALUE:
                result.value[Self.scalar_length + i] = rhs.value[i]

    @always_inline
    def flatten(
        self,
        out result: RuntimeTuple[
            flatten(Self.S), element_type=Self.element_type
        ],
    ):
        """Flattens a potentially nested `RuntimeTuple` into a single-level tuple.
        This method converts a hierarchical structure of tuples into a flat representation,
        preserving all values but removing the nested structure. This is useful for
        operations that need to treat all elements uniformly.

        Returns:
            A new `RuntimeTuple` containing all elements in a flat (non-nested) structure.
        """
        return {self.value}

    def write_to(self, mut writer: Some[Writer]):
        """Writes the RuntimeTuple to a Writer object.

        This method is used by the string conversion system to generate a string
        representation of the RuntimeTuple. It handles both scalar values and
        nested tuple structures, producing a properly formatted output.

        Args:
            writer: The Writer object to write the string representation to.
        """

        comptime if Self.S.is_value():
            writer.write(self.value[0])
        else:
            writer.write("(")

            comptime size = len(Self.S)

            comptime for i in range(size):
                self[i].write_to(writer)

                comptime if i != size - 1:
                    writer.write(", ")
            writer.write(")")

    @always_inline
    def __len__(self) -> Int:
        """Returns the length (number of top-level elements) of the `RuntimeTuple`.

        This method provides the standard Python-like len() functionality,
        giving the number of elements at the top level of the tuple structure.
        For nested tuples, this returns the number of first-level entries,
        not the total number of scalar values.

        Returns:
            The number of top-level elements in the tuple.
        """
        comptime l = len(Self.S)
        return l

    @always_inline
    def cast[
        dtype: DType
    ](self, out result: RuntimeTuple[Self.S, element_type=dtype]):
        """Casts the RuntimeTuple to use a different numeric type.
        This method creates a new RuntimeTuple with the same structure and values
        but using a different underlying numeric type for storage. This is useful
        for changing precision or signedness of the data.

        Parameters:
            dtype: The target DType to cast the elements to.

        Returns:
            A new `RuntimeTuple` with elements cast to the specified type.
        """
        return {self.value.cast[dtype]()}

    @always_inline
    def __int__(self) -> Int:
        """Converts the RuntimeTuple to an integer value.

        This method enables implicit conversion of a RuntimeTuple to an integer,
        but is constrained to only work on scalar tuples (those that contain a single value).

        Returns:
            The integer value of the tuple.
        """
        comptime assert Self.S.is_value(), "tuple must be a single int value"
        return self.value[0]


def is_tuple[t: IntTuple](tuple: RuntimeTuple[t, ...]) -> Bool:
    """Determines if a `RuntimeTuple` represents a tuple rather than a scalar value.

    This function checks the structure of the underlying IntTuple to determine
    if it represents a tuple with multiple elements or a single scalar value.

    Parameters:
        t: The IntTuple type parameter of the RuntimeTuple.

    Args:
        tuple: The `RuntimeTuple` to check.

    Returns:
        True if the `RuntimeTuple` represents a tuple, False if it represents a scalar.
    """
    return t.is_tuple()


def is_int[t: IntTuple](tuple: RuntimeTuple[t, ...]) -> Bool:
    """Determines if a `RuntimeTuple` represents a scalar integer value.

    This function checks if the `RuntimeTuple` holds a single scalar value
    rather than a tuple structure with multiple elements.

    Parameters:
        t: The IntTuple type parameter of the RuntimeTuple.

    Args:
        tuple: The `RuntimeTuple` to check.

    Returns:
        True if the `RuntimeTuple` represents a scalar integer, False otherwise.
    """
    return t.is_value()


@always_inline
def prefix_product[
    t: IntTuple
](tuple: RuntimeTuple[t, ...]) -> RuntimeTuple[prefix_product_int_tuple(t)]:
    """Computes the prefix products of elements in the `RuntimeTuple`.

    This function calculates the running product of elements, where each
    output element is the product of all previous elements in the input.
    This is commonly used in tensor computations to calculate stride values.

    Parameters:
        t: The IntTuple type parameter of the input RuntimeTuple.

    Args:
        tuple: The input `RuntimeTuple`.

    Returns:
        A new `RuntimeTuple` containing the prefix products of the input elements.
    """
    var res = RuntimeTuple[prefix_product_int_tuple(t)]()
    var prefix_res = 1
    for i in range(tuple.scalar_length):
        res.value[i] = prefix_res
        prefix_res *= tuple.value[i]
    return res


@always_inline
def product[t: IntTuple](tuple: RuntimeTuple[t, ...]) -> Int:
    """Computes the product of all elements in the `RuntimeTuple`.

    This function multiplies all scalar values in the tuple, including
    those in nested tuples after flattening. This is commonly used to
    calculate the total size of a tensor from its shape.

    Parameters:
        t: The IntTuple type parameter of the input RuntimeTuple.

    Args:
        tuple: The input `RuntimeTuple`.

    Returns:
        The product of all scalar elements in the tuple.
    """
    var res: Int = 1

    comptime for i in range(tuple.scalar_length):
        res *= tuple.value[i]
    return res


@always_inline
def idx2crd[
    idx_t: IntTuple,
    shape_t: IntTuple,
    stride_t: IntTuple,
](
    idx: RuntimeTuple[idx_t, ...],
    shape: RuntimeTuple[shape_t, ...],
    stride: RuntimeTuple[stride_t, ...],
    out result: RuntimeTuple[
        idx2crd_int_tuple(idx_t, shape_t, stride_t),
        element_type=shape.element_type,
    ],
):
    """Converts a linear index to multi-dimensional coordinates.
    This function transforms a flat index into coordinate values based on
    the provided shape and stride information. This is essential for
    mapping linear memory accesses to multi-dimensional tensor elements.

    Parameters:
        idx_t: IntTuple type of the index.
        shape_t: IntTuple type of the shape.
        stride_t: IntTuple type of the stride.

    Args:
        idx: The linear index to convert.
        shape: The shape of the multi-dimensional array.
        stride: The stride values for each dimension.

    Returns:
        A `RuntimeTuple` containing the multi-dimensional coordinates.

    Constraints:
        The index must be a scalar value (not a tuple).
    """

    comptime assert idx_t.is_value(), "Only scalar index is supported"

    result = {}

    comptime for i in range(result.scalar_length):
        result.value[i] = (Int(idx) // stride.value[i]) % shape.value[i]


# take shape as return type
@always_inline
def idx2crd[
    idx_t: IntTuple,
    shape_t: IntTuple,
](
    idx: RuntimeTuple[idx_t, ...],
    shape: RuntimeTuple[shape_t, ...],
) -> RuntimeTuple[
    idx2crd_int_tuple(idx_t, shape_t, prefix_product_int_tuple(shape_t)),
    element_type=shape.element_type,
]:
    """Converts a linear index to multi-dimensional coordinates using shape-derived strides.
    This is a convenience overload of `idx2crd` that automatically calculates the stride
    values from the shape using `prefix_product`. This is the common case for row-major
    storage order tensors.

    Parameters:
        idx_t: IntTuple type of the index.
        shape_t: IntTuple type of the shape.

    Args:
        idx: The linear index to convert.
        shape: The shape of the multi-dimensional array.

    Returns:
        A `RuntimeTuple` containing the multi-dimensional coordinates calculated using
        automatically derived strides from the shape.
    """
    return idx2crd(idx, shape, prefix_product(shape))


def crd2idx[
    crd_t: IntTuple,
    shape_t: IntTuple,
    stride_t: IntTuple,
    out_type: DType = .uint64,
](
    crd: RuntimeTuple[crd_t, ...],
    shape: RuntimeTuple[shape_t, ...],
    stride: RuntimeTuple[stride_t, ...],
) -> Scalar[out_type]:
    """Converts multi-dimensional coordinates to a linear index.

    This function is the inverse of idx2crd, transforming a set of coordinates
    into a flat index based on the provided shape and stride information.
    This is essential for mapping multi-dimensional tensor elements to linear memory.

    Parameters:
        crd_t: Type of the coordinates.
        shape_t: Type of the shape.
        stride_t: Type of the stride.
        out_type: The output data type for the index (default: uint64).

    Args:
        crd: The coordinates to convert.
        shape: The shape of the multi-dimensional array.
        stride: The stride values for each dimension.

    Returns:
        A scalar value representing the linear index corresponding to the given coordinates.
    """

    comptime if crd_t.is_tuple():
        comptime assert shape_t.is_tuple() and (
            len(crd_t) == len(shape_t) == len(stride_t)
        ), String(
            "Inputs should have same rank but got crd_t: ",
            len(crd_t),
            " shape_t: ",
            len(shape_t),
            " stride_t: ",
            len(stride_t),
        )
        var r: Scalar[out_type] = 0
        comptime size = min(min(len(crd_t), len(shape_t)), len(stride_t))

        comptime for i in range(size):
            r += crd2idx[out_type=out_type](crd[i], shape[i], stride[i])
        return r
    else:
        var int_crd: Scalar[out_type] = Scalar[out_type](
            0 if len(crd) == 0 else Int(crd)
        )

        comptime if shape_t.is_tuple():  # "int" tuple tuple
            comptime assert len(shape_t) == len(
                stride_t
            ), "shape and stride should have same rank"
            var result: Scalar[out_type] = 0

            comptime last_elem_idx = len(shape_t) - 1

            comptime for i in range(last_elem_idx):
                var divisor, quotient = divmod(Int(int_crd), product(shape[i]))
                result += crd2idx[crd_t, out_type=out_type](
                    RuntimeTuple[crd_t, ...](quotient), shape[i], stride[i]
                )
                int_crd = Scalar[out_type](divisor)
            # FIXME(KERN-640): Replace with [-1], currently not giving correct result.
            return result + crd2idx[crd_t, out_type=out_type](
                RuntimeTuple[crd_t, ...](Int(int_crd)),
                shape[last_elem_idx],
                stride[last_elem_idx],
            )
        else:  # "int" "int" "int"
            return int_crd * Scalar[out_type](Int(stride))


# TODO: This isn't necessarily needed. We need to revisit and simplify
# the implementation. We are keeping it here to be consistent with IntTuple
# shape_div.
@always_inline
def signum(a: Int) -> Int:
    """Returns the sign of an integer value.

    This helper function determines whether a number is positive, negative, or zero,
    returning 1 for positive, -1 for negative, and 0 for zero.

    Args:
        a: The integer value to determine the sign of.

    Returns:
        1 if a > 0, -1 if a < 0, 0 if a == 0.
    """
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


def shape_div[
    a_t: IntTuple, b_t: IntTuple
](
    a: RuntimeTuple[a_t, ...],
    b: RuntimeTuple[b_t, ...],
    out result: RuntimeTuple[shape_div_int_tuple(a_t, b_t)],
):
    """Performs specialized shape division between `RuntimeTuple`s.

    This function implements a special division operation specifically designed for
    tensor shape calculations. Unlike standard division, it handles special cases:

    1. If shapes are directly divisible (a % b == 0), returns a standard division (a // b)
    2. If shapes are inversely divisible (b % a == 0), returns the signed reciprocal
    3. If shapes are incompatible, aborts with an error

    This operation is essential for transformations between tensor layouts and computing
    broadcasting semantics.

    Parameters:
        a_t: Type of the first operand.
        b_t: Type of the second operand.

    Args:
        a: The dividend `RuntimeTuple`.
        b: The divisor `RuntimeTuple`.

    Returns:
        A new `RuntimeTuple` containing the result of the shape division.
    """

    comptime if a_t.is_tuple():
        comptime if b_t.is_tuple():
            comptime assert len(a_t) == len(
                b_t
            ), "shape and stride length musth match"
            var res = RuntimeTuple[shape_div_int_tuple(a_t, b_t)]()

            comptime for i in range(len(a_t)):
                var res_i = shape_div(a[i], b[i])

                comptime for j in range(res_i.scalar_length):
                    res.value[i + j] = res_i.value[j]
            return res
        else:
            var res = RuntimeTuple[shape_div_int_tuple(a_t, b_t)]()
            # FIXME: this used to be simpler
            # var vb = Int(to_int(b))
            var vb = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](Int(b))

            comptime for i in range(len(a_t)):
                var res_i = shape_div(a[i], vb)

                comptime for j in range(res_i.scalar_length):
                    res.value[i + j] = res_i.value[j]

                # FIXME: this used to be simpler
                # vb = Int(to_int(shape_div(vb, product(a[i]))))
                var t = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](product(a[i]))
                vb = {Int(shape_div(vb, t))}
            return res
    else:
        comptime if b_t.is_tuple():
            return shape_div(a, b)
        else:
            var va = Int(a)
            var vb = Int(b)

            if not (va % vb == 0 or vb % va == 0):
                abort(t"Incompatible shape values: {va} {vb}")

            return {va // vb if va % vb == 0 else signum(va * vb)}


def to_index_list[
    rank: Int, t: IntTuple
](tuple: RuntimeTuple[t, ...]) -> IndexList[rank]:
    """
    Converts a RuntimeTuple to an IndexList with the same values.

    Parameters:
        rank: The rank of the resulting IndexList.
        t: The IntTuple template parameter of the RuntimeTuple.

    Args:
        tuple: The RuntimeTuple to convert.

    Returns:
        An IndexList filled with the values of the RuntimeTuple.
    """
    var res = IndexList[rank]()
    for i in range(rank):
        res[i] = tuple.value[i]
    return res


def _int_tuple_product_flatten[t: IntTuple]() -> IntTuple:
    comptime rank = len(t)
    var tup = IntTuple(num_elems=rank)

    comptime for i in range(rank):
        comptime known = t[i].all_known()

        comptime if known:
            comptime product = product_int_tuple(t[i])
            tup.replace_entry(i, int_value=product)
        else:
            tup.replace_entry(i, int_value=UNKNOWN_VALUE)

    return tup


def coalesce_nested_tuple[
    t: IntTuple,
    out_t: IntTuple = _int_tuple_product_flatten[t](),
](tuple: RuntimeTuple[t, ...]) -> RuntimeTuple[out_t]:
    """Coalesces a nested `RuntimeTuple` into a single-level `RuntimeTuple`, by multiplying all the
    values together.

    Parameters:
        t: The underlying Compile-time IntTuple backing the RuntimeTuple.
        out_t: The flattened Compile-time IntTuple.

    Args:
        tuple: The RuntimeTuple to convert.

    Returns:
        A new `IntTuple` containing the products of each top level tuple, in a flat structure.
    """

    comptime rank = len(out_t)
    var idxs = IndexList[rank]()

    comptime for i in range(rank):
        comptime known = out_t[i].all_known()

        comptime if known:
            idxs[i] = out_t[i].value()
        else:
            var slice = tuple[i]
            var product = 1

            comptime assert slice.scalar_length > 0, "Slice is empty"

            comptime for j in range(slice.scalar_length):
                product *= Int(slice[j])

            idxs[i] = product

    return RuntimeTuple[out_t](idxs)
