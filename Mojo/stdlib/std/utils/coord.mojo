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
"""Unified layout system for mixed compile-time and runtime indices."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.os import abort
from std.reflection import reflect
from std.collections import Array
from std.utils import IndexList
from std.math.uutils import umod, ufloordiv


trait CoordLike(
    Defaultable, ImplicitlyCopyable, TrivialRegisterPassable, Writable
):
    """Trait for unified layout handling of compile-time and runtime indices."""

    comptime _ParamListType: TypeList[Trait=CoordLike, _]._mlir_type
    """The low-level parameter list of element types."""

    comptime ParamListType: TypeList[Trait=CoordLike, Self._ParamListType]
    """The type list of element types for coordinates."""

    comptime static_value: Int
    """The compile-time value if statically known, -1 otherwise."""

    comptime is_static_value = False
    """True if the value is known at compile time."""

    comptime is_tuple = False
    """True if this is a tuple type (`Coord`), False for scalar values."""

    comptime is_value = not Self.is_tuple
    """True if this is a scalar value, False for tuple types."""

    comptime DTYPE = DType.int
    """The data type used by scalar-returning coordinate operations."""

    # Note that unlike the __len__() from Sized, this is a static method.
    @staticmethod
    def __len__() -> Int:
        """Get the number of elements in this type.

        Returns:
            The number of elements (1 for single values, >1 for tuples).
        """
        ...

    def value(self) -> Scalar[Self.DTYPE]:
        """Get the scalar value of this coordinate.

        Only valid for value types (not tuples).

        Returns:
            The scalar value.
        """
        ...

    def tuple(var self) -> Coord[*Self.ParamListType]:
        """Get this coordinate as a `Coord` tuple.

        Only valid for tuple types.

        Returns:
            The coordinate as a `Coord` tuple.
        """
        ...

    def product(self) -> Scalar[Self.DTYPE]:
        """Calculate the product of all elements.

        Returns:
            The product of all elements.
        """
        ...

    def sum(self) -> Scalar[Self.DTYPE]:
        """Calculate the sum of all elements.

        Returns:
            The sum of all elements.
        """
        ...


struct ComptimeInt[val: Int](CoordLike, TrivialRegisterPassable):
    """Compile-time known index value.

    Parameters:
        val: The compile-time integer value.
    """

    comptime ParamListType = Coord[Self].element_types
    """The element types (Self for scalar types)."""

    comptime _ParamListType = Self.ParamListType.values
    """The low-level parameter list of element types."""

    comptime static_value: Int = Self.val
    """The compile-time value."""

    comptime DTYPE = DType.int
    """The data type (int for compile-time integers)."""

    comptime is_static_value = True
    """True, indicating this is a compile-time known value."""

    def __init__(out self):
        """Initialize a compile-time integer with the specified value."""
        pass

    @staticmethod
    @always_inline("nodebug")
    def __len__() -> Int:
        """Get the length (always 1 for scalar types).

        Returns:
            Always returns 1.
        """
        return 1

    def write_to(self, mut writer: Some[Writer]):
        """Write this compile-time integer to a writer.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.value())

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this compile-time integer to a writer.

        Args:
            writer: The writer to write to.
        """
        t"ComptimeInt[{self.value()}]()".write_to(writer)

    @always_inline("nodebug")
    def product(self) -> Scalar[Self.DTYPE]:
        """Calculate the product (returns the value for scalar types).

        Returns:
            The integer value.
        """
        return self.value()

    @always_inline("nodebug")
    def sum(self) -> Scalar[Self.DTYPE]:
        """Calculate the sum (returns the value for scalar types).

        Returns:
            The integer value.
        """
        return self.value()

    @always_inline("nodebug")
    def value(self) -> Scalar[Self.DTYPE]:
        """Get the scalar value.

        Returns:
            The compile-time integer value.
        """
        return Scalar[Self.DTYPE](Self.val)

    @always_inline("nodebug")
    def tuple(var self) -> Coord[*Self.ParamListType]:
        """Get as a tuple (not valid for `ComptimeInt`).

        Returns:
            Never returns; aborts at compile time.
        """
        comptime assert False, "ComptimeInt is not a tuple type"


comptime Idx[value: Int] = ComptimeInt[value]()
"""A compile-time coordinate index value.

Parameters:
    value: The compile-time integer value.
"""


# ===-----------------------------------------------------------------------===#
# _All — "keep this dimension" marker for TileTensor.__getitem__
# ===-----------------------------------------------------------------------===#


struct _All(CoordLike, TrivialRegisterPassable):
    """Marker type meaning "keep this entire dimension" in a select operation.

    Used with `TileTensor.__getitem__` to indicate that a dimension should be
    preserved in the output rather than collapsed at a specific index.
    Uses `static_value = -2` as a sentinel distinct from `Scalar`'s `-1`.

    Example:

    ```
    # From a 4D tensor (batch, N, heads, head_dim),
    # fix batch and heads, keep N and head_dim:
    var result = tensor.slice(batch_idx, All, head_idx, All)
    # result is a 2D tensor (N, head_dim)
    ```
    """

    comptime ParamListType = Coord[Self].element_types
    """The element types (Self for scalar types)."""

    comptime _ParamListType = Self.ParamListType.values
    """The low-level parameter list of element types."""

    comptime static_value: Int = -2
    """Sentinel value distinguishing `_All` from `Scalar` (-1) and `ComptimeInt` (>=0)."""

    comptime DTYPE = DType.int

    comptime is_static_value = True

    def __init__(out self):
        pass

    @staticmethod
    @always_inline("nodebug")
    def __len__() -> Int:
        return 1

    def write_to(self, mut writer: Some[Writer]):
        writer.write("All")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("All")

    @always_inline("nodebug")
    def product(self) -> Scalar[Self.DTYPE]:
        return Scalar[Self.DTYPE](1)

    @always_inline("nodebug")
    def sum(self) -> Scalar[Self.DTYPE]:
        return Scalar[Self.DTYPE](0)

    @always_inline("nodebug")
    def value(self) -> Scalar[Self.DTYPE]:
        return Scalar[Self.DTYPE](-2)

    @always_inline("nodebug")
    def tuple(var self) -> Coord[*Self.ParamListType]:
        comptime assert False, "_All is not a tuple type"


comptime All = _All()
"""Marker value meaning "keep this entire dimension" in `TileTensor.__getitem__`."""


@fieldwise_init("implicit")
struct Coord[*element_types: CoordLike](
    CoordLike, DevicePassable, Sized, Writable
):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The list of element types that implement `CoordLike`.
    """

    comptime device_type = Self
    """Indicate the type being used on accelerator devices.

    A `Coord` holds only plain integer data (zero-sized `ComptimeInt` dims and
    POD `Scalar` leaves), so its host and device layouts match and it transfers
    by a straight bit-copy, just like `IndexList`."""

    comptime ParamListType = Self.element_types
    """The element types of this `Coord`."""

    comptime _ParamListType = Self.element_types.values
    """The element types (Self for scalar types)."""

    comptime static_value: Int = -1
    """Always -1 for tuple types (value not applicable)."""

    # TODO(GPUA-11): Expand `Coord.DTYPE` so that it can take narrower dtypes.
    comptime DTYPE = DType.int
    """The scalar dtype used for tuple-level aggregate operations."""

    comptime is_tuple = True
    """True, indicating this is a tuple type."""

    comptime all_dims_known = _AllStatic[*Self.element_types]
    """True if all dimensions are statically known at compile time."""

    comptime static_product = _StaticProduct[*Self.element_types]
    """The product of all static dimensions, or -1 if any are dynamic."""

    comptime rank = Self.element_types.length
    """The number of top-level elements in this `Coord`."""

    comptime flat_rank = _Flattened[*Self.element_types].length
    """The total number of leaf elements after flattening nested `Coord`s."""

    comptime is_flat = Self.rank == Self.flat_rank
    """If the `Coord` contains nested items."""

    comptime contains_slices = Self.element_types.contains[type_of(All)]()
    """If the `Coord` contains the `All` symbol."""

    var _storage: _RegTuple[*Self.element_types]
    """The underlying MLIR storage for the tuple elements."""

    def __init__(out self):
        """
        Empty initialize a tensor with static dims.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        comptime for i in range(self.rank):
            self[i] = Self.element_types[i]()

    def __init__[
        rank: Int, dtype: DType
    ](
        out self: Coord[
            *TypeList.splat[Trait=CoordLike, rank, Scalar[dtype]]()
        ],
        index_list: IndexList[rank, element_type=dtype],
    ):
        """Construct a `Coord` from an `IndexList`.

        Parameters:
            rank: The number of elements in the index list.
            dtype: The data type of the index list elements.

        Args:
            index_list: The `IndexList` to convert to a `Coord`.
        """
        self = type_of(self)()

        comptime for i in range(rank):
            Pointer(to=self[i]).write(
                rebind[type_of(self[i])](Scalar[dtype](index_list[i]))
            )

    def __init__[
        rank: Int, dtype: DType
    ](
        out self: Coord[
            *TypeList.splat[Trait=CoordLike, rank, Scalar[dtype]]()
        ],
        array: Array[Scalar[dtype], rank],
    ):
        """Construct an all-dynamic `Coord` from an `Array`.

        Parameters:
            rank: The number of elements in the array.
            dtype: The data type of the array elements.

        Args:
            array: The `Array` to convert to a `Coord`.
        """
        self = type_of(self)()

        comptime for i in range(rank):
            Pointer(to=self[i]).write(rebind[type_of(self[i])](array[i]))

    @staticmethod
    @always_inline("nodebug")
    def size() -> Int:
        """Get the total number of elements including nested ones.

        Returns:
            The total count of all elements.
        """
        var count = 0

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            count += T.__len__()

        return count

    @staticmethod
    def __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return Self.element_types.length

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `Coord` to a writer.

        Args:
            writer: The writer to write the representation to.
        """
        t"Coord({self})".write_to(writer)

    def __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return Self.__len__()

    @always_inline("nodebug")
    def __init__(out self, var *args: *Self.element_types):
        """Construct tuple from variadic arguments.

        Args:
            args: Values for each element.
        """
        self._storage = _RegTuple[*Self.element_types](*args^)

    @implicit
    @always_inline("nodebug")
    def __init__(out self, var tuple: Tuple[*Self.element_types]):
        """Construct from a Tuple with matching element types.

        Args:
            tuple: The Tuple to construct from.
        """
        self = Self()

        comptime for i in range(Self.rank):
            self._storage[i] = tuple[i]

    @always_inline("nodebug")
    def __getitem_param__[
        idx: Int
    ](ref self) -> ref[self._storage] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        return self._storage[idx]

    @always_inline("nodebug")
    def product(self) -> Scalar[Self.DTYPE]:
        """Calculate the product of all elements recursively.

        Returns:
            The product of all leaf values in the `Coord`.
        """
        var result: Scalar[Self.DTYPE] = 1

        # `Coord` is a heterogeneous tuple: children may have different
        # `DTYPE`s (e.g. `CompileTimeInt` with `DType.int` alongside a
        # `Int32`). Aggregating into `Self.DTYPE` is
        # intentional — callers expect a single integer answer at the
        # tuple's dtype, regardless of per-leaf dtype.
        # TODO(GPUA-12): Add comptime asserts to make sure that Coord's
        # product and sum do not overflow.
        comptime for i in range(Self.__len__()):
            result *= Scalar[Self.DTYPE](self[i].product())

        return result

    @always_inline("nodebug")
    def sum(self) -> Scalar[Self.DTYPE]:
        """Calculate the sum of all elements recursively.

        Returns:
            The sum of all leaf values in the `Coord`.
        """
        var result: Scalar[Self.DTYPE] = 0

        # See `product()`: aggregating heterogeneous child `DTYPE`s into
        # `Self.DTYPE` is intentional.
        # TODO(GPUA-12): Add comptime asserts to make sure that Coord's
        # product and sum do not overflow.
        comptime for i in range(Self.__len__()):
            result += Scalar[Self.DTYPE](self[i].sum())

        return result

    @always_inline("nodebug")
    def value(self) -> Scalar[Self.DTYPE]:
        """Get the value (not valid for `Coord` tuples).

        Returns:
            Never returns; aborts at compile time.
        """
        comptime assert False, "Coord is not a value type"

    @always_inline("nodebug")
    def inner_product[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Int:
        """Calculate the inner product with another CoordLike.

        Parameters:
            other_types: The types of the other value.

        Args:
            other: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """
        comptime assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )
        var result = 0

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            comptime if T.is_tuple and U.is_tuple:
                result += Coord(self[i]).inner_product(Coord(other[i]))
            elif T.is_value and U.is_value:
                result += Int(self[i].value()) * Int(other[i].value())
            else:
                comptime assert False, String(
                    "Element ",
                    i,
                    " of Coord must both be a tuple or both be a value",
                )

        return result

    @always_inline("nodebug")
    def __eq__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        """Check if this `Coord` equals another.

        Parameters:
            other_types: The element types of the other `Coord`.

        Args:
            other: The `Coord` to compare with.

        Returns:
            True if all elements are equal, False otherwise.
        """

        comptime assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            comptime if T.is_tuple and U.is_tuple:
                if Coord(self[i]) != Coord(other[i]):
                    return False
            elif T.is_value and U.is_value:
                if Int(self[i].value()) != Int(other[i].value()):
                    return False
            else:
                comptime assert False, String(
                    "Element ",
                    i,
                    " of Coord must both be a tuple or both be",
                    " a value",
                )

        return True

    @always_inline("nodebug")
    def __ne__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        """Check if this `Coord` is not equal to another.

        Parameters:
            other_types: The element types of the other `Coord`.

        Args:
            other: The `Coord` to compare with.

        Returns:
            True if any elements differ, False if all are equal.
        """
        return not self == other

    @always_inline("nodebug")
    def tuple(var self) -> Coord[*Self.ParamListType]:
        """Get this `Coord` as a tuple.

        Returns:
            This `Coord` (identity operation for tuple types).
        """
        return rebind[Coord[*Self.ParamListType]](self)

    @always_inline("nodebug")
    def reverse(
        var self,
    ) -> Coord[*Self.element_types.reverse()]:
        """Reverse the order of elements in this `Coord`.

        Returns:
            A new `Coord` with elements in reverse order.
        """
        return {self._storage.reverse()}

    @always_inline("nodebug")
    def concat[
        *other_element_types: CoordLike
    ](var self, var other: Coord[*other_element_types]) -> Coord[
        *TypeList._concat[
            Self.element_types.values, other_element_types.values
        ]()
    ]:
        """Concatenate this `Coord` with another.

        Parameters:
            other_element_types: The element types of the other `Coord`.

        Args:
            other: The `Coord` to append.

        Returns:
            A new `Coord` containing elements from both `Coord`s.
        """
        return {
            rebind[
                _RegTuple[
                    *TypeList._concat[
                        Self.element_types.values,
                        other_element_types.values,
                    ](),
                ]
            ](self._storage.concat(other._storage))
        }

    @always_inline("nodebug")
    def flatten(
        var self,
    ) -> Coord[*_Flattened[*Self.element_types]]:
        """Convert a nested `Coord` to a flattened `Coord`.


        Returns:
            A flattened `Coord` containing all leaf values in order.

        Examples:
            ```mojo
            from layout import Coord, Idx
            var nested = Coord(
                Idx[5],
                Coord(Idx[3], Idx[2]),
                7
            )
            var flat = nested.flatten()
            # flat is Coord(Idx[5], Idx[3], Idx[2], 7)
            ```
        """
        comptime FlatTypes = _Flattened[*Self.element_types]
        comptime flat_size = FlatTypes.length

        var flat_tuple: _RegTuple[*FlatTypes]

        # Mark the tuple as initialized so we can work on it
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(flat_tuple)
        )

        # Use _get_flattened to access each element by flat index
        comptime for i in range(flat_size):
            comptime FlatType = FlatTypes[i]

            comptime if FlatType.is_static_value:
                # Compile-time known value
                Pointer(to=flat_tuple[i]).write(
                    rebind[FlatType](ComptimeInt[FlatType.static_value]())
                )
            else:
                # Runtime value - use _get_flattened to get the value
                var val = _get_flattened[i](self)
                comptime if FlatType == Int:
                    Pointer(to=flat_tuple[i]).write(rebind[FlatType](val))
                else:
                    Pointer(to=flat_tuple[i]).write(
                        rebind[FlatType](Scalar[FlatType.DTYPE](val))
                    )

        return Coord(flat_tuple)

    @always_inline("nodebug")
    def make_dynamic[
        dtype: DType
    ](self) -> Coord[*_CoordToDynamic[dtype, Self.element_types]]:
        """Convert all elements to `Scalar[dtype]`.

        Parameters:
            dtype: The data type for the resulting `Scalar` values.

        Returns:
            A new `Coord` where all elements are converted to `Scalar[dtype]`.

        Examples:
            ```mojo
            from std.utils.coord import Coord, ComptimeInt
            var c = Coord(ComptimeInt[3](), Int32(5), ComptimeInt[7]())
            var dynamic = c.make_dynamic[.int64]()
            # dynamic is Coord(Int64(3), Int64(5), Int64(7))
            ```
        """
        comptime ResultTypes = _CoordToDynamic[dtype, Self.element_types]
        var result: Coord[*ResultTypes]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(Self.__len__()):
            # Convert all elements to Scalar[dtype]
            Pointer(to=result[i]).write(
                rebind[ResultTypes[i]](Scalar[dtype](self[i].value()))
            )

        return result

    def replace[
        T: CoordLike, //, at: Int
    ](var self, value: T) -> Coord[*_CoordReplaceAt[Self.element_types, at, T]]:
        """Replace the element at `at`, keeping the other elements' types.

        A `Coord`'s element types are part of its type, so a statically known
        element has no runtime storage to assign into. Overwriting one with a
        runtime value therefore produces a `Coord` of a different type rather
        than mutating in place. The untouched elements keep their static
        values.

        Parameters:
            T: The replacement element's type.
            at: The index of the element to replace.

        Args:
            value: The replacement element.

        Returns:
            A new `Coord` with `value` at `at`.

        Examples:
            ```mojo
            from std.utils.coord import Coord, ComptimeInt
            var c = Coord(ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]())
            var moved = c.replace[1](Int64(7))
            # moved is Coord(ComptimeInt[3](), Int64(7), ComptimeInt[5]())
            ```
        """
        comptime assert (
            0 <= at < Self.__len__()
        ), "`at` must be a valid element index"
        comptime ResultTypes = _CoordReplaceAt[Self.element_types, at, T]
        var result: Coord[*ResultTypes]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(Self.__len__()):
            comptime if i != at:
                Pointer(to=result[i]).write(rebind[ResultTypes[i]](self[i]))
            else:
                Pointer(to=result[i]).write(rebind[ResultTypes[i]](value))

        return result

    def write_to(self, mut w: Some[Writer]):
        """Write this `Coord` to a `Writer`.

        Args:
            w: The writer to output to.
        """
        w.write("(")

        comptime for i in range(Self.rank):
            comptime if Self.element_types[i].is_tuple:
                self[i].tuple().write_to(w)
            else:
                w.write(self[i].value())

            comptime if i < Self.rank - 1:
                w.write(", ")
        w.write(")")

    @always_inline("nodebug")
    def cast[
        dtype: DType
    ](self) -> Coord[*_CoordCast[dtype, Self.element_types]]:
        """Cast runtime elements to `Scalar[dtype]`.

        Compile-time elements are preserved as-is, so statically known
        dimensions stay static after the cast.

        Parameters:
            dtype: The target data type for runtime elements.

        Returns:
            A new `Coord` where runtime elements use `dtype` and static
            elements keep their original type.
        """
        comptime assert dtype.is_integral(), "the target type must be integral"
        comptime assert Self.is_flat, "`Coord.cast` only supports flat `Coord`s"

        comptime ResultTypes = _CoordCast[dtype, Self.element_types]
        var result: Coord[*ResultTypes]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(Self.__len__()):
            comptime ResultType = ResultTypes[i]
            comptime if ResultType.is_static_value:
                Pointer(to=result[i]).write(rebind[ResultType](self[i]))
            else:
                Pointer(to=result[i]).write(
                    rebind[ResultType](Scalar[dtype](self[i].value()))
                )

        return result

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        NOTE: This should only be called by `DeviceContext` during invocation
        of accelerator kernels.

        Args:
            encoder: The encoder to convert the host type to a device type.
            target: The target address to store the converted device type.
        """
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Get the human-readable type name for this `Coord`.

        This is used for error messages when passing types to the device.

        Returns:
            A string representation of the type, e.g. "Coord[ComptimeInt[2],
            Int64]".
        """
        var result = String("Coord[")

        comptime for i in range(Self.rank):
            comptime if i > 0:
                result += ", "
            result += reflect[Self.element_types[i]].name()

        result += "]"
        return result


# Helper for flat indexing with nested shape/stride.
def _crd2idx_flat[
    out_type: DType,
](crd_t: Coord, shape_t: Coord, stride_t: Coord) -> Scalar[out_type]:
    """Compute index from flat coordinate with nested shape/stride.

    For nested layouts like blocked_product, this computes the linear index
    by flattening the stride and performing element-wise dot product.

    Parameters:
        out_type: Output scalar type.

    Args:
        crd_t: Flat coordinate tuple.
        shape_t: Nested shape tuple.
        stride_t: Nested stride tuple.

    Returns:
        Linear index computed from flat coords and nested shape/stride.
    """
    # Flatten the stride and compute dot product with flat coord. Also
    # compute the flattened shape *types* (without materializing a runtime
    # value) so that each flat coordinate leaf can be checked against its
    # corresponding flattened shape leaf at compile time.
    var flat_stride = stride_t.flatten()
    var result: Scalar[out_type] = 0
    comptime flat_len = type_of(crd_t).__len__()

    # For each flat dimension, if both the coordinate and the corresponding
    # flattened shape are statically known, assert that the coordinate is in
    # bounds. Runtime dims and runtime coords are left unchecked here.
    comptime _FlatCrdTypes = type_of(crd_t).element_types
    comptime _FlatShapeTypes = _Flattened[*type_of(shape_t).element_types]

    comptime for i in range(flat_len):
        comptime if (
            _FlatCrdTypes[i].is_static_value
            and _FlatShapeTypes[i].is_static_value
        ):
            comptime assert (
                0
                <= _FlatCrdTypes[i].static_value
                < _FlatShapeTypes[i].static_value
            ), String(
                t"crd2idx: static coordinate {_FlatCrdTypes[i].static_value} is"
                t" out of bounds for static shape [0,"
                t" {_FlatShapeTypes[i].static_value}) at flat dim {i}"
            )
        # Each operand is narrowed to `out_type` *before* the multiply, so the
        # entire dot-product is computed at `out_type` precision. This is
        # deliberate: on GPUs with narrow address types (e.g. `uint32`) we
        # don't want a hidden 64-bit multiply widening every index calculation.
        # Callers are responsible for choosing an `out_type` wide enough to
        # hold `max(coord) * max(stride)` summed across dims; the default
        # `DType.int64` covers essentially all real layouts.
        result += Scalar[out_type](crd_t[i].value()) * Scalar[out_type](
            flat_stride[i].value()
        )

    return result


# Implementation based off runtime_tuple.mojo's crd2idx.
def crd2idx[
    Index: CoordLike,
    Shape: CoordLike,
    Stride: CoordLike,
    out_type: DType = .int64,
](crd: Index, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the linear index from a coordinate tuple.

    The dot product is computed at `out_type` precision (each `coord * stride`
    term is evaluated after narrowing both operands to `out_type`). Callers
    that pick a narrow `out_type` (e.g. `uint32`) are responsible for ensuring
    every per-dim product and their sum fit in that type. The default
    `DType.int64` is wide enough for realistic layouts. See `_crd2idx_flat`.

    Parameters:
        Index: The coordinate type (must be CoordLike).
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_type: The output scalar type.

    Args:
        crd: The multi-dimensional coordinate.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        The linear index corresponding to the coordinate.
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()
    comptime crd_len = Index.__len__()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        var result: Scalar[out_type] = 0

        comptime if crd_len > 1:  # tuple tuple tuple
            var crd_t = crd.tuple()

            # Check if crd structure matches shape structure
            comptime if crd_len == shape_len:
                # Hierarchical indexing: crd elements map 1:1 to shape elements
                comptime for i in range(shape_len):
                    result += crd2idx[out_type=out_type](
                        crd_t[i], shape_t[i], stride_t[i]
                    )
            else:
                # Flat indexing: crd is flat, need to compute with flattened strides
                # Use _crd2idx_flat which handles flat coords with nested shape/stride
                return _crd2idx_flat[out_type](crd_t, shape_t, stride_t)

            return result
        else:  # "int" tuple tuple
            var crd_int: Int

            comptime if Index.is_tuple:
                crd_int = 0 if crd_len == 0 else Int(crd.tuple()[0].value())
            else:
                crd_int = 0 if crd_len == 0 else Int(crd.value())

            comptime last_elem_idx = shape_len - 1

            comptime for i in range(last_elem_idx):
                var quotient, remainder = divmod(
                    crd_int, Int(shape_t[i].product())
                )
                result += crd2idx[out_type=out_type](
                    remainder, shape_t[i], stride_t[i]
                )
                crd_int = quotient
            return result + crd2idx[out_type=out_type](
                crd_int, shape_t[last_elem_idx], stride_t[last_elem_idx]
            )
    else:
        comptime if crd_len > 1:
            abort("crd is a tuple but shape and stride are not")
        else:
            # Scalar leaf case: if both crd and shape are statically known,
            # assert that the coord is within the shape's bounds at compile
            # time. Runtime values are not checked here.
            comptime if Index.is_static_value and Shape.is_static_value:
                comptime assert (
                    0 <= Index.static_value < Shape.static_value
                ), String(
                    t"crd2idx: static coordinate {Index.static_value} is out of"
                    t" bounds for static shape [0, {Shape.static_value})"
                )
            # Narrow-first multiply: see `_crd2idx_flat` for rationale.
            return Scalar[out_type](crd.value()) * Scalar[out_type](
                stride.value()
            )


# Implementation based off crd2idx - computes the inverse operation.
# Uses the per-element formula: coord[i] = (idx // stride[i]) % shape[i]
def idx2crd[
    Shape: CoordLike,
    Stride: CoordLike,
    out_dtype: DType = .int64,
](idx: Int, shape: Shape, stride: Stride) -> Coord[
    *_Idx2CrdResultTypes[
        out_dtype,
        Scalar[out_dtype],
        Stride.ParamListType,
        Shape.ParamListType,
    ]
]:
    """Calculate the coordinate tuple from a linear index.

    This is the inverse of crd2idx - given a linear index, shape, and stride,
    it computes the multi-dimensional coordinates using the per-element formula:
    ``coord[i] = (idx // stride[i]) % shape[i]``.

    When a shape dimension is statically known to be 1, the corresponding
    output coordinate is a `ComptimeInt[0]`. Otherwise, coordinates are
    `Scalar[out_dtype]`.

    The idx, and all components of shape and stride must be non-negative.

    Parameters:
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_dtype: The output data type for coordinate values.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A `Coord` containing the coordinate values for each dimension.
        Dimensions with static shape 1 produce `ComptimeInt[0]`, others
        produce `Scalar[out_dtype]`.

    Examples:
        For a 2D tensor with shape (3, 4) and row-major strides (4, 1):

        - idx2crd(0, shape, stride) returns (0, 0).
        - idx2crd(5, shape, stride) returns (1, 1).
        - idx2crd(11, shape, stride) returns (2, 3).
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()

    debug_assert(
        shape_len == stride_len,
        "Shape length (",
        shape_len,
        ") must match stride length (",
        stride_len,
        ")",
    )

    comptime ResultTypes = _Idx2CrdResultTypes[
        out_dtype,
        Scalar[out_dtype],
        Stride.ParamListType,
        Shape.ParamListType,
    ]
    comptime Result = Coord[*ResultTypes]
    var result = Result()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        comptime for i in range(shape_len):
            comptime if Shape.ParamListType[i].is_tuple:
                # Nested shape: recurse into sub-shape/sub-stride.
                var nested = idx2crd[out_dtype=out_dtype](
                    idx, shape_t[i], stride_t[i]
                )
                Pointer(to=result[i]).write(rebind[ResultTypes[i]](nested))
            elif (
                Shape.ParamListType[i].is_static_value
                and Shape.ParamListType[i].static_value == 1
            ):
                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](ComptimeInt[0]())
                )
            else:
                var stride_val = Int(stride_t[i].value())
                var shape_val = Int(shape_t[i].value())
                var coord_val = _linear_idx_to_coord(idx, stride_val, shape_val)

                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](Scalar[out_dtype](coord_val))
                )
    else:
        comptime if Shape.is_static_value and Shape.static_value == 1:
            Pointer(to=result[0]).write(
                rebind[ResultTypes[0]](ComptimeInt[0]())
            )
        else:
            var coord_val = _linear_idx_to_coord(
                idx, Int(stride.value()), Int(shape.value())
            )

            comptime for i in range(shape_len):
                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](Scalar[out_dtype](coord_val))
                )

    return result


def idx2crd[
    Index: CoordLike,
    Shape: CoordLike,
    Stride: CoordLike,
    out_dtype: DType = .int64,
](idx: Index, shape: Shape, stride: Stride) -> Coord[
    *_Idx2CrdResultTypes[
        out_dtype, Index, Stride.ParamListType, Shape.ParamListType
    ]
]:
    """Calculate the coordinate tuple from a CoordLike linear index.

    This overload accepts a CoordLike index, enabling compile-time result
    computation when the index, shape, and stride are all statically known.
    Uses the per-element formula: ``coord[i] = (idx // stride[i]) % shape[i]``.

    The idx, and all components of shape and stride must be non-negative.

    Parameters:
        Index: The index type (must be CoordLike).
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_dtype: The output data type for coordinate values.

    Args:
        idx: The CoordLike linear index to convert.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A `Coord` containing the coordinate values for each dimension.
        When idx, shape, and stride are all compile-time known, produces
        `ComptimeInt` results. Otherwise produces `Scalar[out_dtype]`.
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()

    debug_assert(
        shape_len == stride_len,
        "Shape length (",
        shape_len,
        ") must match stride length (",
        stride_len,
        ")",
    )

    comptime ResultTypes = _Idx2CrdResultTypes[
        out_dtype, Index, Stride.ParamListType, Shape.ParamListType
    ]
    comptime Result = Coord[*ResultTypes]
    var result = Result()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        comptime for i in range(shape_len):
            comptime if Shape.ParamListType[i].is_tuple:
                # Nested shape: recurse into sub-shape/sub-stride.
                var nested = idx2crd[out_dtype=out_dtype](
                    Int(idx.value()), shape_t[i], stride_t[i]
                )
                Pointer(to=result[i]).write(rebind[ResultTypes[i]](nested))
            elif (
                Shape.ParamListType[i].is_static_value
                and Shape.ParamListType[i].static_value == 1
            ):
                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](ComptimeInt[0]())
                )
            elif (
                Index.is_static_value
                and Shape.ParamListType[i].is_static_value
                and Stride.ParamListType[i].is_static_value
            ):
                # All static: result is ComptimeInt, already default-initialized.
                pass
            else:
                var stride_val = Int(stride_t[i].value())
                var shape_val = Int(shape_t[i].value())
                var coord_val = _linear_idx_to_coord(
                    Int(idx.value()), stride_val, shape_val
                )
                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](Scalar[out_dtype](coord_val))
                )
    else:
        comptime if Shape.is_static_value and Shape.static_value == 1:
            Pointer(to=result[0]).write(
                rebind[ResultTypes[0]](ComptimeInt[0]())
            )
        elif (
            Index.is_static_value
            and Shape.is_static_value
            and Stride.is_static_value
        ):
            # All static: result is ComptimeInt, already default-initialized.
            pass
        else:
            var coord_val = _linear_idx_to_coord(
                Int(idx.value()), Int(stride.value()), Int(shape.value())
            )

            comptime for i in range(shape_len):
                Pointer(to=result[i]).write(
                    rebind[ResultTypes[i]](Scalar[out_dtype](coord_val))
                )

    return result


@always_inline
def coord_to_index_list[
    element_types: TypeList[Trait=CoordLike, ...]
](value: Coord[*element_types]) -> IndexList[value.rank]:
    """Convert a flat `Coord` to an `IndexList`.

    Parameters:
        element_types: The list of element types in the `Coord`.

    Args:
        value: The `Coord` to convert.

    Returns:
        An `IndexList` with the same rank and values as the input `Coord`.
    """
    var result = IndexList[value.rank]()

    comptime for i in range(type_of(value).__len__()):
        result[i] = Int(value[i].value())

    return result


@always_inline
def dyn_coord[
    dtype: DType, *element_types: Movable & Deinitable
](
    var values: Tuple[*element_types],
    out result: Coord[
        *TypeList.splat[
            Trait=CoordLike, type_of(values).__len__(), Scalar[dtype]
        ]()
    ],
) where _AllEqual[Int, *element_types]:
    """Create a `Coord` from a tuple of integers with specified dtype.

    Parameters:
        dtype: The data type for the runtime integer values.
        element_types: The types of elements in the input tuple.

    Args:
        values: The runtime integer values.

    Returns:
        A `Coord` instance containing `Scalar` elements for each value.
    """
    result = {}

    comptime for i in range(type_of(values).__len__()):
        Pointer(to=result[i]).write(
            rebind[type_of(result[i])](Scalar[dtype](rebind[Int](values[i])))
        )


# values is a ZST since all elements are comptime
comptime coord[*values: Int]: Coord[*_IntToComptimeInt[*values]] = {}
"""Create a `Coord` from compile-time integer values.

Parameters:
    values: The compile-time integer values.

Returns:
    A `Coord` instance containing `ComptimeInt` elements for each value.
"""


comptime DynamicCoord[dtype: DType, size: Int] = Coord[
    *TypeList.splat[Trait=CoordLike, size, Scalar[dtype]]()
]
"""
Create a Coord full of `size` dynamic elements with `dtype`.

Parameters:
    dtype: The output element DType.
    size: The number of output elements.

Returns:
    A Coord full of `size` dynamic elements with `dtype`.
"""

comptime StaticCoord[value: Int, size: Int] = Coord[
    *TypeList.splat[Trait=CoordLike, size, ComptimeInt[value]]()
]
"""
Create a Coord full of `size` static elements.

Parameters:
    value: The value of each element.
    size: The number of output elements.

Returns:
    A Coord full of `size` static elements.
"""


comptime _FlattenOnceMapper[
    element_type: CoordLike,
]: TypeList.of[Trait=CoordLike]()._mlir_type = element_type._ParamListType

comptime _FlattenOnce[*element_types: CoordLike] = TypeList._concat[
    *element_types.map_to_values[_FlattenOnceMapper]()
]()
"""Peels one level of Coord nesting from a variadic type list.

Each tuple element is replaced by its children; scalar elements are kept.
For example, `(Coord(A, B), C)` becomes `(A, B, C)`, but if `A` is itself
a Coord the result still contains that nested Coord.
"""


comptime _Flatten2[*element_types: CoordLike] = _FlattenOnce[
    *_FlattenOnce[*element_types]
]
"""Two passes of `_FlattenOnce`, handling up to depth-2 nesting."""


comptime _Flattened[*element_types: CoordLike] = _Flatten2[
    *_Flatten2[*element_types]
]
"""Iteratively flatten a variadic type list to its leaf scalar types.

Applies `_FlattenOnce` four times via doubling (`_Flatten2` ∘ `_Flatten2`),
handling up to four levels of Coord nesting.  Once fully flat, additional
passes are no-ops since all elements are scalars.
"""


def _get_flattened_helper[
    element_types: TypeList[Trait=CoordLike, ...],
    //,
    flat_idx: Int,
    current_offset: Int,
    i: Int,
](tuple: Coord[*element_types]) -> Int:
    """Helper function to recursively access flattened elements."""

    comptime assert i < type_of(tuple).__len__(), "flat_idx out of bounds"

    comptime T = element_types[i]

    comptime if T.is_tuple:
        comptime count = _Flattened[*T.ParamListType].length

        comptime if flat_idx >= current_offset and flat_idx < current_offset + count:
            return _get_flattened[flat_idx - current_offset](tuple[i].tuple())
        else:
            return _get_flattened_helper[
                flat_idx, current_offset + count, i + 1
            ](tuple)
    else:
        comptime if flat_idx == current_offset:
            return Int(tuple[i].value())
        else:
            return _get_flattened_helper[flat_idx, current_offset + 1, i + 1](
                tuple
            )


def _get_flattened[
    element_types: TypeList[Trait=CoordLike, ...],
    //,
    flat_idx: Int,
](tuple: Coord[*element_types]) -> Int:
    """Access an element from a nested `Coord` using a flat index.

    Parameters:
        element_types: The variadic element types of the tuple.
        flat_idx: The index into the flattened representation.

    Args:
        tuple: The nested `Coord` to access.

    Returns:
        The value at the given flat index.

    Examples:
        For `tuple = Coord(Idx[5], Coord(Idx[3], Idx[2]), 7)`:
        - `get_flattened[0](tuple)` returns 5  (first element)
        - `get_flattened[1](tuple)` returns 3  (first element of nested tuple)
        - `get_flattened[2](tuple)` returns 2  (second element of nested tuple)
        - `get_flattened[3](tuple)` returns 7  (third top-level element)
    """
    return _get_flattened_helper[flat_idx, 0, 0](tuple)


comptime _IsStaticPredicate[T: CoordLike] = T.is_static_value

comptime _IsNotTuplePredicate[T: CoordLike] = not T.is_tuple

comptime _IsFlat[*element_types: CoordLike] = element_types.all[
    _IsNotTuplePredicate,
]()
"""True iff no element in the variadic is a tuple. Single-pass; lets
`_AllStatic` / `_StaticProduct` skip the multi-pass `_Flattened` rewrite
on the (typical) flat path."""


comptime _AllStaticFlat[*element_types: CoordLike] = element_types.all[
    _IsStaticPredicate,
]()


comptime _AllStatic[*element_types: CoordLike] = _AllStaticFlat[
    *element_types
] if _IsFlat[*element_types] else _AllStaticFlat[*_Flattened[*element_types]]
"""True iff every leaf element in the (possibly nested) variadic is a
compile-time-known dim."""

comptime _AllEqualPredicate[T1: AnyType, T2: type_of(T1)] = T1 == T2

comptime _AllEqual[T: AnyType, *element_types: AnyType] = element_types.all[
    _AllEqualPredicate[T, _],
]()

comptime _StaticProductReducer[
    Prev: Int,
    T: CoordLike,
] = Prev * T.static_value

comptime _StaticProductFlat[*element_types: CoordLike] = element_types.reduce[
    1,
    _StaticProductReducer,
]


comptime _StaticProduct[*element_types: CoordLike] = _StaticProductFlat[
    *element_types
] if _IsFlat[*element_types] else _StaticProductFlat[
    *_Flattened[*element_types]
]
"""Compile-time product of all leaf dimensions in the (possibly nested)
variadic. Caller should gate on `_AllStatic` first — a dynamic leaf
(`static_value == -1`) makes the product meaningless."""

comptime _IntToComptimeIntMapper[
    idx: Int,
]: CoordLike = ComptimeInt[idx]


comptime _IntToComptimeInt[*values: Int] = values.map_to_type[
    _IntToComptimeIntMapper
]()


# ===-----------------------------------------------------------------------===#
# Single-element type replacement
# ===-----------------------------------------------------------------------===#

comptime _CoordReplaceAtTabulator[
    element_types: TypeList[Trait=CoordLike, ...],
    at: Int,
    T: CoordLike,
    idx: Int,
]: CoordLike = T if idx == at else element_types.__getitem_param__[idx]
"Maps index `idx` to `T` at position `at`, otherwise keeps the original"

comptime _CoordReplaceAt[
    element_types: TypeList[Trait=CoordLike, ...], at: Int, T: CoordLike
] = TypeList.tabulate[
    element_types.length,
    _CoordReplaceAtTabulator[element_types, at, T, idx=_],
]()
"Replaces the element at `at` with `T` leaving others untouched"


# ===-----------------------------------------------------------------------===#
# CoordLike to Dynamic conversion
# ===-----------------------------------------------------------------------===#


comptime _CoordToDynamicMapper[
    dtype: DType, type: CoordLike
]: CoordLike = Scalar[dtype]
"""Maps a single CoordLike element to Scalar[dtype].
All elements (ComptimeInt, Scalar of any dtype) are converted to Scalar[dtype].
"""


comptime _CoordToDynamic[
    dtype: DType, element_types: TypeList[Trait=CoordLike, ...]
] = element_types.map[_CoordToDynamicMapper[dtype, _]]()
"""Converts a variadic of CoordLike types to all Scalar[dtype].
All elements are converted to Scalar[dtype], regardless of their original type.

Example:

    ```mojo
    from std.utils.coord import _CoordToDynamic, ComptimeInt, CoordLike
    # All elements become Int64
    comptime types = _CoordToDynamic[.int64, TypeList.of[Trait=CoordLike, ComptimeInt[3], Int32, ComptimeInt[5]]()]
    # types is equivalent to TypeList.of[Trait=CoordLike, Int64, Int64, Int64]()
    ```
"""


# ===-----------------------------------------------------------------------===#
# Nested dynamic coord type computation
# ===-----------------------------------------------------------------------===#


comptime _NestedDynamicCoordMapper[
    dtype: DType,
    coord: CoordLike,
]: CoordLike = Coord[
    *_CoordToDynamic[dtype, coord.ParamListType]
] if coord.is_tuple else Scalar[
    dtype
]
"""Maps a CoordLike type to a nested dynamic type.

Scalar types become Scalar[dtype]. Nested Coord types become
Coord[Scalar[dtype], ...] preserving one level of nesting.
"""

comptime _NestedDynamicCoord[
    dtype: DType, *element_types: CoordLike
] = element_types.map[_NestedDynamicCoordMapper[dtype, _]]()
"""Converts a variadic of CoordLike types to dynamic types preserving nesting.

Scalar types become Scalar[dtype]. Nested Coord types become
Coord[Scalar[dtype], ...], preserving the hierarchical structure.
For flat layouts, this is equivalent to _CoordToDynamic (all Scalar).
For nested layouts (e.g. from zipped_divide), the result mirrors the
nesting with Scalar leaves.

Example:

    For flat types (ComptimeInt[3], ComptimeInt[4]):
        result = (Scalar[dtype], Scalar[dtype])

    For nested types (Coord[ComptimeInt[2], ComptimeInt[2]], Coord[ComptimeInt[3], ComptimeInt[4]]):
        result = (Coord[Scalar[dtype], Scalar[dtype]], Coord[Scalar[dtype], Scalar[dtype]])
"""


# ===-----------------------------------------------------------------------===#
# Deep nested dynamic coord type computation (depth 2 and 3)
# ===-----------------------------------------------------------------------===#
# These extend _NestedDynamicCoord (depth 1) to support deeper nesting.
# Each level uses the previous level for its nested elements:
#   depth 1: _NestedDynamicCoord — tuples become Coord[Scalar, ...]
#   depth 2: _DeepDynamicCoord2 — tuples become Coord[*depth1[...]]
#   depth 3: _DeepDynamicCoord3 — tuples become Coord[*depth2[...]]


comptime _DeepDynamicCoordMapper2[
    dtype: DType,
    coord: CoordLike,
]: CoordLike = Coord[
    *_NestedDynamicCoord[dtype, *coord.ParamListType]
] if coord.is_tuple else Scalar[
    dtype
]

comptime _DeepDynamicCoord2[
    dtype: DType, *element_types: CoordLike
] = element_types.map[_DeepDynamicCoordMapper2[dtype, _]]()
"""Converts CoordLike types to dynamic types preserving up to 2 levels of nesting."""


comptime _DeepDynamicCoordMapper3[
    dtype: DType,
    coord: CoordLike,
]: CoordLike = Coord[
    *_DeepDynamicCoord2[dtype, *coord.ParamListType]
] if coord.is_tuple else Scalar[
    dtype
]

comptime _DeepDynamicCoord3[
    dtype: DType, *element_types: CoordLike
] = element_types.map[_DeepDynamicCoordMapper3[dtype, _]]()
"""Converts CoordLike types to dynamic types preserving up to 3 levels of nesting."""


comptime _DeepDynamicCoordMapper4[
    dtype: DType,
    coord: CoordLike,
]: CoordLike = Coord[
    *_DeepDynamicCoord3[dtype, *coord.ParamListType]
] if coord.is_tuple else Scalar[
    dtype
]


comptime _DeepDynamicCoord4[
    dtype: DType, *element_types: CoordLike
] = element_types.map[_DeepDynamicCoordMapper4[dtype, _]]()
"""Converts CoordLike types to dynamic types preserving up to 4 levels of nesting."""


# ===-----------------------------------------------------------------------===#
# idx2crd result type computation
# ===-----------------------------------------------------------------------===#

comptime _as_CoordLike[x: CoordLike] = x

comptime _Idx2CrdResultTabulator[
    out_dtype: DType,
    idx_type: CoordLike,
    stride_types: TypeList[Trait=CoordLike, ...],
    shape_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Coord[
    *_DeepDynamicCoord4[out_dtype, *shape_types[idx].ParamListType]
] if shape_types[
    idx
].is_tuple else ComptimeInt[
    0  # shape == 1: always ComptimeInt[0]
] if shape_types[
    idx
].is_static_value and shape_types[
    idx
].static_value == 1 else ComptimeInt[
    (idx_type.static_value // stride_types[idx].static_value)
    % shape_types[idx].static_value
] if idx_type.is_static_value and shape_types[
    idx
].is_static_value and stride_types[
    idx
].is_static_value else _as_CoordLike[
    Scalar[out_dtype]
]
"""Computes an idx2crd result type.

Considers shape, stride, and index to determine the result type:
- If shape is a nested tuple, produces a nested Coord with Scalar leaves.
- If shape is statically 1, produces ComptimeInt[0].
- If all of idx, shape, and stride are statically known, produces
  ComptimeInt[(idx // stride) % shape].
- Otherwise produces Scalar[out_dtype].
"""


comptime _Idx2CrdResultTypes[
    out_dtype: DType,
    idx_type: CoordLike,
    stride_types: TypeList[Trait=CoordLike, ...],
    shape_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    shape_types.length,
    _Idx2CrdResultTabulator[
        out_dtype, idx_type, stride_types, shape_types, ...
    ],
]()
"""Computes the result types for idx2crd based on shape, stride, and index.

For each dimension:
- If shape is statically 1, the result is ComptimeInt[0]
- If idx, shape, and stride are all statically known, the result is
  ComptimeInt[(idx // stride) % shape]
- Otherwise, the result is Scalar[out_dtype]

Example:
    ```mojo
    from std.utils.coord import _Idx2CrdResultTypes, ComptimeInt, CoordLike
    comptime stride_t = TypeList.of[Trait=CoordLike, ComptimeInt[4], ComptimeInt[4], ComptimeInt[1]]()
    comptime shape_t = TypeList.of[Trait=CoordLike, ComptimeInt[3], ComptimeInt[1], ComptimeInt[4]]()
    comptime types = _Idx2CrdResultTypes[.int64, Int64, stride_t, shape_t]
    ```
"""


struct _RegTuple[*element_types: CoordLike](
    ImplicitlyCopyable, Sized, TrivialRegisterPassable
):
    """
    A temporary internal type to represent a Tuple where
    all elements are register passable. This should
    be removed once we have conditional conformance.
    """

    comptime _mlir_type = __mlir_type[
        `!kgen.struct<:`,
        type_of(Self.element_types.values),
        Self.element_types.values,
        ` isParamPack>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the tuple."""

    # Overload that crushes down IR generated on the caller side.
    @always_inline("nodebug")
    def __init__(out self: _RegTuple[]):
        """Construct an empty tuple."""
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

    @always_inline("nodebug")
    def __init__(out self, var *args: *Self.element_types):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        # Move each element into the tuple storage.
        @__parameter
        def init_elt[idx: Int](var elt: Self.element_types[idx]):
            Pointer(to=self[idx]).write(elt)

        args^.consume_elements[init_elt]()

    @always_inline("builtin")
    @staticmethod
    def __len__() -> Int:
        """Return the number of elements in the tuple.

        Returns:
            The tuple length.
        """

        comptime result = Self.element_types.length
        return result

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.__len__()

    @always_inline("nodebug")
    def __getitem_param__[
        idx: Int
    ](ref self) -> ref[self] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element to return.

        Returns:
            A reference to the specified element.
        """
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = Pointer(to=self._mlir_value)._get_kgen_pointer()

        # KGenPointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.struct.gep`[
            index=idx.__mlir_index__(),
            _type=Pointer[Self.element_types[idx]]._mlir_type,
        ](storage_kgen_ptr)
        return Pointer[_, origin_of(self)](_mlir_value=elt_kgen_ptr)[]

    @always_inline("nodebug")
    def __init__[*elt_types: CoordLike](out self: _RegTuple[*elt_types]):
        """Construct a tuple with default-initialized elements.

        Parameters:
            elt_types: The types of the elements contained in the Tuple.
        """

        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(type_of(self).__len__()):
            Pointer(to=self[i]).write(elt_types[i]())

    @always_inline("nodebug")
    def reverse(
        self,
        out result: _RegTuple[*Self.element_types.reverse()],
    ):
        """Return a new tuple with the elements in reverse order.

        Returns:
            A new tuple with the elements in reverse order.

        Usage:

        ```mojo
        from std.utils.coord import _RegTuple, ComptimeInt, Idx
        var image_coords = _RegTuple[ComptimeInt[100], ComptimeInt[200]](Idx[100], Idx[200])
        var screen_coords = image_coords.reverse()
        print(screen_coords[0].value(), screen_coords[1].value())  # output: 200, 100
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(type_of(result).__len__()):
            Pointer(to=result[i]).write(
                rebind[type_of(result[i])](
                    self[Self.element_types.length - 1 - i]
                )
            )

    @always_inline("nodebug")
    def concat[
        *other_element_types: CoordLike
    ](
        self,
        other: _RegTuple[*other_element_types],
        out result: _RegTuple[
            *TypeList._concat[
                Self.element_types.values, other_element_types.values
            ]()
        ],
    ):
        """Return a new tuple that concatenates this tuple with another.

        Args:
            other: The other tuple to concatenate.

        Parameters:
            other_element_types: The types of the elements contained in the other _RegTuple.

        Returns:
            A new tuple with the concatenated elements.

        Usage:

        ```
        var rgb = _RegTuple[Int, Int, Int](0xFF, 0xF0, 0x0)
        var rgba = rgb.concat(_RegTuple[Int](0xFF)) # Adds alpha channel
        print(rgba[0], rgba[1], rgba[2], rgba[3]) # 255 240 0 255
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime self_len = Self.__len__()

        comptime for i in range(self_len):
            Pointer(to=result[i]).write(rebind[type_of(result[i])](self[i]))

        comptime for i in range(type_of(other).__len__()):
            Pointer(to=result[self_len + i]).write(
                rebind[type_of(result[self_len + i])](other[i])
            )

    @always_inline("nodebug")
    def __contains__[T: Equatable](self, value: T) -> Bool:
        """Return whether the tuple contains the specified value.

        For example:

        ```mojo
        var t = Tuple(True, 1, 2.5)
        if 1 in t:
            print("t contains 1")
        ```

        Args:
            value: The value to search for.

        Parameters:
            T: The type of the value.

        Returns:
            True if the value is in the tuple, False otherwise.
        """

        comptime for i in range(type_of(self).__len__()):
            comptime if Self.element_types[i] == T:
                if rebind[T](self[i]) == value:
                    return True

        return False


comptime _MultiplyTabulator[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    Lhs[idx].static_value * Rhs[idx].static_value
] if Lhs[
    idx
].is_static_value and Rhs[
    idx
].is_static_value else Scalar[
    Lhs[idx].DTYPE
]


comptime _Multiply[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    Lhs.length,
    _MultiplyTabulator[Lhs, Rhs, ...],
]()


comptime _MultiplyByScalarMapper[
    scalar: Int,
    coord: CoordLike,
]: CoordLike = ComptimeInt[
    coord.static_value * scalar
] if coord.is_static_value else Scalar[
    coord.DTYPE
]


comptime _MultiplyByScalar[
    Types: TypeList[Trait=CoordLike, ...],
    scalar: Int,
] = Types.map[
    _MultiplyByScalarMapper[scalar=scalar, ...],
]()
"""Multiply each element in Types by a scalar value.

Parameters:
    Types: The variadic types to multiply.
    scalar: The scalar value to multiply each element by.

Returns:
    A new variadic of ComptimeInt types with multiplied values.
"""


comptime _DivideTabulator[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    Lhs[idx].static_value // Rhs[idx].static_value
] if Lhs[
    idx
].is_static_value and Rhs[
    idx
].is_static_value else Scalar[
    Lhs[idx].DTYPE
]

comptime _Divide[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    Lhs.length,
    _DivideTabulator[Lhs, Rhs, ...],
]()


comptime _CeilDivTabulator[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    (Lhs[idx].static_value + Rhs[idx].static_value - 1) // Rhs[idx].static_value
] if Lhs[
    idx
].is_static_value and Rhs[
    idx
].is_static_value else Scalar[
    Lhs[idx].DTYPE
]

comptime _CeilDiv[
    Lhs: TypeList[Trait=CoordLike, ...],
    Rhs: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    Lhs.length,
    _CeilDivTabulator[Lhs, Rhs, ...],
]()

comptime _CoordCastMapper[
    dtype: DType, type: CoordLike
]: CoordLike = type if type.is_static_value else Scalar[dtype]
"""Maps dynamic scalar elements to `Scalar[dtype]`, preserving static ones."""


comptime _CoordCast[
    dtype: DType, element_types: TypeList[Trait=CoordLike, ...]
] = element_types.map[_CoordCastMapper[dtype, _]]()
"""Computes the result element types for `Coord.cast[dtype]()`."""


@always_inline
def _linear_idx_to_coord(idx: Int, stride: Int, shape: Int) -> Int:
    return umod(ufloordiv(idx, stride), shape)


@always_inline
def _coerce_dynamic[T: CoordLike](value: Int) -> T:
    comptime if T == Int:
        return rebind[T](value)
    else:
        return rebind[T](Scalar[T.DTYPE](value))
