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
"""Implements the Tuple type.

These are Mojo built-ins, so you don't need to import them.
"""

from std.format._utils import (
    write_sequence_to,
    TypeNames,
    FormatStruct,
)
from std.hashlib.hasher import Hasher

from std.utils._visualizers import lldb_formatter_wrapping_type

# ===-----------------------------------------------------------------------===#
# Tuple
# ===-----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `Tuple` with"
    " non-`Deinitable` elements"
)
struct Tuple[*Ts: Movable](
    Comparable where Ts.all_conforms_to[Comparable](),
    Copyable where Ts.all_conforms_to[Copyable](),
    Defaultable where Ts.all_conforms_to[Defaultable](),
    Deinitable where Ts.all_conforms_to[Deinitable](),
    Equatable where Ts.all_conforms_to[Equatable](),
    Hashable where Ts.all_conforms_to[Hashable](),
    ImplicitlyCopyable where Ts.all_conforms_to[ImplicitlyCopyable](),
    Movable,
    RegisterPassable where Ts.all_conforms_to[RegisterPassable](),
    Sized,
    Writable where Ts.all_conforms_to[Writable](),
):
    """The type of a literal tuple expression.

    A tuple consists of zero or more values, separated by commas.

    Parameters:
        Ts: The elements type.
    """

    @deprecated(use=Ts)
    comptime element_types = Self.Ts
    """The elements type. Deprecated alias for `Ts`."""

    comptime _mlir_type = __mlir_type[
        `!kgen.struct<:`,
        type_of(Self.Ts.values),
        Self.Ts.values,
        ` isParamPack>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the tuple."""

    @always_inline("nodebug")
    def __init__(
        out self,
    ) where Self.Ts.all_conforms_to[Defaultable]():
        """Construct a tuple with default-initialized elements.

        Constraints:
            All `Ts` must conform to `Defaultable`. The constraint
            is enforced via a per-element `comptime assert` in the body
            instead of an explicit `where` clause so that callers whose
            element types come from a comptime reducer (which the solver
            can't reduce through when checking
            `all_conforms_to[Defaultable]()`) can
            still default-construct.
        """
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(Self.__len__()):
            Pointer(to=self[i]).unsafe_write({})

    @always_inline("nodebug")
    def __init__(out self, var *args: *Self.Ts):
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
        def init_elt[idx: Int](var elt: Self.Ts[idx]):
            Pointer(to=self[idx]).unsafe_write(elt^)

        args^.consume_elements[init_elt]()

    def __deinit__(
        deinit self,
    ) where Self.Ts.all_conforms_to[Deinitable]():
        """Destructor that destroys all of the elements.

        Constraints:
            All `Ts` must be `Deinitable`. When any element
            is not, the tuple has no implicit destructor and must be torn down
            with `deinit_with()`.
        """
        # Run the destructor on each member, the destructor of !kgen.struct is
        # trivial and won't do anything.
        comptime for i in range(Self.__len__()):
            Pointer(to=self[i]).unsafe_deinit_pointee()

    def deinit_with[
        deinit_func: def[idx: Int](var elt: Self.Ts[idx]) capturing
    ](deinit self):
        """Consume the tuple, deinitializing each element with a closure.

        Use this to tear down a `Tuple` whose elements are not
        `Deinitable`. Elements are visited in index order.

        Parameters:
            deinit_func: A closure called once per element, receiving ownership
                of the element at that index so it can destroy it.
        """
        self^.consume_elements[deinit_func]()

    @always_inline("nodebug")
    def __init__(
        out self, *, copy: Self
    ) where Self.Ts.all_conforms_to[Copyable]():
        """Copy construct the tuple.

        Args:
            copy: The value to copy from.
        """
        # Mark '_mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(Self.__len__()):
            # TODO: We should not use self[i] as this returns a reference to
            # uninitialized memory.
            Pointer(to=self[i]).unsafe_write(copy=copy[i])

    @always_inline("nodebug")
    def __init__(out self, *, deinit move: Self):
        """Move construct the tuple.

        Args:
            move: The value to move from.
        """
        # Mark '_mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(Self.__len__()):
            # TODO: We should not use self[i] as this returns a reference to
            # uninitialized memory.
            Pointer(to=self[i]).unsafe_write_move_from(Pointer(to=move[i]))
        # Note: The destructor on `move` is auto-disabled in a moveinit.

    @always_inline("builtin")
    @staticmethod
    def __len__() -> Int:
        """Return the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.Ts.length

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.__len__()

    @always_inline("nodebug")
    def __getitem_param__[idx: Int](ref self) -> ref[self] Self.Ts[idx]:
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
            _type=Pointer[Self.Ts[idx]]._mlir_type,
        ](storage_kgen_ptr)
        return Pointer[_, origin_of(self)](_mlir_value=elt_kgen_ptr)[]

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
            comptime if Self.Ts[i] == T:
                if rebind[T](self[i]) == value:
                    return True

        return False

    @always_inline
    def __eq__(
        self, other: Self
    ) -> Bool where Self.Ts.all_conforms_to[Equatable]():
        """Compare this tuple to another tuple using equality comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is equal to the other tuple, False otherwise.
        """
        comptime for i in range(type_of(self).__len__()):
            if self[i] != other[i]:
                return False
        return True

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where Self.Ts.all_conforms_to[Hashable]():
        """Hashes the tuple using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        comptime for i in range(type_of(self).__len__()):
            self[i].__hash__(hasher)

    @no_inline
    def _write_tuple_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where Self.Ts.all_conforms_to[Writable]():
        """Write this tuple's elements to a writer.

        Parameters:
            is_repr: Whether to use repr formatting for elements.

        Args:
            writer: The writer to write to.
        """

        var self_ptr = Pointer(to=self)

        def elements[i: Int](mut writer: Some[Writer]) {self_ptr}:
            comptime if is_repr:
                self_ptr[][i].write_repr_to(writer)
            else:
                self_ptr[][i].write_to(writer)

        write_sequence_to[size=Self.__len__()](
            writer, elements, open="", close=""
        )

        comptime if Self.__len__() == 1:
            writer.write_string(",")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where Self.Ts.all_conforms_to[Writable]():
        """Write this tuple's text representation to a writer.

        Elements are formatted using their `write_to()` representation.
        Single-element tuples include a trailing comma: `(1,)`.

        Args:
            writer: The writer to write to.
        """
        writer.write_string("(")
        self._write_tuple_to[is_repr=False](writer)
        writer.write_string(")")

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where Self.Ts.all_conforms_to[Writable]():
        """Write this tuple's debug representation to a writer.

        Outputs the type name and parameters followed by elements formatted
        using their `write_repr_to()` representation. For example,
        `Tuple[Int, String](Int(0), 'hello')`.

        Args:
            writer: The writer to write to.
        """

        var self_ptr = Pointer(to=self)

        def fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_tuple_to[is_repr=True](w)

        FormatStruct(writer, "Tuple").params(TypeNames[*Self.Ts]()).fields(
            fields
        )

    @always_inline
    def _compare(
        self, other: Self
    ) -> Int where Self.Ts.all_conforms_to[Comparable]():
        comptime for i in range(Self.__len__()):
            if self[i] < other[i]:
                return -1
            if other[i] < self[i]:
                return 1
        return 0

    @always_inline
    def __lt__(
        self, other: Self, /
    ) -> Bool where Self.Ts.all_conforms_to[Comparable]():
        """Compare this tuple to another tuple using less than comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than the other tuple, False otherwise.
        """
        return self._compare(other) < 0

    @always_inline
    def __le__(
        self, other: Self, /
    ) -> Bool where Self.Ts.all_conforms_to[Comparable]():
        """Compare this tuple to another tuple using less than or equal to comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than or equal to the other tuple, False otherwise.
        """
        return self._compare(other) <= 0

    @always_inline
    def __gt__(
        self, other: Self, /
    ) -> Bool where Self.Ts.all_conforms_to[Comparable]():
        """Compare this tuple to another tuple using greater than comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than the other tuple, False otherwise.
        """

        return self._compare(other) > 0

    @always_inline
    def __ge__(
        self, other: Self, /
    ) -> Bool where Self.Ts.all_conforms_to[Comparable]():
        """Compare this tuple to another tuple using greater than or equal to comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than or equal to the other tuple, False otherwise.
        """

        return self._compare(other) >= 0

    @always_inline("nodebug")
    def reverse(deinit self, out result: Tuple[*Self.Ts.reverse()]):
        """Return a new tuple with the elements in reverse order.

        Returns:
            A new tuple with the elements in reverse order.

        Usage:

        ```mojo
        image_coords = Tuple[Int, Int](100, 200) # row-major indexing
        screen_coords = image_coords.reverse() # (col, row) for x,y display
        print(screen_coords[0], screen_coords[1]) # output: 200, 100
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(type_of(result).__len__()):
            Pointer(to=result[i]).unsafe_write_move_from(
                rebind[Pointer[type_of(result[i]), origin_of(self)]](
                    Pointer(to=self[Self.Ts.length - 1 - i])
                )
            )

    @always_inline("nodebug")
    def concat[
        *OtherTs: Movable
    ](
        deinit self,
        deinit other: Tuple[*OtherTs],
        out result: Tuple[*TypeList._concat[Self.Ts.values, OtherTs.values]()],
    ):
        """Return a new tuple that concatenates this tuple with another.

        Args:
            other: The other tuple to concatenate.

        Parameters:
            OtherTs: The types of the elements contained in the other Tuple.

        Returns:
            A new tuple with the concatenated elements.

        Usage:

        ```mojo
        var rgb = Tuple[Int, Int, Int](0xFF, 0xF0, 0x0)
        var rgba = rgb.concat(Tuple[Int](0xFF)) # Adds alpha channel
        print(rgba[0], rgba[1], rgba[2], rgba[3]) # 255 240 0 255
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime self_len = Self.__len__()

        comptime for i in range(self_len):
            Pointer(to=result[i]).unsafe_write_move_from(
                rebind[Pointer[type_of(result[i]), origin_of(self)]](
                    Pointer(to=self[i])
                )
            )

        comptime for i in range(type_of(other).__len__()):
            Pointer(to=result[self_len + i]).unsafe_write_move_from(
                rebind[
                    Pointer[type_of(result[self_len + i]), origin_of(other)]
                ](Pointer(to=other[i]))
            )

    @always_inline("nodebug")
    def consume_elements[
        elt_handler: def[idx: Int](var elt: Self.Ts[idx]) capturing
    ](deinit self):
        """Consume the tuple by transferring ownership of each element into the
        provided closure one at a time.

        Destructuring assignment such as `a, b = t^` desugars to
        reference-returning subscripts, so it copies each element and cannot
        extract elements whose type is `Movable` but not `ImplicitlyCopyable`.
        `consume_elements` hands each element to `elt_handler` by value instead,
        so a tuple of move-only elements can still be taken apart. Elements are
        visited in index order.

        Parameters:
            elt_handler: A function called once for each element of the tuple,
                receiving ownership of the element at that index.

        Example:

        `List` is `Movable` but not `ImplicitlyCopyable`, so its elements can
        be moved out of a tuple but not copied out:

        ```mojo
        # Each `List` is moved out of the tuple, one at a time.
        var t = ([1, 2, 3], [4, 5, 6])

        @__parameter
        def handler[idx: Int](var elt: t.Ts[idx]):
            print(len(elt))  # prints 3, then 3

        t^.consume_elements[handler]()
        ```
        """
        # `deinit self` disables `Tuple.__deinit__`; the underlying `!kgen.struct`
        # destructor is trivial, so moving every element out and letting `self`
        # die leaks nothing.
        comptime for i in range(Self.__len__()):
            var ptr = Pointer(to=self[i])
            elt_handler[i](
                __get_address_as_owned_value(ptr._get_kgen_pointer())
            )
