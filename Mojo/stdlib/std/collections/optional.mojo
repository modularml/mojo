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
"""Defines Optional, a type modeling a value which may or may not be present.

Optional values can be thought of as a type-safe nullable pattern.
Your value can take on a value or `None`, and you need to check
and explicitly extract the value to get it out.

Examples:

```mojo
var a = Optional(1)
var b = Optional[Int](None)
if a:
    print(a.value())  # prints 1
if b:  # Bool(b) is False, so no print
    print(b.value())
var c = a.or_else(2)
var d = b.or_else(2)
print(c)  # prints 1
print(d)  # prints 2
```
"""

from std.os import abort

from std.utils import Variant

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.rebind import downcast, rebind_var
from std.format._utils import FormatStruct, TypeNames, write_to, write_repr_to
from std.hashlib import Hasher
from std.memory import MaybeUninit, forget_deinit
from std.memory.unsafe_pointer import unsafe_cast
from std.reflection import call_location, reflect
from std.utils import StaticTuple
from std.utils._nicheable import (
    UnsafeNicheable,
    UnsafeCustomNicheStorage,
    NicheIndex,
)


@fieldwise_init
struct _NoneType(TrivialRegisterPassable):
    pass


@fieldwise_init
struct EmptyOptionalError[T: AnyType](
    ImplicitlyCopyable, RegisterPassable, Writable
):
    """An error type for when an empty `Optional` is accessed.

    Parameters:
        T: The type of the value that was accessed in the `Optional`.
    """

    def write_to(self, mut writer: Some[Writer]):
        """Write the error to a `Writer`.

        Args:
            writer: The `Writer` to write to.
        """
        FormatStruct(writer, "EmptyOptionalError").params(
            TypeNames[Self.T]()
        ).fields()

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the error to a `Writer`.

        Args:
            writer: The `Writer` to write to.
        """
        self.write_to(writer)


# ===-----------------------------------------------------------------------===#
# Optional
# ===-----------------------------------------------------------------------===#


@stable(since="1.0")
struct Optional[T: AnyType](
    Boolable,
    Copyable where conforms_to(T, Copyable),
    Defaultable,
    Deinitable where conforms_to(T, Deinitable),
    DevicePassable where conforms_to(T, DevicePassable) and conforms_to(
        T, Copyable
    ),
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable),
    Iterable,
    IterableOwned where conforms_to(T, Movable & Deinitable),
    Movable where conforms_to(T, Movable),
    RegisterPassable where conforms_to(T, RegisterPassable),
    Writable where conforms_to(T, Writable),
):
    """A type modeling a value which may or may not be present.

    Parameters:
        T: The type of value stored in the `Optional`.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    ## Layout

    The layout of `Optional` is not guaranteed and may change at any time.
    The implementation may apply niche optimizations (for example, storing the
    `None` sentinel inside spare bits of `T`) that alter the resulting layout.
    Do not rely on `size_of[Optional[T]]()` or `align_of[Optional[T]]()` being
    stable across compiler versions. The only guarantee is that the size and
    alignment will be at least as large as those of `T` itself.

    If you need to inspect the current size or alignment, use `size_of` and
    `align_of`, but treat the results as non-stable implementation details.

    Examples:

    ```mojo
    var a = Optional(1)
    var b = Optional[Int](None)
    if a:
        print(a.value())  # prints 1
    if b:  # Bool(b) is False, so no print
        print(b.value())
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c)  # prints 1
    print(d)  # prints 2
    ```
    """

    # A `where`-gated member can't satisfy a parameterized associated alias, so
    # this alias stays well-formed for every `T` via `downcast` (mirroring
    # `Array`); only the unparameterized `IteratorOwnedType` can be gated.
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _OptionalIter[downcast[Self.T, Movable & Deinitable]]
    """The iterator type for this optional.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # TODO(MOCO-4308): Remove redundant 'Movable' constraint
    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.T, Movable & Deinitable
    ) = _OptionalIter[Self.T]
    """The owned iterator type for this optional."""

    comptime Element = Self.T
    """The element type of this optional."""

    comptime device_type: AnyType = Self
    """The device-side type for this optional."""

    comptime _type = Variant[_NoneType, Self.T]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @stable(since="1.0")
    def __init__(out self):
        """Construct an empty `Optional`.

        Examples:

        ```mojo
        var instance = Optional[String]()
        print(instance) # Output: None
        ```
        """
        self._value = Self._type(_NoneType())

    @stable(since="1.0")
    @implicit
    def __init__(
        out self, var value: Self.T
    ) where conforms_to(Self.T, Movable):
        """Construct an `Optional` containing a value.

        Args:
            value: The value to store in the `Optional`.

        Examples:

        ```mojo
        var instance = Optional[String]("Hello")
        print(instance) # Output: 'Hello'
        ```
        """
        self._value = Self._type(value^)

    def __init__[F: def() -> Self.T](out self, *, init_with: F):
        """Construct an `Optional` holding a value produced in place by `call`.

        The value returned by `call` is constructed directly into the
        `Optional`'s storage without being moved, so this is the only way to
        populate an `Optional` whose element type is not `Movable`. For a
        `Movable` element type, prefer the value constructor.

        The `init_with` keyword is required to disambiguate from the value
        constructor: a closure is itself a storable value, so a positional
        `Optional(f)` would store `f` rather than call it.

        Parameters:
            F: The type of the initializer closure.

        Args:
            init_with: A closure returning the value to store. Called exactly once.

        Examples:

        ```mojo
        @fieldwise_init
        struct Pinned(Movable where False):
            var value: Int

        def make() -> Pinned:
            return Pinned(7)

        var opt = Optional[Pinned](init_with=make)
        print(opt.value().value)  # Output: 7
        ```
        """
        self._value = Self._type(init_with=init_with)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_hidden
    @implicit
    def __init__(out self, value: NoneType._mlir_type):
        """Construct an empty `Optional`.

        Args:
            value: Must be exactly `None`.

        Examples:

        ```mojo
        var instance = Optional[String](None)
        print(instance) # Output: None
        ```
        """
        self = Self(value=NoneType(value))

    @implicit
    def __init__(out self, value: NoneType):
        """Construct an empty `Optional`.

        Args:
            value: Must be exactly `None`.

        Examples:

        ```mojo
        var instance = Optional[String](None)
        print(instance) # Output: None
        ```
        """
        self = Self()

    @always_inline("nodebug")
    @implicit
    @doc_hidden
    def __init__(
        out self, optional_reg: OptionalReg[Self.T]
    ) where conforms_to(Self.T, TrivialRegisterPassable):
        """Implicitly cast an `OptionalReg[T]` to an `Optional[T]`."""
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        Pointer(to=self).unsafe_bitcast[OptionalReg[Self.T]]()[] = optional_reg

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_optional is None:`.
        """
        return not self.__bool__()

    def __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_optional is not None:`.
        """
        return self.__bool__()

    def __eq__(self, rhs: type_of(None)) -> Bool:
        """Return `True` if a value is not present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `True` if a value is not present, `False` otherwise.
        """
        return self is None

    def __eq__(self, rhs: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Return `True` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Args:
            rhs: The value to compare to.

        Returns:
            True if the values are the same.
        """
        if self:
            if rhs:
                return (
                    self._unsafe_unchecked_value()
                    == rhs._unsafe_unchecked_value()
                )
            return False
        return not rhs

    def __ne__(self, rhs: type_of(None)) -> Bool:
        """Return `True` if a value is present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `False` if a value is not present, `True` otherwise.
        """
        return self is not None

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the Optional's possibly contained value.

        Optionals act as a collection of size 0 or 1.

        Returns:
            An iterator over the Optional's value (if present).

        Examples:

        ```mojo
        var instance = Optional("Hello")
        for value in instance:
            print(value) # Output: Hello
        instance = None
        for value in instance:
            print(value) # Does not reach line
        ```
        """
        comptime assert conforms_to(
            Self.T, Movable & Deinitable
        ) and conforms_to(
            Self.T, Copyable
        ), "Cannot iterate over a non-copyable or non-movable Optional."
        # Rebind the copy to the `downcast` element type that the borrow-side
        # `IteratorType` alias names.
        comptime E = downcast[Self.T, Movable & Deinitable]
        return _OptionalIter[E](rebind_var[Optional[E]](self.copy()))

    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.T, Movable & Deinitable):
        """Consume the Optional and return an iterator over its value.

        Optionals act as a collection of size 0 or 1.

        Returns:
            An iterator that owns the Optional's value (if present).
        """
        return _OptionalIter[Self.T](self^)

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Return the bounds of the `Optional`, which is 0 or 1.

        Returns:
            A tuple containing the length (0 or 1) and an `Optional` containing the length.

        Examples:

        ```mojo
        def bounds():
            var empty_instance = Optional[Int]()
            var populated_instance = Optional[Int](50)

            # Bounds returns a tuple: (`bounds`, `Optional` version of `bounds`)
            # with the length of the `Optional`.
            print(empty_instance.bounds()[0])     # 0
            print(populated_instance.bounds()[0]) # 1
            print(empty_instance.bounds()[1])     # 0
            print(populated_instance.bounds()[1]) # 1
        ```
        """
        var len = 1 if self else 0
        return (len, {len})

    @stable(since="1.0")
    @always_inline
    def __bool__(self) -> Bool:
        """Return true if the Optional has a value.

        Returns:
            True if the `Optional` has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    @always_inline
    def __invert__(self) -> Bool:
        """Return False if the `Optional` has a value.

        Returns:
            False if the `Optional` has a value and True otherwise.
        """
        return not self

    @always_inline
    def __getitem__(
        ref self,
    ) raises EmptyOptionalError[Self.T] -> ref[self._value] Self.T:
        """Retrieve a reference to the value inside the `Optional`.

        Returns:
            A reference to the value inside the `Optional`.

        Raises:
            On empty `Optional`.
        """
        if not self:
            raise EmptyOptionalError[Self.T]()
        return self._unsafe_unchecked_value()

    def _write_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where conforms_to(Self.T, Writable):
        if self:
            comptime if is_repr:
                self._unsafe_unchecked_value().write_repr_to(writer)
            else:
                self._unsafe_unchecked_value().write_to(writer)
        else:
            writer.write_string("None")

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional` to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=False](writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional`'s representation to a `Writer`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_to[is_repr=True](w)

        FormatStruct(writer, "Optional").params(TypeNames[Self.T]()).fields(
            fields
        )

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Updates hasher with the hash of the contained value, if present.

        A `None` optional hashes differently from any present value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        if self:
            # Tag the hash so that hash(T) != hash(Optional[T](..)).
            UInt8(1).__hash__(hasher)
            self._unsafe_unchecked_value().__hash__(hasher)
        else:
            UInt8(0).__hash__(hasher)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where conforms_to(Self.T, DevicePassable) and conforms_to(
        Self.T, Copyable
    ):
        """Convert to device type and store at the target address.

        Args:
            encoder: Target specific device type encoder.
            target: The target pointer to store the device type.
        """
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> (
        String
    ) where conforms_to(Self.T, DevicePassable) and conforms_to(
        Self.T, Copyable
    ):
        """Get the human-readable type name for this `Optional` type.

        Returns:
            A string representation of the type, e.g. `Optional[Int]`.
        """
        return String(t"Optional[{reflect[Self.T].name()}]")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def value(ref self) -> ref[self._value] Self.T:
        """Retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will abort on empty `Optional`.

        Examples:

        ```mojo
        var instance = Optional("Hello")
        var x = instance.value()
        print(x) # Hello
        # instance = Optional[String]() # Uncomment both lines to crash
        # print(instance.value())       # Attempts to take value from `None`
        ```
        """
        if not self.__bool__():
            abort(
                (
                    "`Optional.value()` called on empty `Optional`. Consider"
                    " using `if optional:` to check whether the `Optional` is"
                    " empty before calling `.value()`, or use `.or_else()` to"
                    " provide a default value."
                ),
                location=call_location(),
            )

        return self._unsafe_unchecked_value()

    @always_inline
    def _unsafe_unchecked_value(ref self) -> ref[self._value] Self.T:
        return self._value._unsafe_unchecked_get[Self.T]()

    @always_inline
    def unsafe_value(ref self) -> ref[self._value] Self.T:
        """Unsafely retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will **not** abort on empty `Optional`.

        Examples:

        ```mojo
        var instance = Optional("Hello")
        var x = instance.unsafe_value()
        print(x) # Hello
        instance = Optional[String](None)

        # Best practice:
        if instance:
            var y = instance.unsafe_value() # Will not reach this line
            print(y)

        # In debug builds, this will deterministically abort:
        y = instance.unsafe_value()
        print(y)
        ```
        """
        assert self.__bool__(), "`.value()` on empty `Optional`"
        return self._unsafe_unchecked_value()

    def take(mut self) -> Self.T where conforms_to(Self.T, Movable):
        """Move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will abort on empty `Optional`.

        Examples:

        ```mojo
        var instance = Optional("Hello")
        print(instance.bounds()[0])  # Output: 1
        var x = instance.take() # Moves value from `instance` to `x`
        print(x)  # Output: Hello

        # `instance` is now `Optional(None)`
        print(instance.bounds()[0])  # Output: 0
        print(instance)  # Output: None

        # Best practice
        if instance:
            var y = instance.take()  # Won't reach this line
            print(y)

        # Used directly
        # y = instance.take()         # ABORT: `Optional.take()` called on empty `Optional` (via runtime `abort`)
        # print(y)                    # Does not reach this line
        ```
        """
        if not self.__bool__():
            abort(
                "`Optional.take()` called on empty `Optional`. Consider using"
                " `if optional:` to check whether the `Optional` is empty"
                " before calling `.take()`, or use `.or_else()` to provide a"
                " default value."
            )
        return self.unsafe_take()

    def unsafe_take(mut self) -> Self.T where conforms_to(Self.T, Movable):
        """Unsafely move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will **not** abort on empty `Optional`.

        Examples:

        ```mojo
        var instance = Optional("Hello")
        print(instance.bounds()[0])     # Output: 1
        var x = instance.unsafe_take()  # Moves value from `instance` to `x`
        print(x)                        # Output: Hello

        # `instance` is now `Optional(None)`
        print(instance.bounds()[0])     # Output: 0
        print(instance)                 # Output: None

        # Best practice:
        if instance:
            var y = instance.unsafe_take() # Won't reach this line
            print(y)

        # In debug builds, this will deterministically abort:
        y = instance.unsafe_take()      # ABORT: `Optional.take()` called on empty `Optional` (via `debug_assert`)
        print(y)                        # Does not reach this line
        ```
        """
        assert self.__bool__(), "`.unsafe_take()` on empty `Optional`"
        return self._value.unsafe_replace[_NoneType, Self.T](_NoneType())

    def deinit_with[F: def(var Self.T)](deinit self, deinit_func: F, /):
        """Destroy the value contained in this `Optional` in-place using a
        caller-provided deinitializer function.

        This method can be used to destroy `Optional` values whose element
        type is not `Deinitable`. The `__deinit__` on `Optional`
        requires `T: Deinitable`, so explicit-deinit users must
        destroy an `Optional[T]` through this API instead.

        If `self` is empty, `deinit_func` is not called. Otherwise
        `deinit_func` is called exactly once on the contained value.

        Parameters:
            F: The type of the caller-provided deinitializer function.

        Args:
            deinit_func: Caller-provided deinitializer function for destroying
                an instance of `Self.T`. Not called when `self` is empty.

        Examples:

        ```mojo
        @fieldwise_init
        struct ExplicitDeinit(Movable, Deinitable where False):
            var data: Int

            def explicit_deinit(deinit self):
                pass

        var opt = Optional(ExplicitDeinit(5))
        opt^.deinit_with(ExplicitDeinit.explicit_deinit)
        ```
        """
        if self:
            # SAFETY: We just checked that the `Optional` holds a `T`, so
            # `Variant.deinit_with` won't abort.
            self._value^.deinit_with[Self.T](deinit_func)
        else:
            # Retire the empty `Optional` by destroying its `_NoneType`
            # payload through `Variant.deinit_with`. `_NoneType` is
            # trivially destructible, so `_NoneType.__deinit__` is a no-op.
            self._value^.deinit_with[_NoneType](_NoneType.__deinit__)

    def deinit_assert_empty(deinit self):
        """Destroys an empty `Optional`, asserting that it holds no value.

        Use this on an `Optional[T]` whose element type is not
        `Deinitable` when the value is known to be empty. Unlike
        `deinit_with`, it takes no deinitializer function (there is no live
        value to destroy). In safe-assert builds it aborts if the `Optional`
        is non-empty.

        Examples:

        ```mojo
        var opt: Optional[ExplicitDeinit] = None
        opt^.deinit_assert_empty()
        ```
        """
        debug_assert[assert_mode="safe"](
            not self,
            "`deinit_assert_empty()` called on a non-empty `Optional`",
        )
        self._value^.deinit_with[_NoneType](_NoneType.__deinit__)

    def or_else(
        deinit self, var default: Self.T
    ) -> Self.T where conforms_to(Self.T, Movable & Deinitable):
        """Return the underlying value contained in the `Optional` or a default
        value if the `Optional`'s underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the `Optional` or a default value.

        Examples:

        ```mojo
        var instance = Optional("Hello")
        print(instance)                  # Output: 'Hello'
        print(instance.or_else("Bye"))   # Output: Hello
        instance = None
        print(instance)                  # Output: None
        print(instance.or_else("Bye"))   # Output: Bye
        ```
        """
        if self:
            return self._value^._unsafe_unchecked_unwrap[Self.T]()
        return default^

    @__allow_legacy_custom_self_type
    def copied[
        mut: Bool,
        origin: Origin[mut=mut],
        //,
        _T: Copyable,
    ](self: Optional[Pointer[_T, origin]]) -> Optional[_T]:
        """Converts an `Optional` containing a Pointer to an `Optional` of an
        owned value by copying.

        Parameters:
            mut: Mutability of the pointee origin.
            origin: Origin of the contained `Pointer`.
            _T: Type of the owned result value.

        Returns:
            An `Optional` containing an owned copy of the pointee value.

        Examples:

        Copy the value of an `Optional[Pointer[_]]`

        ```mojo
        var data = "foo"
        var opt = Optional(Pointer(to=data))
        var opt_owned: Optional[String] = opt.copied()
        ```

        Notes:
            If `self` is an empty `Optional`, the returned `Optional` will be
            empty as well.
        """
        if self:
            # SAFETY: We just checked that `self` is populated.
            # Perform an implicit copy
            return self.unsafe_value()[].copy()
        else:
            return None

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def _unsafe_nullable[
        U: AnyType, origin: Origin, address_space: AddressSpace
    ](
        self: Optional[Pointer[U, origin, address_space=address_space]],
        out result: type_of(self).T,
    ):
        result = Pointer(to=self).unsafe_bitcast[type_of(result)]()[]

    def map[
        To: Movable,
        //,
        Mapper: def(var Self.T) -> To,
    ](deinit self, mapper: Mapper) -> Optional[To] where conforms_to(
        Self.T, Movable
    ):
        """Applies a function to the contained value (if any), returning an
        `Optional` containing the result.

        Transforms `Optional[T]` into `Optional[To]` by applying `mapper` to
        the contained value. If `self` is empty, returns an empty `Optional[To]`
        without calling `mapper`.

        Parameters:
            To: The result type of the mapping closure.
            Mapper: The type of the mapping closure.

        Args:
            mapper: The closure to apply to the contained value.

        Returns:
            An `Optional[To]` containing the mapped value, or `None` if `self`
            was empty.

        Examples:

        Map the value inside an `Optional` to a different type:

        ```mojo
        var opt = Optional("hello")
        var length = opt.map(String.byte_length)
        print(length.value())  # Output: 5
        ```

        If the `Optional` is empty, the mapper is not called:

        ```mojo
        var opt = Optional[String](None)
        var length = opt.map(String.byte_length)
        print(length.or_else(-1))  # Output: -1
        ```
        """
        if self:
            return {mapper(self._value^._unsafe_unchecked_unwrap[Self.T]())}
        else:
            # SAFETY:
            # We are `None` therefore we can safely forget self.
            forget_deinit(self^)
            return None

    def and_then[
        To: Movable,
        //,
        Mapper: def(var Self.T) -> Optional[To],
    ](deinit self, mapper: Mapper) -> Optional[To] where conforms_to(
        Self.T, Movable
    ):
        """Calls `mapper` on the contained value (if any), returning the result.

        Unlike `map()`, the mapper function itself returns an `Optional`. This
        allows chaining operations that may each independently fail. Sometimes
        called "flat map" in other languages.

        Parameters:
            To: The value type of the `Optional` returned by the mapper.
            Mapper: The type of the mapping function.

        Args:
            mapper: The function to apply to the contained value. Must return
                an `Optional[To]`.

        Returns:
            The `Optional[To]` returned by `mapper`, or `None` if `self` was
            empty.

        Examples:

        Chain operations that may each return `None`:

        ```mojo
        def try_parse_int(s: String) -> Optional[Int]:
            try:
                return Int(s)
            except:
                return None

        def main():
            var opt = Optional("42")
            var parsed = opt.and_then(try_parse_int)
            print(parsed.value())  # Output: 42
        ```

        If the `Optional` is empty, the mapper is not called:

        ```mojo
        def main():
            var opt = Optional[String](None)
            var parsed = opt.and_then(try_parse_int)
            print(parsed.or_else(-1))  # Output: -1
        ```
        """
        if self:
            return mapper(self._value^._unsafe_unchecked_unwrap[Self.T]())
        else:
            # Destroy the empty `Optional` explicitly: an implicit drop here
            # would require `T: Deinitable`, ruling out linear `T`.
            # SAFETY:
            # We are `None` therefore we can safely forget self.
            forget_deinit(self^)
            return None


# ===-----------------------------------------------------------------------===#
# _OptionalIter
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _OptionalIter[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable),
    Deinitable,
    Iterable where conforms_to(T, Copyable),
    IterableOwned,
    Iterator,
    Movable,
):
    """Iterator over an `Optional`'s zero-or-one contained value."""

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime IteratorOwnedType: Iterator = Self

    var _inner: Optional[Self.T]

    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(
        Self.Element, Copyable
    ):
        return self.copy()

    def __next__(mut self) raises StopIteration -> Self.Element:
        if not self._inner:
            raise StopIteration()
        return self._inner.unsafe_take()

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


# ===-----------------------------------------------------------------------===#
# OptionalReg
# ===-----------------------------------------------------------------------===#


trait _OptionalRegStorageTraits(TrivialRegisterPassable):
    def __init__(out self):
        ...

    def __init__[U: TrivialRegisterPassable](out self, value: U):
        ...

    def value[U: TrivialRegisterPassable](self) -> U:
        ...

    def __bool__(self) -> Bool:
        ...


struct _DefaultOptionalRegStorage[T: TrivialRegisterPassable](
    TrivialRegisterPassable, _OptionalRegStorageTraits
):
    comptime _mlir_type = __mlir_type[`!kgen.variant<`, Self.T, `, i1>`]
    var _value: Self._mlir_type

    @always_inline
    def __init__(out self):
        self._value = __mlir_op.`kgen.variant.create`[
            _type=Self._mlir_type, index=SIMDLength(1)._mlir_value
        ](__mlir_attr.false)

    @always_inline
    def __init__[U: TrivialRegisterPassable](out self, value: U):
        comptime assert U == Self.T
        self._value = __mlir_op.`kgen.variant.create`[
            _type=Self._mlir_type, index=SIMDLength(0)._mlir_value
        ](rebind[Self.T](value))

    @always_inline
    def value[U: TrivialRegisterPassable](self) -> U:
        comptime assert U == Self.T
        var value = __mlir_op.`kgen.variant.get`[
            index=SIMDLength(0)._mlir_value
        ](self._value)
        return rebind[U](value)

    @always_inline
    def __bool__(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index=SIMDLength(0)._mlir_value](
            self._value
        )


struct _NicheableOptionalRegStorage[
    T: TrivialRegisterPassable & UnsafeNicheable
](TrivialRegisterPassable, _OptionalRegStorageTraits):
    comptime StorageType: TrivialRegisterPassable = Self.T.NicheStorage if conforms_to(
        Self.T, UnsafeCustomNicheStorage
    ) else StaticTuple[
        Self.T, 1
    ]
    var storage: Self.StorageType

    @always_inline
    def __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        var ptr = Pointer(to=self.storage).unsafe_bitcast[MaybeUninit[Self.T]]()
        Self.T.write_niche[index=0](ptr)

    @always_inline
    def __init__[U: TrivialRegisterPassable](out self, value: U):
        comptime assert U == Self.T
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        var ptr = Pointer(to=self.storage).unsafe_bitcast[Self.T]()
        ptr.unsafe_write(rebind[Self.T](value))

    @always_inline
    def value[U: TrivialRegisterPassable](self) -> U:
        comptime assert U == Self.T
        return Pointer(to=self.storage).unsafe_bitcast[U]()[]

    @always_inline
    def __bool__(self) -> Bool:
        var ptr = Pointer(to=self.storage).unsafe_bitcast[MaybeUninit[Self.T]]()
        return Self.T.classify_niche(ptr) == NicheIndex.NotANiche


comptime _OptionalRegStorageFor[
    T: TrivialRegisterPassable
]: _OptionalRegStorageTraits = _NicheableOptionalRegStorage[T] if conforms_to(
    T, UnsafeNicheable
) else _DefaultOptionalRegStorage[
    T
]


struct OptionalReg[T: TrivialRegisterPassable](
    Boolable, Defaultable, DevicePassable, TrivialRegisterPassable
):
    """A register-passable optional type.

    This struct optionally contains a value. It only works with trivial register
    passable types at the moment.

    Parameters:
        T: The type of value stored in the Optional.
    """

    comptime _Storage = _OptionalRegStorageFor[Self.T]
    var _value: Self._Storage

    comptime device_type: AnyType = Self
    """The device-side type for this optional register."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Get the human-readable type name for this `OptionalReg` type.

        Returns:
            A string representation of the type, e.g. `OptionalReg[Int]`.
        """
        return String(t"OptionalReg[{reflect[Self.T].name()}]")

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Create an optional with a value of None."""
        self = Self(None)

    @always_inline
    @implicit
    def __init__(out self, value: Self.T):
        """Create an optional with a value.

        Args:
            value: The value.
        """
        self._value = Self._Storage(value)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_hidden
    @always_inline
    @implicit
    def __init__(out self, value: NoneType._mlir_type):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value=NoneType(value))

    @always_inline
    @implicit
    def __init__(out self, value: NoneType):
        """Create an optional without a value from a None literal.

        Args:
            value: The None value.
        """
        self._value = {}

    @always_inline
    @implicit
    def __init__(
        out self: OptionalReg[Self.T],
        optional: Optional[Self.T],
    ):
        """Implicitly convert an `Optional[T]` to an `OptionalReg[T]`.

        Args:
            optional: The `Optional` to convert from.
        """
        if optional:
            self = optional.unsafe_value()
        else:
            self = {}

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    def __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __bool__(self) -> Bool:
        """Return true if the optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return self._value.__bool__()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def value(self) -> Self.T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return self.unsafe_value()

    @always_inline
    def unsafe_value(self) -> Self.T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return self._value.value[Self.T]()

    def or_else(var self, var default: Self.T) -> Self.T:
        """Return the underlying value contained in the Optional or a default
        value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self:
            return self.value()
        return default

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def _unsafe_nullable[
        U: AnyType, origin: Origin, address_space: AddressSpace
    ](
        self: OptionalReg[Pointer[U, origin, address_space=address_space]],
        out result: type_of(self).T,
    ):
        result = Pointer(to=self).unsafe_bitcast[type_of(result)]()[]

    @deprecated(
        "Cannot directly dereference an `OptionalReg[Pointer]`."
        " Unwrap the optional first with `.value()` (aborts if NULL) or"
        " `.unsafe_value()` (unchecked), then index the pointer, e.g."
        " `p.unsafe_value()[]` instead of `p[]`."
    )
    @__allow_legacy_custom_self_type
    @doc_hidden
    def __getitem__[
        type: AnyType,
        origin: Origin,
        address_space: AddressSpace,
        //,
    ](self: OptionalReg[Pointer[type, origin, address_space=address_space]],):
        comptime assert (
            False
        ), "Cannot directly dereference an `OptionalReg[Pointer]`"

    @deprecated(
        "Cannot index directly into an `OptionalReg[Pointer]`. Unwrap the"
        " optional first with `.value()` (aborts if NULL) or `.unsafe_value()`"
        " (unchecked), then index the pointer, e.g. `p.unsafe_value()[i]`"
        " instead of `p[i]`."
    )
    @__allow_legacy_custom_self_type
    @doc_hidden
    def __getitem__[
        type: AnyType,
        origin: Origin,
        address_space: AddressSpace,
        //,
    ](
        self: OptionalReg[Pointer[type, origin, address_space=address_space]],
        offset: Some[Indexer],
    ):
        comptime assert (
            False
        ), "Cannot index directly into an `OptionalReg[Pointer]`"
