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
"""Defines a Variant type.

You can use this type to implement variant/sum types. For example:

```mojo
from utils import Variant

alias IntOrString = Variant[Int, String]
fn to_string(inout x: IntOrString) -> String:
  if x.isa[String]():
    return x[String][]
  # x.isa[Int]()
  return str(x[Int])[]

# They have to be mutable for now, and implement CollectionElement
var an_int = IntOrString(4)
var a_string = IntOrString(String("I'm a string!"))
var who_knows = IntOrString(0)
import random
if random.random_ui64(0, 1):
    who_knows.set[String]("I'm actually a string too!")

print(to_string(an_int))
print(to_string(a_string))
print(to_string(who_knows))
```
"""

from sys import alignof, sizeof
from sys.intrinsics import _type_is_eq

from memory import UnsafePointer

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_up(value: Int, alignment: Int) -> Int:
    var div_ceil = (value + alignment - 1)._positive_div(alignment)
    return div_ceil * alignment


# ===----------------------------------------------------------------------=== #
# Variant
# ===----------------------------------------------------------------------=== #


struct Variant[*Ts: CollectionElement](
    CollectionElement,
):
    """A runtime-variant type.

    Data for this type is stored internally. Currently, its size is the
    largest size of any of its variants plus a 16-bit discriminant.

    You can
        - use `isa[T]()` to check what type a variant is
        - use `unsafe_take[T]()` to take a value from the variant
        - use `[T]` to get a value out of a variant
            - This currently does an extra copy/move until we have lifetimes
            - It also temporarily requires the value to be mutable
        - use `set[T](owned new_value: T)` to reset the variant to a new value

    Example:
    ```mojo
    from utils import Variant
    alias IntOrString = Variant[Int, String]
    fn to_string(inout x: IntOrString) -> String:
        if x.isa[String]():
            return x[String][]
        # x.isa[Int]()
        return str(x[Int][])

    # They have to be mutable for now, and implement CollectionElement
    var an_int = IntOrString(4)
    var a_string = IntOrString(String("I'm a string!"))
    var who_knows = IntOrString(0)
    import random
    if random.random_ui64(0, 1):
        who_knows.set[String]("I'm actually a string too!")

    print(to_string(an_int))
    print(to_string(a_string))
    print(to_string(who_knows))
    ```

    Parameters:
      Ts: The elements of the variadic.
    """

    # Fields
    alias _sentinel: Int = -1
    alias _mlir_type = __mlir_type[
        `!kgen.variant<[rebind(:`, __type_of(Ts), ` `, Ts, `)]>`
    ]
    var _impl: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self, *, unsafe_uninitialized: ()):
        """Unsafely create an uninitialized Variant.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer is unsafe.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, Self._mlir_type]

    fn __init__[T: CollectionElement](inout self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state() = Self._check[T]()
        self._get_ptr[T]().init_pointee_move(value^)

    fn __init__(inout self, *, other: Self):
        """Explicitly creates a deep copy of an existing variant.

        Args:
            other: The value to copy from.
        """
        self = Self(unsafe_uninitialized=())
        self._get_state() = other._get_state()

        @parameter
        for i in range(len(VariadicList(Ts))):
            alias T = Ts[i]
            if self._get_state() == i:
                self._get_ptr[T]().initialize_pointee_explicit_copy(
                    other._get_ptr[T]()[]
                )

    # TODO: Enable __copyinit__ only if all elements types have the trait `Copyable` (cheap to copy).

    fn __moveinit__(inout self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state() = other._get_state()

        @parameter
        for i in range(len(VariadicList(Ts))):
            alias T = Ts[i]
            if self._get_state() == i:
                # Calls the correct __moveinit__
                other._get_ptr[T]().move_pointee_into(self._get_ptr[T]())

    fn __del__(owned self):
        """Destroy the variant."""
        self._call_correct_deleter()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__[
        T: CollectionElement
    ](ref [_]self: Self) -> ref [__lifetime_of(self)] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Reference[T]`.
        """
        if not self.isa[T]():
            abort("get: wrong variant type")

        return self.unsafe_get[T]()[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _get_ptr[T: CollectionElement](self) -> UnsafePointer[T]:
        constrained[
            Self._check[T]() != Self._sentinel, "not a union element type"
        ]()
        return UnsafePointer.address_of(self._impl).bitcast[T]()

    fn _get_state(ref [_]self: Self) -> ref [__lifetime_of(self)] Int8:
        var int8_self = UnsafePointer.address_of(self).bitcast[Int8]()
        return (int8_self + Self._size())[]

    @always_inline
    fn _call_correct_deleter(inout self):
        @parameter
        for i in range(len(VariadicList(Ts))):
            if self._get_state() == i:
                self._get_ptr[Ts[i]]().destroy_pointee()

    @always_inline
    fn take[T: CollectionElement](inout self) -> T:
        """Take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[T]():
            abort("taking the wrong type!")

        return self.unsafe_take[T]()

    @always_inline
    fn unsafe_take[T: CollectionElement](inout self) -> T:
        """Unsafely take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[T](), "taking wrong type")
        # don't call the variant's deleter later
        self._get_state() = Self._sentinel
        return self._get_ptr[T]().take_pointee()

    @always_inline
    fn replace[
        Tin: CollectionElement, Tout: CollectionElement
    ](inout self, value: Tin) -> Tout:
        """Replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[Tout]():
            abort("taking out the wrong type!")

        return self.unsafe_replace[Tin, Tout](value)

    @always_inline
    fn unsafe_replace[
        Tin: CollectionElement, Tout: CollectionElement
    ](inout self, value: Tin) -> Tout:
        """Unsafely replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[Tout](), "taking out the wrong type!")

        var x = self.unsafe_take[Tout]()
        self.set[Tin](Tin(other=value))
        return x^

    fn set[T: CollectionElement](inout self, owned value: T):
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self = Self(value^)

    fn isa[T: CollectionElement](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        alias idx = Self._check[T]()
        return self._get_state() == idx

    fn unsafe_get[
        T: CollectionElement
    ](ref [_]self: Self) -> Reference[T, __lifetime_of(self)]:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Reference[T]`.
        """
        debug_assert(self.isa[T](), "get: wrong variant type")
        return self._get_ptr[T]()[]

    @staticmethod
    fn _check[T: CollectionElement]() -> Int8:
        var result = -1

        @parameter
        for i in range(len(VariadicList(Ts))):
            if _type_is_eq[Ts[i], T]():
                result = i
        return result

    @staticmethod
    fn _size() -> Int:
        var size = 0

        @parameter
        for i in range(len(VariadicList(Ts))):
            size = max(size, _align_up(sizeof[Ts[i]](), alignof[Ts[i]]()))
        return _align_up(size, alignof[Int]())
