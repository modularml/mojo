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
fn to_string(mut x: IntOrString) -> String:
  if x.isa[String]():
    return x[String]
  # x.isa[Int]()
  return str(x[Int])

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

from os import abort
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
    ExplicitlyCopyable,
):
    """A runtime-variant type.

    Data for this type is stored internally. Currently, its size is the
    largest size of any of its variants plus a 16-bit discriminant.

    You can
        - use `isa[T]()` to check what type a variant is
        - use `unsafe_take[T]()` to take a value from the variant
        - use `[T]` to get a value out of a variant
            - This currently does an extra copy/move until we have origins
            - It also temporarily requires the value to be mutable
        - use `set[T](owned new_value: T)` to reset the variant to a new value

    Example:
    ```mojo
    from utils import Variant
    alias IntOrString = Variant[Int, String]
    fn to_string(mut x: IntOrString) -> String:
        if x.isa[String]():
            return x[String]
        # x.isa[Int]()
        return str(x[Int])

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

    fn __init__(out self, *, unsafe_uninitialized: ()):
        """Unsafely create an uninitialized Variant.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer is unsafe.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, Self._mlir_type]

    @implicit
    fn __init__[T: CollectionElement](mut self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        alias idx = Self._check[T]()
        self._get_discr() = idx
        self._get_ptr[T]().init_pointee_move(value^)

    fn __init__(out self, *, other: Self):
        """Explicitly creates a deep copy of an existing variant.

        Args:
            other: The value to copy from.
        """
        self = Self(unsafe_uninitialized=())
        self._get_discr() = other._get_discr()

        @parameter
        for i in range(len(VariadicList(Ts))):
            alias T = Ts[i]
            if self._get_discr() == i:
                self._get_ptr[T]().init_pointee_move(other._get_ptr[T]()[])
                return

    fn __copyinit__(out self, other: Self):
        """Creates a deep copy of an existing variant.

        Args:
            other: The variant to copy from.
        """

        # Delegate to explicit copy initializer.
        self = Self(other=other)

    fn __moveinit__(out self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_discr() = other._get_discr()

        @parameter
        for i in range(len(VariadicList(Ts))):
            alias T = Ts[i]
            if self._get_discr() == i:
                # Calls the correct __moveinit__
                other._get_ptr[T]().move_pointee_into(self._get_ptr[T]())
                return

    fn __del__(owned self):
        """Destroy the variant."""

        @parameter
        for i in range(len(VariadicList(Ts))):
            if self._get_discr() == i:
                self._get_ptr[Ts[i]]().destroy_pointee()
                return

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__[T: CollectionElement](ref self) -> ref [self] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            A reference to the internal data.
        """
        if not self.isa[T]():
            abort("get: wrong variant type")

        return self.unsafe_get[T]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn _get_ptr[T: CollectionElement](self) -> UnsafePointer[T]:
        alias idx = Self._check[T]()
        constrained[idx != Self._sentinel, "not a union element type"]()
        var ptr = UnsafePointer.address_of(self._impl).address
        var discr_ptr = __mlir_op.`pop.variant.bitcast`[
            _type = UnsafePointer[T]._mlir_type, index = idx.value
        ](ptr)
        return discr_ptr

    @always_inline("nodebug")
    fn _get_discr(ref self) -> ref [self] UInt8:
        var ptr = UnsafePointer.address_of(self._impl).address
        var discr_ptr = __mlir_op.`pop.variant.discr_gep`[
            _type = __mlir_type.`!kgen.pointer<scalar<ui8>>`
        ](ptr)
        return UnsafePointer(discr_ptr).bitcast[UInt8]()[]

    @always_inline
    fn take[T: CollectionElement](mut self) -> T:
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
    fn unsafe_take[T: CollectionElement](mut self) -> T:
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
        self._get_discr() = Self._sentinel
        return self._get_ptr[T]().take_pointee()

    @always_inline
    fn replace[
        Tin: CollectionElement, Tout: CollectionElement
    ](mut self, owned value: Tin) -> Tout:
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

        return self.unsafe_replace[Tin, Tout](value^)

    @always_inline
    fn unsafe_replace[
        Tin: CollectionElement, Tout: CollectionElement
    ](mut self, owned value: Tin) -> Tout:
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
        self.set[Tin](value^)
        return x^

    fn set[T: CollectionElement](mut self, owned value: T):
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
        return self._get_discr() == idx

    fn unsafe_get[T: CollectionElement](ref self) -> ref [self] T:
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
            The internal data represented as a `Pointer[T]`.
        """
        debug_assert(self.isa[T](), "get: wrong variant type")
        return self._get_ptr[T]()[]

    @staticmethod
    fn _check[T: CollectionElement]() -> Int:
        @parameter
        for i in range(len(VariadicList(Ts))):
            if _type_is_eq[Ts[i], T]():
                return i
        return Self._sentinel
