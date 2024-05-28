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
from memory.unsafe_pointer import (
    initialize_pointee_move,
    move_from_pointee,
    move_pointee,
)
from utils import unroll

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


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct _UnionSize[*Ts: CollectionElement]():
    @staticmethod
    fn compute() -> Int:
        var size = 0

        @parameter
        fn each[i: Int]():
            size = max(size, _align_up(sizeof[Ts[i]](), alignof[Ts[i]]()))

        unroll[each, len(VariadicList(Ts))]()
        return _align_up(size, alignof[Int]())


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct _UnionTypeIndex[T: CollectionElement, *Ts: CollectionElement]:
    @staticmethod
    fn compute() -> Int8:
        var result = -1

        @parameter
        fn each[i: Int]():
            alias q = Ts[i]

            @parameter
            if _type_is_eq[q, T]():
                result = i

        unroll[each, len(VariadicList(Ts))]()
        return result


@value
struct Variant[*Ts: CollectionElement](CollectionElement):
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

    fn __init__[T: CollectionElement](inout self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state()[] = Self._check[T]()
        initialize_pointee_move(self._get_ptr[T](), value^)

    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing variant.

        Args:
            other: The variant to copy from.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                initialize_pointee_move(
                    UnsafePointer.address_of(self._impl).bitcast[T](),
                    UnsafePointer.address_of(other._impl).bitcast[T]()[],
                )

        unroll[each, len(VariadicList(Ts))]()

    fn __moveinit__(inout self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._mlir_type]
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                # Calls the correct __moveinit__
                move_pointee(src=other._get_ptr[T](), dst=self._get_ptr[T]())

        unroll[each, len(VariadicList(Ts))]()

    fn __del__(owned self):
        """Destroy the variant."""
        self._call_correct_deleter()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__[
        T: CollectionElement
    ](self: Reference[Self, _, _]) -> ref [self.lifetime] T:
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
        if not self[].isa[T]():
            abort("get: wrong variant type")

        return self[].unsafe_get[T]()[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _get_ptr[T: CollectionElement](self) -> UnsafePointer[T]:
        constrained[
            Self._check[T]() != Self._sentinel, "not a union element type"
        ]()
        return UnsafePointer.address_of(self._impl).bitcast[T]()

    fn _get_state(
        self: Reference[Self, _, _]
    ) -> Reference[Int8, self.is_mutable, self.lifetime]:
        var int8_self = UnsafePointer(self).bitcast[Int8]()
        return (int8_self + _UnionSize[Ts].compute())[]

    @always_inline
    fn _call_correct_deleter(inout self):
        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias q = Ts[i]
                destroy_pointee(self._get_ptr[q]().address)

        unroll[each, len(VariadicList(Ts))]()

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
        self._get_state()[] = Self._sentinel
        return move_from_pointee(self._get_ptr[T]())

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
        self.set[Tin](value)
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
        return self._get_state()[] == idx

    fn unsafe_get[
        T: CollectionElement
    ](self: Reference[Self, _, _]) -> Reference[
        T, self.is_mutable, self.lifetime
    ]:
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
        debug_assert(self[].isa[T](), "get: wrong variant type")
        return self[]._get_ptr[T]()[]

    @staticmethod
    fn _check[T: CollectionElement]() -> Int8:
        return _UnionTypeIndex[T, Ts].compute()
