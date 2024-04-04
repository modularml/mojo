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
from utils.variant import Variant

alias IntOrString = Variant[Int, String]
fn to_string(inout x: IntOrString) -> String:
  if x.isa[String]():
    return x.get[String]()[]
  # x.isa[Int]()
  return str(x.get[Int]())[]

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

from sys.info import alignof, sizeof
from sys.intrinsics import _mlirtype_is_eq

from memory.unsafe import _LITRef, emplace_ref_unsafe

from utils.loop import unroll
from utils.static_tuple import StaticTuple

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_up(value: Int, alignment: Int) -> Int:
    var div_ceil = (value + alignment - 1)._positive_div(alignment)
    return div_ceil * alignment


@always_inline
fn _max(a: Int, b: Int) -> Int:
    return a if a > b else b


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
            size = _max(size, _align_up(sizeof[Ts[i]](), alignof[Ts[i]]()))

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
            if _mlirtype_is_eq[q, T]():
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
        - use `take[T]()` to take a value from the variant
        - use `get[T]()` to get a value out of a variant
            - This currently does an extra copy/move until we have lifetimes
            - It also temporarily requires the value to be mutable
        - use `set[T](owned new_value: T)` to reset the variant to a new value

    Example:
    ```mojo
    from utils.variant import Variant
    alias IntOrString = Variant[Int, String]
    fn to_string(inout x: IntOrString) -> String:
        if x.isa[String]():
            return x.get[String]()[]
        # x.isa[Int]()
        return str(x.get[Int]()[])

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

    alias _sentinel: Int = -1
    alias _type = __mlir_type[
        `!kgen.variant<[rebind(:`, __type_of(Ts), ` `, Ts, `)]>`
    ]
    var _impl: Self._type

    fn _get_ptr[T: CollectionElement](self) -> AnyPointer[T]:
        constrained[
            Self._check[T]() != Self._sentinel, "not a union element type"
        ]()
        var ptr = Reference(self._impl).get_unsafe_pointer().address
        var result = AnyPointer[T]()
        result.value = __mlir_op.`pop.pointer.bitcast`[
            _type = __mlir_type[`!kgen.pointer<`, T, `>`]
        ](ptr)
        return result

    fn _get_state[
        is_mut: __mlir_type.i1, lt: __mlir_type[`!lit.lifetime<`, is_mut, `>`]
    ](self: _LITRef[Self, is_mut, lt].type) -> Reference[Int8, is_mut, lt]:
        return (
            Reference(self)
            .bitcast_element[Int8]()
            .offset(_UnionSize[Ts].compute())
        )

    fn __init__[T: CollectionElement](inout self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._type]
        self._get_state()[] = Self._check[T]()
        self._get_ptr[T]().emplace_value(value^)

    @always_inline
    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing variant.

        Args:
            other: The variant to copy from.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._type]
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                emplace_ref_unsafe[T](
                    Reference(self._impl).bitcast_element[T](),
                    Reference(other._impl).bitcast_element[T]()[],
                )

        unroll[each, len(VariadicList(Ts))]()

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = __mlir_attr[`#kgen.unknown : `, self._type]
        self._get_state()[] = other._get_state()[]

        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias T = Ts[i]
                # Calls the correct __moveinit__
                self._get_ptr[T]().emplace_value(
                    other._get_ptr[T]().take_value()
                )

        unroll[each, len(VariadicList(Ts))]()

    fn __del__(owned self):
        """Destroy the variant."""
        self._call_correct_deleter()

    @always_inline
    fn _call_correct_deleter(inout self):
        @parameter
        fn each[i: Int]():
            if self._get_state()[] == i:
                alias q = Ts[i]
                __get_address_as_owned_value(self._get_ptr[q]().value).__del__()

        unroll[each, len(VariadicList(Ts))]()

    fn take[T: CollectionElement](owned self) -> T:
        """Take the current value of the variant as the provided type.

        The caller takes ownership of the underlying value. The variant
        type is consumed without calling any deleters.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take.

        Returns:
            The undelying data as an owned value.
        """
        debug_assert(
            Self._check[T]() == self._get_state()[], "taking wrong type"
        )
        self._get_state()[] = (
            Self._sentinel
        )  # don't call the variant's deleter later
        return self._get_ptr[T]().take_value()

    fn set[T: CollectionElement](inout self, owned value: T):
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self._call_correct_deleter()
        self._get_state()[] = Self._check[T]()
        self._get_ptr[T]().emplace_value(value^)

    fn isa[T: CollectionElement](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        alias idx = Self._check[T]()
        return self._get_state()[] == idx

    fn get[
        T: CollectionElement,
        mutability: __mlir_type.`i1`,
        self_life: AnyLifetime[mutability].type,
    ](self: Reference[Self, mutability, self_life].mlir_ref_type) -> Reference[
        T, mutability, self_life
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
            mutability: The inferred mutability of the variant type.
            self_life: The inferred lifetime of the variant type.

        Returns:
            The internal data represented as a `Reference[T]`.
        """
        debug_assert(Reference(self)[].isa[T](), "get: wrong variant type")
        return __mlir_op.`lit.ref.from_pointer`[
            _type = Reference[T, mutability, self_life].mlir_ref_type
        ](Reference(self)[]._get_ptr[T]().value)

    @staticmethod
    fn _check[T: CollectionElement]() -> Int8:
        return _UnionTypeIndex[T, Ts].compute()
