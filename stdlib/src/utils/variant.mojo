# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines a Variant type.

You can use this type to implement variant/sum types. For example:

```mojo
from utils.variant import Variant

alias IntOrString = Variant[Int, String]
fn to_string(inout x: IntOrString) -> String:
  if x.isa[String]():
    return x.get[String]()
  # x.isa[Int]()
  return str(x.get[Int]())

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

from math.math import max
from sys.info import sizeof, alignof
from algorithm.functional import unroll
from utils.static_tuple import StaticTuple
from sys.intrinsics import _mlirtype_is_eq


fn _alignto(value: Int, align: Int) -> Int:
    return (value + align - 1) // align * align


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct _UnionSize[*Ts: CollectionElement]():
    @staticmethod
    fn compute() -> Int:
        var size = 0

        @parameter
        fn each[i: Int]():
            size = max(size, _alignto(sizeof[Ts[i]](), alignof[Ts[i]]()))

        unroll[len(VariadicList(Ts)), each]()
        return size


# FIXME(#27380): Can't pass *Ts to a function parameter, only type parameter.
struct _UnionTypeIndex[T: CollectionElement, *Ts: CollectionElement]:
    @staticmethod
    fn compute() -> Int16:
        var result = -1

        @parameter
        fn each[i: Int]():
            alias q = Ts[i]

            @parameter
            if _mlirtype_is_eq[q, T]():
                result = i

        unroll[len(VariadicList(Ts)), each]()
        return result


@value
struct Variant[*Ts: CollectionElement](CollectionElement):
    """A runtime-variant type.

    Data for this type is stored internally: its size is the largest size
    of any of its variants.

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
            return x.get[String]()
        # x.isa[Int]()
        return str(x.get[Int]())

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
    alias _type = StaticTuple[_UnionSize[Ts].compute(), Int8]
    var _impl: Self._type
    var _state: Int16

    fn _get_ptr[T: CollectionElement](self) -> AnyPointer[T]:
        constrained[
            Self._check[T]() != Self._sentinel, "not a union element type"
        ]()
        let self_ref = __get_ref_from_value(self._impl)
        let ptr = __mlir_op.`lit.ref.to_pointer`(self_ref)
        var result = AnyPointer[T]()
        result.value = __mlir_op.`pop.pointer.bitcast`[
            _type = __mlir_type[
                `!kgen.pointer<:`, CollectionElement, ` `, T, `>`
            ]
        ](ptr)
        return result

    fn __init__[T: CollectionElement](inout self, owned value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`

        Args:
            value: The value to initialize the variant with.
        """
        self._impl = Self._type()
        self._state = Self._check[T]()
        self._get_ptr[T]().emplace_value(value ^)

    fn __copyinit__(inout self, other: Self):
        """Creates a deep copy of an existing variant.

        Args:
            other: The variant to copy from.
        """
        self._impl = Self._type()
        self._state = other._state

        @parameter
        fn each[i: Int]():
            if self._state == i:
                alias T = Ts[i]
                # TODO(27657): reinterpret_cast without a copy
                var _extra_copy_unsafe = other._impl
                let _extra_impl_ptr = Pointer.address_of(
                    _extra_copy_unsafe
                ).address
                var _extra_ptr = AnyPointer[T]()
                _extra_ptr.value = __mlir_op.`pop.pointer.bitcast`[
                    _type = __mlir_type[
                        `!kgen.pointer<:`, CollectionElement, ` `, T, `>`
                    ]
                ](_extra_impl_ptr)
                # Calls the correct __copyinit__ finally (then __moveinit__)
                self._get_ptr[T]().emplace_value(
                    __get_address_as_lvalue(_extra_ptr.value)
                )

        unroll[len(VariadicList(Ts)), each]()

    fn __moveinit__(inout self, owned other: Self):
        """Move initializer for the variant.

        Args:
            other: The variant to move.
        """
        self._impl = Self._type()
        self._state = other._state

        @parameter
        fn each[i: Int]():
            if self._state == i:
                alias T = Ts[i]
                # Calls the correct __moveinit__
                self._get_ptr[T]().emplace_value(
                    other._get_ptr[T]().take_value()
                )

        unroll[len(VariadicList(Ts)), each]()

    fn __del__(owned self):
        """Destroy the variant."""

        @parameter
        fn each[i: Int]():
            if self._state == i:
                alias q = Ts[i]
                __get_address_as_owned_value(self._get_ptr[q]().value).__del__()

        unroll[len(VariadicList(Ts)), each]()

    fn _call_correct_deleter(inout self):
        @parameter
        fn each[i: Int]():
            if self._state == i:
                alias q = Ts[i]
                __get_address_as_owned_value(self._get_ptr[q]().value).__del__()

        unroll[len(VariadicList(Ts)), each]()

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
        debug_assert(Self._check[T]() == self._state, "taking wrong type")
        self._state = Self._sentinel  # don't call the variant's deleter later
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
        self._state = Self._check[T]()
        self._get_ptr[T]().emplace_value(value ^)

    fn isa[T: CollectionElement](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        alias idx = Self._check[T]()
        return self._state == idx

    fn get[T: CollectionElement](self) -> T:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - creates a copy of the internal data instead of returning a reference
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data cast as a T value.
        """
        debug_assert(self.isa[T](), "get: wrong variant type")
        return __get_address_as_lvalue(self._get_ptr[T]().value)

    @staticmethod
    fn _check[T: CollectionElement]() -> Int16:
        return _UnionTypeIndex[T, Ts].compute()
