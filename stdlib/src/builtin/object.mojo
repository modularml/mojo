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
"""Defines the object type, which is used to represent untyped values.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import Dict, List
from sys.intrinsics import _type_is_eq
from sys.ffi import OpaquePointer

from memory import Arc, memcmp, memcpy, UnsafePointer

from utils import StringRef, Variant

# ===----------------------------------------------------------------------=== #
# _ObjectImpl
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _NoneMarker(CollectionElementNew):
    """This is a trivial class to indicate that an object is `None`."""

    fn __init__(inout self, *, other: Self):
        pass


@value
struct _RefCountedList:
    """Python objects have the behavior that bool, int, float, and str are
    passed by value but lists and dictionaries are passed by reference. In order
    to model this behavior, lists and dictionaries are implemented as
    ref-counted data types.
    """

    var impl: Arc[List[object]]
    """The list value."""

    fn __init__(inout self):
        self.impl = Arc[List[object]](List[object]())


@value
struct _RefCountedAttrsDict:
    """This type contains the attribute dictionary for a dynamic object. The
    attribute dictionary is constructed once with a fixed number of elements.
    Those elements can be modified, but elements cannot be added or deleted
    after the dictionary is implemented. Because attribute are accessed
    directly with `x.attr`, the key will always be a `StringLiteral`.
    """

    var impl: Arc[Dict[StringLiteral, object]]
    """The implementation of the map."""

    fn __init__(inout self):
        self.impl = Arc[Dict[StringLiteral, object]](
            Dict[StringLiteral, object]()
        )

    @always_inline
    fn __init__(inout self, values: VariadicListMem[Attr, _]):
        self = Self()
        # Elements can only be added on construction.
        for i in range(len(values)):
            self.impl[]._insert(values[i].key, values[i].value)

    @always_inline
    fn __init__(inout self, values: List[Attr]):
        self = Self()
        # Elements can only be added on construction.
        for i in range(len(values)):
            self.impl[]._insert(values[i].key, values[i].value)

    @always_inline
    fn set(inout self, key: StringLiteral, value: object) raises:
        if key in self.impl[]:
            self.impl[][key] = value
            return
        raise Error(
            "AttributeError: Object does not have an attribute of name '"
            + key
            + "'"
        )

    @always_inline
    fn get(self, key: StringLiteral) raises -> object:
        var iter = self.impl[].find(key)
        if iter:
            return iter.value()
        raise Error(
            "AttributeError: Object does not have an attribute of name '"
            + key
            + "'"
        )


@value
struct Attr:
    """A generic object's attributes are set on construction, after which the
    attributes can be read and modified, but no attributes may be removed or
    added.
    """

    var key: StringLiteral
    """The name of the attribute."""
    var value: object
    """The value of the attribute."""

    @always_inline
    fn __init__(inout self, key: StringLiteral, owned value: object):
        """Initializes the attribute with a key and value.

        Args:
            key: The string literal key.
            value: The object value of the attribute.
        """
        self.key = key
        self.value = value^


@value
struct _RefCountedDict:
    """This type contains the dictionary implementation for a dynamic object."""

    var impl: Arc[Dict[object, object]]
    """The implementation of the map."""

    fn __init__(inout self):
        self.impl = Arc[Dict[object, object]](Dict[object, object]())

    @always_inline
    fn set(self, key: object, value: object) raises:
        self.impl[][key] = value

    @always_inline
    fn get(self, key: object) raises -> ref [self] object:
        return UnsafePointer.address_of(self.impl[].__getitem__(key))[]


@register_passable("trivial")
struct _Function(CollectionElement, CollectionElementNew):
    # The MLIR function type has two arguments:
    # 1. The self value, or the single argument.
    # 2. None, or an additional argument.
    var value: UnsafePointer[Int16]
    """The function pointer."""

    @always_inline
    fn __init__[FnT: AnyTrivialRegType](inout self, value: FnT):
        # FIXME: No "pointer bitcast" for signature function pointers.
        var f = UnsafePointer[Int16]()
        UnsafePointer.address_of(f).bitcast[FnT]()[] = value
        self.value = f

    @always_inline
    fn __init__(inout self, *, other: Self):
        self.value = other.value

    alias fn0 = fn () raises -> object
    """Nullary function type."""
    alias fn1 = fn (object) raises -> object
    """Unary function type."""
    alias fn2 = fn (object, object) raises -> object
    """Binary function type."""
    alias fn3 = fn (object, object, object) raises -> object
    """Ternary function type."""

    @always_inline
    fn invoke(owned self) raises -> object:
        return UnsafePointer.address_of(self.value).bitcast[Self.fn0]()[]()

    @always_inline
    fn invoke(owned self, arg0: object) raises -> object:
        return UnsafePointer.address_of(self.value).bitcast[Self.fn1]()[](arg0)

    @always_inline
    fn invoke(owned self, arg0: object, arg1: object) raises -> object:
        return UnsafePointer.address_of(self.value).bitcast[Self.fn2]()[](
            arg0, arg1
        )

    @always_inline
    fn invoke(
        owned self, arg0: object, arg1: object, arg2: object
    ) raises -> object:
        return UnsafePointer.address_of(self.value).bitcast[Self.fn3]()[](
            arg0, arg1, arg2
        )


struct _ObjectImpl(
    CollectionElement,
    CollectionElementNew,
    Stringable,
    Representable,
    Writable,
):
    """This class is the underlying implementation of the value of an `object`.
    It is a variant of primitive types and pointers to implementations of more
    complex types.

    We choose Int64 and Float64 to store all integer and float values respectively.
    TODO: These should be BigInt and BigFloat one day.
    """

    alias type = Variant[
        _NoneMarker,
        Bool,
        Int64,
        Float64,
        String,
        _RefCountedList,
        _Function,
        _RefCountedAttrsDict,
        _RefCountedDict,
    ]
    """The variant value type."""
    var value: Self.type
    """The value of the object. It is a variant of the possible object values
    kinds."""

    alias none: Int = 0
    """Type discriminator indicating none."""
    alias bool: Int = 1
    """Type discriminator indicating a bool."""
    alias int: Int = 2
    """Type discriminator indicating an int."""
    alias float: Int = 3
    """Type discriminator indicating a float."""
    alias str: Int = 4
    """Type discriminator indicating a string."""
    alias list: Int = 5
    """Type discriminator indicating a list."""
    alias dict: Int = 8
    """Type discriminator indicating a dictionary."""
    alias function: Int = 6
    """Type discriminator indicating a function."""
    alias obj: Int = 7
    """Type discriminator indicating an object."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(inout self, value: Self.type):
        self.value = value

    @always_inline
    fn __init__(inout self):
        self.value = Self.type(_NoneMarker {})

    @always_inline
    fn __init__(inout self, value: Bool):
        self.value = Self.type(value)

    @always_inline
    fn __init__[dt: DType](inout self, value: SIMD[dt, 1]):
        @parameter
        if dt.is_integral():
            self.value = Self.type(value)
        else:
            self.value = Self.type(value)

    @always_inline
    fn __init__(inout self, value: String):
        self.value = Self.type(value)

    @always_inline
    fn __init__(inout self, value: _RefCountedList):
        self.value = Self.type(value)

    @always_inline
    fn __init__(inout self, value: _Function):
        self.value = Self.type(value)

    @always_inline
    fn __init__(inout self, value: _RefCountedAttrsDict):
        self.value = Self.type(value)

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        self = other.value

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        self = existing.value

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self = other.value^

    # ===------------------------------------------------------------------=== #
    # Value Query
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn is_none(self) -> Bool:
        return self.value.isa[_NoneMarker]()

    @always_inline
    fn is_bool(self) -> Bool:
        return self.value.isa[Bool]()

    @always_inline
    fn is_int(self) -> Bool:
        return self.value.isa[Int64]()

    @always_inline
    fn is_float(self) -> Bool:
        return self.value.isa[Float64]()

    @always_inline
    fn is_str(self) -> Bool:
        return self.value.isa[String]()

    @always_inline
    fn is_list(self) -> Bool:
        return self.value.isa[_RefCountedList]()

    @always_inline
    fn is_dict(self) -> Bool:
        return self.value.isa[_RefCountedDict]()

    @always_inline
    fn is_func(self) -> Bool:
        return self.value.isa[_Function]()

    @always_inline
    fn is_obj(self) -> Bool:
        return self.value.isa[_RefCountedAttrsDict]()

    # get a copy
    @always_inline
    fn get_as_bool(self) -> Bool:
        return self.value[Bool]

    @always_inline
    fn get_as_int(self) -> Int64:
        return self.value[Int64]

    @always_inline
    fn get_as_float(self) -> Float64:
        return self.value[Float64]

    @always_inline
    fn get_as_string(self) -> String:
        return self.value[String]

    @always_inline
    fn get_as_list(ref [_]self) -> ref [self.value] _RefCountedList:
        return self.value[_RefCountedList]

    @always_inline
    fn get_as_func(self) -> _Function:
        return self.value[_Function]

    @always_inline
    fn get_obj_attrs(ref [_]self) -> ref [self.value] _RefCountedAttrsDict:
        return self.value[_RefCountedAttrsDict]

    @always_inline
    fn get_as_dict(ref [_]self) -> ref [self.value] _RefCountedDict:
        return self.value[_RefCountedDict]

    @always_inline
    fn get_type_id(self) -> Int:
        if self.is_none():
            return Self.none
        if self.is_bool():
            return Self.bool
        if self.is_int():
            return Self.int
        if self.is_float():
            return Self.float
        if self.is_str():
            return Self.str
        if self.is_list():
            return Self.list
        if self.is_func():
            return Self.function
        if self.is_dict():
            return Self.dict
        debug_assert(self.is_obj(), "expected a generic object")
        return Self.obj

    @always_inline
    fn _get_type_name(self) -> String:
        """Returns the name (in lowercase) of the specific object type."""
        if self.is_none():
            return "none"
        if self.is_bool():
            return "bool"
        if self.is_int():
            return "int"
        if self.is_float():
            return "float"
        if self.is_str():
            return "str"
        if self.is_list():
            return "list"
        if self.is_func():
            return "function"
        if self.is_dict():
            return "dict"
        debug_assert(self.is_obj(), "expected a generic object")
        return "obj"

    # ===------------------------------------------------------------------=== #
    # Type Conversion
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn convert_bool_to_float(self) -> Self:
        return Float64(1.0 if self.get_as_bool() else 0.0)

    @always_inline
    fn convert_bool_to_int(self) -> Self:
        return Int64(1 if self.get_as_bool() else 0)

    @always_inline
    fn convert_int_to_float(self) -> Self:
        return self.get_as_int().cast[DType.float64]()

    @staticmethod
    fn coerce_comparison_type(inout lhs: _ObjectImpl, inout rhs: _ObjectImpl):
        """Coerces two values of arithmetic type to the appropriate
        lowest-common denominator type for performing comparisons, in order of
        increasing priority: bool, int, and then float.
        """
        var lhsId = lhs.get_type_id()
        var rhsId = rhs.get_type_id()
        if lhsId == rhsId:
            return

        @parameter
        fn convert(inout value: _ObjectImpl, id: Int, to: Int):
            if to == Self.int:
                value = value.convert_bool_to_int()
            else:
                if id == Self.bool:
                    value = value.convert_bool_to_float()
                else:
                    value = value.convert_int_to_float()

        if lhsId > rhsId:
            convert(rhs, rhsId, lhsId)
        else:
            convert(lhs, lhsId, rhsId)

    @staticmethod
    fn coerce_arithmetic_type(inout lhs: _ObjectImpl, inout rhs: _ObjectImpl):
        """Coerces two values of arithmetic type to the appropriate
        lowest-common denominator type for performing arithmetic operations.
        Bools are always converted to integers, to match Python's behavior.
        """
        if lhs.is_bool():
            lhs = lhs.convert_bool_to_int()
        if rhs.is_bool():
            rhs = rhs.convert_bool_to_int()
        if lhs.is_float() == rhs.is_float():
            return
        if lhs.is_float():
            rhs = rhs.convert_int_to_float()
        else:
            lhs = lhs.convert_int_to_float()

    @staticmethod
    fn coerce_integral_type(inout lhs: _ObjectImpl, inout rhs: _ObjectImpl):
        """Coerces two values of integral type to the appropriate
        lowest-common denominator type for performing bitwise operations.
        """
        if lhs.is_int() == rhs.is_int():
            return
        if lhs.is_int():
            rhs = rhs.convert_bool_to_int()
        else:
            lhs = lhs.convert_bool_to_int()

    fn write_to[W: Writer](self, inout writer: W):
        """Performs conversion to string according to Python
        semantics.
        """
        if self.is_none():
            writer.write("None")
            return
        if self.is_bool():
            writer.write(str(self.get_as_bool()))
            return
        if self.is_int():
            writer.write(str(self.get_as_int()))
            return
        if self.is_float():
            writer.write(str(self.get_as_float()))
            return
        if self.is_str():
            writer.write("'", self.get_as_string(), "'")
            return
        if self.is_func():
            writer.write(
                "Function at address " + hex(int(self.get_as_func().value))
            )
            return
        if self.is_list():
            writer.write(String("["))
            for j in range(self.get_list_length()):
                if j != 0:
                    writer.write(", ")
                writer.write(repr(self.get_list_element(j)))
            writer.write("]")
            return

        if self.is_dict():
            var ptr = self.get_as_dict().impl
            writer.write(String("{"))
            var print_sep = False
            for entry in ptr[].items():
                if print_sep:
                    writer.write(", ")
                writer.write(
                    repr(entry[].key), " = ", repr(entry[].value)
                )
                print_sep = True
            writer.write("}")
            return

        var ptr = self.get_obj_attrs_ptr()
        writer.write(String("{"))
        var print_sep = False
        for entry in ptr[].items():
            if print_sep:
                writer.write(", ")
            writer.write(
                "'" + str(entry[].key) + "' = " + repr(entry[].value)
            )
            print_sep = True
        writer.write("}")
        return

    @no_inline
    fn __repr__(self) -> String:
        """Performs conversion to string according to Python
        semantics.

        Returns:
            The String representation of the object.
        """
        return self.__str__()

    @no_inline
    fn __str__(self) -> String:
        """Performs conversion to string according to Python
        semantics.

        Returns:
            The String representation of the object.
        """
        return String.write(self)

    # ===------------------------------------------------------------------=== #
    # List Functions
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn get_list_ptr(self) -> Arc[List[object]]:
        return self.get_as_list().impl

    @always_inline
    fn list_append(self, value: object):
        var ptr = self.get_list_ptr()
        ptr[].append(value)

    @always_inline
    fn get_list_length(self) -> Int:
        var ptr = self.get_list_ptr()
        return len(ptr[])

    @always_inline
    fn get_list_element(self, i: Int) -> object:
        var ptr = self.get_list_ptr()
        return ptr[][i]

    @always_inline
    fn set_list_element(self, i: Int, value: object):
        var ptr = self.get_list_ptr()
        ptr[][i] = value

    # ===------------------------------------------------------------------=== #
    # Object Attribute Functions
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn get_obj_attrs_ptr(self) -> Arc[Dict[StringLiteral, object]]:
        return self.get_obj_attrs().impl

    @always_inline
    fn set_obj_attr(self, key: StringLiteral, value: object) raises:
        self.get_obj_attrs_ptr()[][key] = value

    @always_inline
    fn get_obj_attr(self, key: StringLiteral) raises -> object:
        return self.get_obj_attrs_ptr()[][key]


# ===----------------------------------------------------------------------=== #
# object
# ===----------------------------------------------------------------------=== #


struct object(
    IntableRaising, ImplicitlyBoolable, Stringable, Representable, Writable
):
    """Represents an object without a concrete type.

    This is the type of arguments in `def` functions that do not have a type
    annotation, such as the type of `x` in `def f(x): pass`. A value of any type
    can be passed in as the `x` argument in this case, and so that value is
    used to construct this `object` type.
    """

    var _value: _ObjectImpl
    """The underlying value of the object."""

    alias nullary_function = _Function.fn0
    """Nullary function type."""
    alias unary_function = _Function.fn1
    """Unary function type."""
    alias binary_function = _Function.fn2
    """Binary function type."""
    alias ternary_function = _Function.fn3
    """Ternary function type."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(inout self):
        """Initializes the object with a `None` value."""
        self._value = _ObjectImpl()

    @always_inline
    fn __init__(inout self, impl: _ObjectImpl):
        """Initializes the object with an implementation value. This is meant for
        internal use only.

        Args:
            impl: The object implementation.
        """
        self._value = impl

    @always_inline
    fn __init__(inout self, none: NoneType):
        """Initializes a none value object from a `None` literal.

        Args:
            none: None.
        """
        self._value = _ObjectImpl()

    @always_inline
    fn __init__(inout self, value: Int):
        """Initializes the object with an integer value.

        Args:
            value: The integer value.
        """
        self._value = Int64(value)

    @always_inline
    fn __init__[dt: DType](inout self, value: SIMD[dt, 1]):
        """Initializes the object with a generic scalar value. If the scalar
        value type is bool, it is converted to a boolean. Otherwise, it is
        converted to the appropriate integer or floating point type.

        Parameters:
            dt: The scalar value type.

        Args:
            value: The scalar value.
        """

        @parameter
        if dt == DType.bool:
            self._value = value.__bool__()
        else:
            self._value = value

    @always_inline
    fn __init__(inout self, value: Bool):
        """Initializes the object from a bool.

        Args:
            value: The boolean value.
        """
        self._value = value

    @always_inline
    fn __init__(inout self, value: String):
        """Initializes the object from a string.

        Args:
            value: The string value.
        """
        self._value = value

    @always_inline
    fn __init__(inout self, value: StringLiteral):
        """Initializes the object from a string literal.

        Args:
            value: The string value.
        """
        self = object(StringRef(value))

    @always_inline
    fn __init__(inout self, value: StringRef):
        """Initializes the object from a string reference.

        Args:
            value: The string value.
        """
        self._value = String(value)

    @always_inline
    fn __init__[*Ts: CollectionElement](inout self, value: ListLiteral[*Ts]):
        """Initializes the object from a list literal.

        Parameters:
            Ts: The list element types.

        Args:
            value: The list value.
        """
        self._value = _RefCountedList()

        @parameter
        for i in range(len(VariadicList(Ts))):
            # We need to rebind the element to one we know how to convert from.
            # FIXME: This doesn't handle implicit conversions or nested lists.
            alias T = Ts[i]

            @parameter
            if _type_is_eq[T, Int]():
                self._append(value.get[i, Int]())
            elif _type_is_eq[T, Float64]():
                self._append(value.get[i, Float64]())
            elif _type_is_eq[T, Bool]():
                self._append(value.get[i, Bool]())
            elif _type_is_eq[T, String]():
                self._append(value.get[i, String]())
            elif _type_is_eq[T, StringRef]():
                self._append(value.get[i, StringRef]())
            elif _type_is_eq[T, StringLiteral]():
                self._append(value.get[i, StringLiteral]())
            else:
                constrained[
                    False, "cannot convert nested list element to object"
                ]()

    @always_inline
    fn __init__(inout self, func: Self.nullary_function):
        """Initializes an object from a function that takes no arguments.

        Args:
            func: The function.
        """
        self._value = _Function(func)

    @always_inline
    fn __init__(inout self, func: Self.unary_function):
        """Initializes an object from a function that takes one argument.

        Args:
            func: The function.
        """
        self._value = _Function(func)

    @always_inline
    fn __init__(inout self, func: Self.binary_function):
        """Initializes an object from a function that takes two arguments.

        Args:
            func: The function.
        """
        self._value = _Function(func)

    @always_inline
    fn __init__(inout self, func: Self.ternary_function):
        """Initializes an object from a function that takes three arguments.

        Args:
            func: The function.
        """
        self._value = _Function(func)

    @always_inline
    fn __init__(inout self, *attrs: Attr):
        """Initializes the object with a sequence of zero or more attributes.

        Args:
            attrs: Zero or more attributes.
        """
        self._value = _RefCountedAttrsDict(attrs)

    fn __init__(inout self, attrs: List[Attr]):
        """Initializes the object with a list of zero or more attributes.

        Args:
            attrs: Zero or more attributes.
        """
        self._value = _RefCountedAttrsDict(attrs)

    @always_inline
    fn __moveinit__(inout self, owned existing: object):
        """Move the value of an object.

        Args:
            existing: The object to move.
        """
        self._value = existing._value
        existing._value = _ObjectImpl()

    @always_inline
    fn __copyinit__(inout self, existing: object):
        """Copies the object. This clones the underlying string value and
        increases the refcount of lists or dictionaries.

        Args:
            existing: The object to copy.
        """
        self._value = existing._value

    @always_inline
    fn __del__(owned self):
        """Delete the object and release any owned memory."""
        ...

    # ===------------------------------------------------------------------=== #
    # Conversion
    # ===------------------------------------------------------------------=== #

    fn __bool__(self) -> Bool:
        """Performs conversion to bool according to Python semantics. Integers
        and floats are true if they are non-zero, and strings and lists are true
        if they are non-empty.

        Returns:
            Whether the object is considered true.
        """
        if self._value.is_bool():
            return self._value.get_as_bool()
        # Integers or floats are true if they are non-zero.
        if self._value.is_int():
            return (self._value.get_as_int() != 0).__bool__()
        if self._value.is_float():
            return (self._value.get_as_float() != 0.0).__bool__()
        if self._value.is_str():
            # Strings are true if they are non-empty.
            return bool(self._value.get_as_string())
        debug_assert(self._value.is_list(), "expected a list")
        return self._value.get_list_length() != 0

    fn __int__(self) raises -> Int:
        """Performs conversion to integer according to Python
        semantics.

        Returns:
            The Int representation of the object.
        """
        if self._value.is_bool():
            return 1 if self._value.get_as_bool() else 0

        if self._value.is_int():
            return int(self._value.get_as_int())

        if self._value.is_float():
            return int(self._value.get_as_float())

        if self._value.is_str():
            return int(self._value.get_as_string())

        raise "object type cannot be converted to an integer"

    fn __as_bool__(self) -> Bool:
        """Performs conversion to bool according to Python semantics. Integers
        and floats are true if they are non-zero, and strings and lists are true
        if they are non-empty.

        Returns:
            Whether the object is considered true.
        """
        return self.__bool__()

    fn write_to[W: Writer](self, inout writer: W):
        """Performs conversion to string according to Python
        semantics.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The Writer to write to.
        """
        self._value.write_to(writer)

    @no_inline
    fn __str__(self) -> String:
        """Performs conversion to string according to Python
        semantics.

        Returns:
            The String representation of the object.
        """
        if self._value.is_str():
            return self._value.get_as_string()
        return String.write(self._value)

    @no_inline
    fn __repr__(self) -> String:
        """Performs conversion to string according to Python
        semantics.

        Returns:
            The String representation of the object.
        """
        return repr(self._value)

    # ===------------------------------------------------------------------=== #
    # Comparison Operators
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn _comparison_type_check(self) raises:
        """Throws an error if the object cannot be arithmetically compared."""
        if not (
            self._value.is_bool()
            or self._value.is_int()
            or self._value.is_float()
        ):
            raise Error("TypeError: not a valid comparison type")

    @staticmethod
    @always_inline
    fn _comparison_op[
        fp_func: fn (Float64, Float64) -> Scalar[DType.bool],
        int_func: fn (Int64, Int64) -> Scalar[DType.bool],
        bool_func: fn (Bool, Bool) -> Bool,
    ](lhs: object, rhs: object) raises -> object:
        """Dispatches comparison operator depending on the type.

        Parameters:
            fp_func: Floating point comparator.
            int_func: Integer comparator.
            bool_func: Boolean comparator.

        Args:
            lhs: The left hand value.
            rhs: The right hand value.

        Returns:
            The comparison result.
        """
        lhs._comparison_type_check()
        rhs._comparison_type_check()
        var lhsValue = lhs._value
        var rhsValue = rhs._value
        _ObjectImpl.coerce_comparison_type(lhsValue, rhsValue)
        if lhsValue.is_float():
            return fp_func(lhsValue.get_as_float(), rhsValue.get_as_float())
        if lhsValue.is_int():
            return int_func(lhsValue.get_as_int(), rhsValue.get_as_int())
        debug_assert(lhsValue.is_bool(), "expected both values to be bool")
        return bool_func(lhsValue.get_as_bool(), rhsValue.get_as_bool())

    @always_inline
    fn _list_compare(self, rhs: object) raises -> Int:
        var llen = self._value.get_list_length()
        var rlen = self._value.get_list_length()
        var cmp_len = min(llen, rlen)
        for i in range(cmp_len):
            var lelt: object = self._value.get_list_element(i)
            var relt: object = rhs._value.get_list_element(i)
            if lelt < relt:
                return -1
            if lelt > relt:
                return 1
        if llen < rlen:
            return -1
        if llen > rlen:
            return 1
        return 0

    fn __lt__(self, rhs: object) raises -> object:
        """Less-than comparator. This lexicographically compares strings and
        lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the object is less than the right hard argument.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._value.get_as_string() < rhs._value.get_as_string()

        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) < 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return not lhs and rhs

        return Self._comparison_op[Float64.__lt__, Int64.__lt__, bool_fn](
            self, rhs
        )

    fn __le__(self, rhs: object) raises -> object:
        """Less-than-or-equal to comparator. This lexicographically
        compares strings and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the object is less than or equal to the right hard argument.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._value.get_as_string() <= rhs._value.get_as_string()

        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) <= 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs == rhs or not lhs

        return Self._comparison_op[Float64.__le__, Int64.__le__, bool_fn](
            self, rhs
        )

    fn __eq__(self, rhs: object) -> Bool:
        """Equality comparator. This compares the elements of strings
        and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the objects are equal.
        """
        try:
            if self._value.is_str() and rhs._value.is_str():
                return self._value.get_as_string() == rhs._value.get_as_string()
            if self._value.is_list() and rhs._value.is_list():
                return self._list_compare(rhs) == 0

            @always_inline
            fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
                return lhs == rhs

            c = Self._comparison_op[Float64.__eq__, Int64.__eq__, bool_fn](
                self, rhs
            )
            return bool(c)
        except e:
            print(e)
            print("objects are not comparable")
        return False

    fn __ne__(self, rhs: object) -> Bool:
        """Inequality comparator. This compares the elements of strings
        and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the objects are not equal.
        """
        try:
            return not (self == rhs)
        except e:
            print(e)
            print("objects are not comparable")
        return True

    fn __gt__(self, rhs: object) raises -> object:
        """Greater-than comparator. This lexicographically compares the
        elements of strings and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the left hand value is greater.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._value.get_as_string() > rhs._value.get_as_string()
        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) > 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs and not rhs

        return Self._comparison_op[Float64.__gt__, Int64.__gt__, bool_fn](
            self, rhs
        )

    fn __ge__(self, rhs: object) raises -> object:
        """Greater-than-or-equal-to comparator. This lexicographically
        compares the elements of strings and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the left hand value is greater than or equal to the right
            hand value.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._value.get_as_string() >= rhs._value.get_as_string()
        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) >= 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs == rhs or lhs

        return Self._comparison_op[Float64.__ge__, Int64.__ge__, bool_fn](
            self, rhs
        )

    # ===------------------------------------------------------------------=== #
    # Arithmetic Operators
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn _arithmetic_type_check(self) raises:
        """Throws an error if the object is not arithmetic."""
        if not (
            self._value.is_bool()
            or self._value.is_int()
            or self._value.is_float()
        ):
            raise Error("TypeError: not a valid arithmetic type")

    @always_inline
    fn _arithmetic_integral_type_check(self) raises:
        """Throws an error if the object is not an integral type."""
        if not (self._value.is_bool() or self._value.is_int()):
            raise Error("TypeError: not a valid integral type")

    @staticmethod
    @always_inline
    fn _arithmetic_binary_op[
        fp_func: fn (Float64, Float64) -> Float64,
        int_func: fn (Int64, Int64) -> Int64,
    ](lhs: object, rhs: object) raises -> object:
        """Generic arithmetic operator. Bool values are treated as
        integers in arithmetic operators.

        Parameters:
            fp_func: Floating point operator.
            int_func: Integer operator.

        Returns:
            The arithmetic operation result.
        """
        lhs._arithmetic_type_check()
        rhs._arithmetic_type_check()
        var lhsValue = lhs._value
        var rhsValue = rhs._value
        _ObjectImpl.coerce_arithmetic_type(lhsValue, rhsValue)
        if lhsValue.is_float():
            return fp_func(lhsValue.get_as_float(), rhsValue.get_as_float())
        return int_func(lhsValue.get_as_int(), rhsValue.get_as_int())

    @staticmethod
    @always_inline
    fn _arithmetic_bitwise_op[
        int_func: fn (Int64, Int64) -> Int64,
        bool_func: fn (Bool, Bool) -> Bool,
    ](lhs: object, rhs: object) raises -> object:
        """Generic bitwise operator.

        Parameters:
            int_func: Integer operator.
            bool_func: Boolean operator.

        Returns:
            The bitwise operation result.
        """
        lhs._arithmetic_integral_type_check()
        rhs._arithmetic_integral_type_check()
        var lhsValue = lhs._value
        var rhsValue = rhs._value
        _ObjectImpl.coerce_integral_type(lhsValue, rhsValue)
        if lhsValue.is_int():
            return int_func(lhsValue.get_as_int(), rhsValue.get_as_int())
        return bool_func(lhsValue.get_as_bool(), rhsValue.get_as_bool())

    @always_inline
    fn __neg__(self) raises -> object:
        """Negation operator. Only valid for bool, int, and float
        types. Negation on any bool value converts it to an integer.

        Returns:
            The negative of the current value.
        """
        if self._value.is_bool():
            return -self._value.convert_bool_to_int().get_as_int()
        if self._value.is_int():
            return -self._value.get_as_int()
        if self._value.is_float():
            return -self._value.get_as_float()
        raise Error("TypeError: cannot apply negation to this type")

    @always_inline
    fn __invert__(self) raises -> object:
        """Invert value operator. This is only valid for bool and int
        values.

        Returns:
            The inverted value.
        """
        if self._value.is_bool():
            return ~self._value.get_as_bool()
        if self._value.is_int():
            return ~self._value.get_as_int()
        raise Error("TypeError: cannot invert values of this type")

    @always_inline
    fn __add__(self, rhs: object) raises -> object:
        """Addition and concatenation operator. For arithmetic types, this
        function will compute the sum of the left and right hand values. For
        strings and lists, this function will concat the objects.

        Args:
            rhs: Right hand value.

        Returns:
            The sum or concatenated values.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._value.get_as_string() + rhs._value.get_as_string()
        if self._value.is_list() and rhs._value.is_list():
            var result2 = object([])
            for i in range(self.__len__()):
                result2.append(self[i])
            for j in range(rhs.__len__()):
                result2.append(rhs[j])
            return result2

        return Self._arithmetic_binary_op[Float64.__add__, Int64.__add__](
            self, rhs
        )

    @always_inline
    fn __sub__(self, rhs: object) raises -> object:
        """Subtraction operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The difference.
        """
        return Self._arithmetic_binary_op[Float64.__sub__, Int64.__sub__](
            self, rhs
        )

    @always_inline
    fn __mul__(self, rhs: object) raises -> object:
        """Multiplication operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The product.
        """
        return Self._arithmetic_binary_op[Float64.__mul__, Int64.__mul__](
            self, rhs
        )

    @always_inline
    fn __pow__(self, exp: object) raises -> object:
        """Exponentiation operator. Valid only for arithmetic types.

        Args:
            exp: Exponent value.

        Returns:
            The left hand value raised to the power of the right hand value.
        """
        return Self._arithmetic_binary_op[Float64.__pow__, Int64.__pow__](
            self, exp
        )

    @always_inline
    fn __mod__(self, rhs: object) raises -> object:
        """Modulo operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value mod the right hand value.
        """
        return Self._arithmetic_binary_op[Float64.__mod__, Int64.__mod__](
            self, rhs
        )

    @always_inline
    fn __truediv__(self, rhs: object) raises -> object:
        """True division operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value true divide the right hand value.
        """
        return Self._arithmetic_binary_op[
            Float64.__truediv__, Int64.__truediv__
        ](self, rhs)

    @always_inline
    fn __floordiv__(self, rhs: object) raises -> object:
        """Floor division operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value floor divide the right hand value.
        """
        return Self._arithmetic_binary_op[
            Float64.__floordiv__, Int64.__floordiv__
        ](self, rhs)

    @always_inline
    fn __lshift__(self, rhs: object) raises -> object:
        """Left shift operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value left shifted by the right hand value.
        """
        self._arithmetic_integral_type_check()
        rhs._arithmetic_integral_type_check()
        return object(self._value.get_as_int() << rhs._value.get_as_int())

    @always_inline
    fn __rshift__(self, rhs: object) raises -> object:
        """Right shift operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value right shifted by the right hand value.
        """
        self._arithmetic_integral_type_check()
        rhs._arithmetic_integral_type_check()
        return object(self._value.get_as_int() >> rhs._value.get_as_int())

    @always_inline
    fn __and__(self, rhs: object) raises -> object:
        """Bitwise AND operator.

        Args:
            rhs: Right hand value.

        Returns:
            The current value if it is False.
        """
        return Self._arithmetic_bitwise_op[Int64.__and__, Bool.__and__](
            self, rhs
        )

    @always_inline
    fn __or__(self, rhs: object) raises -> object:
        """Bitwise OR operator.

        Args:
            rhs: Right hand value.

        Returns:
            The current value if it is True.
        """
        return Self._arithmetic_bitwise_op[Int64.__or__, Bool.__or__](self, rhs)

    @always_inline
    fn __xor__(self, rhs: object) raises -> object:
        """Bitwise XOR operator.

        Args:
            rhs: Right hand value.

        Returns:
            The current value if it is True.
        """
        return Self._arithmetic_bitwise_op[Int64.__xor__, Bool.__xor__](
            self, rhs
        )

    # ===------------------------------------------------------------------=== #
    # In-Place Operators
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __iadd__(inout self, rhs: object) raises:
        """In-place addition or concatenation operator.

        Args:
            rhs: Right hand value.
        """
        self = self + rhs

    @always_inline
    fn __isub__(inout self, rhs: object) raises:
        """In-place subtraction operator.

        Args:
            rhs: Right hand value.
        """
        self = self - rhs

    @always_inline
    fn __imul__(inout self, rhs: object) raises:
        """In-place multiplication operator.

        Args:
            rhs: Right hand value.
        """
        self = self * rhs

    @always_inline
    fn __ipow__(inout self, rhs: object) raises:
        """In-place exponentiation operator.

        Args:
            rhs: Right hand value.
        """
        self = self**rhs

    @always_inline
    fn __imod__(inout self, rhs: object) raises:
        """In-place modulo operator.

        Args:
            rhs: Right hand value.
        """
        self = self % rhs

    @always_inline
    fn __itruediv__(inout self, rhs: object) raises:
        """In-place true division operator.

        Args:
            rhs: Right hand value.
        """
        self = self / rhs

    @always_inline
    fn __ifloordiv__(inout self, rhs: object) raises:
        """In-place floor division operator.

        Args:
            rhs: Right hand value.
        """
        self = self // rhs

    @always_inline
    fn __ilshift__(inout self, rhs: object) raises:
        """In-place left shift operator.

        Args:
            rhs: Right hand value.
        """
        self = self << rhs

    @always_inline
    fn __irshift__(inout self, rhs: object) raises:
        """In-place right shift operator.

        Args:
            rhs: Right hand value.
        """
        self = self >> rhs

    @always_inline
    fn __iand__(inout self, rhs: object) raises:
        """In-place AND operator.

        Args:
            rhs: Right hand value.
        """
        self = self & rhs

    @always_inline
    fn __ior__(inout self, rhs: object) raises:
        """In-place OR operator.

        Args:
            rhs: Right hand value.
        """
        self = self | rhs

    @always_inline
    fn __ixor__(inout self, rhs: object) raises:
        """In-place XOR operator.

        Args:
            rhs: Right hand value.
        """
        self = self ^ rhs

    # ===------------------------------------------------------------------=== #
    # Reversed Operators
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __radd__(self, lhs: object) raises -> object:
        """Reverse addition or concatenation operator.

        Args:
            lhs: Left hand value.

        Returns:
            The sum or concatenated value.
        """
        return lhs + self

    @always_inline
    fn __rsub__(self, lhs: object) raises -> object:
        """Reverse subtraction operator.

        Args:
            lhs: Left hand value.

        Returns:
            The result of subtracting this from the left-hand-side value.
        """
        return lhs - self

    @always_inline
    fn __rmul__(self, lhs: object) raises -> object:
        """Reverse multiplication operator.

        Args:
            lhs: Left hand value.

        Returns:
            The product.
        """
        return lhs * self

    @always_inline
    fn __rpow__(self, lhs: object) raises -> object:
        """Reverse exponentiation operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value raised to the power of the right hand value.
        """
        return lhs**self

    @always_inline
    fn __rmod__(self, lhs: object) raises -> object:
        """Reverse modulo operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value mod the right hand value.
        """
        return lhs % self

    @always_inline
    fn __rtruediv__(self, lhs: object) raises -> object:
        """Reverse true division operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value divide the right hand value.
        """
        return lhs / self

    @always_inline
    fn __rfloordiv__(self, lhs: object) raises -> object:
        """Reverse floor division operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value floor divide the right hand value.
        """
        return lhs // self

    @always_inline
    fn __rlshift__(self, lhs: object) raises -> object:
        """Reverse left shift operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value left shifted by the right hand value.
        """
        return lhs << self

    @always_inline
    fn __rrshift__(self, lhs: object) raises -> object:
        """Reverse right shift operator.

        Args:
            lhs: Left hand value.

        Returns:
            The left hand value right shifted by the right hand value.
        """
        return lhs >> self

    @always_inline
    fn __rand__(self, lhs: object) raises -> object:
        """Reverse AND operator.

        Args:
            lhs: Left hand value.

        Returns:
            The bitwise AND of the left-hand-side value and this.
        """
        return lhs & self

    @always_inline
    fn __ror__(self, lhs: object) raises -> object:
        """Reverse OR operator.

        Args:
            lhs: Left hand value.

        Returns:
            The bitwise OR of the left-hand-side value and this.
        """
        return lhs | self

    @always_inline
    fn __rxor__(self, lhs: object) raises -> object:
        """Reverse XOR operator.

        Args:
            lhs: Left hand value.

        Returns:
            The bitwise XOR of the left-hand-side value and this.
        """
        return lhs ^ self

    # ===------------------------------------------------------------------=== #
    # Interface
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn append(self, value: object) raises:
        """Appends a value to the list.

        Args:
            value: The value to append.
        """
        if self._value.is_obj():
            _ = self._value.get_obj_attr("append")(self, value)
            return
        if not self._value.is_list():
            raise Error("TypeError: can only append to lists")
        self._append(value)

    @always_inline
    fn _append(self, value: object):
        self._value.list_append(value)

    @always_inline
    fn __len__(self) raises -> Int:
        """Returns the "length" of the object. Only strings, lists, and
        dictionaries have lengths.

        Returns:
            The length of the string value or the number of elements in the list
            or dictionary value.
        """
        if self._value.is_str():
            return len(self._value.get_as_string())
        if self._value.is_list():
            return self._value.get_list_length()
        if self._value.is_dict():
            return len(self._value.get_as_dict().impl[])
        raise Error("TypeError: only strings, lists and dict have length")

    @staticmethod
    @always_inline
    fn _convert_index_to_int(i: object) raises -> Int:
        if i._value.is_bool():
            return i._value.convert_bool_to_int().get_as_int().value
        elif not i._value.is_int():
            raise Error("TypeError: string indices must be integers")
        return i._value.get_as_int().value

    @always_inline
    fn __getitem__(self, i: object) raises -> object:
        """Gets the i-th item from the object. This is only valid for strings,
        lists, and dictionaries.

        Args:
            i: The string or list index, or dictionary key.

        Returns:
            The value at the index or key.
        """
        if self._value.is_obj():
            return self._value.get_obj_attr("__getitem__")(self, i)

        if self._value.is_dict():
            return self._value.get_as_dict().get(i)

        if not self._value.is_str() and not self._value.is_list():
            raise Error("TypeError: can only index into lists and strings")

        var index = Self._convert_index_to_int(i)
        if self._value.is_str():
            # Construct a new single-character string.
            return self._value.get_as_string()[index]
        return self._value.get_list_element(i._value.get_as_int().value)

    @always_inline
    fn __getitem__(self, *index: object) raises -> object:
        """Gets the i-th item from the object, where i is a tuple of indices.

        Args:
            index: A compound index.

        Returns:
            The value at the index.
        """
        var value = self
        for i in index:
            value = value[i[]]
        return value

    @always_inline
    fn __setitem__(self, i: object, value: object) raises -> None:
        """Sets the i-th item in the object. This is only valid for strings,
        lists, and dictionaries.

        Args:
            i: The string or list index, or dictionary key.
            value: The value to set.
        """
        if self._value.is_obj():
            _ = self._value.get_obj_attr("__setitem__")(self, i, value)
            return
        if self._value.is_dict():
            self._value.get_as_dict().set(i, value)
            return
        if self._value.is_str():
            raise Error(
                "TypeError: 'str' object does not support item assignment"
            )
        if not self._value.is_list():
            raise Error("TypeError: can only assign items in lists")
        var index = Self._convert_index_to_int(i)
        self._value.set_list_element(index.value, value)

    @always_inline
    fn __setitem__(self, i: object, j: object, value: object) raises:
        """Sets the (i, j)-th element in the object.

        FIXME: We need this because `obj[i, j] = value` will attempt to invoke
        this method with 3 arguments, and we can only have variadics as the last
        argument.

        Args:
            i: The first index.
            j: The second index.
            value: The value to set.
        """
        self[i][j] = value

    @always_inline
    fn __getattr__(self, key: StringLiteral) raises -> object:
        """Gets the named attribute.

        Args:
            key: The attribute name.

        Returns:
            The attribute value.
        """
        if not self._value.is_obj():
            raise Error(
                "TypeError: Type '"
                + self._value._get_type_name()
                + "' does not have attribute '"
                + key
                + "'"
            )
        return self._value.get_obj_attr(key)

    @always_inline
    fn __setattr__(inout self, key: StringLiteral, value: object) raises:
        """Sets the named attribute.

        Args:
            key: The attribute name.
            value: The attribute value.
        """
        if not self._value.is_obj():
            raise Error(
                "TypeError: Type '"
                + self._value._get_type_name()
                + "' does not have attribute '"
                + key
                + "'"
            )
        self._value.set_obj_attr(key, value)

    @always_inline
    fn __call__(self) raises -> object:
        """Calls the object as a function.

        Returns:
            The function return value, as an object.
        """
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke()

    @always_inline
    fn __call__(self, arg0: object) raises -> object:
        """Calls the object as a function.

        Args:
            arg0: The first function argument.

        Returns:
            The function return value, as an object.
        """
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0)

    @always_inline
    fn __call__(self, arg0: object, arg1: object) raises -> object:
        """Calls the object as a function.

        Args:
            arg0: The first function argument.
            arg1: The second function argument.

        Returns:
            The function return value, as an object.
        """
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0, arg1)

    @always_inline
    fn __call__(
        self, arg0: object, arg1: object, arg2: object
    ) raises -> object:
        """Calls the object as a function.

        Args:
            arg0: The first function argument.
            arg1: The second function argument.
            arg2: The third function argument.

        Returns:
            The function return value, as an object.
        """
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0, arg1, arg2)

    fn __contains__(self, value: Self) raises -> Bool:
        """Returns `True` if value is in `self` (`dict` key or `list` element).

        Args:
            value: The value that is compared.

        Example:
        ```mojo
        a = object([1, "two"])
        if "two" in a:
            print("two is in a")
        ```

        Returns:
            A `Bool` (`True` or `False`).
        """
        if self._value.is_list():
            return value in self._value.get_as_list().impl[]
        if self._value.is_dict():
            return value in self._value.get_as_dict().impl[]
        raise "only lists and dict implements the __contains__ dunder"

    fn pop(self, value: Self) raises -> object:
        """Removes an element and returns it.

        Args:
            value: The index for lists, the key for dicts.

        Returns:
            The element.

        Example
        ```mojo
        x = object.dict()
        x["one"] = 1
        x["two"] = 2
        y = x.pop("two")
        print(y == 2)
        ```
        Note: implemented for list and dictionary.
        """
        # TODO: argument with None as default for pop()
        if self._value.is_list():
            if value._value.is_int():
                tmp_i = int(value._value.get_as_int())
                self_len = self._value.get_list_length()
                if tmp_i < self_len and tmp_i >= 0:
                    ret_val = self._value.get_as_list().impl[].pop(tmp_i)
                    return ret_val
                else:
                    raise "Index should be < len and >= 0"
            else:
                raise "List uses non float numbers as indexes"
        if self._value.is_dict():
            try:
                tmp_ret = self._value.get_as_dict().impl[].pop(value)
                return tmp_ret
            except e:
                raise e
        raise "self is not a list or a dict"

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #
    @staticmethod
    fn dict() -> Self:
        """Construct an empty dictionary.

        Returns:
            The constructed empty dictionary.

        Example:
        ```mojo
        x = object.dict()
        x["one"] = 1
        x[2] = "two"
        x[3.0] = [3.0]
        ```

        Note: supported key types are `Float64`, `Int64` and `String`.

        Values can be object of any types.
        """
        return Self(_ObjectImpl(_RefCountedDict()))

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #
    fn __hash__(self) -> UInt:
        """Compute the hash of `self`.

        Returns:
            An `UInt`.

        Note: `hash(repr(self))` is returned if `self` type is not hashable.

        The hashables types are `String`, `Int64` and `Float64`.
        """
        if self._value.is_str():
            return hash(self._value.get_as_string())
        if self._value.is_int():
            return hash(self._value.get_as_int())
        if self._value.is_float():
            return hash(self._value.get_as_float())
        # FIXME: hash(repr(self)) as fallback
        return hash(repr(self))
