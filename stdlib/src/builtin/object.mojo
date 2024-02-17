# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines the object type, which is used to represent untyped values.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.vector import DynamicVector
from math import min as _min
from os.atomic import Atomic
from sys.intrinsics import _mlirtype_is_eq

from algorithm.functional import unroll
from memory import memcmp, memcpy
from memory.unsafe import DTypePointer, Pointer

from .io import _printf, _put

# ===----------------------------------------------------------------------=== #
# _ObjectImpl
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _NoneMarker:
    """This is a trivial class to indicate that an object is `None`."""

    pass


@register_passable("trivial")
struct _ImmutableString:
    """Python strings are immutable. This class is marked as trivially register
    passable because its memory will be managed by `_ObjectImpl`. It is a
    pointer and integer pair. Memory will be dynamically allocated.
    """

    var data: Pointer[Int8]
    """The pointer to the beginning of the string contents. It is not
    null-terminated."""
    var length: Int
    """The length of the string."""

    @always_inline
    fn __init__(data: Pointer[Int8], length: Int) -> Self:
        return Self {data: data, length: length}

    @always_inline
    fn string_compare(self, rhs: _ImmutableString) -> Int:
        var res = memcmp(self.data, rhs.data, _min(self.length, rhs.length))
        if res != 0:
            return -1 if res < 0 else 1
        if self.length == rhs.length:
            return 0
        return -1 if self.length < rhs.length else 1


struct _RefCountedList:
    """Python objects have the behaviour that bool, int, float, and str are
    passed by value but lists and dictionaries are passed by reference. In order
    to model this behaviour, lists and dictionaries are implemented as
    ref-counted data types.
    """

    fn __init__(inout self):
        self.refcount = 1
        self.impl = DynamicVector[_ObjectImpl]()

    var refcount: Atomic[DType.index]
    """The number of live references to the list."""
    var impl: DynamicVector[_ObjectImpl]
    """The list value."""


@register_passable("trivial")
struct _RefCountedListRef:
    # FIXME(#3335): Use indirection to avoid a recursive struct definition.
    var lst: Pointer[NoneType]
    """The reference to the list."""

    @always_inline
    fn __init__() -> Self:
        var ptr = Pointer[_RefCountedList].alloc(1)
        __get_address_as_uninit_lvalue(ptr.address) = _RefCountedList()
        return Self {lst: ptr.bitcast[NoneType]()}

    @always_inline
    fn copy(self) -> Self:
        _ = __get_address_as_lvalue(
            self.lst.bitcast[_RefCountedList]().address
        ).refcount.fetch_add(1)
        return Self {lst: self.lst}

    fn release(self):
        var ptr = self.lst.bitcast[_RefCountedList]()
        var prev = __get_address_as_lvalue(ptr.address).refcount.fetch_sub(1)
        if prev != 1:
            return

        # Run the destructor on the list elements and then destroy the list.
        var list = __get_address_as_owned_value(ptr.address).impl
        for i in range(len(list)):
            list[i].destroy()
        ptr.free()


@register_passable
struct _AttrsDictEntry(CollectionElement):
    var key: StringLiteral
    """The dictionary entry key. It is always a string literal."""
    var value: _ObjectImpl
    """The dictionary entry value."""

    fn __copyinit__(existing: Self) -> Self:
        return Self {key: existing.key, value: existing.value}


struct _RefCountedAttrsDict:
    """This type contains the attribute dictionary for a dynamic object. The
    attribute dictionary is constructed once with a fixed number of elements.
    Those elements can be modified, but elements cannot be added or devared
    after the dictionary is implemented. Because attribute are accessed
    directly with `x.attr`, the key will always be a `StringLiteral`. Mojo
    string literals are uniqued by the compiler, so we can compare pointers for
    equality.
    """

    var refcount: Atomic[DType.index]
    """The number of live references to the attribute dictionary."""
    # TODO: Actually implement a hash map. This is a list of pairs.
    var impl: DynamicVector[_AttrsDictEntry]
    """The implementation of the map. It is a list of pairs."""

    fn __init__(inout self):
        self.refcount = 1
        self.impl = DynamicVector[_AttrsDictEntry]()

    @always_inline
    fn set(inout self, key: StringLiteral, value: _ObjectImpl) raises:
        for i in range(len(self.impl)):
            var cur = self.impl[i]
            if cur.key.data() == key.data():
                self.impl[i].value.destroy()
                self.impl[i] = _AttrsDictEntry {key: key, value: value}
                return
        raise Error(
            "AttributeError: Object does not have an attribute of this name"
        )

    @always_inline
    fn get(self, key: StringLiteral) raises -> _ObjectImpl:
        for i in range(len(self.impl)):
            var cur = self.impl[i]
            if cur.key.data() == key.data():
                return cur.value
        raise Error(
            "AttributeError: Object does not have an attribute of this name"
        )


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
        self.value = value ^


@register_passable("trivial")
struct _RefCountedAttrsDictRef:
    # FIXME(#3335): Use indirection to avoid a recursive struct definition.
    # FIXME(#12604): Distinguish this type from _RefCountedListRef.
    var attrs: Pointer[Int8]
    """The reference to the dictionary."""

    @always_inline
    fn __init__[  # FIXME(#29464): Should use autoparameterization.
        elt_is_mutable: __mlir_type.i1,
        lifetime: AnyLifetime[elt_is_mutable].type,
    ](values: VariadicListMem[Attr, elt_is_mutable, lifetime]) -> Self:
        var ptr = Pointer[_RefCountedAttrsDict].alloc(1)
        __get_address_as_uninit_lvalue(ptr.address) = _RefCountedAttrsDict()
        # Elements can only be added on construction.
        for i in range(len(values)):
            var entry = _AttrsDictEntry {
                key: values[i].key,
                value: values[i].value._value.copy(),
            }
            __get_address_as_lvalue(ptr.address).impl.push_back(entry)
        return Self {attrs: ptr.bitcast[Int8]()}

    @always_inline
    fn copy(self) -> Self:
        _ = __get_address_as_lvalue(
            self.attrs.bitcast[_RefCountedAttrsDict]().address
        ).refcount.fetch_add(1)
        return Self {attrs: self.attrs}

    fn release(self):
        var ptr = self.attrs.bitcast[_RefCountedAttrsDict]()
        var prev = __get_address_as_lvalue(ptr.address).refcount.fetch_sub(1)
        if prev != 1:
            return

        # Run the destructor on the dictionary elements and then destroy the
        # container.
        var list = __get_address_as_owned_value(ptr.address).impl
        for i in range(len(list)):
            list[i].value.destroy()
        ptr.free()


@register_passable("trivial")
struct _Function:
    # The MLIR function type has two arguments:
    # 1. The self value, or the single argument.
    # 2. None, or an additional argument.
    var value: Pointer[Int16]
    """The function pointer."""

    @always_inline
    fn __init__[FnT: AnyRegType](value: FnT) -> Self:
        # FIXME: No "pointer bitcast" for signature function pointers.
        var f = Pointer[Int16]()
        Pointer(__get_lvalue_as_address(f)).bitcast[FnT]().store(value)
        return Self {value: f}

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
        return (
            Pointer(__get_lvalue_as_address(self.value))
            .bitcast[Self.fn0]()
            .load()()
        )

    @always_inline
    fn invoke(owned self, arg0: object) raises -> object:
        return (
            Pointer(__get_lvalue_as_address(self.value))
            .bitcast[Self.fn1]()
            .load()(arg0)
        )

    @always_inline
    fn invoke(owned self, arg0: object, arg1: object) raises -> object:
        return (
            Pointer(__get_lvalue_as_address(self.value))
            .bitcast[Self.fn2]()
            .load()(arg0, arg1)
        )

    @always_inline
    fn invoke(
        owned self, arg0: object, arg1: object, arg2: object
    ) raises -> object:
        return (
            Pointer(__get_lvalue_as_address(self.value))
            .bitcast[Self.fn3]()
            .load()(arg0, arg1, arg2)
        )


@register_passable
struct _ObjectImpl(CollectionElement):
    """This class is the underlying implementation of the value of an `object`.
    It is a variant of primitive types and pointers to implementations of more
    complex types.

    We choose Int64 and Float64 to store all integer and float values respectively.
    TODO: These should be BigInt and BigFloat one day.
    """

    alias type = __mlir_type[
        `!kgen.variant<`,
        _NoneMarker,
        `, `,
        Bool,
        `, `,
        Int64,
        `, `,
        Float64,
        `, `,
        _ImmutableString,
        `, `,
        _RefCountedListRef,
        `, `,
        _Function,
        `, `,
        _RefCountedAttrsDictRef,
        `>`,
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
    alias dict: Int = 8  # TODO
    """Type discriminator indicating a dictionary."""
    alias function: Int = 6
    """Type discriminator indicating a function."""
    alias obj: Int = 7
    """Type discriminator indicating an object."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(value: Self.type) -> Self:
        return Self {value: value}

    @always_inline
    fn __init__() -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.none.value
        ](_NoneMarker {})

    @always_inline
    fn __init__(value: Bool) -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.bool.value
        ](value)

    @always_inline
    fn __init__[dt: DType](value: SIMD[dt, 1]) -> Self:
        @parameter
        if dt.is_integral():
            return __mlir_op.`kgen.variant.create`[
                _type = Self.type, index = Self.int.value
            ](value.cast[DType.int64]())
        else:
            return __mlir_op.`kgen.variant.create`[
                _type = Self.type, index = Self.float.value
            ](value.cast[DType.float64]())

    @always_inline
    fn __init__(value: _ImmutableString) -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.str.value
        ](value)

    @always_inline
    fn __init__(value: _RefCountedListRef) -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.list.value
        ](value)

    @always_inline
    fn __init__(value: _Function) -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.function.value
        ](value)

    @always_inline
    fn __init__(value: _RefCountedAttrsDictRef) -> Self:
        return __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.obj.value
        ](value)

    @always_inline
    fn __copyinit__(self) -> Self:
        return self.value

    @always_inline
    fn copy(self) -> Self:
        if self.is_str():
            var str = self.get_as_string()
            var impl = _ImmutableString(
                Pointer[Int8].alloc(str.length), str.length
            )
            memcpy(impl.data, DTypePointer[DType.int8](str.data), str.length)
            return impl
        if self.is_list():
            return self.get_as_list().copy()
        if self.is_obj():
            return self.get_obj_attrs().copy()
        return self

    @always_inline
    fn destroy(self):
        if self.is_str():
            self.get_as_string().data.free()
        elif self.is_list():
            self.get_as_list().release()
        elif self.is_obj():
            self.get_obj_attrs().release()

    # ===------------------------------------------------------------------=== #
    # Value Query
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn is_none(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.none.value](self.value)

    @always_inline
    fn is_bool(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.bool.value](self.value)

    @always_inline
    fn is_int(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.int.value](self.value)

    @always_inline
    fn is_float(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.float.value](self.value)

    @always_inline
    fn is_str(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.str.value](self.value)

    @always_inline
    fn is_list(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.list.value](self.value)

    @always_inline
    fn is_dict(self) -> Bool:
        return False

    @always_inline
    fn is_func(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.function.value](
            self.value
        )

    @always_inline
    fn is_obj(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Self.obj.value](self.value)

    @always_inline
    fn get_as_bool(self) -> Bool:
        return __mlir_op.`kgen.variant.take`[index = Self.bool.value](
            self.value
        )

    @always_inline
    fn get_as_int(self) -> Int64:
        return __mlir_op.`kgen.variant.take`[index = Self.int.value](self.value)

    @always_inline
    fn get_as_float(self) -> Float64:
        return __mlir_op.`kgen.variant.take`[index = Self.float.value](
            self.value
        )

    @always_inline
    fn get_as_string(self) -> _ImmutableString:
        return __mlir_op.`kgen.variant.take`[index = Self.str.value](self.value)

    @always_inline
    fn get_as_list(self) -> _RefCountedListRef:
        return __mlir_op.`kgen.variant.take`[index = Self.list.value](
            self.value
        )

    @always_inline
    fn get_as_func(self) -> _Function:
        return __mlir_op.`kgen.variant.take`[index = Self.function.value](
            self.value
        )

    @always_inline
    fn get_obj_attrs(self) -> _RefCountedAttrsDictRef:
        return __mlir_op.`kgen.variant.take`[index = Self.obj.value](self.value)

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
        Bools are always converted to integers, to match Python's behaviour.
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

    # ===------------------------------------------------------------------=== #
    # List Functions
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn get_list_ptr(self) -> Pointer[_RefCountedList]:
        return self.get_as_list().lst.bitcast[_RefCountedList]()

    @always_inline
    fn list_append(self, value: Self):
        __get_address_as_lvalue(self.get_list_ptr().address).impl.push_back(
            value.value
        )

    @always_inline
    fn get_list_length(self) -> Int:
        return len(__get_address_as_lvalue(self.get_list_ptr().address).impl)

    @always_inline
    fn get_list_element(self, i: Int) -> _ObjectImpl:
        return (
            __get_address_as_lvalue(self.get_list_ptr().address).impl[i].copy()
        )

    @always_inline
    fn set_list_element(self, i: Int, value: _ObjectImpl):
        __get_address_as_lvalue(self.get_list_ptr().address).impl[i].destroy()
        __get_address_as_lvalue(self.get_list_ptr().address).impl[i] = value

    # ===------------------------------------------------------------------=== #
    # Object Attribute Functions
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn get_obj_attrs_ptr(self) -> Pointer[_RefCountedAttrsDict]:
        return self.get_obj_attrs().attrs.bitcast[_RefCountedAttrsDict]()

    @always_inline
    fn set_obj_attr(self, key: StringLiteral, value: _ObjectImpl) raises:
        __get_address_as_lvalue(self.get_obj_attrs_ptr().address).set(
            key, value
        )

    @always_inline
    fn get_obj_attr(self, key: StringLiteral) raises -> _ObjectImpl:
        return (
            __get_address_as_lvalue(self.get_obj_attrs_ptr().address)
            .get(key)
            .copy()
        )


# ===----------------------------------------------------------------------=== #
# object
# ===----------------------------------------------------------------------=== #


struct object(IntableRaising, Boolable):
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
    fn __init__(inout self, value: FloatLiteral):
        """Initializes the object with an floating-point value.

        Args:
            value: The float value.
        """
        self._value = Float64(value)

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
        var impl = _ImmutableString(
            Pointer[Int8].alloc(value.length), value.length
        )
        memcpy(impl.data, value.data, value.length)
        self._value = impl

    @always_inline
    fn __init__[*Ts: AnyRegType](inout self, value: ListLiteral[Ts]):
        """Initializes the object from a list literal.

        Parameters:
            Ts: The list element types.

        Args:
            value: The list value.
        """
        self._value = _RefCountedListRef()
        alias types = VariadicList(Ts)

        @parameter
        @always_inline
        fn append[i: Int]():
            # We need to rebind the element to one we know how to convert from.
            # FIXME: This doesn't handle implicit conversions or nested lists.
            alias T = types[i]

            @parameter
            if _mlirtype_is_eq[T, Int]():
                self._append(value.get[i, Int]())
            elif _mlirtype_is_eq[T, FloatLiteral]():
                self._append(value.get[i, FloatLiteral]())
            elif _mlirtype_is_eq[T, Bool]():
                self._append(value.get[i, Bool]())
            elif _mlirtype_is_eq[T, StringRef]():
                self._append(value.get[i, StringRef]())
            elif _mlirtype_is_eq[T, StringLiteral]():
                self._append(value.get[i, StringLiteral]())
            else:
                constrained[
                    False, "cannot convert nested list element to object"
                ]()

        unroll[append, len(types)]()

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
        self._value = _RefCountedAttrsDictRef(attrs)

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
        self._value = existing._value.copy()

    @always_inline
    fn __del__(owned self):
        """Devare the object and release any owned memory."""
        self._value.destroy()

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
            return self._value.get_as_string().length != 0
        debug_assert(self._value.is_list(), "expected a list")
        return self._value.get_list_length() != 0

    fn __int__(self) raises -> Int:
        """Performs conversion to integer conversion according to Python
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

        raise "object type cannot be converted to an integer"

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
    fn _string_compare(self, rhs: object) -> Int:
        return self._value.get_as_string().string_compare(
            rhs._value.get_as_string()
        )

    @always_inline
    fn _list_compare(self, rhs: object) raises -> Int:
        var llen = self._value.get_list_length()
        var rlen = self._value.get_list_length()
        var cmp_len = _min(llen, rlen)
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
            return self._string_compare(rhs) < 0
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
            return self._string_compare(rhs) <= 0
        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) <= 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs == rhs or not lhs

        return Self._comparison_op[Float64.__le__, Int64.__le__, bool_fn](
            self, rhs
        )

    fn __eq__(self, rhs: object) raises -> object:
        """Equality comparator. This compares the elements of strings
        and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the objects are equal.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._string_compare(rhs) == 0
        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) == 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs == rhs

        return Self._comparison_op[Float64.__eq__, Int64.__eq__, bool_fn](
            self, rhs
        )

    fn __ne__(self, rhs: object) raises -> object:
        """Inequality comparator. This compares the elements of strings
        and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the objects are not equal.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._string_compare(rhs) != 0
        if self._value.is_list() and rhs._value.is_list():
            return self._list_compare(rhs) != 0

        @always_inline
        fn bool_fn(lhs: Bool, rhs: Bool) -> Bool:
            return lhs != rhs

        return Self._comparison_op[Float64.__ne__, Int64.__ne__, bool_fn](
            self, rhs
        )

    fn __gt__(self, rhs: object) raises -> object:
        """Greater-than comparator. This lexicographically compares the
        elements of strings and lists.

        Args:
            rhs: Right hand value.

        Returns:
            True if the left hand value is greater.
        """
        if self._value.is_str() and rhs._value.is_str():
            return self._string_compare(rhs) > 0
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
            return self._string_compare(rhs) >= 0
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
            var lhsStr = self._value.get_as_string()
            var rhsStr = rhs._value.get_as_string()
            var length = lhsStr.length + rhsStr.length
            var impl = _ImmutableString(Pointer[Int8].alloc(length), length)
            memcpy(impl.data, lhsStr.data, lhsStr.length)
            memcpy(impl.data.offset(lhsStr.length), rhsStr.data, rhsStr.length)
            var result = object()
            result._value = impl
            return result
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
    fn __pow__(self, rhs: object) raises -> object:
        """Exponentiation operator. Valid only for arithmetic types.

        Args:
            rhs: Right hand value.

        Returns:
            The left hand value raised to the power of the right hand value.
        """
        return Self._arithmetic_binary_op[Float64.__pow__, Int64.__pow__](
            self, rhs
        )

    # TODO: __mod__
    # TODO: __truediv__
    # TODO: __floordiv__

    # TODO: __lshift__
    # TODO: __rshift__

    @always_inline
    fn __and__(self, rhs: object) raises -> object:
        """Bool AND operator. If the left hand value is False, return the
        left-hand value.

        Args:
            rhs: Right hand value.

        Returns:
            The current value if it is False.
        """
        if not self:
            return self
        return rhs

    @always_inline
    fn __or__(self, rhs: object) raises -> object:
        """Bool OR operator. If the left hand value is True, return the
        left-hand value.

        Args:
            rhs: Right hand value.

        Returns:
            The current value if it is True.
        """
        if self:
            return self
        return rhs

    # TODO: __xor__

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

    # TODO: __imod__
    # TODO: __itruediv__
    # TODO: __ifloordiv__

    # TODO: __ilshift__
    # TODO: __irshift__

    @always_inline
    fn __iand__(inout self, rhs: object) raises:
        """In-place AND operator.

        Args:
            rhs: Right hand value.
        """
        if self:
            self = rhs

    @always_inline
    fn __ior__(inout self, rhs: object) raises:
        """In-place OR operator.

        Args:
            rhs: Right hand value.
        """
        if not self:
            self = rhs

    # TODO: __ixor__

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

    # TODO: __rfloordiv__
    # TODO: __rmod__
    # TODO: __rlshift__
    # TODO: __rrshift__

    @always_inline
    fn __rand__(self, lhs: object) raises -> object:
        """Reverse AND operator.

        Args:
            lhs: Left hand value.

        Returns:
            The bitwise AND of the left-hand-side value and this.
        """
        if not lhs:
            return lhs
        return self

    @always_inline
    fn __ror__(self, lhs: object) raises -> object:
        """Reverse OR operator.

        Args:
            lhs: Left hand value.

        Returns:
            The bitwise OR of the left-hand-side value and this.
        """
        if lhs:
            return lhs
        return self

    # TODO: __rxor__

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
            _ = object(self._value.get_obj_attr("append"))(self, value)
            return
        if not self._value.is_list():
            raise Error("TypeError: can only append to lists")
        self._append(value)

    @always_inline
    fn _append(self, value: object):
        self._value.list_append(value._value.copy())

    @always_inline
    fn __len__(self) raises -> Int:
        """Returns the "length" of the object. Only strings, lists, and
        dictionaries have lengths.

        Returns:
            The length of the string value or the number of elements in the list
            or dictionary value.
        """
        if self._value.is_str():
            return self._value.get_as_string().length
        if self._value.is_list():
            return self._value.get_list_length()
        raise Error("TypeError: only strings and lists have length")

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
            return object(self._value.get_obj_attr("__getitem__"))(self, i)
        if not self._value.is_str() and not self._value.is_list():
            raise Error("TypeError: can only index into lists and strings")
        var index = Self._convert_index_to_int(i)
        if self._value.is_str():
            var impl = _ImmutableString(Pointer[Int8].alloc(1), 1)
            impl.data.store(
                self._value.get_as_string().data.offset(index).load()
            )
            return _ObjectImpl(impl)
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
            _ = object(self._value.get_obj_attr("__setitem__"))(self, i, value)
            return
        if self._value.is_str():
            raise Error(
                "TypeError: 'str' object does not support item assignment"
            )
        if not self._value.is_list():
            raise Error("TypeError: can only assign items in lists")
        var index = Self._convert_index_to_int(i)
        self._value.set_list_element(index.value, value._value.copy())

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
        if not self._value.is_obj():
            raise Error(
                "TypeError: Type '"
                + self._value._get_type_name()
                + "' does not have attribute '"
                + key
                + "'"
            )
        self._value.set_obj_attr(key, value._value.copy())

    @always_inline
    fn __call__(self) raises -> object:
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke()

    @always_inline
    fn __call__(self, arg0: object) raises -> object:
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0)

    @always_inline
    fn __call__(self, arg0: object, arg1: object) raises -> object:
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0, arg1)

    @always_inline
    fn __call__(
        self, arg0: object, arg1: object, arg2: object
    ) raises -> object:
        if not self._value.is_func():
            raise Error("TypeError: Object is not a function")
        return self._value.get_as_func().invoke(arg0, arg1, arg2)

    # ===------------------------------------------------------------------=== #
    # Debugging
    # ===------------------------------------------------------------------=== #

    fn print(self):
        """Prints the value of the object."""
        if self._value.is_none():
            _put("None")
        elif self._value.is_bool():
            if self._value.get_as_bool():
                _put("True")
            else:
                _put("False")
        elif self._value.is_int():
            _put(self._value.get_as_int())
        elif self._value.is_float():
            _put(self._value.get_as_float())
        elif self._value.is_str():
            var str = self._value.get_as_string()
            _printf("'%.*s'", str.length, str.data)
        elif self._value.is_list():
            _put("[")
            for j in range(self._value.get_list_length()):
                if j != 0:
                    _put(", ")
                object(self._value.get_list_element(j)).print()
            _put("]")
        elif self._value.is_func():
            _printf("function at %p", self._value.get_as_func().value.address)
        else:
            _put("{")
            var ptr = self._value.get_obj_attrs_ptr().address
            for k in range(len(__get_address_as_lvalue(ptr).impl)):
                var value = __get_address_as_lvalue(ptr).impl[k]
                if k != 0:
                    _put(", ")
                _printf("'%s' = ", value.key.data())
                object(value.value.copy()).print()
            _put("}")
