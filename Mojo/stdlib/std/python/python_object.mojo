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
"""Implements PythonObject.

You can import these APIs from the `python` package. For example:

```mojo
from std.python import PythonObject
```
"""

from . import ConvertibleToPython
from std.os import abort
from std.sys import bit_width_of
from std.ffi import c_double, c_int, c_long, c_size_t, c_ssize_t
import std.format._utils as fmt

from std.reflection import reflect

from ._cpython import (
    CPython,
    GILAcquired,
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject,
    PyObjectPtr,
    PyTypeObject,
    PyTypeObjectPtr,
)
from .bindings import PyMojoObject, _get_type_name, lookup_py_type_object
from .python import Python


struct _PyIter(ImplicitlyCopyable, Iterable, Iterator):
    """A Python iterator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = PythonObject

    var iterator: PythonObject
    """The iterator object that stores location."""
    var next_item: PyObjectPtr
    """The next item to vend or zero if there are no items."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, iter: PythonObject):
        """Initialize an iterator.

        Args:
            iter: A Python iterator instance.
        """
        ref cpy = Python().cpython()
        self.iterator = iter
        self.next_item = cpy.PyIter_Next(iter._obj_ptr)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __next__(mut self) raises StopIteration -> PythonObject:
        """Return the next item and update to point to subsequent item.

        Returns:
            The next item in the traversable object that this iterator
            points to.
        """
        if not self.next_item:
            raise StopIteration()
        ref cpy = Python().cpython()
        var curr_item = self.next_item
        self.next_item = cpy.PyIter_Next(self.iterator._obj_ptr)
        return PythonObject(from_owned=curr_item)

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self


struct PythonObject(
    Boolable,
    Defaultable,
    Identifiable,
    ImplicitlyCopyable,
    Movable,
    RegisterPassable,
    SizedRaising,
    Writable,
):
    """A Python object."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _obj_ptr: PyObjectPtr
    """A pointer to the underlying Python object."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Initialize the object with a `None` value."""
        self = Self(None)

    def __init__(out self, *, from_owned: PyObjectPtr):
        """Initialize this object from an owned reference-counted Python object
        pointer.

        For example, this function should be used to construct a `PythonObject`
        from the pointer returned by "New reference"-type objects from the
        CPython API.

        Args:
            from_owned: An owned pointer to a Python object.

        References:
        - https://docs.python.org/3/glossary.html#term-strong-reference
        """
        self._obj_ptr = from_owned

    def __init__(out self, *, from_borrowed: PyObjectPtr):
        """Initialize this object from a borrowed reference-counted Python
        object pointer.

        For example, this function should be used to construct a `PythonObject`
        from the pointer returned by "Borrowed reference"-type objects from the
        CPython API.

        Args:
            from_borrowed: A borrowed pointer to a Python object.

        References:
        - https://docs.python.org/3/glossary.html#term-borrowed-reference
        """
        ref cpy = Python().cpython()
        # SAFETY:
        #   We were passed a Python "borrowed reference", so for it to be
        #   safe to store this reference, we must increment the reference
        #   count to convert this to a "strong reference".
        cpy.Py_IncRef(from_borrowed)
        self._obj_ptr = from_borrowed

    @implicit
    def __init__(out self, var source: Some[ConvertibleToPython]) raises:
        """Initialize this object from a value of some type that can be
        custom converted to a PythonObject.

        Args:
            source: The value to convert to a PythonObject.

        Raises:
            If the conversion to a PythonObject fails.
        """
        self = source^.to_python_object()

    @always_inline
    def __init__[T: Movable & Deinitable](out self, *, var alloc: T) raises:
        """Allocate a new `PythonObject` and store a Mojo value in it.

        The newly allocated Python object will contain the provided Mojo `T`
        instance directly, without attempting conversion to an equivalent Python
        builtin type.

        Only Mojo types that have a registered Python 'type' object can be stored
        as a Python object. Mojo types are registered using a
        `PythonTypeBuilder`.

        Parameters:
            T: The Mojo type of the value that the resulting Python object
              holds.

        Args:
            alloc: The Mojo value to store in the new Python object.

        Raises:
            If no Python type object has been registered for `T` by a
            `PythonTypeBuilder`.
        """
        # NOTE:
        #   We can't use PythonTypeBuilder.bind[T]() because that constructs a
        #   _new_ PyTypeObject. We want to reference the existing _singleton_
        #   PyTypeObject that represents a given Mojo type.
        var type_obj = lookup_py_type_object[T]()
        var type_obj_ptr = type_obj._obj_ptr.bitcast[PyTypeObject]()
        return _unsafe_alloc_init(type_obj_ptr, alloc^)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_hidden
    @implicit
    def __init__(out self, none: NoneType._mlir_type):
        """Initialize a none value object from a `None` literal.

        Args:
            none: None.
        """
        self = Self(none=NoneType())

    def __init__(out self, none: NoneType):
        """Initialize a none value object from a `None` literal.

        Args:
            none: None.
        """
        ref cpy = Python().cpython()
        self = Self(from_borrowed=cpy.Py_None())

    @implicit
    def __init__(out self, value: Bool):
        """Initialize the object from a bool.

        Args:
            value: The boolean value.
        """
        ref cpy = Python().cpython()
        self = Self(from_owned=cpy.PyBool_FromLong(c_long(Int(value))))

    @implicit
    def __init__[dtype: DType](out self, value: Scalar[dtype]):
        """Initialize the object with a generic scalar value. If the scalar
        value type is bool, it is converted to a boolean. Otherwise, it is
        converted to the appropriate integer or floating point type.

        Parameters:
            dtype: The scalar value type.

        Args:
            value: The scalar value.
        """
        ref cpy = Python().cpython()

        comptime if dtype == .bool:
            var val = c_long(Int(value))
            self = Self(from_owned=cpy.PyBool_FromLong(val))
        elif dtype.is_unsigned():
            var val = c_size_t(value.cast[.uint]())
            self = Self(from_owned=cpy.PyLong_FromSize_t(val))
        elif dtype.is_integral():
            var val = c_ssize_t(value.cast[.int]())
            self = Self(from_owned=cpy.PyLong_FromSsize_t(val))
        else:
            var val = c_double(value.cast[.float64]())
            self = Self(from_owned=cpy.PyFloat_FromDouble(val))

    @implicit
    def __init__(out self, string: StringSlice) raises:
        """Initialize the object from a string.

        Args:
            string: The string value.

        Raises:
            If the string is not valid UTF-8.
        """
        ref cpy = Python().cpython()
        # TODO: This should not be necessary, as `StringSlice` is guaranteed to
        # be valid UTF-8.
        var unicode = cpy.PyUnicode_DecodeUTF8(string)
        if not unicode:
            raise cpy.unsafe_get_error()
        self = Self(from_owned=unicode)

    @implicit
    def __init__(out self, value: StringLiteral) raises:
        """Initialize the object from a string literal.

        Args:
            value: The string literal value.

        Raises:
            If the string is not valid UTF-8.
        """
        self = Self(StringSlice(value))

    @implicit
    def __init__(out self, value: String) raises:
        """Initialize the object from a string.

        Args:
            value: The string value.

        Raises:
            If the string is not valid UTF-8.
        """
        self = Self(StringSlice(value))

    @implicit
    def __init__(out self, slice: Slice):
        """Initialize the object from a Mojo Slice.

        Args:
            slice: The dictionary value.
        """
        self = Self(from_owned=_slice_to_py_object_ptr(slice))

    @always_inline
    def __init__(
        out self, var *values: PythonObject, __list_literal__: NoneType
    ) raises:
        """Construct an Python list of objects.

        Args:
            values: The values to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Returns:
            The constructed Python list.

        Raises:
            If the list construction fails.
        """
        self = Python.list(*values^)

    @always_inline
    def __init__(
        out self, var *values: PythonObject, __set_literal__: NoneType
    ) raises:
        """Construct an Python set of objects.

        Args:
            values: The values to initialize the set with.
            __set_literal__: Tell Mojo to use this method for set literals.

        Returns:
            The constructed Python set.

        Raises:
            If adding an element to the set fails.
        """
        ref cpy = Python().cpython()
        var set_ptr = cpy.PySet_New({})

        for i in range(len(values)):
            # PySet_Add doesn't steal the value reference.
            var value_ptr = values[i].steal_data()
            var errno = cpy.PySet_Add(set_ptr, value_ptr)
            cpy.Py_DecRef(value_ptr)
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=set_ptr)

    def __init__(
        out self,
        var keys: List[PythonObject],
        var values: List[PythonObject],
        __dict_literal__: NoneType,
    ) raises:
        """Construct a Python dictionary from a list of keys and a list of values.

        Args:
            keys: The keys of the dictionary.
            values: The values of the dictionary.
            __dict_literal__: Tell Mojo to use this method for dict literals.

        Raises:
            If setting a dictionary item fails.
        """
        ref cpy = Python().cpython()
        var dict_ptr = cpy.PyDict_New()
        for key, val in zip(keys, values):
            var errno = cpy.PyDict_SetItem(dict_ptr, key._obj_ptr, val._obj_ptr)
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=dict_ptr)

    def __init__(out self, *, copy: Self):
        """Copy the object.

        This increments the underlying refcount of the existing object.

        Args:
            copy: The value to copy.
        """
        self = Self(from_borrowed=copy._obj_ptr)

    def __deinit__(deinit self):
        """Destroy the object.

        Decrements the underlying refcount of the pointed-to object.
        Safe to call from any thread; the GIL is acquired if not
        already held.
        """
        ref cpy = Python().cpython()
        # Skip the PyGILState_Ensure / PyGILState_Release round-trip
        # when the current thread already holds the GIL.
        # PyGILState_Check is a TLS read, much cheaper than the full
        # ensure/release pair, so on the common Python -> Mojo FFI hot
        # path (CPython hands the callee an already-held GIL) the
        # destructor pays just the check.
        if cpy.PyGILState_Check():
            cpy.Py_DecRef(self._obj_ptr)
        else:
            with GILAcquired(Python(cpy)):
                cpy.Py_DecRef(self._obj_ptr)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __iter__(self) raises -> _PyIter:
        """Iterate over the object.

        Returns:
            An iterator object.

        Raises:
            If the object is not iterable.
        """
        ref cpy = Python().cpython()
        var iter_ptr = cpy.PyObject_GetIter(self._obj_ptr)
        if not iter_ptr:
            raise cpy.unsafe_get_error()
        return _PyIter(PythonObject(from_owned=iter_ptr))

    def __getattr__(self, var name: String) raises -> PythonObject:
        """Return the value of the object attribute with the given name.

        Args:
            name: The name of the object attribute to return.

        Returns:
            The value of the object attribute with the given name.

        Raises:
            If the attribute does not exist.
        """
        ref cpy = Python().cpython()
        var attr_ptr = cpy.PyObject_GetAttrString(self._obj_ptr, name^)
        if not attr_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=attr_ptr)

    def __setattr__(self, var name: String, var value: PythonObject) raises:
        """Set the given value for the object attribute with the given name.

        Args:
            name: The name of the object attribute to set.
            value: The new value to be set for that attribute.

        Raises:
            If setting the attribute fails.
        """
        ref cpy = Python().cpython()
        # PyObject_SetAttrString doesn't steal the value reference.
        var value_ptr = value^.steal_data()
        var errno = cpy.PyObject_SetAttrString(self._obj_ptr, name^, value_ptr)
        cpy.Py_DecRef(value_ptr)
        if errno == -1:
            raise cpy.unsafe_get_error()

    def __bool__(self) -> Bool:
        """Evaluate the boolean value of the object.

        Returns:
            Whether the object evaluates as true.
        """
        try:
            return Python().is_true(self)
        except Error:
            # TODO: make this function raise when we can raise parametrically.
            abort("object cannot be converted to bool")

    def __is__(self, other: PythonObject) -> Bool:
        """Test if the PythonObject is the `other` PythonObject, the same as `x is y` in
        Python.

        Args:
            other: The right-hand-side value in the comparison.

        Returns:
            True if they are the same object and False otherwise.
        """
        ref cpy = Python().cpython()
        return cpy.Py_Is(self._obj_ptr, other._obj_ptr) != 0

    def __getitem__(self, *args: PythonObject) raises -> PythonObject:
        """Return the value for the given key or keys.

        Args:
            args: The key or keys to access on this object.

        Returns:
            The value corresponding to the given key for this object.

        Raises:
            If the index is out of bounds or the key does not exist.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = cpy.Py_NewRef(args[0]._obj_ptr)
        else:
            key_ptr = cpy.PyTuple_New(size)
            for i in range(size):
                _ = cpy.PyTuple_SetItem(
                    key_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
                )
        var res_ptr = cpy.PyObject_GetItem(self._obj_ptr, key_ptr)
        cpy.Py_DecRef(key_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    def __getitem__(self, *args: Slice) raises -> PythonObject:
        """Return the sliced value for the given Slice or Slices.

        Args:
            args: The Slice or Slices to apply to this object.

        Returns:
            The sliced value corresponding to the given Slice(s) for this object.

        Raises:
            If the index is out of bounds or the key does not exist.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = _slice_to_py_object_ptr(args[0])
        else:
            key_ptr = cpy.PyTuple_New(size)
            for i in range(size):
                var slice_ptr = _slice_to_py_object_ptr(args[i])
                _ = cpy.PyTuple_SetItem(key_ptr, i, slice_ptr)
        var res_ptr = cpy.PyObject_GetItem(self._obj_ptr, key_ptr)
        cpy.Py_DecRef(key_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    def __setitem__(self, *args: PythonObject, var value: PythonObject) raises:
        """Set the value with the given key or keys.

        Args:
            args: The key or keys to set on this object.
            value: The value to set.

        Raises:
            If setting the item fails.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = cpy.Py_NewRef(args[0]._obj_ptr)
        else:
            key_ptr = cpy.PyTuple_New(size)

            for i in range(size):
                _ = cpy.PyTuple_SetItem(
                    key_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
                )

        # PyObject_SetItem doesn't steal the value reference.
        var value_ptr = value^.steal_data()
        var errno = cpy.PyObject_SetItem(self._obj_ptr, key_ptr, value_ptr)
        cpy.Py_DecRef(key_ptr)
        cpy.Py_DecRef(value_ptr)
        if errno == -1:
            raise cpy.unsafe_get_error()

    def __mul__(self, rhs: PythonObject) raises -> PythonObject:
        """Multiplication.

        Calls the underlying object's `__mul__` method, falling back to
        `rhs.__rmul__` if it is not implemented.

        Args:
            rhs: Right hand value.

        Returns:
            The product.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Multiply](self, rhs)

    def __rmul__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse multiplication.

        Evaluates `lhs * self`, which calls `lhs.__mul__` and falls back to
        this object's `__rmul__` if that is not implemented.

        Args:
            lhs: The left-hand-side value that is multiplied by this object.

        Returns:
            The product of the multiplication.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Multiply](lhs, self)

    # The in-place operators below take `rhs` owned, unlike their non-mutating
    # counterparts, which only borrow it. Their `mut self` would otherwise alias
    # an immutably-borrowed `rhs` and reject `x *= x`; the copy breaks the alias.
    def __imul__(mut self, var rhs: PythonObject) raises:
        """In-place multiplication.

        Calls the underlying object's `__imul__` method, falling back to
        `__mul__` if it is not implemented.

        Args:
            rhs: The right-hand-side value by which this object is multiplied.

        Raises:
            If the operation is not supported.
        """
        self = _binary_op[CPython.PyNumber_InPlaceMultiply](self, rhs)

    def __add__(self, rhs: PythonObject) raises -> PythonObject:
        """Addition and concatenation.

        Calls the underlying object's `__add__` method, falling back to
        `rhs.__radd__` if it is not implemented.

        Args:
            rhs: Right hand value.

        Returns:
            The sum or concatenated values.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Add](self, rhs)

    def __radd__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse addition and concatenation.

        Evaluates `lhs + self`, which calls `lhs.__add__` and falls back to
        this object's `__radd__` if that is not implemented.

        Args:
            lhs: The left-hand-side value to which this object is added or
                 concatenated.

        Returns:
            The sum.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Add](lhs, self)

    def __iadd__(mut self, var rhs: PythonObject) raises:
        """Immediate addition and concatenation.

        Args:
            rhs: The right-hand-side value that is added to this object.

        Raises:
            If the operation is not supported.
        """
        self = _binary_op[CPython.PyNumber_InPlaceAdd](self, rhs)

    def __sub__(self, rhs: PythonObject) raises -> PythonObject:
        """Subtraction.

        Calls the underlying object's `__sub__` method, falling back to
        `rhs.__rsub__` if it is not implemented.

        Args:
            rhs: Right hand value.

        Returns:
            The difference.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Subtract](self, rhs)

    def __rsub__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse subtraction.

        Evaluates `lhs - self`, which calls `lhs.__sub__` and falls back to
        this object's `__rsub__` if that is not implemented.

        Args:
            lhs: The left-hand-side value from which this object is subtracted.

        Returns:
            The result of subtracting this from the given value.

        Raises:
            If the operation is not supported.
        """
        return _binary_op[CPython.PyNumber_Subtract](lhs, self)

    def __isub__(mut self, var rhs: PythonObject) raises:
        """Immediate subtraction.

        Args:
            rhs: The right-hand-side value that is subtracted from this object.

        Raises:
            If the operation is not supported.
        """
        self = _binary_op[CPython.PyNumber_InPlaceSubtract](self, rhs)

    def __floordiv__(self, rhs: PythonObject) raises -> PythonObject:
        """Return the division of self and rhs rounded down to the nearest
        integer.

        Calls the underlying object's `__floordiv__` method, falling back to
        `rhs.__rfloordiv__` if it is not implemented.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing this by the right-hand-side value, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_FloorDivide](self, rhs)

    def __rfloordiv__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse floor division.

        Evaluates `lhs // self`, which calls `lhs.__floordiv__` and falls back to
        this object's `__rfloordiv__` if that is not implemented.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_FloorDivide](lhs, self)

    def __ifloordiv__(mut self, var rhs: PythonObject) raises:
        """Immediate floor division.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceFloorDivide](self, rhs)

    def __truediv__(self, rhs: PythonObject) raises -> PythonObject:
        """Division.

        Calls the underlying object's `__truediv__` method, falling back to
        `rhs.__rtruediv__` if it is not implemented.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing the right-hand-side value by this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_TrueDivide](self, rhs)

    def __rtruediv__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse division.

        Evaluates `lhs / self`, which calls `lhs.__truediv__` and falls back to
        this object's `__rtruediv__` if that is not implemented.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_TrueDivide](lhs, self)

    def __itruediv__(mut self, var rhs: PythonObject) raises:
        """Immediate division.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceTrueDivide](self, rhs)

    def __mod__(self, rhs: PythonObject) raises -> PythonObject:
        """Return the remainder of self divided by rhs.

        Calls the underlying object's `__mod__` method, falling back to
        `rhs.__rmod__` if it is not implemented.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Remainder](self, rhs)

    def __rmod__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse modulo.

        Evaluates `lhs % self`, which calls `lhs.__mod__` and falls back to
        this object's `__rmod__` if that is not implemented.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The remainder from dividing the given value by this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Remainder](lhs, self)

    def __imod__(mut self, var rhs: PythonObject) raises:
        """Immediate modulo.

        Args:
            rhs: The right-hand-side value that is used to divide this object.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceRemainder](self, rhs)

    def __xor__(self, rhs: PythonObject) raises -> PythonObject:
        """Exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is exclusive
                 OR'ed.

        Returns:
            The exclusive OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Xor](self, rhs)

    def __rxor__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse exclusive OR.

        Args:
            lhs: The left-hand-side value that is exclusive OR'ed with this
                 object.

        Returns:
            The exclusive OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Xor](lhs, self)

    def __ixor__(mut self, var rhs: PythonObject) raises:
        """Immediate exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is
                 exclusive OR'ed.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceXor](self, rhs)

    def __or__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Returns:
            The bitwise OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Or](self, rhs)

    def __ror__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise OR.

        Args:
            lhs: The left-hand-side value that is bitwise OR'ed with this
                 object.

        Returns:
            The bitwise OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Or](lhs, self)

    def __ior__(mut self, var rhs: PythonObject) raises:
        """Immediate bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceOr](self, rhs)

    def __and__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Returns:
            The bitwise AND result of this and the given value.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_And](self, rhs)

    def __rand__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise and.

        Args:
            lhs: The left-hand-side value that is bitwise AND'ed with this
                 object.

        Returns:
            The bitwise AND result of the given value and this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_And](lhs, self)

    def __iand__(mut self, var rhs: PythonObject) raises:
        """Immediate bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceAnd](self, rhs)

    def __rshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Returns:
            This value, shifted right by the given value.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Rshift](self, rhs)

    def __rrshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise right shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the right
                 by this object.

        Returns:
            The given value, shifted right by this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Rshift](lhs, self)

    def __irshift__(mut self, var rhs: PythonObject) raises:
        """Immediate bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceRshift](self, rhs)

    def __lshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Returns:
            This value, shifted left by the given value.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Lshift](self, rhs)

    def __rlshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise left shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the left
                 by this object.

        Returns:
            The given value, shifted left by this.

        Raises:
            If the operation fails.
        """
        return _binary_op[CPython.PyNumber_Lshift](lhs, self)

    def __ilshift__(mut self, var rhs: PythonObject) raises:
        """Immediate bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Raises:
            If the operation fails.
        """
        self = _binary_op[CPython.PyNumber_InPlaceLshift](self, rhs)

    def __pow__(self, exp: PythonObject) raises -> PythonObject:
        """Raises this object to the power of the given value.

        Args:
            exp: The exponent.

        Returns:
            The result of raising this by the given exponent.

        Raises:
            If the operation fails.
        """
        return _power_op[CPython.PyNumber_Power](self, exp)

    def __rpow__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse power of.

        Args:
            lhs: The number that is raised to the power of this object.

        Returns:
            The result of raising the given value by this exponent.

        Raises:
            If the operation fails.
        """
        return _power_op[CPython.PyNumber_Power](lhs, self)

    def __ipow__(mut self, var rhs: PythonObject) raises:
        """Immediate power of.

        Args:
            rhs: The exponent.

        Raises:
            If the operation fails.
        """
        self = _power_op[CPython.PyNumber_InPlacePower](self, rhs)

    def __lt__(self, rhs: PythonObject) raises -> PythonObject:
        """Less than (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If neither operand supports `<`, or if the comparison fails.
        """
        return _rich_compare[Py_LT](self, rhs)

    def __le__(self, rhs: PythonObject) raises -> PythonObject:
        """Less than or equal (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If neither operand supports `<=`, or if the comparison fails.
        """
        return _rich_compare[Py_LE](self, rhs)

    def __gt__(self, rhs: PythonObject) raises -> PythonObject:
        """Greater than (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If neither operand supports `>`, or if the comparison fails.
        """
        return _rich_compare[Py_GT](self, rhs)

    def __ge__(self, rhs: PythonObject) raises -> PythonObject:
        """Greater than or equal (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If neither operand supports `>=`, or if the comparison fails.
        """
        return _rich_compare[Py_GE](self, rhs)

    def __eq__(self, rhs: PythonObject) raises -> PythonObject:
        """Equality (rich) comparison operator.

        Operands that do not implement the comparison fall back to an identity
        check and compare unequal, rather than raising.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the comparison fails.
        """
        return _rich_compare[Py_EQ](self, rhs)

    def __ne__(self, rhs: PythonObject) raises -> PythonObject:
        """Inequality (rich) comparison operator.

        Operands that do not implement the comparison fall back to an identity
        check and compare unequal, rather than raising.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the comparison fails.
        """
        return _rich_compare[Py_NE](self, rhs)

    def __pos__(self) raises -> PythonObject:
        """Positive.

        Calls the underlying object's `__pos__` method.

        Returns:
            The result of prefixing this object with a `+` operator. For most
            numerical objects, this does nothing.

        Raises:
            If the operation fails.
        """
        return _unary_op[CPython.PyNumber_Positive](self)

    def __neg__(self) raises -> PythonObject:
        """Negative.

        Calls the underlying object's `__neg__` method.

        Returns:
            The result of prefixing this object with a `-` operator. For most
            numerical objects, this returns the negative.

        Raises:
            If the operation fails.
        """
        return _unary_op[CPython.PyNumber_Negative](self)

    def __invert__(self) raises -> PythonObject:
        """Inversion.

        Calls the underlying object's `__invert__` method.

        Returns:
            The logical inverse of this object: a bitwise representation where
            all bits are flipped, from zero to one, and from one to zero.

        Raises:
            If the operation fails.
        """
        return _unary_op[CPython.PyNumber_Invert](self)

    def __contains__(self, rhs: PythonObject) raises -> Bool:
        """Contains dunder.

        Calls the underlying object's `__contains__` method, falling back to a
        search by iteration if it is not implemented.

        Args:
            rhs: Right hand value.

        Returns:
            True if rhs is in self.

        Raises:
            If the operation fails.
        """
        ref cpy = Python().cpython()
        var result = cpy.PySequence_Contains(self._obj_ptr, rhs._obj_ptr)
        if result == -1:
            raise cpy.unsafe_get_error()
        return result == 1

    # see https://github.com/python/cpython/blob/main/Objects/call.c
    # for decrement rules
    def __call__(
        self, *args: PythonObject, var **kwargs: PythonObject
    ) raises -> PythonObject:
        """Call the underlying object as if it were a function.

        Args:
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Raises:
            If the function cannot be called for any reason.

        Returns:
            The return value from the called object.
        """
        var size = len(args)
        ref cpy = Python().cpython()
        var args_ptr = cpy.PyTuple_New(size)

        for i in range(size):
            _ = cpy.PyTuple_SetItem(
                args_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
            )
        var kwargs_ptr = Python.dict(**kwargs^).steal_data()
        var res_ptr = cpy.PyObject_Call(self._obj_ptr, args_ptr, kwargs_ptr)
        cpy.Py_DecRef(args_ptr)
        cpy.Py_DecRef(kwargs_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) raises -> Int:
        """Returns the length of the object.

        Returns:
            The length of the object.

        Raises:
            If the operation fails.
        """
        ref cpy = Python().cpython()
        var length = Int(cpy.PyObject_Length(self._obj_ptr))
        if length == -1 and cpy.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpy.unsafe_get_error()
        return length

    def __hash__(self) raises -> Int:
        """Returns the hash value of the object.

        Returns:
            The hash value of the object.

        Raises:
            If the operation fails.
        """
        ref cpy = Python().cpython()
        var res = Int(cpy.PyObject_Hash(self._obj_ptr))
        if res == -1 and cpy.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpy.unsafe_get_error()
        return res

    def __int__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `int` (i.e. arbitrary precision
        integer).

        Returns:
            A Python `int` object.

        Raises:
            An error if the conversion failed.
        """
        return Python.int(self)

    def __float__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `float` object.

        Returns:
            A Python `float` object.

        Raises:
            If the conversion fails.
        """
        return Python.float(self)

    @no_inline
    def __str__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `str`.

        Returns:
            A Python `str` object.

        Raises:
            An error if the conversion failed.
        """
        return Python.str(self)

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this Python object to the provided Writer.

        Args:
            writer: The object to write to.
        """

        try:
            # TODO: Avoid this intermediate String allocation, if possible.
            writer.write(String(py=self))
        except e:
            # TODO: make this method raising when we can raise parametrically.
            abort(t"failed to write PythonObject to writer: {e}")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `PythonObject` to a writer.

        Uses Python's `repr()` to get the representation of the underlying
        Python object.

        Args:
            writer: The object to write to.
        """
        try:
            var repr_str = String(py=self.__repr__())
            fmt.FormatStruct(writer, "PythonObject").fields(repr_str)
        except e:
            abort(t"failed to write PythonObject repr to writer: {e}")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def steal_data(var self) -> PyObjectPtr:
        """Take ownership of the underlying pointer from the Python object.

        Returns:
            The underlying data.
        """
        var ptr = self._obj_ptr
        self._obj_ptr = {}
        return ptr

    def unsafe_get_as_pointer[
        dtype: DType
    ](self) raises -> Pointer[Scalar[dtype], MutAnyOrigin]:
        """Reinterpret a Python integer as a Mojo pointer.

        Warning: converting from an integer to a pointer is unsafe! The
        compiler assumes the resulting pointer DOES NOT alias any Mojo-derived
        pointer. This is OK if the pointer originates from and is owned by
        Python, e.g. the data underpinning a torch tensor.

        Parameters:
            dtype: The desired DType of the pointer.

        Returns:
            A pointer for the underlying Python data.

        Raises:
            If the operation fails.
        """
        return Pointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=Int(py=self)
        )

    def downcast_value_ptr[
        T: Deinitable
    ](self, *, func: Optional[StaticString] = None) raises -> Pointer[
        T, MutAnyOrigin
    ]:
        """Get a pointer to the expected contained Mojo value of type `T`.

        This method validates that this object actually contains an instance of
        `T`, and will raise an error if it does not.

        Mojo values are stored as Python objects backed by the `PyMojoObject[T]`
        struct.

        Args:
            func: Optional name of bound Mojo function that the raised
              TypeError should reference if downcasting fails.

        Parameters:
            T: The type of the Mojo value that this Python object is expected
              to contain.

        Returns:
            A pointer to the inner Mojo value.

        Raises:
            If the Python object does not contain an instance of the Mojo `T`
            type.
        """
        var opt = self._try_downcast_value[T]()
        if opt:
            return opt.unsafe_take()

        if func:
            raise Error(
                String.format(
                    (
                        "TypeError: {}() expected Mojo '{}' type argument, got"
                        " '{}'"
                    ),
                    func[],
                    reflect[T].name(),
                    _get_type_name(self),
                )
            )
        else:
            raise Error(
                String.format(
                    "TypeError: expected Mojo '{}' type value, got '{}'",
                    reflect[T].name(),
                    _get_type_name(self),
                )
            )

    def _try_downcast_value[
        T: Deinitable
    ](var self) raises -> Optional[Pointer[T, MutAnyOrigin]]:
        """Try to get a pointer to the expected contained Mojo value of type `T`.

        None will be returned if the type of this object does not match the
        bound Python type of `T`, or if the Mojo value has not been initialized.

        This function will raise if the provided Mojo type `T` has not been
        bound to a Python type using a `PythonTypeBuilder`.

        Parameters:
            T: The type of the Mojo value that this Python object is expected
              to contain.

        Raises:
            If `T` has not been bound to a Python type object.
        """
        ref cpy = Python().cpython()
        var type = PyObjectPtr(upcast_from=cpy.Py_TYPE(self._obj_ptr))
        var expected_type = lookup_py_type_object[T]()._obj_ptr
        if type == expected_type:
            ref mojo_obj = self._obj_ptr.bitcast[PyMojoObject[T]]().value()[]
            if mojo_obj.is_initialized:
                return Pointer(to=mojo_obj.mojo_value).as_unsafe_any_origin()
        return None

    def unchecked_downcast_value_ptr[
        mut: Bool, origin: Origin[mut=mut], //, T: Deinitable
    ](ref[origin] self) -> Pointer[T, origin]:
        """Get a pointer to the expected Mojo value of type `T`.

        This function assumes that this Python object was allocated as an
        instance of `PyMojoObject[T]` and that the Mojo value has been
        initialized.

        Parameters:
            mut: The mutability of self.
            origin: The origin of self.
            T: The type of the Mojo value stored in this object.

        Returns:
            A pointer to the inner Mojo value.

        Safety:

        The user must be certain that this Python object type matches the bound
        Python type object for `T`.
        """
        ref mojo_obj = self._obj_ptr.bitcast[PyMojoObject[T]]().value()[]
        # TODO(MSTDL-950): Should use something like `addr_of!`
        # Safety: The mutability matches that of `self`.
        return Pointer[mut=mut](to=mojo_obj.mojo_value).unsafe_origin_cast[
            origin
        ]()


# ===-----------------------------------------------------------------------===#
# Factory functions for PythonObject
# ===-----------------------------------------------------------------------===#


def _unsafe_alloc[
    T: AnyType
](type_obj_ptr: PyTypeObjectPtr) raises -> PyObjectPtr:
    """Allocate an uninitialized Python object for storing a Mojo value.

    Parameters:
        T: The Mojo type of the value that will be stored in the Python object.

    Args:
        type_obj_ptr: Pointer to the Python type object describing the layout.

    Returns:
        A new Python object pointer with uninitialized storage.

    Raises:
        If the Python object allocation fails.
    """
    ref cpy = Python().cpython()
    var obj_ptr = cpy.PyType_GenericAlloc(type_obj_ptr, 0)
    if not obj_ptr:
        raise Error("Allocation of Python object failed.")
    return obj_ptr


def _unsafe_init[
    T: Movable & Deinitable,
    //,
](obj_ptr: PyObjectPtr, var mojo_value: T) raises:
    """Initialize a Python object pointer with a Mojo value.

    Parameters:
        T: The Mojo type of the value that the resulting Python object holds.

    Args:
        obj_ptr: The Python object pointer to initialize.
            The pointer must have been allocated using the correct type object.
        mojo_value: The Mojo value to store in the Python object.

    # Safety
     `obj_ptr` must be a Python object pointer allocated using the correct
     type object. Use of any other pointer is invalid.
    """
    ref mojo_obj = obj_ptr.bitcast[PyMojoObject[T]]().value()[]
    Pointer(to=mojo_obj.mojo_value).unsafe_write(mojo_value^)
    mojo_obj.is_initialized = True


def _unsafe_alloc_init[
    T: Movable & Deinitable,
    //,
](type_obj_ptr: PyTypeObjectPtr, var mojo_value: T) raises -> PythonObject:
    """Allocate a Python object pointer and initialize it with a Mojo value.

    Parameters:
        T: The Mojo type of the value that the resulting Python object holds.

    Args:
        type_obj_ptr: Must be the Python type object describing `PyTypeObject[T]`.
        mojo_value: The Mojo value to store in the new Python object.

    Returns:
        A new PythonObject containing the Mojo value.

    # Safety
    `type_obj_ptr` must be a Python type object created by `PythonTypeBuilder`,
    whose underlying storage type is the `PyMojoObject` struct. Use of any other
    type object is invalid.
    Raises:
        If the Python object allocation fails.
    """
    var obj_ptr = _unsafe_alloc[T](type_obj_ptr)
    _unsafe_init(obj_ptr, mojo_value^)
    return PythonObject(from_owned=obj_ptr)


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


@always_inline
def _unary_op[
    op: def(CPython, PyObjectPtr) thin -> PyObjectPtr
](obj: PythonObject) raises -> PythonObject:
    """Apply a CPython unary protocol function, such as `PyNumber_Negative`.

    Parameters:
        op: The `CPython` wrapper for the protocol function to apply.

    Args:
        obj: The operand.

    Returns:
        The result of the operation.

    Raises:
        If the operation is not supported, or if it fails.
    """
    ref cpy = Python().cpython()
    var result = op(cpy, obj._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


@always_inline
def _binary_op[
    op: def(CPython, PyObjectPtr, PyObjectPtr) thin -> PyObjectPtr
](lhs: PythonObject, rhs: PythonObject) raises -> PythonObject:
    """Apply a CPython binary protocol function, such as `PyNumber_Add`.

    Parameters:
        op: The `CPython` wrapper for the protocol function to apply.

    Args:
        lhs: The left-hand-side operand.
        rhs: The right-hand-side operand.

    Returns:
        The result of the operation.

    Raises:
        If the operation is not supported, or if it fails.
    """
    ref cpy = Python().cpython()
    var result = op(cpy, lhs._obj_ptr, rhs._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


@always_inline
def _power_op[
    op: def(CPython, PyObjectPtr, PyObjectPtr, PyObjectPtr) thin -> PyObjectPtr
](base: PythonObject, exp: PythonObject) raises -> PythonObject:
    """Apply a CPython power protocol function with no modulus.

    Parameters:
        op: The `CPython` wrapper for the protocol function to apply.

    Args:
        base: The base.
        exp: The exponent.

    Returns:
        `base` raised to the power `exp`.

    Raises:
        If the operation is not supported, or if it fails.
    """
    ref cpy = Python().cpython()
    var result = op(cpy, base._obj_ptr, exp._obj_ptr, cpy.Py_None())
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


@always_inline
def _rich_compare[
    opid: c_int
](lhs: PythonObject, rhs: PythonObject) raises -> PythonObject:
    """Compare two objects with CPython's rich-comparison protocol.

    Parameters:
        opid: The comparison to perform, such as `Py_LT`.

    Args:
        lhs: The left-hand-side operand.
        rhs: The right-hand-side operand.

    Returns:
        The result of the comparison.

    Raises:
        If the comparison is not supported, or if it fails.
    """
    ref cpy = Python().cpython()
    var result = cpy.PyObject_RichCompare(lhs._obj_ptr, rhs._obj_ptr, opid)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _slice_to_py_object_ptr(slice: Slice) -> PyObjectPtr:
    """Convert Mojo Slice to Python slice parameters.

    Deliberately avoids using `span.indices()` here and instead passes
    the Slice parameters directly to Python. Python's C implementation
    already handles such conditions, allowing Python to apply its own slice
    handling and error handling.


    Args:
        slice: A Mojo slice object to be converted.

    Returns:
        PyObjectPtr: The pointer to the Python slice.

    """
    ref cpy = Python().cpython()
    var start = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.start.value())
    ) if slice.start else cpy.Py_None()
    var stop = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.end.value())
    ) if slice.end else cpy.Py_None()
    var step = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.step.value())
    ) if slice.step else cpy.Py_None()
    var res = cpy.PySlice_New(start, stop, step)
    cpy.Py_DecRef(start)
    cpy.Py_DecRef(stop)
    cpy.Py_DecRef(step)
    return res
