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
"""
Mojo bindings functions and types from the CPython C API.

Documentation for these functions can be found online at:
  <https://docs.python.org/3/c-api/stable.html#contents-of-limited-api>
"""

from .python import Python
from .python_object import PythonObject
from std.collections import Array
from std.memory import OpaquePointer
from std.memory.alloc import alloc, Layout
from std.memory.unsafe_pointer import unsafe_cast
from std.os import abort, getenv, setenv
from std.os.path import dirname
from std.pathlib import Path
from std.sys.arg import argv
from std.ffi import (
    external_call,
    _DLHandle,
    OwnedDLHandle,
    CStringSlice,
    c_char,
    c_double,
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
    c_ulong,
)

import std.format._utils as fmt
from std.utils import Variant

comptime Py_ssize_t = c_ssize_t
comptime Py_hash_t = Py_ssize_t

# ===-----------------------------------------------------------------------===#
# Raw Bindings
# ===-----------------------------------------------------------------------===#

# ref: https://github.com/python/cpython/blob/main/Include/compile.h
comptime Py_single_input: c_int = 256
comptime Py_file_input: c_int = 257
comptime Py_eval_input: c_int = 258
comptime Py_func_type_input: c_int = 345

# 0 when Stackless Python is disabled
# ref: https://github.com/python/cpython/blob/main/Include/object.h
comptime Py_TPFLAGS_DEFAULT = 0

# These flags are used to determine if a type is a subclass.
# ref: https://github.com/python/cpython/blob/main/Include/object.h
comptime Py_TPFLAGS_LONG_SUBCLASS = c_ulong(1 << 24)
comptime Py_TPFLAGS_LIST_SUBCLASS = c_ulong(1 << 25)
comptime Py_TPFLAGS_TUPLE_SUBCLASS = c_ulong(1 << 26)
comptime Py_TPFLAGS_BYTES_SUBCLASS = c_ulong(1 << 27)
comptime Py_TPFLAGS_UNICODE_SUBCLASS = c_ulong(1 << 28)
comptime Py_TPFLAGS_DICT_SUBCLASS = c_ulong(1 << 29)
comptime Py_TPFLAGS_BASE_EXC_SUBCLASS = c_ulong(1 << 30)
comptime Py_TPFLAGS_TYPE_SUBCLASS = c_ulong(1 << 31)


# ref: https://docs.python.org/3/c-api/structures.html#c.PyCFunction
comptime PyCFunction = def(PyObjectPtr, PyObjectPtr) thin abi(
    "C"
) -> PyObjectPtr
comptime PyCFunctionWithKeywords = def(
    PyObjectPtr, PyObjectPtr, PyObjectPtr
) thin abi("C") -> PyObjectPtr

# `METH_FASTCALL` ("vectorcall") signature. CPython hands the callee a raw
# C array of borrowed `PyObject*` plus `nargs`, skipping the tuple-packing
# step that `METH_VARARGS` requires. The args pointer is `PyObject *const *`
# in CPython and is guaranteed non-null by the vectorcall protocol (PEP
# 590); we therefore model it as a plain `Pointer` rather than an
# `OptionalPointer`. The pointer is owned by CPython, so the origin
# is `MutUntrackedOrigin`.
# ref: https://docs.python.org/3/c-api/structures.html#c.PyCFunctionFast
comptime PyCFunctionFast = def(
    PyObjectPtr, Pointer[PyObjectPtr, MutUntrackedOrigin], Py_ssize_t
) thin abi("C") -> PyObjectPtr

# Flag passed to newmethodobject
# ref: https://github.com/python/cpython/blob/main/Include/methodobject.h
comptime METH_VARARGS = 0x01
comptime METH_KEYWORDS = 0x02
comptime METH_STATIC = 0x20
comptime METH_FASTCALL = 0x80


# GIL
@fieldwise_init
struct PyGILState_STATE(TrivialRegisterPassable):
    """Represents the state of the Python Global Interpreter Lock (GIL).

    This struct is used to store and manage the state of the GIL, which is
    crucial for thread-safe operations in Python.

    References:
    - https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/pystate.h#L76
    """

    # typedef enum {
    #   PyGILState_LOCKED, PyGILState_UNLOCKED
    # } PyGILState_STATE;

    var current_state: c_int
    """The current state of the GIL."""

    comptime PyGILState_LOCKED = c_int(0)
    comptime PyGILState_UNLOCKED = c_int(1)


struct PyThreadState:
    """This data structure represents the state of a single thread.

    It's an opaque struct.

    References:
    - https://docs.python.org/3/c-api/init.html#c.PyThreadState
    """

    # TODO: add this public data member
    # PyInterpreterState *interp
    pass


@fieldwise_init
struct PyObjectPtr(
    Boolable,
    Defaultable,
    Equatable,
    ImplicitlyCopyable,
    Intable,
    RegisterPassable,
    Writable,
):
    """Equivalent to `PyObject*` in C.

    It is crucial that this type has the same size and alignment as `PyObject*`
    for FFI ABI correctness.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _unsized_obj_ptr: OptionalPointer[PyObject, MutUntrackedOrigin]
    """Raw pointer to the underlying PyObject struct instance.

    It is not valid to read or write a `PyObject` directly from this pointer.

    This is because `PyObject` is an "unsized" or "incomplete" type: typically,
    any allocation containing a `PyObject` contains additional fields holding
    information specific to that Python object instance, e.g. containing its
    "true" value.

    The value behind this pointer is only safe to interact with directly when
    it has been downcasted to a concrete Python object type backing struct, in
    a context where the user has ensured the object value is of that type.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Initialize a null PyObjectPtr."""
        self._unsized_obj_ptr = {}

    @always_inline
    def __init__[
        T: AnyType, //
    ](out self, *, upcast_from: OptionalPointer[T, MutUntrackedOrigin]):
        self._unsized_obj_ptr = unsafe_cast[Type=PyObject](upcast_from)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __eq__(self, rhs: Self) -> Bool:
        """Compare two PyObjectPtr for equality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are equal, False otherwise.
        """
        return self._unsized_obj_ptr == rhs._unsized_obj_ptr

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __bool__(self) -> Bool:
        return Bool(self._unsized_obj_ptr)

    @always_inline
    def __int__(self) -> Int:
        if self._unsized_obj_ptr:
            return Int(self._unsized_obj_ptr.unsafe_value())
        else:
            return 0

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def bitcast[T: AnyType](self) -> OptionalPointer[T, MutUntrackedOrigin]:
        """Bitcasts the `PyObjectPtr` to a pointer of type `T`.

        Parameters:
            T: The target type to cast to.

        Returns:
            A pointer to the underlying object as type `T`.
        """
        return unsafe_cast[Type=T](self._unsized_obj_ptr)

    def write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(self._unsized_obj_ptr)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `PyObjectPtr` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "PyObjectPtr").fields(
            self._unsized_obj_ptr,
        )


@fieldwise_init
struct PythonVersion(ImplicitlyCopyable, RegisterPassable):
    """Represents a Python version with major, minor, and patch numbers."""

    var major: Int
    """The major version number."""
    var minor: Int
    """The minor version number."""
    var patch: Int
    """The patch version number."""

    def __init__(out self, version: StringSlice):
        """Initialize a PythonVersion object from a version string.

        Args:
            version: A string representing the Python version (e.g., "3.9.5").

        The version string is parsed to extract major, minor, and patch numbers.
        If parsing fails for any component, it defaults to -1.
        """
        var components = Array[Int, 3](fill=-1)
        var start = 0
        var next_idx = 0
        var i = 0
        while next_idx < version.byte_length() and i < 3:
            if version[byte=next_idx] == "." or (
                version[byte=next_idx] == " " and i == 2
            ):
                var c = version[byte=start:next_idx]
                try:
                    components[i] = atol(c)
                except:
                    components[i] = -1
                i += 1
                start = next_idx + 1
            next_idx += 1
        self = PythonVersion(components[0], components[1], components[2])


def _py_get_version(lib: _DLHandle) -> StaticString:
    return StaticString(
        unsafe_from_utf8=CStringSlice(
            unsafe_from_ptr=lib.call[
                "Py_GetVersion",
                OptionalPointer[c_char, ImmStaticOrigin],
            ]().value()
        )
    )


@fieldwise_init
struct PyMethodDef(Defaultable, ImplicitlyCopyable):
    """Represents a Python method definition. This struct is used to define
    methods for Python modules or types.

    Notes:
        [Reference](
        https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
        ).
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var method_name: Optional[CStringSlice[ImmStaticOrigin]]
    """A pointer to the name of the method as a C string.

    Notes:
        called `ml_name` in CPython.
    """

    var method_impl: OptionalPointer[NoneType, MutUntrackedOrigin]
    """A function pointer to the implementation of the method."""

    var method_flags: c_int
    """Flags indicating how the method should be called.

    References:
    - https://docs.python.org/3/c-api/structures.html#c.PyMethodDef"""

    var method_docstring: Optional[CStringSlice[ImmStaticOrigin]]
    """The docstring for the method."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Constructs a zero initialized PyModuleDef.

        This is suitable for use terminating an array of PyMethodDef values.
        """
        self.method_name = {}
        self.method_impl = {}
        self.method_flags = 0
        self.method_docstring = {}

    @staticmethod
    def function[
        static_method: Bool = False
    ](
        func: Variant[PyCFunction, PyCFunctionWithKeywords],
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> Self:
        """Create a PyMethodDef for a function.

        Parameters:
            static_method: Whether the function is a static method. Default is
                False.

        Arguments:
            func: The function to wrap.
            func_name: The name of the function.
            docstring: The docstring for the function.
        """
        # TODO(MSTDL-896):
        #   Support a way to get the name of the function from its parameter
        #   type, similar to `get_linkage_name()`?

        var with_kwargs = func.isa[PyCFunctionWithKeywords]()
        var func_ptr = _fn_ptr_as_opaque(
            func[PyCFunctionWithKeywords]
        ) if with_kwargs else _fn_ptr_as_opaque(func[PyCFunction])

        var flags = c_int(
            METH_VARARGS
            | (METH_STATIC if static_method else 0)
            | (METH_KEYWORDS if with_kwargs else 0)
        )
        return PyMethodDef(
            func_name.as_c_string_slice(),
            func_ptr,
            flags,
            docstring.as_c_string_slice(),
        )

    @staticmethod
    def function[
        static_method: Bool = False
    ](
        func: PyCFunctionFast,
        func_name: StaticString,
        docstring: StaticString = StaticString(),
    ) -> Self:
        """Create a PyMethodDef for a `PyCFunctionFast` registered with
        `METH_FASTCALL`.

        Parameters:
            static_method: Whether the function is a static method. Default is
                False.

        Arguments:
            func: The fastcall function to wrap.
            func_name: The name of the function.
            docstring: The docstring for the function.
        """
        var flags = c_int(METH_FASTCALL | (METH_STATIC if static_method else 0))
        return PyMethodDef(
            func_name.as_c_string_slice(),
            _fn_ptr_as_opaque(func),
            flags,
            docstring.as_c_string_slice(),
        )


def _null_fn_ptr[T: TrivialRegisterPassable]() -> T:
    return __mlir_op.`pop.pointer.bitcast`[_type=T](
        __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`
    )


def _fn_ptr_as_opaque[
    T: TrivialRegisterPassable
](func: T) -> OpaquePointer[MutUntrackedOrigin]:
    """Reinterprets a C ABI function as the `void *` CPython stores it in."""
    return {
        _mlir_value = __mlir_op.`pop.pointer.bitcast`[
            _type=OpaquePointer[MutUntrackedOrigin]._mlir_type
        ](func)
    }


comptime PyTypeObjectPtr = OptionalPointer[PyTypeObject, MutUntrackedOrigin]


struct PyTypeObject:
    """The opaque C structure of the objects used to describe types.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyTypeObject
    """

    # TODO(MSTDL-877):
    #   Fill this out based on
    #   https://docs.python.org/3/c-api/typeobj.html#pytypeobject-definition
    pass


@fieldwise_init
struct PyType_Spec(ImplicitlyCopyable, RegisterPassable):
    """Structure defining a type's behavior.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyType_Spec
    """

    var name: Optional[CStringSlice[ImmStaticOrigin]]
    var basicsize: c_int
    var itemsize: c_int
    var flags: c_uint
    var slots: OptionalPointer[PyType_Slot, MutUntrackedOrigin]


# https://github.com/python/cpython/blob/main/Include/typeslots.h
comptime Py_tp_dealloc = 52
comptime Py_tp_init = 60
comptime Py_tp_methods = 64
comptime Py_tp_new = 65
comptime Py_tp_repr = 66

# https://docs.python.org/3/c-api/typeobj.html#slot-type-typedefs

comptime destructor = def(PyObjectPtr) thin abi("C") -> None
"""`typedef void (*destructor)(PyObject*)`."""
comptime reprfunc = def(PyObjectPtr) thin abi("C") -> PyObjectPtr
"""`typedef PyObject *(*reprfunc)(PyObject*)`."""
comptime Typed_initproc = def(
    PyObjectPtr,
    PyObjectPtr,
    PyObjectPtr,  # NULL if no keyword arguments were passed
) thin abi("C") -> c_int
"""`typedef int (*initproc)(PyObject*, PyObject*, PyObject*)`."""
comptime Typed_newfunc = def(
    PyTypeObjectPtr,
    PyObjectPtr,
    PyObjectPtr,
) thin abi("C") -> PyObjectPtr
"""`typedef PyObject *(*newfunc)(PyTypeObject*, PyObject*, PyObject*)`."""


@fieldwise_init
struct PyType_Slot(ImplicitlyCopyable, RegisterPassable):
    """Structure defining optional functionality of a type, containing a slot ID
    and a value pointer.

    References:
    - https://docs.python.org/3/c-api/type.html#c.PyType_Slot
    - https://docs.python.org/3/c-api/typeobj.html#type-object-structures
    """

    var slot: c_int
    var pfunc: OptionalPointer[NoneType, MutUntrackedOrigin]

    @staticmethod
    def tp_dealloc(func: destructor) -> Self:
        return PyType_Slot(
            Py_tp_dealloc,
            _fn_ptr_as_opaque(func),
        )

    @staticmethod
    def tp_init(func: Typed_initproc) -> Self:
        return PyType_Slot(Py_tp_init, _fn_ptr_as_opaque(func))

    @staticmethod
    def tp_methods(
        methods: OptionalPointer[PyMethodDef, MutUntrackedOrigin]
    ) -> Self:
        return PyType_Slot(
            Py_tp_methods,
            unsafe_cast[Type=NoneType](methods),
        )

    @staticmethod
    def tp_new(func: Typed_newfunc) -> Self:
        return PyType_Slot(Py_tp_new, _fn_ptr_as_opaque(func))

    @staticmethod
    def tp_repr(func: reprfunc) -> Self:
        return PyType_Slot(Py_tp_repr, _fn_ptr_as_opaque(func))

    @staticmethod
    def null() -> Self:
        return PyType_Slot(0, None)


@fieldwise_init
struct PyObject(
    Defaultable,
    ImplicitlyCopyable,
    Writable,
):
    """All object types are extensions of this type. This is a type which
    contains the information Python needs to treat a pointer to an object as an
    object. In a normal “release” build, it contains only the object's reference
    count and a pointer to the corresponding type object. Nothing is actually
    declared to be a PyObject, but every pointer to a Python object can be cast
    to a PyObject.

    References:
    - https://docs.python.org/3/c-api/structures.html#c.PyObject
    """

    var object_ref_count: Py_ssize_t
    var object_type: PyTypeObjectPtr

    def __init__(out self):
        self.object_ref_count = 0
        self.object_type = {}

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyObject(")
        writer.write("object_ref_count=", self.object_ref_count, ",")
        writer.write("object_type=", self.object_type)
        writer.write(")")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `PyObject` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "PyObject").fields(
            fmt.Named("object_ref_count", self.object_ref_count),
            fmt.Named("object_type", self.object_type),
        )


# Mojo doesn't have macros, so we define it here for ease.
struct PyModuleDef_Base(Defaultable, Movable, Writable):
    """PyModuleDef_Base.

    References:
    - https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L39
    - https://pyo3.rs/main/doc/pyo3/ffi/struct.pymoduledef_base
    - `PyModuleDef_HEAD_INIT` default inits all of its members (https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L60)
    """

    var object_base: PyObject
    """The initial segment of every `PyObject` in CPython."""

    comptime _init_fn_type = def() thin abi("C") -> PyObjectPtr
    var init_fn: Self._init_fn_type
    """The function used to re-initialize the module."""

    var index: Py_ssize_t
    """The module's index into its interpreter's `modules_by_index` cache."""

    var dict_copy: PyObjectPtr
    """A copy of the module's `__dict__` after the first time it was loaded."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    def __init__(out self):
        self.object_base = {}
        self.init_fn = _null_fn_ptr[Self._init_fn_type]()
        self.index = 0
        self.dict_copy = {}

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyModuleDef_Base(")
        writer.write("object_base=", self.object_base, ",")
        writer.write("init_fn=<unprintable>", ",")
        writer.write("index=", self.index, ",")
        writer.write("dict_copy=", self.dict_copy)
        writer.write(")")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `PyModuleDef_Base` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "PyModuleDef_Base").fields(
            fmt.Named("object_base", fmt.Repr(self.object_base)),
            fmt.Named("index", self.index),
            fmt.Named("dict_copy", fmt.Repr(self.dict_copy)),
        )


@fieldwise_init
struct PyModuleDef_Slot:
    """A struct representing a slot in the module definition.

    References:
    - https://docs.python.org/3/c-api/module.html#c.PyModuleDef_Slot
    """

    var slot: c_int
    var value: OpaquePointer[MutUntrackedOrigin]


struct PyModuleDef(Movable, Writable):
    """The Python module definition structs that holds all of the information
    needed to create a module.

    References:
    - https://docs.python.org/3/c-api/module.html#c.PyModuleDef
    """

    var base: PyModuleDef_Base

    var name: Optional[CStringSlice[ImmStaticOrigin]]
    """Name for the new module."""

    var docstring: Optional[CStringSlice[ImmStaticOrigin]]
    """Points to the contents of the docstring for the module."""

    var size: Py_ssize_t
    """Size of per-module data."""

    var methods: OptionalPointer[PyMethodDef, MutUntrackedOrigin]
    """A pointer to a table of module-level functions. Can be null if there
    are no functions present."""

    var slots: OptionalPointer[PyModuleDef_Slot, MutUntrackedOrigin]
    """An array of slot definitions for multi-phase initialization, terminated
    by a `{0, NULL}` entry."""

    comptime _visitproc_fn_type = def(
        PyObjectPtr, OpaquePointer[MutUntrackedOrigin]
    ) thin abi("C") -> c_int
    comptime _traverse_fn_type = def(
        PyObjectPtr, Self._visitproc_fn_type, OpaquePointer[MutUntrackedOrigin]
    ) thin abi("C") -> c_int
    var traverse_fn: Self._traverse_fn_type
    """A traversal function to call during GC traversal of the module object,
    or `NULL` if not needed."""

    comptime _clear_fn_type = def(PyObjectPtr) thin abi("C") -> c_int
    var clear_fn: Self._clear_fn_type
    """A clear function to call during GC clearing of the module object,
    or `NULL` if not needed."""

    comptime _free_fn_type = def(OpaquePointer[MutUntrackedOrigin]) thin abi(
        "C"
    ) -> OpaquePointer[MutUntrackedOrigin]
    var free_fn: Self._free_fn_type
    """A function to call during deallocation of the module object,
    or `NULL` if not needed."""

    def __init__(out self, name: StaticString):
        self.base = {}
        self.name = name.as_c_string_slice()
        self.docstring = {}
        # setting `size` to -1 means that the module does not support sub-interpreters
        self.size = -1
        self.methods = {}
        self.slots = {}
        self.traverse_fn = _null_fn_ptr[Self._traverse_fn_type]()
        self.clear_fn = _null_fn_ptr[Self._clear_fn_type]()
        self.free_fn = _null_fn_ptr[Self._free_fn_type]()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Formats to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write("PyModuleDef(")
        writer.write("base=", self.base, ",")
        writer.write("name=", self.name, ",")
        writer.write("docstring=", self.docstring, ",")
        writer.write("size=", self.size, ",")
        writer.write("methods=", self.methods, ",")
        writer.write("slots=", self.slots, ",")
        writer.write("traverse_fn=<unprintable>", ",")
        writer.write("clear_fn=<unprintable>", ",")
        writer.write("free_fn=<unprintable>")
        writer.write(")")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this `PyModuleDef` to a writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "PyModuleDef").fields(
            fmt.Named("name", self.name),
            fmt.Named("size", self.size),
            fmt.Named("methods", self.methods),
            fmt.Named("slots", self.slots),
        )


# ===-------------------------------------------------------------------===#
# CPython C API Functions
# ===-------------------------------------------------------------------===#


struct ExternalFunction[
    name: StaticString,
    type: TrivialRegisterPassable,
]:
    @staticmethod
    @always_inline
    def load(lib: _DLHandle) -> Self.type:
        """Loads this external function from an opened dynamic library."""
        return lib._get_function[Self.name, Self.type]()


# external functions for the CPython C API
# ordered based on https://docs.python.org/3/c-api/index.html

# The Very High Level Layer
comptime PyRun_SimpleString = ExternalFunction[
    "PyRun_SimpleString",
    # int PyRun_SimpleString(const char *command)
    def(OptionalPointer[c_char, ImmutAnyOrigin]) thin abi("C") -> c_int,
]
comptime PyRun_String = ExternalFunction[
    "PyRun_String",
    # PyObject *PyRun_String(const char *str, int start, PyObject *globals, PyObject *locals)
    def(
        OptionalPointer[c_char, ImmutAnyOrigin],
        c_int,
        PyObjectPtr,
        PyObjectPtr,
    ) thin abi("C") -> PyObjectPtr,
]
comptime Py_CompileString = ExternalFunction[
    "Py_CompileString",
    # PyObject *Py_CompileString(const char *str, const char *filename, int start)
    def(
        OptionalPointer[c_char, ImmutAnyOrigin],
        OptionalPointer[c_char, ImmutAnyOrigin],
        c_int,
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyEval_EvalCode = ExternalFunction[
    "PyEval_EvalCode",
    # PyObject *PyEval_EvalCode(PyObject *co, PyObject *globals, PyObject *locals)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Reference Counting
comptime Py_NewRef = ExternalFunction[
    "Py_NewRef",
    # PyObject *Py_NewRef(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime Py_IncRef = ExternalFunction[
    "Py_IncRef",
    # void Py_IncRef(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> None,
]
comptime Py_DecRef = ExternalFunction[
    "Py_DecRef",
    # void Py_DecRef(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> None,
]

# Exception Handling
# - Printing and clearing
comptime PyErr_Clear = ExternalFunction[
    "PyErr_Clear",
    # void PyErr_Clear()
    def() thin abi("C") -> None,
]
# - Raising exceptions
comptime PyErr_SetString = ExternalFunction[
    "PyErr_SetString",
    # void PyErr_SetString(PyObject *type, const char *message)
    def(
        PyObjectPtr, OptionalPointer[c_char, ImmutAnyOrigin]
    ) thin abi("C") -> None,
]
comptime PyErr_SetNone = ExternalFunction[
    "PyErr_SetNone",
    # void PyErr_SetNone(PyObject *type)
    def(PyObjectPtr) thin abi("C") -> None,
]
# - Querying the error indicator
comptime PyErr_Occurred = ExternalFunction[
    "PyErr_Occurred",
    # PyObject *PyErr_Occurred()
    def() thin abi("C") -> PyObjectPtr,
]
comptime PyErr_GetRaisedException = ExternalFunction[
    "PyErr_GetRaisedException",
    # PyObject *PyErr_GetRaisedException()
    def() thin abi("C") -> PyObjectPtr,
]
comptime PyErr_Fetch = ExternalFunction[
    "PyErr_Fetch",
    # void PyErr_Fetch(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback)
    def(
        OptionalPointer[PyObjectPtr, MutUntrackedOrigin],
        OptionalPointer[PyObjectPtr, MutUntrackedOrigin],
        OptionalPointer[PyObjectPtr, MutUntrackedOrigin],
    ) thin abi("C") -> None,
]
comptime PyErr_Restore = ExternalFunction[
    "PyErr_Restore",
    # void PyErr_Restore(PyObject *type, PyObject *value, PyObject *traceback)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> None,
]

# Initialization, Finalization, and Threads
comptime PyEval_SaveThread = ExternalFunction[
    "PyEval_SaveThread",
    # PyThreadState *PyEval_SaveThread()
    def() thin abi("C") -> OptionalPointer[PyThreadState, MutUntrackedOrigin],
]
comptime PyEval_RestoreThread = ExternalFunction[
    "PyEval_RestoreThread",
    # void PyEval_RestoreThread(PyThreadState *tstate)
    def(
        OptionalPointer[PyThreadState, MutUntrackedOrigin]
    ) thin abi("C") -> None,
]
comptime PyGILState_Ensure = ExternalFunction[
    "PyGILState_Ensure",
    # PyGILState_STATE PyGILState_Ensure()
    def() thin abi("C") -> PyGILState_STATE,
]
comptime PyGILState_Release = ExternalFunction[
    "PyGILState_Release",
    # void PyGILState_Release(PyGILState_STATE)
    def(PyGILState_STATE) thin abi("C") -> None,
]
comptime PyGILState_Check = ExternalFunction[
    "PyGILState_Check",
    # int PyGILState_Check()
    def() thin abi("C") -> c_int,
]

# Importing Modules
comptime PyImport_ImportModule = ExternalFunction[
    "PyImport_ImportModule",
    # PyObject *PyImport_ImportModule(const char *name)
    def(OptionalPointer[c_char, ImmutAnyOrigin]) thin abi("C") -> PyObjectPtr,
]
comptime PyImport_AddModule = ExternalFunction[
    "PyImport_AddModule",
    # PyObject *PyImport_AddModule(const char *name)
    def(OptionalPointer[c_char, ImmutAnyOrigin]) thin abi("C") -> PyObjectPtr,
]

# Abstract Objects Layer
# Object Protocol
comptime PyObject_HasAttrString = ExternalFunction[
    "PyObject_HasAttrString",
    # int PyObject_HasAttrString(PyObject *o, const char *attr_name)
    def(
        PyObjectPtr, OptionalPointer[c_char, ImmutAnyOrigin]
    ) thin abi("C") -> c_int,
]
comptime PyObject_GetAttrString = ExternalFunction[
    "PyObject_GetAttrString",
    # PyObject *PyObject_GetAttrString(PyObject *o, const char *attr_name)
    def(
        PyObjectPtr, OptionalPointer[c_char, ImmutAnyOrigin]
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyObject_SetAttrString = ExternalFunction[
    "PyObject_SetAttrString",
    # int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v)
    def(
        PyObjectPtr,
        OptionalPointer[c_char, ImmutAnyOrigin],
        PyObjectPtr,
    ) thin abi("C") -> c_int,
]
comptime PyObject_Str = ExternalFunction[
    "PyObject_Str",
    # PyObject *PyObject_Str(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyObject_Hash = ExternalFunction[
    "PyObject_Hash",
    # Py_hash_t PyObject_Hash(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> Py_hash_t,
]
comptime PyObject_IsTrue = ExternalFunction[
    "PyObject_IsTrue",
    # int PyObject_IsTrue(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> c_int,
]
comptime PyObject_Type = ExternalFunction[
    "PyObject_Type",
    # PyTypeObject *PyObject_Type(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyObject_Length = ExternalFunction[
    "PyObject_Length",
    # Py_ssize_t PyObject_Length(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> Py_ssize_t,
]
comptime PyObject_GetItem = ExternalFunction[
    "PyObject_GetItem",
    # PyObject *PyObject_GetItem(PyObject *o, PyObject *key)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyObject_SetItem = ExternalFunction[
    "PyObject_SetItem",
    # int PyObject_SetItem(PyObject *o, PyObject *key, PyObject *v)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int,
]
comptime PyObject_GetIter = ExternalFunction[
    "PyObject_GetIter",
    # PyObject *PyObject_GetIter(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Call Protocol
comptime PyObject_Call = ExternalFunction[
    "PyObject_Call",
    # PyObject *PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyObject_CallObject = ExternalFunction[
    "PyObject_CallObject",
    # PyObject *PyObject_CallObject(PyObject *callable, PyObject *args)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Number Protocol
comptime PyNumber_Long = ExternalFunction[
    "PyNumber_Long",
    # PyObject *PyNumber_Long(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Float = ExternalFunction[
    "PyNumber_Float",
    # PyObject *PyNumber_Float(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Add = ExternalFunction[
    "PyNumber_Add",
    # PyObject *PyNumber_Add(PyObject *o1, PyObject *o2)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Subtract = ExternalFunction[
    "PyNumber_Subtract",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Multiply = ExternalFunction[
    "PyNumber_Multiply",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_TrueDivide = ExternalFunction[
    "PyNumber_TrueDivide",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_FloorDivide = ExternalFunction[
    "PyNumber_FloorDivide",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Remainder = ExternalFunction[
    "PyNumber_Remainder",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Lshift = ExternalFunction[
    "PyNumber_Lshift",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Rshift = ExternalFunction[
    "PyNumber_Rshift",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_And = ExternalFunction[
    "PyNumber_And",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Or = ExternalFunction[
    "PyNumber_Or",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Xor = ExternalFunction[
    "PyNumber_Xor",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Power = ExternalFunction[
    "PyNumber_Power",
    # PyObject *PyNumber_Power(PyObject *o1, PyObject *o2, PyObject *o3)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Negative = ExternalFunction[
    "PyNumber_Negative",
    # PyObject *PyNumber_Negative(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Positive = ExternalFunction[
    "PyNumber_Positive",
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_Invert = ExternalFunction[
    "PyNumber_Invert",
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceAdd = ExternalFunction[
    "PyNumber_InPlaceAdd",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceSubtract = ExternalFunction[
    "PyNumber_InPlaceSubtract",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceMultiply = ExternalFunction[
    "PyNumber_InPlaceMultiply",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceTrueDivide = ExternalFunction[
    "PyNumber_InPlaceTrueDivide",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceFloorDivide = ExternalFunction[
    "PyNumber_InPlaceFloorDivide",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceRemainder = ExternalFunction[
    "PyNumber_InPlaceRemainder",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceLshift = ExternalFunction[
    "PyNumber_InPlaceLshift",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceRshift = ExternalFunction[
    "PyNumber_InPlaceRshift",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceAnd = ExternalFunction[
    "PyNumber_InPlaceAnd",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceOr = ExternalFunction[
    "PyNumber_InPlaceOr",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlaceXor = ExternalFunction[
    "PyNumber_InPlaceXor",
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyNumber_InPlacePower = ExternalFunction[
    "PyNumber_InPlacePower",
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Object Protocol (rich comparison)

# `opid` values for `PyObject_RichCompare`, from CPython's `object.h`.
comptime Py_LT = c_int(0)
comptime Py_LE = c_int(1)
comptime Py_EQ = c_int(2)
comptime Py_NE = c_int(3)
comptime Py_GT = c_int(4)
comptime Py_GE = c_int(5)

comptime PyObject_RichCompare = ExternalFunction[
    "PyObject_RichCompare",
    # PyObject *PyObject_RichCompare(PyObject *o1, PyObject *o2, int opid)
    def(PyObjectPtr, PyObjectPtr, c_int) thin abi("C") -> PyObjectPtr,
]

# Sequence Protocol
comptime PySequence_Contains = ExternalFunction[
    "PySequence_Contains",
    # int PySequence_Contains(PyObject *o, PyObject *value)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int,
]

# Iterator Protocol
comptime PyIter_Check = ExternalFunction[
    "PyIter_Check",
    # int PyIter_Check(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> c_int,
]
comptime PyIter_Next = ExternalFunction[
    "PyIter_Next",
    # PyObject *PyIter_Next(PyObject *o)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Concrete Objects Layer
# Type Objects
comptime PyType_GenericAlloc = ExternalFunction[
    "PyType_GenericAlloc",
    # PyObject *PyType_GenericAlloc(PyTypeObject *type, Py_ssize_t nitems)
    def(PyTypeObjectPtr, Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyType_GetName = ExternalFunction[
    "PyType_GetName",
    # PyObject *PyType_GetName(PyTypeObject *type)
    def(PyTypeObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyType_FromSpec = ExternalFunction[
    "PyType_FromSpec",
    # PyObject *PyType_FromSpec(PyType_Spec *spec)
    def(
        OptionalPointer[PyType_Spec, MutAnyOrigin]
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyType_GetFlags = ExternalFunction[
    "PyType_GetFlags",
    # unsigned long PyType_GetFlags(PyTypeObject *type)
    def(PyTypeObjectPtr) thin abi("C") -> c_ulong,
]
comptime PyType_IsSubtype = ExternalFunction[
    "PyType_IsSubtype",
    # int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b)
    def(PyTypeObjectPtr, PyTypeObjectPtr) thin abi("C") -> c_int,
]

# Integer Objects
comptime PyLong_FromSsize_t = ExternalFunction[
    "PyLong_FromSsize_t",
    # PyObject *PyLong_FromSsize_t(Py_ssize_t v)
    def(Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyLong_FromSize_t = ExternalFunction[
    "PyLong_FromSize_t",
    # PyObject *PyLong_FromSize_t(size_t v)
    def(c_size_t) thin abi("C") -> PyObjectPtr,
]
comptime PyLong_AsSsize_t = ExternalFunction[
    "PyLong_AsSsize_t",
    # Py_ssize_t PyLong_AsSsize_t(PyObject *pylong)
    def(PyObjectPtr) thin abi("C") -> Py_ssize_t,
]
comptime PyLong_AsSize_t = ExternalFunction[
    "PyLong_AsSize_t",
    # size_t PyLong_AsSize_t(PyObject *pylong)
    def(PyObjectPtr) thin abi("C") -> c_size_t,
]

# Boolean Objects
comptime PyBool_FromLong = ExternalFunction[
    "PyBool_FromLong",
    # PyObject *PyBool_FromLong(long v)
    def(c_long) thin abi("C") -> PyObjectPtr,
]

# Floating-Point Objects
comptime PyFloat_FromDouble = ExternalFunction[
    "PyFloat_FromDouble",
    # PyObject *PyFloat_FromDouble(double v)
    def(c_double) thin abi("C") -> PyObjectPtr,
]
comptime PyFloat_AsDouble = ExternalFunction[
    "PyFloat_AsDouble",
    # double PyFloat_AsDouble(PyObject *pyfloat)
    def(PyObjectPtr) thin abi("C") -> c_double,
]

# Unicode Objects and Codecs
comptime PyUnicode_DecodeUTF8 = ExternalFunction[
    "PyUnicode_DecodeUTF8",
    # PyObject *PyUnicode_DecodeUTF8(const char *str, Py_ssize_t size, const char *errors)
    def(
        OptionalPointer[c_char, ImmutAnyOrigin],
        Py_ssize_t,
        OptionalPointer[c_char, ImmutAnyOrigin],
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyUnicode_AsUTF8AndSize = ExternalFunction[
    "PyUnicode_AsUTF8AndSize",
    # const char *PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size)
    def(
        PyObjectPtr,
        OptionalPointer[Py_ssize_t, MutUntrackedOrigin],
    ) thin abi("C") -> OptionalPointer[c_char, ImmutAnyOrigin],
]

# Tuple Objects
comptime PyTuple_New = ExternalFunction[
    "PyTuple_New",
    # PyObject *PyTuple_New(Py_ssize_t len)
    def(Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyTuple_GetItem = ExternalFunction[
    "PyTuple_GetItem",
    # PyObject *PyTuple_GetItem(PyObject *p, Py_ssize_t pos)
    def(PyObjectPtr, Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyTuple_SetItem = ExternalFunction[
    "PyTuple_SetItem",
    # int PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o)
    def(PyObjectPtr, Py_ssize_t, PyObjectPtr) thin abi("C") -> c_int,
]

# List Objects
comptime PyList_New = ExternalFunction[
    "PyList_New",
    # PyObject *PyList_New(Py_ssize_t len)
    def(Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyList_GetItem = ExternalFunction[
    "PyList_GetItem",
    # PyObject *PyList_GetItem(PyObject *list, Py_ssize_t index)
    def(PyObjectPtr, Py_ssize_t) thin abi("C") -> PyObjectPtr,
]
comptime PyList_SetItem = ExternalFunction[
    "PyList_SetItem",
    # int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item)
    def(PyObjectPtr, Py_ssize_t, PyObjectPtr) thin abi("C") -> c_int,
]

# Dictionary Objects
comptime PyDict_New = ExternalFunction[
    "PyDict_New",
    # PyObject *PyDict_New()
    def() thin abi("C") -> PyObjectPtr,
]
comptime PyDict_SetItem = ExternalFunction[
    "PyDict_SetItem",
    # int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int,
]
comptime PyDict_GetItemWithError = ExternalFunction[
    "PyDict_GetItemWithError",
    # PyObject *PyDict_GetItemWithError(PyObject *p, PyObject *key)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyDict_Next = ExternalFunction[
    "PyDict_Next",
    # int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
    def(
        PyObjectPtr,
        OptionalPointer[Py_ssize_t, MutAnyOrigin],
        OptionalPointer[PyObjectPtr, MutAnyOrigin],
        OptionalPointer[PyObjectPtr, MutAnyOrigin],
    ) thin abi("C") -> c_int,
]

# Set Objects
comptime PySet_New = ExternalFunction[
    "PySet_New",
    # PyObject *PySet_New(PyObject *iterable)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PySet_Add = ExternalFunction[
    "PySet_Add",
    # int PySet_Add(PyObject *set, PyObject *key)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int,
]

# Module Objects
comptime PyModule_GetDict = ExternalFunction[
    "PyModule_GetDict",
    # PyObject *PyModule_GetDict(PyObject *module)
    def(PyObjectPtr) thin abi("C") -> PyObjectPtr,
]
comptime PyModule_Create2 = ExternalFunction[
    "PyModule_Create2",
    # PyObject *PyModule_Create2(PyModuleDef *def, int module_api_version)
    def(
        OptionalPointer[PyModuleDef, MutUntrackedOrigin], c_int
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyModule_AddFunctions = ExternalFunction[
    "PyModule_AddFunctions",
    # int PyModule_AddFunctions(PyObject *module, PyMethodDef *functions)
    def(
        PyObjectPtr, OptionalPointer[PyMethodDef, MutAnyOrigin]
    ) thin abi("C") -> c_int,
]
comptime PyModule_AddObjectRef = ExternalFunction[
    "PyModule_AddObjectRef",
    # int PyModule_AddObjectRef(PyObject *module, const char *name, PyObject *value)
    def(
        PyObjectPtr,
        OptionalPointer[c_char, ImmutAnyOrigin],
        PyObjectPtr,
    ) thin abi("C") -> c_int,
]

# Slice Objects
comptime PySlice_New = ExternalFunction[
    "PySlice_New",
    # PyObject *PySlice_New(PyObject *start, PyObject *stop, PyObject *step)
    def(PyObjectPtr, PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr,
]

# Capsules
comptime PyCapsule_Destructor = (
    # typedef void (*PyCapsule_Destructor)(PyObject *)
    destructor
)
comptime PyCapsule_New = ExternalFunction[
    "PyCapsule_New",
    # PyObject *PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
    def(
        OpaquePointer[MutUntrackedOrigin],
        OptionalPointer[c_char, ImmutAnyOrigin],
        PyCapsule_Destructor,
    ) thin abi("C") -> PyObjectPtr,
]
comptime PyCapsule_GetPointer = ExternalFunction[
    "PyCapsule_GetPointer",
    # void *PyCapsule_GetPointer(PyObject *capsule, const char *name)
    def(
        PyObjectPtr, OptionalPointer[c_char, ImmutAnyOrigin]
    ) thin abi("C") -> OpaquePointer[MutUntrackedOrigin],
]
comptime PyCapsule_IsValid = ExternalFunction[
    "PyCapsule_IsValid",
    # int PyCapsule_IsValid(PyObject *capsule, const char *name)
    def(
        PyObjectPtr, OptionalPointer[c_char, ImmutAnyOrigin]
    ) thin abi("C") -> c_int,
]

# Memory Management
comptime PyObject_Free = ExternalFunction[
    "PyObject_Free",
    # void PyObject_Free(void *p)
    def(OptionalPointer[NoneType, MutUntrackedOrigin]) thin abi("C") -> None,
]

# Object Implementation Support
# Common Object Structures
comptime Py_Is = ExternalFunction[
    "Py_Is",
    # int Py_Is(PyObject *x, PyObject *y)
    def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int,
]


def _PyErr_GetRaisedException_dummy() abi("C") -> PyObjectPtr:
    abort("PyErr_GetRaisedException is not available in this Python version")


def _PyType_GetName_dummy(type: PyTypeObjectPtr) abi("C") -> PyObjectPtr:
    abort("PyType_GetName is not available in this Python version")


# ===-------------------------------------------------------------------===#
# Context Managers for Python GIL and Threading
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct GILAcquired(Movable):
    """Context manager for Python Global Interpreter Lock (GIL) operations.

    This struct provides automatic GIL management inspired by nanobind/pybind11.
    It ensures the GIL is acquired on construction and released on destruction,
    making it safe to use Python objects within the managed scope.

    Example:
        ```mojo
        from std.python import Python
        from std.python._cpython import GILAcquired

        var python = Python()
        with GILAcquired(python):
            # Python objects can be safely accessed here
            pass
        # GIL is automatically released here
        ```
    """

    var python: Python
    """Reference to the CPython instance."""
    var gil_state: PyGILState_STATE
    """The GIL state returned by PyGILState_Ensure."""

    def __init__(out self, python: Python):
        """Acquire the GIL and initialize the context manager.

        Args:
            python: The CPython instance to use for GIL operations.
        """
        self.python = python
        self.gil_state = PyGILState_STATE(PyGILState_STATE.PyGILState_UNLOCKED)

    def __enter__(mut self):
        """Acquire the GIL."""
        self.gil_state = self.python.cpython().PyGILState_Ensure()

    def __exit__(mut self):
        """Release the GIL."""
        self.python.cpython().PyGILState_Release(self.gil_state)


@fieldwise_init
struct GILReleased(Movable):
    """Context manager for Python thread state operations.

    This struct provides automatic thread state management for scenarios where
    you need to temporarily release the GIL to allow other threads to run,
    then restore the thread state. This is useful for long-running operations
    that don't need to access Python objects.

    Example:
        ```mojo
        from std.python import Python
        from std.python._cpython import GILReleased

        var python = Python()
        with GILReleased(python):
            # GIL is released here, other threads can run
            # Perform CPU-intensive work without Python object access
            var total = 0
            for i in range(1000):
                total += i
        # Thread state is automatically restored here
        ```
    """

    var python: Python
    """Reference to the CPython instance."""
    var thread_state: OptionalPointer[PyThreadState, MutUntrackedOrigin]
    """The thread state returned by PyEval_SaveThread."""

    def __init__(out self, python: Python):
        """Save the current thread state and release the GIL.

        Args:
            python: The Python instance to use for GIL operations.
        """
        self.python = python
        self.thread_state = {}

    def __enter__(mut self):
        """Save the current thread state and release the GIL."""
        self.thread_state = self.python.cpython().PyEval_SaveThread()

    def __exit__(mut self):
        """Restore the thread state and acquire the GIL."""
        self.python.cpython().PyEval_RestoreThread(self.thread_state)


def _untracked_symbol[
    result_type: AnyType
](mut lib: OwnedDLHandle, name: StringSlice) -> Optional[
    Pointer[result_type, MutUntrackedOrigin]
]:
    """Resolves a symbol and drops the borrow tying it to `lib`.

    `OwnedDLHandle.get_symbol` hands back a pointer borrowed from the handle,
    which is what stops the library being `dlclose`d underneath it. `CPython`
    caches its symbols in fields alongside the handle itself, so the borrow
    cannot be expressed — a field would have to name a sibling field's origin.
    Dropping it is sound here because `lib` and the cached pointers are fields
    of the same struct, so the library outlives every read of them.

    Parameters:
        result_type: The type of the symbol to return.

    Args:
        lib: The library to resolve the symbol in.
        name: The name of the symbol to resolve.

    Returns:
        An optional pointer to the symbol, or `None` if not found.
    """
    var ptr = lib.get_symbol[result_type](name)
    if not ptr:
        return None
    return ptr.unsafe_value().unsafe_origin_cast[MutUntrackedOrigin]()


@fieldwise_init
struct CPython(Defaultable, Movable):
    """Handle to the CPython interpreter present in the current process.

    This type is non-copyable due to its large size. Please refer to it only
    using either a reference, or the `Python` handle type."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var lib: OwnedDLHandle
    """The handle to the CPython shared library."""
    var version: PythonVersion
    """The version of the Python runtime."""
    var init_error: StaticString
    """An error message if initialization failed."""

    # fields holding function pointers to CPython C API functions
    # ordered based on https://docs.python.org/3/c-api/index.html

    # The Very High Level Layer
    var _PyRun_SimpleString: PyRun_SimpleString.type
    var _PyRun_String: PyRun_String.type
    var _Py_CompileString: Py_CompileString.type
    var _PyEval_EvalCode: PyEval_EvalCode.type
    # Reference Counting
    var _Py_NewRef: Py_NewRef.type
    var _Py_IncRef: Py_IncRef.type
    var _Py_DecRef: Py_DecRef.type
    # Exception Handling
    var _PyErr_Clear: PyErr_Clear.type
    var _PyErr_SetString: PyErr_SetString.type
    var _PyErr_SetNone: PyErr_SetNone.type
    var _PyErr_Occurred: PyErr_Occurred.type
    var _PyErr_GetRaisedException: PyErr_GetRaisedException.type
    var _PyErr_Fetch: PyErr_Fetch.type
    var _PyErr_Restore: PyErr_Restore.type
    # Initialization, Finalization, and Threads
    var _PyEval_SaveThread: PyEval_SaveThread.type
    var _PyEval_RestoreThread: PyEval_RestoreThread.type
    var _PyGILState_Ensure: PyGILState_Ensure.type
    var _PyGILState_Release: PyGILState_Release.type
    var _PyGILState_Check: PyGILState_Check.type
    # Importing Modules
    var _PyImport_ImportModule: PyImport_ImportModule.type
    var _PyImport_AddModule: PyImport_AddModule.type
    # Abstract Objects Layer
    # Object Protocol
    var _PyObject_HasAttrString: PyObject_HasAttrString.type
    var _PyObject_GetAttrString: PyObject_GetAttrString.type
    var _PyObject_SetAttrString: PyObject_SetAttrString.type
    var _PyObject_Str: PyObject_Str.type
    var _PyObject_Hash: PyObject_Hash.type
    var _PyObject_IsTrue: PyObject_IsTrue.type
    var _PyObject_Type: PyObject_Type.type
    var _PyObject_Length: PyObject_Length.type
    var _PyObject_GetItem: PyObject_GetItem.type
    var _PyObject_SetItem: PyObject_SetItem.type
    var _PyObject_GetIter: PyObject_GetIter.type
    # Call Protocol
    var _PyObject_Call: PyObject_Call.type
    var _PyObject_CallObject: PyObject_CallObject.type
    # Number Protocol
    var _PyNumber_Long: PyNumber_Long.type
    var _PyNumber_Float: PyNumber_Float.type
    var _PyNumber_Add: PyNumber_Add.type
    var _PyNumber_Subtract: PyNumber_Subtract.type
    var _PyNumber_Multiply: PyNumber_Multiply.type
    var _PyNumber_TrueDivide: PyNumber_TrueDivide.type
    var _PyNumber_FloorDivide: PyNumber_FloorDivide.type
    var _PyNumber_Remainder: PyNumber_Remainder.type
    var _PyNumber_Lshift: PyNumber_Lshift.type
    var _PyNumber_Rshift: PyNumber_Rshift.type
    var _PyNumber_And: PyNumber_And.type
    var _PyNumber_Or: PyNumber_Or.type
    var _PyNumber_Xor: PyNumber_Xor.type
    var _PyNumber_Power: PyNumber_Power.type
    var _PyNumber_Negative: PyNumber_Negative.type
    var _PyNumber_Positive: PyNumber_Positive.type
    var _PyNumber_Invert: PyNumber_Invert.type
    var _PyNumber_InPlaceAdd: PyNumber_InPlaceAdd.type
    var _PyNumber_InPlaceSubtract: PyNumber_InPlaceSubtract.type
    var _PyNumber_InPlaceMultiply: PyNumber_InPlaceMultiply.type
    var _PyNumber_InPlaceTrueDivide: PyNumber_InPlaceTrueDivide.type
    var _PyNumber_InPlaceFloorDivide: PyNumber_InPlaceFloorDivide.type
    var _PyNumber_InPlaceRemainder: PyNumber_InPlaceRemainder.type
    var _PyNumber_InPlaceLshift: PyNumber_InPlaceLshift.type
    var _PyNumber_InPlaceRshift: PyNumber_InPlaceRshift.type
    var _PyNumber_InPlaceAnd: PyNumber_InPlaceAnd.type
    var _PyNumber_InPlaceOr: PyNumber_InPlaceOr.type
    var _PyNumber_InPlaceXor: PyNumber_InPlaceXor.type
    var _PyNumber_InPlacePower: PyNumber_InPlacePower.type
    # Object Protocol (rich comparison)
    var _PyObject_RichCompare: PyObject_RichCompare.type
    # Sequence Protocol
    var _PySequence_Contains: PySequence_Contains.type
    # Iterator Protocol
    var _PyIter_Check: PyIter_Check.type
    var _PyIter_Next: PyIter_Next.type
    # Concrete Objects Layer
    # Type Objects
    var _PyType_GetFlags: PyType_GetFlags.type
    var _PyType_IsSubtype: PyType_IsSubtype.type
    var _PyType_GenericAlloc: PyType_GenericAlloc.type
    var _PyType_GetName: PyType_GetName.type
    var _PyType_FromSpec: PyType_FromSpec.type
    # The None Object
    var _Py_None: PyObjectPtr
    # Integer Objects
    var _PyLong_Type: PyTypeObjectPtr
    var _PyLong_FromSsize_t: PyLong_FromSsize_t.type
    var _PyLong_FromSize_t: PyLong_FromSize_t.type
    var _PyLong_AsSsize_t: PyLong_AsSsize_t.type
    var _PyLong_AsSize_t: PyLong_AsSize_t.type
    # Boolean Objects
    var _PyBool_Type: PyTypeObjectPtr
    var _PyBool_FromLong: PyBool_FromLong.type
    # Floating-Point Objects
    var _PyFloat_Type: PyTypeObjectPtr
    var _PyFloat_FromDouble: PyFloat_FromDouble.type
    var _PyFloat_AsDouble: PyFloat_AsDouble.type
    # Unicode Objects and Codecs
    var _PyUnicode_DecodeUTF8: PyUnicode_DecodeUTF8.type
    var _PyUnicode_AsUTF8AndSize: PyUnicode_AsUTF8AndSize.type
    # Tuple Objects
    var _PyTuple_New: PyTuple_New.type
    var _PyTuple_GetItem: PyTuple_GetItem.type
    var _PyTuple_SetItem: PyTuple_SetItem.type
    # List Objects
    var _PyList_New: PyList_New.type
    var _PyList_GetItem: PyList_GetItem.type
    var _PyList_SetItem: PyList_SetItem.type
    # Dictionary Objects
    var _PyDict_Type: PyTypeObjectPtr
    var _PyDict_New: PyDict_New.type
    var _PyDict_SetItem: PyDict_SetItem.type
    var _PyDict_GetItemWithError: PyDict_GetItemWithError.type
    var _PyDict_Next: PyDict_Next.type
    # Set Objects
    var _PySet_New: PySet_New.type
    var _PySet_Add: PySet_Add.type
    # Module Objects
    var _PyModule_GetDict: PyModule_GetDict.type
    var _PyModule_Create2: PyModule_Create2.type
    var _PyModule_AddFunctions: PyModule_AddFunctions.type
    var _PyModule_AddObjectRef: PyModule_AddObjectRef.type
    # Slice Objects
    var _PySlice_New: PySlice_New.type
    # Capsules
    var _PyCapsule_New: PyCapsule_New.type
    var _PyCapsule_GetPointer: PyCapsule_GetPointer.type
    var _PyCapsule_IsValid: PyCapsule_IsValid.type
    # Memory Management
    var _PyObject_Free: PyObject_Free.type
    # Object Implementation Support
    # Common Object Structures
    var _Py_Is: Py_Is.type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        # Add directory of target file to top of sys.path to find python modules
        var file_dir = dirname(argv()[0])
        if Path(file_dir).is_dir() or file_dir == "":
            var python_path = getenv("PYTHONPATH")
            # A leading `:` will put the current dir at the top of sys.path.
            # If we're doing `mojo run main.mojo` or `./main`, the returned
            # `dirname` will be an empty string.
            if file_dir == "" and not python_path:
                file_dir = ":"
            if python_path:
                _ = setenv("PYTHONPATH", String(t"{file_dir}:{python_path}"))
            else:
                _ = setenv("PYTHONPATH", file_dir)

        # TODO(MOCO-772) Allow raises to propagate through function pointers
        # and make this initialization a raising function.
        self.init_error = StaticString(
            unsafe_from_utf8=CStringSlice(
                unsafe_from_ptr=external_call[
                    "KGEN_CompilerRT_Python_SetPythonPath",
                    Pointer[c_char, ImmStaticOrigin],
                ]()
            )
        )

        var python_lib = getenv("MOJO_PYTHON_LIBRARY")

        # Note:
        #   MOJO_PYTHON_LIBRARY can be "" when the current Mojo program
        #   is a dynamic library being loaded as a Python extension module,
        #   and we need to find CPython symbols that are statically linked
        #   into the `python` main executable. On those platforms where
        #   `python` executable can be statically linked (Linux), it's
        #   important that we don't load a second copy of CPython symbols
        #   into the process by loading the `libpython` dynamic library.
        try:
            # Try to load the library from the current process.
            self.lib = OwnedDLHandle()
            if not self.lib.check_symbol("Py_Initialize"):
                # If the library is not present in the current process, try to load it from the environment variable.
                self.lib = OwnedDLHandle(python_lib)
        except e:
            abort(t"Failed to load libpython from {python_lib}:\n{e}")

        if not self.init_error:
            if not self.lib.check_symbol("Py_Initialize"):
                self.init_error = "compatible Python library not found"
            self.lib.call["Py_Initialize"]()
            self.version = PythonVersion(_py_get_version(self.lib.borrow()))
        else:
            self.version = PythonVersion(0, 0, 0)

        # The Very High Level Layer
        self._PyRun_SimpleString = PyRun_SimpleString.load(self.lib.borrow())
        self._PyRun_String = PyRun_String.load(self.lib.borrow())
        self._Py_CompileString = Py_CompileString.load(self.lib.borrow())
        self._PyEval_EvalCode = PyEval_EvalCode.load(self.lib.borrow())
        # Reference Counting
        self._Py_NewRef = Py_NewRef.load(self.lib.borrow())
        self._Py_IncRef = Py_IncRef.load(self.lib.borrow())
        self._Py_DecRef = Py_DecRef.load(self.lib.borrow())
        # Exception Handling
        self._PyErr_Clear = PyErr_Clear.load(self.lib.borrow())
        self._PyErr_SetString = PyErr_SetString.load(self.lib.borrow())
        self._PyErr_SetNone = PyErr_SetNone.load(self.lib.borrow())
        self._PyErr_Occurred = PyErr_Occurred.load(self.lib.borrow())
        if self.version.minor >= 12:
            self._PyErr_GetRaisedException = PyErr_GetRaisedException.load(
                self.lib.borrow()
            )
        else:
            self._PyErr_GetRaisedException = _PyErr_GetRaisedException_dummy
        self._PyErr_Fetch = PyErr_Fetch.load(self.lib.borrow())
        self._PyErr_Restore = PyErr_Restore.load(self.lib.borrow())
        # Initialization, Finalization, and Threads
        self._PyEval_SaveThread = PyEval_SaveThread.load(self.lib.borrow())
        self._PyEval_RestoreThread = PyEval_RestoreThread.load(
            self.lib.borrow()
        )
        self._PyGILState_Ensure = PyGILState_Ensure.load(self.lib.borrow())
        self._PyGILState_Release = PyGILState_Release.load(self.lib.borrow())
        self._PyGILState_Check = PyGILState_Check.load(self.lib.borrow())
        # Importing Modules
        self._PyImport_ImportModule = PyImport_ImportModule.load(
            self.lib.borrow()
        )
        self._PyImport_AddModule = PyImport_AddModule.load(self.lib.borrow())
        # Abstract Objects Layer
        # Object Protocol
        self._PyObject_HasAttrString = PyObject_HasAttrString.load(
            self.lib.borrow()
        )
        self._PyObject_GetAttrString = PyObject_GetAttrString.load(
            self.lib.borrow()
        )
        self._PyObject_SetAttrString = PyObject_SetAttrString.load(
            self.lib.borrow()
        )
        self._PyObject_Str = PyObject_Str.load(self.lib.borrow())
        self._PyObject_Hash = PyObject_Hash.load(self.lib.borrow())
        self._PyObject_IsTrue = PyObject_IsTrue.load(self.lib.borrow())
        self._PyObject_Type = PyObject_Type.load(self.lib.borrow())
        self._PyObject_Length = PyObject_Length.load(self.lib.borrow())
        self._PyObject_GetItem = PyObject_GetItem.load(self.lib.borrow())
        self._PyObject_SetItem = PyObject_SetItem.load(self.lib.borrow())
        self._PyObject_GetIter = PyObject_GetIter.load(self.lib.borrow())
        # Call Protocol
        self._PyObject_Call = PyObject_Call.load(self.lib.borrow())
        self._PyObject_CallObject = PyObject_CallObject.load(self.lib.borrow())
        # Number Protocol
        self._PyNumber_Long = PyNumber_Long.load(self.lib.borrow())
        self._PyNumber_Float = PyNumber_Float.load(self.lib.borrow())
        self._PyNumber_Add = PyNumber_Add.load(self.lib.borrow())
        self._PyNumber_Subtract = PyNumber_Subtract.load(self.lib.borrow())
        self._PyNumber_Multiply = PyNumber_Multiply.load(self.lib.borrow())
        self._PyNumber_TrueDivide = PyNumber_TrueDivide.load(self.lib.borrow())
        self._PyNumber_FloorDivide = PyNumber_FloorDivide.load(
            self.lib.borrow()
        )
        self._PyNumber_Remainder = PyNumber_Remainder.load(self.lib.borrow())
        self._PyNumber_Lshift = PyNumber_Lshift.load(self.lib.borrow())
        self._PyNumber_Rshift = PyNumber_Rshift.load(self.lib.borrow())
        self._PyNumber_And = PyNumber_And.load(self.lib.borrow())
        self._PyNumber_Or = PyNumber_Or.load(self.lib.borrow())
        self._PyNumber_Xor = PyNumber_Xor.load(self.lib.borrow())
        self._PyNumber_Power = PyNumber_Power.load(self.lib.borrow())
        self._PyNumber_Negative = PyNumber_Negative.load(self.lib.borrow())
        self._PyNumber_Positive = PyNumber_Positive.load(self.lib.borrow())
        self._PyNumber_Invert = PyNumber_Invert.load(self.lib.borrow())
        self._PyNumber_InPlaceAdd = PyNumber_InPlaceAdd.load(self.lib.borrow())
        self._PyNumber_InPlaceSubtract = PyNumber_InPlaceSubtract.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceMultiply = PyNumber_InPlaceMultiply.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceTrueDivide = PyNumber_InPlaceTrueDivide.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceFloorDivide = PyNumber_InPlaceFloorDivide.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceRemainder = PyNumber_InPlaceRemainder.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceLshift = PyNumber_InPlaceLshift.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceRshift = PyNumber_InPlaceRshift.load(
            self.lib.borrow()
        )
        self._PyNumber_InPlaceAnd = PyNumber_InPlaceAnd.load(self.lib.borrow())
        self._PyNumber_InPlaceOr = PyNumber_InPlaceOr.load(self.lib.borrow())
        self._PyNumber_InPlaceXor = PyNumber_InPlaceXor.load(self.lib.borrow())
        self._PyNumber_InPlacePower = PyNumber_InPlacePower.load(
            self.lib.borrow()
        )
        # Object Protocol (rich comparison)
        self._PyObject_RichCompare = PyObject_RichCompare.load(
            self.lib.borrow()
        )
        # Sequence Protocol
        self._PySequence_Contains = PySequence_Contains.load(self.lib.borrow())
        # Iterator Protocol
        self._PyIter_Check = PyIter_Check.load(self.lib.borrow())
        self._PyIter_Next = PyIter_Next.load(self.lib.borrow())
        # Concrete Objects Layer
        # Type Objects
        self._PyType_GetFlags = PyType_GetFlags.load(self.lib.borrow())
        self._PyType_IsSubtype = PyType_IsSubtype.load(self.lib.borrow())
        self._PyType_GenericAlloc = PyType_GenericAlloc.load(self.lib.borrow())
        if self.version.minor >= 11:
            self._PyType_GetName = PyType_GetName.load(self.lib.borrow())
        else:
            self._PyType_GetName = _PyType_GetName_dummy
        self._PyType_FromSpec = PyType_FromSpec.load(self.lib.borrow())
        # The None Object
        if self.version.minor >= 13:
            # Py_GetConstantBorrowed is part of the Stable ABI since version 3.13
            # References:
            # - https://docs.python.org/3/c-api/object.html#c.Py_GetConstantBorrowed
            # - https://docs.python.org/3/c-api/object.html#c.Py_CONSTANT_NONE

            # PyObject *Py_GetConstantBorrowed(unsigned int constant_id)
            self._Py_None = self.lib.call[
                "Py_GetConstantBorrowed", PyObjectPtr
            ](0)
        else:
            # PyObject *Py_None
            # TODO(MOCO-4435): remove this temporary variable.
            var none_ptr = _untracked_symbol[PyObject](
                self.lib, "_Py_NoneStruct"
            )
            self._Py_None = PyObjectPtr(upcast_from=none_ptr)
        # Integer Objects
        # PyTypeObject PyLong_Type
        self._PyLong_Type = _untracked_symbol[PyTypeObject](
            self.lib, "PyLong_Type"
        ).value()
        self._PyLong_FromSsize_t = PyLong_FromSsize_t.load(self.lib.borrow())
        self._PyLong_FromSize_t = PyLong_FromSize_t.load(self.lib.borrow())
        self._PyLong_AsSsize_t = PyLong_AsSsize_t.load(self.lib.borrow())
        self._PyLong_AsSize_t = PyLong_AsSize_t.load(self.lib.borrow())
        # Boolean Objects
        # PyTypeObject PyBool_Type
        self._PyBool_Type = _untracked_symbol[PyTypeObject](
            self.lib, "PyBool_Type"
        ).value()
        self._PyBool_FromLong = PyBool_FromLong.load(self.lib.borrow())
        # Floating-Point Objects
        # PyTypeObject PyFloat_Type
        self._PyFloat_Type = _untracked_symbol[PyTypeObject](
            self.lib, "PyFloat_Type"
        ).value()
        self._PyFloat_FromDouble = PyFloat_FromDouble.load(self.lib.borrow())
        self._PyFloat_AsDouble = PyFloat_AsDouble.load(self.lib.borrow())
        # Unicode Objects and Codecs
        self._PyUnicode_DecodeUTF8 = PyUnicode_DecodeUTF8.load(
            self.lib.borrow()
        )
        self._PyUnicode_AsUTF8AndSize = PyUnicode_AsUTF8AndSize.load(
            self.lib.borrow()
        )
        # Tuple Objects
        self._PyTuple_New = PyTuple_New.load(self.lib.borrow())
        self._PyTuple_GetItem = PyTuple_GetItem.load(self.lib.borrow())
        self._PyTuple_SetItem = PyTuple_SetItem.load(self.lib.borrow())
        # List Objects
        self._PyList_New = PyList_New.load(self.lib.borrow())
        self._PyList_GetItem = PyList_GetItem.load(self.lib.borrow())
        self._PyList_SetItem = PyList_SetItem.load(self.lib.borrow())
        # Dictionary Objects
        # PyTypeObject PyDict_Type
        self._PyDict_Type = _untracked_symbol[PyTypeObject](
            self.lib, "PyDict_Type"
        ).value()
        self._PyDict_New = PyDict_New.load(self.lib.borrow())
        self._PyDict_SetItem = PyDict_SetItem.load(self.lib.borrow())
        self._PyDict_GetItemWithError = PyDict_GetItemWithError.load(
            self.lib.borrow()
        )
        self._PyDict_Next = PyDict_Next.load(self.lib.borrow())
        # Set Objects
        self._PySet_New = PySet_New.load(self.lib.borrow())
        self._PySet_Add = PySet_Add.load(self.lib.borrow())
        # Module Objects
        self._PyModule_GetDict = PyModule_GetDict.load(self.lib.borrow())
        self._PyModule_Create2 = PyModule_Create2.load(self.lib.borrow())
        self._PyModule_AddFunctions = PyModule_AddFunctions.load(
            self.lib.borrow()
        )
        self._PyModule_AddObjectRef = PyModule_AddObjectRef.load(
            self.lib.borrow()
        )
        # Slice Objects
        self._PySlice_New = PySlice_New.load(self.lib.borrow())
        # Capsules
        self._PyCapsule_New = PyCapsule_New.load(self.lib.borrow())
        self._PyCapsule_GetPointer = PyCapsule_GetPointer.load(
            self.lib.borrow()
        )
        self._PyCapsule_IsValid = PyCapsule_IsValid.load(self.lib.borrow())
        # Memory Management
        self._PyObject_Free = PyObject_Free.load(self.lib.borrow())
        # Object Implementation Support
        # Common Object Structures
        self._Py_Is = Py_Is.load(self.lib.borrow())

    def __deinit__(deinit self):
        pass

    def destroy(mut self):
        # https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
        self.lib.call["Py_FinalizeEx"]()
        # Note: self.lib will be automatically closed when CPython is destroyed
        # due to OwnedDLHandle's RAII semantics

    def check_init_error(self) raises:
        """Used for entry points that initialize Python on first use, will
        raise an error if one occurred when initializing the global CPython.
        """
        if self.init_error:
            var mojo_python = getenv("MOJO_PYTHON")
            var python_lib = getenv("MOJO_PYTHON_LIBRARY")
            var python_exe = getenv("PYTHONEXECUTABLE")
            raise Error(
                self.init_error,
                "\nMOJO_PYTHON: " if mojo_python else "",
                mojo_python if mojo_python else "",
                "\nMOJO_PYTHON_LIBRARY: " if python_lib else "",
                python_lib if python_lib else "",
                "\npython executable: " if python_exe else "",
                python_exe if python_exe else "",
                "\n\nMojo/Python interop error, troubleshooting docs at:",
                "\n    https://modul.ar/fix-python\n",
            )

    def unsafe_get_error(self) -> Error:
        """Get the `Error` object corresponding to the current CPython
        interpreter error state.

        Safety:
            The caller MUST be sure that the CPython interpreter is in an error
            state before calling this function.

        This function will clear the CPython error.

        Returns:
            `Error` object describing the CPython error.
        """

        def err_occurred() {self} -> Bool:
            return self.PyErr_Occurred()

        debug_assert(
            err_occurred,
            "invalid unchecked conversion of Python error to Mojo error",
        )

        var err_ptr: PyObjectPtr
        # NOTE: PyErr_Fetch is deprecated since Python 3.12.
        var old_python = self.version.minor < 12
        if old_python:
            err_ptr = self.PyErr_Fetch()
        else:
            err_ptr = self.PyErr_GetRaisedException()
        assert Bool(err_ptr), "Python exception occurred but null was returned"

        var error: String
        try:
            error = String(py=PythonObject(from_owned=err_ptr))
        except e:
            abort(
                "internal error: Python exception occurred but cannot be"
                " converted to String"
            )

        if old_python:
            self.PyErr_Clear()
        return Error(error^)

    def get_error(self) -> Error:
        """Return an `Error` object from the CPython interpreter if it's in an
        error state, or an internal error if it's not.

        This should be used when you expect CPython to be in an error state,
        but want to fail gracefully if it's not.

        Returns:
            An `Error` object from the CPython interpreter if it's in an
            error state, or an internal error if it's not.
        """
        if self.PyErr_Occurred():
            return self.unsafe_get_error()
        return Error("internal error: expected CPython exception not found")

    def get_error_global(
        self,
        global_name: StringSlice,
    ) -> PyObjectPtr:
        """Get a Python read-only reference to the specified global exception
        object.
        """

        # Get pointer to the immortal `global_name` PyObject struct
        # instance.
        var maybe_ptr = self.lib.get_symbol[PyObjectPtr](global_name)

        if not maybe_ptr:
            abort(t"error: symbol `{global_name}` not found in CPython library")
        else:
            # SAFETY: maybe_ptr is checked above
            return maybe_ptr.unsafe_value()[]

    # ===-------------------------------------------------------------------===#
    # Python/C API
    # ref: https://docs.python.org/3/c-api/index.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # The Very High Level Layer
    # ref: https://docs.python.org/3/c-api/veryhigh.html
    # ===-------------------------------------------------------------------===#

    def PyRun_SimpleString(self, var command: String) -> c_int:
        """This is a simplified interface to `PyRun_SimpleStringFlags()` below,
        leaving the `PyCompilerFlags*` argument set to `NULL`.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_SimpleString
        """
        return self._PyRun_SimpleString(
            command.as_c_string_slice().ptr().as_unsafe_any_origin()
        )

    def PyRun_String(
        self,
        var str: String,
        start: c_int,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """Execute Python source code from `str` in the context specified by
        the objects `globals` and `locals`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_String
        """
        return self._PyRun_String(
            str.as_c_string_slice().ptr().as_unsafe_any_origin(),
            start,
            globals,
            locals,
        )

    def Py_CompileString(
        self,
        var str: String,
        var filename: String,
        start: c_int,
    ) -> PyObjectPtr:
        """Parse and compile the Python source code in `str`, returning the
        resulting code object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.Py_CompileString
        """
        return self._Py_CompileString(
            str.as_c_string_slice().ptr().as_unsafe_any_origin(),
            filename.as_c_string_slice().ptr().as_unsafe_any_origin(),
            start,
        )

    def PyEval_EvalCode(
        self,
        co: PyObjectPtr,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """Evaluate a precompiled code object, given a particular environment
        for its evaluation.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/veryhigh.html#c.PyEval_EvalCode
        """
        return self._PyEval_EvalCode(co, globals, locals)

    # ===-------------------------------------------------------------------===#
    # Reference Counting
    # ref: https://docs.python.org/3/c-api/refcounting.html
    # ===-------------------------------------------------------------------===#

    def Py_NewRef(self, o: PyObjectPtr) -> PyObjectPtr:
        """Create a new strong reference to an object: call `Py_INCREF()` on `o`
        and return the object `o`.

        The object `o` must not be `NULL`.

        References:
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_NewRef
        """
        assert Bool(o), "Py_NewRef called with NULL"
        return self._Py_NewRef(o)

    def Py_IncRef(self, ptr: PyObjectPtr):
        """Indicate taking a new strong reference to the object `ptr` points to.

        A function version of `Py_XINCREF()`, which is no-op if `ptr` is `NULL`.

        References:
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_IncRef
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_XINCREF
        """
        self._Py_IncRef(ptr)

    def Py_DecRef(self, ptr: PyObjectPtr):
        """Release a strong reference to the object `ptr` points to.

        A function version of `Py_XDECREF()`, which is no-op if `ptr` is `NULL`.

        References:
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_DecRef
        - https://docs.python.org/3/c-api/refcounting.html#c.Py_XDECREF
        """
        self._Py_DecRef(ptr)

    # This function assumes a specific way PyObjectPtr is implemented, namely
    # that the refcount has offset 0 in that structure. That generally doesn't
    # have to always be the case - but often it is and it's convenient for
    # debugging. We shouldn't rely on this function anywhere - its only purpose
    # is debugging.
    def _Py_REFCNT(self, ptr: PyObjectPtr) -> Py_ssize_t:
        if not ptr:
            return -1
        # NOTE:
        #   The "obvious" way to write this would be:
        #       return ptr._unsized_obj_ptr[].object_ref_count
        #   However, that is not valid, because, as the name suggest, a PyObject
        #   is an "unsized" or "incomplete" type, meaning that a pointer to an
        #   instance of that type doesn't point at the entire allocation of the
        #   underlying "concrete" object instance.
        #
        #   To avoid concerns about whether that's UB or not in Mojo, this
        #   this by just assumes the first field will be the ref count, and
        #   treats the object pointer "as if" it was a pointer to just the first
        #   field.
        # TODO(MSTDL-950): Should use something like `addr_of!`
        return ptr.bitcast[Py_ssize_t]().value()[]

    # ===-------------------------------------------------------------------===#
    # Exception Handling
    # ref: https://docs.python.org/3/c-api/exceptions.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # - Printing and clearing
    # ===-------------------------------------------------------------------===#

    def PyErr_Clear(self):
        """Clear the error indicator. If the error indicator is not set, there
        is no effect.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear
        """
        self._PyErr_Clear()

    # ===-------------------------------------------------------------------===#
    # - Raising exceptions
    # ===-------------------------------------------------------------------===#

    def PyErr_SetString(
        self,
        type: PyObjectPtr,
        message: OptionalPointer[c_char, ImmutAnyOrigin],
    ):
        """This is the most common way to set the error indicator. The first
        argument specifies the exception type; it is normally one of the
        standard exceptions, e.g. `PyExc_RuntimeError`. You need not create a
        new strong reference to it (e.g. with `Py_INCREF()`). The second
        argument is an error message; it is decoded from `'utf-8'`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString
        """
        self._PyErr_SetString(type, message)

    def PyErr_SetNone(self, type: PyObjectPtr):
        """This is a shorthand for `PyErr_SetObject(type, Py_None)`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetNone
        """
        self._PyErr_SetNone(type)

    # ===-------------------------------------------------------------------===#
    # - Querying the error indicator
    # ===-------------------------------------------------------------------===#

    # TODO: fix the return type
    def PyErr_Occurred(self) -> Bool:
        """Test whether the error indicator is set. If set, return the exception
        type (the first argument to the last call to one of the `PyErr_Set*`
        functions or to `PyErr_Restore()`). If not set, return `NULL`.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred
        """
        return Bool(self._PyErr_Occurred())

    def PyErr_GetRaisedException(self) -> PyObjectPtr:
        """Return the exception currently being raised, clearing the error
        indicator at the same time. Return `NULL` if the error indicator is not
        set.

        Return value: New reference. Part of the Stable ABI since version 3.12.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_GetRaisedException
        """
        return self._PyErr_GetRaisedException()

    # TODO: fix the signature to take the type, value, and traceback as args
    def PyErr_Fetch(self) -> PyObjectPtr:
        """Retrieve the error indicator into three variables whose addresses
        are passed.

        Deprecated since version 3.12.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Fetch
        """
        var type = PyObjectPtr()
        var value = PyObjectPtr()
        var traceback = PyObjectPtr()

        self._PyErr_Fetch(
            Pointer(to=type).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=value).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=traceback).unsafe_origin_cast[MutUntrackedOrigin](),
        )

        return value

    def PyErr_FetchTriple(
        self,
    ) -> Tuple[PyObjectPtr, PyObjectPtr, PyObjectPtr]:
        """Retrieve and clear the error indicator as a `(type, value,
        traceback)` triple of new references.

        Unlike `PyErr_Fetch`, this returns all three references so a caller can
        hand them straight back to `PyErr_Restore` without leaking. Works on
        every supported CPython version; the 3.12 deprecation of the underlying
        C function does not remove it.

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Fetch
        """
        var type = PyObjectPtr()
        var value = PyObjectPtr()
        var traceback = PyObjectPtr()

        self._PyErr_Fetch(
            Pointer(to=type).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=value).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=traceback).unsafe_origin_cast[MutUntrackedOrigin](),
        )

        return (type, value, traceback)

    def PyErr_Restore(
        self,
        type: PyObjectPtr,
        value: PyObjectPtr,
        traceback: PyObjectPtr,
    ):
        """Set the error indicator from a `(type, value, traceback)` triple,
        stealing a reference to each argument.

        Pairs with `PyErr_FetchTriple`: passing back exactly what was fetched
        round-trips the indicator (including the null-fields case, which clears
        it).

        References:
        - https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Restore
        """
        self._PyErr_Restore(type, value, traceback)

    # ===-------------------------------------------------------------------===#
    # Initialization, Finalization, and Threads
    # ref: https://docs.python.org/3/c-api/init.html
    # ===-------------------------------------------------------------------===#

    def PyEval_SaveThread(
        self,
    ) -> OptionalPointer[PyThreadState, MutUntrackedOrigin]:
        """Release the global interpreter lock (if it has been created) and
        reset the thread state to `NULL`, returning the previous thread state
        (which is not `NULL`).

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyEval_SaveThread
        """
        return self._PyEval_SaveThread()

    def PyEval_RestoreThread(
        self, state: OptionalPointer[PyThreadState, MutUntrackedOrigin]
    ):
        """Acquire the global interpreter lock (if it has been created) and
        set the thread state to tstate, which must not be `NULL`.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyEval_RestoreThread
        """
        self._PyEval_RestoreThread(state)

    def PyGILState_Ensure(self) -> PyGILState_STATE:
        """Ensure that the current thread is ready to call the Python C API
        regardless of the current state of Python, or of the global interpreter
        lock.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyGILState_Ensure
        """
        return self._PyGILState_Ensure()

    def PyGILState_Release(self, state: PyGILState_STATE):
        """Release any resources previously acquired.

        References:
        - https://docs.python.org/3/c-api/init.html#c.PyGILState_Release
        """
        self._PyGILState_Release(state)

    def PyGILState_Check(self) -> Bool:
        """Check whether the current thread holds the GIL.

        Returns:
            True if the current thread is holding the GIL, False otherwise.

        References:
        - <https://docs.python.org/3/c-api/init.html#c.PyGILState_Check>.
        """
        return self._PyGILState_Check() == 1

    # ===-------------------------------------------------------------------===#
    # Importing Modules
    # ref: https://docs.python.org/3/c-api/import.html
    # ===-------------------------------------------------------------------===#

    def PyImport_ImportModule(self, var name: String) -> PyObjectPtr:
        """This is a wrapper around `PyImport_Import()` which takes a `const char*`
        as an argument instead of a `PyObject*`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/import.html#c.PyImport_ImportModule
        """
        return self._PyImport_ImportModule(
            name.as_c_string_slice().ptr().as_unsafe_any_origin()
        )

    def PyImport_AddModule(self, var name: String) -> PyObjectPtr:
        """Return the module object corresponding to a module name.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/import.html#c.PyImport_AddModule
        """
        return self._PyImport_AddModule(
            name.as_c_string_slice().ptr().as_unsafe_any_origin()
        )

    # ===-------------------------------------------------------------------===#
    # Abstract Objects Layer
    # ref: https://docs.python.org/3/c-api/abstract.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Object Protocol
    # ref: https://docs.python.org/3/c-api/object.html
    # ===-------------------------------------------------------------------===#

    def PyObject_HasAttrString(
        self, obj: PyObjectPtr, var name: String
    ) -> c_int:
        """Returns `1` if `obj` has the attribute `name`, and `0` otherwise.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_HasAttrString
        """
        return self._PyObject_HasAttrString(
            obj, name.as_c_string_slice().ptr().as_unsafe_any_origin()
        )

    def PyObject_GetAttrString(
        self, obj: PyObjectPtr, var name: String
    ) -> PyObjectPtr:
        """Retrieve an attribute named `name` from object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetAttrString
        """
        return self._PyObject_GetAttrString(
            obj, name.as_c_string_slice().ptr().as_unsafe_any_origin()
        )

    def PyObject_SetAttrString(
        self, obj: PyObjectPtr, var name: String, value: PyObjectPtr
    ) -> c_int:
        """Set the value of the attribute named `name`, for object `obj`, to
        `value`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_SetAttrString
        """
        return self._PyObject_SetAttrString(
            obj,
            name.as_c_string_slice().ptr().as_unsafe_any_origin(),
            value,
        )

    def PyObject_Str(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Compute a string representation of object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Str
        """
        return self._PyObject_Str(obj)

    def PyObject_Hash(self, obj: PyObjectPtr) -> Py_hash_t:
        """Compute and return the hash value of an object `obj`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Hash
        """
        return self._PyObject_Hash(obj)

    def PyObject_IsTrue(self, obj: PyObjectPtr) -> c_int:
        """Returns `1` if the object `obj` is considered to be true, and `0`
        otherwise.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_IsTrue
        """
        return self._PyObject_IsTrue(obj)

    def PyObject_Type(self, obj: PyObjectPtr) -> PyObjectPtr:
        """When `obj` is non-`NULL`, returns a type object corresponding to the
        object type of object `obj`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Type
        """
        return self._PyObject_Type(obj)

    def PyObject_TypeCheck(
        self, obj: PyObjectPtr, type: PyTypeObjectPtr
    ) -> c_int:
        """Return non-zero if the object `obj` is of type `type` or a subtype of type,
        and 0 otherwise. Both parameters must be non-NULL.

        Note: this is a static inline function in the Python C API.
        https://github.com/python/cpython/blob/3dab11f888fda34c02734e4468d1acd4c36927fe/Include/object.h#L431

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_TypeCheck
        """
        var type_ptr = self.Py_TYPE(obj)
        return c_int(
            (type_ptr == type) or self._PyType_IsSubtype(type_ptr, type)
        )

    def PyObject_Length(self, obj: PyObjectPtr) -> Py_ssize_t:
        """Return the length of object `obj`.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_Length
        """
        return self._PyObject_Length(obj)

    def PyObject_GetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """Return element of `obj` corresponding to the object `key` or `NULL`
        on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem
        """
        return self._PyObject_GetItem(obj, key)

    def PyObject_SetItem(
        self, obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """Map the object `key` to `value`. Raise an exception and return `-1`
        on failure; return `0` on success.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_SetItem
        """
        return self._PyObject_SetItem(obj, key, value)

    def PyObject_GetIter(self, obj: PyObjectPtr) -> PyObjectPtr:
        """This is equivalent to the Python expression `iter(obj)`. It returns
        a new iterator for the object argument, or the object itself if the
        object is already an iterator.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_GetIter
        """
        return self._PyObject_GetIter(obj)

    # ===-------------------------------------------------------------------===#
    # Call Protocol
    # ref: https://docs.python.org/3/c-api/call.html
    # ===-------------------------------------------------------------------===#

    def PyObject_Call(
        self,
        callable: PyObjectPtr,
        args: PyObjectPtr,
        kwargs: PyObjectPtr,
    ) -> PyObjectPtr:
        """Call a callable Python object `callable`, with arguments given by the
        tuple `args`, and named arguments given by the dictionary `kwargs`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/call.html#c.PyObject_Call
        """
        return self._PyObject_Call(callable, args, kwargs)

    def PyObject_CallObject(
        self,
        callable: PyObjectPtr,
        args: PyObjectPtr,
    ) -> PyObjectPtr:
        """Call a callable Python object `callable`, with arguments given by the
        tuple `args`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/call.html#c.PyObject_CallObject
        """
        return self._PyObject_CallObject(callable, args)

    # ===-------------------------------------------------------------------===#
    # Number Protocol
    # ref: https://docs.python.org/3/c-api/number.html
    # ===-------------------------------------------------------------------===#

    def PyNumber_Long(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Returns the `obj` converted to an integer object on success,
        or `NULL` on failure. This is the equivalent of the Python expression
        `int(obj)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Long
        """
        return self._PyNumber_Long(obj)

    def PyNumber_Float(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Returns the `o` converted to a float object on success, or `NULL` on
        failure. This is the equivalent of the Python expression `float(obj)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Float
        """
        return self._PyNumber_Float(obj)

    def PyNumber_Add(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the result of adding `o1` and `o2`, or `NULL` on failure.
        This is the equivalent of the Python expression `o1 + o2`.

        Unlike a direct `__add__` lookup, this follows the full numeric
        protocol, including the reflected `__radd__` fallback.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Add
        """
        return self._PyNumber_Add(o1, o2)

    def PyNumber_Subtract(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of subtracting `o2` from `o1`, or `NULL` on
        failure. This is the equivalent of the Python expression `o1 - o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Subtract
        """
        return self._PyNumber_Subtract(o1, o2)

    def PyNumber_Multiply(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of multiplying `o1` and `o2`, or `NULL` on
        failure. This is the equivalent of the Python expression `o1 * o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Multiply
        """
        return self._PyNumber_Multiply(o1, o2)

    def PyNumber_TrueDivide(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of dividing `o1` by `o2`, or `NULL` on failure.
        This is the equivalent of the Python expression `o1 / o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_TrueDivide
        """
        return self._PyNumber_TrueDivide(o1, o2)

    def PyNumber_FloorDivide(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the floor of dividing `o1` by `o2`, or `NULL` on failure.
        This is the equivalent of the Python expression `o1 // o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_FloorDivide
        """
        return self._PyNumber_FloorDivide(o1, o2)

    def PyNumber_Remainder(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the remainder of dividing `o1` by `o2`, or `NULL` on failure.
        This is the equivalent of the Python expression `o1 % o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Remainder
        """
        return self._PyNumber_Remainder(o1, o2)

    def PyNumber_Lshift(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the result of left shifting `o1` by `o2`, or `NULL` on
        failure. This is the equivalent of the Python expression `o1 << o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Lshift
        """
        return self._PyNumber_Lshift(o1, o2)

    def PyNumber_Rshift(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the result of right shifting `o1` by `o2`, or `NULL` on
        failure. This is the equivalent of the Python expression `o1 >> o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Rshift
        """
        return self._PyNumber_Rshift(o1, o2)

    def PyNumber_And(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the bitwise AND of `o1` and `o2`, or `NULL` on failure. This
        is the equivalent of the Python expression `o1 & o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_And
        """
        return self._PyNumber_And(o1, o2)

    def PyNumber_Or(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the bitwise OR of `o1` and `o2`, or `NULL` on failure. This
        is the equivalent of the Python expression `o1 | o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Or
        """
        return self._PyNumber_Or(o1, o2)

    def PyNumber_Xor(self, o1: PyObjectPtr, o2: PyObjectPtr) -> PyObjectPtr:
        """Returns the bitwise XOR of `o1` and `o2`, or `NULL` on failure. This
        is the equivalent of the Python expression `o1 ^ o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Xor
        """
        return self._PyNumber_Xor(o1, o2)

    def PyNumber_Power(
        self, o1: PyObjectPtr, o2: PyObjectPtr, o3: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of raising `o1` to the power `o2`, modulo `o3`
        (pass `Py_None` for two-argument power), or `NULL` on failure. This is
        the equivalent of the Python expression `pow(o1, o2, o3)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Power
        """
        return self._PyNumber_Power(o1, o2, o3)

    def PyNumber_Negative(self, o: PyObjectPtr) -> PyObjectPtr:
        """Returns the negation of `o`, or `NULL` on failure. This is the
        equivalent of the Python expression `-o`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Negative
        """
        return self._PyNumber_Negative(o)

    def PyNumber_Positive(self, o: PyObjectPtr) -> PyObjectPtr:
        """Returns `o` with its sign unchanged, or `NULL` on failure. This is
        the equivalent of the Python expression `+o`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Positive
        """
        return self._PyNumber_Positive(o)

    def PyNumber_Invert(self, o: PyObjectPtr) -> PyObjectPtr:
        """Returns the bitwise negation of `o`, or `NULL` on failure. This is
        the equivalent of the Python expression `~o`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_Invert
        """
        return self._PyNumber_Invert(o)

    def PyNumber_InPlaceAdd(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of adding `o1` and `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 += o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceAdd
        """
        return self._PyNumber_InPlaceAdd(o1, o2)

    def PyNumber_InPlaceSubtract(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of subtracting `o2` from `o1`, done in place when
        `o1` supports it, or `NULL` on failure. This is the equivalent of the
        Python statement `o1 -= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceSubtract
        """
        return self._PyNumber_InPlaceSubtract(o1, o2)

    def PyNumber_InPlaceMultiply(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of multiplying `o1` and `o2`, done in place when
        `o1` supports it, or `NULL` on failure. This is the equivalent of the
        Python statement `o1 *= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceMultiply
        """
        return self._PyNumber_InPlaceMultiply(o1, o2)

    def PyNumber_InPlaceTrueDivide(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of dividing `o1` by `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 /= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceTrueDivide
        """
        return self._PyNumber_InPlaceTrueDivide(o1, o2)

    def PyNumber_InPlaceFloorDivide(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the floor of dividing `o1` by `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 //= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceFloorDivide
        """
        return self._PyNumber_InPlaceFloorDivide(o1, o2)

    def PyNumber_InPlaceRemainder(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the remainder of dividing `o1` by `o2`, done in place when
        `o1` supports it, or `NULL` on failure. This is the equivalent of the
        Python statement `o1 %= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceRemainder
        """
        return self._PyNumber_InPlaceRemainder(o1, o2)

    def PyNumber_InPlaceLshift(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of left shifting `o1` by `o2`, done in place when
        `o1` supports it, or `NULL` on failure. This is the equivalent of the
        Python statement `o1 <<= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceLshift
        """
        return self._PyNumber_InPlaceLshift(o1, o2)

    def PyNumber_InPlaceRshift(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of right shifting `o1` by `o2`, done in place when
        `o1` supports it, or `NULL` on failure. This is the equivalent of the
        Python statement `o1 >>= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceRshift
        """
        return self._PyNumber_InPlaceRshift(o1, o2)

    def PyNumber_InPlaceAnd(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the bitwise AND of `o1` and `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 &= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceAnd
        """
        return self._PyNumber_InPlaceAnd(o1, o2)

    def PyNumber_InPlaceOr(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the bitwise OR of `o1` and `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 |= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceOr
        """
        return self._PyNumber_InPlaceOr(o1, o2)

    def PyNumber_InPlaceXor(
        self, o1: PyObjectPtr, o2: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the bitwise XOR of `o1` and `o2`, done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 ^= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlaceXor
        """
        return self._PyNumber_InPlaceXor(o1, o2)

    def PyNumber_InPlacePower(
        self, o1: PyObjectPtr, o2: PyObjectPtr, o3: PyObjectPtr
    ) -> PyObjectPtr:
        """Returns the result of raising `o1` to the power `o2`, modulo `o3`
        (pass `Py_None` for two-argument power), done in place when `o1`
        supports it, or `NULL` on failure. This is the equivalent of the Python
        statement `o1 **= o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/number.html#c.PyNumber_InPlacePower
        """
        return self._PyNumber_InPlacePower(o1, o2, o3)

    def PyObject_RichCompare(
        self, o1: PyObjectPtr, o2: PyObjectPtr, opid: c_int
    ) -> PyObjectPtr:
        """Compares `o1` and `o2` using the operation specified by `opid` (one
        of `Py_LT`, `Py_LE`, `Py_EQ`, `Py_NE`, `Py_GT`, `Py_GE`). Returns the
        result of the comparison on success, or `NULL` on failure. This is the
        equivalent of the Python expression `o1 op o2`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/object.html#c.PyObject_RichCompare
        """
        return self._PyObject_RichCompare(o1, o2, opid)

    def PySequence_Contains(
        self, obj: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """Determines if `obj` contains `value`. Returns `1` if an item in `obj`
        is equal to `value`, `0` if not, and `-1` on error. This is the
        equivalent of the Python expression `value in obj`.

        References:
        - https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
        """
        return self._PySequence_Contains(obj, value)

    # ===-------------------------------------------------------------------===#
    # Iterator Protocol
    # ref: https://docs.python.org/3/c-api/iter.html
    # ===-------------------------------------------------------------------===#

    def PyIter_Check(self, obj: PyObjectPtr) -> c_int:
        """Return non-zero if the object `obj` can be safely passed to `PyIter_Next()`,
        and `0` otherwise.

        References:
        - https://docs.python.org/3/c-api/iter.html#c.PyIter_Check
        """
        return self._PyIter_Check(obj)

    def PyIter_Next(self, obj: PyObjectPtr) -> PyObjectPtr:
        """Return the next value from the iterator `obj`. The object must be an
        iterator according to `PyIter_Check()`. If there are no remaining values,
        returns `NULL` with no exception set. If an error occurs while retrieving
        the item, returns `NULL` and passes along the exception.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/iter.html#c.PyIter_Next
        """
        return self._PyIter_Next(obj)

    # ===-------------------------------------------------------------------===#
    # Concrete Objects Layer
    # ref: https://docs.python.org/3/c-api/concrete.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Type Objects
    # ref: https://docs.python.org/3/c-api/type.html
    # ===-------------------------------------------------------------------===#

    def PyType_GetFlags(
        self,
        type: PyTypeObjectPtr,
    ) -> c_ulong:
        """Return the `tp_flags` member of type.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_GetFlags
        """
        return self._PyType_GetFlags(type)

    def PyType_HasFeature(
        self, ptr: PyTypeObjectPtr, feature: c_ulong
    ) -> c_int:
        """Return non-zero if the type object ptr sets the feature feature. Type features are denoted by single bit flags.

        Note: this is another static helper function in the C API.

        References:
        - https://docs.python.org/3.13/c-api/type.html#c.PyType_HasFeature
        """
        return c_int(self._PyType_GetFlags(ptr) & feature)

    def PyType_IsSubtype(
        self,
        a: PyTypeObjectPtr,
        b: PyTypeObjectPtr,
    ) -> c_int:
        """Return true if *a* is a subtype of *b*.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_IsSubtype
        """
        return self._PyType_IsSubtype(a, b)

    def PyType_GenericAlloc(
        self,
        type: PyTypeObjectPtr,
        nitems: Py_ssize_t,
    ) -> PyObjectPtr:
        """Generic handler for the `tp_alloc` slot of a type object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_GenericAlloc
        """
        return self._PyType_GenericAlloc(type, nitems)

    def PyType_GetName(
        self, type: OptionalPointer[PyTypeObject, MutUntrackedOrigin]
    ) -> PyObjectPtr:
        """Return the type's name.

        Return value: New reference. Part of the Stable ABI since version 3.11.
        This function is patched to work with Python 3.10 and earlier versions.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_GetName
        """
        if self.version.minor < 11:
            return self.PyObject_GetAttrString(
                PyObjectPtr(upcast_from=type), "__name__"
            )
        return self._PyType_GetName(type)

    def PyType_FromSpec(
        self, spec: OptionalPointer[PyType_Spec, MutAnyOrigin]
    ) -> PyObjectPtr:
        """Equivalent to `PyType_FromMetaclass(NULL, NULL, spec, NULL)`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/type.html#c.PyType_FromSpec
        """
        return self._PyType_FromSpec(spec)

    # ===-------------------------------------------------------------------===#
    # The None Object
    # ref: https://docs.python.org/3/c-api/none.html
    # ===-------------------------------------------------------------------===#

    def Py_None(self) -> PyObjectPtr:
        """The Python `None` object, denoting lack of value.

        References:
        - https://docs.python.org/3/c-api/none.html#c.Py_None
        """
        return self._Py_None

    # ===-------------------------------------------------------------------===#
    # Integer Objects
    # ref: https://docs.python.org/3/c-api/long.html
    # ===-------------------------------------------------------------------===#

    def PyLong_Type(self) -> PyTypeObjectPtr:
        """The `PyLong_Type` Object.

        This instance of `PyTypeObject` represents the Python integer type. This is
        the same object as `int` in the Python layer.

        References:
        - https://docs.python.org/3.10/c-api/long.html#c.PyLong_Type
        """
        return self._PyLong_Type

    def PyLong_Check(self, obj: PyObjectPtr) -> c_int:
        """Return true if its argument is a `PyLongObject` or a subtype of
        `PyLongObject`. This function always succeeds.

        Note: this a C macro in the Python C API.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_Check
        - https://github.com/python/cpython/blob/main/Include/longobject.h
        """
        return self.PyType_HasFeature(
            self.Py_TYPE(obj), Py_TPFLAGS_LONG_SUBCLASS
        )

    def PyLong_CheckExact(self, obj: PyObjectPtr) -> c_int:
        """Return true if its argument is a `PyLongObject`, but not a subtype of
        `PyLongObject`. This function always succeeds.

        Note: this a C macro in the Python C API.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_CheckExact
        - https://github.com/python/cpython/blob/main/Include/longobject.h
        """
        return c_int(self.Py_TYPE(obj) == self._PyLong_Type)

    def PyLong_FromSsize_t(self, value: Py_ssize_t) -> PyObjectPtr:
        """Return a new `PyLongObject` object from a C `Py_ssize_t`, or `NULL`
        on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_FromSsize_t
        """
        return self._PyLong_FromSsize_t(value)

    def PyLong_FromSize_t(self, value: c_size_t) -> PyObjectPtr:
        """Return a new `PyLongObject` object from a C `size_t`, or `NULL` on
        failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_FromSize_t
        """
        return self._PyLong_FromSize_t(value)

    def PyLong_AsSsize_t(self, pylong: PyObjectPtr) -> Py_ssize_t:
        """Return a C `Py_ssize_t` representation of `pylong`.

        Raise `OverflowError` if the value of `pylong` is out of range for
        a `Py_ssize_t`.

        Returns `-1` on error. Use `PyErr_Occurred()` to disambiguate.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_AsSsize_t
        """
        return self._PyLong_AsSsize_t(pylong)

    def PyLong_AsSize_t(self, pylong: PyObjectPtr) -> c_size_t:
        """Return a C `size_t` representation of `pylong`.

        Raise `OverflowError` if the value of `pylong` is out of range for
        a `size_t`, including when `pylong` is negative.

        Returns `(size_t)-1` on error. Use `PyErr_Occurred()` to disambiguate.

        References:
        - https://docs.python.org/3/c-api/long.html#c.PyLong_AsSize_t
        """
        return self._PyLong_AsSize_t(pylong)

    # ===-------------------------------------------------------------------===#
    # Boolean Objects
    # ref: https://docs.python.org/3/c-api/bool.html
    # ===-------------------------------------------------------------------===#

    def PyBool_Type(self) -> PyTypeObjectPtr:
        """The `PyBool_Type` Object.

        This instance of `PyTypeObject` represents the Python boolean type; it
        is the same object as `bool` in the Python layer.

        References:
        - https://docs.python.org/3.10/c-api/bool.html#c.PyBool_Type
        """
        return self._PyBool_Type

    def PyBool_Check(self, obj: PyObjectPtr) -> c_int:
        """Return true if `obj` is of type `PyBool_Type`. This function always
        succeeds.

        Note: this a C macro in the Python C API.

        References:
        - https://docs.python.org/3.13/c-api/bool.html#c.PyBool_Check
        - https://github.com/python/cpython/blob/main/Include/boolobject.h
        """
        return c_int(self.Py_TYPE(obj) == self._PyBool_Type)

    def PyBool_FromLong(self, value: c_long) -> PyObjectPtr:
        """Return `Py_True` or `Py_False`, depending on the truth value
        of `value`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/bool.html#c.PyBool_FromLong
        """
        return self._PyBool_FromLong(value)

    # ===-------------------------------------------------------------------===#
    # Floating-Point Objects
    # ref: https://docs.python.org/3/c-api/float.html
    # ===-------------------------------------------------------------------===#

    def PyFloat_Type(self) -> PyTypeObjectPtr:
        """The `PyFloat_Type` Object.

        This instance of `PyTypeObject` represents the Python floating point
        type. This is the same object as `float` in the Python layer.

        References:
        - https://docs.python.org/3.10/c-api/float.html#c.PyFloat_Type
        """
        return self._PyFloat_Type

    def PyFloat_Check(self, obj: PyObjectPtr) -> c_int:
        """Return true if its argument is a `PyFloatObject` or a subtype of
        `PyFloatObject`. This function always succeeds.

        Note: this a C macro in the Python C API.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_Check
        - https://github.com/python/cpython/blob/main/Include/floatobject.h
        """
        return self.PyObject_TypeCheck(obj, self._PyFloat_Type)

    def PyFloat_CheckExact(self, obj: PyObjectPtr) -> c_int:
        """Return true if its argument is a `PyFloatObject`, but not a subtype of
        `PyFloatObject`. This function always succeeds.

        Note: this a C macro in the Python C API.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_CheckExact
        - https://github.com/python/cpython/blob/main/Include/floatobject.h
        """
        return c_int(self.Py_TYPE(obj) == self._PyFloat_Type)

    def PyFloat_FromDouble(self, value: c_double) -> PyObjectPtr:
        """Create a PyFloatObject object from `value`, or `NULL` on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble
        """
        return self._PyFloat_FromDouble(value)

    def PyFloat_AsDouble(self, pyfloat: PyObjectPtr) -> c_double:
        """Return a C double representation of the contents of `pyfloat`.

        This method returns `-1.0` upon failure, so one should call
        `PyErr_Occurred()` to check for errors.

        References:
        - https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble
        """
        return self._PyFloat_AsDouble(pyfloat)

    # ===-------------------------------------------------------------------===#
    # Unicode Objects and Codecs
    # ref: https://docs.python.org/3/c-api/unicode.html
    # ===-------------------------------------------------------------------===#

    # TODO: fix the signature to take str, size, and errors as args
    def PyUnicode_DecodeUTF8(self, s: StringSlice) -> PyObjectPtr:
        """Create a Unicode object by decoding size bytes of the UTF-8 encoded
        string slice `s`. Return `NULL` if an exception was raised by the codec.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_DecodeUTF8
        """
        return self._PyUnicode_DecodeUTF8(
            s.as_bytes()
            .unsafe_ptr()
            .unsafe_bitcast[c_char]()
            .as_imm()
            .as_unsafe_any_origin(),
            Py_ssize_t(s.byte_length()),
            "strict".as_c_string_slice().ptr().as_unsafe_any_origin(),
        )

    # TODO: fix signature to take unicode and size as args
    def PyUnicode_AsUTF8AndSize(
        self, obj: PyObjectPtr
    ) -> Optional[StringSlice[ImmutAnyOrigin]]:
        """Return a pointer to the UTF-8 encoding of the Unicode object, and
        store the size of the encoded representation (in bytes) in `size`.

        References:
        - https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_AsUTF8AndSize
        """
        var length = Py_ssize_t(0)
        var ptr = self._PyUnicode_AsUTF8AndSize(
            obj,
            Pointer(to=length).unsafe_origin_cast[MutUntrackedOrigin](),
        )
        if length == Py_ssize_t(-1):
            return None
        return StringSlice[ImmutAnyOrigin](
            unsafe_from_utf8=Span(
                unsafe_ptr=ptr.value().unsafe_bitcast[Byte](),
                length=Int(length),
            )
        )

    # ===-------------------------------------------------------------------===#
    # Tuple Objects
    # ref: https://docs.python.org/3/c-api/tuple.html
    # ===-------------------------------------------------------------------===#

    def PyTuple_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new tuple object of size `length`, or `NULL` with an
        exception set on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_New
        """
        return self._PyTuple_New(length)

    def PyTuple_GetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `pos` in the tuple `tuple`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_GetItem
        """
        return self._PyTuple_GetItem(tuple, pos)

    def PyTuple_SetItem(
        self,
        tuple: PyObjectPtr,
        pos: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Insert a reference to object `value` at position `pos` of the tuple
        `tuple`.

        This function "steals" a reference to `value` and discards a reference
        to an item already in the tuple at the affected position.

        References:
        - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
        """
        return self._PyTuple_SetItem(tuple, pos, value)

    # ===-------------------------------------------------------------------===#
    # List Objects
    # ref: https://docs.python.org/3/c-api/list.html
    # ===-------------------------------------------------------------------===#

    def PyList_New(self, length: Py_ssize_t) -> PyObjectPtr:
        """Return a new list of length `length` on success, or `NULL` on
        failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_New
        """
        return self._PyList_New(length)

    def PyList_GetItem(
        self,
        list: PyObjectPtr,
        index: Py_ssize_t,
    ) -> PyObjectPtr:
        """Return the object at position `index` in the list `list`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_GetItem
        """
        return self._PyList_GetItem(list, index)

    def PyList_SetItem(
        self,
        list: PyObjectPtr,
        index: Py_ssize_t,
        value: PyObjectPtr,
    ) -> c_int:
        """Set the item at index `index` in `list` to `value`.

        This function "steals" a reference to `value` and discards a reference
        to an item already in the list at the affected position.

        References:
        - https://docs.python.org/3/c-api/list.html#c.PyList_SetItem
        """
        return self._PyList_SetItem(list, index, value)

    # ===-------------------------------------------------------------------===#
    # Dictionary Objects
    # ref: https://docs.python.org/3/c-api/dict.html
    # ===-------------------------------------------------------------------===#

    def PyDict_Type(self) -> PyTypeObjectPtr:
        """This instance of `PyTypeObject` represents the Python dictionary type.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_Type
        """
        return self._PyDict_Type

    def PyDict_New(self) -> PyObjectPtr:
        """Return a new empty dictionary, or `NULL` on failure.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_New
        """
        return self._PyDict_New()

    def PyDict_SetItem(
        self,
        dict: PyObjectPtr,
        key: PyObjectPtr,
        value: PyObjectPtr,
    ) -> c_int:
        """Insert `value` into the dictionary `dict` with a key of `key`.

        This function *does not* steal a reference to `value`.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem
        """
        return self._PyDict_SetItem(dict, key, value)

    def PyDict_GetItemWithError(
        self,
        dict: PyObjectPtr,
        key: PyObjectPtr,
    ) -> PyObjectPtr:
        """Return the object from dictionary `dict` which has a key `key`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItemWithError
        """
        return self._PyDict_GetItemWithError(dict, key)

    def PyDict_Next(
        self,
        dict: PyObjectPtr,
        pos: OptionalPointer[Py_ssize_t, MutAnyOrigin],
        key: OptionalPointer[PyObjectPtr, MutAnyOrigin],
        value: OptionalPointer[PyObjectPtr, MutAnyOrigin],
    ) -> c_int:
        """Iterate over all key-value pairs in the dictionary `dict`.

        References:
        - https://docs.python.org/3/c-api/dict.html#c.PyDict_Next
        """
        return self._PyDict_Next(dict, pos, key, value)

    # ===-------------------------------------------------------------------===#
    # Set Objects
    # ref: https://docs.python.org/3/c-api/set.html
    # ===-------------------------------------------------------------------===#

    def PySet_New(self, iterable: PyObjectPtr) -> PyObjectPtr:
        """Return a new `set` containing objects returned by the `iterable`.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/set.html#c.PySet_New
        """
        return self._PySet_New(iterable)

    def PySet_Add(self, set: PyObjectPtr, key: PyObjectPtr) -> c_int:
        """Add `key` to a `set` instance.

        References:
        - https://docs.python.org/3/c-api/set.html#c.PySet_Add
        """
        return self._PySet_Add(set, key)

    # ===-------------------------------------------------------------------===#
    # Module Objects
    # ref: https://docs.python.org/3/c-api/module.html
    # ===-------------------------------------------------------------------===#

    def PyModule_GetDict(self, module: PyObjectPtr) -> PyObjectPtr:
        """Return the dictionary object that implements `module`'s namespace;
        this object is the same as the `__dict__` attribute of the module
        object.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_GetDict).
        """
        return self._PyModule_GetDict(module)

    def PyModule_Create(self, name: StaticString) -> PyObjectPtr:
        """Create a new module object.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_Create
        """

        # NOTE: See https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1326
        # for what we want to do here.
        var module_def_ptr = alloc(Layout[PyModuleDef].single()).unsafe_leak()
        module_def_ptr.unsafe_write(PyModuleDef(name))

        # TODO: set gil stuff
        # Note: Python automatically calls https://docs.python.org/3/c-api/module.html#c.PyState_AddModule
        # after the caller imports said module.

        # TODO: it would be nice to programmatically call a CPython API to get the value here
        # but I think it's only defined via the `PYTHON_API_VERSION` macro that ships with Python.
        # if this mismatches with the user's Python, then a `RuntimeWarning` is emitted according to the
        # docs.
        comptime module_api_version: c_int = 1013
        return self._PyModule_Create2(module_def_ptr, module_api_version)

    def PyModule_AddFunctions(
        self,
        module: PyObjectPtr,
        functions: OptionalPointer[PyMethodDef, MutAnyOrigin],
    ) -> c_int:
        """Add the functions from the `NULL` terminated `functions` array to
        module.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_AddFunctions
        """
        return self._PyModule_AddFunctions(module, functions)

    def PyModule_AddObjectRef(
        self,
        module: PyObjectPtr,
        name: OptionalPointer[c_char, ImmutAnyOrigin],
        value: PyObjectPtr,
    ) -> c_int:
        """Add an object to `module` as `name`.

        References:
        - https://docs.python.org/3/c-api/module.html#c.PyModule_AddObjectRef
        """
        return self._PyModule_AddObjectRef(module, name, value)

    # ===-------------------------------------------------------------------===#
    # Slice Objects
    # ref: https://docs.python.org/3/c-api/slice.html
    # ===-------------------------------------------------------------------===#

    def PySlice_New(
        self,
        start: PyObjectPtr,
        stop: PyObjectPtr,
        step: PyObjectPtr,
    ) -> PyObjectPtr:
        """Return a new slice object with the given values.

        Return value: New reference.

        References:
        - https://docs.python.org/3/c-api/slice.html#c.PySlice_New
        """
        return self._PySlice_New(start, stop, step)

    # ===-------------------------------------------------------------------===#
    # Capsules
    # ref: https://docs.python.org/3/c-api/capsule.html
    # ===-------------------------------------------------------------------===#

    def PyCapsule_New(
        self,
        pointer: OpaquePointer[MutUntrackedOrigin],
        name: StaticString,
        destructor: PyCapsule_Destructor,
    ) -> PyObjectPtr:
        """Create a `PyCapsule` encapsulating the pointer. The pointer argument
        may not be `NULL`.

        Return value: New reference.

        Note:
            `PyCapsule_New` stores the `name` pointer directly in the capsule
            rather than copying it, so the string must outlive the capsule.
            `name` is therefore a `StaticString` (a nul-terminated string
            literal has a `'static` lifetime); passing a temporary `String`
            would leave the capsule holding a dangling pointer. This is
            intentionally conservative: the C API only requires `name` to
            outlive the capsule, but `StaticString` is the simplest lifetime
            that satisfies that for the string-literal names used in practice.

        References:
        - https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_New
        """
        return self._PyCapsule_New(
            pointer,
            name.as_c_string_slice().ptr().as_unsafe_any_origin(),
            destructor,
        )

    def PyCapsule_GetPointer(
        self,
        capsule: PyObjectPtr,
        var name: String,
    ) raises -> OpaquePointer[MutUntrackedOrigin]:
        """Retrieve the pointer stored in the capsule. On failure, set an
        exception and return `NULL`.

        References:
        - https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_GetPointer
        """
        var r = self._PyCapsule_GetPointer(
            capsule,
            name.as_c_string_slice().ptr().as_unsafe_any_origin(),
        )
        if self.PyErr_Occurred():
            raise self.get_error()
        return r

    def PyCapsule_IsValid(
        self,
        capsule: PyObjectPtr,
        var name: String,
    ) -> Bool:
        """Return whether `capsule` is a valid capsule bearing the given name.

        Does not set an exception and is safe to call with an error already
        pending.

        References:
        - https://docs.python.org/3/c-api/capsule.html#c.PyCapsule_IsValid
        """
        return (
            self._PyCapsule_IsValid(
                capsule,
                name.as_c_string_slice().ptr().as_unsafe_any_origin(),
            )
            != 0
        )

    # ===-------------------------------------------------------------------===#
    # Memory Management
    # ref: https://docs.python.org/3/c-api/memory.html
    # ===-------------------------------------------------------------------===#

    def PyObject_Free(self, ptr: OptionalPointer[NoneType, MutUntrackedOrigin]):
        """Frees the memory block pointed to by `ptr`, which must have been
        returned by a previous call to `PyObject_Malloc()`, `PyObject_Realloc()`
        or PyObject_Calloc()`.

        References:
        - https://docs.python.org/3/c-api/memory.html#c.PyObject_Free
        """
        self._PyObject_Free(ptr)

    # ===-------------------------------------------------------------------===#
    # Object Implementation Support
    # ref: https://docs.python.org/3/c-api/objimpl.html
    # ===-------------------------------------------------------------------===#

    # ===-------------------------------------------------------------------===#
    # Common Object Structures
    # ref: https://docs.python.org/3/c-api/structures.html
    # ===-------------------------------------------------------------------===#

    def Py_Is(self, x: PyObjectPtr, y: PyObjectPtr) -> c_int:
        """Test if the `x` object is the `y` object, the same as `x is y` in
        Python.

        Part of the Stable ABI since version 3.10.

        References:
        - https://docs.python.org/3/c-api/structures.html#c.Py_Is
        """
        return self._Py_Is(x, y)

    def Py_TYPE(self, obj: PyObjectPtr) -> PyTypeObjectPtr:
        """Get the type of the Python object `obj`.

        Return value: Borrowed reference.

        References:
        - https://docs.python.org/3/c-api/structures.html#c.Py_TYPE
        - https://docs.python.org/3/c-api/typeobj.html#c.Py_TYPE
        """
        # Note:
        #   The `Py_TYPE` function is a `static` function in the C API, so
        #   we can't call it directly. Instead we reproduce its (trivial)
        #   behavior here.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return obj._unsized_obj_ptr.value()[].object_type
