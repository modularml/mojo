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
"""
Mojo bindings functions and types from the CPython C API.

Documentation for these functions can be found online at:
  <https://docs.python.org/3/c-api/stable.html#contents-of-limited-api>
"""

from collections import InlineArray, Optional
from os import abort, getenv, setenv
from os.path import dirname
from pathlib import Path
from sys import external_call
from sys.arg import argv
from sys.ffi import (
    DLHandle,
    OpaquePointer,
    c_char,
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
)

from memory import UnsafePointer
from python._bindings import PyMojoObject, Pythonable, Typed_initproc
from python.python import _get_global_python_itf

from utils import StringRef, StringSlice

# ===-----------------------------------------------------------------------===#
# Raw Bindings
# ===-----------------------------------------------------------------------===#

# https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/compile.h#L7
alias Py_single_input = 256
alias Py_file_input = 257
alias Py_eval_input = 258
alias Py_func_type_input = 345

alias Py_tp_dealloc = 52
alias Py_tp_init = 60
alias Py_tp_methods = 64
alias Py_tp_new = 65
alias Py_tp_repr = 66

alias Py_TPFLAGS_DEFAULT = 0

alias Py_ssize_t = c_ssize_t

# TODO(MOCO-1138):
#   This should be a C ABI function pointer, not a Mojo ABI function.
alias PyCFunction = fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
"""[Reference](https://docs.python.org/3/c-api/structures.html#c.PyCFunction).
"""

alias METH_VARARGS = 0x1

alias destructor = fn (PyObjectPtr) -> None

alias reprfunc = fn (PyObjectPtr) -> PyObjectPtr

alias initproc = fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int
alias newfunc = fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr


# GIL
@value
@register_passable("trivial")
struct PyGILState_STATE:
    """Represents the state of the Python Global Interpreter Lock (GIL).

    Notes:
        This struct is used to store and manage the state of the GIL, which is
        crucial for thread-safe operations in Python. [Reference](
        https://github.com/python/cpython/blob/d45225bd66a8123e4a30314c627f2586293ba532/Include/pystate.h#L76
        ).
    """

    var current_state: c_int
    """The current state of the GIL."""

    alias PyGILState_LOCKED = c_int(0)
    alias PyGILState_UNLOCKED = c_int(1)


struct PyThreadState:
    """Opaque struct."""

    pass


@value
@register_passable("trivial")
struct PyKeysValuePair:
    """Represents a key-value pair in a Python dictionary iteration.

    This struct is used to store the result of iterating over a Python dictionary,
    containing the key, value, current position, and success status of the iteration.
    """

    var key: PyObjectPtr
    """The key of the current dictionary item."""
    var value: PyObjectPtr
    """The value of the current dictionary item."""
    var position: c_int
    """The current position in the dictionary iteration."""
    var success: Bool
    """Indicates whether the iteration was successful."""


@value
@register_passable("trivial")
struct PyObjectPtr:
    """Equivalent to `PyObject*` in C.

    It is crucial that this type has the same size and alignment as `PyObject*`
    for FFI ABI correctness.

    This struct provides methods for initialization, null checking,
    equality comparison, and conversion to integer representation.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var unsized_obj_ptr: UnsafePointer[PyObject]

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
    fn __init__(out self):
        """Initialize a null PyObjectPtr."""
        self.unsized_obj_ptr = UnsafePointer[PyObject]()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __eq__(self, rhs: PyObjectPtr) -> Bool:
        """Compare two PyObjectPtr for equality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are equal, False otherwise.
        """
        return int(self.unsized_obj_ptr) == int(rhs.unsized_obj_ptr)

    fn __ne__(self, rhs: PyObjectPtr) -> Bool:
        """Compare two PyObjectPtr for inequality.

        Args:
            rhs: The right-hand side PyObjectPtr to compare.

        Returns:
            Bool: True if the pointers are not equal, False otherwise.
        """
        return not (self == rhs)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn try_cast_to_mojo_value[
        T: AnyType,
    ](
        owned self,
        # TODO: Make this part of the trait bound
        expected_type_name: StringLiteral,
    ) -> Optional[UnsafePointer[T]]:
        var cpython = _get_global_python_itf().cpython()
        var type = cpython.Py_TYPE(self)
        var type_name = PythonObject(cpython.PyType_GetName(type))

        # FIXME(MSTDL-978):
        #   Improve this check. We should do something conceptually equivalent
        #   to:
        #       type == T.python_type_object
        #   where:
        #       trait Pythonable:
        #           var python_type_object: PyTypeObject
        if type_name == PythonObject(expected_type_name):
            return self.unchecked_cast_to_mojo_value[T]()
        else:
            return None

    fn unchecked_cast_to_mojo_object[
        T: AnyType
    ](owned self) -> UnsafePointer[PyMojoObject[T]]:
        """Assume that this Python object contains a wrapped Mojo value."""
        return self.unsized_obj_ptr.bitcast[PyMojoObject[T]]()

    fn unchecked_cast_to_mojo_value[T: AnyType](owned self) -> UnsafePointer[T]:
        var mojo_obj_ptr = self.unchecked_cast_to_mojo_object[T]()

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return UnsafePointer[T].address_of(mojo_obj_ptr[].mojo_value)

    fn is_null(self) -> Bool:
        """Check if the pointer is null.

        Returns:
            Bool: True if the pointer is null, False otherwise.
        """
        return int(self.unsized_obj_ptr) == 0

    # TODO: Consider removing this and inlining int(p.value) into callers
    fn _get_ptr_as_int(self) -> Int:
        """Get the pointer value as an integer.

        Returns:
            Int: The integer representation of the pointer.
        """
        return int(self.unsized_obj_ptr)


@value
@register_passable
struct PythonVersion:
    """Represents a Python version with major, minor, and patch numbers."""

    var major: Int
    """The major version number."""
    var minor: Int
    """The minor version number."""
    var patch: Int
    """The patch version number."""

    @implicit
    fn __init__(out self, version: StringRef):
        """Initialize a PythonVersion object from a version string.

        Args:
            version: A string representing the Python version (e.g., "3.9.5").

        The version string is parsed to extract major, minor, and patch numbers.
        If parsing fails for any component, it defaults to -1.
        """
        var version_string = String(version)
        var components = InlineArray[Int, 3](-1)
        var start = 0
        var next_idx = 0
        var i = 0
        while next_idx < len(version_string) and i < 3:
            if version_string[next_idx] == "." or (
                version_string[next_idx] == " " and i == 2
            ):
                var c = version_string[start:next_idx]
                try:
                    components[i] = atol(c)
                except:
                    components[i] = -1
                i += 1
                start = next_idx + 1
            next_idx += 1
        self = PythonVersion(components[0], components[1], components[2])


fn _py_get_version(lib: DLHandle) -> StringRef:
    return StringRef(ptr=lib.call["Py_GetVersion", UnsafePointer[c_char]]())


fn _py_finalize(lib: DLHandle):
    lib.call["Py_Finalize"]()


@value
struct PyMethodDef:
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

    var method_name: UnsafePointer[c_char]
    """A pointer to the name of the method as a C string.

    Notes:
        called `ml_name` in CPython.
    """

    # TODO(MSTDL-887): Support keyword-argument only methods
    var method_impl: PyCFunction
    """A function pointer to the implementation of the method."""

    var method_flags: c_int
    """Flags indicating how the method should be called. [Reference](
    https://docs.python.org/3/c-api/structures.html#c.PyMethodDef)."""

    var method_docstring: UnsafePointer[c_char]
    """A pointer to the docstring for the method as a C string."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Constructs a zero initialized PyModuleDef.

        This is suitable for use terminating an array of PyMethodDef values.
        """
        self.method_name = UnsafePointer[c_char]()
        self.method_impl = _null_fn_ptr[PyCFunction]()
        self.method_flags = 0
        self.method_docstring = UnsafePointer[c_char]()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @staticmethod
    fn function[
        func: fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr,
        func_name: StringLiteral,
        docstring: StringLiteral = "",
    ]() -> Self:
        """Create a PyMethodDef for a function.

        Parameters:
            func: The function to wrap.
            func_name: The name of the function.
            docstring: The docstring for the function.
        """
        # TODO(MSTDL-896):
        #   Support a way to get the name of the function from its parameter
        #   type, similar to `get_linkage_name()`?

        return PyMethodDef(
            func_name.unsafe_cstr_ptr(),
            func,
            METH_VARARGS,
            docstring.unsafe_cstr_ptr(),
        )


fn _null_fn_ptr[T: AnyTrivialRegType]() -> T:
    return __mlir_op.`pop.pointer.bitcast`[_type=T](
        __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`
    )


struct PyTypeObject:
    """The opaque C structure of the objects used to describe types.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyTypeObject).
    """

    # TODO(MSTDL-877):
    #   Fill this out based on
    #   https://docs.python.org/3/c-api/typeobj.html#pytypeobject-definition
    pass


@value
@register_passable("trivial")
struct PyType_Spec:
    """Structure defining a type's behavior.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyType_Spec).
    """

    var name: UnsafePointer[c_char]
    var basicsize: c_int
    var itemsize: c_int
    var flags: c_uint
    var slots: UnsafePointer[PyType_Slot]


@value
@register_passable("trivial")
struct PyType_Slot:
    """Structure defining optional functionality of a type, containing a slot ID
    and a value pointer.

    Notes:
        [Reference](https://docs.python.org/3/c-api/type.html#c.PyType_Slot).
    """

    var slot: c_int
    var pfunc: OpaquePointer

    @staticmethod
    fn tp_new(func: newfunc) -> Self:
        return PyType_Slot(Py_tp_new, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_init(func: Typed_initproc) -> Self:
        return PyType_Slot(Py_tp_init, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_dealloc(func: destructor) -> Self:
        return PyType_Slot(Py_tp_dealloc, rebind[OpaquePointer](func))

    @staticmethod
    fn tp_methods(methods: UnsafePointer[PyMethodDef]) -> Self:
        return PyType_Slot(Py_tp_methods, rebind[OpaquePointer](methods))

    @staticmethod
    fn tp_repr(func: reprfunc) -> Self:
        return PyType_Slot(Py_tp_repr, rebind[OpaquePointer](func))

    @staticmethod
    fn null() -> Self:
        return PyType_Slot {slot: 0, pfunc: OpaquePointer()}


@value
struct PyObject(Stringable, Representable, Writable):
    """All object types are extensions of this type. This is a type which
    contains the information Python needs to treat a pointer to an object as an
    object. In a normal “release” build, it contains only the object's reference
    count and a pointer to the corresponding type object. Nothing is actually
    declared to be a PyObject, but every pointer to a Python object can be cast
    to a PyObject.

    Notes:
        [Reference](https://docs.python.org/3/c-api/structures.html#c.PyObject).
    """

    var object_ref_count: Int
    var object_type: UnsafePointer[PyTypeObject]

    fn __init__(out self):
        self.object_ref_count = 0
        self.object_type = UnsafePointer[PyTypeObject]()

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDef_Base as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the `PyObject` as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("PyObject(")
        writer.write("object_ref_count=", self.object_ref_count, ",")
        writer.write("object_type=", self.object_type)
        writer.write(")")


# Mojo doesn't have macros, so we define it here for ease.
struct PyModuleDef_Base(Stringable, Representable, Writable):
    """PyModuleDef_Base.

    Notes:
        [Reference 1](
        https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L39
        ). [Reference 2](
        https://pyo3.rs/main/doc/pyo3/ffi/struct.pymoduledef_base
        ). `PyModuleDef_HEAD_INIT` defaults all of its members, [Reference 3](
        https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L60
        ).
    """

    var object_base: PyObject
    """The initial segment of every `PyObject` in CPython."""

    # TODO(MOCO-1138): This is a C ABI function pointer, not Mojo a function.
    alias _init_fn_type = fn () -> UnsafePointer[PyObject]
    """The function used to re-initialize the module."""
    var init_fn: Self._init_fn_type

    var index: Py_ssize_t
    """The module's index into its interpreter's modules_by_index cache."""

    var dict_copy: UnsafePointer[PyObject]
    """A copy of the module's __dict__ after the first time it was loaded."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(out self):
        self.object_base = PyObject()
        self.init_fn = _null_fn_ptr[Self._init_fn_type]()
        self.index = 0
        self.dict_copy = UnsafePointer[PyObject]()

    fn __moveinit__(out self, owned existing: Self):
        self.object_base = existing.object_base
        self.init_fn = existing.init_fn
        self.index = existing.index
        self.dict_copy = existing.dict_copy

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDef_Base as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef_Base as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("PyModuleDef_Base(")
        writer.write("object_base=", self.object_base, ",")
        writer.write("init_fn=<unprintable>", ",")
        writer.write("index=", self.index, ",")
        writer.write("dict_copy=", self.dict_copy)
        writer.write(")")


@value
struct PyModuleDef_Slot:
    """[Reference](
    https://docs.python.org/3/c-api/module.html#c.PyModuleDef_Slot).
    """

    var slot: c_int
    var value: OpaquePointer


struct PyModuleDef(Stringable, Representable, Writable):
    """The Python module definition structs that holds all of the information
    needed to create a module.

    Notes:
        [Reference](https://docs.python.org/3/c-api/module.html#c.PyModuleDef).
    """

    var base: PyModuleDef_Base

    var name: UnsafePointer[c_char]
    """[Reference](https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
    )."""

    var docstring: UnsafePointer[c_char]
    """Points to the contents of the docstring for the module."""

    var size: Py_ssize_t

    var methods: UnsafePointer[PyMethodDef]
    """A pointer to a table of module-level functions.  Can be null if there
    are no functions present."""

    var slots: UnsafePointer[PyModuleDef_Slot]

    # TODO(MOCO-1138): These are C ABI function pointers, not Mojo functions.
    alias _visitproc_fn_type = fn (PyObjectPtr, OpaquePointer) -> c_int
    alias _traverse_fn_type = fn (
        PyObjectPtr, Self._visitproc_fn_type, OpaquePointer
    ) -> c_int
    var traverse_fn: Self._traverse_fn_type

    alias _clear_fn_type = fn (PyObjectPtr) -> c_int
    var clear_fn: Self._clear_fn_type

    alias _free_fn_type = fn (OpaquePointer) -> OpaquePointer
    var free_fn: Self._free_fn_type

    @implicit
    fn __init__(out self, name: String):
        self.base = PyModuleDef_Base()
        self.name = name.unsafe_cstr_ptr()
        self.docstring = UnsafePointer[c_char]()
        # means that the module does not support sub-interpreters
        self.size = -1
        self.methods = UnsafePointer[PyMethodDef]()
        self.slots = UnsafePointer[PyModuleDef_Slot]()

        self.slots = UnsafePointer[PyModuleDef_Slot]()
        self.traverse_fn = _null_fn_ptr[Self._traverse_fn_type]()
        self.clear_fn = _null_fn_ptr[Self._clear_fn_type]()
        self.free_fn = _null_fn_ptr[Self._free_fn_type]()

    fn __moveinit__(out self, owned existing: Self):
        self.base = existing.base^
        self.name = existing.name
        self.docstring = existing.docstring
        self.size = existing.size
        self.methods = existing.methods
        self.slots = existing.slots
        self.traverse_fn = existing.traverse_fn
        self.clear_fn = existing.clear_fn
        self.free_fn = existing.free_fn

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDefe as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef as a string. Returns the same `String` as
        `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to[W: Writer](self, mut writer: W):
        """Formats to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

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


@value
struct CPython:
    """Handle to the CPython interpreter present in the current process."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var lib: DLHandle
    """The handle to the CPython shared library."""
    var dict_type: PyObjectPtr
    """The type object for Python dictionaries."""
    var logging_enabled: Bool
    """Whether logging is enabled."""
    var version: PythonVersion
    """The version of the Python runtime."""
    var total_ref_count: UnsafePointer[Int]
    """The total reference count of all Python objects."""
    var init_error: StringRef
    """An error message if initialization failed."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        var logging_enabled = getenv("MODULAR_CPYTHON_LOGGING") == "ON"
        if logging_enabled:
            print("CPython init")
            print("MOJO_PYTHON:", getenv("MOJO_PYTHON"))
            print("MOJO_PYTHON_LIBRARY:", getenv("MOJO_PYTHON_LIBRARY"))

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
                _ = setenv("PYTHONPATH", file_dir + ":" + python_path)
            else:
                _ = setenv("PYTHONPATH", file_dir)

        # TODO(MOCO-772) Allow raises to propagate through function pointers
        # and make this initialization a raising function.
        self.init_error = external_call[
            "KGEN_CompilerRT_Python_SetPythonPath", UnsafePointer[c_char]
        ]()

        var python_lib = getenv("MOJO_PYTHON_LIBRARY")

        if logging_enabled:
            print("PYTHONEXECUTABLE:", getenv("PYTHONEXECUTABLE"))
            print("libpython selected:", python_lib)

        self.lib = DLHandle(python_lib)
        self.total_ref_count = UnsafePointer[Int].alloc(1)
        self.dict_type = PyObjectPtr()
        self.logging_enabled = logging_enabled
        if not self.init_error:
            if not self.lib.check_symbol("Py_Initialize"):
                self.init_error = "compatible Python library not found"
            self.lib.call["Py_Initialize"]()
            self.version = PythonVersion(_py_get_version(self.lib))
        else:
            self.version = PythonVersion(0, 0, 0)

    fn __del__(owned self):
        pass

    @staticmethod
    fn destroy(mut existing: CPython):
        if existing.logging_enabled:
            print("CPython destroy")
            var remaining_refs = existing.total_ref_count.take_pointee()
            print("Number of remaining refs:", remaining_refs)
            # Technically not necessary since we're working with register
            # passable types, by it's good practice to re-initialize the
            # pointer after a consuming move.
            existing.total_ref_count.init_pointee_move(remaining_refs)
        _py_finalize(existing.lib)
        existing.lib.close()
        existing.total_ref_count.free()

    fn check_init_error(self) raises:
        """Used for entry points that initialize Python on first use, will
        raise an error if one occurred when initializing the global CPython.
        """
        if self.init_error:
            var error = String(self.init_error)
            var mojo_python = getenv("MOJO_PYTHON")
            var python_lib = getenv("MOJO_PYTHON_LIBRARY")
            var python_exe = getenv("PYTHONEXECUTABLE")
            if mojo_python:
                error += "\nMOJO_PYTHON: " + mojo_python
            if python_lib:
                error += "\nMOJO_PYTHON_LIBRARY: " + python_lib
            if python_exe:
                error += "\npython executable: " + python_exe
            error += "\n\nMojo/Python interop error, troubleshooting docs at:"
            error += "\n    https://modul.ar/fix-python\n"
            raise error

    # ===-------------------------------------------------------------------===#
    # Logging
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn log[*Ts: Writable](self, *args: *Ts):
        """If logging is enabled, print the given arguments as a log message.

        Parameters:
            Ts: The argument types.

        Arguments:
            args: The arguments to log.
        """
        if not self.logging_enabled:
            return

        # TODO(MOCO-358):
        #   Once Mojo argument splatting is supported, this should just
        #   be: `print(*args)`
        @parameter
        fn print_arg[T: Writable](arg: T):
            print(arg, sep="", end="", flush=False)

        args.each[print_arg]()

        print(flush=True)

    # ===-------------------------------------------------------------------===#
    # Reference count management
    # ===-------------------------------------------------------------------===#

    fn _inc_total_rc(mut self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v + 1)

    fn _dec_total_rc(mut self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v - 1)

    fn Py_IncRef(mut self, ptr: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/refcounting.html#c.Py_IncRef).
        """

        self.log(ptr._get_ptr_as_int(), " INCREF refcnt:", self._Py_REFCNT(ptr))

        self.lib.call["Py_IncRef"](ptr)
        self._inc_total_rc()

    fn Py_DecRef(mut self, ptr: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/refcounting.html#c.Py_DecRef).
        """

        self.log(ptr._get_ptr_as_int(), " DECREF refcnt:", self._Py_REFCNT(ptr))
        self.lib.call["Py_DecRef"](ptr)
        self._dec_total_rc()

    # This function assumes a specific way PyObjectPtr is implemented, namely
    # that the refcount has offset 0 in that structure. That generally doesn't
    # have to always be the case - but often it is and it's convenient for
    # debugging. We shouldn't rely on this function anywhere - its only purpose
    # is debugging.
    fn _Py_REFCNT(mut self, ptr: PyObjectPtr) -> Int:
        if ptr._get_ptr_as_int() == 0:
            return -1
        # NOTE:
        #   The "obvious" way to write this would be:
        #       return ptr.unsized_obj_ptr[].object_ref_count
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
        return ptr.unsized_obj_ptr.bitcast[Int]()[]

    # ===-------------------------------------------------------------------===#
    # Python GIL and threading
    # ===-------------------------------------------------------------------===#

    fn PyGILState_Ensure(mut self) -> PyGILState_STATE:
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyGILState_Ensure).
        """
        return self.lib.call["PyGILState_Ensure", PyGILState_STATE]()

    fn PyGILState_Release(mut self, state: PyGILState_STATE):
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyGILState_Release).
        """
        self.lib.call["PyGILState_Release"](state)

    fn PyEval_SaveThread(mut self) -> UnsafePointer[PyThreadState]:
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyEval_SaveThread).
        """

        return self.lib.call[
            "PyEval_SaveThread", UnsafePointer[PyThreadState]
        ]()

    fn PyEval_RestoreThread(mut self, state: UnsafePointer[PyThreadState]):
        """[Reference](
        https://docs.python.org/3/c-api/init.html#c.PyEval_RestoreThread).
        """
        self.lib.call["PyEval_RestoreThread"](state)

    # ===-------------------------------------------------------------------===#
    # Python Dict operations
    # ===-------------------------------------------------------------------===#

    fn PyDict_New(mut self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_New).
        """

        var r = self.lib.call["PyDict_New", PyObjectPtr]()

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyDict_New, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    # int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
    fn PyDict_SetItem(
        mut self, dict_obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem).
        """

        var r = self.lib.call["PyDict_SetItem", c_int](dict_obj, key, value)

        self.log(
            "PyDict_SetItem, key: ",
            key._get_ptr_as_int(),
            " value: ",
            value._get_ptr_as_int(),
        )

        return r

    fn PyDict_GetItemWithError(
        mut self, dict_obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_GetItemWithError).
        """

        var r = self.lib.call["PyDict_GetItemWithError", PyObjectPtr](
            dict_obj, key
        )
        self.log("PyDict_GetItemWithError, key: ", key._get_ptr_as_int())
        return r

    fn PyDict_Check(mut self, maybe_dict: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Check).
        """

        var my_type = self.PyObject_Type(maybe_dict)
        var my_type_as_int = my_type._get_ptr_as_int()
        var dict_type = self.PyDict_Type()
        var result = my_type_as_int == dict_type._get_ptr_as_int()
        self.Py_DecRef(my_type)
        return result

    fn PyDict_Type(mut self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Type).
        """
        if self.dict_type.is_null():
            self.dict_type = self.lib.call["PyDict_Type", PyObjectPtr]()
        return self.dict_type

    # int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
    fn PyDict_Next(
        mut self, dictionary: PyObjectPtr, p: Int
    ) -> PyKeysValuePair:
        """[Reference](
        https://docs.python.org/3/c-api/dict.html#c.PyDict_Next).
        """
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        var v = p
        var position = UnsafePointer[Int].address_of(v)
        var result = self.lib.call["PyDict_Next", c_int](
            dictionary,
            position,
            UnsafePointer.address_of(key),
            UnsafePointer.address_of(value),
        )

        self.log(
            dictionary._get_ptr_as_int(),
            " NEWREF PyDict_Next",
            dictionary._get_ptr_as_int(),
            "refcnt:",
            self._Py_REFCNT(dictionary),
            " key: ",
            key._get_ptr_as_int(),
            ", refcnt(key):",
            self._Py_REFCNT(key),
            "value:",
            value._get_ptr_as_int(),
            "refcnt(value)",
            self._Py_REFCNT(value),
        )

        _ = v
        return PyKeysValuePair {
            key: key,
            value: value,
            position: position.take_pointee(),
            success: result == 1,
        }

    # ===-------------------------------------------------------------------===#
    # Python Module operations
    # ===-------------------------------------------------------------------===#

    fn PyImport_ImportModule(
        mut self,
        name: StringRef,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/import.html#c.PyImport_ImportModule).
        """

        var r = self.lib.call["PyImport_ImportModule", PyObjectPtr](name.data)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyImport_ImportModule, str:",
            name,
            ", refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    fn PyImport_AddModule(mut self, name: StringRef) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/import.html#c.PyImport_AddModule).
        """
        return self.lib.call["PyImport_AddModule", PyObjectPtr](
            name.unsafe_ptr().bitcast[c_char]()
        )

    fn PyModule_Create(
        mut self,
        name: String,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_Create).
        """

        # TODO: See https://docs.python.org/3/c-api/module.html#c.PyModule_Create
        # and https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1326
        # for what we want to do essentially here.
        var module_def_ptr = UnsafePointer[PyModuleDef].alloc(1)
        var module_def = PyModuleDef(name)
        module_def_ptr.init_pointee_move(module_def^)

        # TODO: set gil stuff
        # Note: Python automatically calls https://docs.python.org/3/c-api/module.html#c.PyState_AddModule
        # after the caller imports said module.

        # TODO: it would be nice to programatically call a CPython API to get the value here
        # but I think it's only defined via the `PYTHON_API_VERSION` macro that ships with Python.
        # if this mismatches with the user's Python, then a `RuntimeWarning` is emitted according to the
        # docs.
        var module_api_version = 1013
        return self.lib.call["PyModule_Create2", PyObjectPtr](
            module_def_ptr, module_api_version
        )

    fn PyModule_AddFunctions(
        mut self,
        mod: PyObjectPtr,
        functions: UnsafePointer[PyMethodDef],
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_AddFunctions).
        """
        return self.lib.call["PyModule_AddFunctions", c_int](mod, functions)

    fn PyModule_AddObjectRef(
        mut self,
        module: PyObjectPtr,
        name: UnsafePointer[c_char],
        value: PyObjectPtr,
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_AddObjectRef).
        """

        return self.lib.call["PyModule_AddObjectRef", c_int](
            module, name, value
        )

    fn PyModule_GetDict(mut self, name: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/module.html#c.PyModule_GetDict).
        """
        return self.lib.call["PyModule_GetDict", PyObjectPtr](name)

    # ===-------------------------------------------------------------------===#
    # Python Type operations
    # ===-------------------------------------------------------------------===#

    fn Py_TYPE(mut self, ob_raw: PyObjectPtr) -> UnsafePointer[PyTypeObject]:
        """Get the PyTypeObject field of a Python object."""

        # Note:
        #   The `Py_TYPE` function is a `static` function in the C API, so
        #   we can't call it directly. Instead we reproduce its (trivial)
        #   behavior here.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return ob_raw.unsized_obj_ptr[].object_type

    fn PyType_GetName(
        mut self, type: UnsafePointer[PyTypeObject]
    ) -> PyObjectPtr:
        return self.lib.call["PyType_GetName", PyObjectPtr](type)

    fn PyType_FromSpec(
        mut self, spec: UnsafePointer[PyType_Spec]
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/type.html#c.PyType_FromSpec).
        """
        return self.lib.call["PyType_FromSpec", PyObjectPtr](spec)

    fn PyType_GenericAlloc(
        mut self,
        type: UnsafePointer[PyTypeObject],
        nitems: Py_ssize_t,
    ) -> PyObjectPtr:
        return self.lib.call["PyType_GenericAlloc", PyObjectPtr](type, nitems)

    # ===-------------------------------------------------------------------===#
    # Python Evaluation
    # ===-------------------------------------------------------------------===#

    fn PyRun_SimpleString(mut self, strref: StringRef) -> Bool:
        """Executes the given Python code.

        Args:
            strref: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.

        Notes:
            [Reference](
            https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_SimpleString).
        """
        return (
            self.lib.call["PyRun_SimpleString", c_int](strref.unsafe_ptr()) == 0
        )

    fn PyRun_String(
        mut self,
        strref: StringRef,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
        run_mode: Int,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_String).
        """
        var result = self.lib.call["PyRun_String", PyObjectPtr](
            strref.unsafe_ptr(), Int32(run_mode), globals, locals
        )

        self.log(
            result._get_ptr_as_int(),
            " NEWREF PyRun_String, str:",
            strref,
            ", ptr: ",
            result._get_ptr_as_int(),
            ", refcnt:",
            self._Py_REFCNT(result),
        )

        self._inc_total_rc()
        return result

    fn PyEval_EvalCode(
        mut self,
        co: PyObjectPtr,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.PyEval_EvalCode).
        """
        var result = self.lib.call["PyEval_EvalCode", PyObjectPtr](
            co, globals, locals
        )
        self._inc_total_rc()
        return result

    fn PyEval_GetBuiltins(mut self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/reflection.html#c.PyEval_GetBuiltins).
        """
        return self.lib.call["PyEval_GetBuiltins", PyObjectPtr]()

    fn Py_CompileString(
        mut self,
        strref: StringRef,
        filename: StringRef,
        compile_mode: Int,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/veryhigh.html#c.Py_CompileString).
        """

        var r = self.lib.call["Py_CompileString", PyObjectPtr](
            strref.unsafe_ptr(), filename.unsafe_ptr(), Int32(compile_mode)
        )
        self._inc_total_rc()
        return r

    # ===-------------------------------------------------------------------===#
    # Python Object operations
    # ===-------------------------------------------------------------------===#

    fn Py_Is(
        mut self,
        rhs: PyObjectPtr,
        lhs: PyObjectPtr,
    ) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/structures.html#c.Py_Is).
        """

        if self.version.minor >= 10:
            # int Py_Is(PyObject *x, PyObject *y)
            return self.lib.call["Py_Is", c_int](rhs, lhs) > 0
        else:
            return rhs == lhs

    fn PyObject_Type(mut self, obj: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Type).
        """

        var p = self.lib.call["PyObject_Type", PyObjectPtr](obj)
        self._inc_total_rc()
        return p

    fn PyObject_Str(mut self, obj: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Str).
        """

        var p = self.lib.call["PyObject_Str", PyObjectPtr](obj)
        self._inc_total_rc()
        return p

    fn PyObject_GetItem(
        mut self, obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_GetItem).
        """

        var r = self.lib.call["PyObject_GetItem", PyObjectPtr](obj, key)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_GetItem, key:",
            key._get_ptr_as_int(),
            ", refcnt:",
            self._Py_REFCNT(r),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_SetItem(
        mut self, obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_SetItem).
        """

        var r = self.lib.call["PyObject_SetItem", c_int](obj, key, value)

        self.log(
            "PyObject_SetItem result:",
            r,
            ", key:",
            key._get_ptr_as_int(),
            ", value:",
            value._get_ptr_as_int(),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )
        return r

    fn PyObject_HasAttrString(
        mut self,
        obj: PyObjectPtr,
        name: StringRef,
    ) -> Int:
        var r = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[UInt8]) -> Int
        ]("PyObject_HasAttrString")(obj, name.data)
        return r

    fn PyObject_GetAttrString(
        mut self,
        obj: PyObjectPtr,
        name: StringRef,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_GetAttrString).
        """

        var r = self.lib.call["PyObject_GetAttrString", PyObjectPtr](
            obj, name.data
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_GetAttrString, str:",
            name,
            ", refcnt:",
            self._Py_REFCNT(r),
            ", parent obj:",
            obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_SetAttrString(
        mut self, obj: PyObjectPtr, name: StringRef, new_value: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_SetAttrString).
        """

        var r = self.lib.call["PyObject_SetAttrString", c_int](
            obj, name.data, new_value
        )

        self.log(
            "PyObject_SetAttrString str:",
            name,
            ", parent obj:",
            obj._get_ptr_as_int(),
            ", new value:",
            new_value._get_ptr_as_int(),
            " new value ref count: ",
            self._Py_REFCNT(new_value),
        )

        return r

    fn PyObject_CallObject(
        mut self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/call.html#c.PyObject_CallObject).
        """

        var r = self.lib.call["PyObject_CallObject", PyObjectPtr](
            callable_obj, args
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_CallObject, refcnt:",
            self._Py_REFCNT(r),
            ", callable obj:",
            callable_obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_Call(
        mut self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
        kwargs: PyObjectPtr,
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/call.html#c.PyObject_Call).
        """

        var r = self.lib.call["PyObject_Call", PyObjectPtr](
            callable_obj, args, kwargs
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_Call, refcnt:",
            self._Py_REFCNT(r),
            ", callable obj:",
            callable_obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    fn PyObject_IsTrue(mut self, obj: PyObjectPtr) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_IsTrue).
        """
        return self.lib.call["PyObject_IsTrue", c_int](obj)

    fn PyObject_Length(mut self, obj: PyObjectPtr) -> Int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Length).
        """
        return int(self.lib.call["PyObject_Length", Int](obj))

    fn PyObject_Hash(mut self, obj: PyObjectPtr) -> Int:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_Hash).
        """
        return int(self.lib.call["PyObject_Hash", Int](obj))

    fn PyObject_GetIter(
        mut self, traversable_py_object: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/object.html#c.PyObject_GetIter).
        """
        var iterator = self.lib.call["PyObject_GetIter", PyObjectPtr](
            traversable_py_object
        )

        self.log(
            iterator._get_ptr_as_int(),
            " NEWREF PyObject_GetIter, refcnt:",
            self._Py_REFCNT(iterator),
            "referencing ",
            traversable_py_object._get_ptr_as_int(),
            "refcnt of traversable: ",
            self._Py_REFCNT(traversable_py_object),
        )

        self._inc_total_rc()
        return iterator

    # ===-------------------------------------------------------------------===#
    # Python Tuple operations
    # ===-------------------------------------------------------------------===#

    fn PyTuple_New(mut self, count: Int) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/tuple.html#c.PyTuple_New).
        """

        var r = self.lib.call["PyTuple_New", PyObjectPtr](count)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyTuple_New, refcnt:",
            self._Py_REFCNT(r),
            ", tuple size:",
            count,
        )

        self._inc_total_rc()
        return r

    fn PyTuple_GetItem(
        mut self, tuple: PyObjectPtr, pos: Py_ssize_t
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/tuple.html#c.PyTuple_GetItem).
        """
        return self.lib.call["PyTuple_GetItem", PyObjectPtr](tuple, pos)

    fn PyTuple_SetItem(
        mut self, tuple_obj: PyObjectPtr, index: Int, element: PyObjectPtr
    ) -> c_int:
        """[Reference](
        https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem).
        """

        # PyTuple_SetItem steals the reference - the element object will be
        # destroyed along with the tuple
        self._dec_total_rc()
        return self.lib.call["PyTuple_SetItem", c_int](
            tuple_obj, index, element
        )

    # ===-------------------------------------------------------------------===#
    # Python List operations
    # ===-------------------------------------------------------------------===#

    fn PyList_New(mut self, length: Int) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/list.html#c.PyList_New).
        """

        var r = self.lib.call["PyList_New", PyObjectPtr](length)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyList_New, refcnt:",
            self._Py_REFCNT(r),
            ", list size:",
            length,
        )

        self._inc_total_rc()
        return r

    fn PyList_SetItem(
        mut self, list_obj: PyObjectPtr, index: Int, value: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/list.html#c.PyList_SetItem).
        """

        # PyList_SetItem steals the reference - the element object will be
        # destroyed along with the list
        self._dec_total_rc()
        return self.lib.call["PyList_SetItem", PyObjectPtr](
            list_obj, index, value
        )

    fn PyList_GetItem(
        mut self, list_obj: PyObjectPtr, index: Int
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/list.html#c.PyList_GetItem).
        """
        return self.lib.call["PyList_GetItem", PyObjectPtr](list_obj, index)

    # ===-------------------------------------------------------------------===#
    # Concrete Objects
    # ref: https://docs.python.org/3/c-api/concrete.html
    # ===-------------------------------------------------------------------===#

    fn Py_None(mut self) -> PyObjectPtr:
        """Get a None value, of type NoneType. [Reference](
        https://docs.python.org/3/c-api/none.html#c.Py_None)."""

        # Get pointer to the immortal `None` PyObject struct instance.
        # Note:
        #   The name of this global is technical a private part of the
        #   CPython API, but unfortunately the only stable ways to access it are
        #   macros.
        # TODO(MSTDL-977):
        #   Investigate doing this without hard-coding private API details.
        var ptr = self.lib.get_symbol[PyObject]("_Py_NoneStruct")

        if not ptr:
            abort("error: unable to get pointer to CPython `None` struct")

        return PyObjectPtr(ptr)

    # ===-------------------------------------------------------------------===#
    # Boolean Objects
    # ===-------------------------------------------------------------------===#

    fn PyBool_FromLong(mut self, value: c_long) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/bool.html#c.PyBool_FromLong).
        """

        var r = self.lib.call["PyBool_FromLong", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyBool_FromLong, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    # ===-------------------------------------------------------------------===#
    # Integer Objects
    # ===-------------------------------------------------------------------===#

    fn PyLong_FromSsize_t(mut self, value: c_ssize_t) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_FromSsize_t).
        """

        var r = self.lib.call["PyLong_FromSsize_t", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromSsize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyLong_FromSize_t(mut self, value: c_size_t) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_FromSize_t).
        """

        var r = self.lib.call["PyLong_FromSize_t", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromSize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyLong_AsSsize_t(mut self, py_object: PyObjectPtr) -> c_ssize_t:
        """[Reference](
        https://docs.python.org/3/c-api/long.html#c.PyLong_AsSsize_t).
        """
        return self.lib.call["PyLong_AsSsize_t", c_ssize_t](py_object)

    # ===-------------------------------------------------------------------===#
    # Floating-Point Objects
    # ===-------------------------------------------------------------------===#

    fn PyFloat_FromDouble(mut self, value: Float64) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/float.html#c.PyFloat_FromDouble).
        """

        var r = self.lib.call["PyFloat_FromDouble", PyObjectPtr](value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyFloat_FromDouble, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyFloat_AsDouble(mut self, py_object: PyObjectPtr) -> Float64:
        """[Reference](
        https://docs.python.org/3/c-api/float.html#c.PyFloat_AsDouble).
        """
        return self.lib.call["PyFloat_AsDouble", Float64](py_object)

    # ===-------------------------------------------------------------------===#
    # Unicode Objects
    # ===-------------------------------------------------------------------===#

    fn PyUnicode_DecodeUTF8(mut self, strref: StringRef) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_DecodeUTF8).
        """

        var r = self.lib.call["PyUnicode_DecodeUTF8", PyObjectPtr](
            strref.unsafe_ptr().bitcast[Int8](),
            strref.length,
            "strict".unsafe_cstr_ptr(),
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyUnicode_DecodeUTF8, refcnt:",
            self._Py_REFCNT(r),
            ", str:",
            strref,
        )

        self._inc_total_rc()
        return r

    fn PyUnicode_DecodeUTF8(mut self, strslice: StringSlice) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_DecodeUTF8).
        """
        var r = self.lib.call["PyUnicode_DecodeUTF8", PyObjectPtr](
            strslice.unsafe_ptr().bitcast[Int8](),
            strslice.byte_length(),
            "strict".unsafe_cstr_ptr(),
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyUnicode_DecodeUTF8, refcnt:",
            self._Py_REFCNT(r),
            ", str:",
            strslice,
        )

        self._inc_total_rc()
        return r

    fn PySlice_FromSlice(mut self, slice: Slice) -> PyObjectPtr:
        # Convert Mojo Slice to Python slice parameters
        # Note: Deliberately avoid using `span.indices()` here and instead pass
        # the Slice parameters directly to Python. Python's C implementation
        # already handles such conditions, allowing Python to apply its own slice
        # handling.
        var py_start = self.Py_None()
        var py_stop = self.Py_None()
        var py_step = self.Py_None()

        if slice.start:
            py_start = self.PyLong_FromSsize_t(c_ssize_t(slice.start.value()))
        if slice.end:
            py_stop = self.PyLong_FromSsize_t(c_ssize_t(slice.end.value()))
        if slice.end:
            py_step = self.PyLong_FromSsize_t(c_ssize_t(slice.step.value()))

        var py_slice = self.PySlice_New(py_start, py_stop, py_step)

        if py_start != self.Py_None():
            self.Py_DecRef(py_start)
        if py_stop != self.Py_None():
            self.Py_DecRef(py_stop)
        self.Py_DecRef(py_step)

        return py_slice

    fn PyUnicode_AsUTF8AndSize(mut self, py_object: PyObjectPtr) -> StringRef:
        """[Reference](
        https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_AsUTF8AndSize).
        """

        var s = StringRef()
        s.data = self.lib.call[
            "PyUnicode_AsUTF8AndSize", UnsafePointer[c_char]
        ](py_object, UnsafePointer.address_of(s.length)).bitcast[UInt8]()
        return s

    # ===-------------------------------------------------------------------===#
    # Python Error operations
    # ===-------------------------------------------------------------------===#

    fn PyErr_Clear(mut self):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear).
        """
        self.lib.call["PyErr_Clear"]()

    fn PyErr_Occurred(mut self) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Occurred).
        """
        return not self.lib.call["PyErr_Occurred", PyObjectPtr]().is_null()

    fn PyErr_Fetch(mut self) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Fetch).
        """
        var type = PyObjectPtr()
        var value = PyObjectPtr()
        var traceback = PyObjectPtr()

        self.lib.call["PyErr_Fetch"](
            UnsafePointer.address_of(type),
            UnsafePointer.address_of(value),
            UnsafePointer.address_of(traceback),
        )
        var r = value

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyErr_Fetch, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        _ = type
        _ = value
        _ = traceback
        return r

    fn PyErr_SetNone(mut self, type: PyObjectPtr):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetNone).
        """
        self.lib.call["PyErr_SetNone"](type)

    fn PyErr_SetString(
        mut self,
        type: PyObjectPtr,
        message: UnsafePointer[c_char],
    ):
        """[Reference](
        https://docs.python.org/3/c-api/exceptions.html#c.PyErr_SetString).
        """
        self.lib.call["PyErr_SetString"](type, message)

    # ===-------------------------------------------------------------------===#
    # Python Error types
    # ===-------------------------------------------------------------------===#

    fn get_error_global(
        mut self,
        global_name: StringLiteral,
    ) -> PyObjectPtr:
        """Get a Python read-only reference to the specified global exception
        object.
        """

        # Get pointer to the immortal `global_name` PyObject struct
        # instance.
        var ptr = self.lib.get_symbol[PyObjectPtr](global_name)

        if not ptr:
            abort(
                "error: unable to get pointer to CPython `"
                + global_name
                + "` global"
            )

        return ptr[]

    # ===-------------------------------------------------------------------===#
    # Python Iterator operations
    # ===-------------------------------------------------------------------===#

    fn PyIter_Next(mut self, iterator: PyObjectPtr) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/iter.html#c.PyIter_Next).
        """

        var next_obj = self.lib.call["PyIter_Next", PyObjectPtr](iterator)

        self.log(
            next_obj._get_ptr_as_int(),
            " NEWREF PyIter_Next from ",
            iterator._get_ptr_as_int(),
            ", refcnt(obj):",
            self._Py_REFCNT(next_obj),
            "refcnt(iter)",
            self._Py_REFCNT(iterator),
        )

        if next_obj._get_ptr_as_int() != 0:
            self._inc_total_rc()
        return next_obj

    fn PyIter_Check(mut self, obj: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/iter.html#c.PyIter_Check).
        """
        return self.lib.call["PyIter_Check", c_int](obj) != 0

    fn PySequence_Check(mut self, obj: PyObjectPtr) -> Bool:
        """[Reference](
        https://docs.python.org/3/c-api/sequence.html#c.PySequence_Check).
        """
        return self.lib.call["PySequence_Check", c_int](obj) != 0

    # ===-------------------------------------------------------------------===#
    # Python Slice Creation
    # ===-------------------------------------------------------------------===#

    fn PySlice_New(
        mut self, start: PyObjectPtr, stop: PyObjectPtr, step: PyObjectPtr
    ) -> PyObjectPtr:
        """[Reference](
        https://docs.python.org/3/c-api/slice.html#c.PySlice_New).
        """
        var r = self.lib.call["PySlice_New", PyObjectPtr](start, stop, step)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PySlice_New, refcnt:",
            self._Py_REFCNT(r),
            ", start:",
            start._get_ptr_as_int(),
            ", stop:",
            stop._get_ptr_as_int(),
            ", step:",
            step._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r
