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

from collections import InlineArray
from os import getenv, setenv, abort
from os.path import dirname
from pathlib import Path
from sys import external_call
from sys.arg import argv
from sys.ffi import DLHandle, c_char, c_int, c_uint, OpaquePointer
from stdlib.builtin.simd import Size_t, SSize_t
from python.python import _get_global_python_itf
from python._bindings import Typed_initproc

from memory import UnsafePointer

from utils import StringRef


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

alias Py_TPFLAGS_DEFAULT = 0

# TODO(MSTDL-892): Change this to alias ffi.C_ssize_t
alias Py_ssize_t = c_ssize_t

alias Py_size_t = c_size_t
# Ref: https://docs.python.org/3/c-api/structures.html#c.PyCFunction
# TODO(MOCO-1138):
#   This should be a C ABI function pointer, not a Mojo ABI function.
alias PyCFunction = fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr

alias METH_VARARGS = 0x1

alias destructor = fn (PyObjectPtr) -> None

alias initproc = fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int
alias newfunc = fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr


# GIL
@value
@register_passable("trivial")
struct PyGILState_STATE:
    var current_state: c_int

    alias PyGILState_LOCKED = c_int(0)
    alias PyGILState_UNLOCKED = c_int(1)


struct PyThreadState:
    """Opaque struct."""

    pass


@value
@register_passable("trivial")
struct PyKeyValuePair:
    var key: PyObjectPtr
    var value: PyObjectPtr
    var position: c_int
    var success: Bool


@value
@register_passable("trivial")
struct PyObjectPtr:
    var value: UnsafePointer[Int8]

    @always_inline
    fn __init__(inout self):
        self.value = UnsafePointer[Int8]()

    fn is_null(self) -> Bool:
        return int(self.value) == 0

    fn __eq__(self, rhs: PyObjectPtr) -> Bool:
        return int(self.value) == int(rhs.value)

    fn __ne__(self, rhs: PyObjectPtr) -> Bool:
        return not (self == rhs)

    # TODO: Consider removing this and inlining int(p.value) into callers
    fn _get_ptr_as_int(self) -> Int:
        return int(self.value)


@value
@register_passable
struct PythonVersion:
    var major: Int
    var minor: Int
    var patch: Int

    fn __init__(inout self, version: StringRef):
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
    var version_string = lib.get_function[fn () -> UnsafePointer[c_char]](
        "Py_GetVersion"
    )()
    return StringRef(version_string)


fn _py_finalize(lib: DLHandle):
    lib.get_function[fn () -> None]("Py_Finalize")()


# Ref https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
@value
struct PyMethodDef:
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var method_name: UnsafePointer[c_char]  # called ml_name in CPython

    # TODO(MSTDL-887): Support keyword-argument only methods
    # Pointer to the function to call
    var method_impl: PyCFunction

    # Flags bits indicating how the call should be constructed.
    # See https://docs.python.org/3/c-api/structures.html#c.PyMethodDef for the various calling conventions
    var method_flags: c_int

    # Points to the contents of the docstring for the module.
    var method_docstring: UnsafePointer[c_char]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self):
        """Constructs a zero initialized PyModuleDef.

        This is suitable for use terminating an array of PyMethodDef values.
        """
        self.method_name = UnsafePointer[c_char]()
        self.method_impl = _null_fn_ptr[PyCFunction]()
        self.method_flags = 0
        self.method_docstring = UnsafePointer[c_char]()

    fn __init__(inout self, *, other: Self):
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
    # TODO(MSTDL-877):
    #   Fill this out based on
    #   https://docs.python.org/3/c-api/typeobj.html#pytypeobject-definition
    pass


@value
@register_passable("trivial")
struct PyType_Spec:
    var name: UnsafePointer[c_char]
    var basicsize: c_int
    var itemsize: c_int
    var flags: c_uint
    var slots: UnsafePointer[PyType_Slot]


@value
@register_passable("trivial")
struct PyType_Slot:
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
    fn null() -> Self:
        return PyType_Slot {slot: 0, pfunc: OpaquePointer()}


@value
struct PyObject(Stringable, Representable, Formattable):
    var object_ref_count: Int
    # FIXME: should we use `PyObjectPtr`?  I don't think so!
    var object_type: UnsafePointer[PyTypeObject]
    # var object_type: PyObjectPtr

    fn __init__(inout self):
        self.object_ref_count = 0
        self.object_type = UnsafePointer[PyTypeObject]()

    @no_inline
    fn __str__(self) -> String:
        """Get the PyModuleDef_Base as a string.

        Returns:
            A string representation.
        """

        return String.format_sequence(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the `PyObject` as a string. Returns the same `String` as `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn format_to(self, inout writer: Formatter):
        """
        Formats to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write("PyObject(")
        writer.write("object_ref_count=", self.object_ref_count, ",")
        writer.write("object_type=", self.object_type)
        writer.write(")")


# Ref: https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L39
# Ref2: https://pyo3.rs/main/doc/pyo3/ffi/struct.pymoduledef_base
# Mojo doesn't have macros, so we define it here for ease.
# Note: `PyModuleDef_HEAD_INIT` defaults all of its members, see https://github.com/python/cpython/blob/833c58b81ebec84dc24ef0507f8c75fe723d9f66/Include/moduleobject.h#L60
struct PyModuleDef_Base(Stringable, Representable, Formattable):
    # The initial segment of every `PyObject` in CPython
    var object_base: PyObject

    # The function used to re-initialize the module.
    # TODO(MOCO-1138): This is a C ABI function pointer, not Mojo a function.
    alias _init_fn_type = fn () -> UnsafePointer[PyObject]
    var init_fn: Self._init_fn_type

    # The module's index into its interpreter's modules_by_index cache.
    var index: Py_ssize_t

    # A copy of the module's __dict__ after the first time it was loaded.
    var dict_copy: UnsafePointer[PyObject]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self):
        self.object_base = PyObject()
        self.init_fn = _null_fn_ptr[Self._init_fn_type]()
        self.index = 0
        self.dict_copy = UnsafePointer[PyObject]()

    fn __moveinit__(inout self, owned existing: Self):
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

        return String.format_sequence(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef_Base as a string. Returns the same `String` as `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn format_to(self, inout writer: Formatter):
        """
        Formats to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write("PyModuleDef_Base(")
        writer.write("object_base=", self.object_base, ",")
        writer.write("init_fn=<unprintable>", ",")
        writer.write("index=", self.index, ",")
        writer.write("dict_copy=", self.dict_copy)
        writer.write(")")


# Ref: https://docs.python.org/3/c-api/module.html#c.PyModuleDef_Slot
@value
struct PyModuleDef_Slot:
    var slot: c_int
    var value: OpaquePointer


# Ref: https://docs.python.org/3/c-api/module.html#c.PyModuleDef
struct PyModuleDef(Stringable, Representable, Formattable):
    # The Python module definition structs that holds all of the information needed
    # to create a module.  Typically, there is a 1:1 correspondence beteeen a `PyMethodDef`
    # and a module.
    var base: PyModuleDef_Base

    # See https://docs.python.org/3/c-api/structures.html#c.PyMethodDef
    var name: UnsafePointer[c_char]

    # Points to the contents of the docstring for the module.
    var docstring: UnsafePointer[c_char]

    var size: Py_ssize_t

    # A pointer to a table of module-level functions.  Can be null if there
    # are no functions present.
    var methods: UnsafePointer[PyMethodDef]

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

    fn __init__(inout self, name: String):
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

    fn __moveinit__(inout self, owned existing: Self):
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

        return String.format_sequence(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the PyMdouleDef as a string. Returns the same `String` as `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn format_to(self, inout writer: Formatter):
        """
        Formats to the provided formatter.

        Args:
            writer: The formatter to write to.
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


struct CPython:
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var lib: DLHandle
    var dict_type: PyObjectPtr
    var logging_enabled: Bool
    var version: PythonVersion
    var total_ref_count: UnsafePointer[Int]
    var init_error: StringRef

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self):
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
            "KGEN_CompilerRT_Python_SetPythonPath",
            UnsafePointer[c_char],
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
            self.lib.get_function[fn () -> None]("Py_Initialize")()
            self.version = PythonVersion(_py_get_version(self.lib))
            _ = self.Py_None()
            _ = self.PyDict_Type()
        else:
            self.version = PythonVersion(0, 0, 0)

    fn __del__(owned self):
        pass

    fn __copyinit__(inout self, existing: Self):
        self.lib = existing.lib
        self.dict_type = existing.dict_type
        self.logging_enabled = existing.logging_enabled
        self.version = existing.version
        self.total_ref_count = existing.total_ref_count
        self.init_error = existing.init_error

    @staticmethod
    fn destroy(inout existing: CPython):
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
            var error: String = self.init_error
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
    # None
    # ===-------------------------------------------------------------------===#

    fn Py_None(inout self) -> PyObjectPtr:
        """Get a None value, of type NoneType."""

        # Get pointer to the immortal `None` PyObject struct instance.
        # Note:
        #   The name of this global is technical a private part of the
        #   CPython API, but unfortunately the only stable ways to access it are
        #   macros.
        var ptr = self.lib.get_symbol[Int8](
            "_Py_NoneStruct",
        )

        if not ptr:
            abort("error: unable to get pointer to CPython `None` struct")

        return PyObjectPtr(ptr)

    # ===-------------------------------------------------------------------===#
    # Logging
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn log[*Ts: Formattable](self, *args: *Ts):
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
        fn print_arg[T: Formattable](arg: T):
            print(arg, sep="", end="", flush=False)

        args.each[print_arg]()

        print(flush=True)

    # ===-------------------------------------------------------------------===#
    # Reference count management
    # ===-------------------------------------------------------------------===#

    fn _inc_total_rc(inout self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v + 1)

    fn _dec_total_rc(inout self):
        var v = self.total_ref_count.take_pointee()
        self.total_ref_count.init_pointee_move(v - 1)

    fn Py_IncRef(inout self, ptr: PyObjectPtr):
        self.log(ptr._get_ptr_as_int(), " INCREF refcnt:", self._Py_REFCNT(ptr))

        self.lib.get_function[fn (PyObjectPtr) -> None]("Py_IncRef")(ptr)
        self._inc_total_rc()

    fn Py_DecRef(inout self, ptr: PyObjectPtr):
        self.log(ptr._get_ptr_as_int(), " DECREF refcnt:", self._Py_REFCNT(ptr))

        self.lib.get_function[fn (PyObjectPtr) -> None]("Py_DecRef")(ptr)
        self._dec_total_rc()

    # This function assumes a specific way PyObjectPtr is implemented, namely
    # that the refcount has offset 0 in that structure. That generally doesn't
    # have to always be the case - but often it is and it's convenient for
    # debugging. We shouldn't rely on this function anywhere - its only purpose
    # is debugging.
    fn _Py_REFCNT(inout self, ptr: PyObjectPtr) -> Int:
        if ptr._get_ptr_as_int() == 0:
            return -1
        return int(ptr.value.load())

    # ===-------------------------------------------------------------------===#
    # Python GIL and threading
    # ===-------------------------------------------------------------------===#

    fn PyGILState_Ensure(inout self) -> PyGILState_STATE:
        return self.lib.get_function[fn () -> PyGILState_STATE](
            "PyGILState_Ensure"
        )()

    fn PyGILState_Release(inout self, state: PyGILState_STATE):
        self.lib.get_function[fn (PyGILState_STATE) -> None](
            "PyGILState_Release"
        )(state)

    fn PyEval_SaveThread(inout self) -> UnsafePointer[PyThreadState]:
        return self.lib.get_function[fn () -> UnsafePointer[PyThreadState]](
            "PyEval_SaveThread"
        )()

    fn PyEval_RestoreThread(inout self, state: UnsafePointer[PyThreadState]):
        self.lib.get_function[fn (UnsafePointer[PyThreadState]) -> None](
            "PyEval_RestoreThread"
        )(state)

    # ===-------------------------------------------------------------------===#
    # Python Dict operations
    # ===-------------------------------------------------------------------===#

    fn PyDict_New(inout self) -> PyObjectPtr:
        var r = self.lib.get_function[fn () -> PyObjectPtr]("PyDict_New")()

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyDict_New, refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    # int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
    # ref: https://docs.python.org/3/c-api/dict.html#c.PyDict_SetItem
    fn PyDict_SetItem(
        inout self, dict_obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        var r = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int
        ](StringRef("PyDict_SetItem"))(dict_obj, key, value)

        self.log(
            "PyDict_SetItem, key: ",
            key._get_ptr_as_int(),
            " value: ",
            value._get_ptr_as_int(),
        )

        return r

    fn PyDict_GetItemWithError(
        inout self, dict_obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        var result = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
        ](StringRef("PyDict_GetItemWithError"))(dict_obj, key)

        self.log("PyDict_GetItemWithError, key: ", key._get_ptr_as_int())

        return result

    fn PyDict_Check(inout self, maybe_dict: PyObjectPtr) -> Bool:
        var my_type = self.PyObject_Type(maybe_dict)
        var my_type_as_int = my_type._get_ptr_as_int()
        var dict_type = self.PyDict_Type()
        var result = my_type_as_int == dict_type._get_ptr_as_int()
        self.Py_DecRef(my_type)
        return result

    fn PyDict_Type(inout self) -> PyObjectPtr:
        if self.dict_type.is_null():
            self.dict_type = self.lib.get_function[PyObjectPtr]("PyDict_Type")
        return self.dict_type

    # int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
    # ref: https://docs.python.org/3/c-api/dict.html#c.PyDict_Next
    fn PyDict_Next(
        inout self, dictionary: PyObjectPtr, p: Int
    ) -> PyKeyValuePair:
        var key = UnsafePointer[Int8]()
        var value = UnsafePointer[Int8]()
        var v = p
        var position = UnsafePointer[Int].address_of(v)
        var value_ptr = UnsafePointer[UnsafePointer[Int8]].address_of(value)
        var key_ptr = UnsafePointer[UnsafePointer[Int8]].address_of(key)
        var result = self.lib.get_function[
            fn (
                PyObjectPtr,
                UnsafePointer[Int],
                UnsafePointer[UnsafePointer[Int8]],
                UnsafePointer[UnsafePointer[Int8]],
            ) -> c_int
        ]("PyDict_Next")(
            dictionary,
            position,
            key_ptr,
            value_ptr,
        )

        self.log(
            dictionary._get_ptr_as_int(),
            " NEWREF PyDict_Next",
            dictionary._get_ptr_as_int(),
            "refcnt:",
            self._Py_REFCNT(dictionary),
            " key: ",
            PyObjectPtr {value: key}._get_ptr_as_int(),
            ", refcnt(key):",
            self._Py_REFCNT(key),
            "value:",
            PyObjectPtr {value: value}._get_ptr_as_int(),
            "refcnt(value)",
            self._Py_REFCNT(value),
        )

        _ = v
        return PyKeyValuePair {
            key: key,
            value: value,
            position: position.take_pointee(),
            success: result == 1,
        }

    # ===-------------------------------------------------------------------===#
    # Python Module operations
    # ===-------------------------------------------------------------------===#

    fn PyImport_ImportModule(
        inout self,
        name: StringRef,
    ) -> PyObjectPtr:
        var r = self.lib.get_function[fn (UnsafePointer[UInt8]) -> PyObjectPtr](
            "PyImport_ImportModule"
        )(name.data)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyImport_ImportModule, str:",
            name,
            ", refcnt:",
            self._Py_REFCNT(r),
        )

        self._inc_total_rc()
        return r

    fn PyImport_AddModule(inout self, name: StringRef) -> PyObjectPtr:
        var value = self.lib.get_function[
            fn (UnsafePointer[UInt8]) -> UnsafePointer[Int8]
        ]("PyImport_AddModule")(name.data)
        return PyObjectPtr {value: value}

    fn PyModule_Create(
        inout self,
        name: String,
    ) -> PyObjectPtr:
        # TODO: See https://docs.python.org/3/c-api/module.html#c.PyModule_Create
        # and https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1326
        # for what we want to do essentially here.
        var module_def_ptr = UnsafePointer[PyModuleDef].alloc(1)
        var module_def = PyModuleDef(name)
        module_def_ptr.init_pointee_move(module_def^)

        var create_module_fn = self.lib.get_function[
            fn (UnsafePointer[PyModuleDef], Int) -> PyObjectPtr
        ]("PyModule_Create2")

        # TODO: set gil stuff
        # Note: Python automatically calls https://docs.python.org/3/c-api/module.html#c.PyState_AddModule
        # after the caller imports said module.

        # TODO: it would be nice to programatically call a CPython API to get the value here
        # but I think it's only defined via the `PYTHON_API_VERSION` macro that ships with Python.
        # if this mismatches with the user's Python, then a `RuntimeWarning` is emitted according to the
        # docs.
        var module_api_version = 1013
        return create_module_fn(module_def_ptr, module_api_version)

    # int PyModule_AddFunctions(PyObject *module, PyMethodDef *functions)
    # ref: https://docs.python.org/3/c-api/module.html#c.PyModule_AddFunctions
    fn PyModule_AddFunctions(
        inout self,
        mod: PyObjectPtr,
        functions: UnsafePointer[PyMethodDef],
    ) -> c_int:
        var add_functions_fn = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[PyMethodDef]) -> c_int
        ]("PyModule_AddFunctions")

        return add_functions_fn(mod, functions)

    fn PyModule_AddObjectRef(
        inout self,
        module: PyObjectPtr,
        name: UnsafePointer[c_char],
        value: PyObjectPtr,
    ) -> c_int:
        var func = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[c_char], PyObjectPtr) -> c_int
        ]("PyModule_AddObjectRef")

        return func(module, name, value)

    fn PyModule_GetDict(inout self, name: PyObjectPtr) -> PyObjectPtr:
        var value = self.lib.get_function[fn (PyObjectPtr) -> PyObjectPtr](
            "PyModule_GetDict"
        )(name.value)
        return value

    # ===-------------------------------------------------------------------===#
    # Python Type operations
    # ===-------------------------------------------------------------------===#

    fn PyType_FromSpec(
        inout self, spec: UnsafePointer[PyType_Spec]
    ) -> PyObjectPtr:
        var func = self.lib.get_function[
            fn (UnsafePointer[PyType_Spec]) -> PyObjectPtr
        ]("PyType_FromSpec")

        return func(spec)

    # ===-------------------------------------------------------------------===#
    # Python Evaluation
    # ===-------------------------------------------------------------------===#

    # int PyRun_SimpleString(const char *command)
    # ref: https://docs.python.org/3/c-api/veryhigh.html#c.PyRun_SimpleString
    fn PyRun_SimpleString(inout self, strref: StringRef) -> Bool:
        """Executes the given Python code.

        Args:
            strref: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.
        """
        var status = self.lib.get_function[fn (UnsafePointer[UInt8]) -> c_int](
            StringRef("PyRun_SimpleString")
        )(strref.data)
        # PyRun_SimpleString returns 0 on success and -1 if an exception was
        # raised.
        return status == 0

    fn PyRun_String(
        inout self,
        strref: StringRef,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
        run_mode: Int,
    ) -> PyObjectPtr:
        var result = PyObjectPtr(
            self.lib.get_function[
                fn (
                    UnsafePointer[UInt8], Int32, PyObjectPtr, PyObjectPtr
                ) -> UnsafePointer[Int8]
            ]("PyRun_String")(strref.data, Int32(run_mode), globals, locals)
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
        inout self,
        co: PyObjectPtr,
        globals: PyObjectPtr,
        locals: PyObjectPtr,
    ) -> PyObjectPtr:
        var result = PyObjectPtr(
            self.lib.get_function[
                fn (
                    PyObjectPtr, PyObjectPtr, PyObjectPtr
                ) -> UnsafePointer[Int8]
            ]("PyEval_EvalCode")(co, globals, locals)
        )
        self._inc_total_rc()
        return result

    fn PyEval_GetBuiltins(inout self) -> PyObjectPtr:
        return self.lib.get_function[fn () -> PyObjectPtr](
            "PyEval_GetBuiltins"
        )()

    fn Py_CompileString(
        inout self,
        strref: StringRef,
        filename: StringRef,
        compile_mode: Int,
    ) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (
                UnsafePointer[UInt8], UnsafePointer[UInt8], Int32
            ) -> PyObjectPtr
        ]("Py_CompileString")(strref.data, filename.data, Int32(compile_mode))
        self._inc_total_rc()
        return r

    # ===-------------------------------------------------------------------===#
    # Python Object operations
    # ===-------------------------------------------------------------------===#

    # int Py_Is(PyObject *x, PyObject *y)
    # ref: https://docs.python.org/3/c-api/structures.html#c.Py_Is
    fn Py_Is(
        inout self,
        rhs: PyObjectPtr,
        lhs: PyObjectPtr,
    ) -> Bool:
        if self.version.minor >= 10:
            var r = self.lib.get_function[
                fn (PyObjectPtr, PyObjectPtr) -> c_int
            ]("Py_Is")(rhs, lhs)
            return r > 0
        else:
            return rhs == lhs

    fn PyObject_Type(inout self, obj: PyObjectPtr) -> PyObjectPtr:
        var f = self.lib.get_function[fn (PyObjectPtr) -> PyObjectPtr](
            "PyObject_Type"
        )
        self._inc_total_rc()
        return f(obj)

    fn PyObject_Str(inout self, obj: PyObjectPtr) -> PyObjectPtr:
        var f = self.lib.get_function[fn (PyObjectPtr) -> PyObjectPtr](
            "PyObject_Str"
        )
        self._inc_total_rc()
        return f(obj)

    fn PyObject_GetItem(
        inout self, obj: PyObjectPtr, key: PyObjectPtr
    ) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
        ]("PyObject_GetItem")(obj, key)

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
        inout self, obj: PyObjectPtr, key: PyObjectPtr, value: PyObjectPtr
    ) -> c_int:
        var r = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> c_int
        ]("PyObject_SetItem")(obj, key, value)

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

    fn PyObject_GetAttrString(
        inout self,
        obj: PyObjectPtr,
        name: StringRef,
    ) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[UInt8]) -> PyObjectPtr
        ]("PyObject_GetAttrString")(obj, name.data)

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

    # int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v)
    # ref: https://docs.python.org/3/c-api/object.html#c.PyObject_SetAttrString
    fn PyObject_SetAttrString(
        inout self, obj: PyObjectPtr, name: StringRef, new_value: PyObjectPtr
    ) -> c_int:
        var r = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[UInt8], PyObjectPtr) -> c_int
        ]("PyObject_SetAttrString")(obj, name.data, new_value)

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
        inout self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
    ) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr) -> PyObjectPtr
        ]("PyObject_CallObject")(callable_obj, args)

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
        inout self,
        callable_obj: PyObjectPtr,
        args: PyObjectPtr,
        kwargs: PyObjectPtr,
    ) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (PyObjectPtr, PyObjectPtr, PyObjectPtr) -> PyObjectPtr
        ]("PyObject_Call")(callable_obj, args, kwargs)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyObject_Call, refcnt:",
            self._Py_REFCNT(r),
            ", callable obj:",
            callable_obj._get_ptr_as_int(),
        )

        self._inc_total_rc()
        return r

    # int PyObject_IsTrue(PyObject *o)
    # ref: https://docs.python.org/3/c-api/object.html#c.PyObject_IsTrue
    fn PyObject_IsTrue(
        inout self,
        obj: PyObjectPtr,
    ) -> c_int:
        return self.lib.get_function[fn (PyObjectPtr) -> c_int](
            "PyObject_IsTrue"
        )(obj)

    fn PyObject_Length(
        inout self,
        obj: PyObjectPtr,
    ) -> Int:
        return int(
            self.lib.get_function[fn (PyObjectPtr) -> Int]("PyObject_Length")(
                obj
            )
        )

    fn PyObject_Hash(inout self, obj: PyObjectPtr) -> Int:
        return int(
            self.lib.get_function[fn (PyObjectPtr) -> Int]("PyObject_Hash")(obj)
        )

    fn PyObject_GetIter(
        inout self, traversablePyObject: PyObjectPtr
    ) -> PyObjectPtr:
        var iter = self.lib.get_function[fn (PyObjectPtr) -> PyObjectPtr](
            "PyObject_GetIter"
        )(traversablePyObject)

        self.log(
            iter._get_ptr_as_int(),
            " NEWREF PyObject_GetIter, refcnt:",
            self._Py_REFCNT(iter),
            "referencing ",
            traversablePyObject._get_ptr_as_int(),
            "refcnt of traversable: ",
            self._Py_REFCNT(traversablePyObject),
        )

        self._inc_total_rc()
        return iter

    # ===-------------------------------------------------------------------===#
    # Python Tuple operations
    # ===-------------------------------------------------------------------===#

    fn PyTuple_New(inout self, count: Int) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Int) -> PyObjectPtr](
            StringRef("PyTuple_New")
        )(count)

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
        inout self, tuple: PyObjectPtr, pos: Py_ssize_t
    ) -> PyObjectPtr:
        return self.lib.get_function[
            fn (PyObjectPtr, Py_ssize_t) -> PyObjectPtr
        ]("PyTuple_GetItem")(tuple, pos)

    # int PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o)
    # ref: https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
    fn PyTuple_SetItem(
        inout self,
        tuple_obj: PyObjectPtr,
        index: Int,
        element: PyObjectPtr,
    ) -> c_int:
        # PyTuple_SetItem steals the reference - the element object will be
        # destroyed along with the tuple
        self._dec_total_rc()
        return self.lib.get_function[
            fn (PyObjectPtr, Int, PyObjectPtr) -> c_int
        ](StringRef("PyTuple_SetItem"))(tuple_obj, index, element)

    # ===-------------------------------------------------------------------===#
    # Python List operations
    # ===-------------------------------------------------------------------===#

    fn PyList_New(inout self, length: Int) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Int) -> PyObjectPtr]("PyList_New")(
            length
        )

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
        inout self, list_obj: PyObjectPtr, index: Int, value: PyObjectPtr
    ) -> PyObjectPtr:
        # PyList_SetItem steals the reference - the element object will be
        # destroyed along with the list
        self._dec_total_rc()
        return self.lib.get_function[
            fn (PyObjectPtr, Int, PyObjectPtr) -> PyObjectPtr
        ]("PyList_SetItem")(list_obj, index, value)

    fn PyList_GetItem(
        inout self, list_obj: PyObjectPtr, index: Int
    ) -> PyObjectPtr:
        return self.lib.get_function[fn (PyObjectPtr, Int) -> PyObjectPtr](
            "PyList_GetItem"
        )(list_obj, index)

    # ===-------------------------------------------------------------------===#
    # Primitive type conversions
    # ===-------------------------------------------------------------------===#

    fn PyLong_FromLong(inout self, value: Int) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Int) -> PyObjectPtr](
            "PyLong_FromLong"
        )(value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromLong, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyBool_FromLong(inout self, value: Int) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Int) -> PyObjectPtr](
            "PyBool_FromLong"
        )(value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyBool_FromLong, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn toPython(inout self, litString: StringRef) -> PyObjectPtr:
        return self.PyString_FromStringAndSize(litString)

    fn toPython(inout self, litInt: Int) -> PyObjectPtr:
        return self.PyLong_FromLong(litInt.value)

    fn toPython(inout self, litBool: Bool) -> PyObjectPtr:
        return self.PyBool_FromLong(1 if litBool else 0)

    fn PyLong_AsLong(inout self, py_object: PyObjectPtr) -> Int:
        return self.lib.get_function[fn (PyObjectPtr) -> Int]("PyLong_AsLong")(
            py_object
        )

    fn PyFloat_AsDouble(inout self, py_object: PyObjectPtr) -> Float64:
        return self.lib.get_function[fn (PyObjectPtr) -> Float64](
            "PyFloat_AsDouble"
        )(py_object)

    fn PyFloat_FromDouble(inout self, value: Float64) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Float64) -> PyObjectPtr](
            "PyFloat_FromDouble"
        )(value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyFloat_FromDouble, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyFloat_FromDouble(inout self, value: Float32) -> PyObjectPtr:
        return self.PyFloat_FromDouble(value.cast[DType.float64]())


    fn PyLong_FromSsize_t(inout self, value: Py_ssize_t) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Py_ssize_t) -> PyObjectPtr]("PyLong_FromSsize_t")(value)
        self.log(
            r._get_ptr_as_int(),
            "NEWREF PyLong_FromSsize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
        value,
        )
        self._inc_total_rc()
        return r


    fn PyLong_FromSize_t(inout self, value: Size_t) -> PyObjectPtr:
        var r = self.lib.get_function[fn (Py_size_t) -> PyObjectPtr](
            "PyLong_FromSize_t"
        )(value)

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyLong_FromSize_t, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r


    fn PyBool_FromLong(inout self, value: Bool) -> PyObjectPtr:
        var long = 0
        if value:
            long = 1
        var r = self.lib.get_function[fn (Int8) -> PyObjectPtr](
            "PyBool_FromLong"
        )(Int8(long))

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyBool_FromLong, refcnt:",
            self._Py_REFCNT(r),
            ", value:",
            value,
        )

        self._inc_total_rc()
        return r

    fn PyString_FromStringAndSize(inout self, strref: StringRef) -> PyObjectPtr:
        var r = self.lib.get_function[
            fn (
                UnsafePointer[UInt8],
                Int,
                UnsafePointer[c_char],
            ) -> PyObjectPtr
        ](StringRef("PyUnicode_DecodeUTF8"))(
            strref.data, strref.length, "strict".unsafe_cstr_ptr()
        )

        self.log(
            r._get_ptr_as_int(),
            " NEWREF PyString_FromStringAndSize, refcnt:",
            self._Py_REFCNT(r),
            ", str:",
            strref,
        )

        self._inc_total_rc()
        return r

    fn PyUnicode_AsUTF8AndSize(inout self, py_object: PyObjectPtr) -> StringRef:
        var result = self.lib.get_function[
            fn (PyObjectPtr, UnsafePointer[Int]) -> UnsafePointer[c_char]
        ]("PyUnicode_AsUTF8AndSize")(py_object, UnsafePointer[Int]())
        return StringRef(result)

    # ===-------------------------------------------------------------------===#
    # Python Error operations
    # ===-------------------------------------------------------------------===#

    fn PyErr_Clear(inout self):
        self.lib.get_function[fn () -> None]("PyErr_Clear")()

    fn PyErr_Occurred(inout self) -> Bool:
        var value = self.lib.get_function[fn () -> PyObjectPtr](
            "PyErr_Occurred"
        )()
        return not value.is_null()

    fn PyErr_Fetch(inout self) -> PyObjectPtr:
        var type = UnsafePointer[Int8]()
        var value = UnsafePointer[Int8]()
        var traceback = UnsafePointer[Int8]()

        var type_ptr = UnsafePointer[UnsafePointer[Int8]].address_of(type)
        var value_ptr = UnsafePointer[UnsafePointer[Int8]].address_of(value)
        var traceback_ptr = UnsafePointer[UnsafePointer[Int8]].address_of(
            traceback
        )
        var func = self.lib.get_function[
            fn (
                UnsafePointer[UnsafePointer[Int8]],
                UnsafePointer[UnsafePointer[Int8]],
                UnsafePointer[UnsafePointer[Int8]],
            ) -> None
        ]("PyErr_Fetch")(type_ptr, value_ptr, traceback_ptr)
        var r = PyObjectPtr {value: value}

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

    fn PyErr_SetNone(
        inout self,
        type: PyObjectPtr,
    ):
        var func = self.lib.get_function[fn (PyObjectPtr) -> None](
            "PyErr_SetNone"
        )

        return func(type)

    fn PyErr_SetString(
        inout self,
        type: PyObjectPtr,
        message: UnsafePointer[c_char],
    ):
        self.lib.get_function[fn (PyObjectPtr, UnsafePointer[c_char]) -> None](
            "PyErr_SetString"
        )(type, message)

    # ===-------------------------------------------------------------------===#
    # Python Error types
    # ===-------------------------------------------------------------------===#

    fn get_error_global(
        inout self,
        global_name: StringLiteral,
    ) -> PyObjectPtr:
        """Get a Python borrowed reference to the specified global exception object.
        """

        # Get pointer to the immortal `global_name` PyObject struct
        # instance.
        var ptr: UnsafePointer[PyObjectPtr] = self.lib.get_symbol[PyObjectPtr](
            global_name
        )

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

    fn PyIter_Next(inout self, iterator: PyObjectPtr) -> PyObjectPtr:
        var next_obj = self.lib.get_function[fn (PyObjectPtr) -> PyObjectPtr](
            "PyIter_Next"
        )(iterator)

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

    # int PyIter_Check(PyObject *o)
    # ref: https://docs.python.org/3/c-api/iter.html#c.PyIter_Check
    fn PyIter_Check(inout self, obj: PyObjectPtr) -> Bool:
        var follows_iter_protocol = self.lib.get_function[
            fn (PyObjectPtr) -> c_int
        ]("PyIter_Check")(obj)
        return follows_iter_protocol != 0

    # int PySequence_Check(PyObject *o)
    # https://docs.python.org/3/c-api/sequence.html#c.PySequence_Check
    fn PySequence_Check(inout self, obj: PyObjectPtr) -> Bool:
        var follows_seq_protocol = self.lib.get_function[
            fn (PyObjectPtr) -> c_int
        ]("PySequence_Check")(obj)
        return follows_seq_protocol != 0
