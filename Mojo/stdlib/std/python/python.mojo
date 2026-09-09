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
"""Implements Python interoperability.

You can import these APIs from the `python` package. For example:

```mojo
from std.python import Python
```
"""

from std.os import abort
from std.ffi import _Global, c_size_t

from ._cpython import (
    CPython,
    Py_eval_input,
    Py_file_input,
    Py_ssize_t,
    PyMethodDef,
    PyObjectPtr,
)
from .python_object import PythonObject

comptime _PYTHON_GLOBAL = _Global[
    StorageType=_PythonGlobal, name="Python", init_fn=_PythonGlobal.__init__
]


struct _PythonGlobal(Defaultable, Movable):
    var cpython: CPython

    def __init__(out self):
        self.cpython = {}

    def __deinit__(deinit self):
        self.cpython.destroy()


def _get_python_interface() raises -> Pointer[CPython, ImmStaticOrigin]:
    """Returns an immutable static pointer to the CPython global.

    The returned pointer is immutable to prevent invalid shared mutation of
    this global variable. Once it is initialized, it may not be mutated.
    """

    var python = _PYTHON_GLOBAL.get_or_create_indexed_ptr(_Global._python_idx)
    var cpython_instance = (
        Pointer(to=python[].cpython)
        .as_imm()
        .unsafe_origin_cast[ImmStaticOrigin]()
    )
    return cpython_instance


struct Python(Defaultable, ImplicitlyCopyable):
    """Provides methods that help you use Python code in Mojo."""

    var _impl: ImmPointer[CPython, ImmStaticOrigin]
    """The underlying implementation of Mojo's Python interface."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Construct a new `Python` instance."""
        try:
            self._impl = _get_python_interface()
        except e:
            abort[prefix="ERROR:"](String(e))

    def __init__(out self, ref[ImmStaticOrigin] cpython: CPython):
        """Construct a `Python` instance from an existing reference
        to the lower-level singleton `CPython` instance.

        Args:
            cpython: Reference to the `CPython` singleton.
        """
        self._impl = ImmPointer[CPython, MutAnyOrigin](
            to=cpython
        ).unsafe_origin_cast[ImmStaticOrigin]()

    @always_inline
    def cpython(self) -> ref[ImmStaticOrigin] CPython:
        """Handle to the low-level C API of the CPython interpreter present in
        the current process.

        Returns:
            Handle to the CPython interpreter instance in the current process.
        """
        return self._impl[]

    def eval(self, var code: String) -> Bool:
        """Executes the given Python code.

        Args:
            code: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.
        """
        ref cpy = Self().cpython()
        # return 0 if the code executed successfully, -1 if it raised an exception.
        return cpy.PyRun_SimpleString(code^) == 0

    @staticmethod
    def evaluate(
        var expr: String,
        file: Bool = False,
        name: StaticString = "__main__",
    ) raises -> PythonObject:
        """Executes the given Python code.

        Args:
            expr: The Python expression to evaluate.
            file: Evaluate as a file and return the module.
            name: The name of the module (most relevant if `file` is True).

        Returns:
            `PythonObject` containing the result of the evaluation.

        Raises:
            If the operation fails.
        """
        ref cpy = Self().cpython()

        var mod = PythonObject(from_borrowed=cpy.PyImport_AddModule(name))
        var dict_ptr = cpy.PyModule_GetDict(mod._obj_ptr)
        if file:
            # We compile the code as provided and execute in the module
            # context. Note that this may be an existing module if the provided
            # module name is not unique. The name here is used only for this
            # code object, not the module itself.
            #
            # The Py_file_input is the code passed to the parsed to indicate
            # the initial state: this is essentially whether it is expecting
            # to compile an expression, a file or statements (e.g. repl).
            var code_ptr = cpy.Py_CompileString(
                expr^, "<evaluate>", Py_file_input
            )
            if not code_ptr:
                raise cpy.unsafe_get_error()
            # For this evaluation, we pass the dictionary both as the globals
            # and the locals. This is because the globals is defined as the
            # dictionary for the module scope, and locals is defined as the
            # dictionary for the *current* scope. Since we are executing at
            # the module scope for this eval, they should be the same object.
            var res_ptr = cpy.PyEval_EvalCode(code_ptr, dict_ptr, dict_ptr)
            cpy.Py_DecRef(code_ptr)
            if not res_ptr:
                raise cpy.unsafe_get_error()
            cpy.Py_DecRef(res_ptr)
            return mod
        else:
            # We use the result of evaluating the expression directly, and allow
            # all the globals/locals to be discarded. See above re: why the same
            # dictionary is being used here for both globals and locals.
            var res_ptr = cpy.PyRun_String(
                expr^, Py_eval_input, dict_ptr, dict_ptr
            )
            _ = mod^
            if not res_ptr:
                raise cpy.unsafe_get_error()
            return PythonObject(from_owned=res_ptr)

    @staticmethod
    def add_to_path(dir_path: StringSlice) raises:
        """Adds a directory to the Python path.

        This might be necessary to import a Python module via `import_module()`.
        For example:

        ```mojo
        from std.python import Python

        # Specify path to `mypython.py` module
        Python.add_to_path("path/to/module")
        var mypython = Python.import_module("mypython")

        var c = mypython.my_algorithm(2, 3)
        ```

        Args:
            dir_path: The path to a Python module you want to import.

        Raises:
            If the operation fails.
        """
        var sys = Python.import_module("sys")
        _ = sys.path.append(PythonObject(dir_path))

    # ===-------------------------------------------------------------------===#
    # PythonObject "Module" Operations
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def import_module(var module: String) raises -> PythonObject:
        """Imports a Python module.

        This provides you with a module object you can use just like you would
        in Python. For example:

        ```mojo
        from std.python import Python

        # This is equivalent to Python's `import numpy as np`
        np = Python.import_module("numpy")
        a = np.array(Python.list(1, 2, 3))
        ```

        Args:
            module: The Python module name. This module must be visible from the
                list of available Python paths (you might need to add the
                module's path with `add_to_path()`).

        Returns:
            The Python module.

        Raises:
            If the operation fails.
        """
        # Initialize the global interpreter and check for errors.
        ref cpy = Self().cpython()
        cpy.check_init_error()

        var mod_ptr = cpy.PyImport_ImportModule(module^)
        if not mod_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=mod_ptr)

    @staticmethod
    def create_module(name: StaticString) raises -> PythonObject:
        """Creates a Python module using the provided name.

        Inspired by https://github.com/pybind/pybind11/blob/a1d00916b26b187e583f3bce39cd59c3b0652c32/include/pybind11/pybind11.h#L1227

        TODO: allow specifying a doc-string to attach to the module upon creation or lazily added?

        Args:
            name: The Python module name.

        Returns:
            The Python module.

        Raises:
            If the operation fails.
        """
        # Initialize the global interpreter and check for errors.
        ref cpy = Self().cpython()
        cpy.check_init_error()

        var mod_ptr = cpy.PyModule_Create(name)
        if not mod_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=mod_ptr)

    @staticmethod
    def add_functions(
        module: PythonObject,
        var functions: List[PyMethodDef],
    ) raises:
        """Adds functions to a Python module object.

        Args:
            module: The Python module object.
            functions: List of function data.

        Raises:
            If we fail to add the functions to the module.
        """
        # Null terminate the functions list.
        functions.append({})
        # FIXME(MSTDL-910):
        #   This is an intentional memory leak, because we don't store this
        #   in a global variable (yet).
        return Self._unsafe_add_functions(
            module,
            functions.unsafe_take_allocation().unsafe_leak(),
        )

    @staticmethod
    def _unsafe_add_functions(
        module: PythonObject,
        functions: Pointer[PyMethodDef, MutUntrackedOrigin],
    ) raises:
        """Adds functions to a Python module object.

        Safety:
            The provided `functions` pointer must point to data that lives
            for the duration of the associated Python interpreter session.

        Args:
            module: The Python module object.
            functions: A null terminated pointer to function data.

        Raises:
            If we fail to add the functions to the module.
        """
        ref cpy = Self().cpython()
        var errno = cpy.PyModule_AddFunctions(
            # Safety: `module` pointer lives long enough because its reference
            #   argument.
            module._obj_ptr,
            functions.as_unsafe_any_origin(),
        )
        if errno == -1:
            raise cpy.unsafe_get_error()

    @staticmethod
    def add_object(
        module: PythonObject,
        var name: String,
        value: PythonObject,
    ) raises:
        """Add a new object to `module` with the given name and value.

        The provided object can be any type of Python object: an instance,
        a type object, a function, etc.

        The added value will be inserted into the `__dict__` of the provided
        module.

        Args:
            module: The Python module to modify.
            name: The name of the new object.
            value: The python object value.

        Raises:
            If the operation fails.
        """
        ref cpy = Self().cpython()
        var errno = cpy.PyModule_AddObjectRef(
            module._obj_ptr,
            name.as_c_string_slice().ptr().as_unsafe_any_origin(),
            value._obj_ptr,
        )
        if errno == -1:
            raise cpy.unsafe_get_error()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def dict(var **kwargs: PythonObject) raises -> PythonObject:
        """Construct an Python dictionary from keyword arguments.

        Args:
            kwargs: The keyword arguments to construct the dictionary with.

        Returns:
            The constructed Python dictionary.

        Raises:
            On failure to construct the dictionary or convert the values to
            Python objects.
        """
        ref cpy = Self().cpython()
        var dict_ptr = cpy.PyDict_New()
        for entry in kwargs.items():
            var key_ptr = cpy.PyUnicode_DecodeUTF8(StringSlice(entry.key))
            if not key_ptr:
                raise cpy.unsafe_get_error()
            # PyDict_SetItem doesn't steal the value.
            var errno = cpy.PyDict_SetItem(
                dict_ptr, key_ptr, entry.value._obj_ptr
            )
            cpy.Py_DecRef(key_ptr)
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=dict_ptr)

    @staticmethod
    def dict(
        tuples: Span[Tuple[PythonObject, PythonObject], _]
    ) raises -> PythonObject:
        """Construct an Python dictionary from a list of key-value tuples.

        Args:
            tuples: The list of key-value tuples to construct the dictionary
                with.

        Returns:
            The constructed Python dictionary.

        Raises:
            On failure to construct the dictionary or convert the keys or values
            to Python objects.
        """
        ref cpy = Self().cpython()
        var dict_ptr = cpy.PyDict_New()
        for i in range(len(tuples)):
            # PyDict_SetItem doesn't steal the values.
            var errno = cpy.PyDict_SetItem(
                dict_ptr, tuples[i][0]._obj_ptr, tuples[i][1]._obj_ptr
            )
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=dict_ptr)

    @staticmethod
    def list(values: Span[PythonObject, _]) raises -> PythonObject:
        """Initialize the object from a list of values.

        Args:
            values: The values to initialize the list with.

        Returns:
            A PythonObject representing the list.

        Raises:
            If the operation fails.
        """
        ref cpy = Self().cpython()
        var size = len(values)
        var list_ptr = cpy.PyList_New(size)
        for i in range(size):
            _ = cpy.PyList_SetItem(list_ptr, i, values[i].steal_data())
        return PythonObject(from_owned=list_ptr)

    @staticmethod
    def list(var *values: PythonObject) raises -> PythonObject:
        """Construct an Python list of objects.

        Args:
            values: The values to initialize the list with.

        Returns:
            The constructed Python list.

        Raises:
            If the operation fails.
        """
        ref cpy = Self().cpython()
        var list_ptr = cpy.PyList_New(len(values))

        for i in range(len(values)):
            _ = cpy.PyList_SetItem(list_ptr, i, values[i].steal_data())
        return PythonObject(from_owned=list_ptr)

    @staticmethod
    def tuple(var *values: PythonObject) raises -> PythonObject:
        """Construct an Python tuple of objects.

        Args:
            values: The values to initialize the tuple with.

        Returns:
            The constructed Python tuple.

        Raises:
            If the operation fails.
        """
        ref cpy = Self().cpython()
        var size = len(values)
        var tup_ptr = cpy.PyTuple_New(size)

        for i in range(size):
            _ = cpy.PyTuple_SetItem(tup_ptr, i, values[i].steal_data())
        return PythonObject(from_owned=tup_ptr)

    @no_inline
    def as_string_slice(self, obj: PythonObject) -> StringSlice[ImmutAnyOrigin]:
        """Return a string representing the given Python object.

        Args:
            obj: The Python object.

        Returns:
            Mojo string representing the given Python object.
        """
        ref cpy = self.cpython()
        return cpy.PyUnicode_AsUTF8AndSize(obj._obj_ptr).or_else("")

    @staticmethod
    def type(obj: PythonObject) -> PythonObject:
        """Return Type of this PythonObject.

        Args:
            obj: PythonObject we want the type of.

        Returns:
            A PythonObject that holds the type object.
        """
        ref cpy = Self().cpython()
        return PythonObject(from_owned=cpy.PyObject_Type(obj._obj_ptr))

    @staticmethod
    def none() -> PythonObject:
        """Get a `PythonObject` representing `None`.

        Returns:
            `PythonObject` representing `None`.
        """
        return PythonObject(None)

    @staticmethod
    def str(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `str`.

        Args:
            obj: The PythonObject to convert.

        Returns:
            A Python `str` object.

        Raises:
            An error if the conversion failed.
        """
        ref cpy = Self().cpython()
        var str_ptr = cpy.PyObject_Str(obj._obj_ptr)
        if not str_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=str_ptr)

    @staticmethod
    def int(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `int` (i.e. arbitrary precision
        integer).

        Args:
            obj: The PythonObject to convert.

        Returns:
            A PythonObject representing the result of the conversion to `int`.

        Raises:
            If the conversion to `int` fails.
        """
        ref cpy = Self().cpython()
        var int_ptr = cpy.PyNumber_Long(obj._obj_ptr)
        if not int_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=int_ptr)

    @staticmethod
    def float(obj: PythonObject) raises -> PythonObject:
        """Convert a PythonObject to a Python `float` object.

        Args:
            obj: The PythonObject to convert.

        Returns:
            A Python `float` object.

        Raises:
            If the conversion fails.
        """
        ref cpy = Self().cpython()
        var float_ptr = cpy.PyNumber_Float(obj._obj_ptr)
        if not float_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=float_ptr)

    # ===-------------------------------------------------------------------===#
    # Checked Conversions
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def py_long_as_ssize_t(obj: PythonObject) raises -> Py_ssize_t:
        """Get the value of a Python `long` object.

        Args:
            obj: The Python `long` object.

        Returns:
            The value of the `long` object as a `Py_ssize_t`.

        Raises:
            If `obj` is not a Python `long` object, or if the `long` object
            value overflows `Py_ssize_t`.
        """
        ref cpy = Self().cpython()
        var num = cpy.PyLong_AsSsize_t(obj._obj_ptr)
        if num == -1 and cpy.PyErr_Occurred():
            # Note that -1 does not guarantee an error, it just means we need to
            # check if there was an exception.
            raise cpy.unsafe_get_error()
        return num

    @staticmethod
    def py_long_as_size_t(obj: PythonObject) raises -> c_size_t:
        """Get the value of a Python `long` object as an unsigned `size_t`.

        Args:
            obj: The Python `long` object.

        Returns:
            The value of the `long` object as a `size_t`.

        Raises:
            If `obj` is not a Python `long` object, if the `long` object value
            is negative, or if it overflows `size_t`.
        """
        ref cpy = Self().cpython()
        var num = cpy.PyLong_AsSize_t(obj._obj_ptr)
        # `PyLong_AsSize_t` returns `(size_t)-1` on error (e.g. a negative
        # input, which raises `OverflowError`). Unlike the signed path, the
        # sentinel is the maximum `size_t`, so we must check `PyErr_Occurred`.
        if num == c_size_t.MAX and cpy.PyErr_Occurred():
            raise cpy.unsafe_get_error()
        return num

    @staticmethod
    def is_true(obj: PythonObject) raises -> Bool:
        """Check if the PythonObject is truthy.

        Args:
            obj: The PythonObject to check.

        Returns:
            True if the PythonObject is truthy and False otherwise.

        Raises:
            If the boolean value of the PythonObject cannot be determined.
        """
        # TODO: decide if this method should be actually exposed as public,
        # and add tests if so.
        ref cpy = Self().cpython()
        var res = cpy.PyObject_IsTrue(obj._obj_ptr)
        if res == -1:
            raise cpy.unsafe_get_error()
        return res == 1
