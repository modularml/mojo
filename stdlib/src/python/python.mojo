# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements Python interoperability.

You can import these APIs from the `python` package. For example:

```mojo
from python import Python
```
"""

from sys import external_call
from sys.info import sizeof

from memory.unsafe import Pointer

from ._cpython import CPython, Py_eval_input
from .dictionary import Dictionary
from .object import PythonObject

from sys.ffi import _get_global


fn _init_global(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    let ptr = Pointer[CPython].alloc(1)
    __get_address_as_lvalue(ptr.address) = CPython()
    return ptr.bitcast[NoneType]()


fn _destroy_global(python: Pointer[NoneType]):
    let p = python.bitcast[CPython]()
    CPython.destroy(__get_address_as_lvalue(p.address))
    python.free()


@always_inline
fn _get_global_python_itf() -> _PythonInterfaceImpl:
    let ptr = _get_global["Python", _init_global, _destroy_global]()
    return ptr.bitcast[CPython]()


struct _PythonInterfaceImpl:
    var _cpython: Pointer[CPython]

    fn __init__(inout self, cpython: Pointer[CPython]):
        self._cpython = cpython

    fn __copyinit__(inout self, existing: Self):
        self._cpython = existing._cpython

    fn cpython(self) -> CPython:
        return __get_address_as_lvalue(self._cpython.address)


struct Python:
    """Provides methods that help you use Python code in Mojo."""

    var impl: _PythonInterfaceImpl
    """The underlying implementation of Mojo's Python interface."""

    fn __init__(inout self):
        """Default constructor."""
        self.impl = _get_global_python_itf()

    fn __copyinit__(inout self, existing: Self):
        """Copy constructor.

        Args:
            existing: The existing instance to copy from.
        """
        self.impl = existing.impl

    fn eval(inout self, str: StringRef) -> Bool:
        """Executes the given Python code.

        Args:
            str: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.
        """
        var cpython = self.impl.cpython()
        return cpython.PyRun_SimpleString(str)

    @staticmethod
    fn evaluate(str: StringRef) raises -> PythonObject:
        """Executes the given Python code.

        Args:
            str: The Python expression to evaluate.

        Returns:
            `PythonObject` containing the result of the evaluation.
        """
        var cpython = _get_global_python_itf().cpython()
        let module = PythonObject(cpython.PyImport_AddModule("__main__"))
        # PyImport_AddModule returns a borrowed reference - IncRef it to keep it alive.
        cpython.Py_IncRef(module.py_object)
        let dictionary = PythonObject(
            cpython.PyModule_GetDict(module.py_object)
        )
        # PyModule_GetDict returns a borrowed reference - IncRef it to keep it alive.
        cpython.Py_IncRef(dictionary.py_object)
        let result = cpython.PyRun_String(
            str, dictionary.py_object, dictionary.py_object, Py_eval_input
        )
        # We no longer need module and dictionary, release them.
        Python.throw_python_exception_if_error_state(cpython)
        return PythonObject(result)

    @staticmethod
    fn add_to_path(str: StringRef) raises:
        """Adds a directory to the Python path.

        This might be necessary to import a Python module via `import_module()`.
        For example:

        ```mojo
        from python import Python

        # Specify path to `mypython.py` module
        Python.add_to_path("path/to/module")
        let mypython = Python.import_module("mypython")

        let c = mypython.my_algorithm(2, 3)
        ```

        Args:
            str: The path to a Python module you want to import.
        """
        let cpython = _get_global_python_itf().cpython()
        let sys = Python.import_module("sys")
        let directory: PythonObject = str
        _ = sys.path.append(directory)

    @staticmethod
    fn import_module(str: StringRef) raises -> PythonObject:
        """Imports a Python module.

        This provides you with a module object you can use just like you would
        in Python. For example:

        ```mojo
        from python import Python

        # This is equivalent to Python's `import numpy as np`
        let np = Python.import_module("numpy")
        a = np.array([1, 2, 3])
        ```

        Args:
            str: The Python module name. This module must be visible from the
                list of available Python paths (you might need to add the
                module's path with `add_to_path()`).

        Returns:
            The Python module.
        """
        var cpython = _get_global_python_itf().cpython()
        let module_maybe = cpython.PyImport_ImportModule(str)
        Python.throw_python_exception_if_error_state(cpython)
        return PythonObject(module_maybe)

    @staticmethod
    fn dict() -> Dictionary:
        """Construct an empty Python dictionary.

        Returns:
            The constructed empty Python dictionary.
        """
        var cpython = _get_global_python_itf().cpython()
        return Dictionary(cpython.PyDict_New())

    fn __str__(inout self, str: PythonObject) -> StringRef:
        """Return a string representing the given Python object.

        This function allows to convert Python objects to Mojo string type.

        Returns:
            Mojo string representing the given Python object.
        """
        var cpython = self.impl.cpython()
        return cpython.PyUnicode_AsUTF8AndSize(str.py_object)

    @staticmethod
    fn throw_python_exception_if_error_state(inout cpython: CPython) raises:
        """Raise an exception if CPython interpreter is in an error state.

        Args:
            cpython: The cpython instance we wish to error check.
        """
        if cpython.PyErr_Occurred():
            let error = PythonObject(cpython.PyErr_Fetch()).__getattr__(
                "__str__"
            )()
            let err: Error = cpython.PyUnicode_AsUTF8AndSize(error.py_object)
            cpython.PyErr_Clear()
            raise err

    @staticmethod
    fn is_type(x: PythonObject, y: PythonObject) -> Bool:
        """Test if the `x` object is the `y` object, the same as `x is y` in
        Python.

        Args:
            x: The left-hand-side value in the comparison.
            y: The right-hand-side type value in the comparison.

        Returns:
            True if `x` and `y` are the same object and False otherwise.
        """
        var cpython = _get_global_python_itf().cpython()
        return cpython.Py_Is(x.py_object, y.py_object)

    @staticmethod
    fn type(obj: PythonObject) -> PythonObject:
        """Return Type of this PythonObject.

        Args:
            obj: PythonObject we want the type of.

        Returns:
            A PythonObject that holds the type object.
        """
        var cpython = _get_global_python_itf().cpython()
        return cpython.PyObject_Type(obj.py_object)

    @staticmethod
    fn none() -> PythonObject:
        """Get a `PythonObject` representing `None`.

        Returns:
            `PythonObject` representing `None`.
        """
        var cpython = _get_global_python_itf().cpython()
        let nonetype = cpython.Py_NoneType()
        cpython.Py_IncRef(nonetype)
        return PythonObject(nonetype)
