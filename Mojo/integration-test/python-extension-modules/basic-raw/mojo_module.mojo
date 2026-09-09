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

from std.os import abort

from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    """Create a Python module with a function binding for `mojo_count_args`."""

    try:
        var b = PythonModuleBuilder("mojo_module")

        # Test def_py_c_function
        b.def_py_c_function(
            mojo_count_args,
            "mojo_count_args",
            docstring="Count the provided arguments",
        )
        b.def_py_c_function(
            mojo_count_args_with_kwargs,
            "mojo_count_args_with_kwargs",
            docstring="Count the provided arguments and keyword arguments",
        )

        # Test def_py_c_method
        _ = (
            b.add_type[TestCounter]("TestCounter")
            .def_init_defaultable[TestCounter]()
            .def_py_c_method(
                counter_count_args,
                "count_args",
                docstring="Count the provided arguments",
            )
            .def_py_c_method(
                counter_count_args_with_kwargs,
                "count_args_with_kwargs",
                docstring="Count the provided arguments and keyword arguments",
            )
            .def_py_c_method[static_method=True](
                counter_static_count_args,
                "static_count_args",
                docstring="Count the provided arguments",
            )
            .def_py_c_method[static_method=True](
                counter_static_count_args_with_kwargs,
                "static_count_args_with_kwargs",
                docstring="Count the provided arguments and keyword arguments",
            )
        )

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


@export
def mojo_count_args(
    py_self: PyObjectPtr, args: PyObjectPtr
) abi("C") -> PyObjectPtr:
    """Count the provided arguments.

    Return value: New reference.
    """
    return mojo_count_args_with_kwargs(py_self, args, {})


@export
def mojo_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) abi("C") -> PyObjectPtr:
    """Count the provided arguments and keyword arguments.

    Return value: New reference.
    """
    ref cpy = Python().cpython()
    var count = cpy.PyObject_Length(args) + (
        cpy.PyObject_Length(kwargs) if kwargs else 0
    )
    return cpy.PyLong_FromSsize_t(count)


struct TestCounter(Defaultable, ImplicitlyCopyable, Writable):
    var value: Int

    def __init__(out self):
        self.value = 0


@export
def counter_count_args(
    py_self: PyObjectPtr, args: PyObjectPtr
) abi("C") -> PyObjectPtr:
    """PyCFunction method to count arguments."""
    return mojo_count_args(py_self, args)


@export
def counter_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) abi("C") -> PyObjectPtr:
    return mojo_count_args_with_kwargs(py_self, args, kwargs)


@export
def counter_static_count_args(
    py_self: PyObjectPtr, args: PyObjectPtr
) abi("C") -> PyObjectPtr:
    return mojo_count_args(py_self, args)


@export
def counter_static_count_args_with_kwargs(
    py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr
) abi("C") -> PyObjectPtr:
    return mojo_count_args_with_kwargs(py_self, args, kwargs)
