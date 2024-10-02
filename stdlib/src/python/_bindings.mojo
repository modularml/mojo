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

from memory import UnsafePointer

from sys.ffi import c_int

from python import PythonObject, TypedPythonObject
from python.python import _get_global_python_itf
from python._cpython import PyObject, PyObjectPtr, PyCFunction

# ===-----------------------------------------------------------------------===#
# Mojo Object
# ===-----------------------------------------------------------------------===#

# Must be ABI compatible with `initfunc`
alias Typed_initfunc = fn (
    PyObjectPtr, TypedPythonObject["Tuple"], PythonObject
) -> c_int


struct PyMojoObject[T: AnyType]:
    """Storage backing a PyObject* wrapping a Mojo value."""

    var ob_base: PyObject
    var mojo_value: T

    @staticmethod
    fn unsafe_cast_obj(obj_raw_ptr: PyObjectPtr) -> UnsafePointer[T]:
        var mojo_obj_ptr = obj_raw_ptr.value.bitcast[PyMojoObject[T]]()

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return UnsafePointer[T].address_of(mojo_obj_ptr[].mojo_value)


# ===-----------------------------------------------------------------------===#
# PyCFunction Wrappers
# ===-----------------------------------------------------------------------===#


fn create_wrapper_function[
    user_func: fn (PythonObject, TypedPythonObject["Tuple"]) -> PythonObject
]() -> PyCFunction:
    #   > When a C function is called from Python, it borrows references to its
    #   > arguments from the caller. The caller owns a reference to the object,
    #   > so the borrowed reference’s lifetime is guaranteed until the function
    #   > returns. Only when such a borrowed reference must be stored or passed
    #   > on, it must be turned into an owned reference by calling Py_INCREF().
    #   >
    #   >  -- https://docs.python.org/3/extending/extending.html#ownership-rules

    fn wrapper(py_self_ptr: PyObjectPtr, args_ptr: PyObjectPtr) -> PyObjectPtr:
        # SAFETY:
        #   Here we illegally (but carefully) construct _owned_ `PythonObject`
        #   values from the borrowed object reference arguments. We are careful
        #   down below to prevent the destructor for these objects from running
        #   so that we do not illegally decrement the reference count of these
        #   objects we do not own.
        #
        #   This is valid to do, because these are passed using the `borrowed`
        #   argument convention to `user_func`, so logically they are treated
        #   as Python borrowed references.
        var py_self = PythonObject(py_self_ptr)
        var args = TypedPythonObject["Tuple"](
            unsafe_unchecked_from=PythonObject(args_ptr)
        )

        # SAFETY:
        #   Call the user provided function, and take ownership of the
        #   PyObjectPtr of the returned PythonObject.
        var result = user_func(py_self, args).steal_data()

        # Do not destroy the provided PyObjectPtr arguments, since they
        # actually have ownership of the underlying object.
        __mlir_op.`lit.ownership.mark_destroyed`(
            __get_mvalue_as_litref(py_self)
        )

        # SAFETY:
        #   Prevent `args` AND `args._obj` from being destroyed, since we don't
        #   own them.
        # TODO: Use a `mem.forget(args^)` function here in the future.
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(args))
        var _obj = args._obj^
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(_obj))

        return result

    return wrapper


# Wrap a `raises` function
fn create_wrapper_function[
    user_func: fn (
        PythonObject, TypedPythonObject["Tuple"]
    ) raises -> PythonObject
]() -> PyCFunction:
    fn wrapper(
        py_self: PythonObject, args: TypedPythonObject["Tuple"]
    ) -> PythonObject:
        var cpython = _get_global_python_itf().cpython()

        var state = cpython.PyGILState_Ensure()

        try:
            var result = user_func(py_self, args)
            return result
        except e:
            # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
            var error_type = cpython.get_error_global("PyExc_Exception")

            cpython.PyErr_SetString(
                error_type,
                e.unsafe_cstr_ptr(),
            )

            # Return a NULL `PyObject*`.
            return PythonObject(PyObjectPtr())
        finally:
            cpython.PyGILState_Release(state)

    # TODO:
    #   Does this lead to multiple levels of indirect function calls for
    #   `raises` functions? Could we fix that by marking `wrapper` here as
    #   `@always_inline`?
    # Call the non-`raises` overload of `create_wrapper_function`.
    return create_wrapper_function[wrapper]()
