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

from memory import UnsafePointer, Box

from sys.ffi import c_int
from sys.info import sizeof

from os import abort

from collections import Optional

from python import PythonObject, TypedPythonObject
from python.python import _get_global_python_itf
from python._cpython import (
    PyObject,
    PyObjectPtr,
    PyCFunction,
    PyType_Spec,
    PyType_Slot,
    PyMethodDef,
    Py_TPFLAGS_DEFAULT,
    newfunc,
    destructor,
)

# ===-----------------------------------------------------------------------===#
# Mojo Object
# ===-----------------------------------------------------------------------===#

# Must be ABI compatible with `initproc`
alias Typed_initproc = fn (
    PyObjectPtr,
    TypedPythonObject["Tuple"],
    # Will be NULL if no keyword arguments were passed.
    PyObjectPtr,
) -> c_int


trait Pythonable(Defaultable, Representable):
    """Represents a Mojo type that can be used from Python."""

    pass


struct PyMojoObject[T: Pythonable]:
    """Storage backing a PyObject* wrapping a Mojo value."""

    var ob_base: PyObject
    var mojo_value: T

    @staticmethod
    fn python_type_object[
        type_name: StringLiteral,
    ](owned methods: List[PyMethodDef]) raises -> TypedPythonObject["Type"]:
        """Construct a Python 'type' describing PyMojoObject[T].

        Parameters:
            type_name: The name of the Mojo type.
        """

        var cpython = _get_global_python_itf().cpython()

        var slots = List[PyType_Slot](
            # All wrapped Mojo types are allocated generically.
            PyType_Slot.tp_new(
                cpython.lib.get_function[newfunc]("PyType_GenericNew")
            ),
            PyType_Slot.tp_init(empty_tp_init_wrapper[T]),
            PyType_Slot.tp_dealloc(tp_dealloc_wrapper[T]),
            PyType_Slot.tp_repr(tp_repr_wrapper[T]),
        )

        if methods:
            # FIXME: Avoid leaking the methods data pointer in this way.
            slots.append(PyType_Slot.tp_methods(methods.steal_data()))

        # Zeroed item terminator
        slots.append(PyType_Slot.null())

        var type_spec = PyType_Spec {
            # FIXME(MOCO-1306): This should be `T.__name__`.
            name: type_name.unsafe_cstr_ptr(),
            basicsize: sizeof[PyMojoObject[T]](),
            itemsize: 0,
            flags: Py_TPFLAGS_DEFAULT,
            # Note: This pointer is only "borrowed" by PyType_FromSpec.
            slots: slots.unsafe_ptr(),
        }

        # Construct a Python 'type' object from our type spec.
        var type_obj = cpython.PyType_FromSpec(
            UnsafePointer.address_of(type_spec)
        )

        if type_obj.is_null():
            Python.throw_python_exception_if_error_state(cpython)
            return abort[TypedPythonObject["Type"]](
                "expected to raise after getting NULL type object"
            )

        return TypedPythonObject["Type"](
            unsafe_unchecked_from=PythonObject(type_obj)
        )


# Impedance match between:
#
#   Mojo:   fn(UnsafePointer[T]) -> None
#   Python: fn(PyObjectPtr, PyObjectPtr, PyObjectPtr)
#
# The latter is the C function signature that the CPython API expects a
# PyObject initializer function to have. The former is an unsafe form of the
# `fn(inout self)` signature that Mojo types with default constructors provide.
#
# To support CPython calling a Mojo types default constructor, we need to
# provide a wrapper function (around the Mojo constructor) that has the
# signature the C API expects.
#
# This function creates that wrapper function, and returns a pointer pointer to
# it.
fn empty_tp_init_wrapper[
    T: Pythonable
](
    py_self: PyObjectPtr,
    args: TypedPythonObject["Tuple"],
    keyword_args: PyObjectPtr,
) -> c_int:
    """Python-compatible wrapper around a Mojo initializer function.

    Parameters:
        T: The wrapped Mojo type.
    """

    var cpython = _get_global_python_itf().cpython()

    try:
        if len(args) != 0 or keyword_args != PyObjectPtr():
            raise "unexpected arguments passed to default initializer function of wrapped Mojo type"

        var obj_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[
            T
        ]()

        # ------------------------------------------------
        # Call the user-provided initialization function.
        # ------------------------------------------------

        # TODO(MSTDL-950): Avoid forming ref through uninit pointee.
        T.__init__(obj_ptr[])

        return 0
    except e:
        # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
        var error_type = cpython.get_error_global("PyExc_ValueError")

        cpython.PyErr_SetString(
            error_type,
            e.unsafe_cstr_ptr(),
        )

        return -1


fn tp_dealloc_wrapper[T: Pythonable](py_self: PyObjectPtr):
    var self_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[T]()

    # TODO(MSTDL-633):
    #   Is this always safe? Wrap in GIL, because this could
    #   evaluate arbitrary code?
    # Destroy this `Person` instance.
    self_ptr.destroy_pointee()


fn tp_repr_wrapper[T: Pythonable](py_self: PyObjectPtr) -> PyObjectPtr:
    var self_ptr: UnsafePointer[T] = py_self.unchecked_cast_to_mojo_value[T]()

    var repr_str: String = repr(self_ptr[])

    return PythonObject(string=repr_str).steal_data()


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


fn check_arguments_arity(
    func_name: StringLiteral,
    arity: Int,
    args: TypedPythonObject["Tuple"],
) raises:
    """Raise an error if the provided argument count does not match the expected
    function arity.

    If this function returns normally (without raising), then the argument
    count is exactly equal to the expected arity.
    """

    var arg_count = len(args)

    # The error messages raised below are intended to be similar to the
    # equivalent errors in Python.
    if arg_count != arity:
        if arg_count < arity:
            var missing_arg_count = arity - arg_count

            raise Error(
                String.format(
                    "TypeError: {}() missing {} required positional {}",
                    func_name,
                    missing_arg_count,
                    _pluralize(missing_arg_count, "argument", "arguments"),
                )
            )
        else:
            raise Error(
                String.format(
                    "TypeError: {}() takes {} positional {} but {} were given",
                    func_name,
                    arity,
                    _pluralize(arity, "argument", "arguments"),
                    arg_count,
                )
            )


fn check_argument_type[
    T: Pythonable,
](
    func_name: StringLiteral,
    type_name_id: StringLiteral,
    obj: PythonObject,
) raises -> UnsafePointer[T]:
    """Raise an error if the provided Python object does not contain a wrapped
    instance of the Mojo `T` type.
    """

    var opt: Optional[UnsafePointer[T]] = obj.py_object.try_cast_to_mojo_value[
        T
    ](type_name_id)

    if not opt:
        var cpython = _get_global_python_itf().cpython()

        var actual_type = cpython.Py_TYPE(obj.unsafe_as_py_object_ptr())
        var actual_type_name = PythonObject(cpython.PyType_GetName(actual_type))

        raise Error(
            String.format(
                "TypeError: {}() expected Mojo '{}' type argument, got '{}'",
                func_name,
                type_name_id,
                str(actual_type_name),
            )
        )

    # SAFETY: We just validated that this Optional is not empty.
    return opt.unsafe_take()


fn _pluralize(
    count: Int,
    singular: StringLiteral,
    plural: StringLiteral,
) -> StringLiteral:
    if count == 1:
        return singular
    else:
        return plural
