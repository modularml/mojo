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

from collections import Optional
from sys import alignof, sizeof

import python._cpython as cp
from memory import UnsafePointer, stack_allocation
from python import Python, PythonObject, TypedPythonObject
from python._bindings import (  # Imported for use by the compiler
    ConvertibleFromPython,
    PyMojoObject,
    Pythonable,
    check_argument_type,
    check_arguments_arity,
    py_c_function_wrapper,
    python_type_object,
)
from python._cpython import (
    CPython,
    PyMethodDef,
    PyObjectPtr,
    PyType_Slot,
    PyType_Spec,
)
from python.python import _get_global_python_itf

alias PyModule = TypedPythonObject["Module"]


fn get_cpython() -> CPython:
    return _get_global_python_itf().cpython()


fn create_pybind_module[name: StringLiteral]() raises -> PyModule:
    return Python.create_module(name)


fn fail_initialization(owned err: Error) -> PythonObject:
    # TODO(MSTDL-933): Add custom 'MojoError' type, and raise it here.
    cpython = get_cpython()
    error_type = cpython.get_error_global("PyExc_Exception")

    cpython.PyErr_SetString(
        error_type,
        err.unsafe_cstr_ptr(),
    )
    _ = err^
    return PythonObject(PyObjectPtr())


fn pointer_bitcast[
    To: AnyType
](
    ptr: Pointer,
    out result: Pointer[To, ptr.origin, ptr.address_space, *_, **_],
):
    return __type_of(result)(
        _mlir_value=__mlir_op.`lit.ref.from_pointer`[
            _type = __type_of(result)._mlir_type
        ](
            UnsafePointer(__mlir_op.`lit.ref.to_pointer`(ptr._value))
            .bitcast[To]()
            .address
        )
    )


fn gen_pytype_wrapper[
    T: Pythonable,
    name: StringLiteral,
](mut module: PythonObject) raises:
    # TODO(MOCO-1301): Add support for member method generation.
    # TODO(MOCO-1302): Add support for generating member field as computed properties.
    # TODO(MOCO-1307): Add support for constructor generation.

    var type_obj = python_type_object[T, name](
        methods=List[PyMethodDef](),
    )

    # FIXME(MSTDL-957): We should have APIs that explicitly take a `CPython`
    # instance so that callers can pass it around instead of performing a lookup
    # each time.
    # FIXME(MSTDL-969): Bitcast to `TypedPythonObject["Module"]`.
    Python.add_object(
        pointer_bitcast[PyModule](Pointer.address_of(module))[], name, type_obj
    )


fn add_wrapper_to_module[
    wrapper_func: fn (
        PythonObject, TypedPythonObject["Tuple"]
    ) raises -> PythonObject,
    func_name: StringLiteral,
](mut module_obj: PythonObject) raises:
    var module = TypedPythonObject["Module"](unsafe_unchecked_from=module_obj)
    Python.add_functions(
        module,
        List[PyMethodDef](
            PyMethodDef.function[
                py_c_function_wrapper[wrapper_func], func_name
            ]()
        ),
    )


fn check_and_get_arg[
    T: AnyType
](
    func_name: StringLiteral,
    type_name_id: StringLiteral,
    py_args: TypedPythonObject["Tuple"],
    index: Int,
) raises -> UnsafePointer[T]:
    return check_argument_type[T](func_name, type_name_id, py_args[index])


# NOTE:
#   @always_inline is needed so that the stack_allocation() that appears in
#   the definition below is valid in the _callers_ stack frame, effectively
#   allowing us to "return" a pointer to stack-allocated data from this
#   function.
@always_inline
fn check_and_get_or_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StringLiteral,
    type_name_id: StringLiteral,
    py_args: TypedPythonObject["Tuple"],
    index: Int,
) raises -> UnsafePointer[T]:
    # Stack space to hold a converted value for this argument, if needed.
    var converted_arg_ptr: UnsafePointer[T] = stack_allocation[1, T]()

    try:
        return check_and_get_arg[T](func_name, type_name_id, py_args, index)
    except e:
        converted_arg_ptr.init_pointee_move(
            _try_convert_arg[T](
                func_name,
                type_name_id,
                py_args,
                index,
            )
        )
        # Return a pointer to stack data. Only valid because this function is
        # @always_inline.
        return converted_arg_ptr


fn _try_convert_arg[
    T: ConvertibleFromPython
](
    func_name: StringLiteral,
    type_name_id: StringLiteral,
    py_args: TypedPythonObject["Tuple"],
    argidx: Int,
    out result: T,
) raises:
    try:
        result = T.try_from_python(py_args[argidx])
    except convert_err:
        raise Error(
            String.format(
                (
                    "TypeError: {}() expected argument at position {} to be"
                    " instance of (or convertible to) Mojo '{}'; got '{}'."
                    " (Note: attempted conversion failed due to: {})"
                ),
                func_name,
                argidx,
                type_name_id,
                py_args[argidx]._get_type_name(),
                convert_err,
            )
        )
