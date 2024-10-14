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

from sys import sizeof, alignof
from sys.ffi import OpaquePointer

import python._cpython as cp
from python import TypedPythonObject, Python, PythonObject
from python.python import _get_global_python_itf
from python._cpython import (
    PyObjectPtr,
    PyMethodDef,
    PyType_Slot,
    PyType_Spec,
    CPython,
)
from python._bindings import Pythonable, PyMojoObject

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
](ptr: Pointer) -> Pointer[To, ptr.origin, ptr.address_space, *_, **_] as out:
    return __type_of(out)(
        _mlir_value=__mlir_op.`lit.ref.from_pointer`[
            _type = __type_of(out)._mlir_type
        ](
            UnsafePointer(__mlir_op.`lit.ref.to_pointer`(ptr._value))
            .bitcast[To]()
            .address
        )
    )


fn gen_pytype_wrapper[
    T: Pythonable,
    name: StringLiteral,
](inout module: PythonObject) raises:
    # TODO(MOCO-1301): Add support for member method generation.
    # TODO(MOCO-1302): Add support for generating member field as computed properties.
    # TODO(MOCO-1307): Add support for constructor generation.

    var type_obj = PyMojoObject[T].python_type_object[name](
        methods=List[PyMethodDef](),
    )

    # FIXME(MSTDL-957): We should have APIs that explicitly take a `CPython`
    # instance so that callers can pass it around instead of performing a lookup
    # each time.
    # FIXME(MSTDL-969): Bitcast to `TypedPythonObject["Module"]`.
    Python.add_object(
        pointer_bitcast[PyModule](Pointer.address_of(module))[], name, type_obj
    )
