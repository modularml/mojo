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

alias PyModule = TypedPythonObject["Module"]


fn get_cpython() -> CPython:
    return _get_global_python_itf().cpython()


fn create_pybind_module[name: StringLiteral]() raises -> PyModule:
    return Python.create_module(name)


fn pyobj_destroy_as[T: AnyType](pyobj: PyObjectPtr):
    # TODO(MSTDL-633): Is this always safe? Wrap in GIL, because this could
    # evaluate arbitrary code?
    pyobj.value.bitcast[T]().destroy_pointee()


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


alias MutableGlobalLifetime = __mlir_attr[
    `#lit.lifetime.field<`,
    `#lit.static.lifetime : !lit.lifetime<1>`,
    `, "__python_globals__"> : !lit.lifetime<1>`,
]


# FIXME(MOCO-1308): Workaround crash by adding explicit `alignment=1`.
alias PyGlobalPtr = UnsafePointer[lifetime=MutableGlobalLifetime, alignment=1]


@always_inline
fn global_alloc[T: AnyType, len: Int]() -> PyGlobalPtr[T]:
    return __mlir_op.`pop.global_alloc`[
        name = "pyglobal".value,
        count = len.value,
        _type = __mlir_type[`!kgen.pointer<`, T, `>`],
    ]()


fn pointer_bitcast[
    To: AnyType
](ptr: Pointer) -> Pointer[To, ptr.lifetime, ptr.address_space, *_, **_] as out:
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
    T: AnyType, name: StringLiteral
](inout module: PythonObject) raises:
    cpython = get_cpython()

    # TODO(MOCO-1301): Add support for member method generation.
    # TODO(MOCO-1302): Add support for generating member field as computed properties.

    # Zeroed item as terminator
    methods = global_alloc[PyMethodDef, 1]()
    methods[0] = PyMethodDef()

    # TODO(MOCO-1307): Add support for constructor generation.
    slots = global_alloc[PyType_Slot, 3]()
    slots[0] = PyType_Slot(
        cp.Py_tp_new, cpython.lib.get_symbol[NoneType]("PyType_GenericNew")
    )
    alias dtor = pyobj_destroy_as[T]
    slots[1] = PyType_Slot(
        cp.Py_tp_dealloc,
        __mlir_op.`pop.pointer.bitcast`[_type = OpaquePointer._mlir_type](dtor),
    )
    slots[2] = PyType_Slot.null()

    spec = global_alloc[PyType_Spec, 1]()
    spec.init_pointee_move(
        PyType_Spec {
            # FIXME(MOCO-1306): This should be `T.__name__`.
            name: name.unsafe_cstr_ptr(),
            basicsize: sizeof[T](),
            itemsize: 0,
            flags: cp.Py_TPFLAGS_DEFAULT,
            slots: slots,
        }
    )

    type_obj = cpython.PyType_FromSpec(spec)
    # FIXME(MSTDL-957): We should have APIs that explicitly take a `CPython`
    # instance so that callers can pass it around instead of performing a lookup
    # each time.
    # FIXME(MSTDL-969): Bitcast to `TypedPythonObject["Module"]`.
    Python.add_object(
        pointer_bitcast[PyModule](Pointer.address_of(module))[], name, type_obj
    )
