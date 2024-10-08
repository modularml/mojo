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
from python import TypedPythonObject, Python
from python._cpython import (
    PyObjectPtr,
    PyMethodDef,
    PyType_Slot,
    PyType_Spec,
    CPython,
)


fn create_pybind_module[
    name: StringLiteral
]() raises -> TypedPythonObject["Module"]:
    return Python.create_module(name)


@value
@register_passable("trivial")
struct PyTypeDefSizes:
    var pymethod_def_size: Int
    var pymethod_def_align: Int

    var pytype_slot_size: Int
    var pytype_slot_align: Int

    var pytype_spec_size: Int
    var pytype_spec_align: Int


fn get_pytype_def_sizes() -> PyTypeDefSizes:
    return PyTypeDefSizes(
        sizeof[PyMethodDef](),
        alignof[PyMethodDef](),
        sizeof[PyType_Slot](),
        alignof[PyType_Slot](),
        sizeof[PyType_Spec](),
        alignof[PyType_Spec](),
    )


fn pyobj_destroy_as[T: AnyType](pyobj: PyObjectPtr):
    # TODO(MSTDL-633): Is this always safe? Wrap in GIL, because this could
    # evaluate arbitrary code?
    pyobj.value.bitcast[T]().destroy_pointee()


alias MutableGlobalLifetime = __mlir_attr[
    `#lit.lifetime.field<`,
    `#lit.static.lifetime : !lit.lifetime<1>`,
    `, "__python_globals__"> : !lit.lifetime<1>`,
]


# FIXME(MOCO-1308): Workaround crash by adding explicit `alignment=1`.
alias PyGlobalPtr = UnsafePointer[lifetime=MutableGlobalLifetime, alignment=1]


fn gen_pytype_wrapper[
    T: AnyType, name: StringLiteral
](
    inout cpython: CPython,
    inout module: TypedPythonObject["Module"],
    methods: PyGlobalPtr[PyMethodDef],
    slots: PyGlobalPtr[PyType_Slot],
    spec: PyGlobalPtr[PyType_Spec],
) raises:
    # TODO(MOCO-1301): Add support for member method generation.
    # TODO(MOCO-1302): Add support for generating member field as computed properties.

    # Zeroed item as terminator
    methods[0] = PyMethodDef()

    # TODO(MOCO-1307): Add support for constructor generation.
    slots[0] = PyType_Slot(
        cp.Py_tp_new, cpython.lib.get_symbol[NoneType]("PyType_GenericNew")
    )
    alias dtor = pyobj_destroy_as[T]
    slots[1] = PyType_Slot(
        cp.Py_tp_dealloc,
        __mlir_op.`pop.pointer.bitcast`[_type = OpaquePointer._mlir_type](dtor),
    )
    slots[2] = PyType_Slot.null()

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
    Python.add_object(module, name, type_obj)
