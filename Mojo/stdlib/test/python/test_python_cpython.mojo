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

from std.python import Python
from std.python._cpython import (
    CPython,
    Py_eval_input,
    Py_file_input,
    Py_ssize_t,
    PyMethodDef,
    PyModuleDef,
    PyObject,
    PyObjectPtr,
    Py_TPFLAGS_LONG_SUBCLASS,
    Py_TPFLAGS_LIST_SUBCLASS,
)
from std.memory import alloc, dealloc, ThinAllocation, Layout
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def _test_very_high_level_api(cpy: CPython) raises:
    assert_equal(cpy.PyRun_SimpleString("None"), 0)

    var d = cpy.PyDict_New()
    assert_true(cpy.PyRun_String("42", Py_eval_input, d, d))

    var co = cpy.Py_CompileString("5", "test", Py_eval_input)
    assert_true(co)

    assert_true(cpy.PyEval_EvalCode(co, d, d))


def _test_reference_counting_api(cpy: CPython) raises:
    # this is the smallest integer that's GC'd by the Python interpreter
    var n = cpy.PyLong_FromSsize_t(257)
    assert_equal(cpy._Py_REFCNT(n), 1)

    cpy.Py_IncRef(n)
    assert_equal(cpy._Py_REFCNT(n), 2)

    cpy.Py_DecRef(n)
    assert_equal(cpy._Py_REFCNT(n), 1)

    var m = cpy.Py_NewRef(n)
    assert_equal(cpy._Py_REFCNT(m), 2)


def _test_exception_handling_api(cpy: CPython) raises:
    var ValueError = cpy.get_error_global("PyExc_ValueError")
    var msg = "some error message"

    assert_false(cpy.PyErr_Occurred())

    cpy.PyErr_SetNone(ValueError)
    assert_true(cpy.PyErr_Occurred())
    cpy.PyErr_Clear()

    cpy.PyErr_SetString(
        ValueError, msg.as_c_string_slice().ptr().as_unsafe_any_origin()
    )
    assert_true(cpy.PyErr_Occurred())

    if cpy.version.minor < 12:
        # PyErr_Fetch is deprecated since Python 3.12.
        assert_true(cpy.PyErr_Fetch())
        # Manually clear the error indicator.
        cpy.PyErr_Clear()
    else:
        # PyErr_GetRaisedException is new in Python 3.12.
        # PyErr_GetRaisedException clears the error indicator.
        assert_true(cpy.PyErr_GetRaisedException())

    _ = msg


def _test_threading_api(cpy: CPython) raises:
    var gstate = cpy.PyGILState_Ensure()
    var save = cpy.PyEval_SaveThread()
    cpy.PyEval_RestoreThread(save)
    cpy.PyGILState_Release(gstate)


def _test_importing_module_api(cpy: CPython) raises:
    assert_true(cpy.PyImport_ImportModule("builtins"))
    assert_true(cpy.PyImport_AddModule("test"))


def _test_object_protocol_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    var z = cpy.PyLong_FromSsize_t(0)
    var l = cpy.PyList_New(1)
    _ = cpy.PyList_SetItem(l, 0, cpy.Py_NewRef(z))

    assert_equal(cpy.PyObject_HasAttrString(n, "__hash__"), 1)
    assert_true(cpy.PyObject_GetAttrString(n, "__hash__"))
    assert_equal(cpy.PyObject_SetAttrString(n, "attr", cpy.Py_None()), -1)
    cpy.PyErr_Clear()

    assert_true(cpy.PyObject_Str(n))
    assert_equal(cpy.PyObject_Hash(n), 42)
    assert_equal(cpy.PyObject_IsTrue(n), 1)
    assert_true(cpy.PyObject_Type(n))
    assert_equal(cpy.PyObject_Length(l), 1)

    assert_equal(cpy.PyObject_GetItem(l, z), z)
    assert_equal(cpy.PyObject_SetItem(l, z, n), 0)
    assert_equal(cpy.PyObject_GetItem(l, z), n)

    var it = cpy.PyObject_GetIter(l)
    assert_true(it)
    assert_equal(cpy.PyObject_GetIter(it), it)


def _test_call_protocol_api(cpy: CPython) raises:
    var dict_func = PyObjectPtr(upcast_from=cpy.PyDict_Type())
    var t = cpy.PyTuple_New(0)
    var d = cpy.PyDict_New()

    assert_true(cpy.PyObject_CallObject(dict_func, t))
    assert_true(cpy.PyObject_Call(dict_func, t, d))


def _test_number_protocol_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)

    var long_value = cpy.PyNumber_Long(n)
    assert_true(long_value)
    assert_equal(cpy.PyLong_AsSsize_t(long_value), 42)

    var float_value = cpy.PyNumber_Float(n)
    assert_true(float_value)
    assert_equal(cpy.PyFloat_AsDouble(float_value), 42.0)


def _test_iterator_protocol_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    var l = cpy.PyList_New(1)
    _ = cpy.PyList_SetItem(l, 0, cpy.Py_NewRef(n))

    var it = cpy.PyObject_GetIter(l)

    assert_false(cpy.PyIter_Check(n))
    assert_true(it)
    assert_true(cpy.PyIter_Next(it))


def _test_type_object_api(cpy: CPython) raises:
    var dict_type = cpy.PyDict_Type()
    assert_true(cpy.PyType_GetName(dict_type))


def _helper_instantiate_derived_class(
    base_class: String, cpy: CPython
) raises -> PyObjectPtr:
    """Create a derived class from one of the basic types and
    then instantiate it.
    """
    # Setup the environment for PyRun_String.
    var test_mod = cpy.PyModule_Create("test_mod")
    var globals = cpy.PyDict_New()
    var locals = cpy.PyModule_GetDict(test_mod)
    # Create a derived class.
    #
    # Note: the second argument must be Py_file_input otherwise the
    # result is NULL.
    _ = cpy.PyRun_String(
        "class D({}):\n    pass\n\n".format(base_class),
        Py_file_input,
        globals,
        locals,
    )
    # Get a handle to the constructor we have just created.
    var constructor = cpy.PyObject_GetAttrString(test_mod, "D")
    # We need to pass in some arguments from the newly defined constructor.
    var args = cpy.PyTuple_New(0)
    return cpy.PyObject_CallObject(constructor, args)


def _test_integer_object_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(-42)
    assert_true(n)
    assert_equal(cpy.PyLong_AsSsize_t(n), -42)
    assert_true(cpy.PyLong_Check(n))
    assert_true(cpy.PyLong_CheckExact(n))
    assert_true(cpy.PyObject_TypeCheck(n, cpy.PyLong_Type()))

    var z = cpy.PyLong_FromSize_t(57)
    assert_true(z)
    assert_equal(cpy.PyLong_AsSsize_t(z), 57)

    var none = cpy.Py_None()
    assert_false(cpy.PyLong_Check(none))
    assert_false(cpy.PyLong_CheckExact(none))

    # Derive a class from int to be able to test some more the Check* API.
    var instantiated = _helper_instantiate_derived_class("int", cpy)
    assert_true(cpy.PyLong_Check(instantiated))
    assert_false(cpy.PyLong_CheckExact(instantiated))


def _test_boolean_object_api(cpy: CPython) raises:
    var t = cpy.PyBool_FromLong(1)
    assert_true(t)
    assert_equal(cpy.PyObject_IsTrue(t), 1)
    assert_true(cpy.PyBool_Check(t))
    assert_true(cpy.PyObject_TypeCheck(t, cpy.PyBool_Type()))

    var f = cpy.PyBool_FromLong(0)
    assert_true(f)
    assert_equal(cpy.PyObject_IsTrue(f), 0)
    assert_true(cpy.PyBool_Check(t))
    assert_true(cpy.PyObject_TypeCheck(t, cpy.PyBool_Type()))

    var none = cpy.Py_None()
    assert_false(cpy.PyBool_Check(none))


def _test_floating_point_object_api(cpy: CPython) raises:
    var f = cpy.PyFloat_FromDouble(3.14)
    assert_true(f)
    assert_equal(cpy.PyFloat_AsDouble(f), 3.14)
    assert_true(cpy.PyFloat_Check(f))
    assert_true(cpy.PyFloat_CheckExact(f))
    assert_true(cpy.PyObject_TypeCheck(f, cpy.PyFloat_Type()))

    var none = cpy.Py_None()
    assert_false(cpy.PyFloat_Check(none))
    assert_false(cpy.PyFloat_CheckExact(none))
    # Derive a class from float to be able to test some more the Check* API.
    var instantiated = _helper_instantiate_derived_class("float", cpy)
    assert_true(cpy.PyFloat_Check(instantiated))
    assert_false(cpy.PyFloat_CheckExact(instantiated))


def _test_unicode_object_api(cpy: CPython) raises:
    var str = "Hello, World!"

    var py_str = cpy.PyUnicode_DecodeUTF8(str)
    assert_true(py_str)

    var res = cpy.PyUnicode_AsUTF8AndSize(py_str)
    assert_equal(res[], str)


def _test_tuple_object_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    var t = cpy.PyTuple_New(1)
    assert_true(t)

    # PyTuple_SetItem steals a reference to the object
    cpy.Py_IncRef(n)
    assert_equal(cpy.PyTuple_SetItem(t, 0, n), 0)
    assert_equal(cpy.PyTuple_GetItem(t, 0), n)


def _test_list_object_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    var l = cpy.PyList_New(1)
    assert_true(l)

    # PyList_SetItem steals a reference to the object
    cpy.Py_IncRef(n)
    assert_equal(cpy.PyList_SetItem(l, 0, n), 0)
    assert_equal(cpy.PyList_GetItem(l, 0), n)


def _test_dictionary_object_api(cpy: CPython) raises:
    var d = cpy.PyDict_New()
    var b = cpy.PyBool_FromLong(0)

    assert_equal(cpy.PyDict_SetItem(d, b, b), 0)
    assert_equal(cpy.PyDict_GetItemWithError(d, b), b)

    var key = PyObjectPtr()
    var value = PyObjectPtr()
    var pos: Py_ssize_t = 0

    var succ = cpy.PyDict_Next(
        d,
        Pointer(to=pos).as_unsafe_any_origin(),
        Pointer(to=key).as_unsafe_any_origin(),
        Pointer(to=value).as_unsafe_any_origin(),
    )
    assert_equal(pos, 1)
    assert_equal(key, b)
    assert_equal(value, b)
    assert_true(succ)

    succ = cpy.PyDict_Next(
        d,
        Pointer(to=pos).as_unsafe_any_origin(),
        Pointer(to=key).as_unsafe_any_origin(),
        Pointer(to=value).as_unsafe_any_origin(),
    )
    assert_false(succ)


def _test_set_object_api(cpy: CPython) raises:
    var s = cpy.PySet_New({})
    assert_true(s)

    var n = cpy.PyLong_FromSsize_t(42)
    assert_equal(cpy.PySet_Add(s, n), 0)


def _test_module_object_api(cpy: CPython) raises:
    var mod = cpy.PyModule_Create("module")

    assert_true(mod)
    assert_true(cpy.PyModule_GetDict(mod))

    var funcs = Array[PyMethodDef, 1](fill={})
    # returns 0 on success, -1 on failure
    assert_equal(
        cpy.PyModule_AddFunctions(
            mod, funcs.unsafe_ptr().as_unsafe_any_origin()
        ),
        0,
    )
    _ = funcs

    var n = cpy.PyLong_FromSsize_t(0)
    var name = "n"
    # returns 0 on success, -1 on failure
    assert_equal(
        cpy.PyModule_AddObjectRef(
            mod, name.as_c_string_slice().ptr().as_unsafe_any_origin(), n
        ),
        0,
    )
    _ = name


def _test_slice_object_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    assert_true(cpy.PySlice_New(n, n, n))


def _test_capsule_api(cpy: CPython) raises:
    var o = PyObjectPtr()
    with assert_raises(contains="called with invalid PyCapsule object"):
        _ = cpy.PyCapsule_GetPointer(o, "some_name")

    var capsule_impl_layout = Layout[UInt64].single()
    var capsule_impl_ptr = alloc(capsule_impl_layout).unsafe_leak()

    def empty_dtor(capsule: PyObjectPtr) abi("C"):
        pass

    # The capsule stores the `name` pointer directly (CPython does not copy it),
    # so the name must outlive the capsule. `PyCapsule_New` takes a
    # `StaticString` to enforce this: the string literal below has a `'static`
    # lifetime, so the pointer remains valid for the `PyCapsule_GetPointer`
    # lookups after `PyCapsule_New` returns.
    var capsule = cpy.PyCapsule_New(
        capsule_impl_ptr.unsafe_bitcast[NoneType]().unsafe_origin_cast[
            MutUntrackedOrigin
        ](),
        "some_name",
        empty_dtor,
    )
    var capsule_pointer = cpy.PyCapsule_GetPointer(capsule, "some_name")

    # The capsule holds the raw pointer but never dereferences it (its
    # destructor is a no-op and the lookups below fail before any deref), so it
    # is safe to release the storage now. Capture the address first because
    # `assert_equal` can raise, and the allocation must be freed on every path.
    var capsule_impl_addr = Int(capsule_impl_ptr)
    dealloc(
        ThinAllocation(unsafe_owned_ptr=capsule_impl_ptr).unsafe_with_layout(
            capsule_impl_layout
        )
    )
    assert_equal(capsule_impl_addr, Int(capsule_pointer))
    with assert_raises(contains="called with incorrect name"):
        _ = cpy.PyCapsule_GetPointer(capsule, "some_other_name")


def _test_memory_management_api(cpy: CPython) raises:
    var ptr = cpy.lib.call[
        "PyObject_Malloc", OptionalPointer[NoneType, MutUntrackedOrigin]
    ](64)
    assert_true(ptr)

    cpy.PyObject_Free(ptr)


def _test_common_object_structure_api(cpy: CPython) raises:
    var n = cpy.PyLong_FromSsize_t(42)
    assert_true(cpy.Py_Is(n, n))

    var dict_type = cpy.PyDict_Type()
    var d = cpy.PyDict_New()

    var d_type = cpy.Py_TYPE(d)
    assert_equal(
        PyObjectPtr(upcast_from=d_type),
        PyObjectPtr(upcast_from=dict_type),
    )


def test_with_cpython_very_high_level_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_very_high_level_api(cpython)


def test_with_cpython_reference_counting_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_reference_counting_api(cpython)


def test_with_cpython_exception_handling_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_exception_handling_api(cpython)


def test_with_cpython_threading_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_threading_api(cpython)


def test_with_cpython_importing_module_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_importing_module_api(cpython)


def test_with_cpython_object_protocol_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_object_protocol_api(cpython)


def test_with_cpython_call_protocol_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_call_protocol_api(cpython)


def test_with_cpython_number_protocol_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_number_protocol_api(cpython)


def test_with_cpython_iterator_protocol_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_iterator_protocol_api(cpython)


def test_with_cpython_type_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_type_object_api(cpython)


def test_with_cpython_integer_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_integer_object_api(cpython)


def test_with_cpython_boolean_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_boolean_object_api(cpython)


def test_with_cpython_floating_point_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_floating_point_object_api(cpython)


def test_with_cpython_unicode_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_unicode_object_api(cpython)


def test_with_cpython_tuple_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_tuple_object_api(cpython)


def test_with_cpython_list_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_list_object_api(cpython)


def test_with_cpython_dictionary_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_dictionary_object_api(cpython)


def test_with_cpython_set_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_set_object_api(cpython)


def test_with_cpython_module_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_module_object_api(cpython)


def test_with_cpython_slice_object_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_slice_object_api(cpython)


def test_with_cpython_capsule_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_capsule_api(cpython)


def test_with_cpython_memory_management_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_memory_management_api(cpython)


def test_with_cpython_common_object_structure_api() raises:
    var python = Python()
    ref cpython = python.cpython()
    _test_common_object_structure_api(cpython)


def test_pyobjectptr_write_repr_to() raises:
    var ptr = PyObjectPtr()
    var s = String()
    ptr.write_repr_to(s)
    assert_true(s.startswith("PyObjectPtr("))
    assert_true(s.endswith(")"))


def test_pyobject_write_repr_to() raises:
    var obj = PyObject()
    var s = String()
    obj.write_repr_to(s)
    assert_true(s.startswith("PyObject("))
    assert_true("object_ref_count=" in s)
    assert_true("object_type=" in s)


def test_pymoduledef_write_repr_to() raises:
    var mod = PyModuleDef("test_module")
    var s = String()
    mod.write_repr_to(s)
    assert_true(s.startswith("PyModuleDef("))
    assert_true("name=" in s)
    assert_true("size=" in s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
