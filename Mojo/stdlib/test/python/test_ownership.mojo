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

from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true, TestSuite


def _test_import(mut python: Python) raises:
    var my_module: PythonObject = Python.import_module("my_module")
    var py_string = my_module.my_function("Hello")
    var str = String(python.as_string_slice(py_string))
    assert_equal(str, "Formatting the string from Lit with Python: Hello")


def _test_list(mut python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_list: PythonObject = [1, 2.34, "False"]
    var py_string = String(my_list)
    assert_equal(py_string, "[1, 2.34, 'False']")


def _test_tuple(mut python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_tuple = Python.tuple(1, 2.34, "False")
    var py_string = String(my_tuple)
    assert_equal(py_string, "(1, 2.34, 'False')")


def _test_call_ownership(mut python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = String(obj)
    var string = python.as_string_slice(PythonObject(py_string))
    assert_true(string == "[1, '5']")


def _test_getitem_ownership(mut python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = String(obj[1])
    var string = python.as_string_slice(PythonObject(py_string))
    assert_true(string == "5")


def _test_getattr_ownership(mut python: Python) raises:
    var my_module: PythonObject = Python.import_module("my_module")
    var obj = my_module.Foo(4)
    var py_string = String(obj.bar)
    var string = python.as_string_slice(PythonObject(py_string))
    assert_true(string == "4")


def test_with_python_list() raises:
    var python = Python()
    _test_list(python)


def test_with_python_tuple() raises:
    var python = Python()
    _test_tuple(python)


def test_with_python_call_ownership() raises:
    var python = Python()
    _test_call_ownership(python)


def test_with_python_getitem_ownership() raises:
    var python = Python()
    _test_getitem_ownership(python)


def test_with_python_getattr_ownership() raises:
    var python = Python()
    _test_getattr_ownership(python)


def test_with_python_import() raises:
    var python = Python()
    _test_import(python)


def _refcount(obj: PythonObject) raises -> Int:
    var sys = Python.import_module("sys")
    return Int(py=sys.getrefcount(obj))


def test_call_does_not_leak_positional_args() raises:
    var id_fn = Python.import_module("builtins").id
    var obj = Python.evaluate("object()")
    var before = _refcount(obj)
    for _ in range(4):
        _ = id_fn(obj)
    assert_equal(_refcount(obj), before)


def test_setitem_does_not_leak_key_or_value() raises:
    var d = Python.evaluate("{}")
    var key = Python.evaluate("object()")
    var value = Python.evaluate("object()")
    d[key] = value
    var key_before = _refcount(key)
    var value_before = _refcount(value)
    for _ in range(4):
        d[key] = value
    assert_equal(_refcount(key), key_before)
    assert_equal(_refcount(value), value_before)
    # Keep the dict alive through the refcount measurements above.
    _ = d


def test_setattr_does_not_leak_value() raises:
    var obj = Python.evaluate("type('Obj', (), {})()")
    var value = Python.evaluate("object()")
    obj.__setattr__("attr", value)
    var before = _refcount(value)
    for _ in range(4):
        obj.__setattr__("attr", value)
    assert_equal(_refcount(value), before)
    # Keep the attribute holder alive through the refcount measurement above.
    _ = obj


def test_set_literal_does_not_leak_elements() raises:
    var value = Python.evaluate("object()")
    var before = _refcount(value)
    for _ in range(4):
        var s: PythonObject = {value}
        _ = s
    assert_equal(_refcount(value), before)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
