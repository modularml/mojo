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
# XFAIL: asan && !system-darwin
# RUN: %mojo  %s

from pathlib import _dir_of_current_file

from python import Python, PythonObject
from testing import assert_equal


fn test_import(inout python: Python) raises:
    Python.add_to_path(str(_dir_of_current_file()))
    var my_module: PythonObject = Python.import_module("my_module")
    var py_string = my_module.my_function("Hello")
    var str = String(python.__str__(py_string))
    assert_equal(str, "Formatting the string from Lit with Python: Hello")


fn test_list(inout python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_list = PythonObject([1, 2.34, "False"])
    var py_string = str(my_list)
    assert_equal(py_string, "[1, 2.34, 'False']")


fn test_tuple(inout python: Python) raises:
    var b: PythonObject = Python.import_module("builtins")
    var my_tuple = PythonObject((1, 2.34, "False"))
    var py_string = str(my_tuple)
    assert_equal(py_string, "(1, 2.34, 'False')")


fn test_call_ownership(inout python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = str(obj)
    var string = python.__str__(py_string)
    assert_equal(string, "[1, '5']")


fn test_getitem_ownership(inout python: Python) raises:
    var obj: PythonObject = [1, "5"]
    var py_string = str(obj[1])
    var string = python.__str__(py_string)
    assert_equal(string, "5")


fn test_getattr_ownership(inout python: Python) raises:
    Python.add_to_path(str(_dir_of_current_file()))
    var my_module: PythonObject = Python.import_module("my_module")
    var obj = my_module.Foo(4)
    var py_string = str(obj.bar)
    var string = python.__str__(py_string)
    assert_equal(string, "4")


def main():
    # initializing Python instance calls init_python
    var python = Python()

    test_list(python)
    test_tuple(python)
    test_call_ownership(python)
    test_getitem_ownership(python)
    test_getattr_ownership(python)
    test_import(python)
