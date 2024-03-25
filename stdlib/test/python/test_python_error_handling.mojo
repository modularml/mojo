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
# RUN: %mojo -D TEST_DIR=%S %s | FileCheck %s

from sys.param_env import env_get_string

from python import Python
from python._cpython import CPython, PyObjectPtr
from python.object import PythonObject

alias TEST_DIR = env_get_string["TEST_DIR"]()


fn test_python_exception_import() raises:
    try:
        var sys = Python.import_module("my_uninstalled_module")
    except e:
        print(e)


fn test_python_exception_getattr() raises:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var person = my_module.Person()
            var expec_fail = person.undefined()
    except e:
        print(e)


fn test_python_exception_getitem() raises:
    try:
        var list = PythonObject([1, 2, 3])
        var should_fail = list[13]
    except e:
        print(e)


fn test_python_exception_call() raises:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var person = my_module.AbstractPerson()
    except e:
        print(e)


def main():
    # CHECK: No module named 'my_uninstalled_module'
    test_python_exception_import()
    # CHECK: 'Person' object has no attribute 'undefined'
    test_python_exception_getattr()
    # CHECK: list index out of range
    test_python_exception_getitem()
    # CHECK: Can't instantiate abstract class AbstractPerson with abstract method{{s?}} method
    test_python_exception_call()
