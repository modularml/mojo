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
# RUN: %mojo %s

from pathlib import _dir_of_current_file

from python import Python, PythonObject
from testing import assert_equal, assert_raises


fn test_python_exception_import() raises:
    try:
        var sys = Python.import_module("my_uninstalled_module")
    except e:
        assert_equal(str(e), "No module named 'my_uninstalled_module'")


fn test_python_exception_getattr() raises:
    try:
        Python.add_to_path(str(_dir_of_current_file()))
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var person = my_module.Person()
            var expec_fail = person.undefined()
    except e:
        assert_equal(str(e), "'Person' object has no attribute 'undefined'")


fn test_python_exception_getitem() raises:
    try:
        var list = PythonObject([1, 2, 3])
        var should_fail = list[13]
    except e:
        assert_equal(str(e), "list index out of range")


fn test_python_exception_call() raises:
    with assert_raises(
        contains="Can't instantiate abstract class AbstractPerson"
    ):
        Python.add_to_path(str(_dir_of_current_file()))
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var person = my_module.AbstractPerson()


def main():
    test_python_exception_import()
    test_python_exception_getattr()
    test_python_exception_getitem()
    test_python_exception_call()
