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
from std.testing import assert_equal, assert_raises, TestSuite


def test_python_exception_import() raises:
    try:
        var _sys = Python.import_module("my_uninstalled_module")
    except e:
        assert_equal(String(e), "No module named 'my_uninstalled_module'")


def test_python_exception_getattr() raises:
    try:
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var person = my_module.Person()
            var _expect_fail = person.undefined()
    except e:
        assert_equal(String(e), "'Person' object has no attribute 'undefined'")


def test_python_exception_getitem() raises:
    try:
        var list: PythonObject = [1, 2, 3]
        var _should_fail = list[13]
    except e:
        assert_equal(String(e), "list index out of range")


def test_python_exception_call() raises:
    with assert_raises(
        contains="Can't instantiate abstract class AbstractPerson"
    ):
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            _ = my_module.AbstractPerson()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
