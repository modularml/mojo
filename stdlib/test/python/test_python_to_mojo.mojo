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
# RUN: %mojo-no-debug %s | FileCheck %s

from python import Python, PythonObject

from testing import assert_equal


fn test_string_to_python_to_mojo(inout python: Python) raises:
    var py_string = PythonObject("mojo")
    var py_string_capitalized = py_string.capitalize()

    var cap_mojo_string = str(py_string_capitalized)
    assert_equal(cap_mojo_string, "Mojo")


fn test_range() raises:
    var array_size: PythonObject = 2
    # CHECK: 0
    # CHECK-NEXT: 1
    for i in range(array_size):
        print(i)

    var start: PythonObject = 0
    var end: PythonObject = 4
    # CHECK: 0
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    for i in range(start, end):
        print(i)

    var start2: PythonObject = 5
    var end2: PythonObject = 10
    var step: PythonObject = 2
    # CHECK: 5
    # CHECK-NEXT: 7
    # CHECK-NEXT: 9
    for i in range(start2, end2, step):
        print(i)


fn test_python_to_string() raises:
    # CHECK: environ({
    var os = Python.import_module("os")
    print(os.environ)


fn main():
    var python = Python()
    try:
        test_string_to_python_to_mojo(python)
        test_range()
        test_python_to_string()
    except:
        pass
