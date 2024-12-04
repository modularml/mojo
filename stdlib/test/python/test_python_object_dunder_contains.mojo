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

from python import Python, PythonObject
from testing import assert_equal, assert_false, assert_raises, assert_true
from collections import Dict


def test_contains_dunder(mut python: Python):
    with assert_raises(contains="'int' object is not iterable"):
        var z = PythonObject(0)
        _ = 5 in z

    var x = PythonObject([1.1, 2.2])
    assert_true(1.1 in x)
    assert_false(3.3 in x)

    x = PythonObject(["Hello", "World"])
    assert_true("World" in x)

    x = PythonObject((1.5, 2))
    assert_true(1.5 in x)
    assert_false(3.5 in x)

    var y = Dict[PythonObject, PythonObject]()
    y["A"] = "A"
    y["B"] = 5
    x = PythonObject(y)
    assert_true("A" in x)
    assert_false("C" in x)
    assert_true("B" in x)

    # tests with python modules:
    module = python.import_module(
        "module_for_test_python_object_dunder_contains"
    )

    x = module.Class_no_iterable_but_contains()
    assert_true(123 in x)

    x = module.Class_no_iterable_no_contains()
    with assert_raises(
        contains="'Class_no_iterable_no_contains' object is not iterable"
    ):
        _ = 123 in x

    x = module.Class_iterable_no_contains()
    assert_true(123 in x)
    assert_true(456 in x)
    assert_false(234 in x)
    x.data.append(234)
    assert_true(234 in x)


def main():
    # initializing Python instance calls init_python
    var python = Python()
    test_contains_dunder(python)
