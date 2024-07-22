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

# Goal: assert that there are no pollutions between tests.
# see test_testing_environement_is_clean_A for description.

def test_populate_environement_b(inout python: Python):
    with assert_raises(contains = "name 'x' is not defined"):
        _ = python.evaluate("x")

    modules = python.import_module("sys").modules.keys()
    assert_false(
        modules.__getattr__("__contains__")("my_module").__bool__()
    )

def main():
    var python = Python()
    test_populate_environement_b(python)
