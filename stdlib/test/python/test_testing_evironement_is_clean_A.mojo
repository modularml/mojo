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
# Invariant: test A is performed before test B
# The tests are in pairs:
#   - test_testing_environement_is_clean_A
#     (modify the environement)
#   - test_testing_environement_is_clean_B
#     (check that modifications are not reflected there)

def test_populate_environement_a(inout python: Python):
    python.eval("x = 123")
    assert_equal(int(python.evaluate("x")), 123)

    _ = python.import_module("my_module")
    modules = python.import_module("sys").modules.keys()
    assert_true(
        modules.__getattr__("__contains__")("my_module").__bool__()
    )

def main():
    var python = Python()
    test_populate_environement_a(python)
