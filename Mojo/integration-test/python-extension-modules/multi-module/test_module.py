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

# Imports from 'mojo_module.mojo'
import mojo_module_a  # type: ignore[import-not-found]
import mojo_module_b  # type: ignore[import-not-found]

"""
This test suite validates the multi-module Python extension functionality in Mojo.

The test verifies:
1. Module initialization - Both mojo_module_a and mojo_module_b can be imported successfully
2. Cross-module object sharing - A TestStruct instance created in mojo_module_a can be used
   by functions in mojo_module_b
3. Method functionality - The set_a() and set_b() methods work correctly
4. Cross-module function calls - Functions from mojo_module_b can operate on objects
   created by mojo_module_a
5. Data integrity - The shared TestStruct maintains its state across module boundaries

This demonstrates that Mojo's Python extension system supports proper module separation
while allowing shared data structures and cross-module function calls.
"""


def test_pyinit() -> None:
    assert mojo_module_a
    assert mojo_module_b


def test_both_modules() -> None:
    s = mojo_module_a.TestStruct()
    s.set_a(1)
    s.set_b(2)
    mojo_module_b.print_test_struct(s)
    v = mojo_module_b.add(s)
    assert v == 3
