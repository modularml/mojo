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

# Test that --warn-on-unstable-apis emits warnings when unstable magic functions
# are used. Some magic functions (origin_of, type_of, conforms_to,
# __functions_in_module) are considered stable and should not warn.

# RUN: %parse-mojo-isolated -warn-on-unstable-apis -verify-diagnostics %s


def test_stable_magic_functions():
    """Stable magic functions should NOT trigger warnings."""
    var x = 42
    # type_of, origin_of, conforms_to, __functions_in_module are stable.
    comptime t = type_of(x)


def test_unstable_get_current_function_name():
    """Using __get_current_function_name should trigger an unstable warning."""
    # expected-warning @+1 {{use of unstable function '__get_current_function_name'}}
    comptime name = __get_current_function_name()
