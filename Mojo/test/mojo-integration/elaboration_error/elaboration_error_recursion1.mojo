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

# RUN: kgen -elaborate -elaboration-max-depth=128 --elaboration-error-verbose=no-params %s --verify-diagnostics
# RUN: not mojo build -elaboration-max-depth=128 --elaboration-error-verbose=no-params %s 2>&1 | FileCheck %s

from std.collections.string.string_span import _get_kgen_string
from std.sys import get_defined_bool


# expected-note @below {{function instantiation failed}}
# expected-note @below {{remaining errors after}}
# expected-note-re @below {{error recurses {{[0-9]+}} times}}
# expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
def self_recursion[i: Int]() -> Int:
    # expected-note @below {{call expansion failed}}
    # expected-warning @below {{self recursive call will cause an infinite loop}}
    var x = self_recursion[i + 1]()
    return x


# expected-error @below {{function instantiation failed}}
def main():
    # expected-note @below {{call expansion failed}}
    _ = self_recursion[1]()


# CHECK: self recursive call will cause an infinite loop
# CHECK: error recurses {{[0-9]+}} times
# CHECK: elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?
