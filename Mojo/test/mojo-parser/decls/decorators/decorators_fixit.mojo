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

# Tests for @implicit(deprecated=True) fixit suggestions.
# Uses JSON diagnostic format to verify exact fixit positions.

# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Test: Deprecated implicit conversion fixit wraps expression with explicit call
# ===----------------------------------------------------------------------=== #


struct DeprecatedImplicit(Movable where False):
    @implicit(deprecated=True)
    def __init__(out self, value: Int):
        pass


def takes_deprecated_implicit(x: DeprecatedImplicit):
    pass


def main():
    # Variable assignment with implicit conversion.
    # The fixit suggests wrapping the literal with the explicit constructor call.
    # The fixIts are on the note diagnostic, not the warning.
    # CHECK: "fixIts":[{"end":{"column":29,"line":[[#@LINE+2]]},"start":{"column":29,"line":[[#@LINE+2]]},"text":"DeprecatedImplicit("},{"end":{"column":30,"line":[[#@LINE+2]]},"start":{"column":30,"line":[[#@LINE+2]]},"text":")"}]
    # CHECK-SAME: "message":"call 'DeprecatedImplicit(...)' explicitly"
    _: DeprecatedImplicit = 1

    # Function argument with implicit conversion.
    # The fixit suggests wrapping Int(1) with the explicit constructor call.
    # CHECK: "fixIts":[{"end":{"column":31,"line":[[#@LINE+2]]},"start":{"column":31,"line":[[#@LINE+2]]},"text":"DeprecatedImplicit("},{"end":{"column":37,"line":[[#@LINE+2]]},"start":{"column":37,"line":[[#@LINE+2]]},"text":")"}]
    # CHECK-SAME: "message":"call 'DeprecatedImplicit(...)' explicitly"
    takes_deprecated_implicit(Int(1))
