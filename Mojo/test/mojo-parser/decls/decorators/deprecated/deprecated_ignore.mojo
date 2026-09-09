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

# Test that --ignore-deprecated suppresses the deprecation warning only for
# the named declaration, while other @deprecated declarations still warn.

# RUN: %parse-mojo-isolated --ignore-deprecated=deprecated_function --ignore-deprecated=MethodTest.deprecated_method --ignore-deprecated=DeprecatedStructIgnored --ignore-deprecated=deprecated_alias_ignored %s 2>&1 | FileCheck %s

def new_function():
    pass


@deprecated(use=new_function)
def deprecated_function():
    pass


struct MethodTest(Movable where False):
    def __init__(out self):
        pass

    def new_method(self):
        pass

    @deprecated(use=new_method)
    def deprecated_method(self):
        pass


@deprecated("This function is not ignored")
def deprecated_not_ignored():
    pass


@deprecated("DeprecatedStructIgnored is deprecated")
struct DeprecatedStructIgnored(Movable where False):
    pass


@deprecated("DeprecatedStructNotIgnored is deprecated")
struct DeprecatedStructNotIgnored(Movable where False):
    pass


@deprecated("deprecated_alias_ignored is deprecated")
comptime deprecated_alias_ignored = 1


@deprecated("deprecated_alias_not_ignored is deprecated")
comptime deprecated_alias_not_ignored = 1


def main():
    # Ignored by qualified name `deprecated_function` — no warning.
    # CHECK-NOT: warning: 'deprecated_function' is deprecated
    deprecated_function()

    # Ignored by qualified name `MethodTest.deprecated_method` — no warning.
    # CHECK-NOT: warning: 'deprecated_method' is deprecated
    var obj = MethodTest()
    obj.deprecated_method()

    # Not in the ignore list — still warns.
    # CHECK: warning: This function is not ignored
    deprecated_not_ignored()

    # Ignored by name `DeprecatedStructIgnored` — no warning.
    # CHECK-NOT: warning: DeprecatedStructIgnored is deprecated
    var _s: DeprecatedStructIgnored

    # Not in the ignore list — still warns.
    # CHECK: warning: DeprecatedStructNotIgnored is deprecated
    var _s2: DeprecatedStructNotIgnored

    # Ignored by name `deprecated_alias_ignored` — no warning.
    # CHECK-NOT: warning: deprecated_alias_ignored is deprecated
    _ = deprecated_alias_ignored

    # Not in the ignore list — still warns.
    # CHECK: warning: deprecated_alias_not_ignored is deprecated
    _ = deprecated_alias_not_ignored
