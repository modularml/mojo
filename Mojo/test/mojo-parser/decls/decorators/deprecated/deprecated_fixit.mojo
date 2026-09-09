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

# Tests for @deprecated(use=...) fixit suggestions with correct location.
# Uses JSON diagnostic format to verify exact fixit column positions.

# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Test: Function call fixit starts at the function name
# ===----------------------------------------------------------------------=== #


def new_function():
    pass


@deprecated(use=new_function)
def deprecated_function():
    pass


# ===----------------------------------------------------------------------=== #
# Test: Method call fixit should start at the method name, not the receiver
# ===----------------------------------------------------------------------=== #


struct MethodTest(Movable where False):
    def __init__(out self):
        pass

    def new_method(self):
        pass

    @deprecated(use=new_method)
    def deprecated_method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Test: Static method call fixit
# ===----------------------------------------------------------------------=== #


struct StaticMethodTest(Movable where False):
    @staticmethod
    def brand_new_static():
        pass

    @staticmethod
    @deprecated(use=brand_new_static)
    def old_static():
        pass


# ===----------------------------------------------------------------------=== #
# Test: Operator syntax should NOT emit fixit
# ===----------------------------------------------------------------------=== #


struct OpTest(Movable where False):
    def __init__(out self):
        pass

    def add(self, other: Self) -> Self:
        return Self()

    @deprecated(use=add)
    def __add__(self, other: Self) -> Self:
        return Self()


# ===----------------------------------------------------------------------=== #
# Test: Plain @deprecated without use= should NOT emit fixit
# ===----------------------------------------------------------------------=== #


@deprecated("This function is deprecated")
def deprecated_no_use():
    pass


# ===----------------------------------------------------------------------=== #
# Test: Function passed as parameter (cast) should emit fixit
# ===----------------------------------------------------------------------=== #


def new_func_for_cast():
    pass


@deprecated(use=new_func_for_cast)
def deprecated_func_for_cast():
    pass


def takes_func(f: def() thin -> None):
    f()


struct MethodCastTest(Movable where False):
    def __init__(out self):
        pass

    def new_method_for_cast(self):
        pass

    @deprecated(use=new_method_for_cast)
    def deprecated_method_for_cast(self):
        pass


def takes_method(m: def(MethodCastTest) thin -> None):
    pass


def main():
    # Direct function call: fixit covers "deprecated_function" (columns 5-24)
    # CHECK: "fixIts":[{"end":{"column":24,"line":[[#@LINE+1]]},"start":{"column":5,"line":[[#@LINE+1]]},"text":"new_function"}]
    deprecated_function()

    # If the fixit incorrectly included the receiver, start would be column 5.
    # CHECK: "fixIts":[{"end":{"column":26,"line":[[#@LINE+2]]},"start":{"column":9,"line":[[#@LINE+2]]},"text":"new_method"}]
    var obj = MethodTest()
    obj.deprecated_method()

    # Static method call: fixit covers "old_static" (columns 22-32)
    # CHECK: "fixIts":[{"end":{"column":32,"line":[[#@LINE+1]]},"start":{"column":22,"line":[[#@LINE+1]]},"text":"brand_new_static"}]
    StaticMethodTest.old_static()

    # Operator syntax: warning but NO fixit (can't replace operator syntax)
    # The fixIts array is empty because operator syntax doesn't support fixits.
    # CHECK: "fixIts":[]{{.*}}"message":"'__add__' is deprecated
    var a = OpTest()
    var b = OpTest()
    _ = a + b

    # Plain @deprecated: warning but NO fixit (no replacement specified)
    # The fixIts array is empty because no replacement was specified.
    # CHECK: "fixIts":[]{{.*}}"message":"This function is deprecated"
    deprecated_no_use()

    # Function passed as parameter: fixit covers "deprecated_func_for_cast"
    # CHECK: "fixIts":[{"end":{"column":40,"line":[[#@LINE+1]]},"start":{"column":16,"line":[[#@LINE+1]]},"text":"new_func_for_cast"}]
    takes_func(deprecated_func_for_cast)

    # Method reference passed as parameter: fixit covers "deprecated_method_for_cast"
    # CHECK: "fixIts":[{"end":{"column":59,"line":[[#@LINE+1]]},"start":{"column":33,"line":[[#@LINE+1]]},"text":"new_method_for_cast"}]
    takes_method(MethodCastTest.deprecated_method_for_cast)
