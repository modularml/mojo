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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test __get_current_function_name() magic function returns the correct
# function name in various contexts.

# Storage methods for unified closures print before enclosing functions, so
# match the nested name string out of order.
# CHECK-DAG: #StringLiteral <:string "nested_closure">

# CHECK-LABEL: lit.fn @"test_simple_function
def test_simple_function():
    # CHECK: #StringLiteral <:string "test_simple_function">
    var name = __get_current_function_name()
    _ = name


# CHECK-LABEL: lit.struct.decl @MyStruct
struct MyStruct(Movable where False):
    # should return an empty string
    # CHECK: lit.alias.decl *"comptimeField`": !lit.struct<#StringLiteral <:string "">>
    comptime comptimeField = __get_current_function_name()

    def __init__(out self):
        pass

    # CHECK-LABEL: lit.fn @"my_method
    def my_method(self):
        # CHECK: #StringLiteral <:string "my_method">
        var name = __get_current_function_name()
        _ = name

    # CHECK-LABEL: lit.fn @"my_static_method
    @staticmethod
    def my_static_method():
        # CHECK: #StringLiteral <:string "my_static_method">
        var name = __get_current_function_name()
        _ = name


# CHECK-LABEL: lit.trait.decl @MyTrait
trait MyTrait:
    # CHECK-LABEL: lit.fn @"trait_method
    def trait_method(self):
        # CHECK: #StringLiteral <:string "trait_method">
        var name = __get_current_function_name()
        _ = name


struct AnotherStruct(MyTrait, Movable where False):
    def __init__(out self):
        pass


# CHECK-LABEL: lit.extension.decl @"extension:AnotherStruct"
__extension AnotherStruct:
    # CHECK-LABEL: lit.fn @"extension_method
    def extension_method(self: AnotherStruct):
        # CHECK: #StringLiteral <:string "extension_method">
        var name = __get_current_function_name()
        _ = name


# CHECK-LABEL: lit.fn @"test_nested_function
def test_nested_function():
    # CHECK-LABEL: lit.fn *"inner
    def inner() capturing:
        # CHECK: #StringLiteral <:string "inner">
        var name = __get_current_function_name()
        _ = name

    inner()


# Storage methods print before the enclosing function, so the nested name
# string is matched via CHECK-DAG near the top of this file.

# CHECK-LABEL: lit.fn @"test_unified_clsoures
def test_unified_clsoures():
    var capture = 1

    def closure[param: Int](arg: Int) {var capture}:
        def nested_closure[param: Int](arg: Int) {var capture}:
            _ = param + arg + capture
            var name = __get_current_function_name()
            _ = name

        nested_closure[param](arg)

    closure[1](1)


def main():
    test_simple_function()
    test_nested_function()
    var s = MyStruct()
    s.my_method()
    MyStruct.my_static_method()
    var a = AnotherStruct()
    a.extension_method()
    a.trait_method()
    test_unified_clsoures()
