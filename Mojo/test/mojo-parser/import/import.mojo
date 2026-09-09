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

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S/inputs %s | FileCheck %s

# Test import of a module, and we properly allow import of an imported decl.

from imported_module import *

from test_package.module import top_level_alias


# CHECK-LABEL: lit.fn @"foo
def foo():
    var t = top_level_alias


# CHECK-NOT: lit.alias.decl

# Test that a package importing another package at
# the same level is properly supported.

from test_package_user import using_test_package

# CHECK-LABEL: lit.fn @"test_package_user


def test_package_user():
    using_test_package()


# Test import of a package.

from test_package.module import function
from test_package.test_nested_package.module import nested_function
from test_package import *

# CHECK-LABEL: lit.fn @"test_function_calls
# CHECK:  lit.call tail @test_package::@module::@"function()"
# CHECK:  lit.call tail @test_package::@test_nested_package::@module::@"nested_function()"
# CHECK:  lit.call tail @test_package::@__init__::@"method_defined_in_init()"()

# CHECK-LABEL: lit.package @test_package
# CHECK:  lit.file_module @module
# CHECK:    lit.fn @"function()"
# CHECK:      lit.fn @"call_nested_function()"
# CHECK:        lit.call tail @test_package::@test_nested_package::@module::@"nested_function()"
# CHECK:  lit.package @test_nested_package
# CHECK:    lit.file_module @module
# CHECK:      lit.fn @"nested_function()"


def test_function_calls():
    function()
    nested_function()
    method_defined_in_init()
