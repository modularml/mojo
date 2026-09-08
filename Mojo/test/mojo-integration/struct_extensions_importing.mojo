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

# RUN: mkdir -p %t.struct-and-extension
# RUN: mojo precompile %S/inputs/struct_and_extension_package -o %t.struct-and-extension/struct_and_extension_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.struct-and-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from struct_and_extension_package import MyType, ZInt


# CHECK-LABEL: kgen.generator @"struct_extensions_importing::test
def test():
    # Call constructor from the struct
    var obj1 = MyType()
    # CHECK: kgen.call @"struct_and_extension_package::my_type::MyType::__init__()

    # Call constructor from the extension
    var obj2 = MyType(ZInt())
    # CHECK: kgen.call @"struct_and_extension_package::my_type::extension:MyType::__init__(struct_and_extension_package::my_type::ZInt


# TODO(MOCO-522): Combine this with struct_extensions.mojo.
