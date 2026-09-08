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

# RUN: mkdir -p %t.struct-and-conforming-extension
# RUN: mojo precompile %S/inputs/struct_and_conforming_extension_package -o %t.struct-and-conforming-extension/struct_and_conforming_extension_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.struct-and-conforming-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

# This test verifies that extensions with trait conformances survive DCE and
# are properly processed during LowerLIT when imported from a package.

from struct_and_conforming_extension_package import MyStruct, Convertible


def use_convertible[T: Convertible](x: T) -> Int:
    return x.convert()


# CHECK-LABEL: kgen.generator @"extension_conformance_dce::test
def test() -> Int:
    var s = MyStruct(42)

    # This uses the alias defined in the extension, which requires the
    # extension to survive DCE during importing.
    comptime t = MyStruct.ExtensionAlias

    # This call requires the extension conformance to be pulled in from the
    # precompiled file. Without it, this would fail in the elaborator because
    # it can't find the conformance.
    # CHECK: kgen.call @"extension_conformance_dce::use_convertible[::AnyType & struct_and_conforming_extension_package::my_struct::Convertible]
    return use_convertible(s)
