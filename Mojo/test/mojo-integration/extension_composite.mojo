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

# RUN: mkdir -p %t.extension-in-trait-module
# RUN: mojo precompile %S/inputs/simple_struct_package -o %t.extension-in-trait-module/simple_struct_package.mojoc
# RUN: mojo precompile %S/inputs/trait_and_extension_package -I %t.extension-in-trait-module -o %t.extension-in-trait-module/trait_and_extension_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.extension-in-trait-module %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from trait_and_extension_package import MyTrait
from simple_struct_package import MyStruct


def use_trait[T: MyTrait & Copyable](value: T):
    pass


# CHECK-LABEL: kgen.generator @"extension_composite::test
def test(var s: MyStruct):
    # CHECK: kgen.call tail @"extension_composite::use_trait
    use_trait(s^)
