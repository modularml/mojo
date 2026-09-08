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

# RUN: mkdir -p %t.generic-struct
# RUN: mojo precompile %S/inputs/generic_struct_package -o %t.generic-struct/generic_struct_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.generic-struct %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from generic_struct_package import Container


__extension Container:
    def double(self) -> T:
        return self.value


# CHECK-LABEL: kgen.generator @"struct_extensions_generic::my_test
def my_test():
    var container = Container(42)
    # CHECK: kgen.call @"struct_extensions_generic::extension:Container::double
    var result = container.double()
