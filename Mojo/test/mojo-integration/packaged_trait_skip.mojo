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

# Test using a child trait from a package without using the parent trait.

# RUN: mkdir -p %t.packaged-trait-skip
# RUN: mojo precompile %S/inputs/test_package -o %t.packaged-trait-skip/test_package_trait.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.packaged-trait-skip %s --kgen-print-inline-type-values | FileCheck %s

from test_package_trait.module2 import *


# CHECK: lit.struct.decl @MyType({{.*}}PackageChildTrait
struct MyType(PackageChildTrait):
    def method(self):
        pass

    def method2(self):
        pass
