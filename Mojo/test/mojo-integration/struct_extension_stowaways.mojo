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

# RUN: mkdir -p %t.transitive-extension
# RUN: mojo precompile %S/inputs/transitive_extension_package -o %t.transitive-extension/transitive_extension_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.transitive-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

# Note how we're importing IntConfig, not MyStruct here.
# Yet we also want to grab stowaway.mojo's extension for MyStruct.
from transitive_extension_package.base import MyStruct
from transitive_extension_package.stowaway import IntConfig


# CHECK-LABEL: kgen.generator @"struct_extension_stowaways::test
def test():
    var obj = MyStruct(0)

    # Call method from the extension in the stowaway module, even though we
    # didn't explicitly import it by name (we imported IntConfig instead).
    # The extension should have hitchhiked in the IntConfig import to make
    # itself known to the current file so we can call it here.
    # TODO(MOCO-522): Arcana reference here
    obj.intermediate_method()
    # CHECK: kgen.call @"transitive_extension_package::stowaway::extension:MyStruct::intermediate_method
