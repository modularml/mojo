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

# ===----------------------------------------------------------------------=== #
#
# Integration test for packaging LLVM bitcode libraries and using them
#
# This test demonstrates the full workflow:
# 1. Compile a Mojo file with exported functions to LLVM bitcode
# 2. Package a Mojo package with extern declarations and the bitcode library
# 3. Import the package in the main program and use the extern functions
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit=llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Package the bitcode_package with the bitcode library
# RUN: mojo precompile %S/inputs/bitcode_package --bitcode-libs=%t_impl.bc -o %S/bitcode_package.mojoc

# Step 3: Verify the package was created and contains bitcode modules
# RUN: kgen-opt %S/bitcode_package.mojoc | FileCheck %s --check-prefix=CHECK-PACKAGE
# CHECK-PACKAGE: lit.package
# CHECK-PACKAGE-SAME: externLLVMBitcodeModules = #lit<dense_resource_elements_array[dense_resource<[[LLVM_BITCODE_NAME:llvm_bitcode_[[:alnum:]]+]]
# CHECK-PACKAGE: dialect_resources:
# CHECK-PACKAGE: [[LLVM_BITCODE_NAME]]

# Step 4: Compile and run this test file that imports and uses the package
# RUN: mojo %s -I %S | FileCheck %s --check-prefix=CHECK-OUTPUT

# Main program that imports and uses the packaged extern functions
from bitcode_package import double_add, extern_add, extern_multiply


def main():
    # COM: Test a method that directly uses extern functions
    var doubled = double_add(5, 3)
    # COM: Should compute: extern_add(extern_add(5, 3), extern_add(5, 3)) = extern_add(8, 8) = 16
    # CHECK-OUTPUT: Doubled: 16
    print("Doubled:", doubled)

    var a = extern_add(10, 20)
    var b = extern_multiply(5, 6)
    # CHECK-OUTPUT: Direct calls: 30 30
    print("Direct calls:", a, b)
