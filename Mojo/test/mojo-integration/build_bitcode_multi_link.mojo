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
# Integration test for linking multiple LLVM bitcode libraries.
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the original Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit=llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Compile the other Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/another_bitcode_impl.mojo -emit=llvm -o %t_impl_other.ll
# RUN: llvm-as %t_impl_other.ll -o %t_impl_other.bc

# Step 3: Compile and run this test file that links both bitcode libraries.
# RUN: mojo --bitcode-libs=%t_impl.bc --bitcode-libs=%t_impl_other.bc %s | FileCheck %s


@extern("extern_add")
def extern_add(a: Int32, b: Int32) abi("Mojo") -> Int32:
    ...


@extern("extern_sub")
def extern_sub(a: Int32, b: Int32) abi("Mojo") -> Int32:
    ...


def main():
    var a = extern_add(10, 20)
    print("Extern add:", a)
    # CHECK: Extern add: 30

    var b = extern_sub(10, 20)
    print("Extern sub:", b)
    # # CHECK: Extern sub: -10
