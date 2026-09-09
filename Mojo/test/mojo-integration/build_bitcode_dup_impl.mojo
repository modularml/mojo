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
# Integration test for linking multiple LLVM bitcode libraries that conflicts.
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the original Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit=llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Compile the alternative Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl_alt.mojo -emit=llvm -o %t_impl_alt.ll
# RUN: llvm-as %t_impl_alt.ll -o %t_impl_alt.bc

# Step 3: Compile and run this test file that links both packages.
# RUN: mojo --bitcode-libs=%t_impl.bc --bitcode-libs=%t_impl_alt.bc %s | FileCheck %s


@extern("extern_add")
def extern_add(a: Int32, b: Int32) abi("Mojo") -> Int32:
    ...


def main():
    var a = extern_add(10, 20)
    print("Extern add:", a)
    # CHECK: Extern add:
