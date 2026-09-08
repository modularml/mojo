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
#
# Test linking an external bitcode file.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %S/inputs/define_extern_bc_func.mojo --emit=llvm -o %t.ll
# RUN: llvm-as %t.ll -o %t.bc
# RUN: %mojo --bitcode-libs=%t.bc %s | FileCheck %s


@extern("my_add_one")
def my_add_one(x: Pointer[Int32, MutAnyOrigin]) abi("Mojo"):
    ...


def main():
    # CHECK: 3
    var two: Int32 = 2
    my_add_one(Pointer(to=two).as_unsafe_any_origin())
    print(two)
