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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that pointer subtraction rejects pointers with mismatched address
# spaces or pointee types, for both `offset_from()` and the `-` operator.


def main():
    var x = 42
    var y = 4.2
    var p = Pointer(to=x)
    var p_global = p.unsafe_address_space_cast[.GLOBAL]()
    var p_float = Pointer(to=y)

    # CHECK: invalid call to 'offset_from'
    _ = p.offset_from(p_global)

    # CHECK: invalid call to 'offset_from'
    _ = p.offset_from(p_float)

    var u = alloc[Int]({count = 1}).unsafe_leak()
    var u_float = alloc[Float64]({count = 1}).unsafe_leak()

    # CHECK: no matching method in call to '__sub__'
    _ = u - u_float
