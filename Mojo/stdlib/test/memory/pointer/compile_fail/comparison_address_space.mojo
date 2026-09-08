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

# Test that comparison operators reject pointers with different address spaces.


def main():
    var x = 42
    var p_generic = Pointer(to=x)
    var p_global = p_generic.unsafe_address_space_cast[.GLOBAL]()

    # CHECK: no matching method in call to '__eq__'
    _ = p_generic == p_global

    # CHECK: no matching method in call to '__ne__'
    _ = p_generic != p_global

    # CHECK: no matching method in call to '__lt__'
    _ = p_generic < p_global

    # CHECK: no matching method in call to '__le__'
    _ = p_generic <= p_global

    # CHECK: no matching method in call to '__gt__'
    _ = p_generic > p_global

    # CHECK: no matching method in call to '__ge__'
    _ = p_generic >= p_global
