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

# Element-copying operations (iteration, writing) must be rejected on spans
# viewing a non-default address space. `fill` is exempt only for register
# passable element types, which may cross an address-space boundary as a value;
# for every other element type it stays restricted to the default address
# space.

from std.memory import unsafe_stack_allocation


def _shared_tile() -> (
    Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=.SHARED,
    ]
):
    var smem = unsafe_stack_allocation[4, Float32, address_space=.SHARED]()
    return {unsafe_ptr = smem, length = 4}


# `Array[Int, 4]` is trivially copyable but not register passable, so it cannot
# cross an address-space boundary.
def _shared_aggregate_tile() -> (
    Span[
        mut=True,
        Array[Int, 4],
        MutUntrackedOrigin,
        address_space=.SHARED,
    ]
):
    var smem = unsafe_stack_allocation[
        4, Array[Int, 4], address_space=.SHARED
    ]()
    return {unsafe_ptr = smem, length = 4}


def main():
    var tile = _shared_tile()

    # CHECK: error: no matching method in call to '__iter__'
    for x in tile:
        print(x)

    # CHECK: error: no matching function in initialization
    print(String(tile))

    # CHECK: error: invalid call to 'fill': violated constraint
    _shared_aggregate_tile().fill(Array[Int, 4](fill=0))
