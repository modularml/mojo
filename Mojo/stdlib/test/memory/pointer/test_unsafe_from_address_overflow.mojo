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
# `unsafe_from_address` aborts (under `-D ASSERT=all`) when the address does
# not fit in the pointer's address space. This only applies to address spaces
# whose pointers are narrower than `Int`, so it is exercised on a GPU where
# `SHARED` pointers are 32-bit while `Int` is 64-bit.
#
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.memory.address_space import AddressSpace


def _kernel(addr: Int64):
    # `addr` is a runtime value larger than a 32-bit `SHARED` pointer can hold.
    # `Int` is not device-passable, so it arrives as a fixed-width `Int64`.
    var p = Pointer[Int, MutAnyOrigin, address_space=.SHARED](
        unsafe_from_address=Int(addr)
    )
    _ = p


# CHECK-LABEL: == test_fail
def main() raises:
    print("== test_fail")

    with DeviceContext() as ctx:
        # 2**33 fits in `Int` (64-bit) but not a 32-bit `SHARED` pointer.
        # CHECK: Assert Error: address 8589934592 does not fit in this pointer's address space
        ctx.enqueue_function[_kernel](Int64(1 << 33), grid_dim=1, block_dim=1)
        ctx.synchronize()

    # CHECK-NOT: is never reached
    print("is never reached")
