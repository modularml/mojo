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

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import OwnedPointer
from std.testing import *


def kernel(res: MutPointer[UInt32, MutAnyOrigin]):
    res[] = 0


# Here the argument is a host pointer and not a device pointer, so we expect
# an error about an illegal memory address.
def test_function_error(ctx: DeviceContext) raises:
    # CHECK: test_function_error
    print("== test_function_error")
    try:
        var res_ptr = alloc[UInt32](1)
        var res_ptr_owned = OwnedPointer[UInt32](
            unsafe_from_raw_pointer=res_ptr
        )
        var res_host = DeviceBuffer[.uint32](
            ctx, res_ptr_owned.unsafe_ptr(), 1, owning=False
        )
        ctx.enqueue_function[kernel](res_host, block_dim=(1), grid_dim=(1))
        ctx.synchronize()

        # Don't allow early dealloc.
        _ = res_ptr_owned^
    except e:
        # This error should occur at the synchronize call as the kernel launches
        # async by default.
        # CHECK: max/kernels/test/gpu/device_context/test_function_error.mojo
        print(e)


def main() raises:
    with DeviceContext() as ctx:
        # CHECK: To get more accurate error information, set MODULAR_DEBUG=device-sync-mode
        test_function_error(ctx)
