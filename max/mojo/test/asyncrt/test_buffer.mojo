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

from asyncrt_test_utils import create_test_device_context
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import TestSuite


def _run_badbuf(ctx: DeviceContext) raises:
    print("-")
    print("_run_badbuf()")

    comptime alloc_size = 256

    # Construct a bad buffer by adopting the host pointer.
    var host_ptr = alloc[Int8]({count = alloc_size}).unsafe_leak()
    var bad_buf = DeviceBuffer(ctx, host_ptr, alloc_size, owning=True)

    # Make a call that should succeed even with a bad buffer having been constructed.
    ctx.synchronize()

    # Free the pointer now to avoid leaking. This does not change the test.
    host_ptr.unsafe_free()

    try:
        # Release the bad buffer, which should raise an exception in Mojo instead of crashing.
        _ = bad_buf^
        # The deferred cuMemFree_v2 fires during the first synchronize (event
        # handler runs, records a pending error). The error surfaces on the
        # second synchronize via the pending error check.
        ctx.synchronize()
        ctx.synchronize()

    except e:
        print("Correctly raised an exception: ", e)
        return

    raise "Test failed: Should not reach here."


def test_buffer() raises:
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_buffer(" + ctx.name() + "):")

    _run_badbuf(ctx)

    print("Done.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
