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
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def _run_memset[
    dtype: DType
](ctx: DeviceContext, length: Int, val: Scalar[dtype]) raises:
    print("-")
    print("_run_memset(", length, ", ", val, ")")

    var in_host = ctx.enqueue_create_host_buffer[dtype](length)
    var out_host = ctx.enqueue_create_host_buffer[dtype](length)
    var on_dev = ctx.enqueue_create_buffer[dtype](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = Scalar[dtype](i)
        out_host[i] = Scalar[dtype](length + i)

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(on_dev)
    ctx.enqueue_memset(on_dev, val)
    on_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        assert_equal(
            out_host[i],
            val,
            String("at index ", i, " the value is ", out_host[i]),
        )


def _run_memset_cascade[
    dtype: DType
](ctx: DeviceContext, length: Int, val: Scalar[dtype]) raises:
    print("-")
    print("_run_memset_cascade(", length, ", ", val, ")")

    var buf = ctx.enqueue_create_buffer[dtype](length)
    buf.enqueue_fill(val)

    with buf.map_to_host() as buf:
        for i in range(length):
            if i < 10:
                print("buf[", i, "] = ", buf[i])
            assert_equal(
                buf[i], val, String("at index ", i, " the value is ", buf[i])
            )


def test_memset() raises:
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_memset(" + ctx.name() + "):")

    comptime one_mb = 1024 * 1024

    _run_memset[.uint8](ctx, 64, 12)
    _run_memset[.uint8](ctx, one_mb, 13)
    _run_memset_cascade[.uint8](ctx, one_mb, 14)

    _run_memset[.float16](ctx, 64, 1.75)
    _run_memset_cascade[.float16](ctx, 64, 2.5)

    _run_memset[.float32](ctx, 64, 2.3)
    _run_memset_cascade[.float32](ctx, 512, 25.125)

    _run_memset[.float64](ctx, 64, 0)
    _run_memset[.float64](ctx, 64, 1.618033988749)
    _run_memset_cascade[.int64](ctx, one_mb, 1234567890)

    print("Done.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
