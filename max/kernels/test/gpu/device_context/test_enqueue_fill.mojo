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

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true
from std.utils.numerics import isnan, nan


def test_enqueue_fill_host_buffer(ctx: DeviceContext) raises:
    var host_buffer = ctx.enqueue_create_host_buffer[.float64](8)
    host_buffer.enqueue_fill(0.1)
    ctx.synchronize()

    for i in range(8):
        assert_equal(
            host_buffer[i],
            0.1,
            String(t"host_buffer[{i}] should be 0.1"),
        )


def test_enqueue_fill_device_buffer_float64(ctx: DeviceContext) raises:
    """Regression test for when enqueue_fill was O(N) API calls for
    float64 values where the high and low 32-bit halves differ."""
    comptime SIZE = 1024
    var dev_buf = ctx.enqueue_create_buffer[.float64](SIZE)
    dev_buf.enqueue_fill(2.0)

    var host_buf = ctx.enqueue_create_host_buffer[.float64](SIZE)
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(SIZE):
        assert_equal(
            host_buf[i],
            2.0,
            String(t"dev_buf[{i}] should be 2.0"),
        )


def test_enqueue_fill_device_buffer_float64_large(ctx: DeviceContext) raises:
    comptime SIZE = 1 << 20  # 1M elements
    var dev_buf = ctx.enqueue_create_buffer[.float64](SIZE)
    dev_buf.enqueue_fill(2.0)

    # Spot-check a few elements rather than iterating all 1M.
    var host_buf = ctx.enqueue_create_host_buffer[.float64](SIZE)
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    assert_equal(host_buf[0], 2.0, "first element")
    assert_equal(host_buf[1], 2.0, "second element")
    assert_equal(host_buf[SIZE // 2], 2.0, "middle element")
    assert_equal(host_buf[SIZE - 1], 2.0, "last element")


def test_enqueue_fill_device_buffer_float64_small(ctx: DeviceContext) raises:
    var dev_buf = ctx.enqueue_create_buffer[.float64](1)
    dev_buf.enqueue_fill(3.14)

    var host_buf = ctx.enqueue_create_host_buffer[.float64](1)
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    assert_equal(host_buf[0], 3.14, "single element should be 3.14")


def test_enqueue_fill_device_buffer_float64_nan(ctx: DeviceContext) raises:
    """Test filling with NaN. NaN has hi=0x7FF80000, lo=0x00000000 (unequal
    halves), exercising the doubling copy path with a special float value."""
    comptime SIZE = 256
    var val = nan[.float64]()
    var dev_buf = ctx.enqueue_create_buffer[.float64](SIZE)
    dev_buf.enqueue_fill(val)

    var host_buf = ctx.enqueue_create_host_buffer[.float64](SIZE)
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(SIZE):
        assert_true(
            isnan(host_buf[i]),
            String(t"dev_buf[{i}] should be NaN"),
        )


def test_enqueue_fill_device_buffer_float32(ctx: DeviceContext) raises:
    comptime SIZE = 512
    var dev_buf = ctx.enqueue_create_buffer[.float32](SIZE)
    dev_buf.enqueue_fill(Float32(42.0))

    var host_buf = ctx.enqueue_create_host_buffer[.float32](SIZE)
    ctx.enqueue_copy(host_buf, dev_buf)
    ctx.synchronize()

    for i in range(SIZE):
        assert_equal(
            host_buf[i],
            Float32(42.0),
            String(t"dev_buf[{i}] should be 42.0"),
        )


def main() raises:
    with DeviceContext() as ctx:
        test_enqueue_fill_host_buffer(ctx)
        test_enqueue_fill_device_buffer_float64(ctx)
        test_enqueue_fill_device_buffer_float64_large(ctx)
        test_enqueue_fill_device_buffer_float64_small(ctx)
        test_enqueue_fill_device_buffer_float64_nan(ctx)
        test_enqueue_fill_device_buffer_float32(ctx)
