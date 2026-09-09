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
"""Runtime tests for time functions on GPU.

This test verifies that time.sleep() and perf_counter_ns() work correctly on
GPUs, including:
1. Durations longer than 1ms (requires looping since nanosleep has 1ms max)
2. Durations shorter than 1ms (single nanosleep call)
3. Edge cases like zero/negative durations
4. perf_counter_ns() returns nanoseconds (not cycles)
"""

from std.time import global_perf_counter_ns, perf_counter_ns, sleep

from max.gpu.host import DeviceContext
from std.testing import assert_true


def sleep_kernel_100ms(result_ptr: MutPointer[UInt64, MutUntrackedOrigin]):
    """GPU kernel that sleeps for 100ms and stores elapsed time."""
    # Use global_perf_counter_ns() which returns actual nanoseconds on GPUs.
    var start = global_perf_counter_ns()
    sleep(0.1)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


def sleep_kernel_500us(result_ptr: MutPointer[UInt64, MutUntrackedOrigin]):
    """GPU kernel that sleeps for 500 microseconds (sub-1ms)."""
    var start = global_perf_counter_ns()
    sleep(0.0005)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


def sleep_kernel_zero(result_ptr: MutPointer[UInt64, MutUntrackedOrigin]):
    """GPU kernel that sleeps for zero duration (should return immediately)."""
    var start = global_perf_counter_ns()
    sleep(0.0)
    var end = global_perf_counter_ns()
    result_ptr[] = end - start


def test_sleep_100ms(ctx: DeviceContext) raises:
    """Test 100ms sleep (requires loop since nanosleep max is 1ms)."""
    var result_host = ctx.enqueue_create_host_buffer[.uint64](1)
    var result_device = ctx.enqueue_create_buffer[.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_100ms](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_ms_gpu = Float64(result_host[0]) / 1_000_000.0

    # The implementation loops until elapsed >= total_ns using global_perf_counter_ns(),
    # so the measured time should be at least 100ms. NVIDIA's nanosleep can oversleep
    # up to 2x per call, so we allow up to 300ms for the upper bound.
    assert_true(
        elapsed_ms_gpu >= 95.0 and elapsed_ms_gpu <= 300.0,
        "100ms sleep outside expected bounds [95ms, 300ms]",
    )


def test_sleep_500us(ctx: DeviceContext) raises:
    """Test 500 microsecond sleep (sub-1ms, single nanosleep call)."""
    var result_host = ctx.enqueue_create_host_buffer[.uint64](1)
    var result_device = ctx.enqueue_create_buffer[.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_500us](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_us_gpu = Float64(result_host[0]) / 1_000.0

    # The implementation loops until elapsed >= total_ns using global_perf_counter_ns(),
    # so the measured time should be at least 500us. Allow up to 3ms for oversleep.
    assert_true(
        elapsed_us_gpu >= 450.0 and elapsed_us_gpu <= 3000.0,
        "500us sleep outside expected bounds [450us, 3ms]",
    )


def test_sleep_zero(ctx: DeviceContext) raises:
    """Test zero duration sleep (should return immediately)."""
    var result_host = ctx.enqueue_create_host_buffer[.uint64](1)
    var result_device = ctx.enqueue_create_buffer[.uint64](1)

    result_host[0] = 0
    ctx.enqueue_function[sleep_kernel_zero](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var elapsed_us_gpu = Float64(result_host[0]) / 1_000.0

    # Zero sleep should complete very quickly (< 1ms).
    assert_true(
        elapsed_us_gpu < 1000.0,
        "Zero sleep took too long (> 1ms)",
    )


def perf_counter_kernel(
    result_ptr: MutPointer[UInt64, MutUntrackedOrigin],
):
    """GPU kernel that measures a single sleep with both timer functions."""
    # Measure the SAME sleep interval with both counters. If both return
    # nanoseconds, the deltas should be nearly identical. If perf_counter_ns
    # returned cycles, the ratio would equal the GPU clock frequency
    # (typically 1-2 GHz), making the values diverge significantly.
    var pc_start = perf_counter_ns()
    var gpc_start = global_perf_counter_ns()
    sleep(0.01)
    var pc_end = perf_counter_ns()
    var gpc_end = global_perf_counter_ns()
    result_ptr[0] = UInt64(pc_end - pc_start)
    result_ptr[1] = gpc_end - gpc_start


def test_perf_counter_ns(ctx: DeviceContext) raises:
    """Test that perf_counter_ns() returns nanoseconds on GPU, not cycles."""
    var result_host = ctx.enqueue_create_host_buffer[.uint64](2)
    var result_device = ctx.enqueue_create_buffer[.uint64](2)

    result_host[0] = 0
    result_host[1] = 0
    ctx.enqueue_function[perf_counter_kernel](
        result_device, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var pc_ms = Float64(result_host[0]) / 1_000_000.0
    var gpc_ms = Float64(result_host[1]) / 1_000_000.0

    # Both should measure ~10ms. Allow [9ms, 15ms] for oversleep.
    assert_true(
        pc_ms >= 9.0 and pc_ms <= 15.0,
        "perf_counter_ns() 10ms sleep outside expected bounds [9ms, 15ms]",
    )
    assert_true(
        gpc_ms >= 9.0 and gpc_ms <= 15.0,
        (
            "global_perf_counter_ns() 10ms sleep outside expected bounds"
            " [9ms, 15ms]"
        ),
    )

    # Both counters measure the same interval, so they should agree closely.
    # Allow 20% tolerance for the small gap between reading the two clocks.
    # If perf_counter_ns was returning cycles instead of nanoseconds, the
    # ratio would be ~clock_freq_GHz (typically 1.5-2.0x off).
    var ratio = pc_ms / gpc_ms if gpc_ms > 0 else 0.0
    assert_true(
        ratio > 0.8 and ratio < 1.2,
        "perf_counter_ns() and global_perf_counter_ns() disagree by > 20%",
    )


def main() raises:
    with DeviceContext() as ctx:
        test_sleep_100ms(ctx)
        test_sleep_500us(ctx)
        test_sleep_zero(ctx)
        test_perf_counter_ns(ctx)
