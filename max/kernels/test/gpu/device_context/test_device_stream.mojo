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

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream
from std.testing import assert_equal, assert_true


# Simple kernel for testing stream execution
def simple_kernel(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    multiplier: Float32,
):
    """Simple kernel that multiplies input by a multiplier."""
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = input[tid] * multiplier


def test_stream_priority_range(ctx: DeviceContext) raises:
    print("Test that stream_priority_range() returns a valid priority range.")
    var priority_range = ctx.stream_priority_range()

    # The range should be valid (least >= greatest)
    # largest priority is generally -5, while least is 0.
    assert_true(
        priority_range.least >= priority_range.greatest,
        "Priority range should have least >= greatest",
    )


def test_create_stream_default(ctx: DeviceContext) raises:
    print("Test creating a stream with default parameters (blocking=True).")
    var stream = ctx.create_stream()

    # Basic operations should work with the stream
    stream.synchronize()


def test_create_stream_with_priority(ctx: DeviceContext) raises:
    print("Test creating streams with different priority values.")
    var priority_range = ctx.stream_priority_range()

    comptime length = 256
    comptime multiplier = Float32(2.5)

    # Create host and device buffers
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    var output_host_low = ctx.enqueue_create_host_buffer[.float32](length)
    var output_host_high = ctx.enqueue_create_host_buffer[.float32](length)

    # Initialize input data
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    var output_device_low = ctx.enqueue_create_buffer[.float32](length)
    var output_device_high = ctx.enqueue_create_buffer[.float32](length)

    # Copy input data to device
    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Test with lowest priority stream
    var low_priority_stream = ctx.create_stream(priority=priority_range.least)
    var func = ctx.compile_function[simple_kernel]()
    low_priority_stream.enqueue_function(
        func,
        input_device,
        output_device_low,
        Int32(length),
        multiplier,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )
    low_priority_stream.synchronize()

    # Test with highest priority stream
    var high_priority_stream = ctx.create_stream(
        priority=priority_range.greatest
    )
    high_priority_stream.enqueue_function(
        func,
        input_device,
        output_device_high,
        Int32(length),
        multiplier,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )
    high_priority_stream.synchronize()

    # Copy results back and verify
    ctx.enqueue_copy(output_host_low, output_device_low)
    ctx.enqueue_copy(output_host_high, output_device_high)
    ctx.synchronize()

    # Verify results from both streams
    for i in range(min(10, length)):  # Check first 10 elements
        var expected = Float32(i) * multiplier
        assert_equal(output_host_low[i], expected)
        assert_equal(output_host_high[i], expected)

    # Test with middle priority (if range allows)
    if priority_range.least < priority_range.greatest:
        var mid_priority = (priority_range.least + priority_range.greatest) // 2
        var mid_priority_stream = ctx.create_stream(priority=mid_priority)
        var output_device_mid = ctx.enqueue_create_buffer[.float32](length)
        mid_priority_stream.enqueue_function(
            func,
            input_device,
            output_device_mid,
            Int32(length),
            multiplier,
            grid_dim=ceildiv(length, 32),
            block_dim=32,
        )
        mid_priority_stream.synchronize()


def test_multiple_priority_streams(ctx: DeviceContext) raises:
    print(
        "Test creating multiple streams with different priorities and"
        " concurrent kernel execution."
    )
    var priority_range = ctx.stream_priority_range()

    comptime length = 512
    comptime num_kernels = 4

    # Create input data
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Create multiple streams with different priorities
    var streams = List[DeviceStream]()
    var output_devices = List[DeviceBuffer[.float32]]()
    var multipliers = List[Float32]()

    # Create streams with lowest and highest priority
    streams.append(ctx.create_stream(priority=priority_range.least))
    streams.append(ctx.create_stream(priority=priority_range.greatest))
    output_devices.append(ctx.enqueue_create_buffer[.float32](length))
    output_devices.append(ctx.enqueue_create_buffer[.float32](length))
    multipliers.append(Float32(1.5))
    multipliers.append(Float32(3.0))

    # If we have a range of priorities, create some intermediate ones
    if priority_range.greatest > priority_range.least:
        var step = max(1, (priority_range.greatest - priority_range.least) // 4)
        var current_priority = priority_range.least + step
        var multiplier_val = Float32(2.0)
        while (
            current_priority < priority_range.greatest
            and len(streams) < num_kernels
        ):
            streams.append(ctx.create_stream(priority=current_priority))
            output_devices.append(ctx.enqueue_create_buffer[.float32](length))
            multipliers.append(multiplier_val)
            current_priority += step
            multiplier_val += Float32(0.5)

    var func = ctx.compile_function[simple_kernel]()

    # Launch kernels concurrently on all streams
    for i in range(len(streams)):
        streams[i].enqueue_function(
            func,
            input_device,
            output_devices[i],
            Int32(length),
            multipliers[i],
            grid_dim=ceildiv(length, 32),
            block_dim=32,
        )

    # Synchronize all streams
    for i in range(len(streams)):
        streams[i].synchronize()

    # Verify results from each stream
    for stream_idx in range(len(streams)):
        var output_host = ctx.enqueue_create_host_buffer[.float32](length)
        ctx.enqueue_copy(output_host, output_devices[stream_idx])
        ctx.synchronize()

        # Check first few elements
        for i in range(min(5, length)):
            var expected = Float32(i) * multipliers[stream_idx]
            assert_equal(output_host[i], expected)


def test_concurrent_priority_streams(ctx: DeviceContext) raises:
    print("Test concurrent execution on streams with different priorities.")
    var priority_range = ctx.stream_priority_range()

    # Skip if we don't have different priorities available
    if priority_range.least == priority_range.greatest:
        print(
            "Single priority level available, skipping concurrent priority test"
        )
        return

    comptime length = 1024
    comptime iterations = 10

    # Create input data
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Create high and low priority streams
    var high_priority_stream = ctx.create_stream(
        priority=priority_range.greatest
    )
    var low_priority_stream = ctx.create_stream(priority=priority_range.least)

    var high_output_device = ctx.enqueue_create_buffer[.float32](length)
    var low_output_device = ctx.enqueue_create_buffer[.float32](length)

    var func = ctx.compile_function[simple_kernel]()
    # Launch multiple kernels on both streams to test priority behavior
    for i in range(iterations):
        # Launch on low priority stream first
        low_priority_stream.enqueue_function(
            func,
            input_device,
            low_output_device,
            Int32(length),
            Float32(1.0 + Float64(i)),
            grid_dim=((length + 63) // 64),
            block_dim=64,
        )

        # Then immediately launch on high priority stream
        high_priority_stream.enqueue_function(
            func,
            input_device,
            high_output_device,
            Int32(length),
            Float32(2.0 + Float64(i)),
            grid_dim=((length + 63) // 64),
            block_dim=64,
        )

    # Synchronize both streams
    high_priority_stream.synchronize()
    low_priority_stream.synchronize()

    # Verify final results
    var high_output_host = ctx.enqueue_create_host_buffer[.float32](length)
    var low_output_host = ctx.enqueue_create_host_buffer[.float32](length)

    ctx.enqueue_copy(high_output_host, high_output_device)
    ctx.enqueue_copy(low_output_host, low_output_device)
    ctx.synchronize()

    var expected_high = Float32(2.0 + iterations - 1)
    var expected_low = Float32(1.0 + iterations - 1)

    for i in range(min(5, length)):
        var expected_high_val = Float32(i) * expected_high
        var expected_low_val = Float32(i) * expected_low
        assert_equal(high_output_host[i], expected_high_val)
        assert_equal(low_output_host[i], expected_low_val)


def main() raises:
    with DeviceContext() as ctx:
        test_stream_priority_range(ctx)
        test_create_stream_default(ctx)
        test_create_stream_with_priority(ctx)
        test_multiple_priority_streams(ctx)
        test_concurrent_priority_streams(ctx)
