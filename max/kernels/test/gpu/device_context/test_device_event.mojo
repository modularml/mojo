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
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceEvent, DeviceStream
from std.testing import assert_equal


# Simple kernel for testing event synchronization
def simple_kernel(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    multiplier: Float32,
):
    """Simple kernel that multiplies input by a multiplier."""
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = input[tid] * multiplier


# Kernel that does more work to test timing
def heavy_kernel(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len: Int,
    iterations: Int,
):
    """Kernel that does multiple iterations of work."""
    var tid = global_idx.x
    if tid >= len:
        return

    var value = input[tid]
    for _ in range(iterations):
        value = value * 1.001 + 0.001
    output[tid] = value


def test_event_record_and_synchronize(ctx: DeviceContext) raises:
    print("Test event recording and synchronization.")

    comptime length = 256
    comptime multiplier = Float32(3.0)

    # Create buffers
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    var output_host = ctx.enqueue_create_host_buffer[.float32](length)

    # Initialize input data
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    var output_device = ctx.enqueue_create_buffer[.float32](length)

    # Copy input to device
    ctx.enqueue_copy(input_device, input_host)

    # Create event and stream
    var event = ctx.create_event()
    var stream = ctx.create_stream()
    var func = ctx.compile_function[simple_kernel]()

    # Launch kernel on stream
    stream.enqueue_function(
        func,
        input_device,
        output_device,
        Int32(length),
        multiplier,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )

    # Record event after kernel
    stream.record_event(event)

    # Synchronize on event (not stream)
    event.synchronize()

    # Copy result back and verify
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    # Verify results
    for i in range(min(10, length)):
        var expected = Float32(i) * multiplier
        assert_equal(output_host[i], expected)


def test_stream_enqueue_wait_for(ctx: DeviceContext) raises:
    print("Test stream waiting for events from other streams.")

    comptime length = 512
    comptime multiplier1 = Float32(2.0)
    comptime multiplier2 = Float32(3.0)

    # Create buffers
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    var intermediate_host = ctx.enqueue_create_host_buffer[.float32](length)
    var output_host = ctx.enqueue_create_host_buffer[.float32](length)

    # Initialize input
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    var intermediate_device = ctx.enqueue_create_buffer[.float32](length)
    var output_device = ctx.enqueue_create_buffer[.float32](length)

    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Create two streams and an event
    var stream1 = ctx.create_stream()
    var stream2 = ctx.create_stream()
    var event = ctx.create_event()
    var func = ctx.compile_function[simple_kernel]()

    # Stream 1: input -> intermediate (multiply by multiplier1)
    stream1.enqueue_function(
        func,
        input_device,
        intermediate_device,
        Int32(length),
        multiplier1,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )

    # Record event after first kernel completes
    stream1.record_event(event)

    # Stream 2: wait for event, then intermediate -> output (multiply by multiplier2)
    stream2.enqueue_wait_for(event)
    stream2.enqueue_function(
        func,
        intermediate_device,
        output_device,
        Int32(length),
        multiplier2,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )

    # Synchronize both streams
    stream1.synchronize()
    stream2.synchronize()

    # Copy results back and verify
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    # Verify that we got input * multiplier1 * multiplier2
    for i in range(min(10, length)):
        var expected = Float32(i) * multiplier1 * multiplier2
        assert_equal(output_host[i], expected)


def test_multiple_events_synchronization(ctx: DeviceContext) raises:
    print("Test complex synchronization with multiple events.")

    comptime length = 256
    comptime num_streams = 4

    # Create input data
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Create multiple streams, events, and output buffers
    var streams = List[DeviceStream]()
    var events = List[DeviceEvent]()
    var output_devices = List[DeviceBuffer[.float32]]()
    var multipliers = List[Float32]()

    for i in range(num_streams):
        streams.append(ctx.create_stream())
        events.append(ctx.create_event())
        output_devices.append(ctx.enqueue_create_buffer[.float32](length))
        multipliers.append(Float32(i + 1))

    var func = ctx.compile_function[simple_kernel]()

    # Launch kernels on all streams
    for i in range(num_streams):
        streams[i].enqueue_function(
            func,
            input_device,
            output_devices[i],
            Int32(length),
            multipliers[i],
            grid_dim=ceildiv(length, 32),
            block_dim=32,
        )
        # Record event for each stream
        streams[i].record_event(events[i])

    # Synchronize all events
    for i in range(num_streams):
        events[i].synchronize()

    # Verify results from all streams
    for stream_idx in range(num_streams):
        var output_host = ctx.enqueue_create_host_buffer[.float32](length)
        ctx.enqueue_copy(output_host, output_devices[stream_idx])
        ctx.synchronize()

        # Check results
        for i in range(min(5, length)):
            var expected = Float32(i) * multipliers[stream_idx]
            assert_equal(output_host[i], expected)


def test_event_dependency_chain(ctx: DeviceContext) raises:
    print("Test creating a dependency chain using events.")

    comptime length = 128

    # Create input data
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(input_device, input_host)
    ctx.synchronize()

    # Create chain: input -> buffer1 -> buffer2 -> buffer3
    var buffer1 = ctx.enqueue_create_buffer[.float32](length)
    var buffer2 = ctx.enqueue_create_buffer[.float32](length)
    var buffer3 = ctx.enqueue_create_buffer[.float32](length)

    var stream1 = ctx.create_stream()
    var stream2 = ctx.create_stream()
    var stream3 = ctx.create_stream()

    var event1 = ctx.create_event()
    var event2 = ctx.create_event()
    # Ensure move and copy semantics are correct
    var event1_copied = event1
    var event2_moved = event2^

    var func = ctx.compile_function[simple_kernel]()

    # Stage 1: input -> buffer1 (multiply by 2)
    stream1.enqueue_function(
        func,
        input_device,
        buffer1,
        Int32(length),
        Float32(2.0),
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )
    stream1.record_event(event1_copied)

    # Stage 2: buffer1 -> buffer2 (multiply by 3), wait for stage 1
    stream2.enqueue_wait_for(event1_copied)
    stream2.enqueue_function(
        func,
        buffer1,
        buffer2,
        Int32(length),
        Float32(3.0),
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )
    stream2.record_event(event2_moved)

    # Stage 3: buffer2 -> buffer3 (multiply by 5), wait for stage 2
    stream3.enqueue_wait_for(event2_moved)
    stream3.enqueue_function(
        func,
        buffer2,
        buffer3,
        Int32(length),
        Float32(5.0),
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )

    # Synchronize final stream
    stream3.synchronize()

    # Copy result back and verify
    var output_host = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.enqueue_copy(output_host, buffer3)
    ctx.synchronize()

    # Verify that we got input * 2 * 3 * 5 = input * 30
    for i in range(min(10, length)):
        var expected = Float32(i) * 30.0
        assert_equal(output_host[i], expected)


def test_event_across_context_streams(ctx: DeviceContext) raises:
    print(
        "Test event synchronization between default stream and created streams."
    )

    comptime length = 256
    comptime multiplier = Float32(4.0)

    # Create buffers
    var input_host = ctx.enqueue_create_host_buffer[.float32](length)
    var output_host = ctx.enqueue_create_host_buffer[.float32](length)

    for i in range(length):
        input_host[i] = Float32(i)

    var input_device = ctx.enqueue_create_buffer[.float32](length)
    var output_device = ctx.enqueue_create_buffer[.float32](length)

    # Use default stream for input copy
    ctx.enqueue_copy(input_device, input_host)

    # Create event and custom stream
    var event = ctx.create_event()
    var custom_stream = ctx.create_stream()

    # Record event on default stream after copy
    ctx.stream().record_event(event)

    # Wait for copy completion on custom stream
    custom_stream.enqueue_wait_for(event)

    # Launch kernel on custom stream
    var func = ctx.compile_function[simple_kernel]()
    custom_stream.enqueue_function(
        func,
        input_device,
        output_device,
        Int32(length),
        multiplier,
        grid_dim=ceildiv(length, 32),
        block_dim=32,
    )

    # Synchronize and verify
    custom_stream.synchronize()
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    # Verify results
    for i in range(min(10, length)):
        var expected = Float32(i) * multiplier
        assert_equal(output_host[i], expected)


def main() raises:
    with DeviceContext() as ctx:
        test_event_record_and_synchronize(ctx)
        test_stream_enqueue_wait_for(ctx)
        test_multiple_events_synchronization(ctx)
        test_event_dependency_chain(ctx)
        test_event_across_context_streams(ctx)
