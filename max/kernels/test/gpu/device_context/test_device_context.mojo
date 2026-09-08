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

from std.math import iota

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.reflection import reflect
from std.testing import assert_equal


# A Simple Kernel performing the sum of two arrays
def vec_func(
    in0: ImmPointer[Float32, ImmutAnyOrigin],
    in1: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    supplement_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var len = Int(len_dev)
    var supplement = Int(supplement_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = in0[tid] + in1[tid] + Float32(supplement)


def test_declared_arg_types(ctx: DeviceContext) raises:
    var compiled = ctx.compile_function[vec_func]()
    comptime arg_types = type_of(compiled).declared_arg_types

    # The list is always present and reports the kernel's arity.
    assert_equal(arg_types.length, 5)

    # Indexing yields the declared argument types in order.
    comptime assert arg_types[0] == ImmPointer[Float32, ImmutAnyOrigin]
    comptime assert arg_types[2] == MutPointer[Float32, MutAnyOrigin]
    comptime assert arg_types[3] == Int32
    comptime assert arg_types[4] == Int32

    assert_equal(reflect[arg_types[0]].base_name(), "Pointer")


def test_is_compatible(ctx: DeviceContext) raises:
    assert_equal(ctx.is_compatible(), True)


def test_basic(ctx: DeviceContext) raises:
    comptime length = 1024

    # Host memory buffers for input and output data
    var in0_host = ctx.enqueue_create_host_buffer[.float32](length)
    var in1_host = ctx.enqueue_create_host_buffer[.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.synchronize()

    # Initialize inputs
    for i in range(length):
        in0_host[i] = Float32(i)
        in1_host[i] = Float32(2)

    # Device memory buffers for the kernel input and output
    var in0_device = ctx.enqueue_create_buffer[.float32](length)
    var in1_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    # Copy the input data from the Host to the Device memory
    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32
    var supplement = 5

    # Execute the kernel on the device.
    #  - notice the simple function call like invocation
    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        Int32(length),
        Int32(supplement),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Copy the results back from the device to the host
    ctx.enqueue_copy(out_host, out_device)

    # Wait for the computation to be completed
    ctx.synchronize()

    var expected: List[Float32] = [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(10):
        print("at index", i, "the value is", out_host[i])
        assert_equal(out_host[i], expected[i])


def test_move(ctx: DeviceContext) raises:
    var b = ctx
    var c = b^
    c.synchronize()


def test_id(ctx: DeviceContext) raises:
    # CPU always gets id 0 so test for that.
    assert_equal(ctx.id(), 0)


def test_print(ctx: DeviceContext) raises:
    comptime size = 15

    var host_buffer = ctx.enqueue_create_host_buffer[.uint16](size)
    ctx.synchronize()

    iota(host_buffer.as_span())

    var expected_host = (
        "HostBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])"
    )
    assert_equal(String(host_buffer), expected_host)

    var dev_buffer = ctx.enqueue_create_buffer[.uint16](size)
    host_buffer.enqueue_copy_to(dev_buffer)
    ctx.synchronize()

    var expected_dev = (
        "DeviceBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])"
    )
    assert_equal(String(dev_buffer), expected_dev)

    comptime large_size = 1001
    var large_buffer = ctx.enqueue_create_host_buffer[.float32](large_size)
    ctx.synchronize()

    iota(large_buffer.as_span())

    var expected_large = (
        "HostBuffer([0.0, 1.0, 2.0, ..., 998.0, 999.0, 1000.0])"
    )
    assert_equal(String(large_buffer), expected_large)


def test_enqueue_unified(ctx: DeviceContext) raises:
    comptime length = 1024

    # Host memory buffers for input and output data
    var in0_host = ctx.enqueue_create_host_buffer[.float32](length)
    var in1_host = ctx.enqueue_create_host_buffer[.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.synchronize()

    # Initialize inputs
    for i in range(length):
        in0_host[i] = Float32(i)
        in1_host[i] = Float32(2)

    # Device memory buffers for the kernel input and output
    var in0_device = ctx.enqueue_create_buffer[.float32](length)
    var in1_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    # Copy the input data from the Host to the Device memory
    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32
    var supplement: Float32 = 5

    var output = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)
    var in0 = Span(unsafe_ptr=in0_device.unsafe_ptr(), length=length)
    var in1 = Span(unsafe_ptr=in1_device.unsafe_ptr(), length=length)

    def vec_closure() {var supplement, var in0, var in1, var output}:
        var tid = global_idx.x
        if tid >= length:
            return
        output[tid] = in0[tid] + in1[tid] + supplement

    # Execute the kernel on the device.
    #  - notice the simple function call like invocation
    ctx.enqueue_function(
        vec_closure,
        grid_dim=(length // block_dim),
        block_dim=block_dim,
    )

    # Copy the results back from the device to the host
    ctx.enqueue_copy(out_host, out_device)

    # Wait for the computation to be completed
    ctx.synchronize()

    var expected: List[Float32] = [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(10):
        print("at index", i, "the value is", out_host[i])
        assert_equal(out_host[i], expected[i])


def test_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 8

    # Test with List as source (implicitly converts to Span).
    var src_list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var dev_buf = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    var out_host = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.enqueue_copy(out_host, dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(out_host[i], Float32(i + 1))


def test_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 8

    # Set up device buffer with known data.
    var src_list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var dev_buf = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    # Copy device buffer back into a List via Span.
    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32(i + 1))


def test_enqueue_copy_from_span_host_buffer(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [100.0, 200.0, 300.0, 400.0]
    var host_buf = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.enqueue_copy(host_buf, Span(src_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_buf[i], src_list[i])


def test_enqueue_copy_to_span_host_buffer(ctx: DeviceContext) raises:
    comptime length = 4

    var host_buf = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_buf[i] = Float32((i + 1) * 10)

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), host_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32((i + 1) * 10))


def test_device_buffer_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var dev_buf = ctx.enqueue_create_buffer[.float32](length)
    dev_buf.enqueue_copy_from(Span(src_list))

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], src_list[i])


def test_device_buffer_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var dev_buf = ctx.enqueue_create_buffer[.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    dev_buf.enqueue_copy_to(Span(dst_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], src_list[i])


def test_host_buffer_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var host_buf = ctx.enqueue_create_host_buffer[.float32](length)
    host_buf.enqueue_copy_from(Span(src_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_buf[i], src_list[i])


def test_host_buffer_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 4

    var host_buf = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_buf[i] = Float32((i + 1) * 10)

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    host_buf.enqueue_copy_to(Span(dst_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32((i + 1) * 10))


def main() raises:
    # Create an instance of the DeviceContext
    with DeviceContext() as ctx:
        # Execute our test with the context
        test_is_compatible(ctx)
        test_basic(ctx)
        test_declared_arg_types(ctx)
        test_move(ctx)
        test_id(ctx)
        test_print(ctx)
        test_enqueue_unified(ctx)
        test_enqueue_copy_from_span(ctx)
        test_enqueue_copy_to_span(ctx)
        test_enqueue_copy_from_span_host_buffer(ctx)
        test_enqueue_copy_to_span_host_buffer(ctx)
        test_device_buffer_enqueue_copy_from_span(ctx)
        test_device_buffer_enqueue_copy_to_span(ctx)
        test_host_buffer_enqueue_copy_from_span(ctx)
        test_host_buffer_enqueue_copy_to_span(ctx)
