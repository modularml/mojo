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


from std.random import random_float64

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, HostBuffer
from std.testing import assert_equal, TestSuite


def simd_add_kernel[
    width: Int
](
    a_span: Pointer[Float32, MutAnyOrigin],
    b_span: Pointer[Float32, MutAnyOrigin],
    c_span: Pointer[Float32, MutAnyOrigin],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.unsafe_load[width=width](idx)
    var vector_b = b_span.unsafe_load[width=width](idx)
    var vector_c = vector_a + vector_b
    c_span.unsafe_store[width=width](idx, vector_c)


def simd_mult_kernel[
    width: Int
](
    a_span: Pointer[Float32, MutAnyOrigin],
    b_span: Pointer[Float32, MutAnyOrigin],
    c_span: Pointer[Float32, MutAnyOrigin],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.unsafe_load[width=width](idx)
    var vector_b = b_span.unsafe_load[width=width](idx)
    var vector_c = vector_a * vector_b
    c_span.unsafe_store[width=width](idx, vector_c)


def simd_fma_kernel[
    width: Int
](
    a_span: Pointer[Float32, MutAnyOrigin],
    b_span: Pointer[Float32, MutAnyOrigin],
    c_span: Pointer[Float32, MutAnyOrigin],
):
    # Calculate the index for this thread's data
    var idx = (thread_idx.x + block_idx.x * block_dim.x) * width

    var vector_a = a_span.unsafe_load[width=width](idx)
    var vector_b = b_span.unsafe_load[width=width](idx)
    var vector_c = c_span.unsafe_load[width=width](idx)
    vector_c = vector_a.fma(vector_b, vector_c)

    c_span.unsafe_store[width=width](idx, vector_c)


def host_elementwise_add(
    a: HostBuffer[.float32],
    b: HostBuffer[.float32],
    mut c: HostBuffer[.float32],
    size: Int,
):
    for i in range(size):
        c[i] = a[i] + b[i]


def host_elementwise_mult(
    a: HostBuffer[.float32],
    b: HostBuffer[.float32],
    mut c: HostBuffer[.float32],
    size: Int,
):
    for i in range(size):
        c[i] = a[i] * b[i]


def host_elementwise_fma(
    a: HostBuffer[.float32],
    b: HostBuffer[.float32],
    mut c: HostBuffer[.float32],
    size: Int,
):
    for i in range(size):
        var c_temp = a[i] * b[i] + c[i]
        c[i] = c_temp


def _test_arithmetic[width: Int, mode: String](ctx: DeviceContext) raises:
    comptime thread_count = 32
    comptime block_count = 1
    comptime buff_size = thread_count * block_count * width

    var a_host = ctx.enqueue_create_host_buffer[.float32](buff_size)
    var b_host = ctx.enqueue_create_host_buffer[.float32](buff_size)
    var c_host = ctx.enqueue_create_host_buffer[.float32](buff_size)

    ctx.synchronize()

    for i in range(buff_size):
        a_host[i] = random_float64(-1.0, 1.0).cast[.float32]()
        b_host[i] = random_float64(-1.0, 1.0).cast[.float32]()
        c_host[i] = 0.0

    # Create device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[.float32](buff_size)
    var b_device_buffer = ctx.enqueue_create_buffer[.float32](buff_size)
    var c_device_buffer = ctx.enqueue_create_buffer[.float32](buff_size)

    # Copy data from host to device
    ctx.enqueue_copy(a_device_buffer, a_host)
    ctx.enqueue_copy(b_device_buffer, b_host)
    ctx.enqueue_copy(c_device_buffer, c_host)

    # Compute expected result on host
    var c_expected = ctx.enqueue_create_host_buffer[.float32](buff_size)
    c_expected.enqueue_fill(0)
    ctx.synchronize()

    comptime if mode == "add":
        comptime kernel = simd_add_kernel[width]

        ctx.enqueue_function[kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_add(a_host, b_host, c_expected, buff_size)

    elif mode == "mult":
        comptime kernel = simd_mult_kernel[width]

        ctx.enqueue_function[kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_mult(a_host, b_host, c_expected, buff_size)

    else:
        comptime kernel = simd_fma_kernel[width]

        # Execute kernel on GPU
        ctx.enqueue_function[kernel](
            a_device_buffer,
            b_device_buffer,
            c_device_buffer,
            grid_dim=block_count,
            block_dim=thread_count,
        )
        host_elementwise_fma(a_host, b_host, c_expected, buff_size)

    # Copy result back from device to host
    var c_result = ctx.enqueue_create_host_buffer[.float32](buff_size)
    ctx.enqueue_copy(c_result, c_device_buffer)
    ctx.synchronize()

    # Compare results
    for i in range(buff_size):
        assert_equal(c_result[i], c_expected[i])


def test_arithmetic_sm100() raises:
    with DeviceContext() as ctx:
        _test_arithmetic[2, "add"](ctx)
        _test_arithmetic[4, "add"](ctx)
        _test_arithmetic[8, "add"](ctx)
        _test_arithmetic[2, "mult"](ctx)
        _test_arithmetic[4, "mult"](ctx)
        _test_arithmetic[8, "mult"](ctx)
        _test_arithmetic[2, "fma"](ctx)
        _test_arithmetic[4, "fma"](ctx)
        _test_arithmetic[8, "fma"](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
