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
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def vec_func(
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = in0[unsafe_offset=tid] + in1[unsafe_offset=tid]


def test_concurrent_copy() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    _run_test_concurrent_copy(ctx1, ctx2)


def _run_test_concurrent_copy(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    comptime length = 1 * 1024 * 1024
    comptime T = DType.float32

    var in0_dev1 = ctx1.enqueue_create_buffer[T](length)
    var in0_dev2 = ctx1.enqueue_create_buffer[T](length)
    var in0_dev3 = ctx1.enqueue_create_buffer[T](length)

    # Initialize the variable inputs with known values.
    with in0_dev1.map_to_host() as in_host1, in0_dev2.map_to_host() as in_host2, in0_dev3.map_to_host() as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = Float32(index)
            in_host2[i] = Float32(2 * index)
            in_host3[i] = Float32(3 * index)

    # Initialize the fixed (right) inputs.
    var in1_dev1 = ctx1.enqueue_create_buffer[T](length)
    in1_dev1.enqueue_fill(1.0)
    var in1_dev2 = ctx1.enqueue_create_buffer[T](length)
    in1_dev2.enqueue_fill(2.0)
    var in1_dev3 = ctx1.enqueue_create_buffer[T](length)
    in1_dev3.enqueue_fill(3.0)

    # Initialize the device outputs with known bad values.
    var out_dev1 = ctx1.enqueue_create_buffer[T](length)
    out_dev1.enqueue_fill(101.0)
    var out_dev2 = ctx1.enqueue_create_buffer[T](length)
    out_dev2.enqueue_fill(102.0)
    var out_dev3 = ctx1.enqueue_create_buffer[T](length)
    out_dev3.enqueue_fill(103.0)
    # Initialize the result buffer on a second queue with known bad values.
    var out_host1 = ctx2.enqueue_create_host_buffer[T](length)
    out_host1.enqueue_fill(0.1)
    var out_host2 = ctx2.enqueue_create_host_buffer[T](length)
    out_host2.enqueue_fill(0.2)
    var out_host3 = ctx2.enqueue_create_host_buffer[T](length)
    out_host3.enqueue_fill(0.3)

    for i in range(10):
        print(out_host1[i])
        print(out_host2[i])
        print(out_host3[i])

    # Pre-compile and pre-register the device function
    var dev_func = ctx1.compile_function[vec_func]()

    # Make sure both queues are ready to run at this point.
    ctx1.synchronize()
    ctx2.synchronize()

    var block_dim = 1

    ctx1.enqueue_function(
        dev_func,
        in0_dev1,
        in1_dev1,
        out_dev1,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev1.reassign_ownership_to(ctx2)
    out_dev1.enqueue_copy_to(out_host1)
    ctx1.enqueue_function(
        dev_func,
        in0_dev2,
        in1_dev2,
        out_dev2,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev2.reassign_ownership_to(ctx2)
    out_dev2.enqueue_copy_to(out_host2)
    ctx1.enqueue_function(
        dev_func,
        in0_dev3,
        in1_dev3,
        out_dev3,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev3.reassign_ownership_to(ctx2)
    out_dev3.enqueue_copy_to(out_host3)

    # Wait for the copies to be completed.
    ctx2.synchronize()

    for i in range(length):
        var index = i % 2048
        if i < 10:
            print("at index", i, "the value is", out_host1[i])
            print("at index", i, "the value is", out_host2[i])
            print("at index", i, "the value is", out_host3[i])
        assert_equal(
            out_host1[i],
            Float32(Float64(index) + 1.0),
            String(
                "out_host1[",
                i,
                "] is ",
                out_host1[i],
            ),
        )
        assert_equal(
            out_host2[i],
            Float32(2.0 * Float64(index) + 2.0),
            String(
                "out_host2[",
                i,
                "] is ",
                out_host2[i],
            ),
        )
        assert_equal(
            out_host3[i],
            Float32(3.0 * Float64(index) + 3.0),
            String(
                "out_host3[",
                i,
                "] is ",
                out_host3[i],
            ),
        )


def test_concurrent_func() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    _run_test_concurrent_func(ctx1, ctx2)


def _run_test_concurrent_func(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    comptime length = 20 * 1024 * 1024
    comptime T = DType.float32

    # Initialize the variable inputs with known values.
    var in_dev1 = ctx1.enqueue_create_buffer[T](length)
    var in_dev2 = ctx1.enqueue_create_buffer[T](length)
    var in_dev3 = ctx1.enqueue_create_buffer[T](length)
    with in_dev1.map_to_host() as in_host1, in_dev2.map_to_host() as in_host2, in_dev3.map_to_host() as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = Float32(index)
            in_host2[i] = Float32(2 * index)
            in_host3[i] = Float32(3 * index)

    # Initialize the fixed (right) inputs.
    var in_dev4 = ctx1.enqueue_create_buffer[T](length)
    in_dev4.enqueue_fill(1.0)
    var in_dev5 = ctx1.enqueue_create_buffer[T](length)
    in_dev5.enqueue_fill(2.0)

    # Initialize the outputs with known bad values
    var out_dev1 = ctx1.enqueue_create_buffer[T](length)
    out_dev1.enqueue_fill(101.0)
    var out_dev2 = ctx1.enqueue_create_buffer[T](length)
    out_dev2.enqueue_fill(102.0)
    var out_dev3 = ctx1.enqueue_create_buffer[T](length)
    out_dev3.enqueue_fill(103.0)
    var out_dev4 = ctx1.enqueue_create_buffer[T](length)
    out_dev4.enqueue_fill(104.0)

    var out_host = ctx2.enqueue_create_host_buffer[T](length)
    out_host.enqueue_fill(0.5)

    # Pre-compile and pre-register the device function
    var dev_func1 = ctx1.compile_function[vec_func]()
    var dev_func2 = ctx2.compile_function[vec_func]()

    # Ensure the setup has completed.
    ctx1.synchronize()
    ctx2.synchronize()

    var block_dim = 1

    ctx1.enqueue_function(
        dev_func1,
        in_dev1,  # in0 - last use
        in_dev4,  # in1 - last use
        out_dev1,  # output
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    # Wait for out_dev1 to be produced.
    # `ctx2.enqueue_wait_for(ctx1)` is not enough:
    # While it will hold `ctx2` until out_dev1 is ready, it will not extend
    # the lifetime of `out_dev1` for the duration of the `dev_func2` call.
    in_dev2.reassign_ownership_to(ctx2)
    out_dev1.reassign_ownership_to(ctx2)

    out_dev2.reassign_ownership_to(ctx2)  # output of `dev_func2` kernel

    # The following two kernels can execute in parallel.
    ctx2.enqueue_function(
        dev_func2,
        in_dev2,  # in0 - last use
        out_dev1,  # in1 - last use
        out_dev2,  # output
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx1.enqueue_function(
        dev_func1,
        in_dev3,  # in0 - last use
        in_dev5,  # in1 - last use
        out_dev3,  # output
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Wait for output of `dev_func1`.
    out_dev3.reassign_ownership_to(ctx2)

    out_dev4.reassign_ownership_to(ctx2)  # output of `dev_func2` kernel

    ctx2.enqueue_function(
        dev_func2,
        out_dev2,  # in0 - last use
        out_dev3,  # in1 - last use
        out_dev4,  # output
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    out_dev4.enqueue_copy_to(out_host)

    # Reassign ownership of `out_host` to ctx1, then synchronize on ctx1 for the
    #  copies to be completed.
    out_host.reassign_ownership_to(ctx1)
    ctx1.synchronize()

    for i in range(length):
        var index = i % 2048
        var o1 = index + 1
        var o2 = 2 * index + o1
        var o3 = 3 * index + 2
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        assert_equal(
            out_host[i],
            Float32(o2 + o3),
            String(
                "out_host[",
                i,
                "] is ",
                out_host[i],
                " (expected=",
                Float32(o2 + o3),
                ")",
            ),
        )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_concurrent_copy]()
    suite.test[test_concurrent_func]()

    suite^.run()
