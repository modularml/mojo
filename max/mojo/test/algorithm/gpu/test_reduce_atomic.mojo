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
from std.atomic import Atomic

from max.gpu import global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace, unsafe_stack_allocation
from std.testing import assert_equal, TestSuite
from std.sys import is_apple_gpu, has_apple_gpu_accelerator


@fieldwise_init
struct FillStrategy(Equatable, ImplicitlyCopyable):
    var value: Int

    comptime LINSPACE = Self(0)
    comptime NEG_LINSPACE = Self(1)
    comptime SYMMETRIC_LINSPACE = Self(2)
    comptime ZEROS = Self(3)
    comptime ONES = Self(4)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value


def reduce_add(
    res_add: Pointer[Float32, MutAnyOrigin],
    vec: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x

    if tid >= len:
        return

    _ = Atomic.fetch_add(res_add, vec[unsafe_offset=tid])


def reduce_add_via_cas(
    res_add: Pointer[Float32, MutAnyOrigin],
    vec: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x

    if tid >= len:
        return

    # `weak=True` is required on Apple GPU; safe here since a spurious
    # failure just costs one extra loop iteration.
    var expected = Atomic.load(res_add)
    while True:
        var desired = expected + vec[unsafe_offset=tid]
        if Atomic.compare_exchange[weak=True](res_add, expected, desired):
            return


def reduce_add_via_shared_cas(
    res_add: Pointer[Float32, MutAnyOrigin],
    vec: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    """Same CAS-retry-loop reduction as `reduce_add_via_cas`, but on
    threadgroup (`AddressSpace.SHARED`) memory, to exercise Apple GPU's
    local-address-space `cmpxchg` path."""
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var shared = unsafe_stack_allocation[1, Float32, address_space=.SHARED]()

    if thread_idx.x == 0:
        shared[unsafe_offset=0] = 0

    barrier()

    var tid = global_idx.x
    if tid < len:
        var expected = Atomic.load(shared)
        while True:
            var desired = expected + vec[unsafe_offset=tid]
            if Atomic.compare_exchange[weak=True](shared, expected, desired):
                break

    barrier()

    if thread_idx.x == 0:
        _ = Atomic.fetch_add(res_add, shared[unsafe_offset=0])


def reduce_min_max(
    res_min: Pointer[Float32, MutAnyOrigin],
    res_max: Pointer[Float32, MutAnyOrigin],
    vec: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x

    if tid >= len:
        return

    Atomic.min(res_min, vec[unsafe_offset=tid])
    Atomic.max(res_max, vec[unsafe_offset=tid])


def run_reduce(fill_strategy: FillStrategy, ctx: DeviceContext) raises:
    comptime BLOCK_SIZE = 32
    comptime n = 1024
    comptime F32 = DType.float32

    var stack = Array[Float32, n](fill=0)
    var vec_host = Span(stack)

    if fill_strategy == FillStrategy.LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(i)
    elif fill_strategy == FillStrategy.NEG_LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(-i)
    elif fill_strategy == FillStrategy.SYMMETRIC_LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(i - n // 2)
    elif fill_strategy == FillStrategy.ZEROS:
        for i in range(n):
            vec_host[i] = 0
    elif fill_strategy == FillStrategy.ONES:
        for i in range(n):
            vec_host[i] = 1

    var vec_device = ctx.enqueue_create_buffer[F32](n)
    vec_device.enqueue_copy_from(vec_host)

    var res_add_device = ctx.enqueue_create_buffer[F32](1)
    res_add_device.enqueue_fill(0)

    ctx.enqueue_function[reduce_add](
        res_add_device,
        vec_device,
        Int32(n),
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    res_add_device.enqueue_copy_to(Pointer(to=res))

    var res_min = Float32(0)
    var res_max = Float32(0)

    comptime if not has_apple_gpu_accelerator():
        var res_min_device = ctx.enqueue_create_buffer[F32](1)
        res_min_device.enqueue_fill(0)

        var res_max_device = ctx.enqueue_create_buffer[F32](1)
        res_max_device.enqueue_fill(0)
        ctx.enqueue_function[reduce_min_max](
            res_min_device,
            res_max_device,
            vec_device,
            Int32(n),
            grid_dim=ceildiv(n, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )

        res_min_device.enqueue_copy_to(Pointer(to=res_min))
        res_max_device.enqueue_copy_to(Pointer(to=res_max))

    ctx.synchronize()

    if fill_strategy == FillStrategy.LINSPACE:
        assert_equal(res, n * (n - 1) // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, n - 1)
    elif fill_strategy == FillStrategy.NEG_LINSPACE:
        assert_equal(res, -n * (n - 1) // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, -n + 1)
            assert_equal(res_max, 0)
    elif fill_strategy == FillStrategy.SYMMETRIC_LINSPACE:
        assert_equal(res, -n // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, -n // 2)
            assert_equal(res_max, (n - 1) // 2)
    elif fill_strategy == FillStrategy.ZEROS:
        assert_equal(res, 0)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, 0)
    elif fill_strategy == FillStrategy.ONES:
        assert_equal(res, n)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, 1)

    _ = vec_device


def run_reduce_via_cas(ctx: DeviceContext) raises:
    # CAS-loop correctness under contention doesn't depend on the data, so
    # (unlike `run_reduce`) this runs once rather than per `FillStrategy`.
    comptime BLOCK_SIZE = 32
    comptime n = 1024
    comptime F32 = DType.float32

    var vec_host = Array[Float32, n](fill=0)
    for i in range(n):
        vec_host[i] = Float32(i)

    var vec_device = ctx.enqueue_create_buffer[F32](n)
    vec_device.enqueue_copy_from(Span(vec_host))

    var res_device = ctx.enqueue_create_buffer[F32](1)
    res_device.enqueue_fill(0)
    ctx.enqueue_function[reduce_add_via_cas](
        res_device,
        vec_device,
        Int32(n),
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res_shared_device = ctx.enqueue_create_buffer[F32](1)
    res_shared_device.enqueue_fill(0)
    ctx.enqueue_function[reduce_add_via_shared_cas](
        res_shared_device,
        vec_device,
        Int32(n),
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    res_device.enqueue_copy_to(Pointer(to=res))
    var res_shared = Float32(0)
    res_shared_device.enqueue_copy_to(Pointer(to=res_shared))
    ctx.synchronize()

    assert_equal(res, n * (n - 1) // 2)
    assert_equal(res_shared, n * (n - 1) // 2)

    _ = vec_device


def test_reduce_atomic() raises:
    with DeviceContext() as ctx:
        run_reduce(FillStrategy.LINSPACE, ctx)
        run_reduce(FillStrategy.NEG_LINSPACE, ctx)
        run_reduce(FillStrategy.SYMMETRIC_LINSPACE, ctx)
        run_reduce(FillStrategy.ZEROS, ctx)
        run_reduce(FillStrategy.ONES, ctx)
        run_reduce_via_cas(ctx)


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_reduce_atomic]()

    suite^.run()
