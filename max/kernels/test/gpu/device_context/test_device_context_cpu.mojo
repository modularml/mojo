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

from max.gpu.host import DeviceContext, DeviceContextArray
from std.memory import alloc
from std.testing import assert_equal, assert_false, assert_true


def test_empty_func(ctx: DeviceContext) raises:
    def empty() {} -> None:
        pass

    ctx.enqueue_cpu_function(empty)
    ctx.synchronize()


def test_func_writes_to_memory(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    def write_42() {imm} -> None:
        ptr[] = 42

    ctx.enqueue_cpu_function(write_42)
    ctx.synchronize()
    assert_equal(ptr[], 42)
    ptr.free()


def test_func_closure_writes_to_memory(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0
    var expected = 42

    def write_val() {imm}:
        ptr[] = expected

    ctx.enqueue_cpu_function(write_val)
    ctx.synchronize()
    assert_equal(ptr[], 42)
    ptr.free()


def test_multiple_funcs_execute_in_order(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    def write_1() {imm} -> None:
        ptr[] = 1

    def write_2() {imm} -> None:
        ptr[] = 2

    def write_3() {imm} -> None:
        ptr[] = 3

    ctx.enqueue_cpu_function(write_1)
    ctx.enqueue_cpu_function(write_2)
    ctx.enqueue_cpu_function(write_3)
    ctx.synchronize()
    # Stream semantics: functions execute in order, last write wins.
    assert_equal(ptr[], 3)
    ptr.free()


def test_func_accumulates(ctx: DeviceContext) raises:
    var ptr = alloc[Int](1)
    ptr[] = 0

    def increment() {imm} -> None:
        ptr[] += 1

    ctx.enqueue_cpu_function(increment)
    ctx.enqueue_cpu_function(increment)
    ctx.enqueue_cpu_function(increment)
    ctx.synchronize()
    assert_equal(ptr[], 3)
    ptr.free()


def test_range_writes_indices(ctx: DeviceContext) raises:
    comptime count = 16
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = -1

    def write_index(i: Int) {mut} -> None:
        ptr[i] = i

    ctx.enqueue_cpu_range(write_index, count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i)
    ptr.free()


def test_range_large(ctx: DeviceContext) raises:
    comptime count = 1024
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    def write_squared(i: Int) {mut} -> None:
        ptr[i] = i * i

    ctx.enqueue_cpu_range(write_squared, count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i * i)
    ptr.free()


def test_func_then_range(ctx: DeviceContext) raises:
    comptime count = 8
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    def set_all_to_one() {imm} -> None:
        for j in range(count):
            ptr[j] = 1

    def add_index(i: Int) {mut} -> None:
        ptr[i] += i

    # First fill with 1s, then add the index.
    ctx.enqueue_cpu_function(set_all_to_one)
    ctx.enqueue_cpu_range(add_index, count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], 1 + i)
    ptr.free()


def test_two_ranges_sequential(ctx: DeviceContext) raises:
    comptime count = 8
    var ptr = alloc[Int](count)
    for i in range(count):
        ptr[i] = 0

    def write_index(i: Int) {mut} -> None:
        ptr[i] = i

    def double_value(i: Int) {mut} -> None:
        ptr[i] *= 2

    ctx.enqueue_cpu_range(write_index, count=count)
    ctx.enqueue_cpu_range(double_value, count=count)
    ctx.synchronize()
    for i in range(count):
        assert_equal(ptr[i], i * 2)
    ptr.free()


def test_device_context_array(ctx: DeviceContext) raises:
    var arr: DeviceContextArray[2] = [ctx, ctx]
    assert_equal(len(arr), 2)
    assert_equal(arr[0].api(), ctx.api())
    assert_equal(arr[1].api(), ctx.api())

    # The length is inferred from the element count of the literal.
    var inferred: DeviceContextArray[_] = [ctx, ctx, ctx]
    comptime assert type_of(inferred).length == 3
    assert_equal(len(inferred), 3)

    # Direct variadic call, as synthesized by the graph compiler's
    # multi-device lowering.
    var direct = DeviceContextArray[2](ctx, ctx)
    assert_equal(len(direct), 2)
    assert_equal(direct[0].api(), ctx.api())


def test_context_equality(ctx: DeviceContext) raises:
    # A copy of the same context refers to the same underlying runtime
    # context, so it compares equal.
    var copy = ctx
    assert_true(copy == ctx)
    assert_false(copy != ctx)

    # A separately constructed context on the same device is a different
    # runtime context, even though the device ID is identical.
    var fresh = DeviceContext(api="cpu")
    assert_false(fresh == ctx)
    assert_true(fresh != ctx)

    # Copies of the same fresh context are equal to each other.
    var fresh_copy = fresh
    assert_true(fresh_copy == fresh)


def main() raises:
    with DeviceContext(api="cpu") as ctx:
        test_empty_func(ctx)
        test_func_writes_to_memory(ctx)
        test_func_closure_writes_to_memory(ctx)
        test_multiple_funcs_execute_in_order(ctx)
        test_func_accumulates(ctx)
        test_range_writes_indices(ctx)
        test_range_large(ctx)
        test_func_then_range(ctx)
        test_two_ranges_sequential(ctx)
        test_device_context_array(ctx)
        test_context_equality(ctx)
