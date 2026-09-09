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
"""Unit tests for the multi-stream `DeviceContext.select_stream` view API.

`select_stream(id)` returns a "retain-all" view of a single base context: the
view shares the base's full stream set, driver context, and device memory pool,
and only differs in which stream work is enqueued on. These tests exercise that
contract directly at the runtime level (no compiler / model), which is far
cheaper to diagnose than a full multi-stream graph:

- stream 0 selects a view equivalent to the base,
- a buffer produced on the base stream and consumed on a side-stream view reads
  correct data (shared memory pool + cross-stream ownership sync), and
- a view of a view composes (the view keeps the full stream set), and the
  intermediate view can be dropped while the final view stays valid (the base
  context's resources are shared/refcounted).
"""

from asyncrt_test_utils import create_test_device_context
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def add_one(
    inp: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = inp[unsafe_offset=tid] + 1.0


def ensure_stream_count(ctx: DeviceContext, count: Int) raises:
    var current = ctx.num_streams()
    for _ in range(current, count):
        _ = ctx.create_stream()


def test_select_stream_zero_is_base() raises:
    """Selecting stream 0 yields a view that runs work on the default stream."""
    var ctx = create_test_device_context()
    ensure_stream_count(ctx, 1)
    _run_select_stream_zero_is_base(ctx)


def _run_select_stream_zero_is_base(ctx: DeviceContext) raises:
    comptime length = 64 * 1024
    comptime T = DType.float32

    var view = ctx.select_stream(0)
    var func = view.compile_function[add_one]()

    var inp = view.enqueue_create_buffer[T](length)
    inp.enqueue_fill(2.0)
    var out = view.enqueue_create_buffer[T](length)
    out.enqueue_fill(0.0)
    view.enqueue_function(
        func, inp, out, Int32(length), grid_dim=length, block_dim=1
    )

    var host = view.enqueue_create_host_buffer[T](length)
    out.enqueue_copy_to(host)
    view.synchronize()

    for i in range(length):
        assert_equal(host[i], Float32(3.0), String("host[", i, "]"))


def test_cross_stream_buffer_reuse() raises:
    """A buffer produced on the base stream is consumed on a side-stream view.

    The two views share the device memory pool, and `reassign_ownership_to`
    inserts the cross-stream dependency needed for the side stream to read the
    base stream's output correctly. This mirrors the production shared-expert
    pattern (a side stream reads attention output produced on the main stream).
    """
    var ctx = create_test_device_context()
    ensure_stream_count(ctx, 2)
    _run_cross_stream_buffer_reuse(ctx)


def _run_cross_stream_buffer_reuse(ctx: DeviceContext) raises:
    comptime length = 64 * 1024
    comptime T = DType.float32

    var side = ctx.select_stream(1)
    var base_func = ctx.compile_function[add_one]()
    var side_func = side.compile_function[add_one]()

    # Produce `b = a + 1 == 3` on the base stream (stream 0).
    var a = ctx.enqueue_create_buffer[T](length)
    a.enqueue_fill(2.0)
    var b = ctx.enqueue_create_buffer[T](length)
    ctx.enqueue_function(
        base_func, a, b, Int32(length), grid_dim=length, block_dim=1
    )

    # Hand `b` to the side stream and compute `c = b + 1 == 4` there. The
    # ownership reassignment makes the side stream wait for the base stream's
    # kernel, so `c` reflects the produced value rather than uninitialized data.
    b.reassign_ownership_to(side)
    var c = ctx.enqueue_create_buffer[T](length)
    c.reassign_ownership_to(side)
    side.enqueue_function(
        side_func, b, c, Int32(length), grid_dim=length, block_dim=1
    )

    var host = side.enqueue_create_host_buffer[T](length)
    c.enqueue_copy_to(host)
    side.synchronize()

    for i in range(length):
        assert_equal(host[i], Float32(4.0), String("host[", i, "]"))


def test_select_stream_composes() raises:
    """A view of a view keeps the full stream set, so re-selecting composes.

    The intermediate view is a temporary that is destroyed before use; the final
    view must remain valid because the base context's resources are shared.
    """
    var ctx = create_test_device_context()
    ensure_stream_count(ctx, 3)
    _run_select_stream_composes(ctx)


def _run_select_stream_composes(ctx: DeviceContext) raises:
    comptime length = 64 * 1024
    comptime T = DType.float32

    # Re-select stream 2 from a (temporary) stream-1 view.
    var v = ctx.select_stream(1).select_stream(2)
    var func = v.compile_function[add_one]()

    var inp = ctx.enqueue_create_buffer[T](length)
    inp.enqueue_fill(5.0)
    inp.reassign_ownership_to(v)
    var out = ctx.enqueue_create_buffer[T](length)
    out.reassign_ownership_to(v)
    v.enqueue_function(
        func, inp, out, Int32(length), grid_dim=length, block_dim=1
    )

    var host = v.enqueue_create_host_buffer[T](length)
    out.enqueue_copy_to(host)
    v.synchronize()

    for i in range(length):
        assert_equal(host[i], Float32(6.0), String("host[", i, "]"))


def main() raises:
    var suite = TestSuite()

    suite.test[test_select_stream_zero_is_base]()
    suite.test[test_cross_stream_buffer_reuse]()
    suite.test[test_select_stream_composes]()

    suite^.run()
