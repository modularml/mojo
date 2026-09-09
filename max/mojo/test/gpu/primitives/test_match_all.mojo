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
"""Tests for `warp.match_all`.

`match_all(value)` returns the warp's active-lane mask if every lane holds the
same bits, otherwise 0.  One warp is launched over several value patterns and
the (uniform) device result is checked against a host reference, exercising all
three backends: NVIDIA's `match.all.sync`, the CDNA `readfirstlane`/ballot fold,
and the Apple Silicon shuffle check.
"""

from max.gpu import lane_id
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import match_all
from std.testing import assert_equal, TestSuite


def _match_all_probe[
    dtype: DType,
](
    values: Pointer[Scalar[dtype], MutAnyOrigin],
    out_masks: Pointer[UInt64, MutAnyOrigin],
):
    var lane = Int(lane_id())
    out_masks[unsafe_offset=lane] = match_all(values[unsafe_offset=lane]).cast[
        DType.uint64
    ]()


def _check[
    dtype: DType
](ctx: DeviceContext, name: String, values: List[Scalar[dtype]]) raises:
    var n = len(values)
    var v_host = ctx.enqueue_create_host_buffer[dtype](n)
    var out_host = ctx.enqueue_create_host_buffer[.uint64](n)
    ctx.synchronize()
    for i in range(n):
        v_host[i] = values[i]

    var v_dev = ctx.enqueue_create_buffer[dtype](n)
    var out_dev = ctx.enqueue_create_buffer[.uint64](n)
    ctx.enqueue_copy(v_dev, v_host)
    ctx.enqueue_function[_match_all_probe[dtype]](
        v_dev, out_dev, grid_dim=1, block_dim=n
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Uniform result: the full n-lane mask if every lane holds the same value
    # (bit-equal == value-equal for these integer inputs), else 0.
    var all_same = True
    for i in range(n):
        if values[i] != values[0]:
            all_same = False
    var full = ~UInt64(0) if n >= 64 else ((UInt64(1) << UInt64(n)) - 1)
    var expected = full if all_same else UInt64(0)
    for l in range(n):
        assert_equal(out_host[l], expected, String(name, " lane ", l))
    _ = v_dev^
    _ = out_dev^


def test_match_all_all_same() raises:
    # Every lane agrees -> the full warp mask.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        _check(ctx, "all_same", List[Int32](length=w, fill=Int32(7)))


def test_match_all_one_different() raises:
    # A single disagreeing lane -> 0 for every lane.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int32](length=w, fill=Int32(5))
        vals[w // 2] = Int32(9)
        _check(ctx, "one_different", vals)


def test_match_all_all_distinct() raises:
    # No two lanes agree -> 0.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int32](length=w, fill=Int32(0))
        for i in range(w):
            vals[i] = Int32(i)
        _check(ctx, "all_distinct", vals)


def test_match_all_64bit_same() raises:
    # 64-bit values exercise the `match.all.sync.b64` / 64-bit fold path.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        _check(
            ctx, "int64_same", List[Int64](length=w, fill=Int64(0x1_0000_0007))
        )


def test_match_all_64bit_one_different() raises:
    # Differs only in the high 32 bits, so the b64 path must compare them.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int64](length=w, fill=Int64(0x1_0000_0007))
        vals[0] = Int64(0x2_0000_0007)
        _check(ctx, "int64_one_different", vals)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
