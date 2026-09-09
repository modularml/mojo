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
"""Tests for `warp.match_any`.

`match_any(value)` returns, for each warp lane, the mask of lanes whose `value`
has the same bits.  One warp is launched over several value patterns and the
device mask is checked against a host reference, exercising all three backends:
NVIDIA's `match.any.sync`, the CDNA ballot loop, and the Apple Silicon shuffle
emulation.
"""

from max.gpu import lane_id
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import match_any
from std.testing import assert_equal, TestSuite


def _match_any_probe[
    dtype: DType,
](
    values: Pointer[Scalar[dtype], MutAnyOrigin],
    out_masks: Pointer[UInt64, MutAnyOrigin],
):
    var lane = Int(lane_id())
    out_masks[unsafe_offset=lane] = match_any(values[unsafe_offset=lane]).cast[
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
    ctx.enqueue_function[_match_any_probe[dtype]](
        v_dev, out_dev, grid_dim=1, block_dim=n
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Reference: lane `l`'s mask has bit `k` set for every lane `k` with an
    # equal value (bit-equal == value-equal for these integer inputs).
    for l in range(n):
        var expected: UInt64 = 0
        for k in range(n):
            if values[k] == values[l]:
                expected |= UInt64(1) << UInt64(k)
        assert_equal(out_host[l], expected, String(name, " lane ", l))
    _ = v_dev^
    _ = out_dev^


def test_match_any_all_same() raises:
    # One group spanning the whole warp.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        _check(ctx, "all_same", List[Int32](length=w, fill=Int32(7)))


def test_match_any_all_distinct() raises:
    # Every lane its own singleton group.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int32](length=w, fill=Int32(0))
        for i in range(w):
            vals[i] = Int32(i)
        _check(ctx, "all_distinct", vals)


def test_match_any_interleaved_groups() raises:
    # Three interleaved groups.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int32](length=w, fill=Int32(0))
        for i in range(w):
            vals[i] = Int32(i % 3)
        _check(ctx, "mod3", vals)


def test_match_any_with_sentinel() raises:
    # Five groups plus a `-1` sentinel group folding into itself.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int32](length=w, fill=Int32(0))
        for i in range(w):
            vals[i] = Int32(-1) if (i % 4 == 0) else Int32(i % 5)
        _check(ctx, "mod5_with_neg", vals)


def test_match_any_64bit() raises:
    # 64-bit values that differ in the high 32 bits exercise the
    # `match.any.sync.b64` / 64-bit fold path.
    with DeviceContext() as ctx:
        comptime w = WARP_SIZE
        var vals = List[Int64](length=w, fill=Int64(0))
        for i in range(w):
            vals[i] = Int64(0x1_0000_0000) * Int64(i % 3)
        _check(ctx, "int64_mod3", vals)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
