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
"""Tests for `warp.vote` (the warp ballot).

`vote[ret_type](pred)` returns a lane mask whose bit `i` is set iff lane `i`
voted `True`; every lane receives the full mask. One warp is launched over
several predicate patterns and the device mask is checked against a host
reference, exercising all three backends: NVIDIA's `vote.ballot.sync`, AMD's
`ballot`, and the Apple Silicon `simd_ballot`.
"""

from max.gpu import lane_id
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.host.info import GPUInfo
from max.gpu.primitives.warp import vote
from std.sys.info import _accelerator_arch
from std.testing import assert_equal, TestSuite

# WARP_SIZE-appropriate mask type: NVIDIA/Apple SIMD-groups are 32 lanes
# (`uint32`), AMD wavefronts are 64 lanes (`uint64`). Matches the `match_any`
# default and keeps the probe portable across all three backends.
comptime _MASK = DType.uint32 if WARP_SIZE <= 32 else DType.uint64

# The *target-accelerator* vendor, resolved in host context. Do NOT gate the
# `uint64` ballot on `is_nvidia_gpu()`: that predicate reads the *current
# function's* target triple, so inside a host `def` it is False even when
# cross-compiling `--target-accelerator=nvidia:...`, and the `uint64` device
# kernel would still be instantiated -- tripping `vote`'s NVIDIA uint32-only
# constraint. `_accelerator_arch()` reflects the build's accelerator flag (the
# same mechanism `WARP_SIZE` resolves through), so it is correct in host code.
comptime _TARGET_API = GPUInfo.from_name[_accelerator_arch()]().api


def _vote_probe[
    ret_type: DType
](
    preds: Pointer[UInt8, MutAnyOrigin],
    out_masks: Pointer[UInt64, MutAnyOrigin],
):
    var lane = Int(lane_id())
    out_masks[unsafe_offset=lane] = vote[ret_type](
        preds[unsafe_offset=lane] != 0
    ).cast[.uint64]()


def _check[
    ret_type: DType = _MASK
](ctx: DeviceContext, name: String, preds: List[Int]) raises:
    var n = len(preds)  # == WARP_SIZE
    var p_host = ctx.enqueue_create_host_buffer[.uint8](n)
    var out_host = ctx.enqueue_create_host_buffer[.uint64](n)
    ctx.synchronize()
    for i in range(n):
        p_host[i] = UInt8(1) if preds[i] != 0 else UInt8(0)

    var p_dev = ctx.enqueue_create_buffer[.uint8](n)
    var out_dev = ctx.enqueue_create_buffer[.uint64](n)
    ctx.enqueue_copy(p_dev, p_host)
    ctx.enqueue_function[_vote_probe[ret_type]](
        p_dev, out_dev, grid_dim=1, block_dim=n
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # Reference: every lane receives the SAME full ballot, with bit `k` set iff
    # lane `k` voted `True`. The reference never sets a bit `>= WARP_SIZE`, so
    # exact equality also rejects any phantom high bit.
    var expected: UInt64 = 0
    for k in range(n):
        if preds[k] != 0:
            expected |= UInt64(1) << UInt64(k)
    for lane in range(n):
        assert_equal(out_host[lane], expected, String(name, " lane ", lane))
    _ = p_dev^
    _ = out_dev^


def test_vote_all_true() raises:
    # Whole warp votes True -> low WARP_SIZE bits set, no bit above.
    with DeviceContext() as ctx:
        _check(ctx, "all_true", List[Int](length=WARP_SIZE, fill=1))


def test_vote_all_false() raises:
    with DeviceContext() as ctx:
        _check(ctx, "all_false", List[Int](length=WARP_SIZE, fill=0))


def test_vote_alternating() raises:
    # Even lanes vote True (0x...55 pattern).
    with DeviceContext() as ctx:
        var preds = List[Int](length=WARP_SIZE, fill=0)
        for i in range(WARP_SIZE):
            preds[i] = 1 if (i % 2 == 0) else 0
        _check(ctx, "alternating", preds)


def test_vote_single_first_lane() raises:
    with DeviceContext() as ctx:
        var preds = List[Int](length=WARP_SIZE, fill=0)
        preds[0] = 1
        _check(ctx, "single_first", preds)


def test_vote_single_last_lane() raises:
    with DeviceContext() as ctx:
        var preds = List[Int](length=WARP_SIZE, fill=0)
        preds[WARP_SIZE - 1] = 1
        _check(ctx, "single_last", preds)


def test_vote_single_middle_lane() raises:
    with DeviceContext() as ctx:
        var preds = List[Int](length=WARP_SIZE, fill=0)
        preds[13] = 1
        _check(ctx, "single_middle", preds)


def test_vote_uint64_high_bits_masked() raises:
    # Force a `uint64` return on backends where WARP_SIZE < 64 and assert no
    # bit `>= WARP_SIZE` survives (exact equality against a low-bits-only
    # reference). Gated off NVIDIA, which supports only a 32-bit return.
    comptime if _TARGET_API != "cuda" and WARP_SIZE < 64:
        with DeviceContext() as ctx:
            _check[.uint64](
                ctx, "u64_all_true", List[Int](length=WARP_SIZE, fill=1)
            )
            var preds = List[Int](length=WARP_SIZE, fill=0)
            for i in range(WARP_SIZE):
                preds[i] = 1 if (i % 3 == 0) else 0
            _check[.uint64](ctx, "u64_mod3", preds)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
