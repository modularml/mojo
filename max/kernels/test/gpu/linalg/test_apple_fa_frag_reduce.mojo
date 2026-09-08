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
"""Isolated unit test for the Apple M5 FA2-prefill score-fragment row reduction.

The prefill kernel computes `Q.K^T` into the M5 16x16 simdgroup-MMA accumulator,
then needs a per-ROW max and sum over that score tile for the online softmax.
The score row is scattered across the MMA fragment, so the reduction is the one
piece that can't be lifted from the decode kernel (decode is a GEMV + simd_sum).
This test validates that reduction in isolation against a host reference before
it's wired into the full kernel -- a silent fragment-layout bug would otherwise
hide here.

Fragment layout (`_apple_frag_layout`, M5 16x16, ground-truthed by
`test_apple_mma_fragment.mojo` and the matmul epilogue): lane `tid` owns 8 fp32
elements = 2 rows x 4 cols. With `tid = b4 b3 b2 b1 b0`:
  row_lo   = 4*b4 + 2*b2 + b1   (the lane's two rows are row_lo and row_lo+8)
  col_base = 8*b3 + 4*b0        (the lane's 4 cols are col_base .. col_base+3)
  regs[0:4] -> row_lo,   cols col_base..+3
  regs[4:8] -> row_lo+8, cols col_base..+3

So a full 16-wide row is held by the 4 lanes that share (b1,b2,b4) and vary
(b0,b3) -- i.e. lanes at XOR offsets {1, 8}. The row reduction is therefore:
per-lane reduce the 4 cols of each row-half, then a 2-step butterfly
(`shuffle_xor` over masks 1 and 8) across those 4 lanes. No shared memory, no
barriers (the Apple idiom). fp32 is shuffled at width 1 -- the only width the
Apple shuffle intrinsic supports for non-half dtypes.
"""

from max.gpu import WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from std.sys.info import _accelerator_arch

comptime MMA_DIM = 16  # one 16x16 score sub-tile


@always_inline
def _frag_row_max(frag: SIMD[.float32, 8]) -> SIMD[.float32, 2]:
    """Full-row max over a 16x16 score fragment, for the lane's two rows.

    Returns `(max over row_lo, max over row_lo+8)`; after the butterfly every
    lane sharing a row holds the same result.
    """
    var r0 = max(max(frag[0], frag[1]), max(frag[2], frag[3]))
    var r1 = max(max(frag[4], frag[5]), max(frag[6], frag[7]))
    # Butterfly across the 4 row-sharing lanes: XOR 1 flips b0, XOR 8 flips b3.
    r0 = max(r0, shuffle_xor(r0, UInt32(1)))
    r0 = max(r0, shuffle_xor(r0, UInt32(8)))
    r1 = max(r1, shuffle_xor(r1, UInt32(1)))
    r1 = max(r1, shuffle_xor(r1, UInt32(8)))
    return SIMD[.float32, 2](r0, r1)


@always_inline
def _frag_row_sum(frag: SIMD[.float32, 8]) -> SIMD[.float32, 2]:
    """Full-row sum over a 16x16 score fragment, for the lane's two rows."""
    var r0 = frag[0] + frag[1] + frag[2] + frag[3]
    var r1 = frag[4] + frag[5] + frag[6] + frag[7]
    r0 = r0 + shuffle_xor(r0, UInt32(1))
    r0 = r0 + shuffle_xor(r0, UInt32(8))
    r1 = r1 + shuffle_xor(r1, UInt32(1))
    r1 = r1 + shuffle_xor(r1, UInt32(8))
    return SIMD[.float32, 2](r0, r1)


def _frag_reduce_kernel(
    d_ptr: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_sum: MutPointer[Float32, MutAnyOrigin],
):
    """One simdgroup loads a 16x16 fp32 tile into the MMA fragment layout, row-
    reduces, and writes the 16 per-row max/sum results."""
    var lane = Int(lane_id())
    var row_lo = ((lane & 7) // 2) + ((lane & 16) >> 2)
    var col_base = ((lane & 1) << 2) + (lane & 8)

    var frag = SIMD[.float32, 8](0)

    comptime for el in range(8):
        var row = row_lo + (8 if el > 3 else 0)
        var col = col_base + (el & 3)
        frag[el] = d_ptr[row * MMA_DIM + col]

    var rmax = _frag_row_max(frag)
    var rsum = _frag_row_sum(frag)

    # The 8 lanes with col_base == 0 cover row_lo 0..7 uniquely, so they write
    # all 16 rows exactly once -- no races.
    if col_base == 0:
        out_max[row_lo] = rmax[0]
        out_max[row_lo + 8] = rmax[1]
        out_sum[row_lo] = rsum[0]
        out_sum[row_lo + 8] = rsum[1]


def test_frag_row_reduce(ctx: DeviceContext) raises:
    print("== test_frag_row_reduce (16x16 score-fragment row max/sum)")
    comptime N = MMA_DIM * MMA_DIM

    var d_host = ctx.enqueue_create_host_buffer[.float32](N)
    # Deterministic, non-monotonic per row (so the max is not trivially the last
    # column), values in roughly [-125, 125].
    for r in range(MMA_DIM):
        for c in range(MMA_DIM):
            var v = ((r * 131 + c * 977) % 251) - 125
            d_host[r * MMA_DIM + c] = Float32(v)

    var d_dev = ctx.enqueue_create_buffer[.float32](N)
    var max_dev = ctx.enqueue_create_buffer[.float32](MMA_DIM)
    var sum_dev = ctx.enqueue_create_buffer[.float32](MMA_DIM)
    ctx.enqueue_copy(d_dev, d_host)

    ctx.enqueue_function[_frag_reduce_kernel](
        d_dev.unsafe_ptr(),
        max_dev.unsafe_ptr(),
        sum_dev.unsafe_ptr(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    var max_host = ctx.enqueue_create_host_buffer[.float32](MMA_DIM)
    var sum_host = ctx.enqueue_create_host_buffer[.float32](MMA_DIM)
    ctx.enqueue_copy(max_host, max_dev)
    ctx.enqueue_copy(sum_host, sum_dev)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = d_dev^
    _ = max_dev^
    _ = sum_dev^

    var pass_ = True
    for r in range(MMA_DIM):
        var exp_max = Float32(-1.0e30)
        var exp_sum = Float32(0)
        for c in range(MMA_DIM):
            var v = d_host[r * MMA_DIM + c]
            exp_max = max(exp_max, v)
            exp_sum += v
        if max_host[r] != exp_max:
            print("FAIL max row", r, "got", max_host[r], "exp", exp_max)
            pass_ = False
        if abs(sum_host[r] - exp_sum) > Float32(1e-3):
            print("FAIL sum row", r, "got", sum_host[r], "exp", exp_sum)
            pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (16x16 simdgroup MMA fragment)")
        return
    test_frag_row_reduce(ctx)
