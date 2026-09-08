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
"""Isolation unit test for the Apple M5 FA2-prefill online-softmax free funcs.

Drives the register-resident online-softmax state through a TWO-tile sequence and
compares against a host fp32 flash-attention online-softmax reference. Two tiles
exercise the running-max correction (`alpha`), the `l` accumulation, and the final
`normalize` -- a single tile would not move `m`/`l` off their identity seeds.

Calls `_softmax_update` (which reduces via `_softmax_row_max`) x2 then
`_softmax_normalize`, exactly as the kernel does, at a single 16-row block
(NUM_M_MMAS = 1, SQ = 16). The rescaled "output" is a synthetic per-(row, col)
value rather than a real P.V product, so the test isolates the softmax algebra.
"""

from max.gpu import WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from std.math import exp2
from std.sys.info import _accelerator_arch

from linalg.arch.apple.mma import MmaOpApple

from nn.attention.gpu.apple.fa_prefill import (
    _SOFTMAX_FRAG_ROWS,
    _softmax_normalize,
    _softmax_update,
)

comptime NUM_N_MMAS = 2
comptime SQ = 16  # single 16-row query block (NUM_M_MMAS == 1)
comptime SK = NUM_N_MMAS * 16  # KV cols per score tile
comptime NUM_TILES = 2
comptime DEPTH_MMAS = 2  # output (P.V) column fragments; depth = 32 here
comptime OUT_N = DEPTH_MMAS * 16


def _softmax_unit_kernel(
    # Two score tiles (SQ x SK each), row-major, fp32.
    s0_ptr: MutPointer[Float32, MutAnyOrigin],
    s1_ptr: MutPointer[Float32, MutAnyOrigin],
    # Per-tile synthetic "attention output" contribution (SQ x OUT_N) injected
    # directly so the test isolates the softmax; the host reference applies the
    # same recurrence.
    o0_ptr: MutPointer[Float32, MutAnyOrigin],
    o1_ptr: MutPointer[Float32, MutAnyOrigin],
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """One simdgroup: load 2 score tiles + 2 PV-output tiles into the MMA accum
    layout, run `_softmax_update` twice + `_softmax_normalize`, and write the
    final SQ x OUT_N output back."""
    var lane = Int(lane_id())
    var rb = ((lane & 7) >> 1) + ((lane & 16) >> 2)
    var cb = ((lane & 1) << 2) + (lane & 8)

    comptime ScoreMma = MmaOpApple[.float32, .float32, 1, NUM_N_MMAS]
    comptime OutMma = MmaOpApple[.float32, .float32, 1, DEPTH_MMAS]

    # Online-softmax state as the kernel declares it (no-sink), seeded m=-inf, l=0.
    var sm_m = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=Float32(-3.0e38))
    var sm_l = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=Float32(0))

    var output = OutMma.zero_accum()

    @__parameter
    def load_scores(
        src: MutPointer[Float32, MutAnyOrigin],
    ) -> ScoreMma.AccumType:
        var acc = ScoreMma.zero_accum()
        comptime for ni in range(NUM_N_MMAS):
            var frag = SIMD[.float32, 8](0)
            comptime for el in range(8):
                var row = rb + (8 if el > 3 else 0)
                var col = ni * 16 + cb + (el & 3)
                frag[el] = src[row * SK + col]
            acc[ni] = frag
        return acc.copy()

    @__parameter
    def add_output(
        mut acc: OutMma.AccumType,
        src: MutPointer[Float32, MutAnyOrigin],
    ):
        comptime for ni in range(DEPTH_MMAS):
            var frag = acc[ni]
            comptime for el in range(8):
                var row = rb + (8 if el > 3 else 0)
                var col = ni * 16 + cb + (el & 3)
                frag[el] = frag[el] + src[row * OUT_N + col]
            acc[ni] = frag

    # Add each tile's PV contribution AFTER the update's rescale, mirroring the
    # kernel order (rescale old O, then add new P@V).
    var s0 = load_scores(s0_ptr)
    _softmax_update[NUM_N_MMAS, DEPTH_MMAS](sm_m, sm_l, s0, output)
    add_output(output, o0_ptr)

    var s1 = load_scores(s1_ptr)
    _softmax_update[NUM_N_MMAS, DEPTH_MMAS](sm_m, sm_l, s1, output)
    add_output(output, o1_ptr)

    _softmax_normalize[DEPTH_MMAS](sm_l, output)

    # Each lane owns distinct (row, col) pairs, so all lanes write without races.
    comptime for ni in range(DEPTH_MMAS):
        var frag = output[ni]
        comptime for el in range(8):
            var row = rb + (8 if el > 3 else 0)
            var col = ni * 16 + cb + (el & 3)
            out_ptr[row * OUT_N + col] = frag[el]


def test_apple_fa_softmax_unit(ctx: DeviceContext) raises:
    print("== test_apple_fa_softmax_unit (2-tile online softmax + normalize)")

    comptime n_s = SQ * SK
    comptime n_o = SQ * OUT_N

    var s0_h = ctx.enqueue_create_host_buffer[.float32](n_s)
    var s1_h = ctx.enqueue_create_host_buffer[.float32](n_s)
    var o0_h = ctx.enqueue_create_host_buffer[.float32](n_o)
    var o1_h = ctx.enqueue_create_host_buffer[.float32](n_o)

    # Deterministic, non-monotonic scores so the max is not the last column and
    # tile-1's max exceeds tile-0's for some rows (exercising the correction).
    for r in range(SQ):
        for c in range(SK):
            var v0 = Float32(((r * 37 + c * 101) % 211) - 105) * 0.05
            var v1 = Float32(((r * 71 + c * 53) % 197) - 98) * 0.05
            s0_h[r * SK + c] = v0
            s1_h[r * SK + c] = v1
    for r in range(SQ):
        for c in range(OUT_N):
            o0_h[r * OUT_N + c] = Float32(((r * 13 + c * 7) % 100)) * 0.1
            o1_h[r * OUT_N + c] = Float32(((r * 29 + c * 3) % 100)) * 0.1 - 5.0

    var s0_d = ctx.enqueue_create_buffer[.float32](n_s)
    var s1_d = ctx.enqueue_create_buffer[.float32](n_s)
    var o0_d = ctx.enqueue_create_buffer[.float32](n_o)
    var o1_d = ctx.enqueue_create_buffer[.float32](n_o)
    var out_d = ctx.enqueue_create_buffer[.float32](n_o)
    ctx.enqueue_copy(s0_d, s0_h)
    ctx.enqueue_copy(s1_d, s1_h)
    ctx.enqueue_copy(o0_d, o0_h)
    ctx.enqueue_copy(o1_d, o1_h)

    ctx.enqueue_function[_softmax_unit_kernel](
        s0_d.unsafe_ptr(),
        s1_d.unsafe_ptr(),
        o0_d.unsafe_ptr(),
        o1_d.unsafe_ptr(),
        out_d.unsafe_ptr(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    var out_h = ctx.enqueue_create_host_buffer[.float32](n_o)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = s0_d^
    _ = s1_d^
    _ = o0_d^
    _ = o1_d^
    _ = out_d^

    # Host reference: flash-attention online softmax over the two score tiles with
    # the kernel's "add PV after rescale" recurrence. It must mirror the kernel's
    # `exp2` exactly: O is injected raw (not P-weighted) while `l` carries the
    # base-2 weighting, so any base mismatch leaks a per-row factor into O/l.
    # For row r:
    #   m=-inf,l=0,O=0
    #   tile t: m_t=rowmax(s_t); m_new=max(m,m_t); a=exp2(m-m_new)
    #           P=exp2(s_t - m_new); l=l*a + sum(P); O=O*a + o_t
    #           m=m_new
    #   final: O /= l
    var pass_ = True
    for r in range(SQ):
        var m = Float32(-3.0e38)
        var l = Float32(0)
        var o_ref = List[Float32](length=OUT_N, fill=0)

        # tile 0
        var m0 = Float32(-3.0e38)
        for c in range(SK):
            m0 = max(m0, s0_h[r * SK + c])
        var m_new = max(m, m0)
        var a = exp2(m - m_new)
        var psum = Float32(0)
        for c in range(SK):
            psum += exp2(s0_h[r * SK + c] - m_new)
        l = l * a + psum
        for c in range(OUT_N):
            o_ref[c] = o_ref[c] * a + o0_h[r * OUT_N + c]
        m = m_new

        # tile 1
        var m1 = Float32(-3.0e38)
        for c in range(SK):
            m1 = max(m1, s1_h[r * SK + c])
        m_new = max(m, m1)
        a = exp2(m - m_new)
        psum = Float32(0)
        for c in range(SK):
            psum += exp2(s1_h[r * SK + c] - m_new)
        l = l * a + psum
        for c in range(OUT_N):
            o_ref[c] = o_ref[c] * a + o1_h[r * OUT_N + c]

        for c in range(OUT_N):
            var expected = o_ref[c] / l
            var got = out_h[r * OUT_N + c]
            if abs(got - expected) > Float32(1e-3) * (1.0 + abs(expected)):
                print("FAIL row", r, "col", c, "got", got, "exp", expected)
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
    test_apple_fa_softmax_unit(ctx)
