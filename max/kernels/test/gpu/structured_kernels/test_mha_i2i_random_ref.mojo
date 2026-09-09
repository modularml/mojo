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
"""Random-input regression gate for the MHA partial-K softmax dilution.

`test_mha_i2i_repro.mojo` uses UNIFORM inputs (`Q = K = 1`), which
masks the real FLUX i2i bug: with uniform K the valid scores (`~D*scale`)
dominate the SRD-clamp-zeroed OOB columns (score `Q@0 = 0`), so the OOB
dilution is negligible and the test passes even when the kernel is wrong.

With NORMALIZED (non-uniform) inputs the valid scores are `~N(0, 1)` and
score-0 OOB columns are mid-distribution — they take real softmax mass
with a zero numerator and pull the output toward zero. That is the actual
i2i corruption (it compounds over the denoise steps to SSIM ~0.88).

This test runs random bf16 Q/K/V vs a host softmax-attention reference
reading the same rounded bf16 values, at partial-K / partial-Q / both /
aligned shapes, and asserts the per-row max error stays at the bf16 noise
floor. Before the `_apply_kbound_mask_fast` fix, the partial-K cases miss
by ~0.04-0.055 (orders of magnitude over the floor); after, they match the
aligned control. Calls `mha_prefill_v2` DIRECTLY (bypasses the dispatcher
NullMask+partial-K gate) so it gates the kernel, not the dispatcher.
"""

from max.gpu.host import DeviceContext
from std.math import exp2 as math_exp2
from std.testing import assert_true

from layout import LayoutTensor, TileTensor
from layout.coord import Coord, Idx
from layout.runtime_layout import RuntimeLayout
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    mha_prefill_v2,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand


comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256
comptime KV_BLOCK = 64
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime BATCH = 1
comptime LOG2E = Float32(1.4426950408889634)
# A correct kernel matches the f32 host reference (reading the same bf16
# inputs) to the bf16 round-off floor (~2e-4 at depth 128). The partial-K
# dilution bug misses by ~4e-2. 5e-3 cleanly separates the two.
comptime TOL = Float32(5.0e-3)


# Deterministic LCG -> Float32 in [-1, 1) (no RNG-API/`Math.random` dep).
def _rng(mut state: UInt64) -> Float32:
    state = state * 6364136223846793005 + 1442695040888963407
    var top = (state >> 40) & 0xFFFFFF  # 24 bits
    return (Float32(top) / Float32(0x800000)) - 1.0


def run_case[depth: Int, seq_q: Int, num_keys: Int](ctx: DeviceContext) raises:
    comptime SIZE_Q = seq_q * depth
    comptime SIZE_KV = num_keys * depth
    comptime SIZE_OUT = seq_q * depth
    var scale = Float32(1.0) / (Float32(depth) ** Float32(0.5))

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
        output_dtype=DType.float32,
    )

    print(
        "--- depth=",
        depth,
        " seq_q=",
        seq_q,
        " num_keys=",
        num_keys,
        " (seq_q%256=",
        seq_q % BM,
        " num_keys%64=",
        num_keys % KV_BLOCK,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)
    var dev_ref = ctx.enqueue_create_buffer[.float32](SIZE_OUT)

    var st: UInt64 = 0x1234567
    with dev_q.map_to_host() as hq, dev_k.map_to_host() as hk, dev_v.map_to_host() as hv, dev_ref.map_to_host() as href:
        for i in range(SIZE_Q):
            hq[i] = BFloat16(_rng(st))
        for i in range(SIZE_KV):
            hk[i] = BFloat16(_rng(st))
        for i in range(SIZE_KV):
            hv[i] = BFloat16(_rng(st))

        # Host reference: NullMask softmax attention over [0, num_keys),
        # reading the rounded bf16 values so only the kernel algorithm
        # (not input rounding) drives the measured error.
        for i in range(seq_q):
            var m = Float32(-3.0e38)
            for k in range(num_keys):
                var s = Float32(0)
                for d in range(depth):
                    s += Float32(hq[i * depth + d]) * Float32(hk[k * depth + d])
                s *= scale
                if s > m:
                    m = s
            for d in range(depth):
                href[i * depth + d] = 0
            var denom = Float32(0)
            for k in range(num_keys):
                var s = Float32(0)
                for d in range(depth):
                    s += Float32(hq[i * depth + d]) * Float32(hk[k * depth + d])
                s *= scale
                var p = math_exp2((s - m) * LOG2E)
                denom += p
                for d in range(depth):
                    href[i * depth + d] += p * Float32(hv[k * depth + d])
            var inv = Float32(1.0) / denom
            for d in range(depth):
                href[i * depth + d] *= inv

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(Int32(BATCH), Int32(seq_q), Idx[NUM_HEADS], Idx[depth])
        ),
    )
    var k_tt = TileTensor(
        dev_k,
        row_major(
            Coord(Int32(BATCH), Int32(num_keys), Idx[NUM_KV_HEADS], Idx[depth])
        ),
    )
    var v_tt = TileTensor(
        dev_v,
        row_major(
            Coord(Int32(BATCH), Int32(num_keys), Idx[NUM_KV_HEADS], Idx[depth])
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(Int32(BATCH), Int32(seq_q), Idx[NUM_HEADS], Idx[depth])
        ),
    )
    var k_op = LayoutTensorMHAOperand(k_tt)
    var v_op = LayoutTensorMHAOperand(v_tt)

    mha_prefill_v2[CONFIG](
        q_tt, k_op, v_op, o_tt, NullMask(), scale, num_keys, 0, ctx
    )

    var max_err = Float32(0)
    var argq = 0
    with dev_out.map_to_host() as host_out, dev_ref.map_to_host() as href:
        for q in range(seq_q):
            for d in range(depth):
                var diff = abs(host_out[q * depth + d] - href[q * depth + d])
                if diff > max_err:
                    max_err = diff
                    argq = q

    print("  max row-err=", max_err, " at q=", argq, " (tol=", TOL, ")")
    assert_true(max_err < TOL, "kernel output diverged from reference")
    print("  PASSED")


def main() raises:
    print("=" * 64)
    print("MHA i2i random-input vs reference regression")
    print("=" * 64)
    with DeviceContext() as ctx:
        # Aligned, even tile count (N=8) — the t2i-like path.
        run_case[128, 256, 512](ctx)
        # Aligned, ODD tile count (N=5, no partial tile): isolates the
        # software-pipeline even/odd-tile-count parity bug from partial-K.
        run_case[128, 256, 320](ctx)
        # Partial-K, odd N=5 (uniform repro misses this).
        run_case[128, 256, 300](ctx)
        # Partial-Q only (aligned K) — the partial-Q path is clean.
        run_case[128, 300, 512](ctx)
        # Partial-Q AND partial-K (the full i2i situation).
        run_case[128, 300, 300](ctx)
        # The exact FLUX.2-dev i2i K extent: num_keys=8623 = 64*134 + 47
        # -> 135 K tiles (odd) with a 47/64 partial last tile. Small
        # seq_q keeps the host reference cheap while exercising the real
        # FLUX K-side tile geometry.
        run_case[128, 256, 8623](ctx)
        # depth=64: at depth < 128 the K DMA splits each 32-row sub-block
        # across 2 warps (16-row half-strips); the upper strip must carry
        # the `st_32x32_s` swizzle's worker-bit-6 offset or `load_K` reads
        # rows 16-31 of every sub-block from the wrong banks (a separate,
        # shape-independent ~0.05 error the uniform unit test misses).
        run_case[64, 256, 512](ctx)  # aligned even — pure half-strip swizzle
        run_case[64, 256, 320](ctx)  # aligned odd — swizzle + parity
        run_case[64, 256, 300](ctx)  # partial-K odd — swizzle + parity + OOB
    print("=" * 64)
    print("ALL MHA i2i RANDOM-REF TESTS PASSED!")
    print("=" * 64)
