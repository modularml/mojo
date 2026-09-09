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
"""Regression gate for the MHA partial-K-tile SRD over-read.

When `num_keys % KV_BLOCK != 0`, the last K/V tile is partially valid.
Before the Tier-2 SRD fix, `_dma_k` / `_dma_v` sized the per-tile
buffer-resource `num_records` for a full `KV_BLOCK`-row tile, so the
DMA read past `num_keys` into adjacent rows. With `CausalMask` the
causal cap masks those OOB columns to `-inf` before softmax, hiding
the leak; with `NullMask` (FLUX.2-dev i2i: `seq_len=8623 = 64*134 + 47`)
the OOB K/V leak into the output and corrupt it.

This test calls `mha_prefill_v2` DIRECTLY (bypassing the dispatcher's
NullMask+partial-K gate) so it is a real regression gate for the kernel
fix, independent of whether the dispatcher gate is present.

Deterministic construction (no reliance on undefined adjacent memory):

  * Q seq_len = BM (one aligned Q block — isolates the K/V partial-tile
    path from any partial-Q-tile behavior).
  * num_keys is partial (`% KV_BLOCK != 0`). K/V are over-allocated to
    `n_alloc = ceildiv(num_keys, KV_BLOCK) * KV_BLOCK` rows so the buggy
    over-read lands on KNOWN sentinel rows inside the allocation rather
    than on undefined memory past the buffer (which made the original
    repro non-deterministic).
  * Valid keys `[0, num_keys)`: `K = 1`, `V[k] = k / num_keys`.
    With `Q = 1` and NullMask, valid keys get a uniform softmax, so the
    correct (OOB-zeroed) output is the analytic mean
    `(num_keys - 1) / (2 * num_keys)`.
  * OOB rows `[num_keys, n_alloc)`: filled with a sentinel. Two
    variants isolate the two DMAs:
      - V-leak: `K_oob = 1` (OOB keys take equal softmax weight),
        `V_oob` large — a stale-V read inflates the output mean.
      - K-leak: `K_oob` large (OOB keys DOMINATE the softmax),
        `V_oob` a distinct value — a stale-K read steers the whole
        output toward `V_oob`.

  After the fix both K and V OOB lanes hardware-zero, so the output
  equals the analytic mean for every variant. Before the fix the
  output deviates by orders of magnitude.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal

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


comptime Q_BLOCK_SIZE = 32  # per warp
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256 Q rows per block
comptime KV_BLOCK = 64  # MHA d=128 shape


def _ceil_to_block(n: Int, block: Int) -> Int:
    return ((n + block - 1) // block) * block


def run_partial_k_case[
    num_heads: Int,
    depth: Int,
](
    label: StaticString,
    num_keys: Int,
    k_oob: Float32,
    v_oob: Float32,
    ctx: DeviceContext,
    k_valid: Float32 = 1.0,
) raises:
    """One partial-K regression point (see module docstring).

    Q is one BM-aligned block; `num_keys` keys are tiled by `KV_BLOCK`,
    and K/V are over-allocated to the next `KV_BLOCK` multiple so the
    buggy over-read lands on the sentinel rows `[num_keys, n_alloc)`.
    """
    comptime BATCH = 1
    comptime NUM_KV_HEADS = num_heads  # group = 1 (MHA, no GQA)
    comptime SEQ_LEN_Q = BM  # one aligned Q block — no partial-Q confound

    var n_alloc = _ceil_to_block(num_keys, KV_BLOCK)
    var num_oob = n_alloc - num_keys

    var q_size = SEQ_LEN_Q * num_heads * depth
    var kv_size = n_alloc * NUM_KV_HEADS * depth
    var o_size = q_size

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=num_heads,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
        output_dtype=DType.bfloat16,
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](q_size)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](kv_size)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](kv_size)
    var dev_out = ctx.enqueue_create_buffer[.bfloat16](o_size)

    # Q = 1 everywhere → uniform Q@K over the (uniform) valid keys.
    with dev_q.map_to_host() as host_q:
        for i in range(q_size):
            host_q[i] = BFloat16(1)

    # K: valid rows = 1; OOB rows = k_oob. V: valid row k = k/num_keys;
    # OOB rows = v_oob. Row r spans NUM_KV_HEADS*depth contiguous elts
    # (all heads share the same per-row value here).
    var inv_nk = Float32(1) / Float32(num_keys)
    with dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for r in range(n_alloc):
            var is_oob = r >= num_keys
            var k_val = k_oob if is_oob else k_valid
            var v_val = v_oob if is_oob else (Float32(r) * inv_nk)
            var base = r * NUM_KV_HEADS * depth
            for c in range(NUM_KV_HEADS * depth):
                host_k[base + c] = BFloat16(k_val)
                host_v[base + c] = BFloat16(v_val)

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN_Q), Idx[num_heads], Idx[depth])
        ),
    )
    var k_tt = TileTensor(
        dev_k,
        row_major(
            Coord(Int32(BATCH), Int32(n_alloc), Idx[NUM_KV_HEADS], Idx[depth])
        ),
    )
    var v_tt = TileTensor(
        dev_v,
        row_major(
            Coord(Int32(BATCH), Int32(n_alloc), Idx[NUM_KV_HEADS], Idx[depth])
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN_Q), Idx[num_heads], Idx[depth])
        ),
    )

    var k_op = LayoutTensorMHAOperand(k_tt)
    var v_op = LayoutTensorMHAOperand(v_tt)

    mha_prefill_v2[CONFIG](
        q_tt,
        k_op,
        v_op,
        o_tt,
        NullMask(),
        Float32(1.0),
        num_keys,
        0,  # start_pos
        ctx,
    )

    # Correct output: the valid keys share one `k_valid` so their scores
    # are uniform → uniform softmax over [0, num_keys) →
    # mean of V[0..num_keys) = (num_keys - 1) / (2 * num_keys). Holds for
    # any `k_valid` (the shared score cancels in the softmax). A NEGATIVE
    # `k_valid` puts the valid scores BELOW the OOB columns' `Q@0 = 0`, so
    # the OOB columns DOMINATE the softmax (output → 0) unless they are
    # excluded — the stress case for the last-tile OOB masking.
    var expected = Float32(num_keys - 1) / Float32(2 * num_keys)

    var max_diff: Float32 = 0
    var mismatches = 0
    comptime tol: Float32 = 0.02
    with dev_out.map_to_host() as host_out:
        for i in range(o_size):
            var got = Float32(host_out[i])
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1

    print(
        "  [",
        label,
        "] H=",
        num_heads,
        " D=",
        depth,
        " num_keys=",
        num_keys,
        " (% KV_BLOCK=",
        num_keys % KV_BLOCK,
        ", oob_rows=",
        num_oob,
        ") expected=",
        expected,
        " max_diff=",
        max_diff,
        " mismatches=",
        mismatches,
        "/",
        o_size,
    )
    assert_almost_equal(Float32(mismatches), Float32(0))
    print("  PASSED")


def main() raises:
    print("=" * 64)
    print("MhaPrefillV2 partial-K-tile SRD regression (NullMask)")
    print("=" * 64)

    with DeviceContext() as ctx:
        # --- Partial-K: last tile has 47/64 valid keys (FLUX i2i shape). ---
        # V-leak variant: OOB keys take equal softmax weight; a stale-V
        # read inflates the output mean by ~num_oob*V_oob/num_keys.
        run_partial_k_case[num_heads=2, depth=128](
            "partial-K V-leak", num_keys=4143, k_oob=1.0, v_oob=1000.0, ctx=ctx
        )
        # K-leak variant: OOB keys dominate the softmax; a stale-K read
        # steers the entire output toward V_oob.
        run_partial_k_case[num_heads=2, depth=128](
            "partial-K K-leak", num_keys=4143, k_oob=8.0, v_oob=7.0, ctx=ctx
        )
        # depth=64 partial-K (the other supported MHA depth).
        run_partial_k_case[num_heads=2, depth=64](
            "partial-K d=64", num_keys=4143, k_oob=1.0, v_oob=1000.0, ctx=ctx
        )
        # FLUX.2-dev i2i exact sequence length (8623 = 64*134 + 47).
        run_partial_k_case[num_heads=2, depth=128](
            "FLUX i2i seq=8623", num_keys=8623, k_oob=1.0, v_oob=1000.0, ctx=ctx
        )
        # Softmax-dilution stress: valid scores NEGATIVE (k_valid<0), the
        # adversarial worst case for the partial last tile. The SRD clamp
        # zeroes the OOB columns' K AND V, so they add 0 to the numerator
        # (V_oob = 0) and only a bounded `Q@0 = 0` score to the denominator
        # — within tolerance for this SINGLE forward pass, even when every
        # valid score is below that 0. NOTE: this single-pass tolerance does
        # NOT imply E2E parity. The same bounded OOB denominator mass
        # compounds over FLUX's ~50 denoise steps to SSIM ~0.879 (< 0.99),
        # so the dispatcher still routes NullMask + partial-K to FA2. This
        # case guards the memory-safety/leak fix, not softmax fidelity.
        run_partial_k_case[num_heads=2, depth=128](
            "partial-K dilution (k_valid<0)",
            num_keys=4143,
            k_oob=1.0,
            v_oob=1000.0,
            ctx=ctx,
            k_valid=-0.1,
        )

        # --- Control: BM/KV-aligned num_keys (no partial tile). Must pass
        # both before and after the fix — guards against the clamp
        # breaking the common aligned path. ---
        run_partial_k_case[num_heads=2, depth=128](
            "aligned control", num_keys=4096, k_oob=1.0, v_oob=1000.0, ctx=ctx
        )

    print("=" * 64)
    print("ALL partial-K regression cases PASSED!")
    print("=" * 64)
