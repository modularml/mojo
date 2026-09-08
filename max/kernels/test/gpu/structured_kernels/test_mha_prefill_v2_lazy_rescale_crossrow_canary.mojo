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
"""Cross-row leak in AMD `MhaPrefillV2`'s lazy-rescale gate (CDNA4).

`OnlineSoftmax.lazy_rescale_decision` (`mha_softmax.mojo`) decided whether to
adopt a grown running max with a wave-wide AND-ballot
(`warp_vote[.uint64](lane_ok) == 0xFFFFFFFFFFFFFFFF`) instead of each lane's
own predicate. Per that struct's own docstring, "the col_l rt_32x32
accumulator topology gives each lane ownership of one column of the 32x32
fragment, which corresponds to one Q row in the warp's stripe" -- i.e. the 32
lanes of a half-wavefront hold 32 INDEPENDENT Q rows (redundantly duplicated
across the other half), not a single shared row. A sibling row whose growth
crosses `RESCALE_THRESHOLD` forces every other row in the wavefront onto the
"adopt" branch too, which applies a real (non-identity) `exp2` rescale to
`o_reg`/`att_bf16_full` and permanently updates `max_vec`/`max_vec_prev` for
a row whose own growth never asked for it.

Reuses `test_mha_prefill_v2_rescale.mojo`'s exact geometry (`Q_BLOCK_SIZE=32,
NUM_WARPS=8, BM=256, KV_BLOCK=64, NUM_TILES=8, depth=128` -- one warp's Q
rows are exactly `[warp_id*32, warp_id*32+32)`) but drives Q PER-ROW instead
of uniformly: row 0 (observed) always gets a small Q magnitude so its own
tile-0->tile-1 growth (`128 * Q_scale`) stays under `RESCALE_THRESHOLD=8`;
one other row in warp 0 (picked per-seed) gets Q=1.0, whose growth (128) is
far over threshold and forces the wave-wide AND false for the whole warp.
Row 0's Q is byte-identical between the "clean" run (every row in the warp
shares row 0's small Q, so the wave-vote is unanimous and never coerces
anyone) and the "poisoned" run (one sibling forces the vote); a per-lane gate
must produce identical output for row 0 in both.

This kernel is prefill-only (`mha_prefill_v2` has no decode/speculative
variant), so there is no draft-vs-committed-token asymmetry to exploit --
the poison is a same-tile sibling row with different Q magnitude instead of
a never-committed draft token, mirroring the same construction used for the
SM100 depth512 `fuse_gqa` canary (also prefill-only).

CONFIRMED FAILING pre-fix on AMD (MI355X / CDNA4): reproduces for every
seed tried (a single poisoned row's growth of 128 log2-units unconditionally
exceeds the threshold of 8, so the wave-wide AND is always forced false --
this is a deterministic construction, not a data-dependent one, so seed
variation here covers *which* sibling row is poisoned rather than whether
the defect fires at all).
"""

from std.random import random_ui64, seed
from std.testing import assert_equal, assert_true

from layout import LayoutTensor, TileTensor
from layout.coord import Coord, Idx
from layout.runtime_layout import RuntimeLayout
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

from nn.attention.gpu.amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    mha_prefill_v2,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand

comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE
comptime KV_BLOCK = 64
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1
comptime DEPTH = 128

# growth (raw units, tile0->tile1) = DEPTH * (K1 - K0) * q_scale = 128 * q_scale.
# RESCALE_THRESHOLD defaults to 8.0, so 0.03 (growth ~3.84) stays a safe
# "skip" margin and 1.0 (growth 128) is unconditionally a "must rescale".
comptime Q_SMALL = Float32(0.03)
comptime Q_POISON = Float32(1.0)


def _run(poison_row: Int, ctx: DeviceContext) raises -> Float32:
    """Builds Q with `poison_row` at Q_POISON and every other row at
    Q_SMALL (a `poison_row` outside `[0, BM)` poisons nothing), runs
    `mha_prefill_v2`, and returns the sum of row 0's output elements.
    Row 0's `depth` output elements move in lockstep under this
    construction (a uniform rescale multiplies every element by the
    same factor), so the sum is bit-for-bit as sensitive as an
    element-wise comparison here."""
    comptime SIZE_Q = BM * DEPTH
    comptime SIZE_KV = NUM_TILES * KV_BLOCK * DEPTH
    comptime SIZE_OUT = BM * DEPTH

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)

    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for q in range(BM):
            var q_val = Q_POISON if q == poison_row else Q_SMALL
            for d in range(DEPTH):
                host_q[q * DEPTH + d] = BFloat16(q_val)
        for t in range(NUM_TILES):
            var k_val = BFloat16(1) if t == 0 else BFloat16(2)
            for r in range(KV_BLOCK):
                for d in range(DEPTH):
                    host_k[(t * KV_BLOCK + r) * DEPTH + d] = k_val
        for t in range(NUM_TILES):
            for r in range(KV_BLOCK):
                var v_val = Float32(t * KV_BLOCK + r) / Float32(32)
                for m in range(DEPTH):
                    host_v[(t * KV_BLOCK + r) * DEPTH + m] = BFloat16(v_val)

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN), Idx[NUM_HEADS], Idx[DEPTH])
        ),
    )
    var k_tt = TileTensor(
        dev_k,
        row_major(
            Coord(Int32(BATCH), Int32(NUM_KEYS), Idx[NUM_KV_HEADS], Idx[DEPTH])
        ),
    )
    var v_tt = TileTensor(
        dev_v,
        row_major(
            Coord(Int32(BATCH), Int32(NUM_KEYS), Idx[NUM_KV_HEADS], Idx[DEPTH])
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN), Idx[NUM_HEADS], Idx[DEPTH])
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
        NUM_KEYS,
        0,  # start_pos
        ctx,
    )

    var row0_sum = Float32(0)
    with dev_out.map_to_host() as host_out:
        for d in range(DEPTH):
            row0_sum += host_out[0 * DEPTH + d]
    _ = dev_q
    _ = dev_k
    _ = dev_v
    _ = dev_out
    return row0_sum


def test_lazy_rescale_crossrow_canary(ctx: DeviceContext, rng_seed: Int) raises:
    """Row 0's output must not depend on which OTHER row in its warp
    triggers the lazy-rescale gate."""
    seed(rng_seed)
    # A random row in [1, Q_BLOCK_SIZE) -- warp 0's stripe, excluding the
    # observed row 0 itself.
    var poison_row = Int(random_ui64(1, UInt64(Q_BLOCK_SIZE - 1)))
    print(
        "  [AMD lazy-rescale crossrow canary] seed=",
        rng_seed,
        " poison_row=",
        poison_row,
        sep="",
    )

    # Clean run: EVERY row shares row 0's small Q (poison_row = -1, which
    # never matches a real row index), so the wave-vote is unanimous and
    # never coerces anyone.
    var clean = _run(-1, ctx)
    var poisoned = _run(poison_row, ctx)

    print("     row0 sum clean=", clean, " poisoned=", poisoned, sep="")
    assert_equal(
        clean,
        poisoned,
        (
            "row 0's output changed when a sibling row in its warp"
            " triggered the lazy-rescale gate -- cross-row leak in the"
            " kernel"
        ),
    )


def main() raises:
    with DeviceContext() as ctx:
        var failures = List[String]()
        for s in range(24):
            try:
                test_lazy_rescale_crossrow_canary(ctx, s)
            except e:
                failures.append(String("seed=") + String(s))
                print("     CANARY FAILED (seed=", s, "): ", e, sep="")

        assert_true(
            len(failures) == 0,
            (
                "AMD lazy-rescale crossrow canary failed for: "
                + String(failures)
            ),
        )
