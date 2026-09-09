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
"""MhaPrefillV2 correctness test that ACTUALLY exercises rescaling.

The first v2 test used Q=K=1 and a softly-varying V — uniform att,
running max never grows, so the rescale path through `mul_col` /
`exp2(max_prev - max)` was effectively a no-op.

This test makes the per-tile max GROW so the rescaling path fires:

  Q[i, d] = 1
  K[t][r, d] = 1 in tile 0; 2 in tile t>0    (BF16 exact)
  V[t][r, m] = (t*KV + r) / 32                (small monotone)

  att score[k, j] = sum_d K[k, d] * Q[j, d] = DEPTH * K_val_for_k
    -> tile 0 scores = DEPTH = 128
    -> tile 1 scores = 2 * DEPTH = 256

After the eager-rescale fix in `_pv_strip_with_partial_softmax`,
MhaPrefillV2 matches the strict online-softmax ground truth:
exp2(scale*(max_prev - max_new)) zeroes out tile 0's contribution, and
tiles 1..N-1 all carry the same max (=256), so each contributes
weight 1/((N-1)*KV) to the normalized output. The expected per-lane
value is mean(V[KV..N*KV-1, m]).

Prior to the fix (PR #86745 widened the gate to expose this for FLUX),
the kernel's C2/C6 cluster interleaved PV strip 0 with the lazy rescale of
o_reg, then ran strips 1..3 AFTER the rescale while att_bf16 was
still at the OLD max scale — leaving tile 0's strips 1..3 surviving
the rescale at their old scale and contributing a bounded artifact
`mean(V[KV/4..KV-1, m])` to the output. The eager-rescale ordering
removes that artifact entirely.
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


comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE
comptime KV_BLOCK = 64
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8  # reference kernel.cpp prologue requires >= 4
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1


def test_v2_rescale[depth: Int](ctx: DeviceContext) raises:
    comptime SIZE_Q = BM * depth
    comptime SIZE_KV = NUM_TILES * KV_BLOCK * depth
    comptime SIZE_OUT = BM * depth

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
    )

    print(
        "--- MhaPrefillV2 rescale path (tile-1 max > tile-0 max, D=",
        depth,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)

    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
        # K[t=0] = 1, K[t>=1] = 2 (uniform within each tile).
        for t in range(NUM_TILES):
            var k_val = BFloat16(1) if t == 0 else BFloat16(2)
            for r in range(KV_BLOCK):
                for d in range(depth):
                    host_k[(t * KV_BLOCK + r) * depth + d] = k_val
        # V[t][r, m] = (t * KV + r) / 32.
        for t in range(NUM_TILES):
            for r in range(KV_BLOCK):
                var v_val = Float32(t * KV_BLOCK + r) / Float32(32)
                for m in range(depth):
                    host_v[(t * KV_BLOCK + r) * depth + m] = BFloat16(v_val)

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[depth],
            )
        ),
    )
    var k_tt = TileTensor(
        dev_k,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[depth],
            )
        ),
    )
    var v_tt = TileTensor(
        dev_v,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[depth],
            )
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[depth],
            )
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

    # Expected per-lane (strict online-softmax ground truth, eager-rescale).
    #
    # Numerator (o_reg before div_col): tile 0's contribution is
    # zeroed by the rescale (∆max ≈ 128 ≫ RESCALE_THRESHOLD=8, so
    # scale_vec = exp2(-128) ≈ 0 multiplies o_reg before any tile-1
    # strip lands). Tiles 1..NUM_TILES-1 carry the same max=256 and
    # all contribute V[KV..NUM_TILES*KV-1].
    #
    # Denominator (norm_vec): after the iter j=3 C2 rescale fires
    # (tile 1's max jumps from 128 → 256), `scale_vec` is reset to 1
    # by `_pv_strip_with_partial_softmax`'s else-branch on every
    # subsequent no-growth iter, so the epilogue's unconditional
    # `norm_vec *= scale_vec` is identity. Tiles 1..N-1 each
    # contribute `sum(P)=KV_BLOCK` to the denominator → total
    # `(NUM_TILES - 1) * KV_BLOCK`. Tile 0's denominator was zeroed
    # by C4's `norm_vec *= 0` when pending_scale=True fired.
    var sum_tiles_1_n: Float32 = 0
    for r in range(KV_BLOCK, NUM_TILES * KV_BLOCK):
        sum_tiles_1_n += Float32(r) / Float32(32)
    var expected: Float32 = sum_tiles_1_n / Float32((NUM_TILES - 1) * KV_BLOCK)
    print("  expected per-lane =", expected)

    var mismatches = 0
    var max_diff: Float32 = 0
    # Tolerance: BF16 cast slack. Eager rescale removes the prior
    # `mean(V[KV/4..KV-1])` artifact entirely, so the only residual is
    # the BF16 store/round at ~1e-2 of value magnitude.
    var TOL: Float32 = 0.02
    with dev_out.map_to_host() as host_out:
        for q in range(BM):
            for d in range(depth):
                var got = host_out[q * depth + d]
                var diff = abs(got - expected)
                if diff > max_diff:
                    max_diff = diff
                if diff > TOL:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            "MISMATCH q=",
                            q,
                            " d=",
                            d,
                            " got=",
                            got,
                            " expected=",
                            expected,
                        )

    print("  mismatches=", mismatches, " max_diff=", max_diff)
    assert_almost_equal(Float32(mismatches), Float32(0))
    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("MhaPrefillV2 rescale-path GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_v2_rescale[128](ctx)
        test_v2_rescale[64](ctx)

    print("=" * 60)
    print("ALL MhaPrefillV2 RESCALE TESTS PASSED!")
    print("=" * 60)
