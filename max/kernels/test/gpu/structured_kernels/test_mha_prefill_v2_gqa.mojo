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
"""MhaPrefillV2 GQA correctness test.

Exercises the per-(batch, head) view + head_idx interleave + kv_head_idx
mapping at production GQA ratios:

| Variant      | num_heads | num_kv_heads | GROUP | rectangle |
|--------------|----------:|-------------:|------:|-----------|
| Llama3.1 8B  |        32 |            8 |     4 | non-square|
| Llama3.1 70B |        64 |            8 |     8 | square    |
| Gemma3-ish   |         8 |            4 |     2 | non-square|

Analytic pattern: Q[b, q, h, d] = 1, K[b, k, kv, d] = 1, V[b, k, kv, d]
= kv (integer, BF16-exact). With CAUSAL=True and scale=1.0:

  att[k] = sum_d Q*K = DEPTH for k <= q else -inf
  prob[k|q] = 1/(q+1) for k <= q, 0 otherwise (sums to 1)
  o[b, q, h, d] = sum_k prob[k|q] * V[b, k, h//GROUP, d]
               = h // GROUP    (V is k-independent, prob sums to 1)

If `head_idx` or `kv_head_idx` is wrong, output positions will hold
the wrong kv value (or 0 from uninitialized output) and the test fails.
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
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand


comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256
comptime KV_BLOCK = 64
comptime NUM_TILES = 8  # 512 keys total
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1


def test_gqa[
    depth: Int,
    num_heads: Int,
    num_kv_heads: Int,
](ctx: DeviceContext) raises:
    comptime GROUP = num_heads // num_kv_heads
    comptime SIZE_Q = BATCH * SEQ_LEN * num_heads * depth
    comptime SIZE_KV = BATCH * NUM_KEYS * num_kv_heads * depth
    comptime SIZE_OUT = SIZE_Q

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_warps=NUM_WARPS,
    )

    print(
        "--- MhaPrefillV2 GQA (H=",
        num_heads,
        " Hkv=",
        num_kv_heads,
        " G=",
        GROUP,
        " D=",
        depth,
        " seq=",
        SEQ_LEN,
        " keys=",
        NUM_KEYS,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)

    # Zero-init output so any unwritten position reads as 0 (and an
    # `h // GROUP > 0` expected value would catch it).
    with dev_out.map_to_host() as host_out:
        for i in range(SIZE_OUT):
            host_out[i] = Float32(0)

    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
        for i in range(SIZE_KV):
            host_k[i] = BFloat16(1)
        for b in range(BATCH):
            for k in range(NUM_KEYS):
                for kv in range(num_kv_heads):
                    for d in range(depth):
                        var idx = (
                            (b * NUM_KEYS + k) * num_kv_heads + kv
                        ) * depth + d
                        host_v[idx] = BFloat16(Float32(kv))

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[num_heads],
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
                Idx[num_kv_heads],
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
                Idx[num_kv_heads],
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
                Idx[num_heads],
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
        CausalMask(),
        Float32(1.0),
        NUM_KEYS,
        0,  # start_pos
        ctx,
    )

    var mismatches = 0
    var max_diff: Float32 = 0
    var heads_with_any_mismatch = 0
    with dev_out.map_to_host() as host_out:
        for h in range(num_heads):
            var head_mismatch = False
            var expected = Float32(h // GROUP)
            for b in range(BATCH):
                for q in range(SEQ_LEN):
                    for d in range(depth):
                        var idx = (
                            (b * SEQ_LEN + q) * num_heads + h
                        ) * depth + d
                        var got = host_out[idx]
                        var diff = abs(got - expected)
                        if diff > max_diff:
                            max_diff = diff
                        if diff > 0.05:
                            mismatches += 1
                            head_mismatch = True
                            if mismatches <= 5:
                                print(
                                    "MISMATCH b=",
                                    b,
                                    " q=",
                                    q,
                                    " h=",
                                    h,
                                    " kv=",
                                    h // GROUP,
                                    " d=",
                                    d,
                                    " got=",
                                    got,
                                    " expected=",
                                    expected,
                                )
            if head_mismatch:
                heads_with_any_mismatch += 1

    print(
        "  mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " heads_with_mismatch=",
        heads_with_any_mismatch,
        "/",
        num_heads,
    )
    assert_almost_equal(Float32(mismatches), Float32(0))
    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("MhaPrefillV2 GQA GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        # Llama3.1 8B / Mistral 7B shape: H=32, Hkv=8, GROUP=4
        test_gqa[128, 32, 8](ctx)

        # Llama3.1 70B shape: H=64, Hkv=8, GROUP=8 (square — the
        # reference's default config; the formula was already correct for this
        # case, included for regression coverage).
        test_gqa[128, 64, 8](ctx)

        # Gemma3-ish shape: H=8, Hkv=4, GROUP=2 (smallest non-square
        # GQA — fastest iteration shape).
        test_gqa[128, 8, 4](ctx)

    print("=" * 60)
    print("ALL MhaPrefillV2 GQA TESTS PASSED!")
    print("=" * 60)
