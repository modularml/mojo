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
"""MhaPrefillV2 CAUSAL=True correctness test.

Single block, BM=256 Q rows (= 8 warps × 32). num_tiles=8 K/V tiles
(8 × KV_BLOCK=64 = 512 keys). Each warp owns Q rows
[w*32, (w+1)*32). With CAUSAL=True, each Q position q in [0, 256)
should only attend to K positions <= q.

Pattern: K=Q=1, V[k, m] = (k+1) / 512.
  Pre-mask att[k, q] = DEPTH = 128 for all (q, k).
  After mask: att[k, q] = 128 if k <= q else -inf.
  After softmax (subtract max=128): exp(0)=1 for valid k, exp(-inf)=0
    for masked k. col_sum = (q + 1).
  o[q, m] = sum_{k<=q} V[k, m] / (q + 1)
         = sum_{k=0..q} (k+1)/512 / (q+1)
         = (q+1)(q+2)/2 / 512 / (q+1)
         = (q+2) / 1024.
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
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8  # 512 keys total
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1


def test_v2_causal[depth: Int](ctx: DeviceContext) raises:
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
        "--- MhaPrefillV2 CAUSAL=True (BM=",
        BM,
        " KV=",
        KV_BLOCK,
        " D=",
        depth,
        " tiles=",
        NUM_TILES,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)

    var total_k = NUM_TILES * KV_BLOCK
    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
        for i in range(SIZE_KV):
            host_k[i] = BFloat16(1)
        for k in range(total_k):
            var v_val = Float32(k + 1) / Float32(512)
            for m in range(depth):
                host_v[k * depth + m] = BFloat16(v_val)

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
        CausalMask(),
        Float32(1.0),
        NUM_KEYS,
        0,  # start_pos
        ctx,
    )

    var mismatches = 0
    var max_diff: Float32 = 0
    with dev_out.map_to_host() as host_out:
        for q in range(BM):
            var expected = Float32(q + 2) / Float32(1024)
            for d in range(depth):
                var got = host_out[q * depth + d]
                var diff = abs(got - expected)
                if diff > max_diff:
                    max_diff = diff
                if diff > 0.05:
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
    print("MhaPrefillV2 CAUSAL GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_v2_causal[128](ctx)
        test_v2_causal[64](ctx)

    print("=" * 60)
    print("ALL CAUSAL TESTS PASSED!")
    print("=" * 60)
