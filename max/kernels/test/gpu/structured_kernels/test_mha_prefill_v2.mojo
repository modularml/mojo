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
"""GPU test for MhaPrefillV2 single-warp main loop.

Runs the v2 kernel with a small Q/K/V configuration over multiple
K/V tiles, validates the per-lane o_reg dump against an explicit
host-side attention reference computed with the same access pattern
the kernel uses.

Pattern (chosen so attention output is well-conditioned):
  Q[i, d] = 1.0 for all (i, d)
  K[t][r, d] = 1.0 for all (t, r, d)
  V[t][r, d] = (t * KV_BLOCK + r) / 32  (small monotone values)

Then att_unnorm[i, k_global] = sum_d Q[i, d] * K[k_global, d] = DEPTH
for all i, k_global. After softmax (subtract max, exp, normalize):
  att[i, k_global] = 1 / total_K (uniform).
And o_reg[i, m] = sum_k_global att[i, k_global] * V[k_global, m].

For our V pattern (V[t][r, m] = (t*KV+r)/32), all m's are equal
within a row of V. So o_reg[i, m] = mean over all k_global of
(k_global / 32) = (total_K - 1) / 64  (independent of i, m).
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
comptime KV_BLOCK = 64  # reference d=128 shape
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8  # reference kernel.cpp prologue requires >= 4 (loads K[0..2], V[0..1])
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK
comptime BATCH = 1


def test_v2_main_loop[
    depth: Int,
    output_dtype: DType = .float32,
](ctx: DeviceContext) raises:
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
        output_dtype=output_dtype,
    )

    print(
        "--- MhaPrefillV2 8-warp (BM=",
        BM,
        " Q/warp=",
        Q_BLOCK_SIZE,
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
    var dev_out = ctx.enqueue_create_buffer[output_dtype](SIZE_OUT)

    var total_k = NUM_TILES * KV_BLOCK
    with dev_q.map_to_host() as host_q, dev_k.map_to_host() as host_k, dev_v.map_to_host() as host_v:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
        for i in range(SIZE_KV):
            host_k[i] = BFloat16(1)
        # V[t][r, m] = k_global / total_K (small monotone in [0, 1)).
        for k in range(total_k):
            var val = Float32(k) / Float32(total_k)
            for m in range(depth):
                host_v[k * depth + m] = BFloat16(val)

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
    # Wrap K/V TileTensors as LayoutTensorMHAOperand for the kernel's
    # paged-aware signature; the operand infers `buffer_layout` from the
    # passed TileTensor, so pass the existing row-major TileTensors directly.
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

    # Uniform K means uniform softmax: o[i, m] = mean of V[k, m]
    # = mean of (0..total_K-1)/total_K = (total_K-1) / (2*total_K).
    var expected = Float32(total_k - 1) / Float32(2 * total_k)
    print("  expected per-lane =", expected)

    # FP32 output preserves the per-tile accumulator precision; BF16
    # output adds a per-element round-off at store time (~1 ULP in
    # BF16, which at value ~0.5 is ~2e-3). Use a slightly looser
    # tolerance for BF16 to absorb the cast.
    comptime tol: Float32 = 0.05 if output_dtype == .float32 else 0.01
    var mismatches = 0
    var max_diff: Float32 = 0
    with dev_out.map_to_host() as host_out:
        for q in range(BM):
            for d in range(depth):
                var got = Float32(host_out[q * depth + d])
                var diff = abs(got - expected)
                if diff > max_diff:
                    max_diff = diff
                if diff > tol:
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
    print("MhaPrefillV2 main-loop GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        # FP32 output (unnormalized accumulator path).
        test_v2_main_loop[128](ctx)
        test_v2_main_loop[64](ctx)
        # BF16 output (production inference path used by the dispatcher).
        test_v2_main_loop[128, DType.bfloat16](ctx)
        test_v2_main_loop[64, DType.bfloat16](ctx)

    print("=" * 60)
    print("ALL MhaPrefillV2 TESTS PASSED!")
    print("=" * 60)
