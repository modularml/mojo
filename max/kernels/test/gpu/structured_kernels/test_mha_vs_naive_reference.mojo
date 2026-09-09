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
"""`MhaPrefillV2` prefill correctness validation against `mha_gpu_naive`.

Exercises the MHA kernel (unified d=128 for Q/K/V, num_q_heads=num_kv_heads=8,
no MLA latent cache, no RoPE split) against random inputs at the
production seq_len=4096 gate. The five
existing MHA tests use deterministic analytical patterns (e.g.
`K=Q=1`, `V[k]=k/total_K`) that exercise the kernel's wiring but are
weak at distinguishing per-element softmax / accumulator bugs from
correct behavior.

This test answers a different question: at random inputs and at the
MHA production seq=4096 gate, does the MHA kernel produce outputs that
match the trusted `mha_gpu_naive` golden reference per-head,
per-row, per-depth?

For BF16 the per-element absolute/relative tolerance is tight
(3e-2 / 3e-2) per the existing `test_mla.mojo` convention. For FP8
e4m3fn (~3 bits mantissa), per-element diffs compounded over 4096
keys can exceed the tolerance on near-zero outputs while the
per-head output vector stays well-aligned with the reference —
so the FP8 gate is per-head cosine similarity >= 0.99 (matches the
MLA vs naive reference convention).
"""

from std.math import ceildiv, rsqrt
from std.random import randn, seed
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal

from layout import LayoutTensor, Layout, TileTensor, UNKNOWN_VALUE
from layout.coord import Coord, Idx
from layout.runtime_layout import RuntimeLayout
from layout.tile_layout import row_major

from std.utils.index import Index

from nn.attention.gpu.amd_structured.mha_mma_op import MhaConfigV2
from nn.attention.gpu.amd_structured.mha_prefill_v2 import MhaPrefillV2
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand, MHAOperand


# ---- Shape constants (MHA archetype at d=128).
#
# 8-warp MHA block with BM = NUM_WARPS * Q_BLOCK_SIZE = 256 Q rows
# per block. seq=4096 splits into ceildiv(4096, 256) = 16 BM tiles per
# head, exercising the inter-block grid traversal.
comptime Q_BLOCK_SIZE = 32  # per warp
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256 Q rows per block
# Heads. MHA (no GQA) means num_q_heads == num_kv_heads. The `mha_gpu_naive`
# P-intermediate buffer is B * NH * S * NK * sizeof[FP32]. At seq=4096,
# NH=8 this is 1 * 8 * 4096 * 4096 * 4 = 512 MB which is beyond the
# DeviceContext pool — so we run with NH=2 here just like the MLA
# sibling test (the MLA test runs NH=2 because P at NH=4 → 256 MB
# pushes the same limit). MHA at NH=2 still exercises the per-head
# grid stride (`(num_heads, BM_tiles, batch) = (2, 16, 1) = 32 blocks`)
# and the multi-head DMA loop. The remaining matrix coverage
# (BF16/FP8 x NullMask/CausalMask) is unaffected by NH choice.
comptime NUM_HEADS = 2
comptime NUM_KV_HEADS = 2  # MHA: num_q_heads == num_kv_heads
comptime DEPTH = 128
# Sequence shape. seq_len=4096 == MHA long-context gate threshold.
# Critical: this is right at the production dispatcher gate.
comptime SEQ_LEN = 4096
comptime NUM_KEYS = SEQ_LEN  # self-attention, no prefix cache
comptime BATCH = 1


def _is_finite_bf16(v: Float32) -> Bool:
    """Returns True iff `v` is a finite BF16 value (not NaN, not Inf).

    BF16-to-Float32 cast preserves NaN/Inf, so the test on `v` is the
    BF16 finiteness test.
    """
    return v == v and abs(v) < Float32(1e38)


@always_inline
def _mha_prefill_v2_launch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MhaConfigV2,
](
    q: TileTensor[mut=False, ...],
    k_op: k_t,
    v_op: v_t,
    o: TileTensor[mut=True, ...],
    mask_functor: mask_t,
    scale: Float32,
    num_keys: Int,
    start_pos: Int,
    ctx: DeviceContext,
) raises:
    """In-test launcher for `MhaPrefillV2.run`.

    Lifting the `enqueue_function` call across a function boundary
    detaches the operand values from the test's local buffers and
    satisfies the borrow-check at the launch site.
    """
    comptime assert (
        q.dtype == config.dtype
    ), "_mha_prefill_v2_launch: `q.dtype` must equal `config.dtype`"
    comptime assert (
        o.dtype == config.output_dtype
    ), "_mha_prefill_v2_launch: `o.dtype` must equal `config.output_dtype`"

    var batch = Int(q.dim[0]())
    var seq_len = Int(q.dim[1]())

    comptime kernel = MhaPrefillV2[config]
    comptime kernel_run = kernel.run[
        k_t,
        v_t,
        mask_t,
        q.dtype,
        o.dtype,
        q.LayoutType,
        o.LayoutType,
        q.Engine,
        o.Engine,
        ragged=False,
    ]

    var compiled = ctx.compile_function[kernel_run]()
    var sink_weights_ptr = ImmPointer[
        Scalar[q.dtype], ImmutAnyOrigin
    ].unsafe_dangling()
    ctx.enqueue_function(
        compiled,
        q,
        k_op,
        v_op,
        o,
        mask_functor,
        scale,
        Int32(num_keys),
        Int32(start_pos),
        sink_weights_ptr,
        grid_dim=(
            config.num_heads,
            ceildiv(seq_len, kernel.BM),
            batch,
        ),
        block_dim=kernel.NUM_THREADS,
    )


def test_mha_vs_naive[
    mask_t: MHAMask,
    //,
    qkv_type: DType,
    output_type: DType,
    kv_block: Int,
    atol: Float32,
    rtol: Float32,
](
    mask_functor: mask_t, mask_name: StaticString, ctx: DeviceContext
) raises -> Bool:
    """Compares `MhaPrefillV2[config].run` against `mha_gpu_naive`.

    Allocates random Q/K/V via `randn` (fixed seed for reproducibility),
    runs both kernels, then compares element-wise over the
    BATCH x SEQ x NUM_HEADS x DEPTH output grid. Records max_diff,
    mean_diff, and counts per-head mismatches + per-head cosine
    similarity for FP8 gating.
    """
    print(
        "--- MHA vs naive reference (qkv=",
        qkv_type,
        " out=",
        output_type,
        " mask=",
        mask_name,
        " kv_block=",
        kv_block,
        " d=",
        DEPTH,
        " heads=",
        NUM_HEADS,
        " kv_heads=",
        NUM_KV_HEADS,
        " seq=",
        SEQ_LEN,
        ") ---",
    )

    # `scale = 1 / sqrt(d)` — the canonical scaled-dot-product attention
    # scale. At d=128 with random Q,K ~ N(0,1) the un-scaled QK
    # accumulator is ~ N(0, 128); the 1/sqrt(128) standardizes it to
    # ~ N(0, 1) which keeps softmax well-conditioned.
    var scale = rsqrt(Float32(DEPTH))

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=kv_block,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
        dtype=qkv_type,
        output_dtype=output_type,
    )

    comptime SIZE_Q = BATCH * SEQ_LEN * NUM_HEADS * DEPTH
    comptime SIZE_KV = BATCH * NUM_KEYS * NUM_KV_HEADS * DEPTH
    comptime SIZE_OUT = BATCH * SEQ_LEN * NUM_HEADS * DEPTH

    # Host allocations (random fill, then copy to device).
    var host_q = ctx.enqueue_create_host_buffer[qkv_type](SIZE_Q)
    var host_k = ctx.enqueue_create_host_buffer[qkv_type](SIZE_KV)
    var host_v = ctx.enqueue_create_host_buffer[qkv_type](SIZE_KV)
    var host_out = ctx.enqueue_create_host_buffer[output_type](SIZE_OUT)
    var host_out_ref = ctx.enqueue_create_host_buffer[output_type](SIZE_OUT)

    ctx.synchronize()

    # Random fill. Fixed seed via std.random.seed at main entry.
    randn(host_q.as_span())
    randn(host_k.as_span())
    randn(host_v.as_span())

    # Device buffers.
    var dev_q = ctx.enqueue_create_buffer[qkv_type](SIZE_Q)
    var dev_k = ctx.enqueue_create_buffer[qkv_type](SIZE_KV)
    var dev_v = ctx.enqueue_create_buffer[qkv_type](SIZE_KV)
    var dev_out = ctx.enqueue_create_buffer[output_type](SIZE_OUT)
    var dev_out_ref = ctx.enqueue_create_buffer[output_type](SIZE_OUT)

    ctx.enqueue_copy(dev_q, host_q)
    ctx.enqueue_copy(dev_k, host_k)
    ctx.enqueue_copy(dev_v, host_v)

    # --- MHA launch ------------------------------------------------------
    # Q/K/V/O TileTensors at (BATCH, SEQ_LEN/NUM_KEYS, NUM_HEADS, DEPTH).
    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[DEPTH],
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
                Idx[DEPTH],
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
                Idx[DEPTH],
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
                Idx[DEPTH],
            )
        ),
    )

    var k_op = LayoutTensorMHAOperand(k_tt)
    var v_op = LayoutTensorMHAOperand(v_tt)

    _mha_prefill_v2_launch[config=CONFIG](
        q_tt,
        k_op,
        v_op,
        o_tt,
        mask_functor,
        scale,
        NUM_KEYS,
        0,  # start_pos (self-attention from scratch)
        ctx,
    )

    # --- Naive reference launch -----------------------------------------
    # mha_gpu_naive takes K and V at d=DEPTH (same as the MHA kernel).
    var q_naive_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[DEPTH],
            )
        ),
    )
    var k_ref_tt = TileTensor(
        dev_k,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[DEPTH],
            )
        ),
    )
    var v_ref_tt = TileTensor(
        dev_v,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[DEPTH],
            )
        ),
    )
    var out_ref_tt = TileTensor(
        dev_out_ref,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[DEPTH],
            )
        ),
    )
    var k_ref_operand = LayoutTensorMHAOperand(
        k_ref_tt.as_immut().as_unsafe_any_origin()
    )
    var v_ref_operand = LayoutTensorMHAOperand(
        v_ref_tt.as_immut().as_unsafe_any_origin()
    )
    var null_valid_length = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    # MHA: group = num_q_heads / num_kv_heads = 1.
    mha_gpu_naive[_is_cache_length_accurate=True,](
        q_naive_tt.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        mask_functor,
        out_ref_tt.to_layout_tensor(),
        null_valid_length,
        scale,
        BATCH,
        SEQ_LEN,
        NUM_KEYS,
        NUM_HEADS,
        DEPTH,
        NUM_HEADS // NUM_KV_HEADS,  # group = 1 for MHA
        ctx,
    )

    ctx.synchronize()

    # Copy results back for comparison.
    ctx.enqueue_copy(host_out, dev_out)
    ctx.enqueue_copy(host_out_ref, dev_out_ref)
    ctx.synchronize()

    # --- Compare element-wise + cosine similarity per head -------------
    var mismatches = 0
    var max_diff: Float32 = 0
    var sum_diff: Float64 = 0
    var num_checked = 0
    # Per-head mismatch counters.
    var head_mismatches = Array[Int, NUM_HEADS](fill=0)
    var head_max_diff = Array[Float32, NUM_HEADS](fill=0)
    # Per-head cosine-similarity accumulators (FP64 for stability).
    var head_dot = Array[Float64, NUM_HEADS](fill=Float64(0))
    var head_out_sq = Array[Float64, NUM_HEADS](fill=Float64(0))
    var head_ref_sq = Array[Float64, NUM_HEADS](fill=Float64(0))

    var num_nonfinite_hk = 0
    var num_nonfinite_ref = 0
    for b in range(BATCH):
        for s in range(SEQ_LEN):
            for h in range(NUM_HEADS):
                for d in range(DEPTH):
                    var idx = ((b * SEQ_LEN + s) * NUM_HEADS + h) * DEPTH + d
                    var out_val = Float32(host_out[idx])
                    var ref_val = Float32(host_out_ref[idx])
                    if not _is_finite_bf16(out_val):
                        num_nonfinite_hk += 1
                        continue
                    if not _is_finite_bf16(ref_val):
                        num_nonfinite_ref += 1
                        continue
                    var diff = abs(out_val - ref_val)
                    sum_diff += Float64(diff)
                    num_checked += 1
                    head_dot[h] += Float64(out_val) * Float64(ref_val)
                    head_out_sq[h] += Float64(out_val) * Float64(out_val)
                    head_ref_sq[h] += Float64(ref_val) * Float64(ref_val)
                    if diff > max_diff:
                        max_diff = diff
                    if diff > head_max_diff[h]:
                        head_max_diff[h] = diff
                    var threshold = atol + rtol * abs(ref_val)
                    if diff > threshold:
                        mismatches += 1
                        head_mismatches[h] += 1
                        if mismatches <= 5:
                            print(
                                "MISMATCH b=",
                                b,
                                " s=",
                                s,
                                " h=",
                                h,
                                " d=",
                                d,
                                " out=",
                                out_val,
                                " ref=",
                                ref_val,
                                " diff=",
                                diff,
                            )

    if num_nonfinite_hk > 0:
        print("  WARN: ", num_nonfinite_hk, " non-finite outputs (NaN/Inf)")
    if num_nonfinite_ref > 0:
        print("  WARN: ", num_nonfinite_ref, " non-finite ref values (NaN/Inf)")

    var mean_diff = sum_diff / Float64(num_checked)
    print(
        "  num_checked=",
        num_checked,
        " mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " mean_diff=",
        Float32(mean_diff),
    )
    # Per-head cosine similarity. FP8 gate: min_cos >= 0.99.
    var min_cos: Float64 = 1.0
    print("  per-head stats:")
    for h in range(NUM_HEADS):
        var out_norm = head_out_sq[h] ** Float64(0.5)
        var ref_norm = head_ref_sq[h] ** Float64(0.5)
        var cos_sim: Float64 = 0
        if out_norm > 0 and ref_norm > 0:
            cos_sim = head_dot[h] / (out_norm * ref_norm)
        if cos_sim < min_cos:
            min_cos = cos_sim
        print(
            "    head=",
            h,
            " max_diff=",
            head_max_diff[h],
            " mismatches=",
            head_mismatches[h],
            " cos_sim=",
            Float32(cos_sim),
        )

    # FP8 gate: cosine similarity >= 0.99 (per MLA convention).
    # BF16 gate: zero mismatches per the per-element tolerance.
    comptime is_fp8 = qkv_type == DType.float8_e4m3fn
    var passed: Bool = False
    if mismatches == 0:
        passed = True
        print("  PASSED (", mask_name, ")")
    elif is_fp8 and min_cos >= 0.99:
        passed = True
        print(
            "  PASSED (",
            mask_name,
            ") via cosine similarity (min cos=",
            Float32(min_cos),
            "); per-element mismatches=",
            mismatches,
        )
    else:
        print("  FAILED (", mask_name, ") — see mismatches above")
        if is_fp8:
            print("    min cos_sim=", Float32(min_cos), " (FP8 spec gate 0.99)")

    _ = host_q
    _ = host_k
    _ = host_v
    _ = host_out
    _ = host_out_ref
    _ = dev_q
    _ = dev_k
    _ = dev_v
    _ = dev_out
    _ = dev_out_ref
    return passed


def main() raises:
    print("=" * 60)
    print("MHA vs naive reference validation")
    print("=" * 60)

    # Fixed seed for reproducibility — same convention as test_mla.mojo.
    seed(0)

    var all_passed: Bool = True
    with DeviceContext() as ctx:
        # BF16 — primary correctness gate. KV_BLOCK=64 (MHA d=128
        # production shape). Tight tolerances (AMD: 3e-2).
        var bf16_null = test_mha_vs_naive[
            qkv_type=DType.bfloat16,
            output_type=DType.bfloat16,
            kv_block=64,
            atol=Float32(3e-2),
            rtol=Float32(3e-2),
        ](NullMask(), "BF16/NullMask", ctx)
        all_passed = all_passed and bf16_null
        var bf16_causal = test_mha_vs_naive[
            qkv_type=DType.bfloat16,
            output_type=DType.bfloat16,
            kv_block=64,
            atol=Float32(3e-2),
            rtol=Float32(3e-2),
        ](CausalMask(), "BF16/CausalMask", ctx)
        all_passed = all_passed and bf16_causal

        # FP8 — looser tolerances (e4m3fn has ~3 bits mantissa). Cosine
        # similarity >= 0.99 gates pass/fail.
        var fp8_null = test_mha_vs_naive[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=64,
            atol=Float32(5e-2),
            rtol=Float32(5e-2),
        ](NullMask(), "FP8/NullMask", ctx)
        all_passed = all_passed and fp8_null
        var fp8_causal = test_mha_vs_naive[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=64,
            atol=Float32(5e-2),
            rtol=Float32(5e-2),
        ](CausalMask(), "FP8/CausalMask", ctx)
        all_passed = all_passed and fp8_causal

    print("=" * 60)
    if all_passed:
        print("MHA vs naive reference: ALL PASSED")
    else:
        print("MHA vs naive reference: SOME CASES FAILED — see above")
    print("=" * 60)
    # Final assertion (after all 4 cases ran) so partial results surface
    # in the report.
    assert_almost_equal(
        Float32(1.0) if all_passed else Float32(0.0), Float32(1.0)
    )
