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
"""MlaPrefillV2 output validation against an FP32 host reference.

Phase-1 correctness gate for the `MlaPrefillV2` port of the reference
integrated single-schedule MLA-prefill inner loop (gfx950). Buffer-setup
recipe: ONE random single-head K_nope `(B, NK, 1, 128)` + K_rope
`(B, NK, 1, 64)`, FP8
round-tripped through BF16 staging, written into a latent cache
`(B, NK, 1, 576)` with k_nope at cols [0,128), k_rope at [512,576), V =
the nope segment.

`MlaPrefillV2` only supports the FP8 / KV>=128 / 32x32x64 shape
(the reference integrated cadence target), so all cases here are FP8 KV=128.
Each case reports per-head cosine similarity vs a pure-FP32 host
reference computed from the same FP8-roundtripped data; the Phase-1
gate is cos_sim >= 0.99 (especially NullMask KV=128 — the cross-warp
band-share corruption detector).

REQUIRED comptime defines (the BUILD target sets these in `copts`; pass
them explicitly when running `mojo` directly):

    -D fp32_scores=true -D cadence=true

`MlaPrefillV2` delegates every numeric step to `MlaPrefillV2Core[config]`,
whose FP32-scores / reference-cadence path is gated behind
`MlaPrefillV2Core._FP32_SOFTMAX_SCORES` (OFF by default in the shipping
source). These defines enable that path for this compilation only —
they do NOT modify `mla_components.mojo` (the numeric closure that
mirrors the reference). Without them `_SOFTMAX_DTYPE` resolves to FP16 and
the delegated in-place-FP32 helpers are not wired, so the kernel will not
compile.
"""

from std.math import ceildiv, exp, rsqrt
from std.memory import alloc
from std.random import randn, seed
from max.gpu.host import DeviceContext
from std.sys import get_defined_bool, get_defined_int, get_defined_string

from layout import LayoutTensor, TileTensor
from layout.coord import Coord, Idx
from layout.runtime_layout import RuntimeLayout
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.mha_mma_op import MlaConfigV2
from nn.attention.gpu.amd_structured.mla_prefill_v2 import MlaPrefillV2
from nn.attention.gpu.amd_structured.ps_metadata import build_uniform
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand, MHAOperand


# ---- Shape constants. DeepSeek-V3 MLA at TP=4 production shard. -----
comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256 Q rows per block
comptime NUM_KV_HEADS = 1

comptime D_NOPE = 128  # = depth = d_pv
comptime D_ROPE = 64
comptime D_QK = D_NOPE + D_ROPE  # 192
comptime CACHE_DEPTH = 576
comptime ROPE_CACHE_OFFSET = 512  # = cache_depth - d_rope

# Define-driven so the test can be pointed at the exact bench shapes.
# Default 1 = CI behavior.
comptime BATCH = get_defined_int["batch_size", 1]()

# Per-kernel LLVM IGLP tuning, forwarded as `-mllvm` flags via the
# launcher's `compile_options`. MUST mirror `bench_mla_prefill_v2.mojo`
# so THIS file's asm dump reflects the SAME schedule the bench (perf
# path) compiles. `-D exact_solver=0` drops the exact solver
# (heuristic IGLP). `-D dump_asm=<path>` writes the GPU asm. Default 0
# here so the CI correctness test compiles with the default schedule
# (fast) — correctness is schedule-independent. Set
# `-D exact_solver=1` to compile this file's asm dump with the SAME
# exact-solver schedule the bench (perf path) uses.
comptime _EXACT_SOLVER = get_defined_int["exact_solver", 0]()
comptime _PREFILL_IGLP_OPTS: StaticString = "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false" if _EXACT_SOLVER != 0 else ""


def _is_finite(v: Float32) -> Bool:
    """Returns True iff `v` is finite (not NaN, not Inf)."""
    return v == v and abs(v) < Float32(1e38)


def _mla_naive_fp32_ref_chunked(
    host_q_src: ImmPointer[BFloat16, _],
    host_knope_src: ImmPointer[BFloat16, _],
    host_krope_src: ImmPointer[BFloat16, _],
    host_out_ref_fp32: MutPointer[Float32, _],
    batch: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    num_kv_heads: Int,
    d_nope: Int,
    d_rope: Int,
    d_qk: Int,
    scale: Float32,
    start_pos: Int,
    causal: Bool,
    chunk_size: Int,
):
    """Memory-efficient FP32 host-side MLA reference.

    Computes MLA attention output in FP32 from BF16 source buffers that
    already hold the FP8-roundtripped values the kernel sees. V is the
    K_nope segment (DeepSeek-V3 convention).
    """
    var heads_per_kv = num_heads // num_kv_heads
    var score_buf = alloc[Float32](chunk_size * num_keys)

    for b in range(batch):
        for h in range(num_heads):
            var kv_h = h // heads_per_kv
            var q_start = 0
            while q_start < seq_len:
                var q_end = q_start + chunk_size
                if q_end > seq_len:
                    q_end = seq_len
                var cur_chunk = q_end - q_start

                for qi in range(cur_chunk):
                    var q = q_start + qi
                    var q_row_base = ((b * seq_len + q) * num_heads + h) * d_qk
                    for k in range(num_keys):
                        var k_nope_base = (
                            (b * num_keys + k) * num_kv_heads + kv_h
                        ) * d_nope
                        var k_rope_base = (
                            (b * num_keys + k) * num_kv_heads + kv_h
                        ) * d_rope
                        var s_nope: Float32 = 0
                        for d in range(d_nope):
                            var q_v = Float32(host_q_src[q_row_base + d])
                            var k_v = Float32(host_knope_src[k_nope_base + d])
                            s_nope += q_v * k_v
                        var s_rope: Float32 = 0
                        for d in range(d_rope):
                            var q_v = Float32(
                                host_q_src[q_row_base + d_nope + d]
                            )
                            var k_v = Float32(host_krope_src[k_rope_base + d])
                            s_rope += q_v * k_v
                        var s = (s_nope + s_rope) * scale
                        if causal and k > q + start_pos:
                            s = Float32(-1e30)
                        score_buf[qi * num_keys + k] = s

                for qi in range(cur_chunk):
                    var row_base = qi * num_keys
                    var m = score_buf[row_base + 0]
                    for k in range(1, num_keys):
                        var s = score_buf[row_base + k]
                        if s > m:
                            m = s
                    var sum_exp: Float32 = 0
                    for k in range(num_keys):
                        var e = exp(score_buf[row_base + k] - m)
                        score_buf[row_base + k] = e
                        sum_exp += e
                    var inv_sum = Float32(1) / sum_exp
                    for k in range(num_keys):
                        score_buf[row_base + k] = (
                            score_buf[row_base + k] * inv_sum
                        )

                for qi in range(cur_chunk):
                    var q = q_start + qi
                    var out_row_base = (
                        (b * seq_len + q) * num_heads + h
                    ) * d_nope
                    var row_base = qi * num_keys
                    for d in range(d_nope):
                        host_out_ref_fp32[out_row_base + d] = Float32(0)
                    for k in range(num_keys):
                        var p = score_buf[row_base + k]
                        var v_base = (
                            (b * num_keys + k) * num_kv_heads + kv_h
                        ) * d_nope
                        for d in range(d_nope):
                            var v_val = Float32(host_knope_src[v_base + d])
                            host_out_ref_fp32[out_row_base + d] += p * v_val

                q_start = q_end

    score_buf.free()


@always_inline
def _mla_prefill_v2_launch[
    k_nope_t: MHAOperand,
    k_rope_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MlaConfigV2,
](
    q: TileTensor[mut=False, ...],
    k_nope_op: k_nope_t,
    k_rope_op: k_rope_t,
    v_op: v_t,
    o: TileTensor[mut=True, ...],
    mask_functor: mask_t,
    scale: Float32,
    num_keys: Int,
    start_pos: Int,
    work_indptr_ptr: ImmPointer[Int32, _],
    work_info_ptr: ImmPointer[Int32, _],
    num_works: Int,
    num_cu: Int,
    ctx: DeviceContext,
) raises:
    """In-test launcher for `MlaPrefillV2.run` (non-ragged).

    Lifting the `enqueue_function` call across a function boundary
    detaches the operand values from the test's local buffers and
    satisfies the borrow-check at the launch site.
    """
    comptime assert (
        q.dtype == config.dtype
    ), "_mla_prefill_v2_launch: `q.dtype` must equal `config.dtype`"
    comptime assert (
        o.dtype == config.output_dtype
    ), "_mla_prefill_v2_launch: `o.dtype` must equal `config.output_dtype`"

    var batch = Int(q.dim[0]())
    var seq_len = Int(q.dim[1]())

    comptime kernel = MlaPrefillV2[config]
    comptime kernel_run = kernel.run[
        k_nope_t,
        k_rope_t,
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

    comptime _DUMP_ASM: StaticString = get_defined_string["dump_asm", ""]()
    var compiled = ctx.compile_function[
        kernel_run,
        compile_options=_PREFILL_IGLP_OPTS,
        dump_asm=_DUMP_ASM,
    ]()
    # Persistent grid `(num_cu,)` when `-D persistent=true`; else the
    # static `(num_heads, ceildiv(seq, BM), batch)` grid. Both run the same
    # token-major body in `run` (selected by the same comptime flag).
    comptime _PERSISTENT = get_defined_bool["persistent", False]()
    var gx: Int
    var gy: Int
    var gz: Int
    comptime if _PERSISTENT:
        gx = num_cu
        gy = 1
        gz = 1
    else:
        gx = config.num_heads
        gy = ceildiv(seq_len, kernel.BM)
        gz = batch
    ctx.enqueue_function(
        compiled,
        q,
        k_nope_op,
        k_rope_op,
        v_op,
        o,
        mask_functor,
        scale,
        Int32(num_keys),
        Int32(start_pos),
        work_indptr_ptr,
        work_info_ptr,
        Int32(num_works),
        grid_dim=(gx, gy, gz),
        block_dim=kernel.NUM_THREADS,
    )


def test_mla_vs_fp32_ref[
    mask_t: MHAMask,
    //,
    qkv_type: DType,
    output_type: DType,
    kv_block: Int,
    num_heads: Int,
    seq_len: Int,
    is_causal: Bool,
](
    mask_functor: mask_t, mask_name: StaticString, ctx: DeviceContext
) raises -> Bool:
    """Runs `MlaPrefillV2[config].run` and compares per-head cos_sim
    against a pure-FP32 host reference computed from the same
    FP8-roundtripped data.

    Returns True iff min-over-heads cos_sim >= 0.99 (the Phase-1 gate).
    """
    comptime NUM_HEADS = num_heads
    comptime SEQ_LEN = seq_len
    comptime NUM_KEYS = seq_len

    print(
        "--- MlaPrefillV2 vs FP32 ref (qkv=",
        qkv_type,
        " out=",
        output_type,
        " mask=",
        mask_name,
        " kv_block=",
        kv_block,
        " heads=",
        NUM_HEADS,
        " seq=",
        SEQ_LEN,
        ") ---",
    )

    var scale = rsqrt(Float32(D_QK))

    comptime CONFIG = MlaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=kv_block,
        depth=D_NOPE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_qk=D_QK,
        d_rope=D_ROPE,
        cache_depth=CACHE_DEPTH,
        rope_cache_offset=ROPE_CACHE_OFFSET,
        num_warps=NUM_WARPS,
        dtype=qkv_type,
        output_dtype=output_type,
    )

    comptime SIZE_Q = BATCH * SEQ_LEN * NUM_HEADS * D_QK
    comptime SIZE_K_LATENT = BATCH * NUM_KEYS * NUM_KV_HEADS * CACHE_DEPTH
    comptime SIZE_OUT = BATCH * SEQ_LEN * NUM_HEADS * D_NOPE

    # ---- BF16 source buffers + FP8 round-trip --------------------------
    var host_q_src = ctx.enqueue_create_host_buffer[.bfloat16](SIZE_Q)
    var host_knope_src = ctx.enqueue_create_host_buffer[.bfloat16](
        BATCH * NUM_KEYS * NUM_KV_HEADS * D_NOPE
    )
    var host_krope_src = ctx.enqueue_create_host_buffer[.bfloat16](
        BATCH * NUM_KEYS * NUM_KV_HEADS * D_ROPE
    )
    ctx.synchronize()
    randn(host_q_src.as_span())
    randn(host_knope_src.as_span())
    randn(host_krope_src.as_span())

    # Scale to [-0.5, 0.5] before FP8 roundtrip (minimizes FP8 absolute
    # error in the QK accumulator).
    comptime scale_factor = BFloat16(0.5)

    var host_q = ctx.enqueue_create_host_buffer[qkv_type](SIZE_Q)
    for i in range(SIZE_Q):
        var q_bf16 = host_q_src[i] * scale_factor
        var q_t = q_bf16.cast[qkv_type]()
        host_q[i] = q_t
        host_q_src[i] = q_t.cast[.bfloat16]()

    var host_knope = ctx.enqueue_create_host_buffer[qkv_type](
        BATCH * NUM_KEYS * NUM_KV_HEADS * D_NOPE
    )
    for i in range(BATCH * NUM_KEYS * NUM_KV_HEADS * D_NOPE):
        var k_bf16 = host_knope_src[i] * scale_factor
        var k_t = k_bf16.cast[qkv_type]()
        host_knope[i] = k_t
        host_knope_src[i] = k_t.cast[.bfloat16]()

    var host_krope = ctx.enqueue_create_host_buffer[qkv_type](
        BATCH * NUM_KEYS * NUM_KV_HEADS * D_ROPE
    )
    for i in range(BATCH * NUM_KEYS * NUM_KV_HEADS * D_ROPE):
        var k_bf16 = host_krope_src[i] * scale_factor
        var k_t = k_bf16.cast[qkv_type]()
        host_krope[i] = k_t
        host_krope_src[i] = k_t.cast[.bfloat16]()

    # ---- Latent K cache build ------------------------------------------
    var host_k_latent = ctx.enqueue_create_host_buffer[qkv_type](SIZE_K_LATENT)
    for b in range(BATCH):
        for s in range(NUM_KEYS):
            for h in range(NUM_KV_HEADS):
                var row_base = (
                    (b * NUM_KEYS + s) * NUM_KV_HEADS + h
                ) * CACHE_DEPTH
                for d in range(D_NOPE):
                    host_k_latent[row_base + d] = host_knope[
                        ((b * NUM_KEYS + s) * NUM_KV_HEADS + h) * D_NOPE + d
                    ]
                for d in range(ROPE_CACHE_OFFSET - D_NOPE):
                    host_k_latent[row_base + D_NOPE + d] = Scalar[qkv_type](0)
                for d in range(D_ROPE):
                    host_k_latent[
                        row_base + ROPE_CACHE_OFFSET + d
                    ] = host_krope[
                        ((b * NUM_KEYS + s) * NUM_KV_HEADS + h) * D_ROPE + d
                    ]

    ctx.synchronize()

    # ---- Device buffers + copies --------------------------------------
    var dev_q = ctx.enqueue_create_buffer[qkv_type](SIZE_Q)
    var dev_k_latent = ctx.enqueue_create_buffer[qkv_type](SIZE_K_LATENT)
    var dev_out = ctx.enqueue_create_buffer[output_type](SIZE_OUT)
    ctx.enqueue_copy(dev_q, host_q)
    ctx.enqueue_copy(dev_k_latent, host_k_latent)
    ctx.synchronize()

    # ---- Persistent work partition (host-built, uploaded). With
    # `-D persistent=true`, `run` consumes it via the (num_cu,) grid;
    # otherwise it is threaded-but-unused (static grid, cos_sim-neutral).
    # For the persistent gate, `available_tgs = NUM_HEADS` puts one TG per head
    # -> a split-free partition at any seq (no reduce kernel needed). ----
    comptime _PERSISTENT = get_defined_bool["persistent", False]()
    comptime _AVAIL_TGS = NUM_HEADS if _PERSISTENT else 256
    var md = build_uniform(
        BATCH, Int32(SEQ_LEN), Int32(NUM_HEADS), Int32(_AVAIL_TGS)
    )
    var num_cu = len(md.work_indptr) - 1
    var n_indptr = len(md.work_indptr)
    var n_info = len(md.work_info)
    var host_work_indptr = ctx.enqueue_create_host_buffer[.int32](n_indptr)
    var host_work_info = ctx.enqueue_create_host_buffer[.int32](n_info)
    ctx.synchronize()
    for i in range(n_indptr):
        host_work_indptr[i] = md.work_indptr[i]
    for i in range(n_info):
        host_work_info[i] = md.work_info[i]
    var dev_work_indptr = ctx.enqueue_create_buffer[.int32](n_indptr)
    var dev_work_info = ctx.enqueue_create_buffer[.int32](n_info)
    ctx.enqueue_copy(dev_work_indptr, host_work_indptr)
    ctx.enqueue_copy(dev_work_info, host_work_info)
    ctx.synchronize()

    # ---- TileTensor views for the launch ------------------------------
    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN), Idx[NUM_HEADS], Idx[D_QK])
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(Int32(BATCH), Int32(SEQ_LEN), Idx[NUM_HEADS], Idx[D_NOPE])
        ),
    )
    var k_tt = TileTensor(
        dev_k_latent,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[CACHE_DEPTH],
            )
        ),
    )
    var k_nope_op = LayoutTensorMHAOperand(k_tt)
    var k_rope_op = LayoutTensorMHAOperand(k_tt)
    var v_tt = TileTensor(
        dev_k_latent,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(NUM_KEYS),
                Idx[NUM_KV_HEADS],
                Idx[CACHE_DEPTH],
            )
        ),
    )
    var v_op = LayoutTensorMHAOperand(v_tt)

    _mla_prefill_v2_launch[config=CONFIG](
        q_tt,
        k_nope_op,
        k_rope_op,
        v_op,
        o_tt,
        mask_functor,
        scale,
        NUM_KEYS,
        0,
        dev_work_indptr.unsafe_ptr(),
        dev_work_info.unsafe_ptr(),
        md.num_works,
        num_cu,
        ctx,
    )
    ctx.synchronize()

    # ---- Copy result back ---------------------------------------------
    var host_out = ctx.enqueue_create_host_buffer[output_type](SIZE_OUT)
    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    # ---- FP32 host reference (full sweep) -----------------------------
    var size_ref = BATCH * SEQ_LEN * NUM_HEADS * D_NOPE
    var host_out_ref_fp32 = alloc[Float32](size_ref)
    comptime CHUNK_SIZE = 64
    _mla_naive_fp32_ref_chunked(
        host_q_src.unsafe_ptr(),
        host_knope_src.unsafe_ptr(),
        host_krope_src.unsafe_ptr(),
        host_out_ref_fp32,
        BATCH,
        SEQ_LEN,
        NUM_KEYS,
        NUM_HEADS,
        NUM_KV_HEADS,
        D_NOPE,
        D_ROPE,
        D_QK,
        scale,
        0,
        is_causal,
        CHUNK_SIZE,
    )

    # ---- Per-head cosine similarity (kernel vs FP32 ref) --------------
    var dot = Array[Float64, NUM_HEADS](fill=Float64(0))
    var a_sq = Array[Float64, NUM_HEADS](fill=Float64(0))
    var r_sq = Array[Float64, NUM_HEADS](fill=Float64(0))
    var num_nonfinite = 0
    var max_diff: Float32 = 0

    for b in range(BATCH):
        for s in range(SEQ_LEN):
            for h in range(NUM_HEADS):
                for d in range(D_NOPE):
                    var idx = ((b * SEQ_LEN + s) * NUM_HEADS + h) * D_NOPE + d
                    var a_val = host_out[idx].cast[.float32]()
                    var r_val = host_out_ref_fp32[idx]
                    if not _is_finite(a_val):
                        num_nonfinite += 1
                        continue
                    if not _is_finite(r_val):
                        continue
                    var diff = abs(a_val - r_val)
                    if diff > max_diff:
                        max_diff = diff
                    dot[h] += Float64(a_val) * Float64(r_val)
                    a_sq[h] += Float64(a_val) * Float64(a_val)
                    r_sq[h] += Float64(r_val) * Float64(r_val)

    if num_nonfinite > 0:
        print("  WARN: ", num_nonfinite, " non-finite output values (NaN/Inf)")

    var min_cos: Float64 = 1.0
    print("  per-head cos_sim (kernel vs FP32 ref):")
    for h in range(NUM_HEADS):
        var a_norm = a_sq[h] ** Float64(0.5)
        var r_norm = r_sq[h] ** Float64(0.5)
        var cos_sim: Float64 = 0
        if a_norm > 0 and r_norm > 0:
            cos_sim = dot[h] / (a_norm * r_norm)
        if cos_sim < min_cos:
            min_cos = cos_sim
        print("    head=", h, " cos_sim=", Float32(cos_sim))

    var passed = min_cos >= 0.99 and num_nonfinite == 0
    if passed:
        print(
            "  PASSED (",
            mask_name,
            ") — min cos_sim=",
            Float32(min_cos),
            " max_diff=",
            max_diff,
        )
    else:
        print(
            "  FAILED (",
            mask_name,
            ") — min cos_sim=",
            Float32(min_cos),
            " (gate 0.99) max_diff=",
            max_diff,
            " nonfinite=",
            num_nonfinite,
        )

    host_out_ref_fp32.free()
    _ = host_q_src
    _ = host_knope_src
    _ = host_krope_src
    _ = host_q
    _ = host_knope
    _ = host_krope
    _ = host_k_latent
    _ = host_out
    _ = dev_q
    _ = dev_k_latent
    _ = dev_out
    return passed


def main() raises:
    print("=" * 60)
    print("MlaPrefillV2 Phase-1 correctness")
    print("=" * 60)

    seed(0)

    # `-D m0_shape=true` runs a SINGLE define-driven shape
    # (num_heads/seq_len/batch_size/m0_causal) so the test can validate
    # cos_sim at the exact bench shape (e.g. b1/8192 h16 Causal).
    # Default false = the 4-case CI suite.
    comptime _M0_SHAPE = get_defined_bool["m0_shape", False]()
    comptime if _M0_SHAPE:
        comptime _M0_HEADS = get_defined_int["num_heads", 16]()
        comptime _M0_SEQ = get_defined_int["seq_len", 8192]()
        comptime _M0_CAUSAL = get_defined_bool["m0_causal", True]()
        with DeviceContext() as ctx:
            print(
                "\n=== M0 bench-shape cos_sim: heads=",
                _M0_HEADS,
                " seq=",
                _M0_SEQ,
                " batch=",
                BATCH,
                " causal=",
                _M0_CAUSAL,
                " ===",
            )
            comptime if _M0_CAUSAL:
                var p = test_mla_vs_fp32_ref[
                    qkv_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    kv_block=128,
                    num_heads=_M0_HEADS,
                    seq_len=_M0_SEQ,
                    is_causal=True,
                ](CausalMask(), "M0 Causal", ctx)
                if not p:
                    raise Error("M0 shape cos_sim gate failed")
            else:
                var p = test_mla_vs_fp32_ref[
                    qkv_type=DType.float8_e4m3fn,
                    output_type=DType.bfloat16,
                    kv_block=128,
                    num_heads=_M0_HEADS,
                    seq_len=_M0_SEQ,
                    is_causal=False,
                ](NullMask(), "M0 NullMask", ctx)
                if not p:
                    raise Error("M0 shape cos_sim gate failed")
        return

    var all_passed: Bool = True
    with DeviceContext() as ctx:
        # NullMask KV=128 — the cross-warp band-share corruption
        # detector (the Phase-1 must-pass).
        print("\n=== FP8, NullMask, num_heads=32, KV=128 ===")
        var null_kv128 = test_mla_vs_fp32_ref[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=128,
            num_heads=32,
            seq_len=2048,
            is_causal=False,
        ](NullMask(), "FP8/NullMask KV=128", ctx)
        all_passed = all_passed and null_kv128

        # Causal KV=128.
        print("\n=== FP8, Causal, num_heads=32, KV=128 ===")
        var causal_kv128 = test_mla_vs_fp32_ref[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=128,
            num_heads=32,
            seq_len=2048,
            is_causal=True,
        ](CausalMask(), "FP8/Causal KV=128", ctx)
        all_passed = all_passed and causal_kv128

        # DSV-TP8 shard (num_heads=16) KV=128, both masks.
        print("\n=== FP8, NullMask, num_heads=16, KV=128 (DSV-TP8) ===")
        var null_kv128_h16 = test_mla_vs_fp32_ref[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=128,
            num_heads=16,
            seq_len=2048,
            is_causal=False,
        ](NullMask(), "FP8/NullMask KV=128 h16", ctx)
        all_passed = all_passed and null_kv128_h16

        print("\n=== FP8, Causal, num_heads=16, KV=128 (DSV-TP8) ===")
        var causal_kv128_h16 = test_mla_vs_fp32_ref[
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
            kv_block=128,
            num_heads=16,
            seq_len=2048,
            is_causal=True,
        ](CausalMask(), "FP8/Causal KV=128 h16", ctx)
        all_passed = all_passed and causal_kv128_h16

    print("=" * 60)
    if all_passed:
        print("MlaPrefillV2 Phase-1 correctness: PASSED")
    else:
        print("MlaPrefillV2 Phase-1 correctness: FAILED — see above")
    print("=" * 60)
    if not all_passed:
        raise Error("MlaPrefillV2 Phase-1 correctness gate failed")
