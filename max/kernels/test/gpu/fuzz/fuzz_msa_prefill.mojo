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
#
# Fuzz target: MiniMax-M3 block-sparse attention PREFILL (`msa_sm100_prefill_plan`
# + `msa_sm100_prefill_run`, in Kernels/lib/msa/msa_2q.mojo). The deployment does
# chunked prefill, so this is on the hot path; only the decode variant
# (`fuzz_msa_decode`) was covered before. Distinct from the indexer (block
# selection, `fuzz_sparse_indexer_prefill`): this consumes the indexer's
# selected blocks (`q2k`) and computes the attention (QK . softmax . V) over a
# ragged Q batch via an on-device reverse-CSR + block-major forward + combine.
#
# The full path is the two-stage production op:
#   plan = msa_sm100_prefill_plan(...)   # host sizing + allocation, ONCE
#   msa_sm100_prefill_run(plan, o, lse, q, k, v, q2k, cu_q, cu_k, scale, ...)
#     # device reverse-CSR build -> block-major fwd (O/LSE partials per
#     # (query, split-slot)) -> combine (LSE-merge) -> final O.
#
# The fuzzable surface is the *ragged shape*, which drives every device index
# and the CSR work decomposition (split-K over shared rows + q-chunking):
#   - batch + per-batch seqlen_q / seqlen_k (cu_seqlens_q / cu_seqlens_k): the
#     ragged Q/K extents, the block count per batch, and the per-row sharing
#     (high-sharing rows q-chunk into multiple CTAs);
#   - the q2k block selection (deterministic from a seed: distinct in-range
#     blocks per (kv-head, query); a slot left -1 is the C==0 empty-split edge).
#   - topk / num_q_heads / group / causal / Q-K-V dtype are compile-time `-D`
#     (the plan/run take topk + config + group + use_causal as comptime params);
#     head_dim is fixed at the M3 dense value 128.
#
#   -D msap_topk=8 -D msap_num_q_heads=1 -D msap_group=1 -D msap_causal=0
#       [default: M3 head_kv=1, full-block, non-causal]
#   -D msap_topk=16 -D msap_num_q_heads=8 -D msap_group=1 -D msap_causal=1
#       production-width head fan-out + causal
#   -D msap_fp8=1
#       FP8 e4m3 Q/K/V -> BF16 O (the forward asserts a 16-bit output).  Drives
#       BK=64 / MFMA depth 64, a different Q-gather fragment, the fp8 P repack.
#
# Three argv modes (orchestrator-driven, per-case timeout + process isolation):
#   --mode list-specs --seed S --budget B
#   --mode single --batch .. --sq_seed .. --sk_seed .. --q2k_seed .. [--check 1]
#   --mode fuzz --seed S --budget B   (default; in-process batch)
#
# Oracles: memory-safety (memcheck/redzone) catches OOB in the device CSR build
# + block-major scatter (the ragged-shape-driven work decomposition is the risk).
# `ref` (--check 1, DEFAULT) is the f64 block-sparse oracle (mirrors the kernel's
# own authority test, test_msa_d128_prefill.mojo): each query attends its topk
# blocks' surviving (and, if causal, causally-valid) tokens, f64 softmax of
# scale*Q.K^T, weighted V. Gates are shaped like that test's -- per-element
# (atol, rtol) plus an aggregate rel-L2, the PRIMARY gate at fp8 since a
# per-element bound cannot be tight there -- but looser: this target reseeds
# every run, so it cannot calibrate against a fixed phase set (see `_rel_l2_tol`).
# Dual-arch: B200/SM100 (msa_sm100_prefill_run) + AMD MI355 (msa_amd_prefill_run).

from std.math import exp, isinf, isnan, sqrt
from std.random import randn, random_ui64, seed
from std.sys import (
    CompilationTarget,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from std.sys.defines import get_defined_int
from std.utils import IndexList

from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from layout import Idx, TileTensor, row_major

from nn.attention.mha_operand import LayoutTensorMHAOperand, MHAOperand
from nn.attention.mha_utils import MHAConfig
from msa.msa_2q import msa_sm100_prefill_plan, msa_sm100_prefill_run
from msa.amd.prefill import msa_amd_prefill_run

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime BN = 128  # block size (tokens); dtype-independent (only BK varies)
comptime head_dim = 128  # M3 dense head_dim (D=128 prefill kernel)

comptime topk = get_defined_int["msap_topk", 8]()
comptime num_q_heads = get_defined_int["msap_num_q_heads", 1]()
comptime group = get_defined_int["msap_group", 1]()  # qheadperkv (M3: 1)
comptime causal = get_defined_int["msap_causal", 0]() == 1

comptime use_fp8 = get_defined_int["msap_fp8", 0]() == 1
comptime dtype = DType.float8_e4m3fn if use_fp8 else DType.bfloat16  # Q/K/V
# The forward asserts a 16-bit O, so fp8 in -> bf16 out.
comptime out_dtype = DType.bfloat16 if dtype.is_float8() else dtype


@always_inline
def _rel_l2_tol() -> Float64:
    """Aggregate rel-L2 gate, and the primary correctness gate for fp8.

    A per-element bound cannot gate fp8 (it has to admit `2^-4 * max|V|`
    independent of `|O|`); only the aggregate separates noise (measured 2.2e-2
    fp8, 2.1e-3 bf16) from a real layout bug (a Q/K depth permutation measured
    0.65-0.91).  Kept looser than the in-tree gates in
    `Kernels/test/msa/test_msa_d128_prefill.mojo`, which calibrate against a
    fixed phase set this target does not have.
    """
    return Float64(5e-2) if dtype.is_float8() else Float64(2e-2)


comptime head_kv = num_q_heads // group

comptime MAX_BATCH = 4
comptime MAX_BLOCKS = 16  # per-batch KV block cap (seqlen_k <= 16*128)
comptime MAX_Q_PER_BATCH = 192  # per-batch query cap

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied ragged block-sparse prefill shape.

    The ragged batch (per-batch seqlen_q / seqlen_k) + the q2k block selection
    are fully determined by these scalars (so `single` reproduces a case
    exactly). topk / heads / group / causal are compile-time.
    """

    var batch: Int
    var sq_seed: Int  # draws per-batch seqlen_q
    var sk_seed: Int  # draws per-batch seqlen_k
    var q2k_seed: Int  # draws the block selection + empty-split edges

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch=",
            self.batch,
            " sq_seed=",
            self.sq_seed,
            " sk_seed=",
            self.sk_seed,
            " q2k_seed=",
            self.q2k_seed,
            " topk=",
            topk,
            " num_q_heads=",
            num_q_heads,
            " group=",
            group,
            " causal=",
            causal,
            " dtype=",
            dtype,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var b = boundary_int(1, MAX_BATCH, 2)
        var sq = Int(random_ui64(0, 1_000_000))
        var sk = Int(random_ui64(0, 1_000_000))
        var qk = Int(random_ui64(0, 1_000_000))
        specs.append(CaseSpec(b, sq, sk, qk))
    return specs^


# ===----------------------------------------------------------------------=== #
# Deterministic ragged-shape derivation from the scalar spec
# ===----------------------------------------------------------------------=== #


def _derive_cu(
    batch: Int, the_seed: Int, lo: Int, hi: Int, tile: Int
) -> List[Int32]:
    """Per-batch length prefix sum `[batch+1]` (boundary-biased per batch)."""
    seed(the_seed)
    var cu = List[Int32](length=batch + 1, fill=Int32(0))
    for b in range(batch):
        var L = boundary_int(lo, hi, tile)
        cu[b + 1] = cu[b] + Int32(L)
    return cu^


# ===----------------------------------------------------------------------=== #
# f64 block-sparse oracle (mirrors test_msa_d128_prefill._blocksparse_oracle):
# each query attends its topk blocks' surviving (+ causally-valid) tokens.
# K/V are head-major [head_kv, total_k, D].
# ===----------------------------------------------------------------------=== #


def _blocksparse_oracle(
    q2k: List[Int32],  # [head_kv, total_q, topk] batch-local block ids
    q: List[Float64],  # [total_q, head_q, D]
    k: List[Float64],  # [head_kv, total_k, D]
    v: List[Float64],  # [head_kv, total_k, D]
    cu_q: List[Int32],
    cu_k: List[Int32],
    total_q: Int,
    total_k: Int,
    scale: Float64,
) raises -> List[Float64]:
    var batch = len(cu_q) - 1
    var out = List[Float64](
        length=total_q * num_q_heads * head_dim, fill=Float64(0)
    )

    for g in range(total_q):
        var b = 0
        for bb in range(batch):
            if g < Int(cu_q[bb + 1]):
                b = bb
                break
        var k_base_tok = Int(cu_k[b])
        var seqlen_k = Int(cu_k[b + 1]) - k_base_tok
        var seqlen_q = Int(cu_q[b + 1]) - Int(cu_q[b])
        var qloc = g - Int(cu_q[b])
        var causal_q_offset = seqlen_k - seqlen_q

        for h in range(num_q_heads):
            var kh = h // group
            var q_base = (g * num_q_heads + h) * head_dim
            var idx_base = (kh * total_q + g) * topk

            var logits = List[Float64]()
            var slot_tok = List[Int]()
            for t in range(topk):
                var blk = Int(q2k[idx_base + t])
                if blk < 0:
                    continue
                for c in range(BN):
                    var kv_logical = blk * BN + c
                    if kv_logical >= seqlen_k:
                        continue
                    if causal and kv_logical > qloc + causal_q_offset:
                        continue
                    var ktok = k_base_tok + kv_logical
                    var ko = (kh * total_k + ktok) * head_dim
                    var dot = Float64(0)
                    for d in range(head_dim):
                        dot += q[q_base + d] * k[ko + d]
                    logits.append(dot * scale)
                    slot_tok.append(ktok)

            var ncols = len(logits)
            if ncols == 0:
                continue  # C == 0: numerator-zero, O stays 0
            var mx = Float64(-1e300)
            for i in range(ncols):
                mx = max(mx, logits[i])
            var sm = Float64(0)
            for i in range(ncols):
                sm += exp(logits[i] - mx)
            for d in range(head_dim):
                var acc = Float64(0)
                for i in range(ncols):
                    var w = exp(logits[i] - mx) / sm
                    var vo = (kh * total_k + slot_tok[i]) * head_dim
                    acc += w * v[vo + d]
                out[q_base + d] = acc

    return out^


# ===----------------------------------------------------------------------=== #
# One case: build q2k + ragged Q + batch-packed K/V + plan/run, optional ref.
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    comptime scale_f32 = Float32(1.0) / sqrt(Float32(head_dim))
    comptime scale_f64 = Float64(1.0) / sqrt(Float64(head_dim))

    var batch = spec.batch
    # Ragged Q: 1..MAX_Q_PER_BATCH per batch (biased around BM=128 q-chunk pivot).
    var cu_q = _derive_cu(batch, spec.sq_seed, 1, MAX_Q_PER_BATCH, BN)
    # Ragged K: at least 1 block; biased around BN so block tails appear.
    var cu_k = _derive_cu(batch, spec.sk_seed, BN, MAX_BLOCKS * BN, BN)
    var total_q = Int(cu_q[batch])
    var total_k = Int(cu_k[batch])

    var max_sq = 0
    var max_sk = 0
    for b in range(batch):
        max_sq = max(max_sq, Int(cu_q[b + 1] - cu_q[b]))
        max_sk = max(max_sk, Int(cu_k[b + 1] - cu_k[b]))

    # --- q2k [head_kv, total_q, topk]: distinct in-range blocks per (kh, g);
    # one query per kv-head is left all -1 to drive the C==0 empty-split edge.
    seed(spec.q2k_seed)
    var empty_q = Int(random_ui64(0, UInt64(max(1, total_q) - 1)))
    var q2k = List[Int32](length=head_kv * total_q * topk, fill=Int32(-1))
    for kh in range(head_kv):
        for g in range(total_q):
            if g == empty_q:
                continue  # leave all slots -1 -> C == 0
            var b = 0
            for bb in range(batch):
                if g < Int(cu_q[bb + 1]):
                    b = bb
                    break
            var nblk = (Int(cu_k[b + 1] - cu_k[b]) + BN - 1) // BN
            var base = (kh * total_q + g) * topk
            if nblk <= 0:
                continue
            var used = List[Bool](length=max(nblk, 1), fill=False)
            var mult = 37 + 2 * ((kh * 131 + g * 17) % 7)  # odd, coprime
            var shift = (g * 101 + kh * 53 + 7) % nblk
            var npick = min(topk, nblk)
            for t in range(npick):
                var blk = (shift + t * mult) % nblk
                while used[blk]:
                    blk = (blk + 1) % nblk
                used[blk] = True
                q2k[base + t] = Int32(blk)

    # --- Q [total_q, head_q, D]; batch-packed K/V [total_k, head_kv, D] -------
    var q_size = total_q * num_q_heads * head_dim
    var kv_size = total_k * head_kv * head_dim

    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var k_host = ctx.enqueue_create_host_buffer[dtype](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[dtype](kv_size)
    ctx.synchronize()
    randn(q_host.as_span())
    randn(k_host.as_span())
    randn(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    var k_dev = ctx.enqueue_create_buffer[dtype](kv_size)
    var v_dev = ctx.enqueue_create_buffer[dtype](kv_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    # KV operand [1, total_k, head_kv, D] (row_idx == absolute K token).
    var k_tt = TileTensor(
        k_dev, row_major((Idx[1], total_k, Idx[head_kv], Idx[head_dim]))
    )
    var v_tt = TileTensor(
        v_dev, row_major((Idx[1], total_k, Idx[head_kv], Idx[head_dim]))
    )
    var k_op = LayoutTensorMHAOperand(k_tt.as_immut().as_unsafe_any_origin())
    var v_op = LayoutTensorMHAOperand(v_tt.as_immut().as_unsafe_any_origin())

    # --- q2k + cu_seqlens to device (the run is pure-device) ------------------
    var q2k_h = ctx.enqueue_create_host_buffer[.int32](head_kv * total_q * topk)
    var cuq_h = ctx.enqueue_create_host_buffer[.int32](batch + 1)
    var cuk_h = ctx.enqueue_create_host_buffer[.int32](batch + 1)
    ctx.synchronize()
    for i in range(head_kv * total_q * topk):
        q2k_h[i] = q2k[i]
    for i in range(batch + 1):
        cuq_h[i] = cu_q[i]
        cuk_h[i] = cu_k[i]
    var q2k_dev = ctx.enqueue_create_buffer[.int32](head_kv * total_q * topk)
    var cuq_dev = ctx.enqueue_create_buffer[.int32](batch + 1)
    var cuk_dev = ctx.enqueue_create_buffer[.int32](batch + 1)
    ctx.enqueue_copy(q2k_dev, q2k_h)
    ctx.enqueue_copy(cuq_dev, cuq_h)
    ctx.enqueue_copy(cuk_dev, cuk_h)

    var o_dev = ctx.enqueue_create_buffer[out_dtype](q_size)
    var lse_dev = ctx.enqueue_create_buffer[.float32](total_q * num_q_heads)

    # === Kernel under test: plan (host, once) + pure-device run ===============
    comptime config = MHAConfig[dtype](num_q_heads, head_dim)
    var plan = msa_sm100_prefill_plan[
        output_type=out_dtype, config=config, group=group, topk=topk
    ](total_q, total_k, batch, max_sq, max_sk, ctx)
    comptime if has_amd_gpu_accelerator():
        msa_amd_prefill_run[
            config=config, group=group, topk=topk, use_causal=causal
        ](
            plan,
            o_dev,
            lse_dev,
            q_dev,
            k_op,
            v_op,
            q2k_dev,
            cuq_dev,
            cuk_dev,
            scale_f32,
            ctx,
        )
    elif has_nvidia_gpu_accelerator():
        msa_sm100_prefill_run[
            config=config, group=group, topk=topk, use_causal=causal
        ](
            plan,
            o_dev,
            lse_dev,
            q_dev,
            k_op,
            v_op,
            q2k_dev,
            cuq_dev,
            cuk_dev,
            scale_f32,
            ctx,
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()
    ctx.synchronize()

    if check:
        var o_host = ctx.enqueue_create_host_buffer[out_dtype](q_size)
        ctx.enqueue_copy(o_host, o_dev)
        ctx.synchronize()

        # NaN/Inf guard (a degenerate softmax should never go non-finite here).
        for i in range(q_size):
            var x = o_host[i].cast[.float32]()
            if isnan(x) or isinf(x):
                print("FUZZ_NUMERIC_FAIL kind=naninf idx=", i, "val=", x)
                raise Error("MSA prefill output NaN/Inf")

        # f64 mirrors. Kernel KV is token-major [total_k, head_kv, D]; the oracle
        # wants head-major [head_kv, total_k, D]. Build both.
        var q_f64 = List[Float64](length=q_size, fill=Float64(0))
        var k_f64 = List[Float64](length=kv_size, fill=Float64(0))
        var v_f64 = List[Float64](length=kv_size, fill=Float64(0))
        var max_abs_v = Float64(0)  # feeds the derived fp8 atol below
        for i in range(q_size):
            q_f64[i] = q_host[i].cast[.float64]()
        for tok in range(total_k):
            for kh in range(head_kv):
                for d in range(head_dim):
                    var src = (tok * head_kv + kh) * head_dim + d
                    var dst = (kh * total_k + tok) * head_dim + d
                    k_f64[dst] = k_host[src].cast[.float64]()
                    v_f64[dst] = v_host[src].cast[.float64]()
                    max_abs_v = max(max_abs_v, abs(v_f64[dst]))

        var o_ref = _blocksparse_oracle(
            q2k,
            q_f64,
            k_f64,
            v_f64,
            cu_q,
            cu_k,
            total_q,
            total_k,
            scale_f64,
        )

        # fp8's per-element bound is `2^-4 * max|V|` regardless of |O| (2^-4 from
        # the P repack, ~2^-9 from rounding O to bf16), so DERIVE it from the V
        # actually drawn: this target reseeds every run and randn's max grows with
        # the draw count (~4.8 over 1e5 elements), so any constant is a latent
        # false-red.  rel-L2 remains the real gate; this is the backstop.
        comptime fp8_err_per_v = Float64(0.0625) + Float64(0.002)
        var atol = Float64(2e-2)
        comptime if dtype.is_float8():
            atol = fp8_err_per_v * max_abs_v
        comptime rtol = Float64(8e-2) if dtype.is_float8() else Float64(4e-2)
        comptime rel_l2_tol = _rel_l2_tol()
        var max_abs = Float64(0)
        var max_rel = Float64(0)
        var sq_err = Float64(0)
        var sq_ref = Float64(0)
        var n_bad = 0
        for i in range(q_size):
            var got = o_host[i].cast[.float64]()
            var oref = o_ref[i]
            var ad = abs(got - oref)
            max_abs = max(max_abs, ad)
            sq_err += ad * ad
            sq_ref += oref * oref
            if abs(oref) > 0.1:
                var rel = ad / abs(oref)
                max_rel = max(max_rel, rel)
                if ad > atol and rel > rtol:
                    n_bad += 1
            elif ad > atol:
                n_bad += 1
        var rel_l2 = sqrt(sq_err / sq_ref) if sq_ref > 0 else Float64(0)
        if n_bad > 0 or rel_l2 > rel_l2_tol:
            print(
                "FUZZ_NUMERIC_FAIL kind=blocksparse n_bad=",
                n_bad,
                "max_abs=",
                max_abs,
                "max_rel=",
                max_rel,
                "rel_l2=",
                rel_l2,
                "rel_l2_tol=",
                rel_l2_tol,
            )
            raise Error(
                "MSA prefill output mismatch vs f64 block-sparse oracle"
            )

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = q2k_dev
    _ = cuq_dev
    _ = cuk_dev
    _ = o_dev
    _ = lse_dev
    _ = plan^


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch=",
                specs[i].batch,
                "sq_seed=",
                specs[i].sq_seed,
                "sk_seed=",
                specs[i].sk_seed,
                "q2k_seed=",
                specs[i].q2k_seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--batch", 1),
            flag_int(args, "--sq_seed", 1),
            flag_int(args, "--sk_seed", 2),
            flag_int(args, "--q2k_seed", 3),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: standalone in-process fuzz.
    print(
        "=== fuzz_msa_prefill topk=",
        topk,
        "num_q_heads=",
        num_q_heads,
        "group=",
        group,
        "causal=",
        causal,
        "seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
