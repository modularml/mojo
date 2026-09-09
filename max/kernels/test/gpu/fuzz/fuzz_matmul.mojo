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
# Fuzz target: matmul / GEMM (`_matmul_gpu`) (see gpu-kernels-fuzzing-design.md).
#
# On SM100 the tuned tensor-core kernel reads N and K from the tensors' STATIC
# shape, so they must be compile-time (set via `-D N=.. -D K=..`); M is the
# runtime fuzz axis. bf16 + transpose_b is the tuned path (N, K multiples of 8
# keep TMA alignment). Memory-safety oracle (memcheck / redzone).
#
# `determinism` oracle (--rerun N): re-launch the SAME input N times and require
# bit-exact output (atol=rtol=0). On SM100 the M>1 tile GEMM is single
# K-partition and the M=1 GEMV_SPLIT_K reduces K via a fixed-order warp+shared
# tree, so there is no cross-block reduction to reorder -- any run-to-run
# difference is a real race / nondeterminism, not a legit FP wobble.
#
# `batch_variance` oracle (--batch-variance 1): NEGATIVE CONTROL. Dense matmul
# is M-keyed: on SM100 the probe row goes through GEMV_SPLIT_K at M=1 but the
# tuned tensor-core tile GEMM at M>1 -- a different accumulation order. We run
# the SAME probe row in an M=1 batch vs an M>1 batch (row 0 byte-identical; the
# extra rows are fillers that never touch row 0's math in a correct GEMM) and
# require the probe's output row to DIVERGE bit-for-bit. This proves the
# bit-exact invariance oracles have real sensitivity to a dispatch switch; a
# bit-match is reported (FUZZ_CONTRACT_FAIL), not swallowed.

from std.math import ceildiv
from std.random import rand, seed
from std.sys.defines import get_defined_int

from max.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime dtype = DType.bfloat16
comptime N = get_defined_int[
    "N", 2048
]()  # COMPTIME (tuned SM100 reads static N)
comptime K = get_defined_int["K", 2048]()  # COMPTIME
comptime TILE = 128  # matmul block-M tile -- the interesting modulus for M.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


def naive_matmul_ref_kernel(
    c: MutPointer[Scalar[dtype], MutAnyOrigin],
    a: MutPointer[Scalar[dtype], MutAnyOrigin],
    b: MutPointer[Scalar[dtype], MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    """C[m,n] = sum_k A[m,k]*B[n,k] with fp32 accumulation (transpose_b).

    A higher-precision reference than same-precision cuBLAS: it accumulates in
    fp32, exposing shared-rounding/reduction bugs the bf16 tensor-core path and
    a same-precision vendor reference would both hide.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var col = global_idx.x
    var row = global_idx.y
    if row < m and col < n:
        var acc = Float32(0)
        for k_i in range(k):
            acc += (
                a[row * k + k_i].cast[.float32]()
                * b[col * k + k_i].cast[.float32]()
            )
        c[row * n + col] = acc.cast[dtype]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("m=", self.m, " N=", N, " K=", K)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(CaseSpec(boundary_int(1, 4096, TILE)))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False, rerun: Int = 0
) raises:
    var m = spec.m
    var a_size = m * K
    var b_size = N * K  # transpose_b => B is [N, K]
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_size)
    rand(a_host.as_span())
    rand(b_host.as_span())

    var a_dev = ctx.enqueue_create_buffer[dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    # M runtime (bare Int); N, K comptime (Idx[...]) so the tuned kernel engages.
    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()

    if rerun > 0:
        # Run-to-run determinism: re-launch the SAME input `rerun` times and
        # require bit-exact output. No split-K to force (M>1 tile GEMM is single
        # K-partition; M=1 GEMV_SPLIT_K reduces K in-block via a fixed-order
        # warp+shared tree), so any difference is a real race / nondeterminism.
        var first_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        ctx.enqueue_copy(first_h, c_dev)
        ctx.synchronize()
        for _ in range(rerun - 1):
            _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
            ctx.synchronize()
            var rep_h = ctx.enqueue_create_host_buffer[dtype](c_size)
            ctx.enqueue_copy(rep_h, c_dev)
            ctx.synchronize()
            if not numeric_check(
                rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
            ):
                raise Error("matmul run-to-run nondeterminism (rerun)")
    elif check:
        # Numerical oracle: compare against an fp32-accum naive reference.
        var c_ref_dev = ctx.enqueue_create_buffer[dtype](c_size)
        comptime BX = 16
        comptime BY = 16
        ctx.enqueue_function[naive_matmul_ref_kernel](
            c_ref_dev,
            a_dev,
            b_dev,
            Int32(m),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(N, BX), ceildiv(m, BY)),
            block_dim=(BX, BY),
        )
        ctx.synchronize()
        var c_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        var c_ref_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        ctx.enqueue_copy(c_h, c_dev)
        ctx.enqueue_copy(c_ref_h, c_ref_dev)
        ctx.synchronize()
        if not numeric_check(
            c_h.as_span(), c_ref_h.as_span(), atol=2.0, rtol=5e-2
        ):
            raise Error("matmul vs fp32-accum naive mismatch")
        _ = c_ref_dev

    _ = a_dev
    _ = b_dev
    _ = c_dev


# ===----------------------------------------------------------------------=== #
# Batch-variance negative control (M-ladder dispatch breakpoint)
# ===----------------------------------------------------------------------=== #
#
# Each probe row is byte-identical whether run alone (M=1 -> GEMV_SPLIT_K) or
# co-batched (M>1 -> tile GEMM). A correct GEMM computes each output row from
# one row of A, so co-batching cannot change a row's math -- only the M-keyed
# dispatch changes the floating-point accumulation ORDER. We compare each row's
# alone vs co-batched output bit-for-bit and require divergence.
#
# At bf16 / K=2048 the two accumulation orders agree bit-for-bit for a
# meaningful fraction (~15-25%) of individual rows, so a single-probe control
# would flake. We aggregate over N_PROBES rows and PASS if ANY diverges; only
# an ALL-rows bit-match FAILs (which would mean the breakpoint became
# invariant -- a real characterization change, not a flake).
#
# The ~15-25% is measured, not assumed. This mode prints
# `rows_diverged=<d>/N_PROBES` per case, so the per-row match rate is
# `1 - mean(d / N_PROBES)`. Measured on B200 (N=2048, K=2048, bf16) over 512
# rows (budget=32, seeds 12345 and 4242): ~0.24 (per-seed 0.22-0.26). Reproduce
# by building this target and running its binary with
# `--mode fuzz --batch-variance 1 --budget 32`, then aggregating the printed
# `rows_diverged`. N_PROBES=8 follows: a spurious all-match FAIL has probability
# match_rate^N_PROBES ~= 0.24^8 ~= 2e-6. The rate is downstream of dtype and K
# (fewer accumulation steps -> higher match rate), so a different shape needs
# N_PROBES re-derived from a fresh measurement.

comptime B_SEED = 0xB1A5  # B weights; identical across every launch.
comptime N_PROBES = 8  # probe rows compared per case (see flakiness note above).


def _run_rows(
    ctx: DeviceContext, m: Int, base_seed: Int
) raises -> HostBuffer[dtype]:
    """Launches a [m, K] x [N, K]^T matmul where row j of A is seeded from
    `base_seed + j` and B from `B_SEED`. Returns all `m * N` output rows on the
    host. Seeding per row makes a co-batched run (m rows, base_seed) and a
    standalone run (m=1, base_seed + j) share byte-identical inputs for row j.
    """
    var a_size = m * K
    var b_size = N * K
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_size)
    seed(B_SEED)
    rand(b_host.as_span())
    for j in range(m):
        seed(base_seed + j)
        rand(a_host.unsafe_ptr() + j * K, K)

    var a_dev = ctx.enqueue_create_buffer[dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[dtype](c_size)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()
    _ = a_dev
    _ = b_dev
    _ = c_dev
    return c_host^


def _row_bit_diff(
    row_out: HostBuffer[dtype],
    off: Int,
    ref_out: HostBuffer[dtype],
    ref_off: Int,
) -> Tuple[Int, Float64]:
    """Bit-exact element diff of `row_out[off:off+N]` vs `ref_out[ref_off:+N]`.
    Returns (number of differing elements, max abs difference)."""
    var n_diff = 0
    var max_abs = Float64(0)
    for j in range(N):
        var a = row_out[off + j].cast[.float64]()
        var b = ref_out[ref_off + j].cast[.float64]()
        if a != b:
            n_diff += 1
            var ad = abs(a - b)
            if ad > max_abs:
                max_abs = ad
    return (n_diff, max_abs)


def run_batch_variance_case(
    ctx: DeviceContext, spec: CaseSpec, same_regime: Bool = False
) raises:
    """Negative control: assert probe rows DIVERGE across the M=1 -> M>1
    dispatch breakpoint.

    Reference: an M=N_PROBES tile-GEMM batch. Probe side: each row rerun ALONE
    at M=1 (-> GEMV_SPLIT_K). PASS iff at least one row's alone output differs
    bit-for-bit from its co-batched output (divergence observed, proving the
    bit-exact invariance oracles have real sensitivity to the dispatch switch).
    An all-rows bit-match emits FUZZ_CONTRACT_FAIL: the breakpoint is invariant
    for these shapes -- a characterization finding, not a silent pass.

    `same_regime=True` reruns the probe side in the SAME regime (a second
    M=N_PROBES tile GEMM) instead of M=1: identical regime -> all rows bit-match
    -> the FAIL path must fire (inverted positive control)."""
    # This control's teeth come from the M-keyed dispatch switch: SM100 sends
    # bf16 to the GEMV path iff `static_N == 1 or m == 1` and to the tile GEMM
    # otherwise (`matmul_dispatch_sm100`). We rely on N > 1 so that M alone
    # (1 vs N_PROBES) flips the kernel; at N == 1 both regimes take GEMV and the
    # control is vacuous. Pin it at compile time -- the runtime
    # FUZZ_CONTRACT_FAIL below is the complementary guard if the gate changes.
    comptime assert N > 1, (
        "batch-variance control needs N > 1 so the M=1/M>1 switch selects"
        " different kernels (see the matmul_dispatch_sm100 bf16 gate)"
    )
    comptime assert N_PROBES > 1, "co-batched regime requires M = N_PROBES > 1"
    # Vary the probe rows per case (spec.m is the only spec field).
    var base = 0x50B0 ^ (spec.m * 0x9E3779B1)

    # Reference: co-batched tile GEMM (M=N_PROBES > 1).
    var batched = _run_rows(ctx, N_PROBES, base)

    var n_rows_diverged = 0
    var n_diff_total = 0
    var max_abs = Float64(0)

    if same_regime:
        # Inverted positive control: same regime -> must bit-match.
        var same = _run_rows(ctx, N_PROBES, base)
        for i in range(N_PROBES):
            var d = _row_bit_diff(same, i * N, batched, i * N)
            n_diff_total += d[0]
            if d[0] > 0:
                n_rows_diverged += 1
            if d[1] > max_abs:
                max_abs = d[1]
        _ = same^
    else:
        for i in range(N_PROBES):
            var alone = _run_rows(ctx, 1, base + i)  # M=1 -> GEMV_SPLIT_K
            var d = _row_bit_diff(alone, 0, batched, i * N)
            n_diff_total += d[0]
            if d[0] > 0:
                n_rows_diverged += 1
            if d[1] > max_abs:
                max_abs = d[1]
            _ = alone^

    print(
        "BATCH_VARIANCE base_m=",
        spec.m,
        "N=",
        N,
        "K=",
        K,
        "rows_diverged=",
        n_rows_diverged,
        "/",
        N_PROBES,
        "n_diff_total=",
        n_diff_total,
        "max_abs=",
        max_abs,
    )

    if n_rows_diverged == 0:
        # Negative control lost its teeth: every row bit-matched across the
        # M=1 / M>1 breakpoint.
        print(
            "FUZZ_CONTRACT_FAIL batch-variance negative control: all",
            N_PROBES,
            "rows bit-identical across the M=1 / M=",
            N_PROBES,
            "dispatch breakpoint (expected divergence)",
        )
        raise Error("batch-variance negative control observed no divergence")

    _ = batched^


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    var batch_variance = flag_int(args, "--batch-variance", 0) == 1
    # Debug-only inverted positive control: force composition B to M=1 too, so
    # the two regimes bit-match and the negative control's FAIL path fires.
    var bv_same_regime = flag_int(args, "--bv-same-m", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "m=", specs[i].m)
        return

    if mode == "single":
        var m = flag_int(args, "--m", 128)
        print("FUZZ_SINGLE m=", m, "N=", N, "K=", K)
        with DeviceContext() as ctx:
            if batch_variance:
                run_batch_variance_case(
                    ctx, CaseSpec(m), same_regime=bv_same_regime
                )
            elif rerun > 0:
                run_one_case(ctx, CaseSpec(m), rerun=rerun)
            else:
                run_one_case(ctx, CaseSpec(m), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_matmul seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if batch_variance:
                run_batch_variance_case(
                    ctx, specs[i], same_regime=bv_same_regime
                )
            elif rerun > 0:
                run_one_case(ctx, specs[i], rerun=rerun)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
