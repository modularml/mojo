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
# Fuzz target: grouped block-scaled W4A8 SM100 matmul -- E4M3 activations
# against nibble-packed E2M1 weights, both on E8M0 group-32 scales.
#
# W4A8 adds an axis the MXFP8 sibling cannot reach: the two operands no longer
# agree on how many bytes a K-row occupies. The weights stay nibble-packed in
# global memory (`K/2` bytes per row) and the FP4 TMA copy pads them so a K
# extent spans one byte per element, so every K extent inside the kernel is in
# ELEMENTS while the weights' GMEM row is halved. That split is what makes the
# ragged/boundary search worth running here: a TMA transaction size counted in
# the wrong unit hangs the load barrier rather than returning a wrong answer,
# and a scale-group index off by the packing factor reads a neighbouring
# expert's scales. Both are shape-dependent, so fixed-shape tests miss them.
#
# The fuzzed axis is the RAGGED PER-EXPERT TOKEN DISTRIBUTION, as for the MXFP8
# target: the orchestrator carries only integer spec fields, so the
# distribution is encoded as `num_active_experts` + `tok_seed` and expanded
# deterministically inside the target (per-expert counts biased around
# SF_MN_GROUP_SIZE=128). N/K/num_experts are compile-time
# (`-D N=.. -D K=.. -D num_experts=..`); K must stay a multiple of 128, which
# `grouped_matmul_block_scaled` gates with a comptime assert.
#
# This drives `grouped_matmul_block_scaled_sm100_dispatch` rather than the
# config-explicit launcher, so the search also covers scaling-kind inference
# from B's dtype and the launcher's AB_swapped operand-dtype wiring -- both
# only observable when the two operand dtypes differ.
#
# `ref` oracle (--check 1): a host FP32 recompute. No vendor BLAS path takes
# this operand pair, so the reference is exact by construction instead of
# higher-precision: activations are drawn from a set that is exact in E4M3,
# every A*B product is a multiple of 0.25, scales and per-expert alphas are
# powers of two, and the largest partial sum stays far inside FP32's 24-bit
# mantissa. No reordering of the accumulation can change the result, so the
# comparison is BIT-EXACT: the reference is rounded to the output dtype and
# compared at atol=rtol=0. The bound on "exact" is K <= 10922 -- the largest
# partial sum is 384*K and stays exactly representable while 1536*K <= 2^24 --
# which `main` enforces so a `-D K=` override cannot silently invalidate it.
# The recompute is O(M*N*K) on the host, so it checks a strided sample of BOTH
# axes; see the sampling comment for why the column stride is forced odd.
#
# `determinism` oracle (--rerun N): re-launch the SAME input N times and
# require bit-exact output. The kernel is single K-partition with a
# comptime-static config, so any run-to-run difference is a real race.
#
# Batch invariance is NOT covered here, and is not covered by the MXFP8 target
# either: that one builds its config with AB_swapped defaulted False and
# cta_group=1, while W4A8 reaches the regime-tuned launcher with
# AB_swapped=True and cta_group=2 on both prefill regimes. So the config this
# path actually runs is ungated for batch invariance -- a known gap, not a
# claim of coverage.

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from std.math import ceildiv, max
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
    get_scale_factor,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime a_type = W4A8_A_DTYPE
comptime b_type = W4A8_B_DTYPE
comptime out_dtype = DType.bfloat16
comptime scales_dtype = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

# COMPTIME expert geometry. K counts FP4 ELEMENTS; the weights hold K/2 bytes
# per row. The packed-FP4 TMA descriptor requires K % 128 == 0.
comptime N = get_defined_int["N", 1024]()
comptime K = get_defined_int["K", 512]()
comptime num_experts = get_defined_int["num_experts", 8]()
comptime K_PACKED = K // 2

# Largest K for which the host FP32 reference stays bit-exact; see the `ref`
# oracle note in the header. `main` gates `-D K=` on it.
comptime K_EXACT_REFERENCE_MAX = 10922

comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
comptime b_expert_scale_count = (
    n_groups * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
)

# Per-expert token cap. Spans the 128-row scale granule and partial-tile
# remainders while keeping the host reference and the allocations bounded.
comptime MAX_M_PER_EXPERT = 512

# Exact in E4M3, and small enough that every partial sum stays exact in FP32.
comptime A_VALUES = SIMD[.float32, 8](0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 1.5)

# Bounds the strided host recompute at ~SAMPLE_BUDGET*K multiply-adds.
comptime SAMPLE_BUDGET = 4096

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var num_active_experts: Int
    var tok_seed: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "num_active_experts=",
            self.num_active_experts,
            " tok_seed=",
            self.tok_seed,
            " N=",
            N,
            " K=",
            K,
            " num_experts=",
            num_experts,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var nae = boundary_int(1, num_experts, num_experts)
        var ts = Int(random_ui64(1, 1 << 30))
        specs.append(CaseSpec(nae, ts))
    return specs^


def _expand_distribution(
    num_active_experts: Int, tok_seed: Int
) -> Tuple[List[Int], List[Int]]:
    """Deterministically expand (num_active_experts, tok_seed) into the ragged
    per-expert token counts and the distinct expert ids they route to."""
    seed(tok_seed)
    var counts = List[Int]()
    for _ in range(num_active_experts):
        counts.append(boundary_int(1, MAX_M_PER_EXPERT, SF_MN_GROUP_SIZE))
    var ids = List[Int]()
    var base = tok_seed % num_experts
    for i in range(num_active_experts):
        # (base + i) % num_experts is distinct for num_active_experts <=
        # num_experts, and exercises non-contiguous weight/scale slices.
        ids.append((base + i) % num_experts)
    return (counts^, ids^)


def _e8m0_pow2(bits: Int) -> Scalar[scales_dtype]:
    """An E8M0 scale of 2**bits, exact for the small exponents used here."""
    return _convert_f32_to_float8_ue8m0[scales_dtype](Float32(1 << bits))


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False, rerun: Int = 0
) raises:
    comptime transpose_b = True
    var num_active_experts = spec.num_active_experts
    var counts_ids = _expand_distribution(num_active_experts, spec.tok_seed)
    var num_tokens_by_expert = counts_ids[0].copy()
    var expert_ids = counts_ids[1].copy()

    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]

    comptime b_elems = num_experts * N * K_PACKED
    comptime b_scale_elems = num_experts * b_expert_scale_count

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](
        total_num_tokens * K
    )
    var a_host = TileTensor(
        a_host_ptr, row_major(Coord(total_num_tokens, Idx[K]))
    )
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_elems)
    var b_host = TileTensor(
        b_host_ptr, row_major(Coord(Idx[num_experts], Idx[N], Idx[K_PACKED]))
    )
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_dtype](
        total_num_tokens * N
    )
    var c_host = TileTensor(
        c_host_ptr, row_major(Coord(total_num_tokens, Idx[N]))
    )

    var a_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var expert_scales_host = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )

    # Powers of two so the epilogue multiply stays exact.
    for e in range(num_experts):
        expert_scales_host[e] = Float32(1 << (e % 2))

    var a_scale_rows = 0
    a_offsets_host[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_host[i] = UInt32(
            a_scale_rows - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        a_scale_rows += ceildiv(num_tokens_by_expert[i], SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(expert_ids[i])

    var a_scales_shape = row_major(
        Coord(
            Int(max(a_scale_rows, 1)),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scale_elems
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    )

    # ---- Operand fill ---------------------------------------------------
    for m in range(total_num_tokens):
        for k in range(K):
            a_host[m, k] = A_VALUES[Int(random_ui64(0, 7))].cast[a_type]()

    # Two E2M1 codes per byte: element 2i in the low nibble, 2i+1 in the high.
    for e in range(num_experts):
        for n in range(N):
            for kb in range(K_PACKED):
                b_host[e, n, kb] = UInt8(random_ui64(0, 255))

    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[unsafe_offset=i] = Scalar[scales_dtype](0.0)
    for i in range(b_scales_host.num_elements()):
        b_scales_host._storage[unsafe_offset=i] = Scalar[scales_dtype](0.0)

    # Scale rows past an expert's token count are padding the kernel still
    # reads; leaving them nonzero corrupts neighbouring tiles.
    for i in range(num_active_experts):
        var start = Int(a_offsets_host[i])
        var scale_row = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_host[i])
        ) * SF_MN_GROUP_SIZE
        for row in range(scale_row, scale_row + num_tokens_by_expert[i]):
            for col in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host,
                    row,
                    col,
                    _e8m0_pow2(Int(random_ui64(0, 2))),
                )

    for e in range(num_experts):
        var expert_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr().unsafe_offset(
                e * b_expert_scale_count
            ),
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for n in range(N):
            for col in range(0, K, SF_VECTOR_SIZE):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    expert_slice,
                    n,
                    col,
                    _e8m0_pow2(Int(random_ui64(0, 2))),
                )

    # ---- Device copies --------------------------------------------------
    var a_dev = ctx.enqueue_create_buffer[a_type](max(total_num_tokens * K, 1))
    var b_dev = ctx.enqueue_create_buffer[b_type](b_elems)
    var c_dev = ctx.enqueue_create_buffer[out_dtype](
        max(total_num_tokens * N, 1)
    )
    var a_scales_dev = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_shape.product()
    )
    var b_scales_dev = ctx.enqueue_create_buffer[scales_dtype](b_scale_elems)
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        max(num_active_experts, 1)
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](
        max(num_active_experts, 1)
    )
    var expert_scales_dev = ctx.enqueue_create_buffer[.float32](num_experts)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(a_scales_dev, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_dev, b_scales_host_ptr)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(a_scale_offsets_dev, a_scale_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)
    ctx.enqueue_copy(expert_scales_dev, expert_scales_host)
    ctx.enqueue_memset(c_dev, Scalar[out_dtype](0))

    var a_scales_tt = TileTensor(
        a_scales_dev, a_scales_shape
    ).as_unsafe_any_origin()
    var b_scales_tt = TileTensor(
        b_scales_dev,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()

    var c_tt = TileTensor(
        c_dev, row_major(Coord(total_num_tokens, Idx[N]))
    ).as_unsafe_any_origin()
    var a_tt = TileTensor(
        a_dev, row_major(Coord(total_num_tokens, Idx[K]))
    ).as_unsafe_any_origin()
    var b_tt = TileTensor(
        b_dev, row_major(Coord(Idx[num_experts], Idx[N], Idx[K_PACKED]))
    ).as_unsafe_any_origin()
    var a_offsets_tt = TileTensor(
        a_offsets_dev, row_major(Coord(Int(num_active_experts + 1)))
    ).as_unsafe_any_origin()
    var a_scale_offsets_tt = TileTensor(
        a_scale_offsets_dev, row_major(Coord(Int(num_active_experts)))
    ).as_unsafe_any_origin()
    var expert_ids_tt = TileTensor(
        expert_ids_dev, row_major(Coord(Int(num_active_experts)))
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        expert_scales_dev, row_major(Coord(Idx[num_experts]))
    ).as_unsafe_any_origin()

    grouped_matmul_block_scaled_sm100_dispatch[
        transpose_b=transpose_b, target="gpu"
    ](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tt,
        a_scale_offsets_tt,
        expert_ids_tt,
        expert_scales_tt,
        num_active_experts,
        total_num_tokens,
        ctx,
    )

    if rerun > 0:
        # Run-to-run determinism: the kernel is single K-partition with a
        # comptime-static config, so there is no cross-block reduction to
        # reorder and any difference is a real race, not FP wobble.
        ctx.synchronize()
        var first_ptr = ctx.enqueue_create_host_buffer[out_dtype](
            total_num_tokens * N
        )
        ctx.enqueue_copy(first_ptr, c_dev)
        ctx.synchronize()
        for _ in range(rerun - 1):
            grouped_matmul_block_scaled_sm100_dispatch[
                transpose_b=transpose_b, target="gpu"
            ](
                c_tt,
                a_tt,
                b_tt,
                a_scales_tt,
                b_scales_tt,
                a_offsets_tt,
                a_scale_offsets_tt,
                expert_ids_tt,
                expert_scales_tt,
                num_active_experts,
                total_num_tokens,
                ctx,
            )
            ctx.synchronize()
            var rep_ptr = ctx.enqueue_create_host_buffer[out_dtype](
                total_num_tokens * N
            )
            ctx.enqueue_copy(rep_ptr, c_dev)
            ctx.synchronize()
            if not numeric_check(
                rep_ptr.as_span(), first_ptr.as_span(), atol=0.0, rtol=0.0
            ):
                raise Error(
                    "grouped W4A8 matmul run-to-run nondeterminism (rerun)"
                )
    elif check:
        ctx.enqueue_copy(c_host_ptr, c_dev)
        ctx.synchronize()

        # The host recompute is O(M*N*K), so bound the element count rather
        # than loosening the per-element tolerance. Stride BOTH axes: spending
        # the whole budget on rows leaves only a couple of columns, and an even
        # column stride that lands on a multiple of SF_MN_GROUP_SIZE would
        # sample nothing but column 0 of every B-scale group and N tile --
        # exactly the axis a scale-index bug moves. Forcing the stride odd
        # keeps it coprime with the 128-wide granule, so the sample walks
        # across offsets within a group instead of aliasing to its first
        # column.
        var budget_side = 1
        while (budget_side + 1) * (budget_side + 1) <= SAMPLE_BUDGET:
            budget_side += 1
        var m_stride = max(total_num_tokens // budget_side, 1)
        var n_stride = max(N // budget_side, 1)
        if n_stride % 2 == 0:
            n_stride += 1
        var n_cols = ceildiv(N, n_stride)
        var m_rows = ceildiv(total_num_tokens, m_stride)
        var actual = ctx.enqueue_create_host_buffer[out_dtype](
            max(m_rows * n_cols + num_active_experts * n_cols, 1)
        )
        var expected = ctx.enqueue_create_host_buffer[out_dtype](
            max(m_rows * n_cols + num_active_experts * n_cols, 1)
        )
        # Both buffers are compared in full, and the row sampling only fills a
        # prefix (per-expert rounding makes the exact count awkward to
        # precompute), so zero the tails rather than compare uninitialized
        # host memory against itself.
        for i in range(len(actual)):
            actual[i] = Scalar[out_dtype](0)
            expected[i] = Scalar[out_dtype](0)
        var w = 0

        for i in range(num_active_experts):
            var start = Int(a_offsets_host[i])
            var end = Int(a_offsets_host[i + 1])
            var expert_id = Int(expert_ids_host[i])
            var expert_scale = expert_scales_host[expert_id]
            var b_scale_slice = TileTensor(
                b_scales_host_ptr.unsafe_ptr().unsafe_offset(
                    expert_id * b_expert_scale_count
                ),
                row_major(
                    Coord(
                        Idx[n_groups],
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            )
            var scale_row_base = (
                start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_host[i])
            ) * SF_MN_GROUP_SIZE

            # `range(start, end, m_stride)` always yields `start`, so every
            # active expert contributes at least one checked row.
            for m in range(start, end, m_stride):
                var a_scale_row = scale_row_base + (m - start)
                for n in range(0, N, n_stride):
                    var acc = Float32(0.0)
                    for kb in range(0, K, SF_VECTOR_SIZE):
                        var sfa = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE
                        ](a_scales_host, a_scale_row, kb).cast[.float32]()
                        var sfb = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE
                        ](b_scale_slice, n, kb).cast[.float32]()
                        var block = Float32(0.0)
                        for k in range(kb, kb + SF_VECTOR_SIZE):
                            var av = a_host[m, k].cast[.float32]()
                            var packed = Int(b_host[expert_id, n, k // 2])
                            var nibble = (packed >> 4) if k % 2 else (
                                packed & 0xF
                            )
                            block += av * E2M1_TO_FLOAT32[nibble]
                        acc += block * sfa * sfb
                    actual[w] = c_host[m, n]
                    expected[w] = (acc * expert_scale).cast[out_dtype]()
                    w += 1

        # Bit-exact: `expected` is itself rounded to the output dtype, so both
        # sides are the same BF16 rounding of the same exact FP32 value. A
        # nonzero tolerance here would silently absorb a one-ulp error.
        if not numeric_check(
            actual.as_span(), expected.as_span(), atol=0.0, rtol=0.0
        ):
            raise Error("grouped W4A8 matmul vs exact host reference mismatch")
        _ = actual^
        _ = expected^
    else:
        ctx.synchronize()

    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = a_scales_dev^
    _ = b_scales_dev^
    _ = a_offsets_dev^
    _ = a_scale_offsets_dev^
    _ = expert_ids_dev^
    _ = expert_scales_dev^


def main() raises:
    comptime assert K <= K_EXACT_REFERENCE_MAX, String(
        "K=",
        K,
        " exceeds ",
        K_EXACT_REFERENCE_MAX,
        (
            ", past which the host reference stops being bit-exact and --check"
            " reports false failures"
        ),
    )

    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "num_active_experts=",
                specs[i].num_active_experts,
                "tok_seed=",
                specs[i].tok_seed,
            )
        return

    if mode == "single":
        var nae = flag_int(args, "--num_active_experts", 3)
        var ts = flag_int(args, "--tok_seed", 1)
        print(
            "FUZZ_SINGLE num_active_experts=",
            nae,
            "tok_seed=",
            ts,
            "N=",
            N,
            "K=",
            K,
            "num_experts=",
            num_experts,
        )
        with DeviceContext() as ctx:
            if rerun > 0:
                run_one_case(ctx, CaseSpec(nae, ts), rerun=rerun)
            else:
                run_one_case(ctx, CaseSpec(nae, ts), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_grouped_matmul_sm100_w4a8 seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "num_experts=",
        num_experts,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if rerun > 0:
                run_one_case(ctx, specs[i], rerun=rerun)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
