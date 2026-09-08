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
"""AMD block-scaled grouped matmul bench with workload-driven routing.

Inputs describe the workload (total experts, active experts, batch M,
topk, expert skew, optional shared experts) instead of a hand-built
per-slot token list. The bench builds the routing tables (a_offsets,
expert_ids) so we can sweep one knob at a time.

Runs the preshuffled-B grouped matmul (block_scaled_grouped_matmul_amd_preb:
preshuffled B, direct VGPR loads).

`lane_bytes` selects the format: 16 (MXFP4, default), 24 (MXFP6), or 32 (MXFP8).
A's K byte extent is `K * lane_bytes // 32`; the E8M0 scale extent is `K // 32`.
"""

from std.math import align_up, ceildiv, isnan
from std.os import abort
from std.random import random_ui64, seed
from std.sys import (
    get_defined_int,
    get_defined_string,
)
from max.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from internal_utils import arg_parse, CacheBustingBuffer, CACHE_BUST_BYTES
from internal_utils._utils import InitializationType
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler, block_scaled_grouped_matmul_amd_preb


# ===----------------------------------------------------------------------=== #
# Skew parsing
# ===----------------------------------------------------------------------=== #


def _parse_csv_ints(s: String) raises -> List[Int]:
    # Accept "," or ";" as delimiter. kbench evals every yaml value as Python,
    # so a comma-separated string becomes a tuple and expands into a sweep
    # dimension that explodes the build matrix. Use ";" in yaml to keep the
    # value as one literal string.
    var stripped = s.strip("[]\"' ")
    var out = List[Int]()
    if stripped.byte_length() == 0:
        return out^
    var normalized = stripped.replace(";", ",")
    for tok in normalized.split(","):
        var t = String(tok).strip()
        if t.byte_length() == 0:
            continue
        out.append(Int(t))
    return out^


# ===----------------------------------------------------------------------=== #
# Bench name
# ===----------------------------------------------------------------------=== #


def _format_name(lane_bytes: Int) -> String:
    if lane_bytes == 32:
        return "mxfp8"
    if lane_bytes == 24:
        return "mxfp6"
    return "mxfp4"


def _run_name(
    tag: String,
    lane_bytes: Int,
    num_experts: Int,
    num_active_experts: Int,
    M: Int,
    topk: Int,
    N: Int,
    K: Int,
    skew_spec: String,
) -> String:
    return String(
        tag,
        " ",
        _format_name(lane_bytes),
        " : E=",
        num_experts,
        " A=",
        num_active_experts,
        " M=",
        M,
        " topk=",
        topk,
        " N=",
        N,
        " K=",
        K,
        " skew=",
        skew_spec,
    )


# ===----------------------------------------------------------------------=== #
# Correctness verification (MXFP8 only).
#
# `_verify_buffers_gpu` is ported verbatim from `bench_block_scaled_matmul.mojo`
# and `_block_scaled_matmul_fp8_ref` from `test_mxfp8_matmul_amd_preb.mojo`.
# Kept as second copies rather than a shared-module refactor across
# benchmarks/test — that refactor can't be exercised on a GPU in this
# environment, so a copy is the lower-risk choice here (see the AMD grouped
# matmul autotune plan doc for the tradeoff).
# ===----------------------------------------------------------------------=== #


def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: UnsafePointer[Scalar[c_type], ImmutAnyOrigin],
    reference: UnsafePointer[Scalar[c_type], ImmutAnyOrigin],
    length: Int32,
    atol: Float32,
    rtol: Float32,
    result: UnsafePointer[Float32, MutAnyOrigin],
):
    """GPU kernel that computes verification metrics in one pass.

    Each block computes partial reductions and writes 6 Float32 values:
      [0] abs_diff_sum — for relative difference metric (NaN pairs excluded)
      [1] abs_ref_sum  — for relative difference metric (NaN pairs excluded)
      [2] max_violation — max(|x-y| - (atol + rtol*|y|)) over non-NaN pairs,
          <=0 means pass
      [3] out_nz — 1.0 if any output element is nonzero
      [4] ref_nz — 1.0 if any reference element is nonzero
      [5] nan_mismatch — 1.0 if some element is NaN in the output but not the
          reference, or vice versa

    The synthetic activation fill is a uniform random byte pattern (see
    `_rand_e4m3_byte`'s docstring in `test_mxfp8_matmul_amd_preb.mojo`), which
    occasionally lands on the E4M3 NaN encoding — so a NaN that the
    reference also produces is expected input propagation, not a bug.
    `nan_mismatch` instead flags a NaN appearing on only one side, which a
    real kernel bug (e.g. an uninitialized read in a newly-tuned tile) would
    trigger. `max_violation` alone is NaN-blind: `max()` lowers to maxNum
    semantics, which ignore a NaN operand rather than propagate it, so pairs
    where both sides agree on NaN are excluded from the sums/violation to
    keep those metrics meaningful instead of being silently poisoned to NaN.
    """
    var abs_diff_sum: Float32 = 0
    var abs_ref_sum: Float32 = 0
    var max_violation = Float32.MIN_FINITE
    var out_nz: Float32 = 0
    var ref_nz: Float32 = 0
    var nan_mismatch: Float32 = 0

    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(length):
        var x = output[i].cast[.float32]()
        var y = reference[i].cast[.float32]()
        var x_nan = isnan(x)
        var y_nan = isnan(y)
        if x_nan != y_nan:
            nan_mismatch = 1.0
        if not x_nan and not y_nan:
            abs_diff_sum += abs(x - y)
            abs_ref_sum += abs(y)
            max_violation = max(
                max_violation, abs(x - y) - (atol + rtol * abs(y))
            )
        if x != 0:
            out_nz = 1.0
        if y != 0:
            ref_nz = 1.0
        i += stride

    abs_diff_sum = block.sum[block_size=BLOCK_SIZE](abs_diff_sum)
    abs_ref_sum = block.sum[block_size=BLOCK_SIZE](abs_ref_sum)
    max_violation = block.max[block_size=BLOCK_SIZE](max_violation)
    out_nz = block.max[block_size=BLOCK_SIZE](out_nz)
    ref_nz = block.max[block_size=BLOCK_SIZE](ref_nz)
    nan_mismatch = block.max[block_size=BLOCK_SIZE](nan_mismatch)

    if thread_idx.x == 0:
        var base = block_idx.x * 6
        result[base + 0] = abs_diff_sum
        result[base + 1] = abs_ref_sum
        result[base + 2] = max_violation
        result[base + 3] = out_nz
        result[base + 4] = ref_nz
        result[base + 5] = nan_mismatch


def _block_scaled_matmul_fp8_ref(
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_scales_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    M_arg: Int32,
    N_arg: Int32,
    K_arg: Int32,
):
    """Per-element GPU reference for MXFP8 block-scaled matmul.

    No MFMA, no code shared with the real kernel, so a fragment-layout bug
    in the real kernel can't cancel out against a shared bug in the
    reference.
    """
    var M = Int(M_arg)
    var N = Int(N_arg)
    var K = Int(K_arg)
    var m = global_idx.x
    var n = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP8_SF_VECTOR_SIZE

    var am_scales_ptr = a_scales_ptr + m * k_groups
    var bn_scales_ptr = b_scales_ptr + n * k_groups

    # One byte per element at MXFP8, so the row stride is K.
    var am_ptr = (a_ptr + m * K).bitcast[Float8_e4m3fn]()
    var bn_ptr = (b_ptr + n * K).bitcast[Float8_e4m3fn]()

    var accum = Float32(0)

    for ko in range(k_groups):
        var a_scale = am_scales_ptr[ko].cast[.float32]()
        var b_scale = bn_scales_ptr[ko].cast[.float32]()

        var part = Float32(0)
        for ki in range(MXFP8_SF_VECTOR_SIZE):
            part += am_ptr[ki].cast[.float32]() * bn_ptr[ki].cast[.float32]()
        accum += part * a_scale * b_scale

        am_ptr += MXFP8_SF_VECTOR_SIZE
        bn_ptr += MXFP8_SF_VECTOR_SIZE

    c_ptr[m * N + n] = accum


# ===----------------------------------------------------------------------=== #
# Preshuffled-B path (block_scaled_grouped_matmul_amd_preb)
# ===----------------------------------------------------------------------=== #


def bench_preb[
    num_experts: Int,
    N: Int,
    K: Int,
    num_shared_experts: Int,
    topk: Int,
    lane_bytes: Int,
](
    ctx: DeviceContext,
    mut bench: Bench,
    M: Int,
    num_active_experts: Int,
    target_counts: List[Int],
    expert_id_pool: List[Int],
    skew_spec: String,
    init_type: InitializationType,
    cache_bust: Bool = True,
    cache_bust_gb: Float64 = 0.0,
    max_tokens_capacity: Int = 0,
    estimated_total_m: Int = 0,
    verify: Bool = True,
) raises:
    # A's K byte extent per format: K/2 (MXFP4), K*3/4 (MXFP6), K (MXFP8).
    comptime packed_K = (K * lane_bytes) // 32
    comptime scale_K = K // 32

    # Workload is driven by the explicit per-expert token counts (matches the
    # Blackwell `num_tokens_by_expert` bench), not an `M*topk` fan-out.
    var total_routes = 0
    for i in range(num_active_experts):
        total_routes += target_counts[i]
    var total_flops = 2 * total_routes * N * K

    # Routing tables are tiny metadata read once per launch; keep them as plain
    # single buffers (not cache-busted).
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](num_active_experts)

    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var ei_h = ctx.enqueue_create_host_buffer[.int32](num_active_experts)
    a_off_h[0] = 0
    var max_count = 0
    for e in range(num_active_experts):
        var c = target_counts[e]
        a_off_h[e + 1] = a_off_h[e] + UInt32(c)
        ei_h[e] = Int32(expert_id_pool[e])
        if c > max_count:
            max_count = c
    var max_tokens_for_kernel = (
        max_tokens_capacity if max_tokens_capacity > 0 else max_count
    )
    var max_padded_M = align_up(max_tokens_for_kernel, 32)
    ctx.enqueue_copy(a_offsets_dev, a_off_h)
    ctx.enqueue_copy(expert_ids_dev, ei_h)
    ctx.synchronize()

    comptime simd_size = 4
    # B is the dominant operand. A single full copy is num_experts*N*packed_K
    # bytes; the kernel only reads the active experts' slices, so at decode
    # (few active experts) that working set stays resident unless we rotate
    # through many full-B copies. CACHE_BUST_BYTES is 2x the GPU cache, so
    # CACHE_BUST_BYTES * num_experts gives enough windows to evict even a
    # single-expert (M=1) read with a 2x margin (= 24 GiB at 48 experts). The
    # max() keeps >=2 windows if one full copy is itself larger than that.
    # Override with cache_bust_gb to set an explicit footprint.
    var b_full_bytes = num_experts * N * packed_K
    var b_budget = Int(
        cache_bust_gb * (1024.0 * 1024.0 * 1024.0)
    ) if cache_bust_gb > 0.0 else max(
        CACHE_BUST_BYTES * num_experts, 2 * b_full_bytes
    )
    var b_windows = ceildiv(b_budget, align_up(b_full_bytes, simd_size))
    print(
        "  cache_bust=",
        cache_bust,
        " B full=",
        b_full_bytes // (1024 * 1024),
        "MiB budget=",
        b_budget // (1024 * 1024),
        "MiB windows=",
        b_windows,
    )
    var cb_a = CacheBustingBuffer[.uint8](
        total_routes * packed_K, simd_size, ctx, cache_bust
    )
    var cb_b = CacheBustingBuffer[.uint8](
        num_experts * N * packed_K,
        simd_size,
        ctx,
        cache_bust,
        budget_bytes=b_budget,
    )
    var cb_c = CacheBustingBuffer[.float32](
        total_routes * N, simd_size, ctx, cache_bust
    )
    # Scale buffers (uint8 — the dispatcher reads these in scale-4d byte order
    # via PreshuffledScaleLoader). Fill the whole buffer with a valid E8M0 byte
    # (0x7F = magnitude 1).
    var cb_a_sc = CacheBustingBuffer[.uint8](
        num_experts * max_padded_M * scale_K, simd_size, ctx, cache_bust
    )
    var cb_b_sc = CacheBustingBuffer[.uint8](
        num_experts * N * scale_K, simd_size, ctx, cache_bust
    )

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    ctx.enqueue_memset(cb_a_sc.device_buffer(), UInt8(127))
    ctx.enqueue_memset(cb_b_sc.device_buffer(), UInt8(127))

    var aoff_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var ei_tt = TileTensor(expert_ids_dev, row_major(Coord(num_active_experts)))

    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
        var a_tt = TileTensor[mut=False](
            cb_a.offset_ptr(iteration),
            row_major(Coord(total_routes, Idx[packed_K])),
        )
        var b_pre_tt = TileTensor[mut=False](
            cb_b.offset_ptr(iteration), row_major[num_experts, N * packed_K]()
        )
        # Bitcast uint8 → float8_e8m0fnu at the TileTensor wrap to match the
        # dispatcher signature. The kernel internally re-bitcasts to uint8 for
        # V# construction; the dtype is a wrapping convention.
        var sfa_tt = TileTensor[mut=False](
            cb_a_sc.offset_ptr(iteration).bitcast[Float8_e8m0fnu](),
            row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
        )
        var sfb_tt = TileTensor[mut=False](
            cb_b_sc.offset_ptr(iteration).bitcast[Float8_e8m0fnu](),
            row_major[num_experts, N, scale_K](),
        )
        var c_tt = TileTensor[mut=True](
            cb_c.offset_ptr(iteration), row_major(Coord(total_routes, Idx[N]))
        )
        # Pass lane_bytes explicitly: every format is uint8, and (N, packed_K)
        # is not unique across formats.
        block_scaled_grouped_matmul_amd_preb[lane_bytes=lane_bytes](
            c_tt,
            a_tt,
            b_pre_tt,
            sfa_tt,
            sfb_tt,
            aoff_tt,
            ei_tt,
            max_tokens_for_kernel,
            num_active_experts,
            ctx,
            estimated_total_m,
        )

    @always_inline
    def bench_func(mut bencher: Bencher) {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    bench.bench_function(
        bench_func,
        BenchId(
            _run_name(
                "gmm_amd_preb (uint8 -> float32)",
                lane_bytes,
                num_experts,
                num_active_experts,
                M,
                topk,
                N,
                K,
                skew_spec,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    # Correctness verify: reference is only implemented for MXFP8
    # (lane_bytes=32) — the two grouped-matmul shapes this autotune targets. Runs
    # the timed benchmark so a verify failure never affects the reported
    # perf number. `verify` is a runtime Bool so it must be the outer check;
    # a `comptime if`'s branches must all be comptime-decidable, so the
    # `lane_bytes == 32` gate nests inside it rather than sharing one chain.
    if verify:
        comptime if lane_bytes == 32:
            print("  verifying vs. per-element MXFP8 reference...")
            comptime BLOCK_DIM = 32

            # `cb_b`'s iteration-0 slot is random bytes with no prior layout
            # meaning, so it doubles as the natural [num_experts, N,
            # packed_K] source `preshuffle_b_5d` shuffles into a genuinely
            # preshuffled buffer. This step is required: the real kernel's
            # `PreshuffledBLoader` addressing depends on B actually being in
            # preshuffled byte order, unlike the scale buffers below.
            var b_raw_tt = TileTensor[mut=False](
                cb_b.offset_ptr(0), row_major[num_experts, N, packed_K]()
            )
            var verify_b_pre_dev = ctx.enqueue_create_buffer[.uint8](
                num_experts * N * packed_K
            )
            var b_pre_dst_tt = TileTensor[mut=True](
                verify_b_pre_dev,
                Shuffler[num_experts].b_5d_grouped_layout[
                    N=N, K_BYTES=packed_K
                ],
            )
            Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=packed_K](
                b_raw_tt, b_pre_dst_tt, ctx
            )

            # Both scale buffers are a constant E8M0=127 (2^0=1.0) byte
            # everywhere (see the memset above) — a byte permutation of an
            # all-identical buffer is the identity, so `cb_a_sc`/`cb_b_sc`
            # are valid inputs to BOTH the real (preshuffled-addressing)
            # kernel and the natural-layout reference without running the
            # scale preshuffle kernels.
            var verify_sfa_natural_dev = ctx.enqueue_create_buffer[.uint8](
                total_routes * scale_K
            )
            ctx.enqueue_memset(verify_sfa_natural_dev, UInt8(127))

            var verify_c_dev = ctx.enqueue_create_buffer[.float32](
                total_routes * N
            )
            var verify_c_ref_dev = ctx.enqueue_create_buffer[.float32](
                total_routes * N
            )

            var a_tt_v = TileTensor[mut=False](
                cb_a.offset_ptr(0),
                row_major(Coord(total_routes, Idx[packed_K])),
            )
            var b_pre_flat_v = TileTensor[mut=False](
                verify_b_pre_dev, row_major[num_experts, N * packed_K]()
            )
            var sfa_tt_v = TileTensor[mut=False](
                cb_a_sc.offset_ptr(0).bitcast[Float8_e8m0fnu](),
                row_major(Coord(num_experts * max_padded_M, Idx[scale_K])),
            )
            var sfb_tt_v = TileTensor[mut=False](
                cb_b_sc.offset_ptr(0).bitcast[Float8_e8m0fnu](),
                row_major[num_experts, N, scale_K](),
            )
            var c_tt_v = TileTensor[mut=True](
                verify_c_dev, row_major(Coord(total_routes, Idx[N]))
            )

            block_scaled_grouped_matmul_amd_preb[lane_bytes=lane_bytes](
                c_tt_v,
                a_tt_v,
                b_pre_flat_v,
                sfa_tt_v,
                sfb_tt_v,
                aoff_tt,
                ei_tt,
                max_tokens_for_kernel,
                num_active_experts,
                ctx,
                estimated_total_m,
            )

            # Reference: one ungrouped per-element matmul per active expert
            # slot, reading the natural (un-preshuffled) A/B/scale buffers.
            for slot in range(num_active_experts):
                var tok_start = Int(a_off_h[slot])
                var m_slot = target_counts[slot]
                if m_slot == 0:
                    continue
                var eid = Int(ei_h[slot])
                ctx.enqueue_function[_block_scaled_matmul_fp8_ref](
                    (cb_a.offset_ptr(0) + tok_start * packed_K).as_imm(),
                    (cb_b.offset_ptr(0) + eid * N * packed_K).as_imm(),
                    (verify_sfa_natural_dev.unsafe_ptr() + tok_start * scale_K)
                    .bitcast[Float8_e8m0fnu]()
                    .as_imm(),
                    (cb_b_sc.offset_ptr(0) + eid * N * scale_K)
                    .bitcast[Float8_e8m0fnu]()
                    .as_imm(),
                    verify_c_ref_dev.unsafe_ptr() + tok_start * N,
                    Int32(m_slot),
                    Int32(N),
                    Int32(K),
                    grid_dim=(
                        ceildiv(m_slot, BLOCK_DIM),
                        ceildiv(N, BLOCK_DIM),
                    ),
                    block_dim=(BLOCK_DIM, BLOCK_DIM),
                )

            comptime NUM_BLOCKS = 32
            comptime VERIFY_BLOCK_SIZE = 256
            var rtol = Float32(0.05)
            var atol = Float32(0.05)
            var result_device = ctx.enqueue_create_buffer[.float32](
                NUM_BLOCKS * 6
            )
            comptime verify_kernel = _verify_buffers_gpu[
                DType.float32, VERIFY_BLOCK_SIZE
            ]
            ctx.enqueue_function[verify_kernel](
                verify_c_dev.unsafe_ptr(),
                verify_c_ref_dev.unsafe_ptr(),
                Int32(total_routes * N),
                atol,
                rtol,
                result_device,
                grid_dim=NUM_BLOCKS,
                block_dim=VERIFY_BLOCK_SIZE,
            )

            var result_host = List(length=NUM_BLOCKS * 6, fill=Float32(0))
            ctx.enqueue_copy(result_host, result_device)
            ctx.synchronize()

            var total_abs_diff: Float32 = 0
            var total_abs_ref: Float32 = 0
            var worst_violation = Float32.MIN_FINITE
            var out_nz: Float32 = 0
            var ref_nz: Float32 = 0
            var nan_mismatch: Float32 = 0
            for bi in range(NUM_BLOCKS):
                var base = bi * 6
                total_abs_diff += result_host[base + 0]
                total_abs_ref += result_host[base + 1]
                worst_violation = max(worst_violation, result_host[base + 2])
                out_nz = max(out_nz, result_host[base + 3])
                ref_nz = max(ref_nz, result_host[base + 4])
                nan_mismatch = max(nan_mismatch, result_host[base + 5])

            if nan_mismatch != 0:
                raise (
                    "CORRECTNESS: kernel output and reference disagree on"
                    " NaN placement"
                )
            if out_nz == 0:
                raise "CORRECTNESS: kernel output is all zeros"
            if ref_nz == 0:
                raise "CORRECTNESS: reference output is all zeros"

            if total_abs_ref > 0:
                var rel_diff = total_abs_diff / total_abs_ref
                if rel_diff > rtol:
                    raise String(
                        "CORRECTNESS: relative difference ",
                        rel_diff,
                        " > ",
                        rtol,
                    )

            if worst_violation > 0:
                raise String(
                    "CORRECTNESS: element-wise tolerance violated, worst = ",
                    worst_violation,
                )

            print("  verify PASSED (atol=", atol, " rtol=", rtol, ")")
            _ = result_host^
            _ = verify_b_pre_dev^
            _ = verify_sfa_natural_dev^
            _ = verify_c_dev^
            _ = verify_c_ref_dev^
        else:
            print(
                "  verify: skipped (reference only implemented for"
                " lane_bytes=32 / MXFP8)"
            )

    _ = cb_a^
    _ = cb_b^
    _ = cb_a_sc^
    _ = cb_b_sc^
    _ = cb_c^
    _ = a_offsets_dev^
    _ = expert_ids_dev^


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main() raises:
    comptime num_experts = get_defined_int["num_experts", 8]()
    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 7168]()
    comptime topk = get_defined_int["topk", 1]()
    comptime num_shared_experts = get_defined_int["num_shared_experts", 0]()
    comptime algo = get_defined_string["algo", "preb"]()
    # 16 = MXFP4 (default), 24 = MXFP6, 32 = MXFP8.
    comptime lane_bytes = get_defined_int["lane_bytes", 16]()

    comptime assert K % 128 == 0, "K must be a multiple of 128 (MFMA K dim)"
    comptime assert (
        lane_bytes == 16 or lane_bytes == 24 or lane_bytes == 32
    ), "lane_bytes must be 16 (MXFP4), 24 (MXFP6) or 32 (MXFP8)"
    # Preb K granule is 256 at MXFP8 and 512 at MXFP4/MXFP6; smaller K uses
    # the non-preb entry.
    comptime _k_granule = 256 if lane_bytes == 32 else 512
    comptime assert K >= _k_granule and K % _k_granule == 0, (
        "K must be a nonzero multiple of 256 (MXFP8) or 512 (MXFP4/MXFP6) for"
        " the preb path"
    )
    comptime assert (
        num_shared_experts <= topk
    ), "num_shared_experts cannot exceed topk"
    comptime assert (
        algo == "preb"
    ), "algo must be 'preb' (the 'dense' and 'routed' paths were removed)"

    var num_active_experts = Int(arg_parse("num_active_experts", 1))
    var M = Int(arg_parse("M", 256))
    # Expert-parallel degree: experts are sharded across this many GPUs, so a
    # token's `topk` global picks land on this rank's local shard only ~1/N of
    # the time. Per-rank routed-M = M * topk // n_gpus_per_node, matching
    # `nn/moe/expert_parallel.py` (total_tokens * topk // n_gpus_per_node).
    var n_gpus_per_node = Int(arg_parse("n_gpus_per_node", 4))
    var skew_spec = String(arg_parse("expert_skew", "uniform"))
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    # Production simulates a capacity-bound `max_num_tokens_per_expert`
    # (e.g. 8192 = `max_batch_input_tokens` default) regardless of actual
    # routing. When >0 this override is forwarded to the kernel instead of
    # the routing-derived max. Default 0 = use actual.
    var max_tokens_capacity = Int(arg_parse("max_tokens_capacity", 0))
    var cache_bust = Bool(arg_parse("cache_bust", True))
    # Buffer footprint (GiB) for the B-weights cache-busting buffer. 0 = auto:
    # 512 MiB * num_experts (enough windows to evict even a single-expert M=1
    # read; = 24 GiB at 48 experts). Set explicitly to cap or raise the budget.
    var cache_bust_gb = Float64(arg_parse("cache_bust_gb", 0.0))
    # Correctness verify against a per-element reference (MXFP8 only; see
    # bench_preb). Default True: this kernel's whole output is the tuned
    # low-precision path, so it never gets an fp8-disables-default carve-out.
    var verify = Bool(arg_parse("verify", True))
    # Replay a literal serve routing: pass comma-separated per-slot token
    # counts and expert IDs. Both must be set together and both must have
    # length == num_active_experts. Bypasses skew synthesis.
    var target_counts_csv = String(arg_parse("target_counts", ""))
    var expert_ids_csv = String(arg_parse("expert_ids", ""))

    if num_active_experts < num_shared_experts:
        abort(
            "num_active_experts="
            + String(num_active_experts)
            + " < num_shared_experts="
            + String(num_shared_experts)
        )
    if num_active_experts > num_experts:
        abort(
            "num_active_experts="
            + String(num_active_experts)
            + " > num_experts="
            + String(num_experts)
        )
    var target_counts = List[Int]()
    var expert_id_pool = List[Int]()
    var have_counts = target_counts_csv.byte_length() != 0
    var have_ids = expert_ids_csv.byte_length() != 0
    if not have_counts and not have_ids:
        # Synthesize realistic EP-sharded MoE routing. Each of `M` tokens routes
        # to `topk` distinct experts out of the GLOBAL pool
        # (num_experts * n_gpus_per_node); only picks landing on THIS rank's
        # local shard ([0, num_experts)) are processed here. So per-rank routed
        # rows ~= M * topk // n_gpus_per_node, matching
        # `nn/moe/expert_parallel.py` (total_tokens * topk // n_gpus_per_node),
        # and active experts follow coupon-collector over the local shard.
        # n_gpus_per_node=1 reduces to a single-GPU run (all picks local).
        # `seed(0)` keeps the routing reproducible across runs.
        seed(0)
        var num_tokens = M if M > 0 else 1
        var gpus = max(n_gpus_per_node, 1)
        var global_experts = num_experts * gpus
        var k = min(topk, global_experts)
        var counts = List[Int]()
        for _ in range(num_experts):
            counts.append(0)
        for _ in range(num_tokens):
            var picks = List[Int]()
            while len(picks) < k:
                var e = Int(random_ui64(0, UInt64(global_experts - 1)))
                var dup = False
                for i in range(len(picks)):
                    if picks[i] == e:
                        dup = True
                        break
                if not dup:
                    picks.append(e)
            # Count only picks that land on this rank's local shard.
            for i in range(len(picks)):
                if picks[i] < num_experts:
                    counts[picks[i]] += 1
        for e in range(num_experts):
            if counts[e] > 0:
                target_counts.append(counts[e])
                expert_id_pool.append(e)
        num_active_experts = len(target_counts)
    elif not have_counts or not have_ids:
        abort(
            "target_counts and expert_ids must be set together, or both"
            " omitted to synthesize uniform routing from M"
        )
    else:
        target_counts = _parse_csv_ints(target_counts_csv)
        expert_id_pool = _parse_csv_ints(expert_ids_csv)
        if len(target_counts) != num_active_experts:
            abort(
                "target_counts length="
                + String(len(target_counts))
                + " must equal num_active_experts="
                + String(num_active_experts)
            )
        if len(expert_id_pool) != num_active_experts:
            abort(
                "expert_ids length="
                + String(len(expert_id_pool))
                + " must equal num_active_experts="
                + String(num_active_experts)
            )
        for e in expert_id_pool:
            if e < 0 or e >= num_experts:
                abort(
                    "expert_ids contains out-of-range value="
                    + String(e)
                    + " (num_experts="
                    + String(num_experts)
                    + ")"
                )

    print(
        "Config: algo=",
        algo,
        " lane_bytes=",
        lane_bytes,
        " num_experts=",
        num_experts,
        " num_active_experts=",
        num_active_experts,
        " num_shared=",
        num_shared_experts,
        " M=",
        M,
        " topk=",
        topk,
        " n_gpus_per_node=",
        n_gpus_per_node,
        " N=",
        N,
        " K=",
        K,
        " skew=",
        skew_spec,
    )
    print("  target_counts(len=", len(target_counts), "):", end=" ")
    for c in target_counts:
        print(c, end=" ")
    print()
    print("  expert_id_pool(len=", len(expert_id_pool), "):", end=" ")
    for e in expert_id_pool:
        print(e, end=" ")
    print()

    with DeviceContext() as ctx:
        var bench = Bench()

        # `estimated_total_m` drives the preb dispatcher's band +
        # persistent-vs-direct switch. Production computes it from the routing
        # FORMULA (an estimate available at dispatch time), not the exact
        # per-expert counts — see nn/moe/expert_parallel.py:
        #   estimated_total_m = total_tokens * topk // n_gpus_per_node
        var estimated_total_m = M * topk // max(n_gpus_per_node, 1)
        bench_preb[
            num_experts=num_experts,
            N=N,
            K=K,
            num_shared_experts=num_shared_experts,
            topk=topk,
            lane_bytes=lane_bytes,
        ](
            ctx,
            bench,
            M,
            num_active_experts,
            target_counts,
            expert_id_pool,
            skew_spec,
            init_type,
            cache_bust,
            cache_bust_gb,
            max_tokens_capacity,
            estimated_total_m,
            verify,
        )

        bench.dump_report()
