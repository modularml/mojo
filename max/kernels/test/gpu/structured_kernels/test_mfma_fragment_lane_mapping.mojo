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
"""Isolated test of the AMD MFMA `gpu_mma` intrinsic for two shapes.

Shapes covered:

  - BF16 32x32x16 (`v_mfma_f32_32x32x16_bf16`)        - MHA kernel QK/PV
  - FP8  32x32x64 (`v_mfma_scale_f32_32x32x64_f8f6f4`) - MHA kernel FP8 QK

This strips every layer above `gpu_mma`: no SMEM, no swizzle, no loads.
Each lane materializes its A and B fragments directly from per-lane
SIMD literals. We compute one MMA and dump the per-lane C fragment.

Strategy: set A = all-ones, B[k, n] = (k + 1). Then mathematically
`C[m, n] = sum_k A[m, k] * B[k, n] = sum_k (k + 1) = K * (K + 1) / 2`
(uniform across (m, n)). To produce this B-layout from per-lane B
fragments, we apply the inverse of the documented AMD B-operand
lane mapping to determine what each per-lane element should be.

If the dumped per-lane C matches the documented C-fragment layout,
then `gpu_mma` is consistent with the spec used by the MHA/MLA kernel code.
If it fails, the lane-mapping doc is wrong, and all FP8-cast logic
downstream is built on bad assumptions.
"""

from max.gpu import lane_id, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.compute.mma import mma as gpu_mma
from std.memory import AddressSpace
from std.testing import assert_equal

from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation


# v_mfma_f32_32x32x16_bf16 accumulator layout (reference kernel.cpp lines 124-165,
# also documented in `MhaMmaOp.ACC_ROW_OFFSETS_32x32`).
# Per BASE TILE, each lane holds 16 fp32 arranged as 8 packed data[idx]
# (.x, .y) entries, at rows:
comptime ACC_ROW_OFFSETS_32x32 = SIMD[.int32, 16](
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
)
# Plus +4 if lane >= 32. Col within tile is just (lane & 31).


# AMD MFMA fragment counts per lane for wave64:
#   BF16 32x32x16: A and B = (M*K)/64 = 8 BF16 each lane, C = 16 fp32
#   FP8  32x32x64: A and B = (M*K)/64 = 32 FP8 each lane, C = 16 fp32
comptime BF16_FRAG = 8
comptime FP8_FRAG = 32
comptime C_FRAG = 16

# Total bytes per fragment dump (one base tile, 64 lanes, 16 fp32 each)
comptime C_DUMP_SIZE = 64 * C_FRAG


# ---------------------------------------------------------------------------
# C-fragment lane mapping (col_l rt_32x32) decoded into (row, col).
#
# Per element index p in [0, 16):
#   row_in_tile = ACC_ROW_OFFSETS_32x32[p] + (4 if lane >= 32 else 0)
#   col_in_tile = lane % 32
# ---------------------------------------------------------------------------
def _c_row_for(lane: Int, p: Int) -> Int:
    return Int(ACC_ROW_OFFSETS_32x32[p]) + (4 if lane >= 32 else 0)


def _c_col_for(lane: Int) -> Int:
    return lane % 32


# ===========================================================================
# BF16 32x32x16 lane mapping for B-operand
# ===========================================================================
# For v_mfma_f32_32x32x16_bf16, B is a 16x32 (K x N) matrix.
# Each lane holds BF16_FRAG=8 elements of B. The standard CDNA lane mapping
# (matching the reference's load_K row_l fragment formula) is:
#
#   n_in_tile = lane % 32          (B-column)
#   k_in_tile = (lane // 32) * 8 + elt   for elt in [0, 8)
#
# (i.e. the half-warp owns 8 contiguous K-rows; high half owns rows 8..15,
# low half owns rows 0..7. Each lane spans the 8 K rows of its half at
# the same N column.)
def _bf16_b_k_for(lane: Int, elt: Int) -> Int:
    return (lane // 32) * 8 + elt


def _bf16_b_n_for(lane: Int) -> Int:
    return lane % 32


# Similarly for A in BF16 32x32x16: A is 32x16 (M x K). Each lane holds
# BF16_FRAG=8 elements arranged as 8 K-elements at one M row. Lane mapping:
#   m_in_tile = lane % 32
#   k_in_tile = (lane // 32) * 8 + elt
def _bf16_a_m_for(lane: Int) -> Int:
    return lane % 32


def _bf16_a_k_for(lane: Int, elt: Int) -> Int:
    return (lane // 32) * 8 + elt


# ===========================================================================
# FP8 32x32x64 lane mapping for B-operand
# ===========================================================================
# For v_mfma_scale_f32_32x32x64_f8f6f4, B is 64x32 (K x N). Each lane holds
# FP8_FRAG=32 FP8 elements. Standard CDNA4 layout (extrapolated from BF16
# mapping by the K-expansion ratio, 16->64):
#
#   n_in_tile = lane % 32
#   k_in_tile = (lane // 32) * 32 + elt   for elt in [0, 32)
def _fp8_b_k_for(lane: Int, elt: Int) -> Int:
    return (lane // 32) * 32 + elt


def _fp8_b_n_for(lane: Int) -> Int:
    return lane % 32


# A side for FP8 32x32x64: A is 32x64. Each lane holds 32 FP8 at one m row:
#   m_in_tile = lane % 32
#   k_in_tile = (lane // 32) * 32 + elt
def _fp8_a_m_for(lane: Int) -> Int:
    return lane % 32


def _fp8_a_k_for(lane: Int, elt: Int) -> Int:
    return (lane // 32) * 32 + elt


# ===========================================================================
# Kernel: BF16 32x32x16 — uniform C value (sum-of-k validation)
# ===========================================================================
def kernel_bf16_32x32x16(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Per-lane test of v_mfma_f32_32x32x16_bf16 with uniform output.

    A[m, k] = 1.0   (per-lane: a_frag[elt] = 1.0)
    B[k, n] = k + 1 (per-lane: b_frag[elt] = (lane // 32) * 8 + elt + 1)

    Expected: C[m, n] = sum_k (k + 1) = 16 * 17 / 2 = 136 (uniform).
    This validates the sum/K-axis traversal only. Non-uniform tests
    below stress the row/col fragment mapping.

    Dump layout: dump[lane * 16 + p] = c_frag[p].
    """
    var lid = Int(lane_id())

    # A: SIMD[bf16, 8] — all ones (every element of A is 1.0).
    var a_frag = SIMD[.bfloat16, BF16_FRAG](1.0)

    # B: SIMD[bf16, 8] — element `elt` gets value k+1 where k is the
    # K-index this lane/elt owns. With the above mapping:
    #   b_frag[elt] = (lane // 32) * 8 + elt + 1
    var b_frag = SIMD[.bfloat16, BF16_FRAG](0)
    for elt in range(BF16_FRAG):
        b_frag[elt] = BFloat16(_bf16_b_k_for(lid, elt) + 1)

    # C: SIMD[fp32, 16] — zero accumulator.
    var c_frag = SIMD[.float32, C_FRAG](0.0)

    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    # Dump per-lane C frag.
    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Kernel: BF16 32x32x16 — non-uniform C (validates row/col fragment layout)
# ===========================================================================
def kernel_bf16_32x32x16_col(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Stress per-lane (row, col) decomposition of the C fragment.

    A[m, k] = 1.0
    B[k, n] = n     (each B-column has all rows equal to that column index)

    Expected: C[m, n] = sum_k 1 * n = MMA_K * n = 16 * n.
    Per-lane verification: c_frag[p] == 16 * col_in_tile(lane).
    """
    var lid = Int(lane_id())

    var a_frag = SIMD[.bfloat16, BF16_FRAG](1.0)

    # B[k, n] = n. Each lane's per-elt n is constant in elt (= lane % 32).
    var b_frag = SIMD[.bfloat16, BF16_FRAG](0)
    var n_val = BFloat16(_bf16_b_n_for(lid))
    for elt in range(BF16_FRAG):
        b_frag[elt] = n_val

    var c_frag = SIMD[.float32, C_FRAG](0.0)
    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


def kernel_bf16_32x32x16_row(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Stress per-lane row mapping by placing the M-index in A.

    A[m, k] = m     (every K-element of row m has value m)
    B[k, n] = 1

    Expected: C[m, n] = sum_k m * 1 = MMA_K * m = 16 * m.
    Per-lane verification: c_frag[p] == 16 * row_in_tile(lane, p).
    """
    var lid = Int(lane_id())

    # A[m, k] = m. Per-lane: m = lane % 32, constant across elts.
    var a_frag = SIMD[.bfloat16, BF16_FRAG](0)
    var m_val = BFloat16(_bf16_a_m_for(lid))
    for elt in range(BF16_FRAG):
        a_frag[elt] = m_val

    var b_frag = SIMD[.bfloat16, BF16_FRAG](1.0)

    var c_frag = SIMD[.float32, C_FRAG](0.0)
    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Kernel: FP8 32x32x64 — non-uniform variants
# ===========================================================================
def kernel_fp8_32x32x64_col(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """FP8 analogue of `kernel_bf16_32x32x16_col`.

    A[m, k] = 1.0
    B[k, n] = quantized(n / 32), exact-FP8 nearest representable.

    Strategy: encode n as Float32, cast to FP8 e4m3fn (lossy), and use
    that quantized value. Host validation re-casts to FP8 the same way
    so expected matches what FP8 actually carries through the MFMA.
    """
    var lid = Int(lane_id())

    var a_frag = SIMD[.float8_e4m3fn, FP8_FRAG](1.0)

    var b_frag = SIMD[.float8_e4m3fn, FP8_FRAG](0)
    var n_val = Float32(_fp8_b_n_for(lid)) / 32.0
    var n_fp8 = n_val.cast[.float8_e4m3fn]()
    for elt in range(FP8_FRAG):
        b_frag[elt] = n_fp8

    var c_frag = SIMD[.float32, C_FRAG](0.0)
    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


def kernel_fp8_32x32x64_row(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """FP8 analogue of `kernel_bf16_32x32x16_row`. Tests row mapping.

    A[m, k] = quantized(m / 32), exact-FP8 nearest representable.
    B[k, n] = 1.
    """
    var lid = Int(lane_id())

    var a_frag = SIMD[.float8_e4m3fn, FP8_FRAG](0)
    var m_val = Float32(_fp8_a_m_for(lid)) / 32.0
    var m_fp8 = m_val.cast[.float8_e4m3fn]()
    for elt in range(FP8_FRAG):
        a_frag[elt] = m_fp8

    var b_frag = SIMD[.float8_e4m3fn, FP8_FRAG](1.0)

    var c_frag = SIMD[.float32, C_FRAG](0.0)
    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Kernel: BF16 32x32x16 swap_a_b
# ===========================================================================
def kernel_bf16_32x32x16_swap(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Emulates swap_a_b by calling gpu_mma with A and B swapped.

    Set the *other* operand to be the K-pattern; A becomes B and B becomes A.
    With swap, gpu_mma(c, b, a, c) computes B @ A semantically. For B[k,n]=1
    and A[m,k]=k+1, the result is C[n, m] = sum_k 1 * (k+1) = K*(K+1)/2,
    but interpretation of (m, n) in C-fragment is transposed.

    The C-fragment layout is hardware-determined and DOES NOT depend on
    which operand is A vs B in the call; so the dump still uses
    (m=col_in_tile, n=row_in_tile) interpretation. We expect the SAME
    numeric value (136 uniform) because the formula is symmetric:
    sum_k 1*(k+1) == sum_k (k+1)*1.
    """
    var lid = Int(lane_id())

    # Now we want B @ A semantically; B=ones, A=k-pattern.
    # In the swap call, original B operand (passed first) is the new "A
    # role". So make orig_a = k-pattern and orig_b = 1.
    var a_frag = SIMD[.bfloat16, BF16_FRAG](0)
    for elt in range(BF16_FRAG):
        a_frag[elt] = BFloat16(_bf16_a_k_for(lid, elt) + 1)
    var b_frag = SIMD[.bfloat16, BF16_FRAG](1.0)

    var c_frag = SIMD[.float32, C_FRAG](0.0)

    # Swapped argument order: gpu_mma(c, b, a, c).
    gpu_mma(c_frag, b_frag, a_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Kernel: FP8 32x32x64
# ===========================================================================
def kernel_fp8_32x32x64(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Per-lane test of v_mfma_scale_f32_32x32x64_f8f6f4 (e4m3fn).

    A[m, k] = 1.0
    B[k, n] = ((k % 8) + 1) / 8     # FP8 range-safe (<= 1.0)
        - chosen so K=64 sum stays well below FP8 product overflow
          and never aliases the (m, n) row/col indices.

    Expected: C[m, n] = sum_k ((k % 8) + 1) / 8
                     = 8 * (1+2+...+8) / 8     [8 full cycles of 0..7]
                     = 36                       (uniform across all m, n)
    """
    var lid = Int(lane_id())

    # A: SIMD[fp8, 32] — all ones.
    var a_frag = SIMD[.float8_e4m3fn, FP8_FRAG](1.0)

    # B: SIMD[fp8, 32] — element elt gets value ((k % 8) + 1) / 8.
    var b_frag = SIMD[.float8_e4m3fn, FP8_FRAG](0)
    for elt in range(FP8_FRAG):
        var k = _fp8_b_k_for(lid, elt)
        var val = Float32((k % 8) + 1) * 0.125
        b_frag[elt] = val.cast[.float8_e4m3fn]()

    var c_frag = SIMD[.float32, C_FRAG](0.0)

    gpu_mma(c_frag, a_frag, b_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Kernel: FP8 32x32x64 swap_a_b
# ===========================================================================
def kernel_fp8_32x32x64_swap(
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Emulates swap_a_b for FP8 32x32x64. Same idea as BF16 swap test."""
    var lid = Int(lane_id())

    # orig_a = k-pattern, orig_b = ones; swap brings k-pattern into the
    # B-operand slot.
    var a_frag = SIMD[.float8_e4m3fn, FP8_FRAG](0)
    for elt in range(FP8_FRAG):
        var k = _fp8_a_k_for(lid, elt)
        var val = Float32((k % 8) + 1) * 0.125
        a_frag[elt] = val.cast[.float8_e4m3fn]()
    var b_frag = SIMD[.float8_e4m3fn, FP8_FRAG](1.0)

    var c_frag = SIMD[.float32, C_FRAG](0.0)

    gpu_mma(c_frag, b_frag, a_frag, c_frag)

    for p in range(C_FRAG):
        dump_ptr[lid * C_FRAG + p] = c_frag[p]


# ===========================================================================
# Host-side validation: each test inlines the loop with its own host_dump
# binding so we can use the buffer's __getitem__ without needing a generic
# host-buffer parameter type.
# ===========================================================================


def test_bf16_32x32x16(ctx: DeviceContext) raises:
    print("--- BF16 32x32x16 (gpu_mma direct) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    # Zero-init dev buffer so unwritten dump entries don't carry old state.
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_bf16_32x32x16](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    # K=16, sum_{k=0..15} (k+1) = 16*17/2 = 136.
    var expected: Float32 = 136.0
    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  BF16 32x32x16: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  BF16 32x32x16: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " expected=",
        expected,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_bf16_32x32x16_col(ctx: DeviceContext) raises:
    """Validates per-lane C-fragment col mapping for BF16 32x32x16."""
    print("--- BF16 32x32x16 col-pattern (validates fragment col mapping) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_bf16_32x32x16_col](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        var expected = Float32(16 * _c_col_for(lane))
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  BF16 col: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  BF16 32x32x16 col: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_bf16_32x32x16_row(ctx: DeviceContext) raises:
    """Validates per-lane C-fragment row mapping for BF16 32x32x16."""
    print("--- BF16 32x32x16 row-pattern (validates fragment row mapping) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_bf16_32x32x16_row](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var expected = Float32(16 * _c_row_for(lane, p))
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  BF16 row: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  BF16 32x32x16 row: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


@always_inline
def _fp8_quantize(x: Float32) -> Float32:
    """Round-trip a value through FP8 e4m3fn to get the nearest representable value.

    Used by the FP8 col/row validators to compute expected values that
    match what the MFMA actually sees (FP8-quantized inputs).
    """
    return Float32(x.cast[.float8_e4m3fn]())


def test_fp8_32x32x64_col(ctx: DeviceContext) raises:
    """Validates per-lane C-fragment col mapping for FP8 32x32x64."""
    print("--- FP8 32x32x64 col-pattern (KEY DATAPOINT for fragment col) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_fp8_32x32x64_col](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    # Account for FP8 quantization: n/32 is rounded into FP8 e4m3 before
    # the MFMA, so expected is 64 * fp8_quantize(n/32).
    var tol: Float32 = 0.1
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        var n = _c_col_for(lane)
        var n_quant = _fp8_quantize(Float32(n) / 32.0)
        var expected = 64.0 * n_quant
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  FP8 col: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  FP8 32x32x64 col: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_fp8_32x32x64_row(ctx: DeviceContext) raises:
    """Validates per-lane C-fragment row mapping for FP8 32x32x64."""
    print("--- FP8 32x32x64 row-pattern (KEY DATAPOINT for fragment row) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_fp8_32x32x64_row](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    var tol: Float32 = 0.1
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var m = _c_row_for(lane, p)
            var m_quant = _fp8_quantize(Float32(m) / 32.0)
            var expected = 64.0 * m_quant
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  FP8 row: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  FP8 32x32x64 row: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_bf16_32x32x16_swap(ctx: DeviceContext) raises:
    print("--- BF16 32x32x16 swap_a_b ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_bf16_32x32x16_swap](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    # Formula is symmetric: still 136.
    var expected: Float32 = 136.0
    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  BF16 swap: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  BF16 32x32x16 swap: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " expected=",
        expected,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_fp8_32x32x64(ctx: DeviceContext) raises:
    print("--- FP8 32x32x64 (gpu_mma direct) ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_fp8_32x32x64](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    # K=64, B[k,n] = ((k % 8) + 1) / 8. So 8 cycles of {1,2,..,8}/8.
    # Per cycle sum = (1+2+..+8)/8 = 36/8 = 4.5. Total = 8 * 4.5 = 36.
    var expected: Float32 = 36.0
    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  FP8 32x32x64: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " (row=",
                        _c_row_for(lane, p),
                        " col=",
                        _c_col_for(lane),
                        ") got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  FP8 32x32x64: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " expected=",
        expected,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def test_fp8_32x32x64_swap(ctx: DeviceContext) raises:
    print("--- FP8 32x32x64 swap_a_b ---")

    var dev_dump = ctx.enqueue_create_buffer[.float32](C_DUMP_SIZE)
    var host_dump = ctx.enqueue_create_host_buffer[.float32](C_DUMP_SIZE)
    with dev_dump.map_to_host() as init_h:
        for i in range(C_DUMP_SIZE):
            init_h[i] = Float32(0)

    ctx.enqueue_function[kernel_fp8_32x32x64_swap](
        dev_dump, grid_dim=1, block_dim=64
    )
    ctx.enqueue_copy(host_dump, dev_dump)
    ctx.synchronize()

    var expected: Float32 = 36.0
    var tol: Float32 = 1.0
    var mismatches = 0
    var max_diff: Float32 = 0
    for lane in range(64):
        for p in range(C_FRAG):
            var got = host_dump[lane * C_FRAG + p]
            var diff = abs(got - expected)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "  FP8 swap: MISMATCH lane=",
                        lane,
                        " p=",
                        p,
                        " got=",
                        got,
                        " expected=",
                        expected,
                    )
    print(
        "  FP8 32x32x64 swap: mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " expected=",
        expected,
    )
    assert_equal(mismatches, 0)
    _ = dev_dump^
    _ = host_dump^
    print("  PASSED")


def main() raises:
    print("=" * 70)
    print("MFMA fragment lane-mapping isolation test")
    print("=" * 70)

    with DeviceContext() as ctx:
        # Hard gate: BF16 uniform must pass to validate test infrastructure.
        test_bf16_32x32x16(ctx)
        # BF16 non-uniform: validates per-lane row/col fragment decomposition.
        test_bf16_32x32x16_col(ctx)
        test_bf16_32x32x16_row(ctx)
        # BF16 swap_a_b semantic check.
        test_bf16_32x32x16_swap(ctx)
        # Key datapoints: FP8 32x32x64.
        test_fp8_32x32x64(ctx)
        test_fp8_32x32x64_col(ctx)
        test_fp8_32x32x64_row(ctx)
        test_fp8_32x32x64_swap(ctx)

    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
