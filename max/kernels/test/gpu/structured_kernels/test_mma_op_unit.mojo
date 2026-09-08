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
"""Isolated component tests for `MhaMmaOp.mma_QK` and `MhaMmaOp.mma_PV`.

This file targets the two MFMA dispatch helpers without exercising any
of the surrounding cluster body, SMEM loaders, or softmax. The
register tiles are populated DIRECTLY per-lane using the documented
MFMA A/B operand layouts (no LDS round trip, no swizzle, no
ds_read_tr16/tr8). That keeps a regression in `mma_QK` / `mma_PV`
from being masked by a loader bug — the loaders are tested separately
in `test_mha_mma_op_fp8.mojo`.

Coverage:

- `test_mma_QK[BF16]` — `v_mfma_f32_32x32x16_bf16`, MMA_K=16, 8
  elts/lane.
- `test_mma_QK[FP8]`  — `v_mfma_scale_f32_32x32x64_f8f6f4`, MMA_K=64,
  32 elts/lane.
- `test_mma_PV[BF16]` — same MFMA shape, V is A (depth-rows × key-
  cols), P is B (key-rows × q-cols). Output is `o[depth, q]`.
- `test_mma_PV[FP8]`  — FP8 variant of the same.

Each test:

1. Allocates the register tiles via `tt_stack_allocation`.
2. Fills K/Q (or V/P) by computing per-lane (gr, gc) → input-matrix
   entry, using a known 2D pattern over `(KV_BLOCK, DEPTH)` /
   `(Q_BLOCK_SIZE, DEPTH)` / etc.
3. Zeros the accumulator.
4. Calls `MhaMmaOp.mma_QK` / `mma_PV`.
5. Dumps the per-lane accumulator fragment to a gmem buffer.
6. On host, recomputes the same product analytically using the
   col_l rt_32x32 output layout (lane → (row, col)) and compares to
   the dumped values.

Lane geometry references (verified against `mha_mma_op.mojo`):

- MMA_M = MMA_N = 32 for both BF16 and FP8 32×32×{16,64}.
- A operand fragment: lane `lid` owns A[`lid % 32`, `(lid // 32) *
  ROWL_STRIDE + f`] for `f in [0, FRAG_ELTS)`. ROWL_STRIDE = MMA_K //
  2.
- B operand fragment (in B-position, no swap): lane `lid` owns
  B[`(lid // 32) * ROWL_STRIDE + f`, `lid % 32`]. Same partition
  shape as A but with row/col swapped.
- C/D accumulator fragment (col_l rt_32x32 FP32): per base tile,
  lane `lid` holds 16 FP32 values at:
    `row_in_tile = ACC_ROW_OFFSETS_32x32[k_local] + (lid >> 5) * 4`,
    `col_in_tile = lid & 31`,
  with `k_local in [0, 16)`.

Tolerances (max element-wise abs-diff over the whole accumulator):

- BF16: 1e-2 (the input dtype's ULP at value ~1 is ~7.8e-3, so a
  contracting-axis sum of ~16 BF16 products lands within ~1e-2).
- FP8 e4m3: 5e-2 (input ULP at value ~1 is ~6.25e-2; pattern is
  scaled small so the sum of 64 products stays representable).
"""

from max.gpu import lane_id, thread_idx
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.testing import assert_true

from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation

from nn.attention.gpu.amd_structured.mha_mma_op import (
    ACC_ROW_OFFSETS_32x32,
    MhaConfigV2,
    MhaMmaOp,
)


# --------------------------------------------------------------------------- #
# Tile dimensions used by all four tests. Single (KV_BLOCK, Q_BLOCK_SIZE,
# DEPTH) triple keeps the per-lane indexing simple and the host references
# tractable. Q_BLOCK_SIZE=32 / DEPTH=64 fits in one register-tile column
# block for BF16 (DEPTH/MMA_K=4) and one for FP8 (DEPTH/MMA_K=1).
# --------------------------------------------------------------------------- #
comptime Q_BLOCK_SIZE = 32
comptime KV_BLOCK = 64
comptime DEPTH = 64


# --------------------------------------------------------------------------- #
# Patterns. Chosen so K @ Q^T and V @ P fit comfortably in FP32
# accumulator and within BF16 / FP8 precision for the inputs.
#
# Pattern K[gr, gc] = (gr + gc * 0.01)  — values in roughly [0, 100].
# Pattern Q[gr, gc] = (gr - gc * 0.01)  — values in roughly [-1, 32].
# Pattern V[k, m]   = ((k * 0.1 + m) / 32)
# Pattern P[k, n]   = ((k + n * 0.1) / 64)
#
# In all cases the product matrix stays in a range BF16 / FP8 can
# represent without saturation (peak ~16 for QK, ~1 for PV).
#
# Scaled even smaller for FP8 (which has only ~7-bit mantissa for
# values >1). Both BF16 and FP8 patterns are integer-valued-after-cast
# to keep the per-element error from cast quantization away from the
# product noise floor.
# --------------------------------------------------------------------------- #


@always_inline
def _pattern_K_bf16(gr: Int, gc: Int) -> BFloat16:
    return BFloat16(Float32(gr) + Float32(gc) * 0.01)


@always_inline
def _pattern_Q_bf16(gr: Int, gc: Int) -> BFloat16:
    return BFloat16(Float32(gr) - Float32(gc) * 0.01)


@always_inline
def _pattern_V_bf16(k: Int, m: Int) -> BFloat16:
    return BFloat16((Float32(k) * 0.1 + Float32(m)) / 32.0)


@always_inline
def _pattern_P_bf16(k: Int, n: Int) -> BFloat16:
    return BFloat16((Float32(k) + Float32(n) * 0.1) / 64.0)


@always_inline
def _pattern_K_fp8(gr: Int, gc: Int) -> Float8_e4m3fn:
    # FP8 e4m3: peak representable ~448. Keep inputs in [0, ~4).
    return Float8_e4m3fn((Float32(gr) + Float32(gc) * 0.01) / 32.0)


@always_inline
def _pattern_Q_fp8(gr: Int, gc: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn((Float32(gr) - Float32(gc) * 0.01) / 32.0)


@always_inline
def _pattern_V_fp8(k: Int, m: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn((Float32(k) * 0.1 + Float32(m)) / 256.0)


@always_inline
def _pattern_P_fp8(k: Int, n: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn((Float32(k) + Float32(n) * 0.1) / 512.0)


# --------------------------------------------------------------------------- #
# Kernel: test_mma_QK BF16. Each lane fills K_reg and Q_reg directly using
# the MFMA A and B operand fragment layouts (no LDS), then runs mma_QK,
# then dumps att_reg to gmem so the host can compare.
# --------------------------------------------------------------------------- #


def kernel_mma_QK[
    cfg: MhaConfigV2,
    T: DType,
](dump_ptr: MutPointer[Float32, MutAnyOrigin],):
    comptime _Op = MhaMmaOp[T, cfg]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _ROWL_STRIDE = _Op.ROWL_STRIDE
    comptime _FRAG_ELTS = _Op.FRAG_ELTS

    # K register tile: shape (KV_BLOCK / MMA_M, DEPTH / MMA_K, FRAG_ELTS).
    var k_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.K_LAYOUT)
    # Q register tile: shape (Q_BLOCK_SIZE / MMA_M, DEPTH / MMA_K, FRAG_ELTS).
    var q_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.Q_LAYOUT)
    # att register tile: shape (KV_BLOCK / MMA_M, Q_BLOCK_SIZE / MMA_N, 16).
    var att_reg = tt_stack_allocation[.float32, address_space=.LOCAL](
        _Op.ATT_LAYOUT
    )

    var lid = Int(lane_id())
    var row_offset = lid % 32  # 0..31
    var col_offset = _ROWL_STRIDE * (lid // 32)  # 0 or ROWL_STRIDE

    comptime _KH = _Op.K_LAYOUT.static_shape[0]
    comptime _KW = _Op.K_LAYOUT.static_shape[1]
    var k_v = k_reg.vectorize[1, 1, _FRAG_ELTS]()
    comptime for n in range(_KH):
        comptime for kk in range(_KW):
            var frag = SIMD[T, _FRAG_ELTS](0)
            comptime for f in range(_FRAG_ELTS):
                var gr = n * _MMA_M + row_offset
                var gc = kk * _MMA_K + col_offset + f
                comptime if T == .bfloat16:
                    frag[f] = rebind[Scalar[T]](_pattern_K_bf16(gr, gc))
                else:
                    frag[f] = rebind[Scalar[T]](_pattern_K_fp8(gr, gc))
            k_v[n, kk, 0] = rebind[k_v.ElementType](frag)

    comptime _QH = _Op.Q_LAYOUT.static_shape[0]
    comptime _QW = _Op.Q_LAYOUT.static_shape[1]
    var q_v = q_reg.vectorize[1, 1, _FRAG_ELTS]()
    comptime for m in range(_QH):
        comptime for kk in range(_QW):
            var frag = SIMD[T, _FRAG_ELTS](0)
            comptime for f in range(_FRAG_ELTS):
                var gr = m * _MMA_N + row_offset
                var gc = kk * _MMA_K + col_offset + f
                comptime if T == .bfloat16:
                    frag[f] = rebind[Scalar[T]](_pattern_Q_bf16(gr, gc))
                else:
                    frag[f] = rebind[Scalar[T]](_pattern_Q_fp8(gr, gc))
            q_v[m, kk, 0] = rebind[q_v.ElementType](frag)

    # Zero the accumulator.
    comptime _AH = _Op.ATT_LAYOUT.static_shape[0]
    comptime _AW = _Op.ATT_LAYOUT.static_shape[1]
    var att_v = att_reg.vectorize[1, 1, 16]()
    comptime for n in range(_AH):
        comptime for m in range(_AW):
            att_v[n, m, 0] = SIMD[.float32, 16](0.0)

    # The bulk of the test: call mma_QK.
    _Op.mma_QK(att_reg, k_reg, q_reg)

    # Dump per-lane accumulator. Layout: per-lane stride = _AH * _AW * 16.
    comptime _per_lane = _AH * _AW * 16
    comptime for n in range(_AH):
        comptime for m in range(_AW):
            var frag = att_v[n, m, 0]
            comptime for k_local in range(16):
                var idx = lid * _per_lane + (n * _AW + m) * 16 + k_local
                dump_ptr[idx] = frag[k_local]


def test_mma_QK[T: DType](ctx: DeviceContext) raises -> Bool:
    var dtype_name = "BF16" if T == DType.bfloat16 else "FP8"
    print("--- test_mma_QK[", dtype_name, "] ---")

    comptime CFG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=DEPTH,
        num_heads=1,
        num_kv_heads=1,
        dtype=T,
    )
    comptime _Op = MhaMmaOp[T, CFG]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _ROWL_STRIDE = _Op.ROWL_STRIDE
    comptime _FRAG_ELTS = _Op.FRAG_ELTS
    comptime _AH = _Op.ATT_LAYOUT.static_shape[0]
    comptime _AW = _Op.ATT_LAYOUT.static_shape[1]
    comptime _KH = _Op.K_LAYOUT.static_shape[0]
    comptime _KW = _Op.K_LAYOUT.static_shape[1]
    comptime _QH = _Op.Q_LAYOUT.static_shape[0]
    comptime _QW = _Op.Q_LAYOUT.static_shape[1]
    comptime _per_lane = _AH * _AW * 16
    comptime _DUMP_SIZE = 64 * _per_lane

    var dev_dump = ctx.enqueue_create_buffer[.float32](_DUMP_SIZE)

    ctx.enqueue_function[kernel_mma_QK[CFG, T]](
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    # Host reference. The K and Q register tiles each hold one BF16/FP8
    # value per (gr, gc) of the 2D matrix; mma_QK computes
    #   att[n_out * MMA_M + row, m_out * MMA_N + col] +=
    #     sum_kk sum_hw sum_f K_full[n_out * MMA_M + row,
    #                                kk * MMA_K + hw * ROWL_STRIDE + f]
    #                       * Q_full[m_out * MMA_N + col,
    #                                kk * MMA_K + hw * ROWL_STRIDE + f]
    # which over the contracting axis is just K_full @ Q_full^T.
    var mismatches: Int = 0
    var max_diff: Float32 = 0.0
    var pos_count: Int = 0
    var neg_count: Int = 0
    comptime _tol: Float32 = 5e-2 if T == .float8_e4m3fn else 1e-2

    with dev_dump.map_to_host() as host_dump:
        for lid in range(64):
            var col_in_tile = lid & 31
            for n_out in range(_AH):
                for m_out in range(_AW):
                    for k_local in range(16):
                        var row_in_tile = (
                            Int(ACC_ROW_OFFSETS_32x32[k_local]) + (lid >> 5) * 4
                        )
                        var gr_K = n_out * _MMA_M + row_in_tile
                        var gr_Q = m_out * _MMA_N + col_in_tile
                        var expected: Float32 = 0.0
                        for gc in range(DEPTH):
                            var k_val: Float32
                            var q_val: Float32
                            comptime if T == .bfloat16:
                                k_val = Float32(_pattern_K_bf16(gr_K, gc))
                                q_val = Float32(_pattern_Q_bf16(gr_Q, gc))
                            else:
                                k_val = _pattern_K_fp8(gr_K, gc).cast[
                                    DType.float32
                                ]()
                                q_val = _pattern_Q_fp8(gr_Q, gc).cast[
                                    DType.float32
                                ]()
                            expected += k_val * q_val
                        var idx = (
                            lid * _per_lane
                            + (n_out * _AW + m_out) * 16
                            + k_local
                        )
                        var got = host_dump[idx]
                        var diff = abs(got - expected)
                        if diff > max_diff:
                            max_diff = diff
                        if diff > _tol:
                            if got > expected:
                                pos_count += 1
                            else:
                                neg_count += 1
                            if mismatches < 5:
                                print(
                                    "  MISMATCH lid=",
                                    lid,
                                    " n=",
                                    n_out,
                                    " m=",
                                    m_out,
                                    " k_local=",
                                    k_local,
                                    " gr_K=",
                                    gr_K,
                                    " gr_Q=",
                                    gr_Q,
                                    " got=",
                                    got,
                                    " expected=",
                                    expected,
                                    " diff=",
                                    diff,
                                )
                            mismatches += 1

    print(
        "  mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " pos=",
        pos_count,
        " neg=",
        neg_count,
    )
    _ = dev_dump^
    if mismatches == 0:
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False


# --------------------------------------------------------------------------- #
# Kernel: test_mma_PV. Fills V_reg and P_reg per-lane using the A and B
# operand fragment layouts, calls mma_PV, dumps o_reg to gmem.
#
# V is A operand of mma_PV. V_LAYOUT shape is (KV_BLOCK / MMA_K,
# DEPTH / MMA_N, FRAG_ELTS). Per-lane, base tile (kk, n_depth), frag f:
#   V_elem[depth_idx, key_idx] where
#     depth_idx = n_depth * MMA_M + lid % 32
#     key_idx   = kk * MMA_K + (lid // 32) * ROWL_STRIDE + f
# (A operand's row is the M-side of the MFMA, which is the depth axis of
# the output.)
#
# P is B operand. P_LAYOUT shape is (KV_BLOCK / MMA_K, Q_BLOCK_SIZE /
# MMA_N, FRAG_ELTS). Per-lane, base tile (kk, m_q), frag f:
#   P_elem[key_idx, q_idx] where
#     key_idx = kk * MMA_K + (lid // 32) * ROWL_STRIDE + f
#     q_idx   = m_q * MMA_N + lid % 32
#
# Output o_LAYOUT shape is (DEPTH / MMA_M, Q_BLOCK_SIZE / MMA_N, 16) in
# col_l rt_32x32. Per-lane fragment indexing follows
# ACC_ROW_OFFSETS_32x32.
# --------------------------------------------------------------------------- #


def kernel_mma_PV[
    cfg: MhaConfigV2,
    T: DType,
](dump_ptr: MutPointer[Float32, MutAnyOrigin],):
    comptime _Op = MhaMmaOp[T, cfg]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _ROWL_STRIDE = _Op.ROWL_STRIDE
    comptime _FRAG_ELTS = _Op.FRAG_ELTS

    # P uses the ATT_BF16_SUB_LAYOUT (one strip) layout sub-shape: but
    # we don't need the subtile machinery in this isolated test — build
    # a P register tile that matches the same shape as V (kk-outer,
    # m-outer, frag) but the m-outer dimension is Q_BLOCK_SIZE/MMA_N.
    # The mma_PV signature lets P be any RegTile[T, layout_p] with
    # matching shapes; we use ATT_BF16_FULL_LAYOUT shape so the layout
    # exactly matches what the production kernel builds.
    var v_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.V_LAYOUT)
    var p_reg = tt_stack_allocation[T, address_space=.LOCAL](
        _Op.ATT_BF16_FULL_LAYOUT
    )
    var o_reg = tt_stack_allocation[.float32, address_space=.LOCAL](
        _Op.O_LAYOUT
    )

    var lid = Int(lane_id())
    var row_offset = lid % 32
    var col_offset = _ROWL_STRIDE * (lid // 32)

    # Fill V_reg using A-operand per-lane mapping.
    comptime _VH = _Op.V_LAYOUT.static_shape[0]
    comptime _VW = _Op.V_LAYOUT.static_shape[1]
    var v_v = v_reg.vectorize[1, 1, _FRAG_ELTS]()
    comptime for kk in range(_VH):
        comptime for n_depth in range(_VW):
            var frag = SIMD[T, _FRAG_ELTS](0)
            comptime for f in range(_FRAG_ELTS):
                var depth_idx = n_depth * _MMA_M + row_offset
                var key_idx = kk * _MMA_K + col_offset + f
                comptime if T == .bfloat16:
                    frag[f] = rebind[Scalar[T]](
                        _pattern_V_bf16(key_idx, depth_idx)
                    )
                else:
                    frag[f] = rebind[Scalar[T]](
                        _pattern_V_fp8(key_idx, depth_idx)
                    )
            v_v[kk, n_depth, 0] = rebind[v_v.ElementType](frag)

    # Fill P_reg using B-operand per-lane mapping.
    comptime _PH = _Op.ATT_BF16_FULL_LAYOUT.static_shape[0]
    comptime _PW = _Op.ATT_BF16_FULL_LAYOUT.static_shape[1]
    var p_v = p_reg.vectorize[1, 1, _FRAG_ELTS]()
    comptime for kk in range(_PH):
        comptime for m_q in range(_PW):
            var frag = SIMD[T, _FRAG_ELTS](0)
            comptime for f in range(_FRAG_ELTS):
                var key_idx = kk * _MMA_K + col_offset + f
                var q_idx = m_q * _MMA_N + row_offset
                comptime if T == .bfloat16:
                    frag[f] = rebind[Scalar[T]](_pattern_P_bf16(key_idx, q_idx))
                else:
                    frag[f] = rebind[Scalar[T]](_pattern_P_fp8(key_idx, q_idx))
            p_v[kk, m_q, 0] = rebind[p_v.ElementType](frag)

    # Zero o accumulator.
    comptime _OH = _Op.O_LAYOUT.static_shape[0]
    comptime _OW = _Op.O_LAYOUT.static_shape[1]
    var o_v = o_reg.vectorize[1, 1, 16]()
    comptime for n in range(_OH):
        comptime for m in range(_OW):
            o_v[n, m, 0] = SIMD[.float32, 16](0.0)

    # The bulk of the test: call mma_PV.
    _Op.mma_PV(o_reg, v_reg, p_reg)

    # Dump per-lane accumulator.
    comptime _per_lane = _OH * _OW * 16
    comptime for n in range(_OH):
        comptime for m in range(_OW):
            var frag = o_v[n, m, 0]
            comptime for k_local in range(16):
                var idx = lid * _per_lane + (n * _OW + m) * 16 + k_local
                dump_ptr[idx] = frag[k_local]


def test_mma_PV[T: DType](ctx: DeviceContext) raises -> Bool:
    var dtype_name = "BF16" if T == DType.bfloat16 else "FP8"
    print("--- test_mma_PV[", dtype_name, "] ---")

    comptime CFG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=DEPTH,
        num_heads=1,
        num_kv_heads=1,
        dtype=T,
    )
    comptime _Op = MhaMmaOp[T, CFG]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _OH = _Op.O_LAYOUT.static_shape[0]
    comptime _OW = _Op.O_LAYOUT.static_shape[1]
    comptime _per_lane = _OH * _OW * 16
    comptime _DUMP_SIZE = 64 * _per_lane

    var dev_dump = ctx.enqueue_create_buffer[.float32](_DUMP_SIZE)

    ctx.enqueue_function[kernel_mma_PV[CFG, T]](
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    # Host reference: o[depth, q] = sum_key V_full[depth, key] * P_full[key, q].
    var mismatches: Int = 0
    var max_diff: Float32 = 0.0
    var pos_count: Int = 0
    var neg_count: Int = 0
    comptime _tol: Float32 = 5e-2 if T == .float8_e4m3fn else 1e-2

    with dev_dump.map_to_host() as host_dump:
        for lid in range(64):
            var col_in_tile = lid & 31
            for n_depth in range(_OH):
                for m_q in range(_OW):
                    for k_local in range(16):
                        var row_in_tile = (
                            Int(ACC_ROW_OFFSETS_32x32[k_local]) + (lid >> 5) * 4
                        )
                        var depth_idx = n_depth * _MMA_M + row_in_tile
                        var q_idx = m_q * _MMA_N + col_in_tile
                        var expected: Float32 = 0.0
                        for key in range(KV_BLOCK):
                            var v_val: Float32
                            var p_val: Float32
                            comptime if T == .bfloat16:
                                v_val = Float32(_pattern_V_bf16(key, depth_idx))
                                p_val = Float32(_pattern_P_bf16(key, q_idx))
                            else:
                                v_val = _pattern_V_fp8(key, depth_idx).cast[
                                    DType.float32
                                ]()
                                p_val = _pattern_P_fp8(key, q_idx).cast[
                                    DType.float32
                                ]()
                            expected += v_val * p_val
                        var idx = (
                            lid * _per_lane
                            + (n_depth * _OW + m_q) * 16
                            + k_local
                        )
                        var got = host_dump[idx]
                        var diff = abs(got - expected)
                        if diff > max_diff:
                            max_diff = diff
                        if diff > _tol:
                            if got > expected:
                                pos_count += 1
                            else:
                                neg_count += 1
                            if mismatches < 5:
                                print(
                                    "  MISMATCH lid=",
                                    lid,
                                    " n_depth=",
                                    n_depth,
                                    " m_q=",
                                    m_q,
                                    " k_local=",
                                    k_local,
                                    " depth=",
                                    depth_idx,
                                    " q=",
                                    q_idx,
                                    " got=",
                                    got,
                                    " expected=",
                                    expected,
                                    " diff=",
                                    diff,
                                )
                            mismatches += 1

    print(
        "  mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " pos=",
        pos_count,
        " neg=",
        neg_count,
    )
    _ = dev_dump^
    if mismatches == 0:
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False


def main() raises:
    print("=" * 60)
    print("MhaMmaOp mma_QK / mma_PV unit tests")
    print("=" * 60)

    with DeviceContext() as ctx:
        var ok_qk_bf16 = test_mma_QK[.bfloat16](ctx)
        var ok_qk_fp8 = test_mma_QK[.float8_e4m3fn](ctx)
        var ok_pv_bf16 = test_mma_PV[.bfloat16](ctx)
        var ok_pv_fp8 = test_mma_PV[.float8_e4m3fn](ctx)

        print("=" * 60)
        print("Summary:")
        print(
            "  mma_QK[BF16] =",
            "PASS" if ok_qk_bf16 else "FAIL",
        )
        print(
            "  mma_QK[FP8 ] =",
            "PASS" if ok_qk_fp8 else "FAIL",
        )
        print(
            "  mma_PV[BF16] =",
            "PASS" if ok_pv_bf16 else "FAIL",
        )
        print(
            "  mma_PV[FP8 ] =",
            "PASS" if ok_pv_fp8 else "FAIL",
        )
        print("=" * 60)

        assert_true(
            ok_qk_bf16 and ok_qk_fp8 and ok_pv_bf16 and ok_pv_fp8,
            "one or more mma_QK / mma_PV tests failed",
        )
