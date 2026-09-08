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
"""Pure-FP32-reference test for `load_K + load_Q + mma_QK` chain.

Phase 14c diagnostic: verify the K loader + Q loader + `mma_QK` chain
produces a numerically-correct attention output against a host-computed
pure-FP32 reference, for BOTH `BF16` and `float8_e4m3fn`.

Critical: existing component tests in `test_mha_mma_op_fp8.mojo`,
`test_mfma_fragment_lane_mapping.mojo`, and `test_mma_op_unit.mojo`
PASS for FP8 — but they all validate the kernel's ASSUMED layout. If
the assumed layout were wrong, the kernel and those tests would still
agree (both reading from the same lane-mapping formula) while computing
wrong attention.

This test computes its host reference using only matrix-level operations
(no per-lane assumptions): `att[m_kv, n_q] = sum_d K[m_kv, d] * Q[n_q, d]`
in pure-FP32, then FP8-quantizes K and Q and recomputes the reference
to match what the FP8 MFMA actually contracts.

Test design:

1. Build K_fp32 and Q_fp32 matrices on host with smooth pattern:
   - K[m_kv, d] = (m_kv * 1.0 + d * 0.01) / 32.0 -> values in ~[0, 6]
   - Q[n_q, d] = (n_q * 0.5 - d * 0.02) / 32.0 -> values in ~[-0.08, 0.5]
2. Quantize K, Q to BF16 / FP8 e4m3fn.
3. Upload quantized K to SMEM in the swizzled layout the production K
   loader expects (same swizzle pattern as `_dma_k`). Q register tile
   is populated DIRECTLY per-lane using the documented `row_l rt_32xMMA_K`
   formula (mirrors `test_mma_op_unit.mojo`).
4. Call `MhaMmaOp.load_K`, populate `q_reg` per-lane, then
   `MhaMmaOp.mma_QK`. Dump per-lane FP32 accumulator to gmem.
5. Host: de-map per-lane (lid, n_out, m_out, k_local) -> (key, q_row)
   via `ACC_ROW_OFFSETS_32x32` (KNOWN CORRECT — verified by sister test
   `test_mfma_fragment_lane_mapping`), recompute reference using
   quantized values in pure FP32, compare.

Tolerance:
- BF16: max element-wise abs-diff <= 1e-2 (BF16 ULP ~7.8e-3 at ~1)
- FP8 e4m3fn: max element-wise abs-diff <= 0.05

If FP8 passes -> K/Q chain is correct, FP8 attention bug is downstream
(softmax cast / OnlineSoftmax FP32 path / etc).

If FP8 fails -> K loader / mma_QK chain is wrong. The mismatch pattern
((m, n) positions, sign of error) localizes which loader is at fault.
"""

from max.gpu import lane_id, thread_idx
from max.gpu.sync import barrier
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
# Matrix dimensions. KV_BLOCK=64 is the MHA kernel's default; Q_BLOCK_SIZE=32 is
# the per-warp Q slab. DEPTH=128 exercises 2 K-direction MFMA base tiles
# at FP8 (MMA_K=64) and 8 at BF16 (MMA_K=16).
# --------------------------------------------------------------------------- #
comptime Q_BLOCK_SIZE = 32
comptime KV_BLOCK = 64
comptime DEPTH = 128


# --------------------------------------------------------------------------- #
# Pattern: smooth, value-range chosen so the contracting-axis sum
# remains representable in FP8 e4m3fn without saturation. Peak K @ Q^T
# magnitude ~ DEPTH * 6 * 0.5 / 32 ~= 12.
# --------------------------------------------------------------------------- #


@always_inline
def _K_fp32(m_kv: Int, d: Int) -> Float32:
    # Distinct (m_kv, d) coefficients so a swap or off-by-one in the loader
    # produces wrong values. Range ~[-0.25, 1.5] -> FP8 e4m3fn representable
    # with ~5-6 bits of fractional precision, so the per-element
    # quantization error is non-zero (test isn't degenerate).
    return (Float32(m_kv) * 0.1 + Float32(d) * 0.05 - 0.3) / 16.0


@always_inline
def _Q_fp32(n_q: Int, d: Int) -> Float32:
    # Different coefficients so K and Q can't trivially "swap" without
    # detection. Negative-tilt vs K so the product range spans sign.
    return (Float32(n_q) * 0.07 - Float32(d) * 0.04 + 0.5) / 16.0


# --------------------------------------------------------------------------- #
# Kernel: SMEM-K + per-lane-Q + mma_QK + dump.
#
# K is loaded via the PRODUCTION path: SMEM has been filled by the host
# with the K matrix pre-swizzled (matching what `_dma_k` produces in the
# real kernel), `MhaMmaOp.load_K` unswizzles into the K register tile.
#
# Q is populated DIRECTLY per-lane using the row_l rt_32xMMA_K formula
# (mirroring how `RegTileLoader` distributes a 2D Q tile across the warp).
# We use direct fill rather than `RegTileLoader` here so the test stays
# self-contained (no TileTensor + buffer-resource plumbing) and isolates
# the K loader. A failure here pins the bug to load_K or mma_QK.
# --------------------------------------------------------------------------- #


def kernel_qk_chain[
    cfg: MhaConfigV2,
    T: DType,
](
    src_k_swz_ptr: MutPointer[Scalar[T], MutAnyOrigin],
    src_q_ptr: MutPointer[Scalar[T], MutAnyOrigin],
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Loads K from pre-swizzled SMEM, fills Q per-lane from gmem, calls
    `mma_QK`, dumps per-lane FP32 accumulator to `dump_ptr`.

    SMEM K layout: `row_major[KV_BLOCK * (DEPTH / K_SUB_COLS), K_SUB_COLS]`
    with each `(K_SUB_ROWS x K_SUB_COLS)` sub-block holding the K rows
    `[sub_row * K_SUB_ROWS, (sub_row+1) * K_SUB_ROWS)` x cols
    `[sub_col * K_SUB_COLS, (sub_col+1) * K_SUB_COLS)` of the logical K
    matrix, byte-positionally swizzled via `_swizzle_K_sub`.

    Q gmem layout: contiguous `(Q_BLOCK_SIZE, DEPTH)` row-major.
    """
    comptime _Op = MhaMmaOp[T, cfg]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _ROWL_STRIDE = _Op.ROWL_STRIDE
    comptime _FRAG_ELTS = _Op.FRAG_ELTS
    comptime _K_SUB_COLS = _Op.K_SUB_COLS
    comptime _NUM_BLOCK_COLS_K = cfg.depth // _K_SUB_COLS
    comptime _K_SLOT_ROWS = cfg.kv_block * _NUM_BLOCK_COLS_K
    comptime smem_layout_k = row_major[_K_SLOT_ROWS, _K_SUB_COLS]()

    # ---- K SMEM allocation + cooperative fill from gmem image -------------
    var k_smem = tt_stack_allocation[T, address_space=.SHARED](smem_layout_k)
    var tid = Int(thread_idx.x)
    comptime _smem_total = _K_SLOT_ROWS * _K_SUB_COLS
    var i = tid
    while i < _smem_total:
        k_smem._storage[i] = src_k_swz_ptr[i]
        i += 64
    barrier()

    # ---- K register tile + load_K -----------------------------------------
    var k_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.K_LAYOUT)
    _Op.load_K(k_reg, k_smem)

    # ---- Q register tile + per-lane fill from gmem ------------------------
    # row_l rt_32xMMA_K lane formula (mirror of how RegTileLoader fills
    # `q_reg` from a `(Q_BLOCK_SIZE, DEPTH)` 2D gmem tile via
    # `col_major[Q_BLOCK_SIZE, WARP_SIZE / Q_BLOCK_SIZE]` thread layout):
    #     Q_reg[register_row, register_col].frag[f]
    #         = Q[register_row * MMA_M + (lid % 32),
    #             register_col * MMA_K + ROWL_STRIDE * (lid // 32) + f]
    var q_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.Q_LAYOUT)
    var lid = Int(lane_id())
    var row_offset = lid % 32
    var col_offset = _ROWL_STRIDE * (lid // 32)
    comptime _QH = _Op.Q_LAYOUT.static_shape[0]
    comptime _QW = _Op.Q_LAYOUT.static_shape[1]
    var q_v = q_reg.vectorize[1, 1, _FRAG_ELTS]()
    comptime for m in range(_QH):
        comptime for kk in range(_QW):
            var frag = SIMD[T, _FRAG_ELTS](0)
            comptime for f in range(_FRAG_ELTS):
                var gr = m * _MMA_M + row_offset
                var gc = kk * _MMA_K + col_offset + f
                var src_idx = gr * cfg.depth + gc
                frag[f] = src_q_ptr[src_idx]
            q_v[m, kk, 0] = rebind[q_v.ElementType](frag)

    # ---- Accumulator + mma_QK ---------------------------------------------
    var att_reg = tt_stack_allocation[.float32, address_space=.LOCAL](
        _Op.ATT_LAYOUT
    )
    comptime _AH = _Op.ATT_LAYOUT.static_shape[0]
    comptime _AW = _Op.ATT_LAYOUT.static_shape[1]
    var att_v = att_reg.vectorize[1, 1, 16]()
    comptime for n in range(_AH):
        comptime for m in range(_AW):
            att_v[n, m, 0] = SIMD[.float32, 16](0.0)

    _Op.mma_QK(att_reg, k_reg, q_reg)

    # ---- Dump per-lane accumulator ----------------------------------------
    comptime _per_lane = _AH * _AW * 16
    comptime for n in range(_AH):
        comptime for m in range(_AW):
            var frag = att_v[n, m, 0]
            comptime for k_local in range(16):
                var idx = lid * _per_lane + (n * _AW + m) * 16 + k_local
                dump_ptr[idx] = frag[k_local]


# --------------------------------------------------------------------------- #
# Host helper: cast Float32 -> dtype T. BF16 / FP8 e4m3fn paths share the
# Mojo standard cast semantics (round-to-nearest-even, saturate to dtype
# limits). Centralized so the test path matches whatever the production
# kernel sees.
# --------------------------------------------------------------------------- #


@always_inline
def _quantize_to_K[T: DType](v: Float32) -> Scalar[T]:
    comptime if T == .bfloat16:
        return rebind[Scalar[T]](BFloat16(v))
    else:
        return rebind[Scalar[T]](Float8_e4m3fn(v))


@always_inline
def _dequantize_to_f32[T: DType](v: Scalar[T]) -> Float32:
    return v.cast[.float32]()


# --------------------------------------------------------------------------- #
# Test driver: runs the chain for one dtype, compares to host FP32 reference.
# --------------------------------------------------------------------------- #


def test_qk_chain[T: DType](ctx: DeviceContext) raises -> Bool:
    var dtype_name = "BF16" if T == DType.bfloat16 else "FP8"
    print("--- test_qk_chain[", dtype_name, "] ---")

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
    comptime _K_SUB_ROWS = _Op.K_SUB_ROWS
    comptime _K_SUB_COLS = _Op.K_SUB_COLS
    comptime _NUM_BLOCK_COLS_K = DEPTH // _K_SUB_COLS
    comptime _K_SLOT_ROWS = KV_BLOCK * _NUM_BLOCK_COLS_K
    comptime _SMEM_SIZE = _K_SLOT_ROWS * _K_SUB_COLS  # T-elements
    comptime _Q_SIZE = Q_BLOCK_SIZE * DEPTH
    comptime _AH = _Op.ATT_LAYOUT.static_shape[0]
    comptime _AW = _Op.ATT_LAYOUT.static_shape[1]
    comptime _per_lane = _AH * _AW * 16
    comptime _DUMP_SIZE = 64 * _per_lane
    comptime _tol: Float32 = 5e-2 if T == .float8_e4m3fn else 1e-2

    var dev_k_swz = ctx.enqueue_create_buffer[T](_SMEM_SIZE)
    var dev_q = ctx.enqueue_create_buffer[T](_Q_SIZE)
    var dev_dump = ctx.enqueue_create_buffer[.float32](_DUMP_SIZE)

    # ---- Build the pre-swizzled K SMEM image on host ----------------------
    # For each logical (m_kv, d), quantize K_fp32 to T, then write it into
    # the K SMEM slot at the swizzled byte offset.
    with dev_k_swz.map_to_host() as host_k:
        for i in range(_SMEM_SIZE):
            host_k[i] = _quantize_to_K[T](Float32(0.0))

        comptime _T_size = 1 if T == DType.float8_e4m3fn else 2  # bytes/elt
        var sub_elts = _K_SUB_ROWS * _K_SUB_COLS  # elements per sub-block
        for m_kv in range(KV_BLOCK):
            for d in range(DEPTH):
                var sub_row = m_kv // _K_SUB_ROWS
                var sub_col = d // _K_SUB_COLS
                var sub_id = sub_row * _NUM_BLOCK_COLS_K + sub_col
                var sr = m_kv % _K_SUB_ROWS
                var sc = d % _K_SUB_COLS
                var swz_bytes = _Op._swizzle_K_sub(sr, sc)
                # swz_bytes is byte-positional; convert to element index
                # within this sub-block.
                var swz_elts = swz_bytes // _T_size
                var slot_idx = sub_id * sub_elts + swz_elts
                host_k[slot_idx] = _quantize_to_K[T](_K_fp32(m_kv, d))

    # ---- Build the contiguous Q gmem image on host ------------------------
    with dev_q.map_to_host() as host_q:
        for n_q in range(Q_BLOCK_SIZE):
            for d in range(DEPTH):
                host_q[n_q * DEPTH + d] = _quantize_to_K[T](_Q_fp32(n_q, d))

    # ---- Launch the kernel ------------------------------------------------
    ctx.enqueue_function[kernel_qk_chain[CFG, T]](
        dev_k_swz.unsafe_ptr(),
        dev_q.unsafe_ptr(),
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    # ---- Host reference computation --------------------------------------- #
    # Build the FP8/BF16-quantized K and Q matrices in host arrays for the
    # reference computation. Each matrix is `(rows, DEPTH)` flattened. We
    # back the host arrays with DeviceContext buffers so we don't need
    # manual pointer allocation plumbing (the test never launches a
    # kernel against these buffers).
    var dev_k_quant = ctx.enqueue_create_buffer[.float32](KV_BLOCK * DEPTH)
    var dev_q_quant = ctx.enqueue_create_buffer[.float32](Q_BLOCK_SIZE * DEPTH)
    var dev_att_ref = ctx.enqueue_create_buffer[.float32](
        KV_BLOCK * Q_BLOCK_SIZE
    )

    # ---- Compare dumped per-lane accumulator to reference -----------------
    # De-map: per-lane fragment idx (lid, n_out, m_out, k_local) maps to
    # global (m_kv, n_q) via:
    #     m_kv = n_out * MMA_M + ACC_ROW_OFFSETS_32x32[k_local] + (lid>>5)*4
    #     n_q  = m_out * MMA_N + (lid & 31)
    # `ACC_ROW_OFFSETS_32x32` is hardware-determined and verified by
    # `test_mfma_fragment_lane_mapping` — known correct independent of
    # any FP8 / loader assumption.
    var mismatches: Int = 0
    var max_diff: Float32 = 0.0
    var sumsq_err: Float32 = 0.0
    var sumsq_ref: Float32 = 0.0
    var dot_got_ref: Float32 = 0.0
    var sumsq_got: Float32 = 0.0
    var worst_m: Int = 0
    var worst_n: Int = 0

    with dev_k_quant.map_to_host() as k_quant, dev_q_quant.map_to_host() as q_quant, dev_att_ref.map_to_host() as att_ref, dev_dump.map_to_host() as host_dump:
        # Quantize K and Q to T then dequant back to FP32 -> bit-exact
        # mirror of what the kernel sees, computed in pure FP32.
        for m_kv in range(KV_BLOCK):
            for d in range(DEPTH):
                k_quant[m_kv * DEPTH + d] = _dequantize_to_f32[T](
                    _quantize_to_K[T](_K_fp32(m_kv, d))
                )
        for n_q in range(Q_BLOCK_SIZE):
            for d in range(DEPTH):
                q_quant[n_q * DEPTH + d] = _dequantize_to_f32[T](
                    _quantize_to_K[T](_Q_fp32(n_q, d))
                )

        # Pure-FP32 reference:
        #   att_ref[m_kv, n_q] = sum_d K_q[m_kv, d] * Q_q[n_q, d]
        # Computed sequentially in FP32 — bypasses ALL lane-mapping
        # assumptions. No SIMD, no per-lane offsets — just matrix multiply
        # at the (m, n) level.
        for m_kv in range(KV_BLOCK):
            for n_q in range(Q_BLOCK_SIZE):
                var acc: Float32 = 0.0
                for d in range(DEPTH):
                    acc += k_quant[m_kv * DEPTH + d] * q_quant[n_q * DEPTH + d]
                att_ref[m_kv * Q_BLOCK_SIZE + n_q] = acc

        # De-map per-lane fragment to (m_kv, n_q) and compare.
        for lid in range(64):
            var col_in_tile = lid & 31
            for n_out in range(_AH):
                for m_out in range(_AW):
                    for k_local in range(16):
                        var row_in_tile = (
                            Int(ACC_ROW_OFFSETS_32x32[k_local]) + (lid >> 5) * 4
                        )
                        var m_kv = n_out * _MMA_M + row_in_tile
                        var n_q = m_out * _MMA_N + col_in_tile
                        var idx = (
                            lid * _per_lane
                            + (n_out * _AW + m_out) * 16
                            + k_local
                        )
                        var got = host_dump[idx]
                        var expected = att_ref[m_kv * Q_BLOCK_SIZE + n_q]
                        var diff = abs(got - expected)
                        if diff > max_diff:
                            max_diff = diff
                            worst_m = m_kv
                            worst_n = n_q
                        sumsq_err += diff * diff
                        sumsq_ref += expected * expected
                        sumsq_got += got * got
                        dot_got_ref += got * expected
                        if diff > _tol:
                            if mismatches < 8:
                                print(
                                    "  MISMATCH lid=",
                                    lid,
                                    " n_out=",
                                    n_out,
                                    " m_out=",
                                    m_out,
                                    " k_local=",
                                    k_local,
                                    " m_kv=",
                                    m_kv,
                                    " n_q=",
                                    n_q,
                                    " got=",
                                    got,
                                    " expected=",
                                    expected,
                                    " diff=",
                                    diff,
                                )
                            mismatches += 1

    var cos_sim: Float32 = 0.0
    var denom = (sumsq_got * sumsq_ref) ** Float32(0.5)
    if denom > Float32(0.0):
        cos_sim = dot_got_ref / denom

    print(
        "  mismatches=",
        mismatches,
        " max_diff=",
        max_diff,
        " cos_sim=",
        cos_sim,
        " worst@(m=",
        worst_m,
        " n=",
        worst_n,
        ")",
    )

    _ = dev_k_swz^
    _ = dev_q^
    _ = dev_dump^
    _ = dev_k_quant^
    _ = dev_q_quant^
    _ = dev_att_ref^

    if mismatches == 0:
        print("  PASSED")
        return True
    else:
        print("  FAILED (", mismatches, " elements exceeded tol=", _tol, ")")
        return False


# --------------------------------------------------------------------------- #
# Main: runs the chain test for BF16 and FP8.
# --------------------------------------------------------------------------- #


def main() raises:
    print("=" * 60)
    print("Phase 14c: load_K + Q + mma_QK chain pure-FP32 reference test")
    print("=" * 60)

    with DeviceContext() as ctx:
        var ok_bf16 = test_qk_chain[.bfloat16](ctx)
        var ok_fp8 = test_qk_chain[.float8_e4m3fn](ctx)

        print("=" * 60)
        print("Summary:")
        print("  BF16 = ", "PASS" if ok_bf16 else "FAIL")
        print("  FP8  = ", "PASS" if ok_fp8 else "FAIL")
        print("=" * 60)

        assert_true(ok_bf16, "BF16 QK chain test failed")
        assert_true(
            ok_fp8,
            (
                "FP8 QK chain test failed — bug is in K loader, Q load formula"
                " (row_l rt_32xMMA_K), or mma_QK FP8 dispatch"
            ),
        )
