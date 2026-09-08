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
"""Pure-FP32-reference test for `load_V + mma_PV` chain (Phase 14e).

This is the symmetric counterpart of `test_qk_chain_fp32_ref.mojo` for
the PV side. Phase 14e probe: verify the V loader + `mma_PV` chain
produces a numerically-correct output against a host-computed pure-FP32
reference, for BOTH `BF16` and `float8_e4m3fn`.

Critical methodological point (matches the QK chain test): the existing
`test_mha_mma_op_fp8.mojo::test_load_V_fp8` fills V SMEM in a
contiguous (BN × BK) depth-block layout, then verifies `load_V` reads
that same layout consistently — i.e. it validates the loader's
assumption against itself. The production DMA producer
`SubTileLoaderLDS_st_8x32` writes V SMEM in a **sub-tile-major** byte
layout: each 8 × V_SUB_COLS sub-tile is written contiguously, with
sub-tiles ordered by row-direction first then column-direction. For
DEPTH > V_SUB_COLS the two layouts diverge.

This test fills V SMEM using the **DMA producer's** sub-tile-major byte
layout (mirroring the production DMA exactly), then runs `load_V`,
`mma_PV`, and a pure-FP32 host reference computed at the (key, depth)
matrix level — bypassing every per-lane assumption.

Test design:

1. Build V_fp32 (KV_BLOCK × DEPTH) and P_fp32 (KV_BLOCK × Q_BLOCK_SIZE)
   matrices on host with smooth, non-trivial patterns.
2. Quantize V and P to BF16 / FP8.
3. Upload V to SMEM in sub-tile-major byte order (matches the
   production DMA producer `SubTileLoaderLDS_st_8x32`).
4. Populate P register tile (PV-B operand) per-lane using the
   documented FP8/BF16 B-operand lane mapping.
5. Call `MhaMmaOp.load_V`, then `MhaMmaOp.mma_PV`. Dump per-lane FP32
   accumulator to gmem.
6. Host reference: `ref[depth, q] = sum_key V_q[key, depth] * P_q[key, q]`
   in pure FP32. De-map per-lane (n_out, m_out, k_local) to (depth, q)
   via `ACC_ROW_OFFSETS_32x32`. Compare.

Tolerance:
- BF16: max abs-diff <= 5e-2 (BF16 ULP at ~10 ~= 7.8e-3, accumulated
  over 64 keys this can reach ~5e-2 in worst case).
- FP8 e4m3fn: max abs-diff <= 0.5 (FP8 quantization noise dominates;
  cos_sim is the load-bearing metric).

Diagnosis on failure:
- If FP8 FAILS but BF16 PASSES -> bug is in `load_V` FP8 path's
  per-lane (key, depth) -> SMEM byte computation OR in `mma_PV` FP8
  dispatch. The (key, depth) pattern of the worst-mismatched output
  cells localizes which lanes carry wrong fragments.
- If both pass -> `load_V` + `mma_PV` chain is correct against the
  actual DMA byte layout; the FP8 attention bug is downstream
  (OnlineSoftmax FP32 path / FP8 P-cast interaction).
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
# the per-warp Q slab. DEPTH=128 exercises 4 N-direction MFMA base tiles
# at FP8 (MMA_N=32) and 2 sub-tile columns in the V SMEM slab.
# --------------------------------------------------------------------------- #
comptime Q_BLOCK_SIZE = 32
comptime KV_BLOCK = 64
comptime DEPTH = 128


# --------------------------------------------------------------------------- #
# Patterns. Values chosen so:
#   * FP8 e4m3fn can represent both with non-zero quantization error
#     (test isn't degenerate), but no saturation.
#   * Each (key, depth) and (key, q) cell is unique so a swap or off-by-
#     one in the loader/lane-formula produces wrong values.
#   * The contracted output sum_key V[key, depth] * P[key, q] doesn't
#     overflow FP8 e4m3fn's max (~448) — V ~ [-0.25, 1.5], P ~ [0, 0.1]
#     (post-softmax-style: small probabilities), so peak sum < 64 * 1.5
#     * 0.1 = 9.6.
# --------------------------------------------------------------------------- #


@always_inline
def _V_fp32(key: Int, depth: Int) -> Float32:
    # Smooth, distinct per (key, depth). Range ~[-0.3, 1.4].
    return (Float32(key) * 0.05 + Float32(depth) * 0.03 - 0.3) / 16.0


@always_inline
def _P_fp32(key: Int, q: Int) -> Float32:
    # Post-softmax-like: small positives. Range ~[0.005, 0.1].
    return (Float32(key) * 0.001 + Float32(q) * 0.003 + 0.005) / 1.0


# --------------------------------------------------------------------------- #
# Host helpers: type-generic quantize/dequant. BF16 and FP8 e4m3fn share
# Mojo's standard `.cast` semantics (round-to-nearest-even, saturate to
# dtype limits) — exactly what the kernel sees.
# --------------------------------------------------------------------------- #


@always_inline
def _quantize[T: DType](v: Float32) -> Scalar[T]:
    comptime if T == .bfloat16:
        return rebind[Scalar[T]](BFloat16(v))
    else:
        return rebind[Scalar[T]](Float8_e4m3fn(v))


@always_inline
def _dequant_to_f32[T: DType](v: Scalar[T]) -> Float32:
    return v.cast[.float32]()


# --------------------------------------------------------------------------- #
# Kernel: SMEM-V (sub-tile-major) + per-lane-P + load_V + mma_PV + dump.
#
# V SMEM is pre-filled by the host with the sub-tile-major byte layout
# the production DMA producer `SubTileLoaderLDS_st_8x32` writes:
#   sub-tile (V_SUB_ROWS x V_SUB_COLS) at (sub_row, sub_col) lives in
#   contiguous byte range [sub_id * sub_bytes, (sub_id+1) * sub_bytes),
#   where sub_id = sub_row * subtiles_per_row + sub_col.
#
# P register tile (PV-B operand) is filled DIRECTLY per-lane using the
# documented B-operand lane mapping for the MFMA shape:
#   - BF16 32x32x16: n_in_tile = lane % 32, k_in_tile = (lane // 32) * 8
#                    + elt, for elt in [0, FRAG_ELTS=8).
#   - FP8 32x32x64:  n_in_tile = lane % 32, k_in_tile = (lane // 32) *
#                    32 + elt, for elt in [0, FRAG_ELTS=32).
# This mirrors `_fp8_b_k_for` / `_bf16_b_k_for` in
# `test_mfma_fragment_lane_mapping.mojo` — a sister test that verifies
# this B-side mapping is the hardware-correct convention for both
# shapes.
# --------------------------------------------------------------------------- #


def kernel_pv_chain[
    cfg: MhaConfigV2,
    T: DType,
](
    src_v_subtile_ptr: MutPointer[Scalar[T], MutAnyOrigin],
    src_p_ptr: MutPointer[Scalar[T], MutAnyOrigin],
    dump_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Loads V from sub-tile-major SMEM, fills P per-lane from gmem,
    calls `mma_PV`, dumps per-lane FP32 accumulator to `dump_ptr`.

    SMEM V layout: `row_major[KV_BLOCK * (DEPTH / V_SUB_COLS), V_SUB_COLS]`
    with each `(V_SUB_ROWS x V_SUB_COLS)` sub-block holding a contiguous
    byte range produced by the DMA producer.

    P gmem layout: contiguous `(KV_BLOCK, Q_BLOCK_SIZE)` row-major.
    """
    comptime _Op = MhaMmaOp[T, cfg]
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_N = _Op.MMA_N
    comptime _MMA_K = _Op.MMA_K
    comptime _FRAG_ELTS = _Op.FRAG_ELTS
    comptime _V_SUB_COLS = _Op.V_SUB_COLS
    comptime _NUM_BLOCK_COLS_V = cfg.depth // _V_SUB_COLS
    comptime _V_SLOT_ROWS = cfg.kv_block * _NUM_BLOCK_COLS_V
    comptime smem_layout_v = row_major[_V_SLOT_ROWS, _V_SUB_COLS]()

    # ---- V SMEM allocation + cooperative fill from gmem image -------------
    var v_smem = tt_stack_allocation[T, address_space=.SHARED](smem_layout_v)
    var tid = Int(thread_idx.x)
    comptime _smem_total = _V_SLOT_ROWS * _V_SUB_COLS
    var i = tid
    while i < _smem_total:
        v_smem._storage[i] = src_v_subtile_ptr[i]
        i += 64
    barrier()

    # ---- V register tile + load_V (the unit under test) -------------------
    var v_reg = tt_stack_allocation[T, address_space=.LOCAL](_Op.V_LAYOUT)
    _Op.load_V(v_reg, v_smem)

    # ---- P register tile + per-lane fill from gmem ------------------------
    # B-operand lane mapping (B-side of `mma_PV` = `gpu_mma(d, v, p, d)`):
    #   For each subtile (i_k, j_n) in (KV_BLOCK/MMA_K, Q_BLOCK/MMA_N):
    #     P_reg[i_k, j_n].frag[elt] = P[
    #         key = i_k * MMA_K + (lid // 32) * (MMA_K // 2) + elt,
    #         q   = j_n * MMA_N + (lid % 32),
    #     ]
    # where MMA_K // 2 == FRAG_ELTS (FP8: 32, BF16: 8 — the half-warp
    # stride). Validated by `test_mfma_fragment_lane_mapping`.
    var p_reg = tt_stack_allocation[T, address_space=.LOCAL](
        _Op.ATT_BF16_FULL_LAYOUT
    )
    var lid = Int(lane_id())
    var q_lane = lid % 32
    var key_half = (lid // 32) * (_MMA_K // 2)
    comptime _PH = _Op.ATT_BF16_FULL_LAYOUT.static_shape[0]
    comptime _PW = _Op.ATT_BF16_FULL_LAYOUT.static_shape[1]
    comptime _PV_A_FRAG = _Op.PV_A_FRAG_ELTS
    var p_v = p_reg.vectorize[1, 1, _PV_A_FRAG]()
    comptime for i_k in range(_PH):
        comptime for j_n in range(_PW):
            var frag = SIMD[T, _PV_A_FRAG](0)
            comptime for f in range(_PV_A_FRAG):
                var key = i_k * _MMA_K + key_half + f
                var q = j_n * _MMA_N + q_lane
                var src_idx = key * cfg.q_block_size + q
                frag[f] = src_p_ptr[src_idx]
            p_v[i_k, j_n, 0] = rebind[p_v.ElementType](frag)

    # ---- Accumulator + mma_PV ---------------------------------------------
    var o_reg = tt_stack_allocation[.float32, address_space=.LOCAL](
        _Op.O_LAYOUT
    )
    comptime _OH = _Op.O_LAYOUT.static_shape[0]
    comptime _OW = _Op.O_LAYOUT.static_shape[1]
    var o_v = o_reg.vectorize[1, 1, 16]()
    comptime for n in range(_OH):
        comptime for m in range(_OW):
            o_v[n, m, 0] = SIMD[.float32, 16](0.0)

    _Op.mma_PV(o_reg, v_reg, p_reg)

    # ---- Dump per-lane accumulator ----------------------------------------
    comptime _per_lane = _OH * _OW * 16
    comptime for n in range(_OH):
        comptime for m in range(_OW):
            var frag = o_v[n, m, 0]
            comptime for k_local in range(16):
                var idx = lid * _per_lane + (n * _OW + m) * 16 + k_local
                dump_ptr[idx] = frag[k_local]


# --------------------------------------------------------------------------- #
# Test driver: runs the chain for one dtype, compares to host FP32 reference.
# --------------------------------------------------------------------------- #


def test_pv_chain[T: DType](ctx: DeviceContext) raises -> Bool:
    var dtype_name = "BF16" if T == DType.bfloat16 else "FP8"
    print("--- test_pv_chain[", dtype_name, "] ---")

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
    comptime _V_SUB_ROWS = _Op.V_SUB_ROWS
    comptime _V_SUB_COLS = _Op.V_SUB_COLS
    comptime _NUM_BLOCK_COLS_V = DEPTH // _V_SUB_COLS
    comptime _V_SLOT_ROWS = KV_BLOCK * _NUM_BLOCK_COLS_V
    comptime _SMEM_SIZE = _V_SLOT_ROWS * _V_SUB_COLS  # T-elements
    comptime _P_SIZE = KV_BLOCK * Q_BLOCK_SIZE
    comptime _OH = _Op.O_LAYOUT.static_shape[0]
    comptime _OW = _Op.O_LAYOUT.static_shape[1]
    comptime _per_lane = _OH * _OW * 16
    comptime _DUMP_SIZE = 64 * _per_lane
    # FP8 e4m3fn has very limited precision (~3 mantissa bits); per-cell
    # quantization error can be ~6% and accumulates over 64 keys. cos_sim
    # is the load-bearing metric for FP8; max-diff is mainly a sanity
    # check.
    comptime _tol: Float32 = 0.5 if T == .float8_e4m3fn else 5e-2

    var dev_v_subtile = ctx.enqueue_create_buffer[T](_SMEM_SIZE)
    var dev_p = ctx.enqueue_create_buffer[T](_P_SIZE)
    var dev_dump = ctx.enqueue_create_buffer[.float32](_DUMP_SIZE)

    # ---- Build the sub-tile-major V SMEM image on host --------------------
    # This is the layout the production DMA producer
    # `SubTileLoaderLDS_st_8x32` writes (see
    # `max/kernels/src/structured_kernels/amd_tile_io.mojo` line 1568).
    # For every SMEM byte position `b`:
    #   sub_id            = b / sub_bytes
    #   sub_lane_byte     = b % sub_bytes
    #   sub_row, sub_col  = divmod(sub_id, subtiles_per_row)
    #   row_in_sub, col_byte = divmod(sub_lane_byte, sub_row_bytes)
    #   col_in_sub        = col_byte // size_of[dtype]
    #   global_key        = sub_row * V_SUB_ROWS + row_in_sub
    #   global_depth      = sub_col * V_SUB_COLS + col_in_sub
    #   SMEM[b]           = V[global_key, global_depth]
    #
    # We fill it the equivalent forward way: for each (key, depth),
    # compute its SMEM element index from the sub-tile-major byte
    # formula:
    #   sub_row = key // V_SUB_ROWS
    #   sub_col = depth // V_SUB_COLS
    #   sub_id  = sub_row * subtiles_per_row + sub_col
    #   row_in_sub = key % V_SUB_ROWS
    #   col_in_sub = depth % V_SUB_COLS
    #   slot_row = sub_id * V_SUB_ROWS + row_in_sub
    #   elt_idx  = slot_row * V_SUB_COLS + col_in_sub
    # (FP8 = 1 B/elt and BF16 = 2 B/elt; element index = byte // sizeof
    # already.)
    with dev_v_subtile.map_to_host() as host_v:
        for i in range(_SMEM_SIZE):
            host_v[i] = _quantize[T](Float32(0.0))

        for key in range(KV_BLOCK):
            for depth in range(DEPTH):
                var sub_row = key // _V_SUB_ROWS
                var sub_col = depth // _V_SUB_COLS
                var sub_id = sub_row * _NUM_BLOCK_COLS_V + sub_col
                var row_in_sub = key % _V_SUB_ROWS
                var col_in_sub = depth % _V_SUB_COLS
                var slot_row = sub_id * _V_SUB_ROWS + row_in_sub
                var elt_idx = slot_row * _V_SUB_COLS + col_in_sub
                host_v[elt_idx] = _quantize[T](_V_fp32(key, depth))

    # ---- Build the contiguous P gmem image on host ------------------------
    with dev_p.map_to_host() as host_p:
        for key in range(KV_BLOCK):
            for q in range(Q_BLOCK_SIZE):
                host_p[key * Q_BLOCK_SIZE + q] = _quantize[T](_P_fp32(key, q))

    # ---- Launch the kernel ------------------------------------------------
    ctx.enqueue_function[kernel_pv_chain[CFG, T]](
        dev_v_subtile.unsafe_ptr(),
        dev_p.unsafe_ptr(),
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    # ---- Host reference computation --------------------------------------- #
    # Quantize V and P to T, then dequantize back to FP32 -> bit-exact
    # mirror of what the kernel sees, computed in pure FP32. Then
    # ref[depth, q] = sum_key V_q[key, depth] * P_q[key, q] sequentially
    # in FP32 — bypasses every per-lane assumption.
    var dev_v_quant = ctx.enqueue_create_buffer[.float32](KV_BLOCK * DEPTH)
    var dev_p_quant = ctx.enqueue_create_buffer[.float32](
        KV_BLOCK * Q_BLOCK_SIZE
    )
    var dev_o_ref = ctx.enqueue_create_buffer[.float32](DEPTH * Q_BLOCK_SIZE)

    var mismatches: Int = 0
    var max_diff: Float32 = 0.0
    var sumsq_err: Float32 = 0.0
    var sumsq_ref: Float32 = 0.0
    var sumsq_got: Float32 = 0.0
    var dot_got_ref: Float32 = 0.0
    var worst_depth: Int = 0
    var worst_q: Int = 0

    with dev_v_quant.map_to_host() as v_quant, dev_p_quant.map_to_host() as p_quant, dev_o_ref.map_to_host() as o_ref, dev_dump.map_to_host() as host_dump:
        for key in range(KV_BLOCK):
            for depth in range(DEPTH):
                v_quant[key * DEPTH + depth] = _dequant_to_f32[T](
                    _quantize[T](_V_fp32(key, depth))
                )
        for key in range(KV_BLOCK):
            for q in range(Q_BLOCK_SIZE):
                p_quant[key * Q_BLOCK_SIZE + q] = _dequant_to_f32[T](
                    _quantize[T](_P_fp32(key, q))
                )

        # Pure-FP32 reference: o[depth, q] = sum_key V[key, depth] * P[key, q]
        for depth in range(DEPTH):
            for q in range(Q_BLOCK_SIZE):
                var acc: Float32 = 0.0
                for key in range(KV_BLOCK):
                    acc += (
                        v_quant[key * DEPTH + depth]
                        * p_quant[key * Q_BLOCK_SIZE + q]
                    )
                o_ref[depth * Q_BLOCK_SIZE + q] = acc

        # De-map per-lane fragment -> (depth, q) and compare.
        # C-output fragment lane mapping (col_l rt_32x32):
        #   row_in_tile = ACC_ROW_OFFSETS_32x32[k_local] + (lid >> 5) * 4
        #   col_in_tile = lid & 31
        # For mma_PV: o[n, m] is at depth = n * MMA_M + row_in_tile,
        # q = m * MMA_N + col_in_tile.
        for lid in range(64):
            var col_in_tile = lid & 31
            for n_out in range(_OH):
                for m_out in range(_OW):
                    for k_local in range(16):
                        var row_in_tile = (
                            Int(ACC_ROW_OFFSETS_32x32[k_local]) + (lid >> 5) * 4
                        )
                        var depth = n_out * _MMA_M + row_in_tile
                        var q = m_out * _MMA_N + col_in_tile
                        var idx = (
                            lid * _per_lane
                            + (n_out * _OW + m_out) * 16
                            + k_local
                        )
                        var got = host_dump[idx]
                        var expected = o_ref[depth * Q_BLOCK_SIZE + q]
                        var diff = abs(got - expected)
                        if diff > max_diff:
                            max_diff = diff
                            worst_depth = depth
                            worst_q = q
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
                                    " depth=",
                                    depth,
                                    " q=",
                                    q,
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
        " worst@(depth=",
        worst_depth,
        " q=",
        worst_q,
        ")",
    )

    _ = dev_v_subtile^
    _ = dev_p^
    _ = dev_dump^
    _ = dev_v_quant^
    _ = dev_p_quant^
    _ = dev_o_ref^

    # cos_sim >= 0.99 is the headline-correctness metric for FP8 (the
    # FP8 quantization noise can push max-diff up; cos_sim is robust to
    # that).
    var ok_cos = cos_sim > 0.99
    var ok_diff = mismatches == 0

    if ok_cos and ok_diff:
        print("  PASSED")
        return True
    else:
        print(
            "  FAILED (mismatches=",
            mismatches,
            " cos_sim=",
            cos_sim,
            " tol=",
            _tol,
            ")",
        )
        return False


# --------------------------------------------------------------------------- #
# Main: runs the chain test for BF16 and FP8.
# --------------------------------------------------------------------------- #


def main() raises:
    print("=" * 60)
    print("Phase 14e: load_V + mma_PV chain pure-FP32 reference test")
    print("=" * 60)

    with DeviceContext() as ctx:
        var ok_bf16 = test_pv_chain[.bfloat16](ctx)
        var ok_fp8 = test_pv_chain[.float8_e4m3fn](ctx)

        print("=" * 60)
        print("Summary:")
        print("  BF16 = ", "PASS" if ok_bf16 else "FAIL")
        print("  FP8  = ", "PASS" if ok_fp8 else "FAIL")
        print("=" * 60)

        assert_true(ok_bf16, "BF16 PV chain test failed")
        assert_true(
            ok_fp8,
            (
                "FP8 PV chain test failed — bug is in V loader (sub-tile"
                "-major byte addressing), P-side per-lane B-operand layout,"
                " or mma_PV FP8 dispatch"
            ),
        )
