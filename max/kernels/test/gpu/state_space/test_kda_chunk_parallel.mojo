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
"""Tests for the KDA CHUNK-PARALLEL chunk-blocked prefill GPU path (M2c).

Correctness gate for the 3-kernel chunk-parallel prefill:
``kda_chunk_prepare_gpu`` (L1, parallel per (chunk, head)) ->
``kda_chunk_scan_gpu`` (L2, sequential inter-chunk state scan per sequence) ->
``kda_chunk_output_gpu`` (L3, parallel output contraction). This is the grid
that fixes the seq-parallel scaffold + M2b ``proportional-to-T`` latency.

The chunk-parallel path reassociates the fp32 arithmetic (interaction matrices +
triangular solve, S_in-affine split), so it is NOT bit-identical; it is checked
WITHIN the M1 decode tolerances against:
  - the M1 decode recurrence ``kda_decode_gpu`` (same inputs), and
  - the M0 CPU scalar reference ``kda_decode_ref`` (independent oracle).

Output error AND final-state error are tracked SEPARATELY (tol: 0.005/0.006
output, 0.005 fp32 state, 0.05 bf16 state — the M1 tolerances).

Axes: single/multi/ragged chunks; varlen {fixed, ragged, empty seq}; gate
{original, safe}; beta {logits, probability}; layout {K_FIRST, V_FIRST}; GVA
{HV/H = 2, 4}; state {fp32, bf16}; zero + nonzero init. CHUNK_SIZE is fixed at
16 (matching the M2b algebra reuse; the scan cost is chunk-size independent).

``num_value_heads`` is 4 everywhere a non-GQA baseline is wanted (GQA cases
scale both heads to keep their ratio), never 1-3: L1's one-shot TMA load reads
``beta_logits`` through a 4-column box (``chunk_fwd.mojo:1592-1599``), which
needs the tensor's own row stride to be a 16-byte multiple, i.e.
``num_value_heads % 4 == 0`` -- a precondition of the TMA path itself, not a
choice made to route around a result.

``kda_chunk_prepare_gpu`` here keeps its default ``p_contract="pd"`` rather
than the production driver's ``"qs_b"``: this file's L2 launch uses
``block_dim=(VALUE_HEAD_DIM,)`` (128), below the tc2 warp-specialized body's
``WARPSPEC_BLOCK`` (192) floor, so ``kda_chunk_scan_gpu`` always falls through
to its M3 (non-fused) scan body -- the body the "pd" workspace layout was
built for. "qs_b" pairs only with the tc2-fused body (measured: swapping it in
here, unfused, turns several cases to NaN). Per the kernel-opt journal
(kda-chunk-prefill.md, "SEPARATE PRE-EXISTING DEFECT") this "pd" path already
carries a known, unresolved bug from before this file last compiled (a
Phase-D arena overlay read as decay under ``if use_tc:``) -- see the FAIL
below, root-caused but not yet fixed here.
"""

import std.math
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext

from layout import TileTensor, row_major
from layout.tma_async import create_tma_tile
from kda.chunk_fwd import (
    kda_chunk_prepare_gpu,
    kda_chunk_scan_gpu,
    kda_chunk_output_gpu,
    kda_chunk_seg_reduce_gpu,
    kda_chunk_seg_scan_gpu,
    kda_chunk_seg_apply_gpu,
)
from kda.recurrent import kda_decode_gpu
from kda.reference import kda_decode_ref


comptime CHUNK: Int = 16


def _rel_err_ptr(
    gold: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    cand: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float64:
    """RMSE(gold - cand) / RMSE(gold)."""
    var g_sq = Float64(0.0)
    var d_sq = Float64(0.0)
    for i in range(n):
        var g = Float64(gold[i])
        var d = Float64(gold[i]) - Float64(cand[i])
        g_sq = g_sq + g * g
        d_sq = d_sq + d * d
    var gold_rmse = sqrt(g_sq / Float64(n))
    var diff_rmse = sqrt(d_sq / Float64(n))
    return diff_rmse / (gold_rmse + Float64(1e-8))


def _run_both[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](
    batch_size: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    total_T: Int,
    seq_lengths: List[Int],
    q_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    k_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    v_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    rg_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    bl_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    al_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    dt_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    state_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    ctx: DeviceContext,
) raises -> Tuple[
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Returns (decode_out, decode_state, chunk_out, chunk_state), all fp32."""
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V

    # --- cu_seqlens (decode reference) ---
    var cu_h = alloc[Scalar[DType.int32]](batch_size + 1)
    var cumsum = 0
    cu_h[0] = Int32(0)
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        cu_h[b + 1] = Int32(cumsum)

    # --- state indices: batch b -> slot b ---
    var si_h = alloc[Scalar[DType.int32]](batch_size)
    for b in range(batch_size):
        si_h[b] = Int32(b)

    # --- Chunk map: seq_chunk_offsets [N+1], chunk_tok_offsets [C+1]. ---
    var seq_co = List[Int32]()
    var chunk_to = List[Int32]()
    seq_co.append(Int32(0))
    chunk_to.append(Int32(0))
    var chunk_count = 0
    var tok_cursor = 0
    for b in range(batch_size):
        var L = seq_lengths[b]
        var nc = (L + CHUNK - 1) // CHUNK
        for c in range(nc):
            var rem = L - c * CHUNK
            var this_len = CHUNK if rem >= CHUNK else rem
            tok_cursor += this_len
            chunk_to.append(Int32(tok_cursor))
        chunk_count += nc
        seq_co.append(Int32(chunk_count))
    var num_chunks = len(chunk_to) - 1

    var sco_h = alloc[Scalar[DType.int32]](batch_size + 1)
    for i in range(batch_size + 1):
        sco_h[i] = seq_co[i]
    var cto_h = alloc[Scalar[DType.int32]](num_chunks + 1)
    for i in range(num_chunks + 1):
        cto_h[i] = chunk_to[i]

    # --- Device buffers ---
    var q_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var k_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var v_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * HV * V)
    var rg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * K)
    var bl_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var al_dev = ctx.enqueue_create_buffer[DType.float32](HV)
    var dt_dev = ctx.enqueue_create_buffer[DType.float32](HV * K)
    var cu_dev = ctx.enqueue_create_buffer[DType.int32](batch_size + 1)
    var si_dev = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var sco_dev = ctx.enqueue_create_buffer[DType.int32](batch_size + 1)
    var cto_dev = ctx.enqueue_create_buffer[DType.int32](num_chunks + 1)
    var pool_dec_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var pool_chk_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_dec_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)
    var out_chk_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

    # --- Workspace (fp32, FIXED internal [num_chunks, HV, ., .] layout). ---
    var nch = num_chunks * HV
    var ws_W_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_Uv_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_KE_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_glast_dev = ctx.enqueue_create_buffer[DType.float32](nch * K)
    var ws_P_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_d_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_Sin_dev = ctx.enqueue_create_buffer[DType.float32](nch * K * V)

    with ctx.push_context():
        ctx.enqueue_copy(q_dev, q_h)
        ctx.enqueue_copy(k_dev, k_h)
        ctx.enqueue_copy(v_dev, v_h)
        ctx.enqueue_copy(rg_dev, rg_h)
        ctx.enqueue_copy(bl_dev, bl_h)
        ctx.enqueue_copy(al_dev, al_h)
        ctx.enqueue_copy(dt_dev, dt_h)
        ctx.enqueue_copy(cu_dev, cu_h)
        ctx.enqueue_copy(si_dev, si_h)
        ctx.enqueue_copy(sco_dev, sco_h)
        ctx.enqueue_copy(cto_dev, cto_h)

    var state_src = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        state_src[i] = Scalar[state_dtype](state_h[i])
    with ctx.push_context():
        ctx.enqueue_copy(pool_dec_dev, state_src)
        ctx.enqueue_copy(pool_chk_dev, state_src)
    state_src.free()

    var q_tt = TileTensor(q_dev, row_major(total_T, H * K))
    var k_tt = TileTensor(k_dev, row_major(total_T, H * K))
    var v_tt = TileTensor(v_dev, row_major(total_T, HV * V))
    var rg_tt = TileTensor(rg_dev, row_major(total_T, HV * K))
    var bl_tt = TileTensor(bl_dev, row_major(total_T, HV))
    var al_tt = TileTensor(al_dev, row_major(HV))
    var dt_tt = TileTensor(dt_dev, row_major(HV, K))
    var cu_tt = TileTensor(cu_dev, row_major(batch_size + 1))
    var si_tt = TileTensor(si_dev, row_major(batch_size))
    var sco_tt = TileTensor(sco_dev, row_major(batch_size + 1))
    var cto_tt = TileTensor(cto_dev, row_major(num_chunks + 1))
    var out_dec_tt = TileTensor(out_dec_dev, row_major(total_T, HV * V))
    var out_chk_tt = TileTensor(out_chk_dev, row_major(total_T, HV * V))

    var ws_W_tt = TileTensor(ws_W_dev, row_major(nch * CHUNK * K))
    var ws_Uv_tt = TileTensor(ws_Uv_dev, row_major(nch * CHUNK * V))
    var ws_KE_tt = TileTensor(ws_KE_dev, row_major(nch * CHUNK * K))
    var ws_glast_tt = TileTensor(ws_glast_dev, row_major(nch * K))
    var ws_P_tt = TileTensor(ws_P_dev, row_major(nch * CHUNK * K))
    var ws_d_tt = TileTensor(ws_d_dev, row_major(nch * CHUNK * V))
    var ws_Sin_tt = TileTensor(ws_Sin_dev, row_major(nch * K * V))

    var num_dec_blocks = batch_size * HV
    var num_chunk_blocks = num_chunks * HV
    out_dec_dev.enqueue_fill(0.0)
    out_chk_dev.enqueue_fill(0.0)

    # --- TMA descriptors for L1's one-shot GMEM loads (FlashKDA K1 port). ---
    var q_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, q_tt.to_layout_tensor()
    )
    var k_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, k_tt.to_layout_tensor()
    )
    var rg_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, rg_tt.to_layout_tensor()
    )
    var bl_tma = create_tma_tile[CHUNK, 4](ctx, bl_tt.to_layout_tensor())
    var dt_tma = create_tma_tile[1, KEY_HEAD_DIM](ctx, dt_tt.to_layout_tensor())

    # --- L1: parallel prepare (state-independent; layout-agnostic). ---
    ctx.enqueue_function[
        kda_chunk_prepare_gpu[
            qkv_dtype,
            DType.float32,
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            q_tt.LayoutType,
            k_tt.LayoutType,
            v_tt.LayoutType,
            rg_tt.LayoutType,
            bl_tt.LayoutType,
            al_tt.LayoutType,
            dt_tt.LayoutType,
            cto_tt.LayoutType,
            ws_W_tt.LayoutType,
            ws_Uv_tt.LayoutType,
            ws_KE_tt.LayoutType,
            ws_glast_tt.LayoutType,
            ws_P_tt.LayoutType,
            ws_d_tt.LayoutType,
            gate_mode,
            beta_mode,
        ]
    ](
        Int32(HV),
        Int32(H),
        q_tt,
        k_tt,
        v_tt,
        rg_tt,
        bl_tt,
        al_tt,
        dt_tt,
        cto_tt,
        q_tma,
        k_tma,
        rg_tma,
        bl_tma,
        dt_tma,
        ws_W_tt,
        ws_Uv_tt,
        ws_KE_tt,
        ws_glast_tt,
        ws_P_tt,
        ws_d_tt,
        UInt32(H * K),
        UInt32(K),
        UInt32(1),
        UInt32(H * K),
        UInt32(K),
        UInt32(1),
        UInt32(HV * V),
        UInt32(V),
        UInt32(1),
        UInt32(HV * K),
        UInt32(K),
        UInt32(1),
        UInt32(HV),
        UInt32(1),
        UInt32(K),
        UInt32(1),
        grid_dim=(num_chunk_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    comptime if state_layout == "K_FIRST":
        var pool_dec_tt = TileTensor(
            pool_dec_dev, row_major(batch_size, HV, K, V)
        )
        var pool_chk_tt = TileTensor(
            pool_chk_dev, row_major(batch_size, HV, K, V)
        )
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_dec_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_dec_tt.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_dec_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_dec_tt,
            si_tt,
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV),
            UInt32(1),
            UInt32(K),
            UInt32(1),
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        ctx.enqueue_function[
            kda_chunk_scan_gpu[
                state_dtype,
                DType.float32,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                ws_W_tt.LayoutType,
                ws_Uv_tt.LayoutType,
                ws_KE_tt.LayoutType,
                ws_glast_tt.LayoutType,
                ws_Sin_tt.LayoutType,
                ws_P_tt.LayoutType,
                ws_d_tt.LayoutType,
                pool_chk_tt.LayoutType,
                si_tt.LayoutType,
                out_chk_tt.LayoutType,
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            sco_tt,
            cto_tt,
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Sin_tt,
            ws_P_tt,
            ws_d_tt,
            pool_chk_tt,
            si_tt,
            out_chk_tt,
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    else:  # V_FIRST
        var pool_dec_tt = TileTensor(
            pool_dec_dev, row_major(batch_size, HV, V, K)
        )
        var pool_chk_tt = TileTensor(
            pool_chk_dev, row_major(batch_size, HV, V, K)
        )
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_dec_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_dec_tt.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "V_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_dec_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_dec_tt,
            si_tt,
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV),
            UInt32(1),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V * K),
            UInt32(V * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        ctx.enqueue_function[
            kda_chunk_scan_gpu[
                state_dtype,
                DType.float32,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                ws_W_tt.LayoutType,
                ws_Uv_tt.LayoutType,
                ws_KE_tt.LayoutType,
                ws_glast_tt.LayoutType,
                ws_Sin_tt.LayoutType,
                ws_P_tt.LayoutType,
                ws_d_tt.LayoutType,
                pool_chk_tt.LayoutType,
                si_tt.LayoutType,
                out_chk_tt.LayoutType,
                "V_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            sco_tt,
            cto_tt,
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Sin_tt,
            ws_P_tt,
            ws_d_tt,
            pool_chk_tt,
            si_tt,
            out_chk_tt,
            UInt32(HV * V * K),
            UInt32(V * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    # --- L3: parallel output contraction (layout-agnostic). ---
    ctx.enqueue_function[
        kda_chunk_output_gpu[
            DType.float32,
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            out_chk_tt.LayoutType,
            cto_tt.LayoutType,
            ws_Sin_tt.LayoutType,
            ws_P_tt.LayoutType,
            ws_d_tt.LayoutType,
        ]
    ](
        Int32(HV),
        out_chk_tt,
        cto_tt,
        ws_Sin_tt,
        ws_P_tt,
        ws_d_tt,
        UInt32(HV * V),
        UInt32(V),
        UInt32(1),
        grid_dim=(num_chunk_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    var dec_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    var chk_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    var dec_pool_typed = alloc[Scalar[state_dtype]](pool_size)
    var chk_pool_typed = alloc[Scalar[state_dtype]](pool_size)
    var dec_pool = alloc[Scalar[DType.float32]](pool_size)
    var chk_pool = alloc[Scalar[DType.float32]](pool_size)

    with ctx.push_context():
        ctx.enqueue_copy(dec_out, out_dec_dev)
        ctx.enqueue_copy(chk_out, out_chk_dev)
        ctx.enqueue_copy(dec_pool_typed, pool_dec_dev)
        ctx.enqueue_copy(chk_pool_typed, pool_chk_dev)
    ctx.synchronize()

    for i in range(pool_size):
        dec_pool[i] = Scalar[DType.float32](dec_pool_typed[i])
        chk_pool[i] = Scalar[DType.float32](chk_pool_typed[i])

    dec_pool_typed.free()
    chk_pool_typed.free()
    cu_h.free()
    si_h.free()
    sco_h.free()
    cto_h.free()

    return dec_out, dec_pool, chk_out, chk_pool


def _check[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](
    tag: String,
    batch_size: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    total_T: Int,
    seq_lengths: List[Int],
    ctx: DeviceContext,
    nonzero_init: Bool = True,
    tol_output: Float64 = 0.006,
    tol_state: Float64 = 0.005,
) raises:
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V

    var q_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var k_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var v_h = alloc[Scalar[qkv_dtype]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_h = alloc[Scalar[DType.float32]](pool_size)

    import std.math

    for i in range(total_T * H * K):
        q_h[i] = Scalar[qkv_dtype](
            std.math.sin(Float32(i + 1) * Float32(0.313))
        )
        k_h[i] = Scalar[qkv_dtype](
            std.math.cos(Float32(i + 1) * Float32(0.217))
        )
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[qkv_dtype](
            std.math.sin(Float32(i + 7) * Float32(0.491)) * Float32(0.5)
        )
    for i in range(total_T * HV * K):
        rg_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 3) * Float32(0.137))
        )
    for i in range(total_T * HV):
        bl_h[i] = Scalar[DType.float32](
            std.math.cos(Float32(i + 5) * Float32(0.274))
        )
    for i in range(HV):
        al_h[i] = Scalar[DType.float32](
            Float32(-0.5) + Float32(i) * Float32(0.1)
        )
    for i in range(HV * K):
        dt_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 2) * Float32(0.057)) * Float32(0.1)
        )
    for i in range(pool_size):
        if nonzero_init:
            state_h[i] = Scalar[DType.float32](
                std.math.sin(Float32(i + 11) * Float32(0.193)) * Float32(0.2)
            )
        else:
            state_h[i] = Scalar[DType.float32](0.0)

    var ref_state_h = alloc[Scalar[DType.float32]](pool_size)
    comptime if state_layout == "V_FIRST":
        for n in range(batch_size):
            for hv in range(HV):
                for kd in range(K):
                    for vd in range(V):
                        var src = n * HV * K * V + hv * K * V + kd * V + vd
                        var dst = n * HV * V * K + hv * V * K + vd * K + kd
                        ref_state_h[dst] = state_h[src]
    else:
        for i in range(pool_size):
            ref_state_h[i] = state_h[i]

    var dec_out, dec_state, chk_out, chk_state = _run_both[
        qkv_dtype,
        state_dtype,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        gate_mode,
        beta_mode,
        state_layout,
    ](
        batch_size,
        HV,
        H,
        total_T,
        seq_lengths,
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        ref_state_h,
        ctx,
    )

    var ref_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    for i in range(total_T * HV * V):
        ref_out[i] = Scalar[DType.float32](0.0)

    var si_ref = alloc[Scalar[DType.int32]](batch_size)
    var cu_ref = alloc[Scalar[DType.int32]](batch_size + 1)
    var cumsum = 0
    cu_ref[0] = Int32(0)
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        cu_ref[b + 1] = Int32(cumsum)
        si_ref[b] = Int32(b)

    var state_dim1: Int
    var state_dim2: Int
    comptime if state_layout == "K_FIRST":
        state_dim1 = V
        state_dim2 = 1
    else:
        state_dim1 = K
        state_dim2 = 1

    kda_decode_ref[
        qkv_dtype,
        DType.float32,
        DType.float32,
        DType.float32,
        gate_mode,
        beta_mode,
        state_layout,
    ](
        batch_size=batch_size,
        num_value_heads=HV,
        num_key_heads=H,
        key_head_dim=K,
        value_head_dim=V,
        output_ptr=ref_out.bitcast[Scalar[DType.float32]](),
        q_ptr=q_h.bitcast[Scalar[qkv_dtype]](),
        k_ptr=k_h.bitcast[Scalar[qkv_dtype]](),
        v_ptr=v_h.bitcast[Scalar[qkv_dtype]](),
        raw_gate_ptr=rg_h.bitcast[Scalar[DType.float32]](),
        beta_logits_ptr=bl_h.bitcast[Scalar[DType.float32]](),
        a_log_ptr=al_h.bitcast[Scalar[DType.float32]](),
        dt_bias_ptr=dt_h.bitcast[Scalar[DType.float32]](),
        cu_seqlens_ptr=cu_ref.bitcast[Scalar[DType.int32]](),
        state_pool_ptr=ref_state_h.bitcast[Scalar[DType.float32]](),
        state_indices_ptr=si_ref.bitcast[Scalar[DType.int32]](),
        q_seqlen_stride=H * K,
        q_head_stride=K,
        q_key_stride=1,
        k_seqlen_stride=H * K,
        k_head_stride=K,
        k_key_stride=1,
        v_seqlen_stride=HV * V,
        v_head_stride=V,
        v_value_stride=1,
        raw_gate_seqlen_stride=HV * K,
        raw_gate_head_stride=K,
        raw_gate_key_stride=1,
        beta_seqlen_stride=HV,
        beta_head_stride=1,
        dt_bias_head_stride=K,
        dt_bias_key_stride=1,
        state_slot_stride=HV * K * V,
        state_head_stride=K * V,
        state_dim1_stride=state_dim1,
        state_dim2_stride=state_dim2,
        out_seqlen_stride=HV * V,
        out_head_stride=V,
        out_value_stride=1,
    )

    var err_o_dec = _rel_err_ptr(dec_out, chk_out, total_T * HV * V)
    var err_s_dec = _rel_err_ptr(dec_state, chk_state, pool_size)
    var err_o_ref = _rel_err_ptr(ref_out, chk_out, total_T * HV * V)
    var err_s_ref = _rel_err_ptr(ref_state_h, chk_state, pool_size)

    print(
        tag
        + " | cp-vs-decode out="
        + String(err_o_dec)
        + " state="
        + String(err_s_dec)
        + " | cp-vs-M0ref out="
        + String(err_o_ref)
        + " state="
        + String(err_s_ref)
    )

    assert_true(
        err_o_dec < tol_output,
        tag + " cp-vs-decode output rel_err=" + String(err_o_dec),
    )
    assert_true(
        err_s_dec < tol_state,
        tag + " cp-vs-decode state rel_err=" + String(err_s_dec),
    )
    assert_true(
        err_o_ref < tol_output,
        tag + " cp-vs-M0ref output rel_err=" + String(err_o_ref),
    )
    assert_true(
        err_s_ref < tol_state,
        tag + " cp-vs-M0ref state rel_err=" + String(err_s_ref),
    )

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_h.free()
    ref_state_h.free()
    ref_out.free()
    si_ref.free()
    cu_ref.free()
    dec_out.free()
    dec_state.free()
    chk_out.free()
    chk_state.free()


def test_cp_t16_single_original_kfirst() raises:
    """T=16 == CHUNK: single full chunk, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cp-t16",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=16,
        seq_lengths=[16],
        ctx=ctx,
    )


def test_cp_t64_multichunk_safe_kfirst() raises:
    """T=64: 4 full chunks of 16, safe/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "K_FIRST"
    ](
        "cp-t64-safe",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=64,
        seq_lengths=[64],
        ctx=ctx,
    )


def test_cp_t100_ragged_original_vfirst() raises:
    """T=100: 7 chunks (6*16 + 4) — ragged last chunk, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "V_FIRST"
    ](
        "cp-t100-ragged",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=100,
        seq_lengths=[100],
        ctx=ctx,
    )


def test_cp_t256_original_kfirst() raises:
    """T=256: 16 full chunks, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cp-t256",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=256,
        seq_lengths=[256],
        ctx=ctx,
    )


def test_cp_varlen_ragged_original_kfirst() raises:
    """Varlen ragged [200, 56], original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cp-varlen",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=4,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
    )


def test_cp_varlen_empty_seq_safe_vfirst() raises:
    """Varlen with an EMPTY sequence [0, 64, 10], safe/logits, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "V_FIRST"
    ](
        "cp-varlen-empty",
        batch_size=3,
        num_value_heads=4,
        num_key_heads=4,
        total_T=74,
        seq_lengths=[0, 64, 10],
        ctx=ctx,
    )


def test_cp_gqa_hv2_original_kfirst() raises:
    """GVA HV/H=2, varlen [64, 130], original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cp-gqa-hv2",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=2,
        total_T=194,
        seq_lengths=[64, 130],
        ctx=ctx,
    )


def test_cp_gqa_hv4_safe_probability_vfirst() raises:
    """GVA HV/H=4, varlen [80, 33], safe/probability, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "probability",
        "V_FIRST",
    ](
        "cp-gqa-hv4",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=1,
        total_T=113,
        seq_lengths=[80, 33],
        ctx=ctx,
    )


def test_cp_zero_init_original_kfirst() raises:
    """Zero initial state, T=100, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cp-zero-init",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=100,
        seq_lengths=[100],
        ctx=ctx,
        nonzero_init=False,
    )


def test_cp_bf16_state_original_kfirst() raises:
    """BF16 state, T=64, original/logits, K_FIRST (looser state tol)."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16,
        DType.bfloat16,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        "cp-bf16-state",
        batch_size=1,
        num_value_heads=4,
        num_key_heads=4,
        total_T=64,
        seq_lengths=[64],
        ctx=ctx,
        tol_state=0.05,
    )


def test_cp_k3_headdim_t64_original_kfirst() raises:
    """K3's own (key_head_dim, value_head_dim) = (32, 32), not (128, 128).

    Every other case in this file runs Kimi-Linear's head dims; K3 (via
    `kimi_k3_kda_attention` / the `kda_chunk` graph op in `kda.mojo`) uses
    (32, 32) with num_heads=8 and no GQA. `kda_chunk_scan_gpu`'s tc2
    tensor-core arm only compiles for KEY_HEAD_DIM==VALUE_HEAD_DIM==128, so
    this comptime-falls to the portable M3 scan body -- the same body every
    other case in this file already exercises -- but the (32, 32) geometry
    itself (TMA box widths, workspace strides, warp/lane mapping) is
    otherwise untested until this case.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 32, 32, "original", "logits", "K_FIRST"
    ](
        "cp-k3-headdim-t64",
        batch_size=1,
        num_value_heads=8,
        num_key_heads=8,
        total_T=64,
        seq_lengths=[64],
        ctx=ctx,
    )


def test_cp_k3_headdim_varlen_ragged_vfirst() raises:
    """K3 head dims, varlen ragged [200, 56], V_FIRST -- the production shape.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 32, 32, "original", "logits", "V_FIRST"
    ](
        "cp-k3-headdim-varlen",
        batch_size=2,
        num_value_heads=8,
        num_key_heads=8,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
    )


# ===----------------------------------------------------------------------=== #
# M3-full slice 1: two-level segmented associative scan (A -> B -> C).
#
# Compares the segmented path (L1 -> A -> B -> C -> L3) against the M3 leader
# sequential scan (L1 -> kda_chunk_scan_gpu -> L3) sharing one L1 workspace, plus
# the M0 CPU reference. Scope: N=1, equal-head, fp32 workspace, K==V==128,
# CHUNK==16, ALIGNED T (num_chunks % seg_len == 0), fixed seg_len.
# ===----------------------------------------------------------------------=== #


def _run_segmented[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](
    batch_size: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    total_T: Int,
    seq_lengths: List[Int],
    seg_len: Int,
    q_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    k_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    v_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    rg_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    bl_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    al_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    dt_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    state_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    ctx: DeviceContext,
) raises -> Tuple[
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Returns (scan_out, scan_state, seg_out, seg_state), all fp32.

    ``scan_*`` = the M3 sequential scan (reference); ``seg_*`` = the segmented
    A->B->C path. Both share one L1 workspace and the same pool init.
    """
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V

    # --- state indices: batch b -> slot b ---
    var si_h = alloc[Scalar[DType.int32]](batch_size)
    for b in range(batch_size):
        si_h[b] = Int32(b)

    # --- Chunk map: seq_chunk_offsets [N+1], chunk_tok_offsets [C+1]. ---
    var seq_co = List[Int32]()
    var chunk_to = List[Int32]()
    seq_co.append(Int32(0))
    chunk_to.append(Int32(0))
    var chunk_count = 0
    var tok_cursor = 0
    for b in range(batch_size):
        var L = seq_lengths[b]
        var nc = (L + CHUNK - 1) // CHUNK
        for c in range(nc):
            var rem = L - c * CHUNK
            var this_len = CHUNK if rem >= CHUNK else rem
            tok_cursor += this_len
            chunk_to.append(Int32(tok_cursor))
        chunk_count += nc
        seq_co.append(Int32(chunk_count))
    var num_chunks = len(chunk_to) - 1

    if num_chunks % seg_len != 0:
        raise Error("_run_segmented requires aligned T (num_chunks % seg_len)")
    var num_segments = num_chunks // seg_len

    var sco_h = alloc[Scalar[DType.int32]](batch_size + 1)
    for i in range(batch_size + 1):
        sco_h[i] = seq_co[i]
    var cto_h = alloc[Scalar[DType.int32]](num_chunks + 1)
    for i in range(num_chunks + 1):
        cto_h[i] = chunk_to[i]

    # --- Device buffers ---
    var q_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var k_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var v_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * HV * V)
    var rg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * K)
    var bl_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var al_dev = ctx.enqueue_create_buffer[DType.float32](HV)
    var dt_dev = ctx.enqueue_create_buffer[DType.float32](HV * K)
    var si_dev = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var sco_dev = ctx.enqueue_create_buffer[DType.int32](batch_size + 1)
    var cto_dev = ctx.enqueue_create_buffer[DType.int32](num_chunks + 1)
    var pool_ref_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var pool_seg_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_ref_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)
    var out_seg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

    # --- L1 workspace (fp32, FIXED internal [num_chunks, HV, ., .] layout). ---
    var nch = num_chunks * HV
    var ws_W_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_Uv_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_KE_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_glast_dev = ctx.enqueue_create_buffer[DType.float32](nch * K)
    var ws_P_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_d_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_Sin_ref_dev = ctx.enqueue_create_buffer[DType.float32](nch * K * V)
    var ws_Sin_seg_dev = ctx.enqueue_create_buffer[DType.float32](nch * K * V)

    # --- Segmented workspace ([num_segments, HV, ., .]). ---
    var nseg = num_segments * HV
    var ws_Mseg_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * K)
    var ws_cseg_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * V)
    var ws_Sseg_in_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * V)

    with ctx.push_context():
        ctx.enqueue_copy(q_dev, q_h)
        ctx.enqueue_copy(k_dev, k_h)
        ctx.enqueue_copy(v_dev, v_h)
        ctx.enqueue_copy(rg_dev, rg_h)
        ctx.enqueue_copy(bl_dev, bl_h)
        ctx.enqueue_copy(al_dev, al_h)
        ctx.enqueue_copy(dt_dev, dt_h)
        ctx.enqueue_copy(si_dev, si_h)
        ctx.enqueue_copy(sco_dev, sco_h)
        ctx.enqueue_copy(cto_dev, cto_h)

    var state_src = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        state_src[i] = Scalar[state_dtype](state_h[i])
    with ctx.push_context():
        ctx.enqueue_copy(pool_ref_dev, state_src)
        ctx.enqueue_copy(pool_seg_dev, state_src)
    state_src.free()

    var q_tt = TileTensor(q_dev, row_major(total_T, H * K))
    var k_tt = TileTensor(k_dev, row_major(total_T, H * K))
    var v_tt = TileTensor(v_dev, row_major(total_T, HV * V))
    var rg_tt = TileTensor(rg_dev, row_major(total_T, HV * K))
    var bl_tt = TileTensor(bl_dev, row_major(total_T, HV))
    var al_tt = TileTensor(al_dev, row_major(HV))
    var dt_tt = TileTensor(dt_dev, row_major(HV, K))
    var si_tt = TileTensor(si_dev, row_major(batch_size))
    var sco_tt = TileTensor(sco_dev, row_major(batch_size + 1))
    var cto_tt = TileTensor(cto_dev, row_major(num_chunks + 1))
    var out_ref_tt = TileTensor(out_ref_dev, row_major(total_T, HV * V))
    var out_seg_tt = TileTensor(out_seg_dev, row_major(total_T, HV * V))

    var ws_W_tt = TileTensor(ws_W_dev, row_major(nch * CHUNK * K))
    var ws_Uv_tt = TileTensor(ws_Uv_dev, row_major(nch * CHUNK * V))
    var ws_KE_tt = TileTensor(ws_KE_dev, row_major(nch * CHUNK * K))
    var ws_glast_tt = TileTensor(ws_glast_dev, row_major(nch * K))
    var ws_P_tt = TileTensor(ws_P_dev, row_major(nch * CHUNK * K))
    var ws_d_tt = TileTensor(ws_d_dev, row_major(nch * CHUNK * V))
    var ws_Sin_ref_tt = TileTensor(ws_Sin_ref_dev, row_major(nch * K * V))
    var ws_Sin_seg_tt = TileTensor(ws_Sin_seg_dev, row_major(nch * K * V))
    var ws_Mseg_tt = TileTensor(ws_Mseg_dev, row_major(nseg * K * K))
    var ws_cseg_tt = TileTensor(ws_cseg_dev, row_major(nseg * K * V))
    var ws_Sseg_in_tt = TileTensor(ws_Sseg_in_dev, row_major(nseg * K * V))

    var num_dec_blocks = batch_size * HV
    var num_chunk_blocks = num_chunks * HV
    var num_seg_blocks = num_segments * HV
    out_ref_dev.enqueue_fill(0.0)
    out_seg_dev.enqueue_fill(0.0)

    # --- TMA descriptors for L1's one-shot GMEM loads (FlashKDA K1 port). ---
    var q_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, q_tt.to_layout_tensor()
    )
    var k_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, k_tt.to_layout_tensor()
    )
    var rg_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, rg_tt.to_layout_tensor()
    )
    var bl_tma = create_tma_tile[CHUNK, 4](ctx, bl_tt.to_layout_tensor())
    var dt_tma = create_tma_tile[1, KEY_HEAD_DIM](ctx, dt_tt.to_layout_tensor())

    # --- L1: parallel prepare (state-independent, shared by both paths). ---
    ctx.enqueue_function[
        kda_chunk_prepare_gpu[
            qkv_dtype,
            DType.float32,
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            q_tt.LayoutType,
            k_tt.LayoutType,
            v_tt.LayoutType,
            rg_tt.LayoutType,
            bl_tt.LayoutType,
            al_tt.LayoutType,
            dt_tt.LayoutType,
            cto_tt.LayoutType,
            ws_W_tt.LayoutType,
            ws_Uv_tt.LayoutType,
            ws_KE_tt.LayoutType,
            ws_glast_tt.LayoutType,
            ws_P_tt.LayoutType,
            ws_d_tt.LayoutType,
            gate_mode,
            beta_mode,
        ]
    ](
        Int32(HV),
        Int32(H),
        q_tt,
        k_tt,
        v_tt,
        rg_tt,
        bl_tt,
        al_tt,
        dt_tt,
        cto_tt,
        q_tma,
        k_tma,
        rg_tma,
        bl_tma,
        dt_tma,
        ws_W_tt,
        ws_Uv_tt,
        ws_KE_tt,
        ws_glast_tt,
        ws_P_tt,
        ws_d_tt,
        UInt32(H * K),
        UInt32(K),
        UInt32(1),
        UInt32(H * K),
        UInt32(K),
        UInt32(1),
        UInt32(HV * V),
        UInt32(V),
        UInt32(1),
        UInt32(HV * K),
        UInt32(K),
        UInt32(1),
        UInt32(HV),
        UInt32(1),
        UInt32(K),
        UInt32(1),
        grid_dim=(num_chunk_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    # --- Reference: M3 sequential scan (pool_ref, ws_Sin_ref). ---
    comptime if state_layout == "K_FIRST":
        var pool_ref_tt = TileTensor(
            pool_ref_dev, row_major(batch_size, HV, K, V)
        )
        ctx.enqueue_function[
            kda_chunk_scan_gpu[
                state_dtype,
                DType.float32,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                ws_W_tt.LayoutType,
                ws_Uv_tt.LayoutType,
                ws_KE_tt.LayoutType,
                ws_glast_tt.LayoutType,
                ws_Sin_ref_tt.LayoutType,
                ws_P_tt.LayoutType,
                ws_d_tt.LayoutType,
                pool_ref_tt.LayoutType,
                si_tt.LayoutType,
                out_ref_tt.LayoutType,
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            sco_tt,
            cto_tt,
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Sin_ref_tt,
            ws_P_tt,
            ws_d_tt,
            pool_ref_tt,
            si_tt,
            out_ref_tt,
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    else:  # V_FIRST
        var pool_ref_tt = TileTensor(
            pool_ref_dev, row_major(batch_size, HV, V, K)
        )
        ctx.enqueue_function[
            kda_chunk_scan_gpu[
                state_dtype,
                DType.float32,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                ws_W_tt.LayoutType,
                ws_Uv_tt.LayoutType,
                ws_KE_tt.LayoutType,
                ws_glast_tt.LayoutType,
                ws_Sin_ref_tt.LayoutType,
                ws_P_tt.LayoutType,
                ws_d_tt.LayoutType,
                pool_ref_tt.LayoutType,
                si_tt.LayoutType,
                out_ref_tt.LayoutType,
                "V_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            sco_tt,
            cto_tt,
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Sin_ref_tt,
            ws_P_tt,
            ws_d_tt,
            pool_ref_tt,
            si_tt,
            out_ref_tt,
            UInt32(HV * V * K),
            UInt32(V * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    # --- Segmented: A (reduce) -> B (segment scan) -> C (apply). ---
    ctx.enqueue_function[
        kda_chunk_seg_reduce_gpu[
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            ws_W_tt.LayoutType,
            ws_Uv_tt.LayoutType,
            ws_KE_tt.LayoutType,
            ws_glast_tt.LayoutType,
            ws_Mseg_tt.LayoutType,
            ws_cseg_tt.LayoutType,
        ]
    ](
        Int32(HV),
        Int32(seg_len),
        ws_W_tt,
        ws_Uv_tt,
        ws_KE_tt,
        ws_glast_tt,
        ws_Mseg_tt,
        ws_cseg_tt,
        grid_dim=(num_seg_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    comptime if state_layout == "K_FIRST":
        var pool_seg_tt = TileTensor(
            pool_seg_dev, row_major(batch_size, HV, K, V)
        )
        ctx.enqueue_function[
            kda_chunk_seg_scan_gpu[
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                ws_Mseg_tt.LayoutType,
                ws_cseg_tt.LayoutType,
                ws_Sseg_in_tt.LayoutType,
                pool_seg_tt.LayoutType,
                si_tt.LayoutType,
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(num_segments),
            ws_Mseg_tt,
            ws_cseg_tt,
            ws_Sseg_in_tt,
            pool_seg_tt,
            si_tt,
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    else:  # V_FIRST
        var pool_seg_tt = TileTensor(
            pool_seg_dev, row_major(batch_size, HV, V, K)
        )
        ctx.enqueue_function[
            kda_chunk_seg_scan_gpu[
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                ws_Mseg_tt.LayoutType,
                ws_cseg_tt.LayoutType,
                ws_Sseg_in_tt.LayoutType,
                pool_seg_tt.LayoutType,
                si_tt.LayoutType,
                "V_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(num_segments),
            ws_Mseg_tt,
            ws_cseg_tt,
            ws_Sseg_in_tt,
            pool_seg_tt,
            si_tt,
            UInt32(HV * V * K),
            UInt32(V * K),
            UInt32(K),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    ctx.enqueue_function[
        kda_chunk_seg_apply_gpu[
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            ws_W_tt.LayoutType,
            ws_Uv_tt.LayoutType,
            ws_KE_tt.LayoutType,
            ws_glast_tt.LayoutType,
            ws_Sseg_in_tt.LayoutType,
            ws_Sin_seg_tt.LayoutType,
        ]
    ](
        Int32(HV),
        Int32(seg_len),
        ws_W_tt,
        ws_Uv_tt,
        ws_KE_tt,
        ws_glast_tt,
        ws_Sseg_in_tt,
        ws_Sin_seg_tt,
        grid_dim=(num_seg_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    # --- L3: output contraction for both paths. ---
    ctx.enqueue_function[
        kda_chunk_output_gpu[
            DType.float32,
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            out_ref_tt.LayoutType,
            cto_tt.LayoutType,
            ws_Sin_ref_tt.LayoutType,
            ws_P_tt.LayoutType,
            ws_d_tt.LayoutType,
        ]
    ](
        Int32(HV),
        out_ref_tt,
        cto_tt,
        ws_Sin_ref_tt,
        ws_P_tt,
        ws_d_tt,
        UInt32(HV * V),
        UInt32(V),
        UInt32(1),
        grid_dim=(num_chunk_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )
    ctx.enqueue_function[
        kda_chunk_output_gpu[
            DType.float32,
            DType.float32,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            CHUNK,
            out_seg_tt.LayoutType,
            cto_tt.LayoutType,
            ws_Sin_seg_tt.LayoutType,
            ws_P_tt.LayoutType,
            ws_d_tt.LayoutType,
        ]
    ](
        Int32(HV),
        out_seg_tt,
        cto_tt,
        ws_Sin_seg_tt,
        ws_P_tt,
        ws_d_tt,
        UInt32(HV * V),
        UInt32(V),
        UInt32(1),
        grid_dim=(num_chunk_blocks,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    var ref_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    var seg_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    var ref_pool_typed = alloc[Scalar[state_dtype]](pool_size)
    var seg_pool_typed = alloc[Scalar[state_dtype]](pool_size)
    var ref_pool = alloc[Scalar[DType.float32]](pool_size)
    var seg_pool = alloc[Scalar[DType.float32]](pool_size)

    with ctx.push_context():
        ctx.enqueue_copy(ref_out, out_ref_dev)
        ctx.enqueue_copy(seg_out, out_seg_dev)
        ctx.enqueue_copy(ref_pool_typed, pool_ref_dev)
        ctx.enqueue_copy(seg_pool_typed, pool_seg_dev)
    ctx.synchronize()

    for i in range(pool_size):
        ref_pool[i] = Scalar[DType.float32](ref_pool_typed[i])
        seg_pool[i] = Scalar[DType.float32](seg_pool_typed[i])

    ref_pool_typed.free()
    seg_pool_typed.free()
    si_h.free()
    sco_h.free()
    cto_h.free()

    return ref_out, ref_pool, seg_out, seg_pool


def _check_seg[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](
    tag: String,
    total_T: Int,
    seg_len: Int,
    ctx: DeviceContext,
    nonzero_init: Bool = True,
    tol_output: Float64 = 0.006,
    tol_state: Float64 = 0.005,
) raises:
    """Single-sequence (N=1) segmented-path correctness vs the M3 scan + M0 ref.
    """
    var batch_size = 1
    var HV = 4
    var H = 4
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V
    var seq_lengths = List[Int]([total_T])

    var q_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var k_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var v_h = alloc[Scalar[qkv_dtype]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_h = alloc[Scalar[DType.float32]](pool_size)

    import std.math

    for i in range(total_T * H * K):
        q_h[i] = Scalar[qkv_dtype](
            std.math.sin(Float32(i + 1) * Float32(0.313))
        )
        k_h[i] = Scalar[qkv_dtype](
            std.math.cos(Float32(i + 1) * Float32(0.217))
        )
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[qkv_dtype](
            std.math.sin(Float32(i + 7) * Float32(0.491)) * Float32(0.5)
        )
    for i in range(total_T * HV * K):
        rg_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 3) * Float32(0.137))
        )
    for i in range(total_T * HV):
        bl_h[i] = Scalar[DType.float32](
            std.math.cos(Float32(i + 5) * Float32(0.274))
        )
    for i in range(HV):
        al_h[i] = Scalar[DType.float32](
            Float32(-0.5) + Float32(i) * Float32(0.1)
        )
    for i in range(HV * K):
        dt_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 2) * Float32(0.057)) * Float32(0.1)
        )
    for i in range(pool_size):
        if nonzero_init:
            state_h[i] = Scalar[DType.float32](
                std.math.sin(Float32(i + 11) * Float32(0.193)) * Float32(0.2)
            )
        else:
            state_h[i] = Scalar[DType.float32](0.0)

    # Layout-physical seed for the GPU pool + M0 ref (V_FIRST transpose).
    var ref_state_h = alloc[Scalar[DType.float32]](pool_size)
    comptime if state_layout == "V_FIRST":
        for kd in range(K):
            for vd in range(V):
                ref_state_h[vd * K + kd] = state_h[kd * V + vd]
    else:
        for i in range(pool_size):
            ref_state_h[i] = state_h[i]

    var scan_out, scan_state, seg_out, seg_state = _run_segmented[
        qkv_dtype,
        state_dtype,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        gate_mode,
        beta_mode,
        state_layout,
    ](
        batch_size,
        HV,
        H,
        total_T,
        seq_lengths,
        seg_len,
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        ref_state_h,
        ctx,
    )

    # M0 CPU reference (independent oracle).
    var ref_out = alloc[Scalar[DType.float32]](total_T * HV * V)
    for i in range(total_T * HV * V):
        ref_out[i] = Scalar[DType.float32](0.0)
    var m0_state_h = alloc[Scalar[DType.float32]](pool_size)
    for i in range(pool_size):
        m0_state_h[i] = ref_state_h[i]
    var si_ref = alloc[Scalar[DType.int32]](batch_size)
    var cu_ref = alloc[Scalar[DType.int32]](batch_size + 1)
    cu_ref[0] = Int32(0)
    cu_ref[1] = Int32(total_T)
    si_ref[0] = Int32(0)

    var state_dim1: Int
    var state_dim2: Int
    comptime if state_layout == "K_FIRST":
        state_dim1 = V
        state_dim2 = 1
    else:
        state_dim1 = K
        state_dim2 = 1

    kda_decode_ref[
        qkv_dtype,
        DType.float32,
        DType.float32,
        DType.float32,
        gate_mode,
        beta_mode,
        state_layout,
    ](
        batch_size=batch_size,
        num_value_heads=HV,
        num_key_heads=H,
        key_head_dim=K,
        value_head_dim=V,
        output_ptr=ref_out.bitcast[Scalar[DType.float32]](),
        q_ptr=q_h.bitcast[Scalar[qkv_dtype]](),
        k_ptr=k_h.bitcast[Scalar[qkv_dtype]](),
        v_ptr=v_h.bitcast[Scalar[qkv_dtype]](),
        raw_gate_ptr=rg_h.bitcast[Scalar[DType.float32]](),
        beta_logits_ptr=bl_h.bitcast[Scalar[DType.float32]](),
        a_log_ptr=al_h.bitcast[Scalar[DType.float32]](),
        dt_bias_ptr=dt_h.bitcast[Scalar[DType.float32]](),
        cu_seqlens_ptr=cu_ref.bitcast[Scalar[DType.int32]](),
        state_pool_ptr=m0_state_h.bitcast[Scalar[DType.float32]](),
        state_indices_ptr=si_ref.bitcast[Scalar[DType.int32]](),
        q_seqlen_stride=H * K,
        q_head_stride=K,
        q_key_stride=1,
        k_seqlen_stride=H * K,
        k_head_stride=K,
        k_key_stride=1,
        v_seqlen_stride=HV * V,
        v_head_stride=V,
        v_value_stride=1,
        raw_gate_seqlen_stride=HV * K,
        raw_gate_head_stride=K,
        raw_gate_key_stride=1,
        beta_seqlen_stride=HV,
        beta_head_stride=1,
        dt_bias_head_stride=K,
        dt_bias_key_stride=1,
        state_slot_stride=HV * K * V,
        state_head_stride=K * V,
        state_dim1_stride=state_dim1,
        state_dim2_stride=state_dim2,
        out_seqlen_stride=HV * V,
        out_head_stride=V,
        out_value_stride=1,
    )

    var err_o_scan = _rel_err_ptr(scan_out, seg_out, total_T * HV * V)
    var err_s_scan = _rel_err_ptr(scan_state, seg_state, pool_size)
    var err_o_ref = _rel_err_ptr(ref_out, seg_out, total_T * HV * V)
    var err_s_ref = _rel_err_ptr(m0_state_h, seg_state, pool_size)

    print(
        tag
        + " | seg-vs-scan out="
        + String(err_o_scan)
        + " state="
        + String(err_s_scan)
        + " | seg-vs-M0ref out="
        + String(err_o_ref)
        + " state="
        + String(err_s_ref)
    )

    assert_true(
        err_o_scan < tol_output,
        tag + " seg-vs-scan output rel_err=" + String(err_o_scan),
    )
    assert_true(
        err_s_scan < tol_state,
        tag + " seg-vs-scan state rel_err=" + String(err_s_scan),
    )
    assert_true(
        err_o_ref < tol_output,
        tag + " seg-vs-M0ref output rel_err=" + String(err_o_ref),
    )
    assert_true(
        err_s_ref < tol_state,
        tag + " seg-vs-M0ref state rel_err=" + String(err_s_ref),
    )

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_h.free()
    ref_state_h.free()
    m0_state_h.free()
    ref_out.free()
    si_ref.free()
    cu_ref.free()
    scan_out.free()
    scan_state.free()
    seg_out.free()
    seg_state.free()


def test_seg_t256_lseg4_kfirst() raises:
    """T=256 (16 chunks), L_seg=4 -> 4 segments, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ]("seg-t256-l4", total_T=256, seg_len=4, ctx=ctx)


def test_seg_t1024_lseg8_kfirst() raises:
    """T=1024 (64 chunks), L_seg=8 -> 8 segments, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ]("seg-t1024-l8", total_T=1024, seg_len=8, ctx=ctx)


def test_seg_t2048_lseg16_kfirst() raises:
    """T=2048 (128 chunks), L_seg=16 -> 8 segments, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ]("seg-t2048-l16", total_T=2048, seg_len=16, ctx=ctx)


def test_seg_t8192_lseg16_kfirst() raises:
    """T=8192 (512 chunks), L_seg=16 -> 32 segments, K_FIRST fp32 (long drift).
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ]("seg-t8192-l16", total_T=8192, seg_len=16, ctx=ctx)


def test_seg_t1024_lseg8_safe_vfirst() raises:
    """T=1024, L_seg=8, safe gate, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "V_FIRST"
    ]("seg-t1024-l8-vfirst", total_T=1024, seg_len=8, ctx=ctx)


def test_seg_t1024_lseg8_zero_init() raises:
    """Zero initial state, T=1024, L_seg=8, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_seg[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ]("seg-t1024-l8-zero", total_T=1024, seg_len=8, ctx=ctx, nonzero_init=False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
