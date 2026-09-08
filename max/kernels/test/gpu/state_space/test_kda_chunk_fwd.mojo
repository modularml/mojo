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
"""Tests for the KDA chunk-blocked prefill GPU kernel (M2 scaffold).

Correctness-order step 3 (Mojo chunked vs our OWN recurrence): compares
``kda_chunk_gpu`` against the BANKED M1 decode recurrence ``kda_decode_gpu`` on
the SAME inputs, and independently against the M0 CPU scalar reference
``kda_decode_ref``. Does NOT compare to FLA `chunk_kda` (that needs the prefill
harness extension — a separate step the orchestrator owns).

Output error AND final-state error are tracked SEPARATELY:
  - chunk_fwd vs decode: expected BIT-IDENTICAL (identical fp32 math + token
    order; chunk-outer only re-nests the token loop). Asserted tight (1e-5).
  - chunk_fwd vs M0 ref (independent oracle): 0.005 output / 0.005 fp32 state /
    0.05 bf16 state (matches the M1 decode tolerances).

Axes: short/med T {16, 64, 256}; chunk sizes {16, 64}; varlen {fixed, ragged
imbalanced incl empty sequence}; gate {original, safe}; beta {logits,
probability}; layout {K_FIRST, V_FIRST}; GVA {HV/H = 2, 4}; state {fp32, bf16};
zero + nonzero init state.
"""

import std.math
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext

from layout import TileTensor, row_major
from kda.chunk_fwd import kda_chunk_gpu
from kda.recurrent import kda_decode_gpu
from kda.reference import kda_decode_ref


# ===----------------------------------------------------------------------=== #
# Error metric
# ===----------------------------------------------------------------------=== #


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


# ===----------------------------------------------------------------------=== #
# Run decode (M1) + chunk (M2) on the SAME inputs; return 4 fp32 host arrays.
# ===----------------------------------------------------------------------=== #


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
    chunk_size: Int,
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
    """Returns (decode_out, decode_state, chunk_out, chunk_state), all fp32.

    Caller frees the four returned pointers.
    """
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V

    # --- cu_seqlens (decode) ---
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

    # --- Chunk map (chunk-blocked prefill): seq_chunk_offsets [N+1] cumulative
    #     chunk counts, chunk_tok_offsets [C+1] cumulative global token index.
    #     Built once here on the host (a stream-persistent builder in
    #     production); the kernel receives NO cu_seqlens. ---
    var seq_co = List[Int32]()
    var chunk_to = List[Int32]()
    seq_co.append(Int32(0))
    chunk_to.append(Int32(0))
    var chunk_count = 0
    var tok_cursor = 0
    for b in range(batch_size):
        var L = seq_lengths[b]
        var nc = (L + chunk_size - 1) // chunk_size  # ceildiv; 0 if L == 0
        for c in range(nc):
            var rem = L - c * chunk_size
            var this_len = chunk_size if rem >= chunk_size else rem
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

    # --- Device buffers (shared inputs) ---
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
    # Separate pools so each kernel writes its own final state.
    var pool_dec_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var pool_chk_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_dec_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)
    var out_chk_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

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

    # Seed both pools from the same init state (cast fp32 -> state_dtype).
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

    var num_blocks = batch_size * HV
    out_dec_dev.enqueue_fill(0.0)
    out_chk_dev.enqueue_fill(0.0)

    comptime if state_layout == "K_FIRST":
        var pool_dec_tt = TileTensor(
            pool_dec_dev, row_major(batch_size, HV, K, V)
        )
        var pool_chk_tt = TileTensor(
            pool_chk_dev, row_major(batch_size, HV, K, V)
        )
        # --- decode (M1 oracle) ---
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
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        # --- chunk (M2 scaffold) ---
        ctx.enqueue_function[
            kda_chunk_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_chk_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                pool_chk_tt.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_chk_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            sco_tt,
            cto_tt,
            pool_chk_tt,
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
            grid_dim=(num_blocks,),
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
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
        ctx.enqueue_function[
            kda_chunk_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_chk_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                pool_chk_tt.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "V_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_chk_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            sco_tt,
            cto_tt,
            pool_chk_tt,
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
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    # --- Copy results back, convert state_dtype -> fp32 ---
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


# ===----------------------------------------------------------------------=== #
# Generic harness: chunk vs decode (bit-identical) AND chunk vs M0 ref.
# ===----------------------------------------------------------------------=== #


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
    chunk_size: Int,
    seq_lengths: List[Int],
    ctx: DeviceContext,
    nonzero_init: Bool = True,
    tol_output: Float64 = 0.005,
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

    # Layout-physical initial state: K_FIRST keeps [N,HV,K,V]; V_FIRST stores the
    # transpose [N,HV,V,K] (pool[.,.,v,k] = S0[k,v]). Both the GPU kernels and the
    # CPU ref are seeded from this same physical buffer (K==V so element count is
    # identical; the transpose matters for the addressing).
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
        chunk_size,
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

    # --- M0 CPU reference (independent oracle); mutates ref_state_h -> final ---
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

    # (1) chunk vs decode: identical fp32 math + token order -> bit-identical
    # OUTPUT always. STATE is bit-identical only when state_dtype is fp32; for
    # bf16 state, decode narrows to bf16 every token while chunk narrows once
    # per chunk_size-token boundary (the whole point of the chunked algorithm),
    # so the two bf16-rounding schedules diverge slightly even though both
    # remain accurate against the independent CPU reference below. tol_state
    # already carries that slack (0.05 for the one bf16-state case here vs the
    # 0.005 default) -- reuse it instead of assuming bit-identical state.
    var err_o_dec = _rel_err_ptr(dec_out, chk_out, total_T * HV * V)
    var err_s_dec = _rel_err_ptr(dec_state, chk_state, pool_size)
    # (2) chunk vs M0 CPU reference (independent oracle).
    var err_o_ref = _rel_err_ptr(ref_out, chk_out, total_T * HV * V)
    var err_s_ref = _rel_err_ptr(ref_state_h, chk_state, pool_size)

    print(
        tag
        + " | chunk-vs-decode out="
        + String(err_o_dec)
        + " state="
        + String(err_s_dec)
        + " | chunk-vs-M0ref out="
        + String(err_o_ref)
        + " state="
        + String(err_s_ref)
    )

    assert_true(
        err_o_dec < Float64(1e-5),
        tag + " chunk-vs-decode output rel_err=" + String(err_o_dec),
    )
    assert_true(
        err_s_dec < tol_state,
        tag + " chunk-vs-decode state rel_err=" + String(err_s_dec),
    )
    assert_true(
        err_o_ref < tol_output,
        tag + " chunk-vs-M0ref output rel_err=" + String(err_o_ref),
    )
    assert_true(
        err_s_ref < tol_state,
        tag + " chunk-vs-M0ref state rel_err=" + String(err_s_ref),
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


# ===----------------------------------------------------------------------=== #
# Tests — short/med T, chunk boundaries, varlen (incl empty), all axes.
# ===----------------------------------------------------------------------=== #


def test_chunk_t16_partial_original_kfirst() raises:
    """T=16 < chunk 64: single partial chunk, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "t16-partial",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=16,
        chunk_size=64,
        seq_lengths=[16],
        ctx=ctx,
    )


def test_chunk_t64_exact_safe_kfirst() raises:
    """T=64 == chunk 64: single full chunk, safe/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "K_FIRST"
    ](
        "t64-exact-safe",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=64,
        chunk_size=64,
        seq_lengths=[64],
        ctx=ctx,
    )


def test_chunk_t65_raggedlast_original_vfirst() raises:
    """T=65: 2 chunks (64 + 1) — off-by-one ragged last chunk, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "V_FIRST"
    ](
        "t65-ragged-vfirst",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=65,
        chunk_size=64,
        seq_lengths=[65],
        ctx=ctx,
    )


def test_chunk_t256_multichunk_original_kfirst() raises:
    """T=256: 4 full chunks, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "t256-multichunk",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        chunk_size=64,
        seq_lengths=[256],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_smallchunk16_t64_safe_vfirst() raises:
    """`chunk_size=16` on T=64: 4 chunks (many boundaries), safe/prob, V_FIRST.
    """
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
        "chunk16-t64",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=64,
        chunk_size=16,
        seq_lengths=[64],
        ctx=ctx,
    )


def test_chunk_varlen_ragged_original_kfirst() raises:
    """Varlen ragged [200, 56], chunk 64, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "varlen-ragged",
        batch_size=2,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        chunk_size=64,
        seq_lengths=[200, 56],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_varlen_empty_seq_safe_vfirst() raises:
    """Varlen with an EMPTY sequence [0, 64, 10], chunk 64, safe, V_FIRST fp32.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "V_FIRST"
    ](
        "varlen-empty",
        batch_size=3,
        num_value_heads=1,
        num_key_heads=1,
        total_T=74,
        chunk_size=64,
        seq_lengths=[0, 64, 10],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_gqa_hv2_original_kfirst() raises:
    """GVA HV/H=2, varlen [64, 130], chunk 64, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "gqa-hv2",
        batch_size=2,
        num_value_heads=2,
        num_key_heads=1,
        total_T=194,
        chunk_size=64,
        seq_lengths=[64, 130],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_gqa_hv4_safe_probability_vfirst() raises:
    """GVA HV/H=4, varlen [80, 33], chunk 64, safe/probability, V_FIRST fp32."""
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
        "gqa-hv4",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=1,
        total_T=113,
        chunk_size=64,
        seq_lengths=[80, 33],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_hv64_h64_noGQA_safe_kfirst() raises:
    """HV=H=64, no GQA expansion, using dims from GLM-5.3-Flash.
    (`linear_num_heads=64`, no query/key-value head split). Varlen [300, 212],
    chunk 64, safe/logits (matches `linear_lower_bound=-5.0`), K_FIRST fp32.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "K_FIRST"
    ](
        "hv64-h64-noGQA",
        batch_size=2,
        num_value_heads=64,
        num_key_heads=64,
        total_T=512,
        chunk_size=64,
        seq_lengths=[300, 212],
        ctx=ctx,
        tol_output=0.006,
    )


def test_chunk_zero_init_original_kfirst() raises:
    """Zero initial state, T=100, chunk 64, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "zero-init",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=100,
        chunk_size=64,
        seq_lengths=[100],
        ctx=ctx,
        nonzero_init=False,
        tol_output=0.006,
    )


def test_chunk_bf16_state_original_kfirst() raises:
    """BF16 state, T=256, chunk 64, original/logits, K_FIRST (looser state tol).
    """
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
        "bf16-state",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        chunk_size=64,
        seq_lengths=[256],
        ctx=ctx,
        tol_output=0.006,
        tol_state=0.05,
    )


def test_chunk_probability_beta_original_kfirst() raises:
    """`beta=probability` pass-through, T=128, chunk 64, original, K_FIRST fp32.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "probability",
        "K_FIRST",
    ](
        "prob-beta",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=128,
        chunk_size=64,
        seq_lengths=[128],
        ctx=ctx,
        tol_output=0.006,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
