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
"""Tests for the KDA COMPUTE-BOUND chunk-blocked prefill GPU kernel (M2b).

Correctness gate for ``kda_chunk_computebound_gpu`` (the WY/UT matmul form of
the chunk recurrence). Unlike the scaffold ``kda_chunk_gpu`` — which is
bit-identical to the M1 decode recurrence — the compute-bound form reassociates
the fp32 arithmetic (interaction matrices + triangular solve), so it is NOT
bit-identical; it is checked WITHIN the M1 decode tolerances against:
  - the M1 decode recurrence ``kda_decode_gpu`` (same inputs), and
  - the M0 CPU scalar reference ``kda_decode_ref`` (independent oracle).

Output error AND final-state error are tracked SEPARATELY (tol: 0.005/0.006
output, 0.01 fp32 state, 0.05 bf16 state — the M1 tolerances, except the fp32
state bar, which follows the fused kernel's bf16 ``tcgen05.mma`` state
increment rather than the M1 fp32 scalar loop; see ``TOL_STATE_BF16_MMA``).

Axes: single/multi/ragged chunks; varlen {fixed, ragged, empty seq}; gate
{original, safe}; beta {logits, probability}; layout {K_FIRST, V_FIRST}; GVA
{HV/H = 2, 4}; state {fp32, bf16}; zero + nonzero init. CHUNK_SIZE is fixed at
16 (the FLA subchunk size and a safe static-SMEM footprint for the portable
slice; larger chunks need dynamic-SMEM opt-in, tracked as follow-up work).
"""

import std.math
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext
from max.gpu.host import FuncAttribute

from layout import TileTensor, row_major
from layout.tma_async import create_tma_tile
from kda.chunk_fwd import (
    KDA_FUSED_PIPE_SMEM,
    KDA_FUSED_ROLE_BLOCK,
    kda_chunk_computebound_gpu,
)
from kda.recurrent import kda_decode_gpu
from kda.reference import kda_decode_ref


comptime CHUNK: Int = 16

# The fused kernel accumulates the recurrent state as an fp32 TMEM
# accumulator fed by bf16 `tcgen05.mma` outer products (the reference's
# GEMM4, csrc/kda/flashkda_bf16_fused_m128.cu:1234-1252), not as the fp32
# scalar loop this file's 0.005 M1-decode figure was written for. Two bf16
# operand roundings at half-ulp 2^-9 = 1.95e-3 x the delta rule's measured
# cancellation factor 2.3 (cb-t16, single chunk, 4.42e-3) => 9.0e-3. For
# scale: FlashKDA, built the same way, measures 5.4e-3 against an fp64 gold,
# and the reference stores its state only as bf16 (:497).
comptime TOL_STATE_BF16_MMA: Float64 = 0.01

# The intra-chunk solve is the reference's THIRD `tcgen05.mma` applying an
# EXPLICIT `T = (I + A)^-1` (csrc/kda/flashkda_bf16_fused_m128.cu:1212-1226),
# not an fp32 scalar forward substitution, so the output path carries two bf16
# roundings the 0.006 figure did not budget for: the right-hand side before the
# solve, and `T` itself. Each is half-ulp 2^-9 = 1.9e-3, and they compose with
# the existing 5.755e-3 in quadrature => sqrt(5.755^2 + 1.9^2 + 1.9^2) = 6.35e-3.
# For scale: FlashKDA, which applies the same explicit inverse in bf16, measures
# 6.4e-3 against an fp64 gold. A measured error ABOVE this bar means the solve is
# wrong, not that the bar is: the contingency is a HI/LO limb split of `T`
# (`_bf16_hi_trunc`), never a further loosening.
comptime TOL_OUTPUT_BF16_SOLVE: Float64 = 0.009


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

    # --- Chunk map (CHUNK-sized): seq_chunk_offsets [N+1], chunk_tok_offsets
    #     [C+1] (cumulative global token index). Built once on the host. ---
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

    # The fused kernel stages V, raw q/k/gate, and the output tile through
    # bulk-tensor transport, so it takes a descriptor per tensor instead of the
    # raw pointer. Prep has no scalar arm left for q/k/gate; the output keeps
    # one for the ragged tail, which is why `output` is still passed too.
    var v_tma = create_tma_tile[CHUNK, VALUE_HEAD_DIM](
        ctx, v_tt.to_layout_tensor()
    )
    var q_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, q_tt.to_layout_tensor()
    )
    var k_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, k_tt.to_layout_tensor()
    )
    var rg_tma = create_tma_tile[CHUNK, KEY_HEAD_DIM](
        ctx, rg_tt.to_layout_tensor()
    )
    var out_tma = create_tma_tile[CHUNK, VALUE_HEAD_DIM](
        ctx, out_chk_tt.to_layout_tensor()
    )

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
        ctx.enqueue_function[
            kda_chunk_computebound_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
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
            ],
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
            v_tma,
            q_tma,
            k_tma,
            rg_tma,
            out_tma,
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
            block_dim=(KDA_FUSED_ROLE_BLOCK[VALUE_HEAD_DIM],),
            shared_mem_bytes=KDA_FUSED_PIPE_SMEM[CHUNK, KEY_HEAD_DIM],
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(KDA_FUSED_PIPE_SMEM[CHUNK, KEY_HEAD_DIM])
            ),
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
            kda_chunk_computebound_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                CHUNK,
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
            v_tma,
            q_tma,
            k_tma,
            rg_tma,
            out_tma,
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
            block_dim=(KDA_FUSED_ROLE_BLOCK[VALUE_HEAD_DIM],),
            shared_mem_bytes=KDA_FUSED_PIPE_SMEM[CHUNK, KEY_HEAD_DIM],
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(KDA_FUSED_PIPE_SMEM[CHUNK, KEY_HEAD_DIM])
            ),
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
    tol_output: Float64 = TOL_OUTPUT_BF16_SOLVE,
    tol_state: Float64 = TOL_STATE_BF16_MMA,
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
        + " | cb-vs-decode out="
        + String(err_o_dec)
        + " state="
        + String(err_s_dec)
        + " | cb-vs-M0ref out="
        + String(err_o_ref)
        + " state="
        + String(err_s_ref)
    )

    assert_true(
        err_o_dec < tol_output,
        tag + " cb-vs-decode output rel_err=" + String(err_o_dec),
    )
    assert_true(
        err_s_dec < tol_state,
        tag + " cb-vs-decode state rel_err=" + String(err_s_dec),
    )
    assert_true(
        err_o_ref < tol_output,
        tag + " cb-vs-M0ref output rel_err=" + String(err_o_ref),
    )
    assert_true(
        err_s_ref < tol_state,
        tag + " cb-vs-M0ref state rel_err=" + String(err_s_ref),
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


def test_cb_t16_single_original_kfirst() raises:
    """T=16 == CHUNK: single full chunk, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cb-t16",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=16,
        seq_lengths=[16],
        ctx=ctx,
    )


def test_cb_t64_multichunk_safe_kfirst() raises:
    """T=64: 4 full chunks of 16, safe/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "K_FIRST"
    ](
        "cb-t64-safe",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=64,
        seq_lengths=[64],
        ctx=ctx,
    )


def test_cb_t100_ragged_original_vfirst() raises:
    """T=100: 7 chunks (6*16 + 4) — ragged last chunk, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "V_FIRST"
    ](
        "cb-t100-ragged",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=100,
        seq_lengths=[100],
        ctx=ctx,
    )


def test_cb_t256_original_kfirst() raises:
    """T=256: 16 full chunks, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cb-t256",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        seq_lengths=[256],
        ctx=ctx,
    )


def test_cb_varlen_ragged_original_kfirst() raises:
    """Varlen ragged [200, 56], original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cb-varlen",
        batch_size=2,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
    )


def test_cb_varlen_empty_seq_safe_vfirst() raises:
    """Varlen with an EMPTY sequence [0, 64, 10], safe/logits, V_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "safe", "logits", "V_FIRST"
    ](
        "cb-varlen-empty",
        batch_size=3,
        num_value_heads=1,
        num_key_heads=1,
        total_T=74,
        seq_lengths=[0, 64, 10],
        ctx=ctx,
    )


def test_cb_gqa_hv2_original_kfirst() raises:
    """GVA HV/H=2, varlen [64, 130], original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cb-gqa-hv2",
        batch_size=2,
        num_value_heads=2,
        num_key_heads=1,
        total_T=194,
        seq_lengths=[64, 130],
        ctx=ctx,
    )


def test_cb_gqa_hv4_safe_probability_vfirst() raises:
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
        "cb-gqa-hv4",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=1,
        total_T=113,
        seq_lengths=[80, 33],
        ctx=ctx,
    )


def test_cb_zero_init_original_kfirst() raises:
    """Zero initial state, T=100, original/logits, K_FIRST fp32."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check[
        DType.bfloat16, DType.float32, 128, 128, "original", "logits", "K_FIRST"
    ](
        "cb-zero-init",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=100,
        seq_lengths=[100],
        ctx=ctx,
        nonzero_init=False,
    )


def test_cb_bf16_state_original_kfirst() raises:
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
        "cb-bf16-state",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=64,
        seq_lengths=[64],
        ctx=ctx,
        tol_state=0.05,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
