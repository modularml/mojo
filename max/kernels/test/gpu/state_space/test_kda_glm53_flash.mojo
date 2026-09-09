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
"""GLM-5.3-Flash KDA correctness + dispatch suite.

Exercises the SM100 prefill kernels from PR #97619 at the GLM-5.3-Flash
linear-attention geometry: H == HV == 64, K == V == 128, no GQA, the safe
forget gate (gate_mode="safe", lower_bound = -5.0 == GLM's
`linear_lower_bound`), beta_mode="logits", state layout V_FIRST, bf16 q/k/v/g
and state, fp32 A_log/dt_bias. The prefill kernels bake safe/logits and take
`lower_bound` as a runtime float, so GLM is "-5.0" at the call site --- no
gate-mode axis here.

Two prefill kernels exist because H=64 batch=1 starves the M128 grid (64 CTAs
on a 148-SM B200); the M64 variant value-splits to 2 CTAs/head. This suite
therefore routes batch=1 to `sm100_kda_prefill_m64` and batch>1 to
`sm100_kda_prefill` (M128), matching PR #97619's best-route-per-shape table.

The harness mirrors `test_sm100_kda_prefill_m128.mojo` (PrefillInputs +
_run_prefill + _run_reference + _check_case), reused for the M128 cases, plus
a `_run_m64_prefill`/`_m64_check_case` for batch=1 and a new
`_dispatch_check` that runs a prefill kernel and the decode kernel
(`kda_decode_gpu`, gate_mode="safe" V_FIRST) token-by-token on identical
inputs and asserts the two outputs/states agree --- the kernel-side proxy for
the layer's T>1 -> prefill / T=1 -> decode dispatch.

Oracle vs chunk-vs-decode budget: the fp64 oracle (`kda_decode_ref`) is
O(T*H*K*V) per head and ~32x the H=2 oracle the M128 suite already runs. It
backs only the small Group A cases and the one prime-varlen case; larger
real prefill shapes (Group B) and the batch-8 prime case (Group C) gate on
chunk-vs-decode (the decode kernel is GPU and is the un-reassociated
recurrence, so it independently catches the chunk reassociation).

Raggedness comes from prime sequence lengths: primes are not multiples of
32 (the chunk size), so every prime-length sequence has a non-trivial tail
that exercises the zero-filled cp.async loads and the scalar-store epilogue.
Primes picked to span tail sizes mod 32: 97 (tail 1), 127 (tail 31, the max
just under a chunk), 101/89/113; mixed batches include a multiple of 32 as
an in-batch exact-tail oracle anchor.

Out of scope (by kernel contract, not test gaps): gate_mode="original" (K3
softplus) and beta_mode="probability" --- the SM100 prefill kernels bake
safe/logits; GQA (H != HV) --- H == HV by contract; FP32 state --- state is
bf16 by contract (the decode helper uses fp32 state only because the
decode kernel's bf16-state path ptxas-fails at this geometry, and the
comparison promotes both sides to fp32); the BT=16 decomposition kernels
(`sm100_kda_prep_bt16`/`chain_bt16`) --- a separate prefill path, not needed
for GLM correctness; the graph-compiler `kda_chunk` op --- still unregistered,
so this suite calls the kernels directly (a layer-bring-up gap, part 2).
"""

from std.io import print
from std.math import sqrt
from std.memory import alloc, bitcast
from std.random import random_float64, seed
from std.testing import TestSuite, assert_true
from std.utils import IndexList

from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor

from kda.recurrent import kda_decode_gpu
from kda.reference import kda_decode_ref
from kda.sm100_kda_prefill_m128 import (
    BetaTmaTile as M128BetaTmaTile,
    QkOutTmaTile as M128QkOutTmaTile,
    SMEM_TOTAL as M128_SMEM_TOTAL,
    VgTmaTile as M128VgTmaTile,
    sm100_kda_prefill,
)
from kda.sm100_kda_prefill_m64 import (
    BetaTmaTile as M64BetaTmaTile,
    OutTmaTile as M64OutTmaTile,
    QkTmaTile as M64QkTmaTile,
    SMEM_TOTAL as M64_SMEM_TOTAL,
    VTmaTile as M64VTmaTile,
    GateTmaTile as M64GateTmaTile,
    sm100_kda_prefill_m64,
)

# GLM-5.3-Flash linear-attention geometry (all comptime so kernel comptime
# slots and stride arithmetic get named values, not bare literals). The SM100
# prefill kernels are D == 128-specialized; the TMA descriptor box/swizzle
# integers (32, 8, 64, 128 trailing the IndexLists) are kernel-schedule
# constants tied to the swizzle mode, not geometry --- left literal.
comptime H: Int = 64  # query/key head count
comptime HV: Int = 64  # value head count (== H; no GQA)
comptime K: Int = 128  # key/query head dim
comptime V: Int = 128  # value head dim
comptime D: Int = 128  # head dim (K == V for KDA)
comptime HK: Int = H * K  # q/k per-token row stride
comptime HVK: Int = HV * K  # gate per-token row stride
comptime HVV: Int = HV * V  # v/output per-token row stride
comptime STATE_TILE: Int = HV * K * V  # one head's state [h, v, k]
comptime STATE_HEAD: Int = K * V  # state head stride

comptime BF16 = Scalar[DType.bfloat16]
comptime LOWER_BOUND: Float32 = -5.0
comptime SCALE: Float32 = 0.08838834764831845  # 1/sqrt(D)
comptime OUT_TOL = 0.01
comptime STATE_TOL = 0.02


def _rel_err(
    want: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    got: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float64:
    """RMSE(want - got) / RMSE(want)."""
    var r_sq = Float64(0.0)
    var d_sq = Float64(0.0)
    for i in range(n):
        var w = Float64(want[i])
        var g = Float64(got[i])
        r_sq = r_sq + w * w
        var diff = w - g
        d_sq = d_sq + diff * diff
    return sqrt(d_sq) / (sqrt(r_sq) + 1e-8)


def _fill_bf16(
    p: UnsafePointer[BF16, MutUntrackedOrigin], n: Int, lo: Float64, hi: Float64
):
    for i in range(n):
        p[i] = BF16(Float32(lo + (hi - lo) * random_float64()))


def _fill_f32(
    p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
    lo: Float64,
    hi: Float64,
):
    for i in range(n):
        p[i] = Float32(lo + (hi - lo) * random_float64())


def _to_f32(
    src: UnsafePointer[BF16, MutUntrackedOrigin], n: Int
) -> UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]:
    var out = alloc[Scalar[DType.float32]](n)
    for i in range(n):
        out[i] = Float32(src[i])
    return out


# ===----------------------------------------------------------------------=== #
# Host inputs (general: varlen M128; the M64 helper builds its own single-seq
# inputs since the M64 kernel has no state_indices / checkpoint support).
# ===----------------------------------------------------------------------=== #


struct M128Inputs:
    """Host-side inputs for one M128 (varlen) prefill case. H carried by runner.
    """

    var num_seqs: Int
    var t_total: Int
    var num_slots: Int
    var q: UnsafePointer[BF16, MutUntrackedOrigin]
    var k: UnsafePointer[BF16, MutUntrackedOrigin]
    var v: UnsafePointer[BF16, MutUntrackedOrigin]
    var g: UnsafePointer[BF16, MutUntrackedOrigin]
    var beta: UnsafePointer[BF16, MutUntrackedOrigin]
    var alog: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var dt: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var cu: UnsafePointer[Scalar[DType.int64], MutUntrackedOrigin]
    var state0: UnsafePointer[BF16, MutUntrackedOrigin]
    var state_indices: UnsafePointer[Scalar[DType.int32], MutUntrackedOrigin]
    var ckpt_starts: UnsafePointer[Scalar[DType.int64], MutUntrackedOrigin]

    def __init__(
        out self, seq_lengths: List[Int], num_slots: Int, num_ckpts: Int
    ):
        self.num_seqs = len(seq_lengths)
        self.t_total = 0
        for i in range(self.num_seqs):
            self.t_total += seq_lengths[i]
        self.num_slots = num_slots
        _ = num_ckpts
        var n_qkv = self.t_total * HK
        var n_state = num_slots * STATE_TILE
        self.q = alloc[BF16](n_qkv)
        self.k = alloc[BF16](n_qkv)
        self.v = alloc[BF16](n_qkv)
        self.g = alloc[BF16](n_qkv)
        self.beta = alloc[BF16](self.t_total * HV)
        self.alog = alloc[Scalar[DType.float32]](HV)
        self.dt = alloc[Scalar[DType.float32]](HVK)
        self.cu = alloc[Scalar[DType.int64]](self.num_seqs + 1)
        self.state0 = alloc[BF16](n_state)
        self.state_indices = alloc[Scalar[DType.int32]](self.num_seqs)
        self.ckpt_starts = alloc[Scalar[DType.int64]](self.num_seqs)
        _fill_bf16(self.q, n_qkv, -1.0, 1.0)
        _fill_bf16(self.k, n_qkv, -1.0, 1.0)
        _fill_bf16(self.v, n_qkv, -1.0, 1.0)
        _fill_bf16(self.g, n_qkv, -1.0, 1.0)
        _fill_bf16(self.beta, self.t_total * HV, -2.0, 2.0)
        _fill_f32(self.alog, HV, -1.0, 0.0)
        _fill_f32(self.dt, HVK, -0.1, 0.1)
        for i in range(n_state):
            self.state0[i] = BF16(0.0)
        var cum = 0
        self.cu[0] = Int64(0)
        for i in range(self.num_seqs):
            cum += seq_lengths[i]
            self.cu[i + 1] = Int64(cum)
        for i in range(self.num_seqs):
            self.state_indices[i] = Int32(i)
            self.ckpt_starts[i] = Int64(0)

    def __deinit__(deinit self):
        self.q.free()
        self.k.free()
        self.v.free()
        self.g.free()
        self.beta.free()
        self.alog.free()
        self.dt.free()
        self.cu.free()
        self.state0.free()
        self.state_indices.free()
        self.ckpt_starts.free()


# ===----------------------------------------------------------------------=== #
# M128 prefill run + reference + check (varlen, H=64).
# ===----------------------------------------------------------------------=== #


def _run_m128_prefill(
    ctx: DeviceContext,
    inp: M128Inputs,
    *,
    use_initial_state: Bool,
    store_final_state: Bool,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    var num_seqs = inp.num_seqs
    var t_total = inp.t_total
    var n_qkv = t_total * HK
    var n_state = inp.num_slots * STATE_TILE
    comptime padded_heads = (64 + 7) // 8 * 8  # 64, already 8-aligned
    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * H)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](H)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](HVK)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](num_seqs + 1)
    var d_seqorder = ctx.enqueue_create_buffer[DType.int32](num_seqs)
    var d_state0 = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_state_final = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_si = ctx.enqueue_create_buffer[DType.int32](num_seqs)
    var d_out = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_ckpt = ctx.enqueue_create_buffer[DType.bfloat16](
        num_seqs * STATE_TILE
    )
    var d_ckpt_starts = ctx.enqueue_create_buffer[DType.int64](num_seqs)

    ctx.enqueue_copy(d_q, inp.q)
    ctx.enqueue_copy(d_k, inp.k)
    ctx.enqueue_copy(d_v, inp.v)
    ctx.enqueue_copy(d_g, inp.g)
    ctx.enqueue_copy(d_beta, inp.beta)
    ctx.enqueue_copy(d_alog, inp.alog)
    ctx.enqueue_copy(d_dt, inp.dt)
    ctx.enqueue_copy(d_cu, inp.cu)
    ctx.enqueue_copy(d_state0, inp.state0)
    ctx.enqueue_copy(d_si, inp.state_indices)
    ctx.enqueue_copy(d_ckpt_starts, inp.ckpt_starts)
    var final_init_h = alloc[BF16](n_state)
    for i in range(n_state):
        final_init_h[i] = BF16(0.0)
    ctx.enqueue_copy(d_state_final, final_init_h)
    var ckpt_init_h = alloc[BF16](num_seqs * STATE_TILE)
    for i in range(num_seqs * STATE_TILE):
        ckpt_init_h[i] = BF16(0.0)
    ctx.enqueue_copy(d_ckpt, ckpt_init_h)
    var seqorder_h = alloc[Scalar[DType.int32]](num_seqs)
    for i in range(num_seqs):
        seqorder_h[i] = Int32(i)
    ctx.enqueue_copy(d_seqorder, seqorder_h)

    comptime h128 = HK
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_v,
        IndexList[3](t_total, 64, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_g,
        IndexList[3](t_total, 64, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var beta_desc = create_tma_descriptor[
        DType.bfloat16, 2, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_beta,
        IndexList[2](t_total, padded_heads),
        IndexList[2](padded_heads, 1),
        IndexList[2](32, 8),
    )
    var out_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),  # M128: box dim0=2 (compute+epilogue pair)
    )
    var q_tma = M128QkOutTmaTile(q_desc)
    var k_tma = M128QkOutTmaTile(k_desc)
    var v_tma = M128VgTmaTile(v_desc)
    var g_tma = M128VgTmaTile(g_desc)
    var beta_tma = M128BetaTmaTile(beta_desc)
    var out_tma = M128QkOutTmaTile(out_desc)

    var q_t = TileTensor[linear_idx_type=DType.int32](
        d_q.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var k_t = TileTensor[linear_idx_type=DType.int32](
        d_k.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var v_t = TileTensor[linear_idx_type=DType.int32](
        d_v.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var g_t = TileTensor[linear_idx_type=DType.int32](
        d_g.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var out_tt = TileTensor[linear_idx_type=DType.int32](
        d_out.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var beta_t = TileTensor[linear_idx_type=DType.int32](
        d_beta.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV])),
    )
    var alog_t = TileTensor[linear_idx_type=DType.int32](
        d_alog.unsafe_ptr().as_unsafe_any_origin(), row_major(Idx[HV])
    )
    var dt_t = TileTensor[linear_idx_type=DType.int32](
        d_dt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[HV], Idx[D])),
    )
    var cu_t = TileTensor[linear_idx_type=DType.int32](
        d_cu.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs + 1)
    )
    var so_t = TileTensor[linear_idx_type=DType.int32](
        d_seqorder.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs)
    )
    var si_t = TileTensor[linear_idx_type=DType.int32](
        d_si.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs)
    )
    var state0_t = TileTensor[linear_idx_type=DType.int32](
        d_state0.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(inp.num_slots, Idx[HV], Idx[D], Idx[D])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_state_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(inp.num_slots, Idx[HV], Idx[D], Idx[D])),
    )
    var ckpt_t = TileTensor[linear_idx_type=DType.int32](
        d_ckpt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(num_seqs, Idx[HV], Idx[D], Idx[D])),
    )
    var ckpt_starts_t = TileTensor[linear_idx_type=DType.int32](
        d_ckpt_starts.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs)
    )

    comptime kernel = sm100_kda_prefill[
        H,
        QkvLayout=type_of(q_t).LayoutType,
        BetaLayout=type_of(beta_t).LayoutType,
        HVecLayout=type_of(alog_t).LayoutType,
        DtLayout=type_of(dt_t).LayoutType,
        I64Layout=type_of(cu_t).LayoutType,
        I32Layout=type_of(so_t).LayoutType,
        StateLayout=type_of(state0_t).LayoutType,
        TensorOrigin=type_of(q_t).origin,
    ]
    ctx.enqueue_function[kernel](
        q_t,
        q_tma,
        k_t,
        k_tma,
        v_t,
        v_tma,
        g_t,
        g_tma,
        beta_t,
        beta_tma,
        alog_t,
        dt_t,
        cu_t,
        so_t,
        state0_t,
        out_tt,
        out_tma,
        final_t,
        Int32(1) if use_initial_state else Int32(0),
        Int32(1) if store_final_state else Int32(0),
        SCALE,
        LOWER_BOUND,
        si_t,
        ckpt_t,
        ckpt_starts_t,
        Int32(1),  # use_state_indices
        Int32(0),  # checkpoint_every_n_tokens
        grid_dim=(num_seqs * HV,),
        block_dim=(1024,),
        shared_mem_bytes=M128_SMEM_TOTAL,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(M128_SMEM_TOTAL)
        ),
    )

    var out_bf16_h = alloc[BF16](n_qkv)
    var final_bf16_h = alloc[BF16](n_state)
    ctx.enqueue_copy(out_bf16_h, d_out)
    ctx.enqueue_copy(final_bf16_h, d_state_final)
    ctx.synchronize()
    var out_h = _to_f32(out_bf16_h, n_qkv)
    var final_h = _to_f32(final_bf16_h, n_state)
    out_bf16_h.free()
    final_bf16_h.free()
    final_init_h.free()
    ckpt_init_h.free()
    seqorder_h.free()
    return (out_h, final_h)


def _run_m128_reference(
    inp: M128Inputs,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    var H = 64
    var n_qkv = inp.t_total * HK
    var n_state = inp.num_slots * STATE_TILE
    var tile = STATE_TILE
    var q_f = _to_f32(inp.q, n_qkv)
    var k_f = _to_f32(inp.k, n_qkv)
    var v_f = _to_f32(inp.v, n_qkv)
    var g_f = _to_f32(inp.g, n_qkv)
    var beta_f = _to_f32(inp.beta, inp.t_total * H)
    var out_f = alloc[Scalar[DType.float32]](n_qkv)
    var state_f = _to_f32(inp.state0, n_state)
    for s in range(inp.num_seqs):
        var seq_start = Int(inp.cu[s])
        var seq_end = Int(inp.cu[s + 1])
        var slot = Int(inp.state_indices[s])
        var slot_i32 = alloc[Scalar[DType.int32]](1)
        slot_i32[0] = Int32(slot)
        var window = alloc[Scalar[DType.int32]](2)
        window[0] = Int32(seq_start)
        window[1] = Int32(seq_end)
        kda_decode_ref[
            DType.float32,
            DType.float32,
            DType.float32,
            DType.float32,
            gate_mode="safe",
            beta_mode="logits",
            state_layout="V_FIRST",
        ](
            1,
            H,
            H,
            K,
            V,
            out_f,
            q_f,
            k_f,
            v_f,
            g_f,
            beta_f,
            inp.alog,
            inp.dt,
            window,
            state_f,
            slot_i32,
            H * 128,
            128,
            1,  # q
            H * 128,
            128,
            1,  # k
            H * 128,
            128,
            1,  # v
            H * 128,
            128,
            1,  # g
            H,
            1,  # beta
            128,
            1,  # dt_bias
            tile,
            128 * 128,
            128,
            1,  # state [slot,h,v,k]
            H * 128,
            128,
            1,  # out
        )
        window.free()
        slot_i32.free()
    q_f.free()
    k_f.free()
    v_f.free()
    g_f.free()
    beta_f.free()
    return (out_f, state_f)


def _m128_check(name: String, ctx: DeviceContext, inp: M128Inputs) raises:
    var got = _run_m128_prefill(
        ctx, inp, use_initial_state=False, store_final_state=True
    )
    var want = _run_m128_reference(inp)
    var n_qkv = inp.t_total * HK
    var tile = STATE_TILE
    var e_out = _rel_err(want[0], got[0], n_qkv)
    var e_state = Float64(0.0)
    for s in range(inp.num_seqs):
        var slot = Int(inp.state_indices[s])
        var e = _rel_err(want[1] + slot * tile, got[1] + slot * tile, tile)
        e_state = max(e_state, e)
    print(name, ": rel_err out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": final state rel_err too high")
    got[0].free()
    got[1].free()
    want[0].free()
    want[1].free()


# ===----------------------------------------------------------------------=== #
# M64 prefill run + reference + check (single-seq, H=64 baked).
# ===----------------------------------------------------------------------=== #


struct M64Inputs:
    """Host-side inputs for one single-sequence M64 prefill case (H=64 baked).
    """

    var t_total: Int
    var q: UnsafePointer[BF16, MutUntrackedOrigin]
    var k: UnsafePointer[BF16, MutUntrackedOrigin]
    var v: UnsafePointer[BF16, MutUntrackedOrigin]
    var g: UnsafePointer[BF16, MutUntrackedOrigin]
    var beta: UnsafePointer[BF16, MutUntrackedOrigin]
    var alog: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var dt: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var cu: UnsafePointer[Scalar[DType.int64], MutUntrackedOrigin]
    var state0: UnsafePointer[BF16, MutUntrackedOrigin]

    def __init__(out self, t_total: Int, *, init_state: Bool = False):
        self.t_total = t_total
        var n_qkv = t_total * HK
        var n_state = STATE_TILE
        self.q = alloc[BF16](n_qkv)
        self.k = alloc[BF16](n_qkv)
        self.v = alloc[BF16](n_qkv)
        self.g = alloc[BF16](n_qkv)
        self.beta = alloc[BF16](t_total * HV)
        self.alog = alloc[Scalar[DType.float32]](HV)
        self.dt = alloc[Scalar[DType.float32]](HVK)
        self.cu = alloc[Scalar[DType.int64]](2)
        self.state0 = alloc[BF16](n_state)
        _fill_bf16(self.q, n_qkv, -1.0, 1.0)
        _fill_bf16(self.k, n_qkv, -1.0, 1.0)
        _fill_bf16(self.v, n_qkv, -1.0, 1.0)
        _fill_bf16(self.g, n_qkv, -1.0, 1.0)
        _fill_bf16(self.beta, t_total * HV, -2.0, 2.0)
        _fill_f32(self.alog, HV, -1.0, 0.0)
        _fill_f32(self.dt, HVK, -0.1, 0.1)
        if init_state:
            _fill_bf16(self.state0, n_state, -0.5, 0.5)
        else:
            for i in range(n_state):
                self.state0[i] = BF16(0.0)
        self.cu[0] = Int64(0)
        self.cu[1] = Int64(t_total)

    def __deinit__(deinit self):
        self.q.free()
        self.k.free()
        self.v.free()
        self.g.free()
        self.beta.free()
        self.alog.free()
        self.dt.free()
        self.cu.free()
        self.state0.free()


def _run_m64_prefill(
    ctx: DeviceContext,
    inp: M64Inputs,
    *,
    use_initial_state: Bool,
    store_final_state: Bool,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    var t_total = inp.t_total
    var n_qkv = t_total * HK
    var n_state = STATE_TILE
    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * HV)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](HV)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](HVK)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](2)
    var d_seqorder = ctx.enqueue_create_buffer[DType.int32](1)
    var d_state0 = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_state_final = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_out = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    ctx.enqueue_copy(d_q, inp.q)
    ctx.enqueue_copy(d_k, inp.k)
    ctx.enqueue_copy(d_v, inp.v)
    ctx.enqueue_copy(d_g, inp.g)
    ctx.enqueue_copy(d_beta, inp.beta)
    ctx.enqueue_copy(d_alog, inp.alog)
    ctx.enqueue_copy(d_dt, inp.dt)
    ctx.enqueue_copy(d_cu, inp.cu)
    ctx.enqueue_copy(d_state0, inp.state0)
    var final_init_h = alloc[BF16](n_state)
    for i in range(n_state):
        final_init_h[i] = BF16(0.0)
    ctx.enqueue_copy(d_state_final, final_init_h)
    var seqorder_h = alloc[Scalar[DType.int32]](1)
    seqorder_h[0] = Int32(0)
    ctx.enqueue_copy(d_seqorder, seqorder_h)

    comptime h128 = HK
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_v,
        IndexList[3](t_total, 64, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 64),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_g,
        IndexList[3](t_total, 64, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var beta_desc = create_tma_descriptor[
        DType.bfloat16, 2, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_beta,
        IndexList[2](t_total, 64),
        IndexList[2](64, 1),
        IndexList[2](32, 8),
    )
    var out_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](2, 64, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](1, 1, 32, 64),
    )
    var q_tma = M64QkTmaTile(q_desc)
    var k_tma = M64QkTmaTile(k_desc)
    var v_tma = M64VTmaTile(v_desc)
    var g_tma = M64GateTmaTile(g_desc)
    var beta_tma = M64BetaTmaTile(beta_desc)
    var out_tma = M64OutTmaTile(out_desc)

    var q_t = TileTensor[linear_idx_type=DType.int32](
        d_q.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var k_t = TileTensor[linear_idx_type=DType.int32](
        d_k.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var v_t = TileTensor[linear_idx_type=DType.int32](
        d_v.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var g_t = TileTensor[linear_idx_type=DType.int32](
        d_g.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var out_tt = TileTensor[linear_idx_type=DType.int32](
        d_out.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV], Idx[D])),
    )
    var beta_t = TileTensor[linear_idx_type=DType.int32](
        d_beta.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[HV])),
    )
    var alog_t = TileTensor[linear_idx_type=DType.int32](
        d_alog.unsafe_ptr().as_unsafe_any_origin(), row_major(Idx[HV])
    )
    var dt_t = TileTensor[linear_idx_type=DType.int32](
        d_dt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[HV], Idx[D])),
    )
    var cu_t = TileTensor[linear_idx_type=DType.int32](
        d_cu.unsafe_ptr().as_unsafe_any_origin(), row_major(2)
    )
    var so_t = TileTensor[linear_idx_type=DType.int32](
        d_seqorder.unsafe_ptr().as_unsafe_any_origin(), row_major(1)
    )
    var state0_t = TileTensor[linear_idx_type=DType.int32](
        d_state0.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[HV], Idx[D], Idx[D])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_state_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[HV], Idx[D], Idx[D])),
    )

    comptime kernel = sm100_kda_prefill_m64[
        H,
        QkvLayout=type_of(q_t).LayoutType,
        BetaLayout=type_of(beta_t).LayoutType,
        HVecLayout=type_of(alog_t).LayoutType,
        DtLayout=type_of(dt_t).LayoutType,
        I64Layout=type_of(cu_t).LayoutType,
        I32Layout=type_of(so_t).LayoutType,
        StateLayout=type_of(state0_t).LayoutType,
        TensorOrigin=type_of(q_t).origin,
    ]
    ctx.enqueue_function[kernel](
        q_t,
        q_tma,
        k_t,
        k_tma,
        v_t,
        v_tma,
        g_t,
        g_tma,
        beta_t,
        beta_tma,
        alog_t,
        dt_t,
        cu_t,
        so_t,
        state0_t,
        out_tt,
        out_tma,
        final_t,
        Int32(1) if use_initial_state else Int32(0),
        Int32(1) if store_final_state else Int32(0),
        SCALE,
        LOWER_BOUND,
        grid_dim=(2 * HV,),
        block_dim=(1024,),
        shared_mem_bytes=M64_SMEM_TOTAL,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(M64_SMEM_TOTAL)
        ),
    )

    var out_bf16_h = alloc[BF16](n_qkv)
    var final_bf16_h = alloc[BF16](n_state)
    ctx.enqueue_copy(out_bf16_h, d_out)
    ctx.enqueue_copy(final_bf16_h, d_state_final)
    ctx.synchronize()
    var out_h = _to_f32(out_bf16_h, n_qkv)
    var final_h = _to_f32(final_bf16_h, n_state)
    out_bf16_h.free()
    final_bf16_h.free()
    final_init_h.free()
    seqorder_h.free()
    return (out_h, final_h)


def _run_m64_reference(
    inp: M64Inputs,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    var n_qkv = inp.t_total * HK
    var n_state = STATE_TILE
    var q_f = _to_f32(inp.q, n_qkv)
    var k_f = _to_f32(inp.k, n_qkv)
    var v_f = _to_f32(inp.v, n_qkv)
    var g_f = _to_f32(inp.g, n_qkv)
    var beta_f = _to_f32(inp.beta, inp.t_total * HV)
    var out_f = alloc[Scalar[DType.float32]](n_qkv)
    var state_f = _to_f32(inp.state0, n_state)
    var window = alloc[Scalar[DType.int32]](2)
    window[0] = Int32(0)
    window[1] = Int32(inp.t_total)
    var slot_i32 = alloc[Scalar[DType.int32]](1)
    slot_i32[0] = Int32(0)
    kda_decode_ref[
        DType.float32,
        DType.float32,
        DType.float32,
        DType.float32,
        gate_mode="safe",
        beta_mode="logits",
        state_layout="V_FIRST",
    ](
        1,
        HV,
        H,
        K,
        V,
        out_f,
        q_f,
        k_f,
        v_f,
        g_f,
        beta_f,
        inp.alog,
        inp.dt,
        window,
        state_f,
        slot_i32,
        HK,
        K,
        1,
        HK,
        K,
        1,
        HK,
        K,
        1,
        HK,
        K,
        1,
        HV,
        1,
        K,
        1,
        STATE_TILE,
        STATE_HEAD,
        K,
        1,
        HVV,
        V,
        1,
    )
    window.free()
    slot_i32.free()
    q_f.free()
    k_f.free()
    v_f.free()
    g_f.free()
    beta_f.free()
    return (out_f, state_f)


def _m64_check(name: String, ctx: DeviceContext, inp: M64Inputs) raises:
    var got = _run_m64_prefill(
        ctx, inp, use_initial_state=False, store_final_state=True
    )
    var want = _run_m64_reference(inp)
    var n_qkv = inp.t_total * HK
    var n_state = STATE_TILE
    var e_out = _rel_err(want[0], got[0], n_qkv)
    var e_state = _rel_err(want[1], got[1], n_state)
    print(name, ": rel_err out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": final state rel_err too high")
    got[0].free()
    got[1].free()
    want[0].free()
    want[1].free()


# ===----------------------------------------------------------------------=== #
# Decode-only runner: `kda_decode_gpu` (gate_mode="safe", V_FIRST). Runs one
# cu_seqlens window per launch on the same pool, token by token for the
# dispatch check. Returns fp32 host copies of (output, final-state).
# ===----------------------------------------------------------------------=== #


def _run_decode[
    H: Int = 64,  # GLM: H == HV == 64, K == V == 128
](
    ctx: DeviceContext,
    inp: M128Inputs,
    *,
    window_starts: List[Int],
    total_T: Int,
    per_seq: Bool,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Run `kda_decode_gpu` over the given packed-token windows on the inp's
    state pool, one launch per window (state carried in the pool across
    launches). For a single token, window_starts = [0,1,2,...] token-by-token;
    for prefill-vs-decode on a whole batch, one window per sequence.

    GLM geometry only: H == HV == 64, K == V == 128 (comptime here so the
    decode-kernel's comptime slots get literals, not runtime vars)."""
    comptime assert H == 64, "_run_decode is specialized to GLM H == 64"
    var HV = 64
    var n_state = inp.num_slots * STATE_TILE
    var n_out = total_T * HVV
    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](total_T * HK)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](total_T * HK)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](total_T * HVV)
    # kda_decode_gpu takes q/k/v at qkv_dtype (bf16) but raw_gate and beta_logits
    # at gate_dtype (fp32) --- unlike the M128 prefill kernel, which takes g/
    # beta at bf16 (its QkvLayout). Promote the bf16 inputs to fp32 here.
    var d_g = ctx.enqueue_create_buffer[DType.float32](total_T * HVK)
    var d_beta = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](HV)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](HVK)
    var d_cu = ctx.enqueue_create_buffer[DType.int32](2)
    # window_starts / per_seq are legacy of an earlier whole-seq-window scheme;
    # _run_decode now always decodes token-by-token (the real T=1 dispatch path),
    # routing each token's state to its owning sequence's slot via inp.cu. Kept
    # in the signature so callers need not change.
    _ = window_starts
    _ = per_seq
    var d_si = ctx.enqueue_create_buffer[DType.int32](1)
    var d_pool = ctx.enqueue_create_buffer[DType.float32](n_state)
    var d_out = ctx.enqueue_create_buffer[DType.float32](n_out)
    ctx.enqueue_copy(d_q, inp.q)
    ctx.enqueue_copy(d_k, inp.k)
    ctx.enqueue_copy(d_v, inp.v)
    # g/beta are bf16 in inp; promote to fp32 for the decode kernel's gate_dtype.
    var g_f32_h = _to_f32(inp.g, total_T * HVK)
    var beta_f32_h = _to_f32(inp.beta, total_T * HV)
    ctx.enqueue_copy(d_g, g_f32_h)
    ctx.enqueue_copy(d_beta, beta_f32_h)
    g_f32_h.free()
    beta_f32_h.free()
    ctx.enqueue_copy(d_alog, inp.alog)
    ctx.enqueue_copy(d_dt, inp.dt)
    # state pool init: copy inp.state0 (already layout [slot,h,v,k] V_FIRST).
    # state pool init: copy inp.state0 (bf16) promoted to fp32, so the decode
    # kernel's fp32 state pool starts from the same values the prefill saw.
    var state0_f32_h = _to_f32(inp.state0, n_state)
    ctx.enqueue_copy(d_pool, state0_f32_h)
    state0_f32_h.free()
    var si_h = alloc[Scalar[DType.int32]](1)
    # state_indices[0] is rewritten per window so seq w writes to slot w (the
    # decode kernel is batch_size=1 per launch; the prefill scatters into
    # slots 0..N-1, and the dispatch check compares per-slot states, so each
    # window's final state must land in its own slot, not all in slot 0).
    si_h[0] = Int32(0)
    ctx.enqueue_copy(d_si, si_h)

    var q_tt = TileTensor(d_q, row_major(total_T, HK))
    var k_tt = TileTensor(d_k, row_major(total_T, HK))
    var v_tt = TileTensor(d_v, row_major(total_T, HVV))
    var g_tt = TileTensor(d_g, row_major(total_T, HVK))
    var bl_tt = TileTensor(d_beta, row_major(total_T, HV))
    var al_tt = TileTensor(d_alog, row_major(64))
    var dt_tt = TileTensor(d_dt, row_major(HV, K))
    var cu_tt = TileTensor(d_cu, row_major(2))
    var si_tt = TileTensor(d_si, row_major(1))
    var pool_tt = TileTensor(
        d_pool, row_major(inp.num_slots, HV, V, K)  # V_FIRST: [slot,h,v,k]
    )
    var out_tt = TileTensor(d_out, row_major(total_T, HK))
    # zero the output (kda_decode_gpu accumulates into it); DeviceBuffer has
    # enqueue_fill, not the TileTensor.
    d_out.enqueue_fill(0.0)

    # Token-by-token decode: one window per absolute token [t, t+1), routed
    # to the slot of the sequence that owns token t (via inp.cu). This matches
    # the real T=1 decode dispatch and avoids the whole-sequence-window path.
    # (prefill vs decode agree on the recurrence; the per-token path is what
    # the layer actually runs at decode time.)
    var cu_h = alloc[Scalar[DType.int32]](2)
    for t in range(total_T):
        cu_h[0] = Int32(t)
        cu_h[1] = Int32(t + 1)
        ctx.enqueue_copy(d_cu, cu_h)
        # find which sequence owns token t (largest s with cu[s] <= t).
        var slot = 0
        for s in range(inp.num_seqs):
            if Int(inp.cu[s]) <= t:
                slot = s
            else:
                break
        si_h[0] = Int32(slot)
        ctx.enqueue_copy(d_si, si_h)
        ctx.enqueue_function[
            kda_decode_gpu[
                DType.bfloat16,
                DType.float32,
                DType.float32,
                DType.float32,
                K,
                V,
                type_of(out_tt).LayoutType,
                type_of(q_tt).LayoutType,
                type_of(k_tt).LayoutType,
                type_of(v_tt).LayoutType,
                type_of(g_tt).LayoutType,
                type_of(bl_tt).LayoutType,
                type_of(al_tt).LayoutType,
                type_of(dt_tt).LayoutType,
                type_of(cu_tt).LayoutType,
                type_of(pool_tt).LayoutType,
                type_of(si_tt).LayoutType,
                "safe",
                "logits",
                "V_FIRST",
            ]
        ](
            Int32(1),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            g_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt,
            si_tt,
            UInt32(HK),
            UInt32(K),
            UInt32(1),  # q       [t, H, K]
            UInt32(HK),
            UInt32(K),
            UInt32(1),  # k       [t, H, K]
            UInt32(HVV),
            UInt32(V),
            UInt32(1),  # v       [t, HV, V]
            UInt32(HVK),
            UInt32(K),
            UInt32(1),  # g       [t, HV, K]
            UInt32(HV),
            UInt32(1),  # beta    [t, HV]
            UInt32(K),
            UInt32(1),  # dt_bias [HV, K]
            UInt32(STATE_TILE),
            UInt32(STATE_HEAD),
            UInt32(V),
            UInt32(1),  # state V_FIRST
            UInt32(HVV),
            UInt32(V),
            UInt32(1),  # out     [t, HV, V]
            UInt32(1),  # state_indices_seq_stride
            grid_dim=(HV,),
            block_dim=(V,),
        )
    ctx.synchronize()
    var out_h = alloc[Scalar[DType.float32]](n_out)
    var pool_h = alloc[Scalar[DType.float32]](n_state)
    ctx.enqueue_copy(out_h, d_out)
    ctx.enqueue_copy(pool_h, d_pool)
    ctx.synchronize()
    cu_h.free()
    si_h.free()
    return (out_h, pool_h)


# ===----------------------------------------------------------------------=== #
# Dispatch: prefill kernel vs `kda_decode_gpu` run token-by-token, on identical
# inputs from the same init state. The two implement the same safe gate, so the
# outputs and final states must agree. This is the kernel-side proxy for the
# layer's T>1 -> prefill / T=1 -> decode dispatch.
#
# `kda_decode_gpu` writes output at the ABSOLUTE token index
# (flat_token_idx * out_seqlen_stride, where flat_token_idx = cu_seqlens[0]),
# so the decode output buffer is sized to the whole sequence; per-token
# windows are [t, t+1).
#
# The prefill kernel writes bf16 state/output; the decode kernel (here) writes
# bf16 state and fp32 output. The dispatch check compares the prefill output
# (promoted to fp32) against the decode fp32 output, and the prefill bf16
# state (promoted) against the decode bf16 state (promoted).
# ===----------------------------------------------------------------------=== #


def _m64_inputs_from_m128(src: M128Inputs, T: Int) -> M64Inputs:
    """Build a single-seq M64Inputs with the SAME q/k/v/g/beta/A_log/dt/state0
    as `src` (slot 0), so the M64 prefill and the decode see byte-identical
    inputs. Re-seeding would diverge (M64Inputs and M128Inputs consume the RNG
    in different orders), so we copy instead. For batch=1 the two structs' q/k/
    v/g/beta/A_log/dt buffers are the same flat [t_total, H*D] shape, and slot 0
    of M128's state0 is the M64 single-slot state0."""
    var dst = M64Inputs(T)
    var n_qkv = T * HK
    for i in range(n_qkv):
        dst.q[i] = src.q[i]
        dst.k[i] = src.k[i]
        dst.v[i] = src.v[i]
        dst.g[i] = src.g[i]
    for i in range(T * HV):
        dst.beta[i] = src.beta[i]
    for i in range(HV):
        dst.alog[i] = src.alog[i]
    for i in range(HVK):
        dst.dt[i] = src.dt[i]
    for i in range(STATE_TILE):
        dst.state0[i] = src.state0[i]  # slot 0
    dst.cu[0] = Int64(0)
    dst.cu[1] = Int64(T)
    return dst^


def _dispatch_check_m64(name: String, ctx: DeviceContext, T: Int) raises:
    """Batch=1, T tokens: M64 prefill vs decode token-by-token, identical inputs.
    """
    seed(101)
    var seq_lens = List[Int]()
    seq_lens.append(T)
    # Build ONE source of truth (M128Inputs, which the decode helper takes),
    # then derive the M64Inputs by copying so both paths see identical q/k/v/...
    var inp = M128Inputs(seq_lens, 1, 1)
    var minp = _m64_inputs_from_m128(inp, T)
    var got_pre = _run_m64_prefill(
        ctx, minp, use_initial_state=False, store_final_state=True
    )
    var slippery: List[Int] = List[Int]()
    for t in range(T):
        slippery.append(t)
    slippery.append(T)
    var got_dec = _run_decode(
        ctx, inp, window_starts=slippery, total_T=T, per_seq=False
    )
    var n_qkv = T * HK
    var n_state = STATE_TILE
    var e_out = _rel_err(got_pre[0], got_dec[0], n_qkv)
    var e_state = _rel_err(
        got_pre[1] + 0 * STATE_TILE, got_dec[1] + 0 * STATE_TILE, n_state
    )
    print(name, ": dispatch out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": dispatch output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": dispatch state rel_err too high")
    got_pre[0].free()
    got_pre[1].free()
    got_dec[0].free()
    got_dec[1].free()
    _ = minp^
    _ = inp^


def _dispatch_check_m128(
    name: String, ctx: DeviceContext, seq_lens: List[Int]
) raises:
    """Batch>1: M128 prefill vs decode per-sequence, identical inputs."""
    seed(102)
    var total = 0
    for L in seq_lens:
        total += L
    # Build ONE source of truth and feed both the prefill and the decode so
    # the two paths see byte-identical q/k/v/g/beta/A_log/dt/state0.
    var inp = M128Inputs(seq_lens, len(seq_lens), len(seq_lens))
    var got_pre = _run_m128_prefill(
        ctx, inp, use_initial_state=False, store_final_state=True
    )
    # decode each sequence on its own init state (the dispatch contract: the
    # layer runs decode one seq at a time, state per seq). Run windows per seq.
    var slippery: List[Int] = List[Int]()
    var cursor = 0
    for L in seq_lens:
        slippery.append(cursor)
        slippery.append(cursor + L)
        cursor += L
    var got_dec = _run_decode(
        ctx, inp, window_starts=slippery, total_T=total, per_seq=True
    )
    var tile = STATE_TILE
    var e_out = _rel_err(got_pre[0], got_dec[0], total * HK)
    var e_state = Float64(0.0)
    for s in range(len(seq_lens)):
        var e = _rel_err(got_pre[1] + s * tile, got_dec[1] + s * tile, tile)
        e_state = max(e_state, e)
    print(name, ": dispatch out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": dispatch output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": dispatch state rel_err too high")
    got_pre[0].free()
    got_pre[1].free()
    got_dec[0].free()
    got_dec[1].free()
    _ = inp^


# ===----------------------------------------------------------------------=== #
# Group A — GLM geometry, oracle-backed (small T; P0 coverage).
# ===----------------------------------------------------------------------=== #


def test_glm53_flash_m64_t64() raises:
    """M64, 1x64 (2 chunks, exact tail). Oracle."""
    seed(201)
    with DeviceContext() as ctx:
        var inp = M64Inputs(64)
        _m64_check("glm53-flash-m64-t64", ctx, inp)


def test_glm53_flash_m64_t33() raises:
    """M64, 1x33 (chunk+1, off-by-one either side of the chunk boundary). Oracle.
    """
    seed(202)
    with DeviceContext() as ctx:
        var inp = M64Inputs(33)
        _m64_check("glm53-flash-m64-t33", ctx, inp)


def test_glm53_flash_m64_t100() raises:
    """M64, 1x100 (3 full chunks + 4-token ragged tail). Oracle."""
    seed(203)
    with DeviceContext() as ctx:
        var inp = M64Inputs(100)
        _m64_check("glm53-flash-m64-t100", ctx, inp)


def test_glm53_flash_m128_2x128() raises:
    """M128, [128, 128] (varlen batch=2, both exact tails). Oracle."""
    seed(204)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(128)
        seq_lens.append(128)
        var inp = M128Inputs(seq_lens, 2, 2)
        _m128_check("glm53-flash-m128-2x128", ctx, inp)


def test_glm53_flash_m128_3x64() raises:
    """M128, [64, 64, 64] (batch=3). Oracle."""
    seed(205)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(64)
        seq_lens.append(64)
        seq_lens.append(64)
        var inp = M128Inputs(seq_lens, 3, 3)
        _m128_check("glm53-flash-m128-3x64", ctx, inp)


# ===----------------------------------------------------------------------=== #
# Group B — real GLM prefill shapes, chunk-vs-decode gate (no oracle; fast).
# Decode is the un-reassociated recurrence; the comparison independently
# catches the chunk reassociation without the fp64 oracle's O(T*H*K*V) cost.
# ===----------------------------------------------------------------------=== #


def test_glm53_flash_prefill_m64_1x512() raises:
    """M64, 1x512 (16 chunks). Chunk-vs-decode."""
    seed(211)
    with DeviceContext() as ctx:
        _dispatch_check_m64("glm53-flash-prefill-m64-1x512", ctx, 512)


def test_glm53_flash_prefill_m64_1x1024() raises:
    """M64, 1x1024 (32 chunks). Chunk-vs-decode."""
    seed(212)
    with DeviceContext() as ctx:
        _dispatch_check_m64("glm53-flash-prefill-m64-1x1024", ctx, 1024)


def test_glm53_flash_prefill_m128_6x3063() raises:
    """M128, 6x3063 (ragged brochure shape). Chunk-vs-decode."""
    seed(213)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        for _ in range(6):
            seq_lens.append(3063)
        _dispatch_check_m128("glm53-flash-prefill-m128-6x3063", ctx, seq_lens)


def test_glm53_flash_prefill_m128_8x1024() raises:
    """M128, 8x1024 (batch=8). Chunk-vs-decode."""
    seed(214)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        for _ in range(8):
            seq_lens.append(1024)
        _dispatch_check_m128("glm53-flash-prefill-m128-8x1024", ctx, seq_lens)


def test_glm53_flash_prefill_m128_2x2048() raises:
    """M128, 2x2048 (batch=2, long — cross-chunk carry over 64 chunks)."""
    seed(215)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(2048)
        seq_lens.append(2048)
        _dispatch_check_m128("glm53-flash-prefill-m128-2x2048", ctx, seq_lens)


# ===----------------------------------------------------------------------=== #
# Group C — ragged prime lengths (bounds-checking stress).
# Primes aren't multiples of 32, so every prime-length sequence has a
# non-trivial ragged tail exercising the zero-filled cp.async loads + the
# scalar-store epilogue. Picked to span tail sizes mod 32.
# ===----------------------------------------------------------------------=== #


def test_glm53_flash_prime_m64_t97() raises:
    """M64, 1x97 (prime; 3 full chunks + 1-token tail). Oracle."""
    seed(221)
    with DeviceContext() as ctx:
        var inp = M64Inputs(97)
        _m64_check("glm53-flash-prime-m64-t97", ctx, inp)


def test_glm53_flash_prime_m64_t127() raises:
    """M64, 1x127 (prime; 3 full + 31-token tail, max just under a chunk). Oracle.
    """
    seed(222)
    with DeviceContext() as ctx:
        var inp = M64Inputs(127)
        _m64_check("glm53-flash-prime-m64-t127", ctx, inp)


def test_glm53_flash_prime_m128_varlen() raises:
    """M128, [97, 127, 64] (two primes + one multiple-of-32 anchor). Oracle."""
    seed(223)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(97)
        seq_lens.append(127)
        seq_lens.append(64)
        var inp = M128Inputs(seq_lens, 3, 3)
        _m128_check("glm53-flash-prime-m128-varlen", ctx, inp)


def test_glm53_flash_prime_m128_batch8() raises:
    """M128, 8 primes [97,127,101,89,64,32,113,127] (mostly primes, total 750).
    Chunk-vs-decode (8x64 state pool is fine; 8x fp64 oracle too slow)."""
    seed(224)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        for L in [97, 127, 101, 89, 64, 32, 113, 127]:
            seq_lens.append(L)
        _dispatch_check_m128("glm53-flash-prime-m128-batch8", ctx, seq_lens)


# ===----------------------------------------------------------------------=== #
# Group D — dispatch: prefill vs token-by-token decode (the kernel-side
# proxy for the layer's T>1 -> prefill / T=1 -> decode split).
# ===----------------------------------------------------------------------=== #


def test_glm53_flash_dispatch_m64_t64() raises:
    """Batch=1, T=64: M64 prefill vs decode token-by-token. Small; oracle-equivalent.
    """
    seed(231)
    with DeviceContext() as ctx:
        _dispatch_check_m64("glm53-flash-dispatch-m64-t64", ctx, 64)


def test_glm53_flash_dispatch_m128_2x64() raises:
    """Batch=2, [64,64]: M128 prefill vs decode per-sequence."""
    seed(232)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(64)
        seq_lens.append(64)
        _dispatch_check_m128("glm53-flash-dispatch-m128-2x64", ctx, seq_lens)


def test_glm53_flash_dispatch_m128_prime_mix() raises:
    """Batch=3, [97, 64, 127]: M128 prefill vs decode, prime tails on the outer seqs.
    """
    seed(233)
    with DeviceContext() as ctx:
        var seq_lens = List[Int]()
        seq_lens.append(97)
        seq_lens.append(64)
        seq_lens.append(127)
        _dispatch_check_m128(
            "glm53-flash-dispatch-m128-prime-mix", ctx, seq_lens
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
