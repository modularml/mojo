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
"""Reference-based tests for the SM100 KDA prefill kernel, M128 variant.

The reference (`kda.reference.kda_decode_ref`) computes the KDA recurrence
token by token in FP64; run over a whole sequence it is a prefill oracle.
The prefill kernel's math corresponds to `gate_mode="safe"` (its lower_bound
of -5 is the safe gate's constant, applied in log2 space) and
`beta_mode="logits"` (its tanh(x/2)/2 + 1/2 beta transform is sigmoid).

Mirrors the decode kernel's unit-test matrix (test_kda_gpu.mojo) where the
axes exist on this kernel:
  - fixed shapes at several H, chunk-aligned and ragged lengths
  - varlen batches with mixed ragged lengths
  - nonzero initial state in/out
  - state_indices scatter into a larger state pool
  - speculative checkpoints (state snapshots every N tokens)
  - store_final_state=0 leaves the final-state buffer untouched
Axes that do not exist here and are intentionally untested: gate_mode
"original" and beta_mode "probability" (this kernel bakes safe/logits),
GQA head grouping (H == HV by contract), state-layout selection (the
kernel's state is V_FIRST by contract: TMEM datapath rows are the value
dim, so the global tile is [slot, h, v, k]) and packed QKV (fixed I/O
contract), FP32 state (bf16 by contract).

All comparisons use rel_err = RMSE(ref - gpu) / RMSE(ref). The output and
state tolerances absorb the kernel's internal bf16 operand narrowing
(state snapshot, residuals, corrected errors) against the oracle.
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

from kda.reference import kda_decode_ref
from kda.sm100_kda_prefill_m128 import (
    BetaTmaTile,
    QkOutTmaTile,
    SMEM_TOTAL,
    VgTmaTile,
    sm100_kda_prefill,
)

comptime BF16 = Scalar[DType.bfloat16]

comptime OUT_TOL = 0.01
comptime STATE_TOL = 0.02  # bf16 state quantisation (measured ~0.004)


def _rel_err(
    want: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    got: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float64:
    """RMSE(want - got) / RMSE(want)."""
    var r_sq = Float64(0.0)
    var d_sq = Float64(0.0)
    for i in range(n):
        var r = Float64(want[i])
        var d = r - Float64(got[i])
        r_sq += r * r
        d_sq += d * d
    return sqrt(d_sq / Float64(n)) / (sqrt(r_sq / Float64(n)) + Float64(1e-8))


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


struct PrefillInputs:
    """Host-side inputs for one prefill case. H is carried by the runner."""

    var num_seqs: Int
    var t_total: Int
    var num_slots: Int
    var num_ckpts: Int
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
        out self, H: Int, seq_lengths: List[Int], num_slots: Int, num_ckpts: Int
    ):
        self.num_seqs = len(seq_lengths)
        self.t_total = 0
        for i in range(len(seq_lengths)):
            self.t_total += seq_lengths[i]
        self.num_slots = num_slots
        self.num_ckpts = num_ckpts

        var n_qkv = self.t_total * H * 128
        var n_state = num_slots * H * 128 * 128
        self.q = alloc[BF16](n_qkv)
        self.k = alloc[BF16](n_qkv)
        self.v = alloc[BF16](n_qkv)
        self.g = alloc[BF16](n_qkv)
        self.beta = alloc[BF16](self.t_total * H)
        self.alog = alloc[Scalar[DType.float32]](H)
        self.dt = alloc[Scalar[DType.float32]](H * 128)
        self.cu = alloc[Scalar[DType.int64]](self.num_seqs + 1)
        self.state0 = alloc[BF16](n_state)
        self.state_indices = alloc[Scalar[DType.int32]](self.num_seqs)
        self.ckpt_starts = alloc[Scalar[DType.int64]](self.num_seqs)

        _fill_bf16(self.q, n_qkv, -1.0, 1.0)
        _fill_bf16(self.k, n_qkv, -1.0, 1.0)
        _fill_bf16(self.v, n_qkv, -1.0, 1.0)
        _fill_bf16(self.g, n_qkv, -1.0, 1.0)
        _fill_bf16(self.beta, self.t_total * H, -2.0, 2.0)
        _fill_f32(self.alog, H, -1.0, 0.0)
        _fill_f32(self.dt, H * 128, -0.1, 0.1)
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


struct PrefillResults:
    """Kernel outputs copied back to the host as FP32."""

    var out: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var final_state: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var ckpts: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]

    def __init__(
        out self,
        out_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
        final_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
        ckpt_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    ):
        self.out = out_p
        self.final_state = final_p
        self.ckpts = ckpt_p

    def __deinit__(deinit self):
        self.out.free()
        self.final_state.free()
        self.ckpts.free()


def _run_prefill[
    H: Int
](
    ctx: DeviceContext,
    inp: PrefillInputs,
    *,
    use_initial_state: Bool,
    store_final_state: Bool,
    use_state_indices: Bool,
    checkpoint_every: Int,
    final_sentinel: Float32 = 0.0,
) raises -> PrefillResults:
    """Launch the prefill kernel on `inp`; return FP32 host copies of the
    output, the final-state pool, and the checkpoint pool."""
    var num_seqs = inp.num_seqs
    var t_total = inp.t_total
    var n_qkv = t_total * H * 128
    var n_state = inp.num_slots * H * 128 * 128
    var n_ckpt = inp.num_ckpts * H * 128 * 128
    comptime padded_heads = (H + 7) // 8 * 8

    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * H)
    var d_beta_tma = ctx.enqueue_create_buffer[DType.bfloat16](
        t_total * padded_heads
    )
    var d_alog = ctx.enqueue_create_buffer[DType.float32](H)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](H * 128)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](num_seqs + 1)
    var d_seqorder = ctx.enqueue_create_buffer[DType.int32](num_seqs)
    var d_state0 = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_state_final = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_ckpt = ctx.enqueue_create_buffer[DType.bfloat16](n_ckpt)
    var d_si = ctx.enqueue_create_buffer[DType.int32](num_seqs)
    var d_ckpt_starts = ctx.enqueue_create_buffer[DType.int64](num_seqs)
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
    ctx.enqueue_copy(d_si, inp.state_indices)
    ctx.enqueue_copy(d_ckpt_starts, inp.ckpt_starts)

    # The final-state pool starts as a sentinel fill so store_final_state=0
    # cases can verify it is left untouched.
    var final_init_h = alloc[BF16](n_state)
    for i in range(n_state):
        final_init_h[i] = BF16(final_sentinel)
    ctx.enqueue_copy(d_state_final, final_init_h)
    var ckpt_init_h = alloc[BF16](n_ckpt)
    for i in range(n_ckpt):
        ckpt_init_h[i] = BF16(0.0)
    ctx.enqueue_copy(d_ckpt, ckpt_init_h)

    var seqorder_h = alloc[Scalar[DType.int32]](num_seqs)
    for i in range(num_seqs):
        seqorder_h[i] = Int32(i)
    ctx.enqueue_copy(d_seqorder, seqorder_h)

    # Beta padded to the next 8-head boundary for the TMA box.
    var beta_pad_h = alloc[BF16](t_total * padded_heads)
    for t in range(t_total):
        for h in range(padded_heads):
            var val = BF16(0.0)
            if h < H:
                val = inp.beta[t * H + h]
            beta_pad_h[t * padded_heads + h] = val
    ctx.enqueue_copy(d_beta_tma, beta_pad_h)

    comptime h128 = H * 128
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](2, H, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](2, H, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_v,
        IndexList[3](t_total, H, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_g,
        IndexList[3](t_total, H, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var beta_desc = create_tma_descriptor[
        DType.bfloat16, 2, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_beta_tma,
        IndexList[2](t_total, padded_heads),
        IndexList[2](padded_heads, 1),
        IndexList[2](32, 8),
    )
    var out_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](2, H, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var q_tma = QkOutTmaTile(q_desc)
    var k_tma = QkOutTmaTile(k_desc)
    var v_tma = VgTmaTile(v_desc)
    var g_tma = VgTmaTile(g_desc)
    var beta_tma = BetaTmaTile(beta_desc)
    var out_tma = QkOutTmaTile(out_desc)

    var q_t = TileTensor[linear_idx_type=DType.int32](
        d_q.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var k_t = TileTensor[linear_idx_type=DType.int32](
        d_k.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var v_t = TileTensor[linear_idx_type=DType.int32](
        d_v.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var g_t = TileTensor[linear_idx_type=DType.int32](
        d_g.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var out_tt = TileTensor[linear_idx_type=DType.int32](
        d_out.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var beta_t = TileTensor[linear_idx_type=DType.int32](
        d_beta.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H])),
    )
    var alog_t = TileTensor[linear_idx_type=DType.int32](
        d_alog.unsafe_ptr().as_unsafe_any_origin(), row_major(Idx[H])
    )
    var dt_t = TileTensor[linear_idx_type=DType.int32](
        d_dt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], Idx[128])),
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
        row_major(Coord(inp.num_slots, Idx[H], Idx[128], Idx[128])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_state_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(inp.num_slots, Idx[H], Idx[128], Idx[128])),
    )
    var ckpt_t = TileTensor[linear_idx_type=DType.int32](
        d_ckpt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(inp.num_ckpts, Idx[H], Idx[128], Idx[128])),
    )
    var ckpt_starts_t = TileTensor[linear_idx_type=DType.int32](
        d_ckpt_starts.unsafe_ptr().as_unsafe_any_origin(),
        row_major(num_seqs),
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
        Float32(0.08838834764831845),  # 1/sqrt(128)
        Float32(-5.0),  # lower_bound: the safe gate's constant
        si_t,
        ckpt_t,
        ckpt_starts_t,
        Int32(1) if use_state_indices else Int32(0),
        Int32(checkpoint_every),
        grid_dim=(num_seqs * H,),
        block_dim=(1024,),
        shared_mem_bytes=SMEM_TOTAL,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_TOTAL)
        ),
    )

    var out_bf16_h = alloc[BF16](n_qkv)
    var final_bf16_h = alloc[BF16](n_state)
    var ckpt_bf16_h = alloc[BF16](n_ckpt)
    ctx.enqueue_copy(out_bf16_h, d_out)
    ctx.enqueue_copy(final_bf16_h, d_state_final)
    ctx.enqueue_copy(ckpt_bf16_h, d_ckpt)
    ctx.synchronize()

    var out_h = _to_f32(out_bf16_h, n_qkv)
    var final_h = _to_f32(final_bf16_h, n_state)
    var ckpt_h = _to_f32(ckpt_bf16_h, n_ckpt)
    out_bf16_h.free()
    final_bf16_h.free()
    ckpt_bf16_h.free()
    final_init_h.free()
    ckpt_init_h.free()
    seqorder_h.free()
    beta_pad_h.free()
    return PrefillResults(out_h, final_h, ckpt_h)


struct RefResults:
    """Reference output, final states, and checkpoint snapshots, in FP32
    buffers; the oracle itself accumulates in FP64."""

    var out: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var state: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]
    var ckpts: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]

    def __init__(
        out self,
        out_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
        state_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
        ckpt_p: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    ):
        self.out = out_p
        self.state = state_p
        self.ckpts = ckpt_p

    def __deinit__(deinit self):
        self.out.free()
        self.state.free()
        self.ckpts.free()


def _run_reference(
    H: Int, inp: PrefillInputs, *, checkpoint_every: Int
) raises -> RefResults:
    """Run the scalar reference over `inp`, one checkpoint segment at a
    time, snapshotting the state pool at every segment boundary."""
    var n_qkv = inp.t_total * H * 128
    var n_state = inp.num_slots * H * 128 * 128
    var n_ckpt = inp.num_ckpts * H * 128 * 128
    var tile = H * 128 * 128

    var q_f = _to_f32(inp.q, n_qkv)
    var k_f = _to_f32(inp.k, n_qkv)
    var v_f = _to_f32(inp.v, n_qkv)
    var g_f = _to_f32(inp.g, n_qkv)
    var beta_f = _to_f32(inp.beta, inp.t_total * H)
    var out_f = alloc[Scalar[DType.float32]](n_qkv)
    var state_f = _to_f32(inp.state0, n_state)
    var ckpt_f = alloc[Scalar[DType.float32]](n_ckpt)
    for i in range(n_ckpt):
        ckpt_f[i] = 0.0

    for s in range(inp.num_seqs):
        var seq_start = Int(inp.cu[s])
        var seq_end = Int(inp.cu[s + 1])
        var slot = Int(inp.state_indices[s])
        var slot_i32 = alloc[Scalar[DType.int32]](1)
        slot_i32[0] = Int32(slot)
        var seg = checkpoint_every if checkpoint_every > 0 else (
            seq_end - seq_start
        )
        var a = seq_start
        var ckpt_idx = 0
        while a < seq_end:
            if checkpoint_every > 0:
                # Checkpoint = state at this segment boundary.
                var base = (Int(inp.ckpt_starts[s]) + ckpt_idx) * tile
                for i in range(tile):
                    ckpt_f[base + i] = state_f[slot * tile + i]
                ckpt_idx += 1
            var b = min(a + seg, seq_end)
            var window = alloc[Scalar[DType.int32]](2)
            window[0] = Int32(a)
            window[1] = Int32(b)
            kda_decode_ref[
                DType.float32,
                DType.float32,
                DType.float32,
                DType.float32,
                gate_mode="safe",
                beta_mode="logits",
                state_layout="V_FIRST",
            ](
                1,  # batch_size (one window at a time)
                H,  # num_value_heads
                H,  # num_key_heads
                128,  # key_head_dim
                128,  # value_head_dim
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
                1,  # q strides
                H * 128,
                128,
                1,  # k strides
                H * 128,
                128,
                1,  # v strides
                H * 128,
                128,
                1,  # raw_gate strides
                H,
                1,  # beta strides
                128,
                1,  # dt_bias strides
                tile,
                128 * 128,
                128,
                1,  # state strides: [slot, h, v, k] rows are the value dim
                H * 128,
                128,
                1,  # out strides
            )
            window.free()
            a = b
        slot_i32.free()

    q_f.free()
    k_f.free()
    v_f.free()
    g_f.free()
    beta_f.free()
    return RefResults(out_f, state_f, ckpt_f)


def _check_case[
    H: Int
](
    name: String,
    ctx: DeviceContext,
    inp: PrefillInputs,
    *,
    use_initial_state: Bool = False,
    use_state_indices: Bool = False,
    checkpoint_every: Int = 0,
) raises:
    var got = _run_prefill[H](
        ctx,
        inp,
        use_initial_state=use_initial_state,
        store_final_state=True,
        use_state_indices=use_state_indices,
        checkpoint_every=checkpoint_every,
    )
    var want = _run_reference(H, inp, checkpoint_every=checkpoint_every)

    var n_qkv = inp.t_total * H * 128
    var tile = H * 128 * 128
    var e_out = _rel_err(want.out, got.out, n_qkv)
    # Compare only the slots this case touches.
    var e_state = Float64(0.0)
    for s in range(inp.num_seqs):
        var slot = Int(inp.state_indices[s])
        var e = _rel_err(
            want.state + slot * tile, got.final_state + slot * tile, tile
        )
        e_state = max(e_state, e)
    print(name, ": rel_err out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": final state rel_err too high")

    if checkpoint_every > 0:
        var n_ckpt = inp.num_ckpts * tile
        var e_ckpt = _rel_err(want.ckpts, got.ckpts, n_ckpt)
        print(name, ": rel_err checkpoints =", e_ckpt)
        assert_true(e_ckpt < STATE_TOL, name + ": checkpoint rel_err too high")
    _ = got^
    _ = want^


# ===----------------------------------------------------------------------=== #
# Tests (mirroring the decode suite's matrix where the axes exist here)
# ===----------------------------------------------------------------------=== #


def test_fixed_h2() raises:
    seed(11)
    with DeviceContext() as ctx:
        var lens: List[Int] = [256]
        var inp = PrefillInputs(2, lens, 1, 1)
        _check_case[2]("fixed_h2 T=256", ctx, inp)


def test_long_sequence() raises:
    """T=8192 in 256 chunks, the deepest serial walk covered here: enough
    for state drift or error growth across the recursion to show. Two heads
    keeps the scalar FP64 oracle affordable, and head count buys grid
    parallelism rather than recursion depth."""
    seed(19)
    with DeviceContext() as ctx:
        var lens: List[Int] = [8192]
        var inp = PrefillInputs(2, lens, 1, 1)
        _check_case[2]("long_sequence T=8192", ctx, inp)


def test_fixed_h4() raises:
    seed(12)
    with DeviceContext() as ctx:
        var lens: List[Int] = [512]
        var inp = PrefillInputs(4, lens, 1, 1)
        _check_case[4]("fixed_h4 T=512", ctx, inp)


def test_ragged_tail() raises:
    # T=100: three full 32-token chunks plus a 4-token ragged tail, which
    # exercises the zero-filled cp.async loads and the scalar-store epilogue.
    seed(13)
    with DeviceContext() as ctx:
        var lens: List[Int] = [100]
        var inp = PrefillInputs(2, lens, 1, 1)
        _check_case[2]("ragged_tail T=100", ctx, inp)


def test_varlen_mixed() raises:
    seed(14)
    with DeviceContext() as ctx:
        var lens: List[Int] = [64, 100, 32, 231]
        var inp = PrefillInputs(2, lens, 4, 1)
        _check_case[2]("varlen_mixed", ctx, inp)


def test_single_token() raises:
    """One token: 31 of the 32 chunk slots masked off, the extreme of the
    ragged path."""
    seed(31)
    with DeviceContext() as ctx:
        var lens: List[Int] = [1]
        var inp = PrefillInputs(2, lens, 1, 1)
        _check_case[2]("single_token T=1", ctx, inp)


def test_chunk_plus_one() raises:
    """T=33: a single token in the second chunk, the off-by-one either side
    of a chunk boundary."""
    seed(32)
    with DeviceContext() as ctx:
        var lens: List[Int] = [33]
        var inp = PrefillInputs(2, lens, 1, 1)
        _check_case[2]("chunk_plus_one T=33", ctx, inp)


def test_empty_sequence() raises:
    """A zero-length sequence between two real ones: it owns a state slot
    and has to hand it back untouched."""
    seed(33)
    with DeviceContext() as ctx:
        var lens: List[Int] = [64, 0, 96]
        var inp = PrefillInputs(2, lens, 3, 1)
        _check_case[2]("empty_sequence", ctx, inp)


def test_initial_state() raises:
    seed(15)
    with DeviceContext() as ctx:
        var lens: List[Int] = [128, 96]
        var inp = PrefillInputs(2, lens, 2, 1)
        _fill_bf16(inp.state0, 2 * 2 * 128 * 128, -0.5, 0.5)
        _check_case[2]("initial_state", ctx, inp, use_initial_state=True)


def test_state_indices_scatter() raises:
    # Three sequences scattered into slots {4, 0, 2} of a six-slot pool.
    seed(16)
    with DeviceContext() as ctx:
        var lens: List[Int] = [96, 64, 128]
        var inp = PrefillInputs(2, lens, 6, 1)
        inp.state_indices[0] = Int32(4)
        inp.state_indices[1] = Int32(0)
        inp.state_indices[2] = Int32(2)
        _fill_bf16(inp.state0, 6 * 2 * 128 * 128, -0.5, 0.5)
        _check_case[2](
            "state_indices_scatter",
            ctx,
            inp,
            use_initial_state=True,
            use_state_indices=True,
        )


def test_checkpoints() raises:
    # Snapshots every 64 tokens: seq0 (T=160) checkpoints at tokens 0/64/128,
    # seq1 (T=96) at 0/64 -- five tiles at cu_starts [0, 3].
    seed(17)
    with DeviceContext() as ctx:
        var lens: List[Int] = [160, 96]
        var inp = PrefillInputs(2, lens, 2, 5)
        inp.ckpt_starts[0] = Int64(0)
        inp.ckpt_starts[1] = Int64(3)
        _fill_bf16(inp.state0, 2 * 2 * 128 * 128, -0.5, 0.5)
        _check_case[2](
            "checkpoints N=64",
            ctx,
            inp,
            use_initial_state=True,
            checkpoint_every=64,
        )


def test_h96_short() raises:
    seed(18)
    with DeviceContext() as ctx:
        var lens: List[Int] = [64]
        var inp = PrefillInputs(96, lens, 1, 1)
        _check_case[96]("h96 T=64", ctx, inp)


def test_no_store_final() raises:
    # store_final_state=0: the output must still be right and the final-state
    # pool must keep its sentinel fill.
    seed(19)
    with DeviceContext() as ctx:
        var lens: List[Int] = [128]
        var inp = PrefillInputs(2, lens, 1, 1)
        var got = _run_prefill[2](
            ctx,
            inp,
            use_initial_state=False,
            store_final_state=False,
            use_state_indices=False,
            checkpoint_every=0,
            final_sentinel=7.0,
        )
        var want = _run_reference(2, inp, checkpoint_every=0)
        var n_qkv = inp.t_total * 2 * 128
        var e_out = _rel_err(want.out, got.out, n_qkv)
        print("no_store_final: rel_err out =", e_out)
        assert_true(e_out < OUT_TOL, "no_store_final: output rel_err too high")
        var tile = 2 * 128 * 128
        for i in range(tile):
            assert_true(
                got.final_state[i] == Float32(7.0),
                "no_store_final: final-state pool was written",
            )
        # Results own their buffers; keep them alive past the last pointer
        # read above (ASAP destruction would otherwise free the memory the
        # comparison is still reading).
        _ = got^
        _ = want^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
