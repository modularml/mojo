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
"""Correctness for the BT=16 KDA decomposition: prep + chain end to end.

Generates inputs on the host, runs both kernels on the GPU, and checks the
output and the final state against `kda.reference.kda_decode_ref`, the
in-tree FP64 CPU oracle. No external data, so this travels with the repo.

The oracle walks the unfactored recurrence one token at a time, so agreement
here also confirms the chunked factorisation itself, not just the kernels
that implement it.

Cases span a single chunk, a multi-chunk walk, a ragged tail, a non-zero
initial state, a length that wraps the eight-deep operand ring, packed
varlen sequences, and a T=8192 walk of 512 chunks. Head count is held low
where possible: the oracle costs O(T * H * K * V), and heads buy grid
parallelism rather than recursion depth.
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.utils import IndexList

from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor

from kda.reference import kda_decode_ref
from kda.sm100_kda_prep_bt16 import (
    QkGateTmaTile,
    WsTileTmaTile,
    sm100_kda_prep_bt16,
)
from kda.sm100_kda_chain_bt16 import (
    CHAIN_SMEM_BYTES,
    ChainDiagTmaTile,
    ChainQkTmaTile,
    ChainVTmaTile,
    ChainWsTmaTile,
    sm100_kda_chain_bt16,
)

comptime BF16 = Scalar[DType.bfloat16]
comptime BT = 16
comptime DK = 128
comptime SCALE = 0.08838834764831845  # 1 / sqrt(128)
comptime LOWER_BOUND = -5.0

# The decomposition rounds its factor workspace and its state to bf16, so it
# is held to the same tolerances as the fused ports rather than to the
# oracle's FP64 precision.
comptime OUT_TOL = 0.01
comptime STATE_TOL = 0.02


def _rel_err(
    want: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    got: UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float64:
    """Compute RMSE(want - got) / RMSE(want)."""
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


def _run_case[
    H: Int
](
    ctx: DeviceContext,
    name: String,
    seq_lens: List[Int],
    nonzero_init: Bool,
    use_initial_state: Bool = True,
    store_final_state: Bool = True,
) raises:
    """Generate one case, run prep + chain against the oracle, and compare."""
    seed(0xC0FFEE)
    var num_seqs = len(seq_lens)
    var t_total = 0
    for i in range(num_seqs):
        t_total += seq_lens[i]

    # Chunk map: chunks are per sequence, so a ragged tail costs a part chunk.
    var num_chunks = 0
    for i in range(num_seqs):
        num_chunks += (seq_lens[i] + BT - 1) // BT
    var ws_rows = num_chunks * BT

    var n_qkv = t_total * H * DK
    var n_state = num_seqs * H * DK * DK

    # ---------------- host inputs ----------------
    var q_h = alloc[BF16](n_qkv)
    var k_h = alloc[BF16](n_qkv)
    var v_h = alloc[BF16](n_qkv)
    var g_h = alloc[BF16](n_qkv)
    var beta_h = alloc[BF16](t_total * H)
    var alog_h = alloc[Scalar[DType.float32]](H)
    var dt_h = alloc[Scalar[DType.float32]](H * DK)
    var state0_h = alloc[BF16](n_state)
    _fill_bf16(q_h, n_qkv, -1.0, 1.0)
    _fill_bf16(k_h, n_qkv, -1.0, 1.0)
    _fill_bf16(v_h, n_qkv, -1.0, 1.0)
    _fill_bf16(g_h, n_qkv, -2.0, 2.0)
    _fill_bf16(beta_h, t_total * H, -2.0, 2.0)
    _fill_f32(alog_h, H, -1.0, 0.0)
    _fill_f32(dt_h, H * DK, -0.5, 0.5)
    if nonzero_init or not use_initial_state:
        _fill_bf16(state0_h, n_state, -0.2, 0.2)
    else:
        for i in range(n_state):
            state0_h[i] = BF16(0)

    var cu_h = alloc[Scalar[DType.int64]](num_seqs + 1)
    var cuc_h = alloc[Scalar[DType.int32]](num_seqs + 1)
    cu_h[0] = 0
    cuc_h[0] = 0
    for s in range(num_seqs):
        cu_h[s + 1] = cu_h[s] + Int64(seq_lens[s])
        cuc_h[s + 1] = cuc_h[s] + Int32((seq_lens[s] + BT - 1) // BT)

    # ---------------- device buffers ----------------
    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * H)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](H)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](H * DK)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](num_seqs + 1)
    var d_cuc = ctx.enqueue_create_buffer[DType.int32](num_seqs + 1)
    var d_out = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_state0 = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_final = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var n_tile = H * ws_rows * DK
    var d_ws_kd = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_qd = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_w = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_qk = ctx.enqueue_create_buffer[DType.bfloat16](
        H * num_chunks * 256
    )
    var d_ws_diag = ctx.enqueue_create_buffer[DType.float32](
        H * num_chunks * DK
    )

    ctx.enqueue_copy(d_q, q_h)
    ctx.enqueue_copy(d_k, k_h)
    ctx.enqueue_copy(d_v, v_h)
    ctx.enqueue_copy(d_g, g_h)
    ctx.enqueue_copy(d_beta, beta_h)
    ctx.enqueue_copy(d_alog, alog_h)
    ctx.enqueue_copy(d_dt, dt_h)
    ctx.enqueue_copy(d_cu, cu_h)
    ctx.enqueue_copy(d_cuc, cuc_h)
    ctx.enqueue_copy(d_state0, state0_h)
    d_out.enqueue_fill(BF16(0))
    # A sentinel the kernel never writes, so `store_final_state=0` can be
    # checked for what it does NOT do.
    comptime SENTINEL = 12.5
    d_final.enqueue_fill(BF16(SENTINEL))

    var h128 = H * DK

    # ---------------- descriptors ----------------
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](1, H, t_total, DK),
        IndexList[4](t_total * h128, DK, h128, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](1, H, t_total, DK),
        IndexList[4](t_total * h128, DK, h128, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_g,
        IndexList[4](1, H, t_total, DK),
        IndexList[4](t_total * h128, DK, h128, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var ws_kd_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_kd,
        IndexList[4](1, H, ws_rows, DK),
        IndexList[4](ws_rows * h128, ws_rows * DK, DK, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var ws_qd_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_qd,
        IndexList[4](1, H, ws_rows, DK),
        IndexList[4](ws_rows * h128, ws_rows * DK, DK, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var ws_w_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_w,
        IndexList[4](1, H, ws_rows, DK),
        IndexList[4](ws_rows * h128, ws_rows * DK, DK, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_v,
        IndexList[4](1, H, t_total, DK),
        IndexList[4](t_total * h128, DK, h128, 1),
        IndexList[4](1, 1, BT, 64),
    )
    var diag_desc = create_tma_descriptor[
        DType.float32, 4, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_ws_diag,
        IndexList[4](1, H, num_chunks, DK),
        IndexList[4](num_chunks * h128, num_chunks * DK, DK, 1),
        IndexList[4](1, 1, 1, DK),
    )
    var qk_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_ws_qk,
        IndexList[4](1, H, num_chunks, 256),
        IndexList[4](num_chunks * H * 256, num_chunks * 256, 256, 1),
        IndexList[4](1, 1, 1, 256),
    )
    var o_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](1, H, t_total, DK),
        IndexList[4](t_total * h128, DK, h128, 1),
        IndexList[4](1, 1, BT, 64),
    )

    # ---------------- tensors ----------------
    var beta_t = TileTensor[linear_idx_type=DType.int32](
        d_beta.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H])),
    )
    var alog_t = TileTensor[linear_idx_type=DType.int32](
        d_alog.unsafe_ptr().as_unsafe_any_origin(), row_major(Idx[H])
    )
    var dt_t = TileTensor[linear_idx_type=DType.int32](
        d_dt.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], Idx[DK])),
    )
    var cu_t = TileTensor[linear_idx_type=DType.int32](
        d_cu.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs + 1)
    )
    var cuc_t = TileTensor[linear_idx_type=DType.int32](
        d_cuc.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs + 1)
    )
    var ws_qk_t = TileTensor[linear_idx_type=DType.int32](
        d_ws_qk.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], num_chunks, Idx[256])),
    )
    var ws_diag_t = TileTensor[linear_idx_type=DType.int32](
        d_ws_diag.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], num_chunks, Idx[DK])),
    )
    var out_t = TileTensor[linear_idx_type=DType.int32](
        d_out.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[DK])),
    )
    var state0_t = TileTensor[linear_idx_type=DType.int32](
        d_state0.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(num_seqs, Idx[H], Idx[DK], Idx[DK])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(num_seqs, Idx[H], Idx[DK], Idx[DK])),
    )

    # ---------------- k1: factor prep ----------------
    comptime prep = sm100_kda_prep_bt16[
        H,
        BetaLayout=type_of(beta_t).LayoutType,
        HVecLayout=type_of(alog_t).LayoutType,
        DtLayout=type_of(dt_t).LayoutType,
        I64Layout=type_of(cu_t).LayoutType,
        I32Layout=type_of(cuc_t).LayoutType,
        WsQkLayout=type_of(ws_qk_t).LayoutType,
        WsDiagLayout=type_of(ws_diag_t).LayoutType,
        TensorOrigin=type_of(beta_t).origin,
    ]
    ctx.enqueue_function[prep](
        QkGateTmaTile(q_desc),
        QkGateTmaTile(k_desc),
        QkGateTmaTile(g_desc),
        WsTileTmaTile(ws_kd_desc),
        WsTileTmaTile(ws_qd_desc),
        WsTileTmaTile(ws_w_desc),
        beta_t,
        alog_t,
        dt_t,
        cu_t,
        cuc_t,
        ws_qk_t,
        ws_diag_t,
        Int32(num_chunks),
        Int32(num_seqs),
        Float32(LOWER_BOUND),
        grid_dim=((num_chunks + 3) // 4, H),
        block_dim=(128,),
        shared_mem_bytes=44 * 1024,
    )

    # ---------------- k2: serial chain ----------------
    comptime chain = sm100_kda_chain_bt16[
        H,
        QkvLayout=type_of(out_t).LayoutType,
        I64Layout=type_of(cu_t).LayoutType,
        I32Layout=type_of(cuc_t).LayoutType,
        StateLayout=type_of(state0_t).LayoutType,
        TensorOrigin=type_of(out_t).origin,
    ]
    ctx.enqueue_function[chain](
        ChainWsTmaTile(ws_kd_desc),
        ChainWsTmaTile(ws_w_desc),
        ChainWsTmaTile(ws_qd_desc),
        ChainVTmaTile(v_desc),
        ChainDiagTmaTile(diag_desc),
        ChainQkTmaTile(qk_desc),
        ChainVTmaTile(o_desc),
        out_t,
        cu_t,
        cuc_t,
        state0_t,
        final_t,
        Int32(1) if use_initial_state else Int32(0),
        Int32(1) if store_final_state else Int32(0),
        Float32(SCALE),
        grid_dim=(num_seqs, H * 2),
        block_dim=(512,),
        shared_mem_bytes=CHAIN_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(CHAIN_SMEM_BYTES)
        ),
    )

    var got_out_bf = alloc[BF16](n_qkv)
    var got_state_bf = alloc[BF16](n_state)
    ctx.enqueue_copy(got_out_bf, d_out)
    ctx.enqueue_copy(got_state_bf, d_final)
    ctx.synchronize()

    # ---------------- oracle ----------------
    var q_f = _to_f32(q_h, n_qkv)
    var k_f = _to_f32(k_h, n_qkv)
    var v_f = _to_f32(v_h, n_qkv)
    var g_f = _to_f32(g_h, n_qkv)
    var beta_f = _to_f32(beta_h, t_total * H)
    var want_out = alloc[Scalar[DType.float32]](n_qkv)
    var want_state = _to_f32(state0_h, n_state)
    if not use_initial_state:
        for i in range(n_state):
            want_state[i] = 0.0
    for s in range(num_seqs):
        var window = alloc[Scalar[DType.int32]](2)
        window[0] = Int32(cu_h[s])
        window[1] = Int32(cu_h[s + 1])
        var slot = alloc[Scalar[DType.int32]](1)
        slot[0] = Int32(s)
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
            DK,
            DK,
            want_out,
            q_f,
            k_f,
            v_f,
            g_f,
            beta_f,
            alog_h,
            dt_h,
            window,
            want_state,
            slot,
            h128,
            DK,
            1,
            h128,
            DK,
            1,
            h128,
            DK,
            1,
            h128,
            DK,
            1,
            H,
            1,
            DK,
            1,
            H * DK * DK,
            DK * DK,
            DK,
            1,
            h128,
            DK,
            1,
        )
        window.free()
        slot.free()

    var got_out = _to_f32(got_out_bf, n_qkv)
    var got_state = _to_f32(got_state_bf, n_state)
    var e_out = _rel_err(want_out, got_out, n_qkv)
    if not store_final_state:
        for i in range(n_state):
            assert_true(
                Float64(got_state[i]) == Float64(SENTINEL),
                name + ": final state written despite store_final_state=0",
            )
        for i in range(n_state):
            want_state[i] = Float32(SENTINEL)
    var e_state = _rel_err(want_state, got_state, n_state)
    print(
        name,
        ": T =",
        t_total,
        " H =",
        H,
        " chunks =",
        num_chunks,
        " out rel_err =",
        e_out,
        " state rel_err =",
        e_state,
    )
    assert_true(e_out < OUT_TOL, name + ": output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": final state rel_err too high")

    q_h.free()
    k_h.free()
    v_h.free()
    g_h.free()
    beta_h.free()
    alog_h.free()
    dt_h.free()
    state0_h.free()
    cu_h.free()
    cuc_h.free()
    got_out_bf.free()
    got_state_bf.free()
    q_f.free()
    k_f.free()
    v_f.free()
    g_f.free()
    beta_f.free()
    want_out.free()
    want_state.free()
    got_out.free()
    got_state.free()


def test_bt16_single_chunk() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [16]
            _run_case[2](ctx, "single_chunk", lens, False)


def test_bt16_multichunk() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [128]
            _run_case[2](ctx, "multichunk", lens, False)


def test_bt16_ragged_tail() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [100]
            _run_case[4](ctx, "ragged_tail", lens, False)


def test_bt16_initial_state() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [64]
            _run_case[2](ctx, "initial_state", lens, True)


def test_bt16_ring_wrap() raises:
    """More chunks than the 8-deep operand ring, so the ring wraps and the
    unrolled issuer body runs its steady-state path."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [320]
            _run_case[2](ctx, "ring_wrap", lens, True)


def test_bt16_long_sequence() raises:
    """T=8192 in 512 chunks, the deepest serial walk covered here: enough
    for state drift or error growth across the recursion to show. Held at
    two heads so the scalar FP64 oracle, whose cost is O(T * H * K * V),
    stays affordable -- head count buys grid parallelism, not recursion
    depth, so it is the wrong axis to grow here."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [8192]
            _run_case[2](ctx, "long_sequence", lens, True)


def test_bt16_unroll_tail_boundary() raises:
    """T=144 is nine chunks: one full eight-chunk unrolled issuer block plus
    a single chunk in the ragged tail loop. The tightest handoff between the
    two, where the tail has to resume the ring index and the alternating
    barrier phases exactly where the unrolled body left them."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [144]
            _run_case[2](ctx, "unroll_tail_boundary", lens, True)


def test_bt16_unroll_tail_full() raises:
    """T=240 is fifteen chunks: one unrolled block plus a seven-chunk tail,
    the widest the tail loop ever runs."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [240]
            _run_case[2](ctx, "unroll_tail_full", lens, True)


def test_bt16_single_token() raises:
    """One token: fifteen of the sixteen chunk slots masked off, the extreme
    of the ragged path."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [1]
            _run_case[2](ctx, "single_token", lens, True)


def test_bt16_chunk_plus_one() raises:
    """T=17: a single token in the second chunk, the off-by-one either side
    of a chunk boundary."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [17]
            _run_case[2](ctx, "chunk_plus_one", lens, True)


def test_bt16_many_heads() raises:
    """H=96 puts 192 CTAs on 148 SMs, so the chain spills into a second wave
    and head indexing is exercised at the count the benchmarks use. Held to
    two chunks to keep the O(T * H * K * V) oracle cheap."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [32]
            _run_case[96](ctx, "many_heads", lens, True)


def test_bt16_ignore_initial_state() raises:
    """Check that use_initial_state=0 is honoured: the state buffer holds
    non-zero data the kernel must not read, and the oracle starts from
    zero."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [64]
            _run_case[2](
                ctx,
                "ignore_initial_state",
                lens,
                True,
                use_initial_state=False,
            )


def test_bt16_no_store_final() raises:
    """Check that store_final_state=0 is honoured: the output still has to
    be right and the final state buffer has to be left exactly as it was
    found."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [64]
            _run_case[2](
                ctx, "no_store_final", lens, True, store_final_state=False
            )


def test_bt16_many_short_sequences() raises:
    """Sixteen short sequences, all ragged, which is what a serving batch of
    small requests looks like: sixteen state slots and sixteen independent
    chunk ranges rather than a few long walks."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [
                7,
                16,
                1,
                33,
                12,
                48,
                5,
                20,
                64,
                3,
                17,
                9,
                31,
                2,
                40,
                11,
            ]
            _run_case[2](ctx, "many_short_sequences", lens, True)


def test_bt16_empty_sequence() raises:
    """A zero-length sequence between two real ones. It contributes no
    chunks, so its chain CTA launches with nothing to walk and still has to
    balance its barrier arrivals and hand back its initial state."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [64, 0, 48]
            _run_case[2](ctx, "empty_sequence", lens, True)


def test_bt16_varlen() raises:
    """Multiple packed sequences, each with its own state slot and its own
    ragged tail chunk."""
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var lens: List[Int] = [64, 100, 32, 48]
            _run_case[2](ctx, "varlen", lens, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
