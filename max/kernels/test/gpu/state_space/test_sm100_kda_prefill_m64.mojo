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
"""Reference-based tests for the SM100 KDA prefill kernel, M64 variant.

The reference (`kda.reference.kda_decode_ref`) computes the KDA recurrence
in FP64 scalar arithmetic on the host; the kernel is compared against it with
relative-RMSE tolerances that absorb the bf16 I/O quantisation.

The M64 variant is FlashInfer's value-split schedule for a single fixed-layout
sequence with H=64: each (sequence, head) task runs on two CTAs (grid.x =
2*B*H) that each own a 64-wide half of the value dimension.  The cases here
stay inside that dispatch contract (one sequence, H=64) and cover the fresh
and carried-in state paths, the ragged tail (the load warp's zero-filled
cp.async plus the scalar-store epilogue), and the single-chunk minimum.
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.utils import IndexList
from layout import Coord, Idx, TileTensor, row_major
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor

from kda.reference import kda_decode_ref
from kda.sm100_kda_prefill_m64 import (
    BetaTmaTile,
    GateTmaTile,
    OutTmaTile,
    QkTmaTile,
    VTmaTile,
    sm100_kda_prefill_m64,
    SMEM_TOTAL,
)

comptime BF16 = Scalar[DType.bfloat16]
comptime H = 64  # the M64 variant's dispatch contract

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
    """Host-side inputs for one single-sequence M64 prefill case."""

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

    def __init__(out self, t_total: Int):
        self.t_total = t_total
        var n_qkv = t_total * H * 128
        var n_state = H * 128 * 128
        self.q = alloc[BF16](n_qkv)
        self.k = alloc[BF16](n_qkv)
        self.v = alloc[BF16](n_qkv)
        self.g = alloc[BF16](n_qkv)
        self.beta = alloc[BF16](t_total * H)
        self.alog = alloc[Scalar[DType.float32]](H)
        self.dt = alloc[Scalar[DType.float32]](H * 128)
        self.cu = alloc[Scalar[DType.int64]](2)
        self.state0 = alloc[BF16](n_state)

        _fill_bf16(self.q, n_qkv, -1.0, 1.0)
        _fill_bf16(self.k, n_qkv, -1.0, 1.0)
        _fill_bf16(self.v, n_qkv, -1.0, 1.0)
        _fill_bf16(self.g, n_qkv, -1.0, 1.0)
        _fill_bf16(self.beta, t_total * H, -2.0, 2.0)
        _fill_f32(self.alog, H, -1.0, 0.0)
        _fill_f32(self.dt, H * 128, -0.1, 0.1)
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


def _run_prefill(
    ctx: DeviceContext,
    inp: PrefillInputs,
    *,
    use_initial_state: Bool,
    store_final_state: Bool,
    final_sentinel: Float32 = 0.0,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Launch the M64 prefill kernel on `inp`; return FP32 host copies of the
    output and the final state."""
    var t_total = inp.t_total
    var n_qkv = t_total * H * 128
    var n_state = H * 128 * 128

    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * H)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](H)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](H * 128)
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

    # The final-state pool starts as a sentinel fill so store_final_state=0
    # cases can verify it is left untouched.
    var final_init_h = alloc[BF16](n_state)
    for i in range(n_state):
        final_init_h[i] = BF16(final_sentinel)
    ctx.enqueue_copy(d_state_final, final_init_h)

    var seqorder_h = alloc[Scalar[DType.int32]](1)
    seqorder_h[0] = Int32(0)
    ctx.enqueue_copy(d_seqorder, seqorder_h)

    # TMA descriptors: q/k/g/beta as in the m128 harness (H == 64 is already
    # a multiple of the beta box, so beta needs no padding); V is the 64-wide
    # 128B-swizzled slice box and out writes one 64-column half per store.
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
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_v,
        IndexList[3](t_total, H, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 64),
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
        d_beta,
        IndexList[2](t_total, H),
        IndexList[2](H, 1),
        IndexList[2](32, 8),
    )
    var out_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](2, H, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](1, 1, 32, 64),
    )
    var q_tma = QkTmaTile(q_desc)
    var k_tma = QkTmaTile(k_desc)
    var v_tma = VTmaTile(v_desc)
    var g_tma = GateTmaTile(g_desc)
    var beta_tma = BetaTmaTile(beta_desc)
    var out_tma = OutTmaTile(out_desc)

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
        d_cu.unsafe_ptr().as_unsafe_any_origin(), row_major(2)
    )
    var so_t = TileTensor[linear_idx_type=DType.int32](
        d_seqorder.unsafe_ptr().as_unsafe_any_origin(), row_major(1)
    )
    var state0_t = TileTensor[linear_idx_type=DType.int32](
        d_state0.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[H], Idx[128], Idx[128])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_state_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[H], Idx[128], Idx[128])),
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
        Float32(0.08838834764831845),  # 1/sqrt(128)
        Float32(-5.0),  # lower_bound: the safe gate's constant
        grid_dim=(2 * H,),  # two CTAs per task: the value split
        block_dim=(1024,),
        shared_mem_bytes=SMEM_TOTAL,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_TOTAL)
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


def _run_reference(
    inp: PrefillInputs,
) raises -> Tuple[
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
    UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Run the scalar reference over `inp`'s single sequence."""
    var n_qkv = inp.t_total * H * 128
    var n_state = H * 128 * 128

    var q_f = _to_f32(inp.q, n_qkv)
    var k_f = _to_f32(inp.k, n_qkv)
    var v_f = _to_f32(inp.v, n_qkv)
    var g_f = _to_f32(inp.g, n_qkv)
    var beta_f = _to_f32(inp.beta, inp.t_total * H)
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
        1,  # batch_size
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
        H * 128 * 128,
        128 * 128,
        128,
        1,  # state strides: [slot, h, v, k] rows are the value dim
        H * 128,
        128,
        1,  # out strides
    )
    window.free()
    slot_i32.free()

    q_f.free()
    k_f.free()
    v_f.free()
    g_f.free()
    beta_f.free()
    return (out_f, state_f)


def _check_case(
    name: String,
    ctx: DeviceContext,
    inp: PrefillInputs,
    *,
    use_initial_state: Bool = False,
) raises:
    var got = _run_prefill(
        ctx,
        inp,
        use_initial_state=use_initial_state,
        store_final_state=True,
    )
    var want = _run_reference(inp)

    var n_qkv = inp.t_total * H * 128
    var n_state = H * 128 * 128
    var e_out = _rel_err(want[0], got[0], n_qkv)
    var e_state = _rel_err(want[1], got[1], n_state)
    print(name, ": rel_err out =", e_out, " state =", e_state)
    assert_true(e_out < OUT_TOL, name + ": output rel_err too high")
    assert_true(e_state < STATE_TOL, name + ": final state rel_err too high")
    got[0].free()
    got[1].free()
    want[0].free()
    want[1].free()


def test_m64_fixed() raises:
    seed(21)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(256)
        _check_case("m64 fixed T=256", ctx, inp)


def test_m64_long_sequence() raises:
    """T=1024 in 32 chunks, six times the pipeline's ring depth, so state
    drift across a long walk would show. The M64 contract pins H at 64, so
    the scalar FP64 oracle's O(T * H * K * V) cost scales with T alone;
    T=8192 depth for the same recursion is covered by the M128 and BT=16
    suites, whose head count is free."""
    seed(26)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(1024)
        _check_case(
            "m64 long_sequence T=1024", ctx, inp, use_initial_state=True
        )


def test_m64_single_chunk() raises:
    seed(22)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(32)
        _check_case("m64 single chunk T=32", ctx, inp)


def test_m64_single_token() raises:
    """One token: 31 of the 32 chunk slots masked off."""
    seed(31)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(1)
        _check_case("m64 single_token T=1", ctx, inp)


def test_m64_chunk_plus_one() raises:
    """T=33: a single token in the second chunk."""
    seed(32)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(33)
        _check_case("m64 chunk_plus_one T=33", ctx, inp)


def test_m64_ragged_tail() raises:
    # T=100: three full 32-token chunks plus a 4-token ragged tail, which
    # exercises the load warp's zero-filled cp.async and the scalar-store
    # epilogue.
    seed(23)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(100)
        _check_case("m64 ragged_tail T=100", ctx, inp)


def test_m64_initial_state() raises:
    seed(24)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(256)
        _fill_bf16(inp.state0, H * 128 * 128, -0.5, 0.5)
        _check_case("m64 initial_state T=256", ctx, inp, use_initial_state=True)


def test_m64_no_store_final() raises:
    # store_final_state=0 must leave the final-state pool untouched; the
    # output must still be right.
    seed(25)
    with DeviceContext() as ctx:
        var inp = PrefillInputs(128)
        var got = _run_prefill(
            ctx,
            inp,
            use_initial_state=False,
            store_final_state=False,
            final_sentinel=7.0,
        )
        var want = _run_reference(inp)
        var n_qkv = inp.t_total * H * 128
        var n_state = H * 128 * 128
        var e_out = _rel_err(want[0], got[0], n_qkv)
        var untouched = True
        for i in range(n_state):
            if got[1][i] != 7.0:
                untouched = False
                break
        print("m64 no_store_final T=128 : rel_err out =", e_out)
        assert_true(e_out < OUT_TOL, "no_store_final: output rel_err too high")
        assert_true(untouched, "no_store_final: final-state pool was written")
        got[0].free()
        got[1].free()
        want[0].free()
        want[1].free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
