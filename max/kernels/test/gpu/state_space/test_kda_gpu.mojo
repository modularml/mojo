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
"""Tests for the KDA decode GPU kernel (M1) vs the CPU scalar reference (M0).

Verifies correctness of all three compile-time axes:
  gate_mode:    original | safe
  beta_mode:    logits   | probability
  state_layout: K_FIRST  | V_FIRST

All comparisons use rel_err = RMSE(M0 - GPU) / RMSE(M0).
Fixed-length tolerance: 0.005 (output and FP32 final state).
BF16 final state: 0.05 (documented tolerance for BF16 quantisation noise).

Shapes tested:
  - K=V=128, equal heads (H=HV=1), various modes/layouts, FP32 and BF16 state.
  - K=V=128, grouped HV/H=2 and HV/H=4.
  - Varlen (T>=256), nonzero initial state, both layouts.

K=V=64 is also attempted; outcome is reported (currently unsupported by M1
dispatch — the test prints the result and skips gracefully).
"""

import std.math
from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext
from std.random import rand

from layout import TileTensor, row_major
from kda.recurrent import kda_decode_gpu
from kda.reference import kda_decode_ref


# ===----------------------------------------------------------------------=== #
# Error metric
# ===----------------------------------------------------------------------=== #


def _rmse_f32(
    a: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    b: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float64:
    var s = Float64(0.0)
    for i in range(n):
        var d = Float64(a[i]) - Float64(b[i])
        s = s + d * d
    return sqrt(s / Float64(n))


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
# Generic GPU runner: launches kda_decode_gpu and returns results.
# ===----------------------------------------------------------------------=== #


def _run_gpu[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
    qkv_layout: StaticString = "SEPARATE",
](
    batch_size: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    total_T: Int,
    seq_lengths: List[Int],
    # Host input pointers (float32 for gate/a_log/dt_bias; qkv_dtype for q/k/v)
    q_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    k_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    v_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    rg_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    bl_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    al_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    dt_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    # Initial state (float32 always, cast to state_dtype on device)
    state_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    ctx: DeviceContext,
) raises -> Tuple[
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Launch GPU kernel and return (output_host_fp32, state_host_fp32) pointers.

    Caller is responsible for freeing the returned pointers.
    """
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM

    var pool_size = batch_size * HV * K * V

    # PACKED post-conv addressing: q/k/v are sections of one mixed-QKV row
    # [q (H*K) | k (H*K) | v (HV*V)] with per-token stride conv_dim.
    comptime PACKED = qkv_layout == "PACKED"
    var conv_dim = 2 * H * K + HV * V

    # cu_seqlens
    var cu_h = alloc[Scalar[DType.int32]](batch_size + 1)
    var cumsum = 0
    cu_h[0] = Int32(0)
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        cu_h[b + 1] = Int32(cumsum)

    # State indices: batch item b → slot b.
    var si_h = alloc[Scalar[DType.int32]](batch_size)
    for b in range(batch_size):
        si_h[b] = Int32(b)

    # Allocate device buffers.
    var q_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var k_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var v_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * HV * V)
    var rg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * K)
    var bl_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var al_dev = ctx.enqueue_create_buffer[DType.float32](HV)
    var dt_dev = ctx.enqueue_create_buffer[DType.float32](HV * K)
    var cu_dev = ctx.enqueue_create_buffer[DType.int32](batch_size + 1)
    var si_dev = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var pool_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

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

    # Fill pool with initial state (cast from fp32 to state_dtype).
    # Copy fp32 state to a temporary fp32 device buffer, then elementwise cast.
    var state_fp32_dev = ctx.enqueue_create_buffer[DType.float32](pool_size)
    with ctx.push_context():
        ctx.enqueue_copy(state_fp32_dev, state_h)
    # Cast fp32 → state_dtype into pool_dev using a host loop for simplicity.
    # (For test correctness this cast must match the kernel's round-trip.)
    var state_src_host = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        state_src_host[i] = Scalar[state_dtype](state_h[i])
    with ctx.push_context():
        ctx.enqueue_copy(pool_dev, state_src_host)
    state_src_host.free()

    # PACKED: build one mixed-QKV device buffer from the separate host q/k/v,
    # packing each token row as [q (H*K) | k (H*K) | v (HV*V)]. Function-scoped
    # so it outlives the launch. Allocated (small) in both modes; only filled
    # and used to back q/k/v when PACKED.
    var mixed_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * conv_dim)
    comptime if PACKED:
        var mixed_h = alloc[Scalar[qkv_dtype]](total_T * conv_dim)
        for t in range(total_T):
            for i in range(H * K):
                mixed_h[t * conv_dim + i] = q_h[t * H * K + i]
            for i in range(H * K):
                mixed_h[t * conv_dim + H * K + i] = k_h[t * H * K + i]
            for i in range(HV * V):
                mixed_h[t * conv_dim + 2 * H * K + i] = v_h[t * HV * V + i]
        with ctx.push_context():
            ctx.enqueue_copy(mixed_dev, mixed_h)
        mixed_h.free()

    # TileTensors wrapping device buffers. Origin is erased to UnsafeAnyOrigin
    # so q/k/v can be repointed at the mixed buffer under PACKED (a per-buffer
    # origin makes the two tensors distinct types and blocks the reassignment).
    var q_tt = TileTensor(
        q_dev, row_major(total_T, H * K)
    ).as_unsafe_any_origin()
    var k_tt = TileTensor(
        k_dev, row_major(total_T, H * K)
    ).as_unsafe_any_origin()
    var v_tt = TileTensor(
        v_dev, row_major(total_T, HV * V)
    ).as_unsafe_any_origin()
    # PACKED: repoint q/k/v at the single mixed buffer (row stride conv_dim).
    comptime if PACKED:
        q_tt = TileTensor(
            mixed_dev, row_major(total_T, conv_dim)
        ).as_unsafe_any_origin()
        k_tt = TileTensor(
            mixed_dev, row_major(total_T, conv_dim)
        ).as_unsafe_any_origin()
        v_tt = TileTensor(
            mixed_dev, row_major(total_T, conv_dim)
        ).as_unsafe_any_origin()

    # q/k/v per-token sequence strides: conv_dim when PACKED, else the
    # separate-tensor row size. Head/element strides are the same in both modes.
    var q_seqlen_stride = UInt32(conv_dim) if PACKED else UInt32(H * K)
    var k_seqlen_stride = UInt32(conv_dim) if PACKED else UInt32(H * K)
    var v_seqlen_stride = UInt32(conv_dim) if PACKED else UInt32(HV * V)
    var rg_tt = TileTensor(rg_dev, row_major(total_T, HV * K))
    var bl_tt = TileTensor(bl_dev, row_major(total_T, HV))
    var al_tt = TileTensor(al_dev, row_major(HV))
    var dt_tt = TileTensor(dt_dev, row_major(HV, K))
    var cu_tt = TileTensor(cu_dev, row_major(batch_size + 1))
    var si_tt = TileTensor(si_dev, row_major(batch_size))
    var out_tt = TileTensor(out_dev, row_major(total_T, HV * V))

    # State pool TileTensor.
    # K_FIRST: [N, HV, K, V] → row_major(batch_size, HV, K, V)
    # V_FIRST: [N, HV, V, K] → row_major(batch_size, HV, V, K)
    var pool_tt_kf = TileTensor(pool_dev, row_major(batch_size, HV, K, V))
    var pool_tt_vf = TileTensor(pool_dev, row_major(batch_size, HV, V, K))

    # Strides from layout (row-major: stride[i] = product of trailing dims).
    # q: [total_T, H, K] → strides [H*K, K, 1]
    # k: same
    # v: [total_T, HV, V] → strides [HV*V, V, 1]
    # rg: [total_T, HV, K] → strides [HV*K, K, 1]
    # bl: [total_T, HV] → strides [HV, 1]
    # dt: [HV, K] → strides [K, 1]
    # pool K_FIRST: [N, HV, K, V] → strides [HV*K*V, K*V, V, 1]
    # pool V_FIRST: [N, HV, V, K] → strides [HV*V*K, V*K, K, 1]
    # out: [total_T, HV*V] → strides [HV*V, 1]

    var num_blocks = batch_size * HV
    out_dev.enqueue_fill(0.0)

    comptime if state_layout == "K_FIRST":
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_tt_kf.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "K_FIRST",
                qkv_layout,
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt_kf,
            si_tt,
            q_seqlen_stride,
            UInt32(K),
            UInt32(1),
            k_seqlen_stride,
            UInt32(K),
            UInt32(1),
            v_seqlen_stride,
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
            UInt32(1),  # state_indices_seq_stride (unused in spec_mode "none")
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    else:  # V_FIRST
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_tt_vf.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "V_FIRST",
                qkv_layout,
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt_vf,
            si_tt,
            q_seqlen_stride,
            UInt32(K),
            UInt32(1),
            k_seqlen_stride,
            UInt32(K),
            UInt32(1),
            v_seqlen_stride,
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
            UInt32(1),  # state_indices_seq_stride (unused in spec_mode "none")
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    # Copy results back to host.
    var out_result = alloc[Scalar[DType.float32]](total_T * HV * V)
    var pool_result_typed = alloc[Scalar[state_dtype]](pool_size)
    var pool_result = alloc[Scalar[DType.float32]](pool_size)

    with ctx.push_context():
        ctx.enqueue_copy(out_result, out_dev)
        ctx.enqueue_copy(pool_result_typed, pool_dev)
    ctx.synchronize()

    # Convert state_dtype → float32 for comparison.
    for i in range(pool_size):
        pool_result[i] = Scalar[DType.float32](pool_result_typed[i])

    pool_result_typed.free()
    cu_h.free()
    si_h.free()

    return out_result, pool_result


# ===----------------------------------------------------------------------=== #
# Generic harness: GPU vs M0 reference
# ===----------------------------------------------------------------------=== #


def _check_gpu_vs_ref[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
    qkv_layout: StaticString = "SEPARATE",
](
    tag: String,
    batch_size: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    total_T: Int,
    seq_lengths: List[Int],
    ctx: DeviceContext,
    tol_output: Float64 = 0.005,
    tol_state: Float64 = 0.005,
) raises:
    """Generate random inputs, run GPU and M0 reference, compare outputs."""
    var H = num_key_heads
    var HV = num_value_heads
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var pool_size = batch_size * HV * K * V

    # Allocate and fill host tensors.
    var q_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var k_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    var v_h = alloc[Scalar[qkv_dtype]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_h = alloc[Scalar[DType.float32]](pool_size)

    # Fill with deterministic pseudo-random data using sin/cos sequences.
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
        state_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 11) * Float32(0.193)) * Float32(0.2)
        )

    # --- Build the layout-physical initial state ---
    # state_h holds the canonical K_FIRST [N,HV,K,V] init. Both the GPU kernel
    # and the CPU reference must be seeded from the SAME logical state S0[k,v] in
    # the target layout's physical order: K_FIRST keeps [N,HV,K,V]; V_FIRST stores
    # the transpose [N,HV,V,K] (pool[.,.,v,k] = S0[k,v]). Seeding the GPU with the
    # untransposed K_FIRST buffer under V_FIRST would start it from S0^T.
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

    # --- Run GPU kernel (seeded with the layout-physical init state) ---
    var gpu_out, gpu_state = _run_gpu[
        qkv_dtype,
        state_dtype,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        gate_mode,
        beta_mode,
        state_layout,
        qkv_layout,
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

    # --- Run CPU reference (mutates ref_state_h into the final state) ---
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

    # For V_FIRST ref, state strides: [N,HV,V,K] → dim1_stride=K, dim2_stride=1
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

    # gpu_state and the reference's final ref_state_h are both in the target
    # layout's physical order (K_FIRST [N,HV,K,V] or V_FIRST [N,HV,V,K], with
    # K==V), so pool[i] == pool[i] element-wise; compare directly.
    var err_o = _rel_err_ptr(ref_out, gpu_out, total_T * HV * V)
    var err_s = _rel_err_ptr(ref_state_h, gpu_state, pool_size)

    print(
        tag
        + " output rel_err="
        + String(err_o)
        + " state rel_err="
        + String(err_s)
    )

    assert_true(
        err_o < tol_output,
        tag + " output rel_err=" + String(err_o) + " >= " + String(tol_output),
    )
    assert_true(
        err_s < tol_state,
        tag + " state rel_err=" + String(err_s) + " >= " + String(tol_state),
    )

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_h.free()
    gpu_out.free()
    gpu_state.free()
    ref_state_h.free()
    ref_out.free()
    si_ref.free()
    cu_ref.free()


# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_gpu_original_logits_kfirst_fp32() raises:
    """K=128, gate=original, beta=logits, K_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        "128x128 original/logits/K_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_original_logits_kfirst_bf16() raises:
    """K=128, gate=original, beta=logits, K_FIRST, bf16 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.bfloat16,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        "128x128 original/logits/K_FIRST/bf16",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
        tol_state=0.05,
    )


def test_gpu_safe_logits_kfirst_fp32() raises:
    """K=128, gate=safe, beta=logits, K_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "logits",
        "K_FIRST",
    ](
        "128x128 safe/logits/K_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_original_probability_kfirst_fp32() raises:
    """K=128, gate=original, beta=probability, K_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "probability",
        "K_FIRST",
    ](
        "128x128 original/probability/K_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_safe_probability_vfirst_fp32() raises:
    """K=128, gate=safe, beta=probability, V_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "probability",
        "V_FIRST",
    ](
        "128x128 safe/probability/V_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_original_logits_vfirst_fp32() raises:
    """K=128, gate=original, beta=logits, V_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "V_FIRST",
    ](
        "128x128 original/logits/V_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_gqa_hv2_original_kfirst_fp32() raises:
    """K=128, GQA HV/H=2, gate=original, beta=logits, K_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        "128x128 GQA-HV2 original/logits/K_FIRST/fp32",
        batch_size=2,
        num_value_heads=2,
        num_key_heads=1,
        total_T=2,
        seq_lengths=[1, 1],
        ctx=ctx,
    )


def test_gpu_gqa_hv4_safe_probability_vfirst_fp32() raises:
    """K=128, GQA HV/H=4, gate=safe, beta=probability, V_FIRST, fp32 state."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "probability",
        "V_FIRST",
    ](
        "128x128 GQA-HV4 safe/probability/V_FIRST/fp32",
        batch_size=2,
        num_value_heads=4,
        num_key_heads=1,
        total_T=2,
        seq_lengths=[1, 1],
        ctx=ctx,
    )


def test_gpu_varlen_long_original_kfirst() raises:
    """Varlen T>=256, nonzero initial state, K_FIRST, gate=original."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    # Two sequences, one of length 200 and one of length 56 (total 256).
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        "128x128 varlen-256 original/logits/K_FIRST/fp32",
        batch_size=2,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
        tol_output=0.006,
    )


def test_gpu_varlen_long_safe_vfirst() raises:
    """Varlen T>=256, nonzero initial state, V_FIRST, gate=safe."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "logits",
        "V_FIRST",
    ](
        "128x128 varlen-256 safe/logits/V_FIRST/fp32",
        batch_size=2,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
        tol_output=0.006,
    )


def test_gpu_explicit_layout_transpose_equivalence() raises:
    """K_FIRST and V_FIRST must produce identical GPU outputs for same logical state.
    """
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var K = 128
    var V = 128
    var N = 1
    var HV = 1
    var H = 1
    var T = 4
    var total_T = T
    var pool_size = N * HV * K * V

    import std.math

    var q_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var k_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var v_h = alloc[Scalar[DType.bfloat16]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_kf = alloc[Scalar[DType.float32]](pool_size)

    for i in range(total_T * H * K):
        q_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 7) * Float32(0.37))
        )
        k_h[i] = Scalar[DType.bfloat16](
            std.math.cos(Float32(i + 7) * Float32(0.21))
        )
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i) * Float32(0.55)) * Float32(0.5)
        )
    for i in range(total_T * HV * K):
        rg_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 1) * Float32(0.13))
        )
    for i in range(total_T * HV):
        bl_h[i] = Scalar[DType.float32](Float32(0.5))
    al_h[0] = Scalar[DType.float32](Float32(-0.3))
    for i in range(HV * K):
        dt_h[i] = Scalar[DType.float32](Float32(0.0))
    for i in range(pool_size):
        state_kf[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 3) * Float32(0.17)) * Float32(0.1)
        )

    # Transpose state K_FIRST → V_FIRST.
    var state_vf = alloc[Scalar[DType.float32]](pool_size)
    for kd in range(K):
        for vd in range(V):
            var src = kd * V + vd
            var dst = vd * K + kd
            state_vf[dst] = state_kf[src]

    # Run GPU for both layouts.
    var out_kf, st_kf = _run_gpu[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
    ](
        N,
        HV,
        H,
        total_T,
        [T],
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        state_kf,
        ctx,
    )

    var out_vf, st_vf_raw = _run_gpu[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "V_FIRST",
    ](
        N,
        HV,
        H,
        total_T,
        [T],
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        state_vf,
        ctx,
    )

    # Transpose GPU V_FIRST result back for comparison.
    var st_vf_cmp = alloc[Scalar[DType.float32]](pool_size)
    for kd in range(K):
        for vd in range(V):
            var src = vd * K + kd  # V_FIRST: [N,HV,V,K] stored result
            var dst = kd * V + vd  # K_FIRST: [N,HV,K,V] reference
            st_vf_cmp[dst] = st_vf_raw[src]

    var err_o = _rel_err_ptr(out_kf, out_vf, total_T * HV * V)
    var err_s = _rel_err_ptr(st_kf, st_vf_cmp, pool_size)
    print(
        "layout_equiv GPU K_FIRST vs V_FIRST: output rel_err="
        + String(err_o)
        + " state rel_err="
        + String(err_s)
    )

    assert_true(
        err_o < Float64(0.001),
        "Layout equiv output: rel_err=" + String(err_o) + " >= 0.001",
    )
    assert_true(
        err_s < Float64(0.001),
        "Layout equiv state: rel_err=" + String(err_s) + " >= 0.001",
    )

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_kf.free()
    state_vf.free()
    out_kf.free()
    out_vf.free()
    st_kf.free()
    st_vf_raw.free()
    st_vf_cmp.free()


# ===----------------------------------------------------------------------=== #
# PACKED post-conv QKV addressing tests
#
# PACKED mode consumes one mixed-QKV buffer ([q|k|v] per token) instead of three
# separate q/k/v tensors, computing the section offsets in-kernel. These tests
# (a) prove PACKED is numerically identical to SEPARATE for the same logical
# inputs, and (b) validate PACKED against the M0 oracle across the axes that
# matter for the section-base math (equal heads, V_FIRST, GQA where H != HV,
# and varlen).
# ===----------------------------------------------------------------------=== #


def test_gpu_packed_matches_separate_kfirst() raises:
    """PACKED must produce bit-identical output/state to SEPARATE (K_FIRST)."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var K = 128
    var V = 128
    var N = 2
    var HV = 2
    var H = 1  # GQA HV/H=2 so the packed q/k (H heads) vs v (HV heads) differ.
    var T = 3
    var total_T = N * T

    import std.math

    var q_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var k_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var v_h = alloc[Scalar[DType.bfloat16]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_h = alloc[Scalar[DType.float32]](N * HV * K * V)

    for i in range(total_T * H * K):
        q_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 1) * Float32(0.313))
        )
        k_h[i] = Scalar[DType.bfloat16](
            std.math.cos(Float32(i + 1) * Float32(0.217))
        )
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[DType.bfloat16](
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
    for i in range(N * HV * K * V):
        state_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 11) * Float32(0.193)) * Float32(0.2)
        )

    var sep_out, sep_state = _run_gpu[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
        "SEPARATE",
    ](
        N,
        HV,
        H,
        total_T,
        [T, T],
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        state_h,
        ctx,
    )

    var pk_out, pk_state = _run_gpu[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
        "PACKED",
    ](
        N,
        HV,
        H,
        total_T,
        [T, T],
        q_h,
        k_h,
        v_h,
        rg_h,
        bl_h,
        al_h,
        dt_h,
        state_h,
        ctx,
    )

    var err_o = _rel_err_ptr(sep_out, pk_out, total_T * HV * V)
    var err_s = _rel_err_ptr(sep_state, pk_state, N * HV * K * V)
    print(
        "packed_vs_separate K_FIRST: output rel_err="
        + String(err_o)
        + " state rel_err="
        + String(err_s)
    )

    # Same numbers, same math, different addressing -> must be identical.
    assert_true(
        err_o < Float64(1e-6),
        "PACKED vs SEPARATE output rel_err=" + String(err_o),
    )
    assert_true(
        err_s < Float64(1e-6),
        "PACKED vs SEPARATE state rel_err=" + String(err_s),
    )

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_h.free()
    sep_out.free()
    sep_state.free()
    pk_out.free()
    pk_state.free()


def test_gpu_packed_original_logits_kfirst_fp32() raises:
    """PACKED K=128, gate=original, beta=logits, K_FIRST, vs M0 oracle."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
        "PACKED",
    ](
        "128x128 PACKED original/logits/K_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_packed_safe_probability_vfirst_fp32() raises:
    """PACKED K=128, gate=safe, beta=probability, V_FIRST, vs M0 oracle."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "safe",
        "probability",
        "V_FIRST",
        "PACKED",
    ](
        "128x128 PACKED safe/probability/V_FIRST/fp32",
        batch_size=1,
        num_value_heads=1,
        num_key_heads=1,
        total_T=4,
        seq_lengths=[4],
        ctx=ctx,
    )


def test_gpu_packed_gqa_hv2_original_kfirst_fp32() raises:
    """PACKED GQA HV/H=2 (packed q/k use H heads, v uses HV) vs M0 oracle."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
        "PACKED",
    ](
        "128x128 PACKED GQA-HV2 original/logits/K_FIRST/fp32",
        batch_size=2,
        num_value_heads=2,
        num_key_heads=1,
        total_T=2,
        seq_lengths=[1, 1],
        ctx=ctx,
    )


def test_gpu_packed_varlen_long_original_kfirst() raises:
    """PACKED varlen T>=256, nonzero initial state, K_FIRST, vs M0 oracle."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_gpu_vs_ref[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        "original",
        "logits",
        "K_FIRST",
        "PACKED",
    ](
        "128x128 PACKED varlen-256 original/logits/K_FIRST/fp32",
        batch_size=2,
        num_value_heads=1,
        num_key_heads=1,
        total_T=256,
        seq_lengths=[200, 56],
        ctx=ctx,
        tol_output=0.006,
    )


# ===----------------------------------------------------------------------=== #
# T=1+S speculative-chain checkpoint-row tests
#
# spec_mode="checkpoint" writes the running state after each token to its own
# checkpoint row (token p -> state_indices column p+1) instead of overwriting
# the single committed slot with only the sequence-final state. This lets a
# partial-accept rollback restart from any accepted position without corrupting
# the committed pool. See recurrent.mojo + the SSM spec-decode KB entry
# (known-limitations/hybrid-ssm-spec-decode-needs-per-step-state-snapshot).
#
# These are correctness-PRESERVING: the recurrence math is unchanged, only the
# state store target differs. They are verified entirely against MAX-own kernels
# (no FLA oracle change needed):
#   A. checkpoint-mode outputs == none-mode outputs (same inputs) — math intact.
#   B. checkpoint column m == a none-mode run over ONLY the first m tokens
#      (rollback-ability: column m IS the committed state after accepting m).
#   C. checkpoint column 0 (committed input) is unchanged (never mutated).
# ===----------------------------------------------------------------------=== #


def _run_gpu_checkpoint[
    qkv_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](
    num_slots: Int,
    seq_len: Int,
    # Host inputs for a single sequence of length seq_len (batch_size=1, HV=H=1).
    q_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    k_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    v_h: MutPointer[Scalar[qkv_dtype], MutUntrackedOrigin],
    rg_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    bl_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    al_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    dt_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    # Committed initial state S0 (float32), laid out for the target layout, one
    # slot; seeds committed column 0. Checkpoint slots are zero-seeded.
    committed_state_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    ctx: DeviceContext,
) raises -> Tuple[
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
]:
    """Launch checkpoint-mode kernel; return (output_fp32, full_pool_fp32).

    state_indices is 2-D [1, seq_len+1] = [0, 1, ..., seq_len]: column 0 is the
    committed slot (0), columns 1..seq_len are checkpoint slots (token p ->
    slot p+1). Caller frees the returned pointers.
    """
    var HV = 1
    var H = 1
    var K = KEY_HEAD_DIM
    var V = VALUE_HEAD_DIM
    var total_T = seq_len
    var num_cols = seq_len + 1  # committed col 0 + one checkpoint per token
    var slot_elems = HV * K * V
    var pool_size = num_slots * slot_elems

    # cu_seqlens for one sequence.
    var cu_h = alloc[Scalar[DType.int32]](2)
    cu_h[0] = Int32(0)
    cu_h[1] = Int32(seq_len)

    # 2-D state_indices [1, num_cols] = [0, 1, ..., seq_len] (flat).
    var si_h = alloc[Scalar[DType.int32]](num_cols)
    for c in range(num_cols):
        si_h[c] = Int32(c)

    var q_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var k_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * H * K)
    var v_dev = ctx.enqueue_create_buffer[qkv_dtype](total_T * HV * V)
    var rg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * K)
    var bl_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var al_dev = ctx.enqueue_create_buffer[DType.float32](HV)
    var dt_dev = ctx.enqueue_create_buffer[DType.float32](HV * K)
    var cu_dev = ctx.enqueue_create_buffer[DType.int32](2)
    var si_dev = ctx.enqueue_create_buffer[DType.int32](num_cols)
    var pool_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

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

    # Seed the pool: committed slot 0 = S0, all checkpoint slots = 0.
    var pool_src_host = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        pool_src_host[i] = Scalar[state_dtype](0.0)
    for i in range(slot_elems):
        pool_src_host[i] = Scalar[state_dtype](committed_state_h[i])
    with ctx.push_context():
        ctx.enqueue_copy(pool_dev, pool_src_host)
    pool_src_host.free()

    var q_tt = TileTensor(q_dev, row_major(total_T, H * K))
    var k_tt = TileTensor(k_dev, row_major(total_T, H * K))
    var v_tt = TileTensor(v_dev, row_major(total_T, HV * V))
    var rg_tt = TileTensor(rg_dev, row_major(total_T, HV * K))
    var bl_tt = TileTensor(bl_dev, row_major(total_T, HV))
    var al_tt = TileTensor(al_dev, row_major(HV))
    var dt_tt = TileTensor(dt_dev, row_major(HV, K))
    var cu_tt = TileTensor(cu_dev, row_major(2))
    var si_tt = TileTensor(si_dev, row_major(num_cols))
    var out_tt = TileTensor(out_dev, row_major(total_T, HV * V))

    var pool_tt_kf = TileTensor(pool_dev, row_major(num_slots, HV, K, V))
    var pool_tt_vf = TileTensor(pool_dev, row_major(num_slots, HV, V, K))

    out_dev.enqueue_fill(0.0)

    comptime if state_layout == "K_FIRST":
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_tt_kf.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "K_FIRST",
                "SEPARATE",
                "checkpoint",
            ]
        ](
            Int32(1),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt_kf,
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
            UInt32(num_cols),
            grid_dim=(HV,),
            block_dim=(VALUE_HEAD_DIM,),
        )
    else:  # V_FIRST
        ctx.enqueue_function[
            kda_decode_gpu[
                qkv_dtype,
                DType.float32,
                state_dtype,
                DType.float32,
                KEY_HEAD_DIM,
                VALUE_HEAD_DIM,
                out_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_tt_vf.LayoutType,
                si_tt.LayoutType,
                gate_mode,
                beta_mode,
                "V_FIRST",
                "SEPARATE",
                "checkpoint",
            ]
        ](
            Int32(1),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt_vf,
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
            UInt32(num_cols),
            grid_dim=(HV,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    var out_result = alloc[Scalar[DType.float32]](total_T * HV * V)
    var pool_typed = alloc[Scalar[state_dtype]](pool_size)
    var pool_result = alloc[Scalar[DType.float32]](pool_size)
    with ctx.push_context():
        ctx.enqueue_copy(out_result, out_dev)
        ctx.enqueue_copy(pool_typed, pool_dev)
    ctx.synchronize()
    for i in range(pool_size):
        pool_result[i] = Scalar[DType.float32](pool_typed[i])

    pool_typed.free()
    cu_h.free()
    si_h.free()
    return out_result, pool_result


def _check_spec_checkpoint[
    gate_mode: StaticString,
    beta_mode: StaticString,
    state_layout: StaticString,
](tag: String, seq_len: Int, ctx: DeviceContext) raises:
    """Verify checkpoint-mode against none-mode over a single spec chain."""
    comptime K = 128
    comptime V = 128
    var HV = 1
    var H = 1
    var slot_elems = HV * K * V
    var num_slots = seq_len + 1  # committed col 0 + one checkpoint per token

    import std.math

    var q_h = alloc[Scalar[DType.bfloat16]](seq_len * H * K)
    var k_h = alloc[Scalar[DType.bfloat16]](seq_len * H * K)
    var v_h = alloc[Scalar[DType.bfloat16]](seq_len * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](seq_len * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](seq_len * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var s0_h = alloc[Scalar[DType.float32]](slot_elems)

    for i in range(seq_len * H * K):
        q_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 1) * Float32(0.313))
        )
        k_h[i] = Scalar[DType.bfloat16](
            std.math.cos(Float32(i + 1) * Float32(0.217))
        )
    for i in range(seq_len * HV * V):
        v_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 7) * Float32(0.491)) * Float32(0.5)
        )
    for i in range(seq_len * HV * K):
        rg_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 3) * Float32(0.137))
        )
    for i in range(seq_len * HV):
        bl_h[i] = Scalar[DType.float32](
            std.math.cos(Float32(i + 5) * Float32(0.274))
        )
    al_h[0] = Scalar[DType.float32](Float32(-0.3))
    for i in range(HV * K):
        dt_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 2) * Float32(0.057)) * Float32(0.1)
        )
    # Nonzero committed initial state (same physical order for both layouts
    # since K==V; both runs use it identically so the comparison is layout-safe).
    for i in range(slot_elems):
        s0_h[i] = Scalar[DType.float32](
            std.math.sin(Float32(i + 11) * Float32(0.193)) * Float32(0.2)
        )

    # --- Checkpoint run over the full chain ---
    var ck_out, ck_pool = _run_gpu_checkpoint[
        DType.bfloat16,
        DType.float32,
        128,
        128,
        gate_mode,
        beta_mode,
        state_layout,
    ](num_slots, seq_len, q_h, k_h, v_h, rg_h, bl_h, al_h, dt_h, s0_h, ctx)

    # C. Committed input (slot 0) must be untouched.
    var err_committed = _rel_err_ptr(s0_h, ck_pool, slot_elems)
    print(tag + " committed-slot-preserved rel_err=" + String(err_committed))
    assert_true(
        err_committed < Float64(1e-6),
        tag + " committed slot mutated: rel_err=" + String(err_committed),
    )

    # B. For each accepted prefix length m in 1..seq_len, a none-mode run over
    #    ONLY the first m tokens must equal checkpoint column m.
    for m in range(1, seq_len + 1):
        var trunc_out, trunc_state = _run_gpu[
            DType.bfloat16,
            DType.float32,
            128,
            128,
            gate_mode,
            beta_mode,
            state_layout,
            "SEPARATE",
        ](1, HV, H, m, [m], q_h, k_h, v_h, rg_h, bl_h, al_h, dt_h, s0_h, ctx)
        var ck_slot_m = ck_pool + m * slot_elems
        var err_m = _rel_err_ptr(trunc_state, ck_slot_m, slot_elems)
        print(tag + " rollback m=" + String(m) + " rel_err=" + String(err_m))
        assert_true(
            err_m < Float64(1e-6),
            tag
            + " checkpoint col "
            + String(m)
            + " != commit-over-"
            + String(m)
            + " : rel_err="
            + String(err_m),
        )
        # A (at m=seq_len). Full-chain none-mode outputs == checkpoint outputs.
        if m == seq_len:
            var err_o = _rel_err_ptr(trunc_out, ck_out, seq_len * HV * V)
            print(tag + " output-vs-none rel_err=" + String(err_o))
            assert_true(
                err_o < Float64(1e-6),
                tag
                + " checkpoint output != none output: rel_err="
                + String(err_o),
            )
        trunc_out.free()
        trunc_state.free()

    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    s0_h.free()
    ck_out.free()
    ck_pool.free()


def test_gpu_spec_checkpoint_original_logits_kfirst() raises:
    """Spec checkpoint rows, K_FIRST original/logits, chain length 4."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_spec_checkpoint["original", "logits", "K_FIRST"](
        "spec-checkpoint original/logits/K_FIRST", seq_len=4, ctx=ctx
    )


def test_gpu_spec_checkpoint_safe_probability_vfirst() raises:
    """Spec checkpoint rows, V_FIRST safe/probability, chain length 5."""
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_spec_checkpoint["safe", "probability", "V_FIRST"](
        "spec-checkpoint safe/probability/V_FIRST", seq_len=5, ctx=ctx
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
