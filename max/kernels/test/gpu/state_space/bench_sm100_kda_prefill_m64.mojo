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
"""Benchmark for the SM100 KDA prefill kernel, M64 value-split variant.

Runs the one shape the M64 variant exists for -- a single fixed-layout
sequence with H=64, where the M128 kernel launches only 64 CTAs and starves a
148-SM B200 -- with the same methodology used to measure the reference CUDA
kernels on this box (device events, 5 warmup iters, 30 timed iters, sync per
iter, median, no L2 flush):

Two views of H=64 fixed [8192], because the eager brackets are not
kernel-comparable across runtimes:

  eager wall clock per call (events around the call, FlashInfer's own
  bench bracket -- includes each runtime's host-side launch path):
      Mojo m64: 0.474    CUDA fused_m64: 0.5675    CuTe-DSL: 0.569
      Mojo m128: 0.548   CUDA fused_m128: 0.614

  kernel-only device time (nsys / torch-profiler steady state; what a
  CUDA-graph deployment pays):
      Mojo m64: 0.469    CUDA fused_m64: 0.479     CuTe-DSL: 0.425
      Mojo m128: 0.540   CUDA fused_m128: 0.504    (CuTe-DSL = prep
      0.170 + chain_dv2 0.255, its two-kernel chain, BT=16)

The FlashInfer eager Python wrapper spends ~90-140 us of host time inside
its event bracket (the stream idles between the start event and the kernel
launch), while this harness's bracket is a single enqueue (~5 us of launch
overhead), so cross-runtime claims should quote the kernel-only row.
Within one harness the bracket is still a valid regression gate.
"""

from std.utils import IndexList
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor

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


def _fill_bf16(
    ptr: UnsafePointer[BF16, MutUntrackedOrigin], n: Int, seed: UInt64
):
    # xorshift64* pseudo-normal-ish fill: average of two uniforms in [-1, 1).
    var s = seed
    for i in range(n):
        s ^= s >> 12
        s ^= s << 25
        s ^= s >> 27
        var r = s * 0x2545F4914F6CDD1D
        var u1 = Float32(Int32(r & 0xFFFF)) / 32768.0 - 1.0
        var u2 = Float32(Int32((r >> 16) & 0xFFFF)) / 32768.0 - 1.0
        ptr[i] = BF16((u1 + u2) * 0.7)


def _fill_f32(
    ptr: UnsafePointer[Float32, MutUntrackedOrigin], n: Int, seed: UInt64
):
    var s = seed
    for i in range(n):
        s ^= s >> 12
        s ^= s << 25
        s ^= s >> 27
        var r = s * 0x2545F4914F6CDD1D
        ptr[i] = Float32(Int32(r & 0xFFFF)) / 32768.0 - 1.0


def _median(times: List[Float64]) -> Float64:
    # Insertion sort into a copy; 30 elements.
    var s = List[Float64]()
    for t in times:
        var inserted = False
        for j in range(len(s)):
            if t < s[j]:
                s.insert(j, t)
                inserted = True
                break
        if not inserted:
            s.append(t)
    return s[len(s) // 2]


def _bench_shape[
    H: Int
](ctx: DeviceContext, num_seqs: Int, seq_len: Int) raises -> Float64:
    # B (num_seqs) and T are runtime; H and D are compile time.
    var num_heads = H
    var t_total = num_seqs * seq_len
    var n_qkv = t_total * num_heads * 128
    var n_beta = t_total * num_heads
    var n_state = num_seqs * num_heads * 128 * 128

    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](n_beta)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](num_heads)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](num_heads * 128)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](num_seqs + 1)
    var d_seqorder = ctx.enqueue_create_buffer[DType.int32](num_seqs)
    var d_state = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_out = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)

    var host_qkv = alloc[BF16](n_qkv)
    _fill_bf16(host_qkv, n_qkv, 0x9E3779B97F4A7C15)
    ctx.enqueue_copy(d_q, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0xBF58476D1CE4E5B9)
    ctx.enqueue_copy(d_k, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0x94D049BB133111EB)
    ctx.enqueue_copy(d_v, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0xD6E8FEB86659FD93)
    ctx.enqueue_copy(d_g, host_qkv)

    var host_beta = alloc[BF16](n_beta)
    _fill_bf16(host_beta, n_beta, 0xA0761D6478BD642F)
    ctx.enqueue_copy(d_beta, host_beta)

    var host_f32 = alloc[Float32](num_heads * 128)
    _fill_f32(host_f32, num_heads, 0xE7037ED1A0B428DB)
    ctx.enqueue_copy(d_alog, host_f32)
    _fill_f32(host_f32, num_heads * 128, 0x8EBC6AF09C88C6E3)
    ctx.enqueue_copy(d_dt, host_f32)

    var host_cu = alloc[Int64](num_seqs + 1)
    for i in range(num_seqs + 1):
        host_cu[i] = Int64(i * seq_len)
    ctx.enqueue_copy(d_cu, host_cu)

    var host_so = alloc[Int32](num_seqs)
    for i in range(num_seqs):
        host_so[i] = Int32(i)
    ctx.enqueue_copy(d_seqorder, host_so)

    d_state.enqueue_fill(BF16(0))

    var h128 = num_heads * 128
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](2, num_heads, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](2, num_heads, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](2, 1, 32, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_v,
        IndexList[3](t_total, num_heads, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 64),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 3, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_g,
        IndexList[3](t_total, num_heads, 128),
        IndexList[3](h128, 128, 1),
        IndexList[3](32, 1, 128),
    )
    var beta_desc = create_tma_descriptor[
        DType.bfloat16, 2, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_beta,
        IndexList[2](t_total, num_heads),
        IndexList[2](num_heads, 1),
        IndexList[2](32, 8),
    )
    var out_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](2, num_heads, t_total, 64),
        IndexList[4](64, 128, h128, 1),
        IndexList[4](1, 1, 32, 64),
    )
    # The descriptors travel by value as grid_constant kernel params, the
    # in-tree TMATensorTile way; no device-side descriptor buffer or fence.
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
        d_cu.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs + 1)
    )
    var so_t = TileTensor[linear_idx_type=DType.int32](
        d_seqorder.unsafe_ptr().as_unsafe_any_origin(), row_major(num_seqs)
    )
    var state_t = TileTensor[linear_idx_type=DType.int32](
        d_state.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(num_seqs, Idx[H], Idx[128], Idx[128])),
    )

    @always_inline
    def launch(
        lctx: DeviceContext,
    ) raises {
        imm q_t,
        imm k_t,
        imm v_t,
        imm g_t,
        imm beta_t,
        imm alog_t,
        imm dt_t,
        imm cu_t,
        imm so_t,
        imm state_t,
        imm out_tt,
        imm q_tma,
        imm k_tma,
        imm v_tma,
        imm g_tma,
        imm beta_tma,
        imm out_tma,
        imm num_seqs,
    }:
        comptime kernel = sm100_kda_prefill_m64[
            H,
            QkvLayout=type_of(q_t).LayoutType,
            BetaLayout=type_of(beta_t).LayoutType,
            HVecLayout=type_of(alog_t).LayoutType,
            DtLayout=type_of(dt_t).LayoutType,
            I64Layout=type_of(cu_t).LayoutType,
            I32Layout=type_of(so_t).LayoutType,
            StateLayout=type_of(state_t).LayoutType,
            TensorOrigin=type_of(q_t).origin,
        ]
        lctx.enqueue_function[kernel](
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
            state_t,
            out_tt,
            out_tma,
            state_t,
            Int32(1),
            Int32(1),
            Float32(0.08838834764831845),
            Float32(-5.0),
            grid_dim=(2 * num_seqs * H,),  # two CTAs per task: the value split
            block_dim=(1024,),
            shared_mem_bytes=SMEM_TOTAL,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(SMEM_TOTAL)
            ),
        )

    for _ in range(5):
        launch(ctx)
        ctx.synchronize()

    var times = List[Float64]()
    for _ in range(30):
        var ns = ctx.execution_time(launch, 1)
        times.append(Float64(ns) / 1e6)
    var med = _median(times)

    host_qkv.free()
    host_beta.free()
    host_f32.free()
    host_cu.free()
    host_so.free()
    return med


def main() raises:
    var ctx = DeviceContext()
    print(
        "shape                         median_ms   cuda_m64_ms   mojo_m128_ms"
    )
    var m1 = _bench_shape[64](ctx, 1, 8192)
    print("H=64 fixed [8192]           ", m1, "  0.5675   0.548")
