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
"""Benchmark for the SM100 KDA BT=16 two-kernel decomposition.

Runs the chunk-parallel factor-prep kernel (k1) and the serial chain kernel
(k2) on the H=64 fixed [8192] shape with the same methodology used for the
fused kernels (device events, 5 warmup iters, 30 timed iters, sync per
iter, median, no L2 flush).

Reference kernel-only device times on this box (nsys / torch-profiler
steady state) for the same shape:

    CuTe-DSL BT=16:  prep 0.170 + chain_dv2 0.255 = 0.425 total
    Mojo fused m64:  0.469     CUDA fused_m64: 0.479
    Mojo fused m128: 0.540     CUDA fused_m128: 0.504

The decomposition's win over the fused kernels comes from the chunk-parallel
prep saturating all SMs while the serial chain only has to run the small
recurrent update.
"""

from std.utils import IndexList
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor

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
comptime PREP_SMEM_BYTES = 44 * 1024


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


def _bench[H: Int](ctx: DeviceContext, seq_len: Int) raises:
    var num_heads = H
    var t_total = seq_len
    var num_chunks = (seq_len + 15) // 16
    var n_qkv = t_total * num_heads * 128
    var n_state = num_heads * 128 * 128
    var ws_rows = num_chunks * 16
    var n_tile = num_heads * ws_rows * 128

    var d_q = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_k = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_v = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_g = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_beta = ctx.enqueue_create_buffer[DType.bfloat16](t_total * num_heads)
    var d_alog = ctx.enqueue_create_buffer[DType.float32](num_heads)
    var d_dt = ctx.enqueue_create_buffer[DType.float32](num_heads * 128)
    var d_cu = ctx.enqueue_create_buffer[DType.int64](2)
    var d_cuc = ctx.enqueue_create_buffer[DType.int32](2)
    var d_out = ctx.enqueue_create_buffer[DType.bfloat16](n_qkv)
    var d_state0 = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_state_final = ctx.enqueue_create_buffer[DType.bfloat16](n_state)
    var d_ws_kd = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_qd = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_w = ctx.enqueue_create_buffer[DType.bfloat16](n_tile)
    var d_ws_qk = ctx.enqueue_create_buffer[DType.bfloat16](
        num_heads * num_chunks * 256
    )
    var d_ws_diag = ctx.enqueue_create_buffer[DType.float32](
        num_heads * num_chunks * 128
    )

    var host_qkv = alloc[BF16](n_qkv)
    _fill_bf16(host_qkv, n_qkv, 0x9E3779B97F4A7C15)
    ctx.enqueue_copy(d_q, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0xBF58476D1CE4E5B9)
    ctx.enqueue_copy(d_k, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0x94D049BB133111EB)
    ctx.enqueue_copy(d_v, host_qkv)
    _fill_bf16(host_qkv, n_qkv, 0xD6E8FEB86659FD93)
    ctx.enqueue_copy(d_g, host_qkv)
    var host_beta = alloc[BF16](t_total * num_heads)
    _fill_bf16(host_beta, t_total * num_heads, 0xA0761D6478BD642F)
    ctx.enqueue_copy(d_beta, host_beta)
    var host_f32 = alloc[Float32](num_heads * 128)
    _fill_f32(host_f32, num_heads, 0xE7037ED1A0B428DB)
    ctx.enqueue_copy(d_alog, host_f32)
    _fill_f32(host_f32, num_heads * 128, 0x8EBC6AF09C88C6E3)
    ctx.enqueue_copy(d_dt, host_f32)
    var host_cu = alloc[Int64](2)
    host_cu[0] = 0
    host_cu[1] = Int64(t_total)
    ctx.enqueue_copy(d_cu, host_cu)
    var host_cuc = alloc[Int32](2)
    host_cuc[0] = 0
    host_cuc[1] = Int32(num_chunks)
    ctx.enqueue_copy(d_cuc, host_cuc)
    d_state0.enqueue_fill(BF16(0))

    var h128 = num_heads * 128
    var q_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_q,
        IndexList[4](1, num_heads, t_total, 128),
        IndexList[4](t_total * h128, 128, h128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var k_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_k,
        IndexList[4](1, num_heads, t_total, 128),
        IndexList[4](t_total * h128, 128, h128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var g_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_g,
        IndexList[4](1, num_heads, t_total, 128),
        IndexList[4](t_total * h128, 128, h128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var ws_kd_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_kd,
        IndexList[4](1, num_heads, ws_rows, 128),
        IndexList[4](ws_rows * num_heads * 128, ws_rows * 128, 128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var ws_qd_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_qd,
        IndexList[4](1, num_heads, ws_rows, 128),
        IndexList[4](ws_rows * num_heads * 128, ws_rows * 128, 128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var ws_w_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_ws_w,
        IndexList[4](1, num_heads, ws_rows, 128),
        IndexList[4](ws_rows * num_heads * 128, ws_rows * 128, 128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var v_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_v,
        IndexList[4](1, num_heads, t_total, 128),
        IndexList[4](t_total * h128, 128, h128, 1),
        IndexList[4](1, 1, 16, 64),
    )
    var diag_desc = create_tma_descriptor[
        DType.float32, 4, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_ws_diag,
        IndexList[4](1, num_heads, num_chunks, 128),
        IndexList[4](num_chunks * num_heads * 128, num_chunks * 128, 128, 1),
        IndexList[4](1, 1, 1, 128),
    )
    var qk_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_NONE
    ](
        d_ws_qk,
        IndexList[4](1, num_heads, num_chunks, 256),
        IndexList[4](num_chunks * num_heads * 256, num_chunks * 256, 256, 1),
        IndexList[4](1, 1, 1, 256),
    )
    var o_desc = create_tma_descriptor[
        DType.bfloat16, 4, TensorMapSwizzle.SWIZZLE_128B
    ](
        d_out,
        IndexList[4](1, num_heads, t_total, 128),
        IndexList[4](t_total * h128, 128, h128, 1),
        IndexList[4](1, 1, 16, 64),
    )

    var q_tma = QkGateTmaTile(q_desc)
    var k_tma = QkGateTmaTile(k_desc)
    var g_tma = QkGateTmaTile(g_desc)
    var ws_kd_tma = WsTileTmaTile(ws_kd_desc)
    var ws_qd_tma = WsTileTmaTile(ws_qd_desc)
    var ws_w_tma = WsTileTmaTile(ws_w_desc)
    var kd_tma = ChainWsTmaTile(ws_kd_desc)
    var w_tma = ChainWsTmaTile(ws_w_desc)
    var qd_tma = ChainWsTmaTile(ws_qd_desc)
    var v_tma = ChainVTmaTile(v_desc)
    var diag_tma = ChainDiagTmaTile(diag_desc)
    var qk_tma = ChainQkTmaTile(qk_desc)
    var o_tma = ChainVTmaTile(o_desc)

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
    var cuc_t = TileTensor[linear_idx_type=DType.int32](
        d_cuc.unsafe_ptr().as_unsafe_any_origin(), row_major(2)
    )
    var ws_qk_t = TileTensor[linear_idx_type=DType.int32](
        d_ws_qk.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], num_chunks, Idx[256])),
    )
    var ws_diag_t = TileTensor[linear_idx_type=DType.int32](
        d_ws_diag.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Idx[H], num_chunks, Idx[128])),
    )
    var out_t = TileTensor[linear_idx_type=DType.int32](
        d_out.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(t_total, Idx[H], Idx[128])),
    )
    var state0_t = TileTensor[linear_idx_type=DType.int32](
        d_state0.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[H], Idx[128], Idx[128])),
    )
    var final_t = TileTensor[linear_idx_type=DType.int32](
        d_state_final.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(1, Idx[H], Idx[128], Idx[128])),
    )

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
    comptime chain = sm100_kda_chain_bt16[
        H,
        QkvLayout=type_of(out_t).LayoutType,
        I64Layout=type_of(cu_t).LayoutType,
        I32Layout=type_of(cuc_t).LayoutType,
        StateLayout=type_of(state0_t).LayoutType,
        TensorOrigin=type_of(out_t).origin,
    ]

    @always_inline
    def launch_prep(
        lctx: DeviceContext,
    ) raises {
        imm q_tma,
        imm k_tma,
        imm g_tma,
        imm ws_kd_tma,
        imm ws_qd_tma,
        imm ws_w_tma,
        imm beta_t,
        imm alog_t,
        imm dt_t,
        imm cu_t,
        imm cuc_t,
        imm ws_qk_t,
        imm ws_diag_t,
        imm num_chunks,
    }:
        lctx.enqueue_function[prep](
            q_tma,
            k_tma,
            g_tma,
            ws_kd_tma,
            ws_qd_tma,
            ws_w_tma,
            beta_t,
            alog_t,
            dt_t,
            cu_t,
            cuc_t,
            ws_qk_t,
            ws_diag_t,
            Int32(num_chunks),
            Int32(1),
            Float32(-5.0),
            grid_dim=((num_chunks + 3) // 4, H),
            block_dim=(128,),
            shared_mem_bytes=PREP_SMEM_BYTES,
        )

    @always_inline
    def launch_chain(
        lctx: DeviceContext,
    ) raises {
        imm kd_tma,
        imm w_tma,
        imm qd_tma,
        imm v_tma,
        imm diag_tma,
        imm qk_tma,
        imm o_tma,
        imm out_t,
        imm cu_t,
        imm cuc_t,
        imm state0_t,
        imm final_t,
    }:
        lctx.enqueue_function[chain](
            kd_tma,
            w_tma,
            qd_tma,
            v_tma,
            diag_tma,
            qk_tma,
            o_tma,
            out_t,
            cu_t,
            cuc_t,
            state0_t,
            final_t,
            Int32(1),
            Int32(1),
            Float32(0.08838834764831845),
            grid_dim=(1, H * 2),
            block_dim=(512,),
            shared_mem_bytes=CHAIN_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(CHAIN_SMEM_BYTES)
            ),
        )

    @always_inline
    def launch_both(
        lctx: DeviceContext,
    ) raises {imm launch_prep, imm launch_chain}:
        launch_prep(lctx)
        launch_chain(lctx)

    for _ in range(5):
        launch_both(ctx)
        ctx.synchronize()

    var t_prep = List[Float64]()
    var t_chain = List[Float64]()
    var t_both = List[Float64]()
    for _ in range(30):
        t_prep.append(Float64(ctx.execution_time(launch_prep, 1)) / 1e6)
        t_chain.append(Float64(ctx.execution_time(launch_chain, 1)) / 1e6)
        t_both.append(Float64(ctx.execution_time(launch_both, 1)) / 1e6)

    print(
        "H=" + String(H) + " fixed [" + String(seq_len) + "]   ",
        "prep",
        _median(t_prep),
        " chain",
        _median(t_chain),
        " total",
        _median(t_both),
    )

    host_qkv.free()
    host_beta.free()
    host_f32.free()
    host_cu.free()
    host_cuc.free()


def main() raises:
    var ctx = DeviceContext()
    print(
        "shape                  median_ms (CuTe-DSL bar: prep 0.170  chain"
        " 0.255  total 0.425)"
    )
    _bench[64](ctx, 8192)
    _bench[96](ctx, 8192)
