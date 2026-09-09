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
"""Benchmark for the in-tree KDA chunk-parallel prefill pipeline.

Times the L1/L2/L3 stages of `kda_chunk_{prepare,scan,output}_gpu` at the
shapes the SM100 ports are benchmarked on, with the same methodology as
`bench_sm100_kda_bt16.mojo` (device events, 5 warmup iters, 30 timed
iters, median, no L2 flush), so the two decompositions can be compared
stage by stage. CHUNK is 16 in both, which makes the comparison direct:

    L1 prepare  (grid num_chunks*HV)  <-> sm100_kda_prep_bt16   (grid C/4 x H)
    L2 scan     (grid N*HV)           <-> sm100_kda_chain_bt16  (grid N x 2H)
    L3 output   (grid num_chunks*HV)  <-> fused into the chain above

The launch sequence is lifted from `test_kda_chunk_parallel.mojo`'s
K_FIRST path so the two agree on setup; only the reference/decode
launches and the correctness comparison are dropped.

Note the two pipelines do NOT carry the same precision contract: this one
keeps its workspace, gate, beta and state in FP32, while the SM100
decomposition writes a BF16 workspace and a BF16 state. The workspace
traffic therefore differs by 2x, which matters for a bandwidth-bound
stage and is called out with the numbers rather than normalised away.
"""

import std.math
from std.utils import IndexList
from layout import TileTensor, row_major
from layout.tma_async import create_tma_tile
from max.gpu.host import DeviceContext

from max.gpu.host import FuncAttribute

from kda.chunk_fwd import (
    KDA_FUSED_PIPE_SMEM,
    KDA_FUSED_ROLE_BLOCK,
    kda_chunk_computebound_gpu,
    kda_chunk_output_gpu,
    kda_chunk_prepare_gpu,
    kda_chunk_scan_gpu,
    kda_chunk_seg_apply_gpu,
    kda_chunk_seg_reduce_gpu,
    kda_chunk_seg_scan_gpu,
)

comptime CHUNK: Int = 16
comptime K: Int = 128
comptime V: Int = 128


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


def _bench(
    ctx: DeviceContext, num_heads: Int, seq_len: Int, num_seqs: Int
) raises:
    comptime qkv_dtype = DType.bfloat16
    var HV = num_heads
    var H = num_heads
    var batch_size = num_seqs
    var total_T = num_seqs * seq_len
    var pool_size = batch_size * HV * K * V

    # Chunk map: seq_chunk_offsets [N+1], chunk_tok_offsets [C+1].
    var sco = List[Int32]()
    var cto = List[Int32]()
    sco.append(Int32(0))
    cto.append(Int32(0))
    var chunk_count = 0
    var tok_cursor = 0
    for _ in range(batch_size):
        var nc = (seq_len + CHUNK - 1) // CHUNK
        for c in range(nc):
            var rem = seq_len - c * CHUNK
            tok_cursor += CHUNK if rem >= CHUNK else rem
            cto.append(Int32(tok_cursor))
        chunk_count += nc
        sco.append(Int32(chunk_count))
    var num_chunks = len(cto) - 1

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
    var pool_dev = ctx.enqueue_create_buffer[DType.float32](pool_size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

    var nch = num_chunks * HV
    var ws_W_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_Uv_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_KE_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_glast_dev = ctx.enqueue_create_buffer[DType.float32](nch * K)
    var ws_P_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * K)
    var ws_d_dev = ctx.enqueue_create_buffer[DType.float32](nch * CHUNK * V)
    var ws_Sin_dev = ctx.enqueue_create_buffer[DType.float32](nch * K * V)
    # Segmented-path workspaces: SEG_LEN chunks per segment (her T=8192
    # correctness case uses 16).
    comptime SEG_LEN = 16
    var num_segments = num_chunks // SEG_LEN
    var nseg = num_segments * HV
    var ws_Mseg_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * K)
    var ws_cseg_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * V)
    var ws_Sseg_in_dev = ctx.enqueue_create_buffer[DType.float32](nseg * K * V)

    # Deterministic host fills, same trig seeding as the correctness test.
    var qk_h = alloc[Scalar[qkv_dtype]](total_T * H * K)
    for i in range(total_T * H * K):
        qk_h[i] = Scalar[qkv_dtype](std.math.sin(Float32(i + 1) * 0.313))
    ctx.enqueue_copy(q_dev, qk_h)
    for i in range(total_T * H * K):
        qk_h[i] = Scalar[qkv_dtype](std.math.cos(Float32(i + 1) * 0.217))
    ctx.enqueue_copy(k_dev, qk_h)
    var v_h = alloc[Scalar[qkv_dtype]](total_T * HV * V)
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[qkv_dtype](std.math.sin(Float32(i + 7) * 0.491) * 0.5)
    ctx.enqueue_copy(v_dev, v_h)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    for i in range(total_T * HV * K):
        rg_h[i] = std.math.sin(Float32(i + 3) * 0.137)
    ctx.enqueue_copy(rg_dev, rg_h)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    for i in range(total_T * HV):
        bl_h[i] = std.math.cos(Float32(i + 5) * 0.274)
    ctx.enqueue_copy(bl_dev, bl_h)
    var al_h = alloc[Scalar[DType.float32]](HV)
    for i in range(HV):
        al_h[i] = Float32(-0.5) + Float32(i) * 0.1
    ctx.enqueue_copy(al_dev, al_h)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    for i in range(HV * K):
        dt_h[i] = std.math.sin(Float32(i + 2) * 0.057) * 0.1
    ctx.enqueue_copy(dt_dev, dt_h)
    var si_h = alloc[Scalar[DType.int32]](batch_size)
    for b in range(batch_size):
        si_h[b] = Int32(b)
    ctx.enqueue_copy(si_dev, si_h)
    var sco_h = alloc[Scalar[DType.int32]](batch_size + 1)
    for i in range(batch_size + 1):
        sco_h[i] = sco[i]
    ctx.enqueue_copy(sco_dev, sco_h)
    var cto_h = alloc[Scalar[DType.int32]](num_chunks + 1)
    for i in range(num_chunks + 1):
        cto_h[i] = cto[i]
    ctx.enqueue_copy(cto_dev, cto_h)
    pool_dev.enqueue_fill(Float32(0))
    out_dev.enqueue_fill(Float32(0))

    # BF16 state and output, so the fused kernel is measured under the same
    # precision contract as the SM100 ports. Its gate stays FP32: the raw
    # gate lands in place over the kernel's FP32 cumsum, so a 2-byte gate is
    # not expressible, and that configuration measures within ~2% anyway.
    var pool_bf = ctx.enqueue_create_buffer[DType.bfloat16](pool_size)
    var out_bf = ctx.enqueue_create_buffer[DType.bfloat16](total_T * HV * V)
    pool_bf.enqueue_fill(Scalar[DType.bfloat16](0))
    out_bf.enqueue_fill(Scalar[DType.bfloat16](0))

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
    var out_tt = TileTensor(out_dev, row_major(total_T, HV * V))
    var pool_tt = TileTensor(pool_dev, row_major(batch_size, HV, K, V))

    var poolb_tt = TileTensor(pool_bf, row_major(batch_size, HV, K, V))
    var outb_tt = TileTensor(out_bf, row_major(total_T, HV * V))

    var ws_W_tt = TileTensor(ws_W_dev, row_major(nch * CHUNK * K))
    var ws_Uv_tt = TileTensor(ws_Uv_dev, row_major(nch * CHUNK * V))
    var ws_KE_tt = TileTensor(ws_KE_dev, row_major(nch * CHUNK * K))
    var ws_glast_tt = TileTensor(ws_glast_dev, row_major(nch * K))
    var ws_P_tt = TileTensor(ws_P_dev, row_major(nch * CHUNK * K))
    var ws_d_tt = TileTensor(ws_d_dev, row_major(nch * CHUNK * V))
    var ws_Sin_tt = TileTensor(ws_Sin_dev, row_major(nch * K * V))
    var ws_Mseg_tt = TileTensor(ws_Mseg_dev, row_major(nseg * K * K))
    var ws_cseg_tt = TileTensor(ws_cseg_dev, row_major(nseg * K * V))
    var ws_Sseg_in_tt = TileTensor(ws_Sseg_in_dev, row_major(nseg * K * V))

    var num_dec_blocks = batch_size * HV
    var num_chunk_blocks = num_chunks * HV
    var num_seg_blocks = num_segments * HV

    var q_tma = create_tma_tile[CHUNK, K](ctx, q_tt.to_layout_tensor())
    var k_tma = create_tma_tile[CHUNK, K](ctx, k_tt.to_layout_tensor())
    var rg_tma = create_tma_tile[CHUNK, K](ctx, rg_tt.to_layout_tensor())
    var bl_tma = create_tma_tile[CHUNK, 4](ctx, bl_tt.to_layout_tensor())
    var dt_tma = create_tma_tile[1, K](ctx, dt_tt.to_layout_tensor())
    var v_tma = create_tma_tile[CHUNK, V](ctx, v_tt.to_layout_tensor())
    var out_tma = create_tma_tile[CHUNK, V](ctx, out_tt.to_layout_tensor())
    var outb_tma = create_tma_tile[CHUNK, V](ctx, outb_tt.to_layout_tensor())

    @always_inline
    def launch_l1(
        lctx: DeviceContext,
    ) raises {
        imm q_tt,
        imm k_tt,
        imm v_tt,
        imm rg_tt,
        imm bl_tt,
        imm al_tt,
        imm dt_tt,
        imm cto_tt,
        imm q_tma,
        imm k_tma,
        imm rg_tma,
        imm bl_tma,
        imm dt_tma,
        imm ws_W_tt,
        imm ws_Uv_tt,
        imm ws_KE_tt,
        imm ws_glast_tt,
        imm ws_P_tt,
        imm ws_d_tt,
        imm HV,
        imm H,
        imm num_chunk_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_prepare_gpu[
                qkv_dtype,
                DType.float32,
                DType.float32,
                K,
                V,
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
            block_dim=(V,),
        )

    @always_inline
    def launch_l2(
        lctx: DeviceContext,
    ) raises {
        imm sco_tt,
        imm cto_tt,
        imm ws_W_tt,
        imm ws_Uv_tt,
        imm ws_KE_tt,
        imm ws_glast_tt,
        imm ws_Sin_tt,
        imm ws_P_tt,
        imm ws_d_tt,
        imm pool_tt,
        imm si_tt,
        imm out_tt,
        imm batch_size,
        imm HV,
        imm num_dec_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_scan_gpu[
                DType.float32,
                DType.float32,
                DType.float32,
                K,
                V,
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
                pool_tt.LayoutType,
                si_tt.LayoutType,
                out_tt.LayoutType,
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
            pool_tt,
            si_tt,
            out_tt,
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(V,),
        )

    @always_inline
    def launch_seg_reduce(
        lctx: DeviceContext,
    ) raises {
        imm ws_W_tt,
        imm ws_Uv_tt,
        imm ws_KE_tt,
        imm ws_glast_tt,
        imm ws_Mseg_tt,
        imm ws_cseg_tt,
        imm HV,
        imm num_seg_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_seg_reduce_gpu[
                DType.float32,
                K,
                V,
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
            Int32(SEG_LEN),
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Mseg_tt,
            ws_cseg_tt,
            grid_dim=(num_seg_blocks,),
            block_dim=(V,),
        )

    @always_inline
    def launch_seg_scan(
        lctx: DeviceContext,
    ) raises {
        imm ws_Mseg_tt,
        imm ws_cseg_tt,
        imm ws_Sseg_in_tt,
        imm pool_tt,
        imm si_tt,
        imm batch_size,
        imm HV,
        imm num_segments,
        imm num_dec_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_seg_scan_gpu[
                DType.float32,
                DType.float32,
                K,
                V,
                ws_Mseg_tt.LayoutType,
                ws_cseg_tt.LayoutType,
                ws_Sseg_in_tt.LayoutType,
                pool_tt.LayoutType,
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
            pool_tt,
            si_tt,
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_dec_blocks,),
            block_dim=(V,),
        )

    @always_inline
    def launch_seg_apply(
        lctx: DeviceContext,
    ) raises {
        imm ws_W_tt,
        imm ws_Uv_tt,
        imm ws_KE_tt,
        imm ws_glast_tt,
        imm ws_Sseg_in_tt,
        imm ws_Sin_tt,
        imm HV,
        imm num_seg_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_seg_apply_gpu[
                DType.float32,
                K,
                V,
                CHUNK,
                ws_W_tt.LayoutType,
                ws_Uv_tt.LayoutType,
                ws_KE_tt.LayoutType,
                ws_glast_tt.LayoutType,
                ws_Sseg_in_tt.LayoutType,
                ws_Sin_tt.LayoutType,
            ]
        ](
            Int32(HV),
            Int32(SEG_LEN),
            ws_W_tt,
            ws_Uv_tt,
            ws_KE_tt,
            ws_glast_tt,
            ws_Sseg_in_tt,
            ws_Sin_tt,
            grid_dim=(num_seg_blocks,),
            block_dim=(V,),
        )

    @always_inline
    def launch_l3(
        lctx: DeviceContext,
    ) raises {
        imm out_tt,
        imm cto_tt,
        imm ws_Sin_tt,
        imm ws_P_tt,
        imm ws_d_tt,
        imm HV,
        imm num_chunk_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_output_gpu[
                DType.float32,
                DType.float32,
                K,
                V,
                CHUNK,
                out_tt.LayoutType,
                cto_tt.LayoutType,
                ws_Sin_tt.LayoutType,
                ws_P_tt.LayoutType,
                ws_d_tt.LayoutType,
            ]
        ](
            Int32(HV),
            out_tt,
            cto_tt,
            ws_Sin_tt,
            ws_P_tt,
            ws_d_tt,
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            grid_dim=(num_chunk_blocks,),
            block_dim=(V,),
        )

    @always_inline
    def launch_fused(
        lctx: DeviceContext,
    ) raises {
        imm outb_tt,
        imm q_tt,
        imm k_tt,
        imm v_tt,
        imm rg_tt,
        imm bl_tt,
        imm al_tt,
        imm dt_tt,
        imm sco_tt,
        imm cto_tt,
        imm poolb_tt,
        imm si_tt,
        imm v_tma,
        imm q_tma,
        imm k_tma,
        imm rg_tma,
        imm outb_tma,
        imm batch_size,
        imm HV,
        imm H,
        imm num_dec_blocks,
    }:
        lctx.enqueue_function[
            kda_chunk_computebound_gpu[
                qkv_dtype,
                DType.float32,
                DType.bfloat16,
                DType.bfloat16,
                K,
                V,
                CHUNK,
                outb_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                sco_tt.LayoutType,
                cto_tt.LayoutType,
                poolb_tt.LayoutType,
                si_tt.LayoutType,
                "safe",
                "logits",
                "K_FIRST",
            ]
        ](
            Int32(batch_size),
            Int32(HV),
            Int32(H),
            outb_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            sco_tt,
            cto_tt,
            poolb_tt,
            si_tt,
            v_tma,
            q_tma,
            k_tma,
            rg_tma,
            outb_tma,
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
            grid_dim=(num_dec_blocks,),
            block_dim=(KDA_FUSED_ROLE_BLOCK[V],),
            shared_mem_bytes=KDA_FUSED_PIPE_SMEM[CHUNK, K],
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(KDA_FUSED_PIPE_SMEM[CHUNK, K])
            ),
        )

    for _ in range(5):
        launch_l1(ctx)
        launch_l2(ctx)
        launch_l3(ctx)
        launch_fused(ctx)
        ctx.synchronize()

    # The three stages are serial on one stream, so the pipeline total is
    # their sum plus two launch gaps; timing them separately keeps the
    # per-stage numbers comparable to the SM100 decomposition's.
    var t1 = List[Float64]()
    var t2 = List[Float64]()
    var t3 = List[Float64]()
    var tsr = List[Float64]()
    var tss = List[Float64]()
    var tsa = List[Float64]()
    var tf = List[Float64]()
    for _ in range(30):
        t1.append(Float64(ctx.execution_time(launch_l1, 1)) / 1e6)
        t2.append(Float64(ctx.execution_time(launch_l2, 1)) / 1e6)
        t3.append(Float64(ctx.execution_time(launch_l3, 1)) / 1e6)
        tsr.append(Float64(ctx.execution_time(launch_seg_reduce, 1)) / 1e6)
        tss.append(Float64(ctx.execution_time(launch_seg_scan, 1)) / 1e6)
        tsa.append(Float64(ctx.execution_time(launch_seg_apply, 1)) / 1e6)
        tf.append(Float64(ctx.execution_time(launch_fused, 1)) / 1e6)

    print(
        "H=" + String(num_heads),
        "T=" + String(seq_len),
        "x" + String(num_seqs),
        " L1",
        _median(t1),
        " L2",
        _median(t2),
        " L3",
        _median(t3),
        " sum",
        _median(t1) + _median(t2) + _median(t3),
    )
    print(
        "    segmented (SEG_LEN=16): reduce",
        _median(tsr),
        " segscan",
        _median(tss),
        " apply",
        _median(tsa),
        " sum(L1+A+B+C+L3)",
        _median(t1) + _median(tsr) + _median(tss) + _median(tsa) + _median(t3),
    )
    print(
        "    fused computebound, safe gate, bf16 state+out (one kernel):",
        _median(tf),
    )

    qk_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    si_h.free()
    sco_h.free()
    cto_h.free()


def main() raises:
    var ctx = DeviceContext()
    print("kda_chunk_parallel (L1 prepare / L2 scan / L3 output), median ms")
    _bench(ctx, 64, 8192, 1)
    _bench(ctx, 96, 8192, 1)
    _bench(ctx, 64, 1024, 8)
    _bench(ctx, 96, 1024, 8)
