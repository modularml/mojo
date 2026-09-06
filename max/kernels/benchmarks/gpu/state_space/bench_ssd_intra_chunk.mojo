# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Microbenchmark sweep for ``ssd_intra_chunk_fwd_gpu``.

Sweeps several dim configs to isolate where time goes: small/medium/large
chunk_len, head_dim, state_dim. Times wall-clock per call (kernel + sync).
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from state_space.ssd_chunk import (
    ssd_intra_chunk_fwd_gpu,
    ssd_intra_chunk_fwd_gpu_fused,
    ssd_intra_chunk_fwd_gpu_static,
)


def time_one[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    use_fused: Bool = False,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    comptime cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len

    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    var C_tt = TileTensor(
        C_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var Y_tt = TileTensor(
        Y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    for _ in range(warmups):
        comptime if use_fused:
            ssd_intra_chunk_fwd_gpu_fused[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
        else:
            ssd_intra_chunk_fwd_gpu_static[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
    ctx.synchronize()

    var min_ns: Int = -1
    var sum_ns: Int = 0
    for _ in range(iters):
        var t0 = perf_counter_ns()
        comptime if use_fused:
            ssd_intra_chunk_fwd_gpu_fused[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
        else:
            ssd_intra_chunk_fwd_gpu_static[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed = Int(t1 - t0)
        sum_ns += elapsed
        if min_ns < 0 or elapsed < min_ns:
            min_ns = elapsed

    var min_ms = Float64(min_ns) / 1.0e6
    var mean_ms = Float64(sum_ns) / Float64(iters) / 1.0e6
    print(
        "  nc=",
        n_chunks,
        " nh=",
        n_heads,
        " L=",
        chunk_len,
        " P=",
        head_dim,
        " N=",
        state_dim,
        "  min=",
        min_ms,
        "ms  mean=",
        mean_ms,
        "ms",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()

    # Single-slice baselines to expose where time goes.
    print("Single-slice (batch=1, n_chunks=1, n_heads=1) sweep:")
    time_one[DType.float32, 1, 1, 1, 64, 64, 32](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 64, 64, 128](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 128, 64, 128](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 256, 64, 128](ctx, 5, 20)

    print("Multi-slice sweep (Mamba2-130m prefill profile) — two-kernel:")
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 10)

    print("Multi-slice sweep (Mamba2-130m prefill profile) — FUSED (RFC 0009):")
    time_one[DType.float32, 1, 4, 24, 256, 64, 128, use_fused=True](ctx, 5, 50)
    time_one[DType.float32, 1, 8, 24, 256, 64, 128, use_fused=True](ctx, 5, 50)
