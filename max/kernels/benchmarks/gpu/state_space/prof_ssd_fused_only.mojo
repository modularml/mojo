# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Minimal driver: run ONLY the fused intra-chunk MMA path (RFC 0009).

Scoped target for ncu single-metric roofline collection on GB10. Mamba2-130m
prefill profile.
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from state_space.ssd_chunk import ssd_intra_chunk_fwd_gpu_fused


def main() raises:
    var ctx = DeviceContext()
    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime chunk_len = 256
    comptime head_dim = 64
    comptime state_dim = 128

    comptime cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len

    var C = ctx.enqueue_create_buffer[dtype](cb_count)
    var B = ctx.enqueue_create_buffer[dtype](cb_count)
    var X = ctx.enqueue_create_buffer[dtype](xy_count)
    var A = ctx.enqueue_create_buffer[dtype](a_count)
    var Y = ctx.enqueue_create_buffer[dtype](xy_count)

    var C_tt = TileTensor(
        C, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var B_tt = TileTensor(
        B, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(A, row_major(batch, n_chunks, n_heads, chunk_len))
    var Y_tt = TileTensor(
        Y, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    for _ in range(4):
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
    ctx.synchronize()
