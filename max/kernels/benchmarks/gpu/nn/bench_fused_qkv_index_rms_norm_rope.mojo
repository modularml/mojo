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
"""Kernel-level A/B benchmark: one fused QKV+indexer launch vs two unfused launches.

Compares, at identical inputs:
    fused    -- one launch of `fused_dual_qk_rms_norm_rope_ragged_paged`,
                norming+RoPE-ing the main-attention Q/K and the indexer Q/K
                together.
    unfused  -- two back-to-back launches of `fused_qk_rms_norm_rope_ragged_paged`
                (main pair, then index pair), which is what MiniMax-M3 did before
                the dual op landed.

The fusion removes one launch's fixed overhead, so the interesting regime is
small decode batch (one token per sequence), where per-launch overhead dominates
the tiny per-token work.

MiniMax-M3 shapes (BF16, the only path that fires this fusion on AMD):
    head_dim=128, rope_dim=64 (partial, non-interleaved), weight_offset=1.0
    main:  q=64, kv=4     index: q=4, kv=1

`cache_len` is fixed at 0. It does not affect the kernel's work: the number of
new K-cache writes is `total_seq_len * num_kv_heads` regardless of where in the
cache they land, and the RoPE position only indexes the freqs table. Keeping it 0
bounds the paged-cache allocation at large batch while leaving the launch-overhead
vs work-bound comparison unchanged.

Cache busting (cold HBM per iter): the norm+RoPE is bandwidth/overhead bound, so
after warmup a reused buffer sits in L2 and reports optimistic times. Each timed
iteration reads/writes a fresh window of a ring buffer that exceeds the MI355
last-level cache (see `CB_BUDGET`), so every iter touches cold HBM. Busted:
the main/index Q reads and all four K caches (each K cache is read then written
in place, so read and write target the SAME per-iter window; the window advances
across iters). Fused and unfused hold independent K rings so their in-place
writes never alias. L2-hot (small, shared): gammas, freqs, row_offsets,
cache_lengths, paged_lut, and the Q output write targets (write-only and
identical for both variants, so they do not bias the fused-vs-unfused gap).

Both variants move the SAME HBM bytes per launch-set: the only
difference is one launch vs two, so any GB/s gap is pure launch overhead.

Run locally (AMD MI355):
    ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qkv_index_rms_norm_rope
"""

from std.math import ceildiv
from std.random import seed
from std.sys import size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType
from nn.kv_cache import (
    fused_dual_qk_rms_norm_rope_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
)
from std.utils import Index, IndexList

# Ring size per cache-busted tensor. 1 GiB comfortably exceeds a few x the MI355
# last-level cache and, even at the largest window (bs=512 decode main-K is
# ~128 MiB), holds multiple windows so `offset()` never collapses to 0. Bounded
# and independent of iteration count. Six busted tensors => ~6 GiB, trivial on a
# 288+ GB device (no shape here comes close to OOM, so no ring is ever capped).
comptime CB_BUDGET = 1024 * 1024 * 1024
# Element alignment for ring windows: 8 bf16 elts = 16 B, safe for 128-bit
# vector loads/stores.
comptime CB_ALIGN = 8


def _bench_name[
    dtype: DType,
    head_dim: Int,
    rope_dim: Int,
    variant: StaticString,
](batch_size: Int, seq_len: Int) -> String:
    # fmt: off
    return String(
        variant, "(", dtype, ")",
        " hd=", head_dim,
        " rd=", rope_dim,
        " bs=", batch_size,
        " sl=", seq_len,
    )
    # fmt: on


def bench_fused_qkv_index_rms_norm_rope[
    dtype: DType,
    freq_dtype: DType,
    head_dim: Int,
    rope_dim: Int,
    main_q_heads: Int,
    main_kv_heads: Int,
    index_q_heads: Int,
    index_kv_heads: Int,
](ctx: DeviceContext, mut m: Bench, batch_size: Int, seq_len: Int) raises:
    """Benchmarks the fused single launch vs the unfused two-launch sequence."""
    comptime main_params = KVCacheStaticParams(
        num_heads=main_kv_heads, head_size=head_dim
    )
    comptime index_params = KVCacheStaticParams(
        num_heads=index_kv_heads, head_size=head_dim
    )
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime weight_offset = 1.0
    # Covers any (cache_len=0)+seq_len position we index into `freqs`.
    comptime max_seq_len = 4096

    var cache_len = UInt32(0)
    var total_seq_len = batch_size * seq_len
    var pages_per_seq = ceildiv(Int(cache_len) + seq_len, page_size)
    var num_paged_blocks = batch_size * pages_per_seq

    var main_kv_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, main_kv_heads, head_dim
    )
    var index_kv_block_shape = IndexList[6](
        num_paged_blocks, 2, num_layers, page_size, index_kv_heads, head_dim
    )
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime paged_lut_layout = Layout.row_major[2]()

    # Cache-busted tensors (cold HBM per iter). Q reads are read-only and
    # identical for both variants, so a single ring is shared. The K caches are
    # read+written in place, so fused and unfused get independent rings.
    var cb_q_main = CacheBustingBuffer[dtype](
        total_seq_len * main_q_heads * head_dim,
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    var cb_q_index = CacheBustingBuffer[dtype](
        total_seq_len * index_q_heads * head_dim,
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    var cb_main_kv_unfused = CacheBustingBuffer[dtype](
        main_kv_block_shape.flattened_length(),
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    var cb_main_kv_fused = CacheBustingBuffer[dtype](
        main_kv_block_shape.flattened_length(),
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    var cb_index_kv_unfused = CacheBustingBuffer[dtype](
        index_kv_block_shape.flattened_length(),
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    var cb_index_kv_fused = CacheBustingBuffer[dtype](
        index_kv_block_shape.flattened_length(),
        CB_ALIGN,
        ctx,
        budget_bytes=CB_BUDGET,
    )
    cb_q_main.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_q_index.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_main_kv_unfused.init_on_device(
        InitializationType.uniform_distribution, ctx
    )
    cb_main_kv_fused.init_on_device(
        InitializationType.uniform_distribution, ctx
    )
    cb_index_kv_unfused.init_on_device(
        InitializationType.uniform_distribution, ctx
    )
    cb_index_kv_fused.init_on_device(
        InitializationType.uniform_distribution, ctx
    )

    # L2-hot auxiliaries and write-only outputs (fixed buffers).
    var row_offsets_d = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths_d = ctx.enqueue_create_buffer[.uint32](batch_size)
    var q_main_out_unfused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * main_q_heads * head_dim
    )
    var q_main_out_fused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * main_q_heads * head_dim
    )
    var q_index_out_unfused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * index_q_heads * head_dim
    )
    var q_index_out_fused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * index_q_heads * head_dim
    )
    var gamma_q_main_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var gamma_k_main_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var gamma_q_index_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var gamma_k_index_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var paged_lut_d = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    var freqs_d = ctx.enqueue_create_buffer[freq_dtype](max_seq_len * rope_dim)

    var row_offsets_h = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var cache_lengths_h = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var paged_lut_h = ctx.enqueue_create_host_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    for i in range(batch_size + 1):
        row_offsets_h[i] = UInt32(i * seq_len)
    for i in range(batch_size):
        cache_lengths_h[i] = cache_len
        for j in range(pages_per_seq):
            paged_lut_h[i * pages_per_seq + j] = UInt32(i * pages_per_seq + j)
    ctx.enqueue_copy(row_offsets_d, row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(paged_lut_d, paged_lut_h)

    comptime gamma_layout = Layout.row_major(head_dim)
    var gamma_rt = RuntimeLayout[gamma_layout].row_major(Index(head_dim))
    with gamma_q_main_d.map_to_host() as h:
        random(LayoutTensor[dtype, gamma_layout](h, gamma_rt))
    with gamma_k_main_d.map_to_host() as h:
        random(LayoutTensor[dtype, gamma_layout](h, gamma_rt))
    with gamma_q_index_d.map_to_host() as h:
        random(LayoutTensor[dtype, gamma_layout](h, gamma_rt))
    with gamma_k_index_d.map_to_host() as h:
        random(LayoutTensor[dtype, gamma_layout](h, gamma_rt))

    comptime freqs_static_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_rt = RuntimeLayout[freqs_static_layout].row_major(
        IndexList[2](max_seq_len, rope_dim)
    )
    with freqs_d.map_to_host() as h:
        random(LayoutTensor[freq_dtype, freqs_static_layout](h, freqs_rt))
    ctx.synchronize()

    # Runtime layouts rebuilt per iter onto each ring window.
    var main_kv_rt = RuntimeLayout[kv_block_layout].row_major(
        main_kv_block_shape
    )
    var index_kv_rt = RuntimeLayout[kv_block_layout].row_major(
        index_kv_block_shape
    )

    var q_main_out_unfused_tile = TileTensor(
        q_main_out_unfused_d,
        row_major((total_seq_len, Idx[main_q_heads], Idx[head_dim])),
    )
    var q_main_out_fused_tile = TileTensor(
        q_main_out_fused_d,
        row_major((total_seq_len, Idx[main_q_heads], Idx[head_dim])),
    )
    var q_index_out_unfused_tile = TileTensor(
        q_index_out_unfused_d,
        row_major((total_seq_len, Idx[index_q_heads], Idx[head_dim])),
    )
    var q_index_out_fused_tile = TileTensor(
        q_index_out_fused_d,
        row_major((total_seq_len, Idx[index_q_heads], Idx[head_dim])),
    )
    var gamma_q_main_tile = TileTensor(gamma_q_main_d, row_major[head_dim]())
    var gamma_k_main_tile = TileTensor(gamma_k_main_d, row_major[head_dim]())
    var gamma_q_index_tile = TileTensor(gamma_q_index_d, row_major[head_dim]())
    var gamma_k_index_tile = TileTensor(gamma_k_index_d, row_major[head_dim]())
    var freqs_tile = TileTensor(freqs_d, row_major[max_seq_len, rope_dim]())
    var row_offsets_tile = TileTensor(row_offsets_d, row_major(batch_size + 1))

    var cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, cache_lengths_layout
    ](
        cache_lengths_d,
        RuntimeLayout[cache_lengths_layout].row_major(Index(batch_size)),
    )
    var paged_lut_tensor = LayoutTensor[mut=False, .uint32, paged_lut_layout](
        paged_lut_d,
        RuntimeLayout[paged_lut_layout].row_major(paged_lut_shape),
    )
    var max_prompt_len = UInt32(seq_len)
    var max_cache_len = cache_len

    # Real HBM traffic the norm+RoPE moves per launch-set. Q and K are each read
    # then written (2x); freqs is read once per token position (rows_that_rope =
    # total_seq_len); gammas are tiny. Identical for fused and unfused -- equal
    # traffic is the point, so any GB/s gap is pure launch overhead.
    comptime elt = size_of[dtype]()
    comptime felt = size_of[freq_dtype]()
    var rw_bytes = (
        2
        * (
            total_seq_len * main_q_heads * head_dim
            + total_seq_len * index_q_heads * head_dim
            + total_seq_len * main_kv_heads * head_dim
            + total_seq_len * index_kv_heads * head_dim
        )
        * elt
    )
    var freqs_bytes = total_seq_len * rope_dim * felt
    var gamma_bytes = 4 * head_dim * elt
    var bytes_per_iter = rw_bytes + freqs_bytes + gamma_bytes

    @always_inline
    def bench_unfused(
        mut b: Bencher,
    ) {
        var cb_q_main,
        var cb_q_index,
        var cb_main_kv_unfused,
        var cb_index_kv_unfused,
        var main_kv_rt,
        var index_kv_rt,
        var cache_lengths_tensor,
        var paged_lut_tensor,
        var q_main_out_unfused_tile,
        var q_index_out_unfused_tile,
        var gamma_q_main_tile,
        var gamma_k_main_tile,
        var gamma_q_index_tile,
        var gamma_k_index_tile,
        var freqs_tile,
        var row_offsets_tile,
        var max_prompt_len,
        var max_cache_len,
        var total_seq_len,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            # Named vars bind the per-iter ring-window pointer's origin before
            # it flows into the cache collection / input lambdas.
            var main_kv_lt = LayoutTensor[dtype, kv_block_layout](
                cb_main_kv_unfused.offset_ptr(iteration), main_kv_rt
            )
            var index_kv_lt = LayoutTensor[dtype, kv_block_layout](
                cb_index_kv_unfused.offset_ptr(iteration), index_kv_rt
            )
            var main_kv = PagedKVCacheCollection[dtype, main_params, page_size](
                main_kv_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            var index_kv = PagedKVCacheCollection[
                dtype, index_params, page_size
            ](
                index_kv_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            var q_main_src = TileTensor(
                cb_q_main.offset_ptr(iteration),
                row_major((total_seq_len, Idx[main_q_heads], Idx[head_dim])),
            ).as_immut()
            var q_index_src = TileTensor(
                cb_q_index.offset_ptr(iteration),
                row_major((total_seq_len, Idx[index_q_heads], Idx[head_dim])),
            ).as_immut()

            @always_inline
            @__parameter
            @__copy_capture(q_main_src)
            def q_main_fn[
                width: Int, alignment: Int
            ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
                return q_main_src.load[width=width](
                    Coord(Index(token, head, col))
                )

            @always_inline
            @__parameter
            @__copy_capture(q_index_src)
            def q_index_fn[
                width: Int, alignment: Int
            ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
                return q_index_src.load[width=width](
                    Coord(Index(token, head, col))
                )

            fused_qk_rms_norm_rope_ragged_paged[
                target="gpu",
                multiply_before_cast=True,
                interleaved=False,
                q_input_fn=q_main_fn,
            ](
                main_kv,
                gamma_q_main_tile.as_immut(),
                gamma_k_main_tile.as_immut(),
                freqs_tile.as_immut(),
                Float32(1e-6),
                Scalar[dtype](weight_offset),
                UInt32(layer_idx),
                row_offsets_tile.as_immut(),
                q_main_out_unfused_tile,
                ctx,
            )
            fused_qk_rms_norm_rope_ragged_paged[
                target="gpu",
                multiply_before_cast=True,
                interleaved=False,
                q_input_fn=q_index_fn,
            ](
                index_kv,
                gamma_q_index_tile.as_immut(),
                gamma_k_index_tile.as_immut(),
                freqs_tile.as_immut(),
                Float32(1e-6),
                Scalar[dtype](weight_offset),
                UInt32(layer_idx),
                row_offsets_tile.as_immut(),
                q_index_out_unfused_tile,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_unfused,
        BenchId(
            _bench_name[dtype, head_dim, rope_dim, "unfused"](
                batch_size, seq_len
            )
        ),
        [ThroughputMeasure(BenchMetric.bytes, bytes_per_iter)],
    )

    @always_inline
    def bench_fused(
        mut b: Bencher,
    ) {
        var cb_q_main,
        var cb_q_index,
        var cb_main_kv_fused,
        var cb_index_kv_fused,
        var main_kv_rt,
        var index_kv_rt,
        var cache_lengths_tensor,
        var paged_lut_tensor,
        var q_main_out_fused_tile,
        var q_index_out_fused_tile,
        var gamma_q_main_tile,
        var gamma_k_main_tile,
        var gamma_q_index_tile,
        var gamma_k_index_tile,
        var freqs_tile,
        var row_offsets_tile,
        var max_prompt_len,
        var max_cache_len,
        var total_seq_len,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var main_kv_lt = LayoutTensor[dtype, kv_block_layout](
                cb_main_kv_fused.offset_ptr(iteration), main_kv_rt
            )
            var index_kv_lt = LayoutTensor[dtype, kv_block_layout](
                cb_index_kv_fused.offset_ptr(iteration), index_kv_rt
            )
            var main_kv = PagedKVCacheCollection[dtype, main_params, page_size](
                main_kv_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            var index_kv = PagedKVCacheCollection[
                dtype, index_params, page_size
            ](
                index_kv_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            var q_main_src = TileTensor(
                cb_q_main.offset_ptr(iteration),
                row_major((total_seq_len, Idx[main_q_heads], Idx[head_dim])),
            ).as_immut()
            var q_index_src = TileTensor(
                cb_q_index.offset_ptr(iteration),
                row_major((total_seq_len, Idx[index_q_heads], Idx[head_dim])),
            ).as_immut()

            @always_inline
            @__parameter
            @__copy_capture(q_main_src)
            def q_main_fn[
                width: Int, alignment: Int
            ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
                return q_main_src.load[width=width](
                    Coord(Index(token, head, col))
                )

            @always_inline
            @__parameter
            @__copy_capture(q_index_src)
            def q_index_fn[
                width: Int, alignment: Int
            ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
                return q_index_src.load[width=width](
                    Coord(Index(token, head, col))
                )

            # Exercises the merged dual op (`mo.fused_qk_rms_norm_rope.ragged.paged.dual`,
            # landed separately), which collapses the two single-pair launches above
            # into one. It takes a per-band epsilon, so `main_epsilon`/`index_epsilon`
            # are passed twice below with the same value the unfused pair uses.
            fused_dual_qk_rms_norm_rope_ragged_paged[
                target="gpu",
                multiply_before_cast=True,
                interleaved=False,
                main_q_input_fn=q_main_fn,
                index_q_input_fn=q_index_fn,
            ](
                main_kv,
                index_kv,
                gamma_q_main_tile.as_immut(),
                gamma_k_main_tile.as_immut(),
                gamma_q_index_tile.as_immut(),
                gamma_k_index_tile.as_immut(),
                freqs_tile.as_immut(),
                Float32(1e-6),
                Float32(1e-6),
                Scalar[dtype](weight_offset),
                UInt32(layer_idx),
                row_offsets_tile.as_immut(),
                q_main_out_fused_tile,
                q_index_out_fused_tile,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_fused,
        BenchId(
            _bench_name[dtype, head_dim, rope_dim, "fused"](batch_size, seq_len)
        ),
        [ThroughputMeasure(BenchMetric.bytes, bytes_per_iter)],
    )

    _ = cb_q_main^
    _ = cb_q_index^
    _ = cb_main_kv_unfused^
    _ = cb_main_kv_fused^
    _ = cb_index_kv_unfused^
    _ = cb_index_kv_fused^
    _ = row_offsets_d^
    _ = cache_lengths_d^
    _ = q_main_out_unfused_d^
    _ = q_main_out_fused_d^
    _ = q_index_out_unfused_d^
    _ = q_index_out_fused_d^
    _ = gamma_q_main_d^
    _ = gamma_k_main_d^
    _ = gamma_q_index_d^
    _ = gamma_k_index_d^
    _ = paged_lut_d^
    _ = freqs_d^


def main() raises:
    comptime dtype = DType.bfloat16
    comptime freq_dtype = DType.float32
    comptime head_dim = 128
    comptime rope_dim = 64
    # MiniMax-M3 (single device): main 64Q/4KV, indexer 4Q/1KV.
    comptime main_q_heads = 64
    comptime main_kv_heads = 4
    comptime index_q_heads = 4
    comptime index_kv_heads = 1

    seed(0)

    # Exactly 20 warmup iters + one measured batch of 100 iters: min_runtime=0
    # forces the batch size to `max_iters`, and the second loop pass breaks once
    # 100 iters are logged (see std.benchmark._run_impl).
    var m = Bench(
        BenchConfig(
            num_warmup_iters=20,
            max_iters=100,
            min_runtime_secs=0.0,
            max_runtime_secs=10.0,
        )
    )
    with DeviceContext() as ctx:
        # Decode: one token per sequence, sweeping batch by powers of two.
        var decode_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        for i in range(len(decode_batches)):
            bench_fused_qkv_index_rms_norm_rope[
                dtype,
                freq_dtype,
                head_dim,
                rope_dim,
                main_q_heads,
                main_kv_heads,
                index_q_heads,
                index_kv_heads,
            ](ctx, m, batch_size=decode_batches[i], seq_len=1)
        # Prefill points (work-bound regime, for context).
        var prefill_seqs = [2048, 4096]
        for i in range(len(prefill_seqs)):
            bench_fused_qkv_index_rms_norm_rope[
                dtype,
                freq_dtype,
                head_dim,
                rope_dim,
                main_q_heads,
                main_kv_heads,
                index_q_heads,
                index_kv_heads,
            ](ctx, m, batch_size=1, seq_len=prefill_seqs[i])

    m.dump_report()
