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
"""Kernel-level perf benchmark: FUSED vs UNFUSED MiniMax-M3 QKV + indexer-QKV.

Compares, at the raw-kernel level (no graph op / emitter / model wiring):

  * FUSED   : ONE call to `generic_fused_qkv_index_matmul_kv_cache_paged_ragged`
              over the stacked weight [Wq|Wk|Wv|Wiq|Wik] (N_total=2560).
  * UNFUSED : the current production path — TWO calls to the existing
              `generic_fused_qkv_matmul_kv_cache_paged_ragged`: main
              [Wq|Wk|Wv] (N=2304) then indexer [Wiq|Wik] (N=256).

Both paths do the SAME total work (2*M*N_total*K FLOPs) and the SAME KV
scatter; the only difference is one GEMM+launch vs two, so this isolates the
fusion's launch/scheduling benefit on the two small decode-regime GEMMs.

Shapes: M3 per-device (TP=4), native BF16. Sweeps the DECODE regime
(total_seq == decode batch size, one token each) across
{1, 8, 16, 32, 64, 128, 256}, plus one PREFILL shape (2 prompts x 256 tokens)
for completeness. Cache topologies match the differential test: MAIN = non-MLA
GQA (K+V, 1 KV head); INDEX = MLA (K-only, 1 latent head).

Timing: stdlib `benchmark` `Bench` / `iter_custom`, matching the SM100 SwiGLU
fusion benchmark (`profile_grouped_matmul_swiglu_nvfp4.mojo`). The hidden state
and the stacked weight are cache-busted (`CacheBustingBuffer` + per-iteration
`offset_ptr`) so every iteration reads cold HBM rather than an L2-resident copy
-- decode QKV is weight-bandwidth-bound, so this is what keeps the numbers
honest. Reports the per-iteration mean plus GFLOP/s and GB/s via
`ThroughputMeasure`. The UNFUSED entry enqueues both kernels inside one timed
closure, so its number is the sum of the two calls.

Run directly:  mojo max/kernels/benchmarks/gpu/nn/bench_fused_qkv_index_matmul.mojo
Or via bazel:  ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qkv_index_matmul
"""

from std.random import seed

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
)

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType

from std.math import ceildiv
from std.sys import size_of
from std.utils import IndexList

# ---- M3 per-device (TP=4) BF16 shapes (match the differential test) ----
comptime DATA_DTYPE = DType.bfloat16
comptime HEAD_SIZE = 128
comptime NUM_Q_HEADS = 16  # q_dim = 2048
comptime MAIN_KV_HEADS = 1  # kv_dim = 128
comptime NUM_INDEX_HEADS = 1  # iq_dim = 128

comptime hidden = 6144  # K
comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 2048
comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 128
comptime ik_dim = HEAD_SIZE  # 128
comptime qkv_n = q_dim + 2 * kv_dim  # 2304 (main matmul N)
comptime idx_n = iq_dim + ik_dim  # 256 (indexer matmul N)
comptime n_total = qkv_n + idx_n  # 2560 (stacked N)
comptime combined_out = q_dim + iq_dim  # 2176 (fused visible output)

# Cache config (single layer is enough for a kernel microbench).
comptime page_size = 512
comptime num_pages = 512  # >= max decode batch (256) and prefill pages
comptime num_layers = 1
comptime layer_idx = 0

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(
    num_heads=1, head_size=HEAD_SIZE, is_mla=True
)

comptime MainCollection = PagedKVCacheCollection[
    DATA_DTYPE, main_kv_params, page_size, ...
]
comptime IndexCollection = PagedKVCacheCollection[
    DATA_DTYPE, index_kv_params, page_size, ...
]


def bench_shape(
    ctx: DeviceContext,
    mut m: Bench,
    prompt_lens: List[Int],
    regime: String,
) raises:
    """Build device inputs / caches for `prompt_lens` and register the FUSED and
    UNFUSED `iter_custom` benchmark functions for this shape."""
    var batch_size = len(prompt_lens)

    # ---- ragged offsets + (empty) cache lengths ----
    var total_seq = 0
    var max_seq = 0
    var iro_host = List[UInt32](length=batch_size + 1, fill=UInt32(0))
    for i in range(batch_size):
        iro_host[i] = UInt32(total_seq)
        total_seq += prompt_lens[i]
        max_seq = max(max_seq, prompt_lens[i])
    iro_host[batch_size] = UInt32(total_seq)
    var max_ctx = max_seq  # cache_lengths are all 0 here

    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(iro_dev, iro_host)
    var iro_tensor = LayoutTensor[
        mut=False, .uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        iro_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    var cache_lengths_host = List[UInt32](length=batch_size, fill=UInt32(0))
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_dev.unsafe_ptr(),
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
    )

    # ---- paged lookup table (sequential distinct blocks; shared by both
    # caches since main/index blocks are separate allocations) ----
    var lut_cols = ((ceildiv(max_ctx, page_size) + 7) // 8) * 8 + 16
    var lut_host = List[UInt32](length=batch_size * lut_cols, fill=UInt32(0))
    var block_counter = 0
    for b in range(batch_size):
        var pages = ceildiv(prompt_lens[b], page_size)
        for p in range(pages):
            lut_host[b * lut_cols + p] = UInt32(block_counter)
            block_counter += 1
    var lut_dev = ctx.enqueue_create_buffer[.uint32](batch_size * lut_cols)
    ctx.enqueue_copy(lut_dev, lut_host)
    var lut_tensor = LayoutTensor[mut=False, .uint32, Layout.row_major[2]()](
        lut_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[2]()].row_major(
            IndexList[2](batch_size, lut_cols)
        ),
    )

    # ---- cache-busting inputs: hidden state (M, K) and stacked weight
    # (N_total, K), both bf16. Each timed iteration reads a fresh, cold window
    # (see module docstring) so L2 can't hide HBM traffic. The `hs_tensor` and
    # weight views (`w_full`, and the unfused sub-views `w_qkv`=[Wq|Wk|Wv],
    # `w_idx`=[Wiq|Wik]) are rebuilt per iteration from `offset_ptr` inside the
    # timed closures below.
    comptime simd_size = 4
    var cb_hs = CacheBustingBuffer[DATA_DTYPE](
        total_seq * hidden, simd_size, ctx
    )
    var cb_w = CacheBustingBuffer[DATA_DTYPE](n_total * hidden, simd_size, ctx)
    cb_hs.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_w.init_on_device(InitializationType.uniform_distribution, ctx)

    # ---- output buffers ----
    var fused_out_dev = ctx.enqueue_create_buffer[DATA_DTYPE](
        total_seq * combined_out
    )
    var fused_out = LayoutTensor[
        DATA_DTYPE, Layout.row_major(UNKNOWN_VALUE, combined_out)
    ](
        fused_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, combined_out)].row_major(
            IndexList[2](total_seq, combined_out)
        ),
    )
    var q_out_dev = ctx.enqueue_create_buffer[DATA_DTYPE](total_seq * q_dim)
    var q_out = LayoutTensor[
        DATA_DTYPE, Layout.row_major(UNKNOWN_VALUE, q_dim)
    ](
        q_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, q_dim)].row_major(
            IndexList[2](total_seq, q_dim)
        ),
    )
    var iq_out_dev = ctx.enqueue_create_buffer[DATA_DTYPE](total_seq * iq_dim)
    var iq_out = LayoutTensor[
        DATA_DTYPE, Layout.row_major(UNKNOWN_VALUE, iq_dim)
    ](
        iq_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, iq_dim)].row_major(
            IndexList[2](total_seq, iq_dim)
        ),
    )

    # ---- KV cache blocks (main: K+V, 1 head; index: K-only MLA, 1 head) ----
    comptime block_layout = Layout.row_major[6]()
    var main_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks_dev = ctx.enqueue_create_buffer[DATA_DTYPE](
        main_block_shape.flattened_length()
    )
    var main_blocks = LayoutTensor[DATA_DTYPE, block_layout](
        main_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(main_block_shape),
    )
    var index_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks_dev = ctx.enqueue_create_buffer[DATA_DTYPE](
        index_block_shape.flattened_length()
    )
    var index_blocks = LayoutTensor[DATA_DTYPE, block_layout](
        index_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(index_block_shape),
    )

    # `as_unsafe_any_origin`: the fused QKV matmul writes both the k and v cache
    # views (disjoint kv_idx halves of one blocks buffer sharing its origin), so
    # the nested-origin exclusivity check would reject passing both. Opt out.
    var main_collection = MainCollection(
        main_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor,
        lut_tensor,
        UInt32(max_seq),
        UInt32(max_ctx),
    )
    var index_collection = IndexCollection(
        index_blocks.as_unsafe_any_origin(),
        cache_lengths_tensor,
        lut_tensor,
        UInt32(max_seq),
        UInt32(max_ctx),
    )

    # Useful compute is identical for both paths. Byte traffic differs only in
    # the activation read: the unfused chain re-reads the hidden state for its
    # second (indexer) GEMM; both write the same visible Q/IndexQ output and the
    # same KV scatter (main K+V + index K).
    var flops = 2 * total_seq * n_total * hidden
    comptime elt = size_of[DATA_DTYPE]()
    var write_elems = (
        total_seq * combined_out  # visible Q | IndexQ
        + 2 * total_seq * kv_dim  # main K + V scatter
        + total_seq * ik_dim  # index K scatter
    )
    var fused_bytes = (
        n_total * hidden + total_seq * hidden + write_elems
    ) * elt
    var unfused_bytes = (
        n_total * hidden + 2 * total_seq * hidden + write_elems
    ) * elt

    # ============ FUSED: one GEMM over the stacked weight ============
    @always_inline
    def fused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_hs, mut cb_w, mut fused_out, imm,}:
        var hs_tensor = LayoutTensor[
            mut=False, DATA_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var w_full = LayoutTensor[
            mut=False, DATA_DTYPE, Layout.row_major(n_total, hidden)
        ](
            cb_w.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(n_total, hidden)].row_major(
                IndexList[2](n_total, hidden)
            ),
        )
        generic_fused_qkv_index_matmul_kv_cache_paged_ragged[target="gpu"](
            hs_tensor,
            iro_tensor,
            w_full,
            main_collection,
            index_collection,
            UInt32(layer_idx),
            iq_dim,
            fused_out,
            ctx,
        )

    @always_inline
    def fused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, fused_launch, ctx)

    m.bench_function(
        fused_bench,
        BenchId("fused   " + regime + " total_seq=" + String(total_seq)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, fused_bytes),
        ],
    )

    # ============ UNFUSED: main QKV then indexer QKV (2 calls) ============
    @always_inline
    def unfused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_hs, mut cb_w, mut q_out, mut iq_out, imm,}:
        var hs_tensor = LayoutTensor[
            mut=False, DATA_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var w_qkv = LayoutTensor[
            mut=False, DATA_DTYPE, Layout.row_major(qkv_n, hidden)
        ](
            cb_w.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(qkv_n, hidden)].row_major(
                IndexList[2](qkv_n, hidden)
            ),
        )
        var w_idx = LayoutTensor[
            mut=False, DATA_DTYPE, Layout.row_major(idx_n, hidden)
        ](
            cb_w.offset_ptr(iteration) + qkv_n * hidden,
            RuntimeLayout[Layout.row_major(idx_n, hidden)].row_major(
                IndexList[2](idx_n, hidden)
            ),
        )
        generic_fused_qkv_matmul_kv_cache_paged_ragged[target="gpu"](
            hs_tensor,
            iro_tensor,
            w_qkv,
            main_collection,
            UInt32(layer_idx),
            q_out,
            ctx,
        )
        generic_fused_qkv_matmul_kv_cache_paged_ragged[target="gpu"](
            hs_tensor,
            iro_tensor,
            w_idx,
            index_collection,
            UInt32(layer_idx),
            iq_out,
            ctx,
        )

    @always_inline
    def unfused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, unfused_launch, ctx)

    m.bench_function(
        unfused_bench,
        BenchId("unfused " + regime + " total_seq=" + String(total_seq)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, unfused_bytes),
        ],
    )

    # Keep device buffers alive until all iterations have run.
    _ = cb_hs^
    _ = cb_w^
    _ = iro_dev^
    _ = cache_lengths_dev^
    _ = lut_dev^
    _ = fused_out_dev^
    _ = q_out_dev^
    _ = iq_out_dev^
    _ = main_blocks_dev^
    _ = index_blocks_dev^


def main() raises:
    seed(0)
    var m = Bench()
    with DeviceContext() as ctx:
        # DECODE regime: total_seq == batch size (one token each).
        for bs in [1, 8, 16, 32, 64, 128, 256]:
            var decode_lens = List[Int](length=bs, fill=1)
            bench_shape(ctx, m, decode_lens, "decode")

        # PREFILL shape for completeness: 2 prompts x 256 tokens.
        bench_shape(ctx, m, [256, 256], "prefill")
    m.dump_report()
