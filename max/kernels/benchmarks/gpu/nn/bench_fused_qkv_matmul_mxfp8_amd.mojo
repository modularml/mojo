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
"""Kernel-level perf benchmark: MXFP8 QKV projections on CDNA4, fused vs not.

Covers both attention flavours behind one `has_indexer` argument, since the
harness (ragged offsets, paged caches, cache-busted operands, band slicing) is
identical and only the band list and the fused entry point differ:

  * DENSE  (`has_indexer=False`) -- layers with no sparse indexer. Bands
    `[Wq|Wk|Wv]`, N_total=2304. FUSED is one
    `generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4`; UNFUSED is
    three `block_scaled_matmul_amd` calls (N = 2048, 128, 128) plus two paged
    `kv_cache_store` launches to place K/V.
  * SPARSE (`has_indexer=True`) -- layers that also project the indexer's
    IndexQ/IndexK. Bands `[Wq|Wk|Wv|Wiq|Wik]`, N_total=2560. FUSED is one
    `generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4`;
    UNFUSED is five band GEMMs (N = 2048, 128, 128, 128, 128) plus three paged
    stores.

Either way both paths do the same 2*M*N_total*K FLOPs against the same cold
weight bytes, so the difference is the GEMM count plus the paged stores the
unfused path needs to place K/V (and IndexK), which the fused epilogue does from
the GEMM's own store path. Everything is inside the timed closure.

Shapes: M3 per-device (TP=4), MXFP8 operands with E8M0 scales over 32-element K
blocks. `batch_size` prompts of `seq_len` tokens each, so decode is seq_len=1
across a batch sweep and prefill is a few long prompts; the regime label follows
from seq_len. Cache topologies: MAIN = non-MLA GQA (K+V, 1 KV head); INDEX = MLA
(K-only, 1 latent head).

Timing: stdlib `benchmark` `Bench` / `iter_custom`. Operands and both scale
tensors are cache-busted (`CacheBustingBuffer` + per-iteration `offset_ptr`) so
each iteration reads cold HBM -- decode QKV is weight-bandwidth-bound. Reports
per-iteration mean plus GFLOP/s and GB/s via `ThroughputMeasure`.

CDNA4 only: the fused epilogue and the block-scaled AMD matmul are MI355X paths.

A run covers ONE variant at ONE shape, defaulting to dense at decode batch 1.
The sibling yaml holds the sweep: `$has_indexer` over both variants crossed with
`$batch_size` / `$seq_len` over the decode, verify, and prefill shapes.

Run the default (dense, decode bs=1):
    ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qkv_matmul_mxfp8_amd
One other point (sparse, decode bs=64):
    ... -- --has_indexer=True --batch_size=64 --seq_len=1
The whole sweep:
    python max/kernels/benchmarks/autotune/kbench.py \\
        max/kernels/benchmarks/gpu/nn/bench_fused_qkv_matmul_mxfp8_amd.yaml
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
from layout.tile_tensor import lt_to_tt
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from linalg.matmul.gpu.amd import block_scaled_matmul_amd
from nn.kv_cache_ragged import (
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
    kv_cache_store_ragged,
)

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType, arg_parse

from std.math import ceildiv
from std.sys import size_of
from std.utils import IndexList

comptime OPERAND_DTYPE = DType.float8_e4m3fn
comptime OUT_DTYPE = DType.bfloat16
comptime SCALE_DTYPE = DType.float8_e8m0fnu
comptime SF_VECTOR_SIZE = 32

comptime HEAD_SIZE = 128
comptime NUM_Q_HEADS = 16  # q_dim = 2048
comptime MAIN_KV_HEADS = 1  # kv_dim = 128
comptime NUM_INDEX_HEADS = 1  # iq_dim = 128

comptime hidden = 6144  # K
comptime q_dim = NUM_Q_HEADS * HEAD_SIZE  # 2048
comptime kv_dim = MAIN_KV_HEADS * HEAD_SIZE  # 128
comptime iq_dim = NUM_INDEX_HEADS * HEAD_SIZE  # 128
comptime ik_dim = HEAD_SIZE  # 128
comptime k_scales = hidden // SF_VECTOR_SIZE  # 192

comptime page_size = 512
comptime num_pages = 512
comptime num_layers = 1
comptime layer_idx = 0

comptime main_kv_params = KVCacheStaticParams(
    num_heads=MAIN_KV_HEADS, head_size=HEAD_SIZE
)
comptime index_kv_params = KVCacheStaticParams(num_heads=1, head_size=HEAD_SIZE)
comptime MainCollection = PagedKVCacheCollection[
    OUT_DTYPE, main_kv_params, page_size, ...
]
comptime IndexCollection = PagedKVCacheCollection[
    OUT_DTYPE, index_kv_params, page_size, ...
]


@always_inline
def _any(
    ptr: MutPointer[Scalar[OUT_DTYPE], ...],
) -> MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin]:
    """Erase a pointer's origin so one helper serves every band."""
    return MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin](
        unsafe_from_address=Int(ptr)
    )


# Column offset of each band in the stacked weight, for the unfused slices. The
# index bands sit past the QKV ones, so the dense layout is a prefix of the
# sparse one and these offsets serve both variants.
comptime k_off = q_dim
comptime v_off = k_off + kv_dim
comptime iq_off = v_off + kv_dim
comptime ik_off = iq_off + iq_dim


def bench_shape[
    HAS_INDEXER: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    prompt_lens: List[Int],
    regime: String,
) raises:
    """Build device inputs / caches for `prompt_lens`, register both entries."""
    # Sparse stacks IndexQ/IndexK past the QKV bands; dense stops after V.
    comptime n_total = q_dim + 2 * kv_dim + (iq_dim + ik_dim) * Int(HAS_INDEXER)
    comptime variant = "sparse" if HAS_INDEXER else "dense "
    comptime num_bands = 3 + 2 * Int(HAS_INDEXER)

    var batch_size = len(prompt_lens)

    var total_seq = 0
    var max_seq = 0
    var iro_host = List[UInt32](length=batch_size + 1, fill=UInt32(0))
    for i in range(batch_size):
        iro_host[i] = UInt32(total_seq)
        total_seq += prompt_lens[i]
        max_seq = max(max_seq, prompt_lens[i])
    iro_host[batch_size] = UInt32(total_seq)
    var max_ctx = max_seq

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

    # ---- cache-busting inputs: fp8 operands plus both E8M0 scale tensors ----
    comptime simd_size = 4
    var cb_hs = CacheBustingBuffer[OPERAND_DTYPE](
        total_seq * hidden, simd_size, ctx
    )
    var cb_w = CacheBustingBuffer[OPERAND_DTYPE](
        n_total * hidden, simd_size, ctx
    )
    # E8M0 has no zero encoding, so a `CacheBustingBuffer` of it trips an
    # APFloat assertion in the compiler; hold the scale bytes as uint8 and
    # reinterpret them at the tensor seam instead.
    var cb_asf = CacheBustingBuffer[.uint8](
        total_seq * k_scales, simd_size, ctx
    )
    var cb_bsf = CacheBustingBuffer[.uint8](n_total * k_scales, simd_size, ctx)
    cb_hs.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_w.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_asf.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_bsf.init_on_device(InitializationType.uniform_distribution, ctx)

    # ---- outputs: fused writes Q (+ IndexQ); unfused writes one per band ----
    var q_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * q_dim)
    var q_out = LayoutTensor[OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, q_dim)](
        q_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, q_dim)].row_major(
            IndexList[2](total_seq, q_dim)
        ),
    )
    # IndexQ and the index cache are allocated for both variants: they are a few
    # hundred KiB, sit outside the timed closure, and keeping them unconditional
    # lets one closure body serve dense and sparse alike.
    var iq_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](total_seq * iq_dim)
    var iq_out = LayoutTensor[
        OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, iq_dim)
    ](
        iq_out_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, iq_dim)].row_major(
            IndexList[2](total_seq, iq_dim)
        ),
    )
    # K, V and IndexK land in dense buffers on the unfused path; the fused
    # epilogue scatters them straight into the caches instead.
    var kv_out_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        3 * total_seq * kv_dim
    )
    var kv_out_ptr = kv_out_dev.unsafe_ptr()

    # ---- KV cache blocks (main: K+V; index: MLA K-only) ----
    comptime block_layout = Layout.row_major[6]()
    var main_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, MAIN_KV_HEADS, HEAD_SIZE
    )
    var main_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        main_block_shape.flattened_length()
    )
    var main_blocks = LayoutTensor[OUT_DTYPE, block_layout](
        main_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(main_block_shape),
    )
    var index_block_shape = IndexList[6](
        num_pages, 2, num_layers, page_size, 1, HEAD_SIZE
    )
    var index_blocks_dev = ctx.enqueue_create_buffer[OUT_DTYPE](
        index_block_shape.flattened_length()
    )
    var index_blocks = LayoutTensor[OUT_DTYPE, block_layout](
        index_blocks_dev.unsafe_ptr(),
        RuntimeLayout[block_layout].row_major(index_block_shape),
    )

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

    var flops = 2 * total_seq * n_total * hidden
    # Both paths read the same cold weight and scale bytes; the unfused chain
    # re-reads the activation once per band. Every band is written exactly once,
    # so the write volume is N_total wide either way.
    var operand_bytes = n_total * hidden + n_total * k_scales
    var act_bytes = total_seq * hidden + total_seq * k_scales
    var write_elems = total_seq * n_total
    var fused_bytes = (
        operand_bytes + act_bytes + write_elems * size_of[OUT_DTYPE]()
    )
    var unfused_bytes = (
        operand_bytes
        + num_bands * act_bytes
        + write_elems * size_of[OUT_DTYPE]()
    )

    # ============ FUSED: one GEMM, scatter from the epilogue ============
    @always_inline
    def fused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_hs,
        mut cb_w,
        mut cb_asf,
        mut cb_bsf,
        mut q_out,
        mut iq_out,
        imm,
    }:
        var hs = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var w = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(n_total, hidden)
        ](
            cb_w.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(n_total, hidden)].row_major(
                IndexList[2](n_total, hidden)
            ),
        )
        var asf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(UNKNOWN_VALUE, k_scales)
        ](
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, k_scales)].row_major(
                IndexList[2](total_seq, k_scales)
            ),
        )
        var bsf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(n_total, k_scales)
        ](
            cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(n_total, k_scales)].row_major(
                IndexList[2](n_total, k_scales)
            ),
        )
        comptime if HAS_INDEXER:
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
            ](
                hs,
                iro_tensor,
                w,
                asf,
                bsf,
                Float32(1.0),
                main_collection,
                index_collection,
                UInt32(layer_idx),
                iq_dim,
                q_out,
                iq_out,
                ctx,
            )
        else:
            generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, target="gpu"
            ](
                hs,
                iro_tensor,
                w,
                asf,
                bsf,
                Float32(1.0),
                main_collection,
                UInt32(layer_idx),
                q_out,
                ctx,
            )

    @always_inline
    def fused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, fused_launch, ctx)

    m.bench_function(
        fused_bench,
        BenchId(
            "fused   "
            + variant
            + " "
            + regime
            + " total_seq="
            + String(total_seq)
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, fused_bytes),
        ],
    )

    # ============ UNFUSED: one dense GEMM per output band ============
    @always_inline
    def unfused_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_hs,
        mut cb_w,
        mut cb_asf,
        mut cb_bsf,
        mut q_out,
        mut iq_out,
        imm,
    }:
        var hs = LayoutTensor[
            mut=False, OPERAND_DTYPE, Layout.row_major(UNKNOWN_VALUE, hidden)
        ](
            cb_hs.offset_ptr(iteration),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, hidden)].row_major(
                IndexList[2](total_seq, hidden)
            ),
        )
        var asf = LayoutTensor[
            mut=False, SCALE_DTYPE, Layout.row_major(UNKNOWN_VALUE, k_scales)
        ](
            cb_asf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]](),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE, k_scales)].row_major(
                IndexList[2](total_seq, k_scales)
            ),
        )
        var hs_tt = lt_to_tt(hs).bitcast[.uint8]()
        var asf_tt = lt_to_tt(asf)

        # Q band: the only wide one (N=2048); the rest are N=128.
        @__parameter
        @always_inline
        def band[
            band_n: Int
        ](
            col_off: Int,
            out_ptr: MutPointer[Scalar[OUT_DTYPE], MutAnyOrigin],
        ) raises:
            var w = LayoutTensor[
                mut=False, OPERAND_DTYPE, Layout.row_major(band_n, hidden)
            ](
                cb_w.offset_ptr(iteration) + col_off * hidden,
                RuntimeLayout[Layout.row_major(band_n, hidden)].row_major(
                    IndexList[2](band_n, hidden)
                ),
            )
            var bsf = LayoutTensor[
                mut=False, SCALE_DTYPE, Layout.row_major(band_n, k_scales)
            ](
                cb_bsf.offset_ptr(iteration).bitcast[Scalar[SCALE_DTYPE]]()
                + col_off * k_scales,
                RuntimeLayout[Layout.row_major(band_n, k_scales)].row_major(
                    IndexList[2](band_n, k_scales)
                ),
            )
            var c = LayoutTensor[
                OUT_DTYPE, Layout.row_major(UNKNOWN_VALUE, band_n)
            ](
                out_ptr,
                RuntimeLayout[
                    Layout.row_major(UNKNOWN_VALUE, band_n)
                ].row_major(IndexList[2](total_seq, band_n)),
            )
            block_scaled_matmul_amd[lane_bytes=32](
                lt_to_tt(c),
                hs_tt,
                lt_to_tt(w).bitcast[.uint8](),
                asf_tt,
                lt_to_tt(bsf),
                ctx,
            )

        band[q_dim](0, _any(q_out.ptr))
        band[kv_dim](k_off, _any(kv_out_ptr))
        band[kv_dim](v_off, _any(kv_out_ptr) + total_seq * kv_dim)
        comptime if HAS_INDEXER:
            band[iq_dim](iq_off, _any(iq_out.ptr))
            band[ik_dim](ik_off, _any(kv_out_ptr) + 2 * total_seq * kv_dim)

        # Placing K/V (and IndexK) is the other half of what the fused epilogue
        # does, so the unfused path pays for those paged-store launches on top
        # of its band GEMMs.
        @always_inline
        @__copy_capture(kv_out_ptr)
        def k_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (_any(kv_out_ptr) + idx[0] * kv_dim + idx[2]).load[
                width=width
            ]()

        @always_inline
        @__copy_capture(kv_out_ptr, total_seq)
        def v_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (
                _any(kv_out_ptr) + total_seq * kv_dim + idx[0] * kv_dim + idx[2]
            ).load[width=width]()

        @always_inline
        @__copy_capture(kv_out_ptr, total_seq)
        def ik_in[
            width: Int, alignment: Int
        ](idx: IndexList[3]) capturing -> SIMD[OUT_DTYPE, width]:
            return (
                _any(kv_out_ptr)
                + 2 * total_seq * kv_dim
                + idx[0] * ik_dim
                + idx[2]
            ).load[width=width]()

        kv_cache_store_ragged[target="gpu", input_fn=k_in](
            main_collection.get_key_cache(layer_idx),
            IndexList[3](total_seq, MAIN_KV_HEADS, HEAD_SIZE),
            iro_tensor,
            ctx,
        )
        kv_cache_store_ragged[target="gpu", input_fn=v_in](
            main_collection.get_value_cache(layer_idx),
            IndexList[3](total_seq, MAIN_KV_HEADS, HEAD_SIZE),
            iro_tensor,
            ctx,
        )
        comptime if HAS_INDEXER:
            kv_cache_store_ragged[target="gpu", input_fn=ik_in](
                index_collection.get_key_cache(layer_idx),
                IndexList[3](total_seq, 1, HEAD_SIZE),
                iro_tensor,
                ctx,
            )

    @always_inline
    def unfused_bench(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, unfused_launch, ctx)

    m.bench_function(
        unfused_bench,
        BenchId(
            "unfused "
            + variant
            + " "
            + regime
            + " total_seq="
            + String(total_seq)
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, unfused_bytes),
        ],
    )

    _ = cb_hs^
    _ = cb_w^
    _ = cb_asf^
    _ = cb_bsf^
    _ = iro_dev^
    _ = cache_lengths_dev^
    _ = lut_dev^
    _ = q_out_dev^
    _ = iq_out_dev^
    _ = kv_out_dev^
    _ = main_blocks_dev^
    _ = index_blocks_dev^


def main() raises:
    # One variant at one shape per run; the yaml holds the sweep, which is what
    # keeps a default invocation short. All three knobs are runtime args, so a
    # sweep reuses one build: `bench_shape` is instantiated for both variants
    # and selected here rather than behind a `-D` define.
    var has_indexer = arg_parse("has_indexer", False)
    var batch_size = Int(arg_parse("batch_size", 1))
    var seq_len = Int(arg_parse("seq_len", 1))
    var is_verify = arg_parse("is_verify", False)

    seed(0)
    var m = Bench()
    with DeviceContext() as ctx:
        var prompt_lens = List[Int](length=batch_size, fill=seq_len)
        var regime = "prefill"
        if is_verify:
            regime = "verify"
        elif seq_len == 1:
            regime = "decode"
        if has_indexer:
            bench_shape[True](ctx, m, prompt_lens, regime)
        else:
            bench_shape[False](ctx, m, prompt_lens, regime)
    m.dump_report()
