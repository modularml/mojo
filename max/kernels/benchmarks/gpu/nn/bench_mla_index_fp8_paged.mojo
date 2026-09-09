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
"""Benchmark for the paged MLA FP8 indexer (`mla_indexer_ragged_float8_paged`).

Unlike `bench_fp8_index` (contiguous scorer only), this drives the full
production op — scores alloc/fill, SM100 scorer, top-k, and invalid-fill —
against a paged K cache, at MTP decode shapes.

The `frozen_cache_len` knob sets the cache collection's `max_cache_length`
metadata independently of the actual per-row cache lengths. Captured decode
device graphs bake grid dims, the score-buffer allocation, its `-inf` fill,
and the top-k scan width from this metadata at capture time (one captured
graph serves every cache length that maps to the same dispatch key, and it is
captured at the largest), so replay does metadata-proportional work no matter
the batch's real lengths. Sweeping `frozen_cache_len` at a fixed actual
`cache_len` measures exactly that gap.

Prefill shapes exercise a second axis: past `scores_budget_mb` the op scores the
matrix one row window at a time instead of materializing it, so the whole
memory/latency trade shows up here. The cost tracks the ROWS PER CHUNK the budget
works out to (`budget / (max_num_keys * 4)`), not the chunk count -- below ~500
rows it collapses. Sweep it at a prefill shape, e.g.

    ... -D scores_budget_mb=2048 -- --batch_size=8 --seq_len=512 \\
        --cache_len=76000 --frozen_cache_len=76000
"""

from std.random import rand, seed
from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    mla_indexer_ragged_float8_paged,
)
from nn.attention.mha_mask import MaskName
from std.math import ceildiv
from std.utils.index import IndexList


def _run_name[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    scores_budget_bytes: Int,
](
    batch_size: Int, seq_len: Int, cache_len: Int, frozen_cache_len: Int
) -> String:
    # fmt: off
    return String(
        "mla_indexer_fp8_paged : ",
        "num_heads=", num_heads, ", ",
        "depth=", depth, ", ",
        "page_size=", page_size, ", ",
        "top_k=", top_k, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len, ", ",
        "frozen_cache_len=", frozen_cache_len, ", ",
        "scores_budget_mb=", scores_budget_bytes // (1024 * 1024),
    )
    # fmt: on


def execute_mla_indexer_paged[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    scores_budget_bytes: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    frozen_cache_len: Int,
) raises:
    """Benchmark the paged indexer at one (actual, frozen-metadata) shape.

    Args:
        ctx: Device context.
        m: Bench harness collecting results.
        batch_size: Number of sequences (decode requests).
        seq_len: New tokens per sequence (1 + num_speculative_tokens for MTP).
        cache_len: Actual cached tokens per sequence.
        frozen_cache_len: The `max_cache_length` metadata the op sees. Must be
            >= `cache_len`. Equal reproduces eager execution;
            larger reproduces a decode graph captured at that cache length.
    """
    if frozen_cache_len < cache_len:
        raise Error("frozen_cache_len must be >= cache_len")
    var total_seq_len = batch_size * seq_len

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth, is_mla=True
    )
    comptime num_layers = 1

    # Pool holds the actual tokens; the LUT view is as wide as the frozen
    # metadata implies (mirroring capture-time `runtime_inputs`), with unused
    # tail slots pointing at block 0. The kernel never dereferences past each
    # row's real key count, so the tail is address-safety padding only.
    var real_keys_per_seq = cache_len + seq_len
    var real_pages_per_seq = ceildiv(real_keys_per_seq, page_size)
    var lut_pages_per_seq = ceildiv(frozen_cache_len + seq_len, page_size)
    var num_blocks = batch_size * real_pages_per_seq + 1

    var q_size = total_seq_len * num_heads * depth
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var qs_size = total_seq_len * num_heads
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    with qs_device.map_to_host() as qs_host:
        rand(qs_host.as_span())

    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    with input_row_offsets_device.map_to_host() as iro_host:
        for i in range(batch_size + 1):
            iro_host[i] = UInt32(i * seq_len)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_len)

    var k_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, lut_pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(lut_pages_per_seq):
                var block_idx = 0
                if page_idx < real_pages_per_seq:
                    block_idx = 1 + bs * real_pages_per_seq + page_idx
                k_lut_host[bs * lut_pages_per_seq + page_idx] = UInt32(
                    block_idx
                )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    var k_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ](
        LayoutTensor[.float8_e4m3fn, k_block_layout](
            k_block_device,
            k_block_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, cache_lengths_layout](
            cache_lengths_device,
            cache_lengths_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, paged_lut_layout](
            k_lut_device,
            paged_lut_runtime_layout,
        ),
        UInt32(seq_len),
        UInt32(frozen_cache_len),
        LayoutTensor[.float32, ks_block_layout](
            ks_block_device,
            ks_block_runtime_layout,
        ),
    )

    var o_device = ctx.enqueue_create_buffer[.int32](total_seq_len * top_k)

    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )
    var o_tile = TileTensor(o_device, row_major(total_seq_len, top_k))

    @always_inline
    def kernel_launch(
        launch_ctx: DeviceContext,
    ) raises {mut o_tile, imm}:
        mla_indexer_ragged_float8_paged[
            DType.float8_e4m3fn,
            type_of(k_collection),
            num_heads,
            depth,
            top_k,
            MaskName.CAUSAL.name,
            scores_budget_bytes,
        ](
            o_tile,
            q_tile,
            qs_tile,
            input_row_offsets_tile,
            k_collection,
            UInt32(0),
            launch_ctx,
        )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            _run_name[num_heads, depth, page_size, top_k, scores_budget_bytes](
                batch_size, seq_len, cache_len, frozen_cache_len
            )
        ),
    )

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_device


def main() raises:
    # The indexer is REPLICATED per tensor-parallel rank, not sharded: the
    # `Indexer` layer computes an `n_local_heads` but never uses it, reshaping
    # to the full `index_n_heads` instead. So GLM 5.2 puts 32 heads through this
    # kernel on every rank and DeepSeek V3.2 puts 64, whatever the TP degree.
    # 4 and 8 are reachable only where a caller shards the heads itself.
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime depth = get_defined_int["depth", 128]()
    comptime page_size = get_defined_int["page_size", 128]()
    comptime top_k = get_defined_int["top_k", 2048]()
    # 0 = follow the op's own default, so this cannot drift from production when
    # that default changes; nonzero overrides it to sweep the chunking.
    comptime budget_mb = get_defined_int["scores_budget_mb", 0]()
    comptime scores_budget_bytes = (
        _SCORES_BUDGET_BYTES if budget_mb == 0 else budget_mb * 1024 * 1024
    )

    var batch_size = arg_parse("batch_size", 8)
    # 1 + num_speculative_tokens: the MTP verify width GLM 5.2 decodes at.
    var seq_len = arg_parse("seq_len", 6)
    var cache_len = arg_parse("cache_len", 76000)
    # 0 sweeps {cache_len, GLM recipe pin 163840, GLM max_position 1048576}.
    var frozen_cache_len = arg_parse("frozen_cache_len", 0)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        if frozen_cache_len != 0:
            execute_mla_indexer_paged[
                num_heads, depth, page_size, top_k, scores_budget_bytes
            ](ctx, m, batch_size, seq_len, cache_len, frozen_cache_len)
        else:
            var frozen_sweep = [cache_len, 163840, 1048576]
            for frozen in frozen_sweep:
                if frozen < cache_len:
                    continue
                execute_mla_indexer_paged[
                    num_heads, depth, page_size, top_k, scores_budget_bytes
                ](ctx, m, batch_size, seq_len, cache_len, frozen)

    m.dump_report()
