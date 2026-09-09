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
"""Kernel-level A/B benchmark: two-step (RMSNorm + RoPE) vs fused kernel.

Measures PERF-2718: replacing the separate `fused_qk_rms_norm_ragged_paged` +
`fused_qk_rope_ragged` pair with `fused_qk_rms_norm_rope_ragged_paged` for
MiniMax-M3 decode shapes.

M3 main-attention shape (per TP4 device, decode):
    num_q_heads=16, num_kv_heads=1, head_dim=128, rope_dim=128, seq_len=1

M3 indexer shape (per TP4 device, decode):
    num_q_heads=1,  num_kv_heads=1, head_dim=128, rope_dim=64,  seq_len=1

Run locally (1×B200):
    ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_fused_qk_rms_norm_rope -- num_q_heads=16 num_kv_heads=1 head_dim=128 rope_dim=128 seq_len=1 batch_size=12

Via kbench:
    cd max/kernels && python benchmarks/autotune/kbench.py benchmarks/gpu/nn/bench_fused_qk_rms_norm_rope.yaml
"""

from std.math import ceildiv
from std.random import seed
from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
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
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache import (
    fused_qk_rms_norm_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
)
from std.utils import Index, IndexList


def _bench_name[
    dtype: DType,
    head_dim: Int,
    rope_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    variant: StaticString,
](batch_size: Int, seq_len: Int) -> String:
    # fmt: off
    return String(
        variant, "(", dtype, ")",
        " q=", num_q_heads,
        " k=", num_kv_heads,
        " hd=", head_dim,
        " rd=", rope_dim,
        " bs=", batch_size,
        " sl=", seq_len,
    )
    # fmt: on


def bench_fused_qk_rms_norm_rope[
    dtype: DType,
    head_dim: Int,
    rope_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
](ctx: DeviceContext, mut m: Bench, batch_size: Int, seq_len: Int) raises:
    """Benchmarks two-step (norm+rope) vs fused at a given shape."""
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=head_dim
    )
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    # Must exceed cache_len (80 000) so freqs[cache_len, :] is in-bounds.
    comptime max_seq_len = 131072

    var cache_len = UInt32(80_000)  # M3 production context length
    var total_seq_len = batch_size * seq_len
    var pages_per_seq = ceildiv(Int(cache_len) + seq_len, page_size)
    var num_paged_blocks = batch_size * pages_per_seq

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        num_kv_heads,
        head_dim,
    )
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)

    comptime kv_block_layout = Layout.row_major[6]()
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime paged_lut_layout = Layout.row_major[2]()

    # row_offsets is a cumulative per-batch offset array (cu_seqlens-style):
    # one entry per batch boundary, not per token -- must be sized
    # batch_size + 1, not total_seq_len + 1 (which left the tail unfilled
    # and fed garbage into get_batch_from_row_offsets' binary search for
    # any seq_len > ~3).
    var row_offsets_d = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths_d = ctx.enqueue_create_buffer[.uint32](batch_size)
    # Separate input Q buffers so each benchmark reads the same original data.
    # bench_two_step wrote its RoPE output back into q_d, which would corrupt
    # bench_fused's input if they shared a buffer.
    var q_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_q_heads * head_dim
    )
    var q_fused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_q_heads * head_dim
    )
    var q_out_ref_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_q_heads * head_dim
    )
    var q_out_fused_d = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_q_heads * head_dim
    )
    var gamma_q_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var gamma_k_d = ctx.enqueue_create_buffer[dtype](head_dim)
    var kv_blocks_ref_d = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_blocks_fused_d = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var paged_lut_d = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    comptime freqs_static_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_d = ctx.enqueue_create_buffer[dtype](max_seq_len * rope_dim)

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

    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    var q_ragged_rt = RuntimeLayout[q_ragged_layout].row_major(
        IndexList[3](total_seq_len, num_q_heads, head_dim)
    )
    with q_d.map_to_host() as q_h:
        random(LayoutTensor[dtype, q_ragged_layout](q_h, q_ragged_rt))
    ctx.enqueue_copy(q_fused_d, q_d)

    comptime gamma_layout = Layout.row_major(head_dim)
    var gamma_rt = RuntimeLayout[gamma_layout].row_major(Index(head_dim))
    with gamma_q_d.map_to_host() as gq_h:
        random(LayoutTensor[dtype, gamma_layout](gq_h, gamma_rt))
    with gamma_k_d.map_to_host() as gk_h:
        random(LayoutTensor[dtype, gamma_layout](gk_h, gamma_rt))

    var freqs_rt = RuntimeLayout[freqs_static_layout].row_major(
        IndexList[2](max_seq_len, rope_dim)
    )
    with freqs_d.map_to_host() as fr_h:
        random(LayoutTensor[dtype, freqs_static_layout](fr_h, freqs_rt))

    var kv_block_rt = RuntimeLayout[kv_block_layout].row_major(kv_block_shape)
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    random(
        LayoutTensor[dtype, kv_block_layout](
            kv_block_host.unsafe_ptr(), kv_block_rt
        )
    )
    ctx.enqueue_copy(kv_blocks_ref_d, kv_block_host)
    ctx.enqueue_copy(kv_blocks_fused_d, kv_block_host)
    ctx.synchronize()

    var q_tile = TileTensor(
        q_d, row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim]))
    )
    var q_fused_tile = TileTensor(
        q_fused_d, row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim]))
    )
    var q_out_ref_tile = TileTensor(
        q_out_ref_d,
        row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var q_out_fused_tile = TileTensor(
        q_out_fused_d,
        row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim])),
    )
    var gamma_q_tile = TileTensor(gamma_q_d, row_major[head_dim]())
    var gamma_k_tile = TileTensor(gamma_k_d, row_major[head_dim]())
    comptime freqs_tile_layout = row_major[max_seq_len, rope_dim]()
    var freqs_tile = TileTensor(freqs_d, freqs_tile_layout)
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
    var max_cache_len = UInt32(Int(cache_len))

    # Named vars for kv block tensors — avoids GenericLayoutTensorType in
    # the collection constructor (inline construction of LayoutTensor from
    # DeviceBuffer produces an unbound origin; a named var binds it).
    var kv_blocks_ref_lt = LayoutTensor[dtype, kv_block_layout](
        kv_blocks_ref_d, kv_block_rt
    )
    var kv_blocks_fused_lt = LayoutTensor[dtype, kv_block_layout](
        kv_blocks_fused_d, kv_block_rt
    )

    @always_inline
    def bench_two_step(
        mut b: Bencher,
    ) {
        var kv_blocks_ref_lt,
        var cache_lengths_tensor,
        var paged_lut_tensor,
        var q_tile,
        var q_out_ref_tile,
        var gamma_q_tile,
        var gamma_k_tile,
        var freqs_tile,
        var row_offsets_tile,
        var max_prompt_len,
        var max_cache_len,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            var kv_ref = PagedKVCacheCollection[dtype, kv_params, page_size](
                kv_blocks_ref_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            fused_qk_rms_norm_ragged_paged[
                target="gpu", multiply_before_cast=True
            ](
                q_tile.as_immut(),
                kv_ref,
                gamma_q_tile.as_immut(),
                gamma_k_tile.as_immut(),
                Float32(1e-6),
                Scalar[dtype](1.0),
                UInt32(layer_idx),
                row_offsets_tile.as_immut(),
                q_out_ref_tile,
                ctx,
            )
            # rope output goes to q_tile (original Q buf) — avoids aliasing
            # q_out_ref_tile as both mutable output and immutable input.
            fused_qk_rope_ragged[
                kv_ref.CacheType, interleaved=False, target="gpu"
            ](
                q_proj=q_out_ref_tile.as_immut(),
                input_row_offsets=row_offsets_tile.as_immut(),
                kv_collection=kv_ref,
                freqs_cis=freqs_tile.as_immut(),
                position_ids=None,
                layer_idx=UInt32(layer_idx),
                output=q_tile,
                context=ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_two_step,
        BenchId(
            _bench_name[
                dtype, head_dim, rope_dim, num_q_heads, num_kv_heads, "two_step"
            ](batch_size, seq_len)
        ),
    )

    @always_inline
    def bench_fused(
        mut b: Bencher,
    ) {
        var kv_blocks_fused_lt,
        var cache_lengths_tensor,
        var paged_lut_tensor,
        var q_fused_tile,
        var q_out_fused_tile,
        var gamma_q_tile,
        var gamma_k_tile,
        var freqs_tile,
        var row_offsets_tile,
        var max_prompt_len,
        var max_cache_len,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            var kv_fused = PagedKVCacheCollection[dtype, kv_params, page_size](
                kv_blocks_fused_lt,
                cache_lengths_tensor,
                paged_lut_tensor,
                max_prompt_len,
                max_cache_len,
            )
            var q_src = q_fused_tile.as_immut()

            @always_inline
            @__parameter
            @__copy_capture(q_src)
            def q_input_fn[
                width: Int, alignment: Int
            ](token: Int, head: Int, col: Int) -> SIMD[dtype, width]:
                return q_src.load[width=width](Coord(Index(token, head, col)))

            fused_qk_rms_norm_rope_ragged_paged[
                target="gpu",
                multiply_before_cast=True,
                interleaved=False,
                q_input_fn=q_input_fn,
            ](
                kv_fused,
                gamma_q_tile.as_immut(),
                gamma_k_tile.as_immut(),
                freqs_tile.as_immut(),
                Float32(1e-6),
                Scalar[dtype](1.0),
                UInt32(layer_idx),
                row_offsets_tile.as_immut(),
                q_out_fused_tile,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_fused,
        BenchId(
            _bench_name[
                dtype, head_dim, rope_dim, num_q_heads, num_kv_heads, "fused"
            ](batch_size, seq_len)
        ),
    )


def main() raises:
    comptime dtype = DType.bfloat16

    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime rope_dim = get_defined_int["rope_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 16]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 1]()

    var batch_size = arg_parse("batch_size", 12)
    var seq_len = arg_parse("seq_len", 1)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        bench_fused_qk_rms_norm_rope[
            dtype, head_dim, rope_dim, num_q_heads, num_kv_heads
        ](ctx, m, batch_size, seq_len)

    m.dump_report()
