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

from std.math import ceildiv, sqrt
from std.memory import alloc, dealloc
from std.random import randn
from std.sys import get_defined_dtype, get_defined_int, get_defined_bool

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import *
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from internal_utils import arg_parse, CacheBustingBuffer
from internal_utils._utils import InitializationType
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
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_prefill
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse import mla_prefill_sparse
from nn.attention.mha_mask import CausalMask

from std.utils.index import Index, IndexList


def bench_decode[
    qkv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    decoding_warp_split_k: Bool = False,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    num_partitions: Int,
    mode: String,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    # MLA decode output has the V (no-rope) portion only: depth - 64.
    comptime v_depth = depth - 64

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = batch_size * num_heads * seq_len * v_depth

    # For cache busting: calculate strides and larger buffer sizes.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[output_type](
        o_size, simd_size, ctx, cache_busting
    )

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
    ](batch_size, num_keys, 1, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {var cb_q, var cb_k, var cb_o, var scalar_args_buf_lt, imm,}:
        @always_inline
        def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var q_device = TileTensor(
                cb_q.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size,
                        seq_len,
                        Idx[num_heads],
                        Idx[depth],
                    )
                ),
            )
            var k_device = TileTensor(
                cb_k.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size,
                        num_keys,
                        Idx[kv_num_heads],
                        Idx[depth],
                    )
                ),
            )
            var output_device = TileTensor(
                cb_o.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size,
                        seq_len,
                        Idx[num_heads],
                        Idx[v_depth],
                    )
                ),
            )
            var scalar_args_tt = TileTensor(
                scalar_args_buf_lt.ptr, row_major[3]()
            )

            flare_mla_decoding[
                config=MHAConfig[qkv_type](num_heads, depth),
                decoding_warp_split_k=decoding_warp_split_k,
            ](
                output_device.as_unsafe_any_origin(),
                q_device,
                k_device,
                CausalMask(),
                scale,
                ctx,
                scalar_args_tt,
                num_partitions=num_partitions,
            )

        bencher_iter_custom(b, _kernel_launch, ctx)

    def compute_flops() {imm} -> Int:
        return 4 * batch_size * num_heads * seq_len * num_keys * depth

    m.bench_function(
        bench_func,
        BenchId(
            "mla_decode",
            # fmt: off
        input_id=String(
            "qkv_type=", qkv_type,
            "/num_heads=", num_heads,
            "/seq_len=", seq_len,
            "/num_keys=", num_keys,
            "/batch_size=", batch_size,
            "/mode=", mode,
            "/cache_busting=", cache_busting,
        ),
            # fmt: on
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_o
    _ = mla_args


def bench_prefill[
    qkv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    # For cache busting: calculate strides and larger buffer sizes.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_cache = CacheBustingBuffer[qkv_type](
        cache_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[output_type](
        o_size, simd_size, ctx, cache_busting
    )

    # input row offsets and cache row offsets
    var input_row_offsets = alloc[UInt32](
        {count = batch_size + 1}
    ).unsafe_leak()
    var cache_row_offsets = alloc[UInt32](
        {count = batch_size + 1}
    ).unsafe_leak()
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)
    cb_cache.init_on_device(random_distribution, ctx)

    # Copy from host to device
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # Row offsets tensors (these don't need cache busting offsets).
    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr,
        row_major(Coord(batch_size + 1)),
    )
    var cache_row_offsets_device = TileTensor(
        cache_row_offsets_device_ptr,
        row_major(Coord(batch_size + 1)),
    )

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {
        var cb_q,
        var cb_k,
        var cb_v,
        var cb_cache,
        var cb_o,
        var input_row_offsets_device,
        var cache_row_offsets_device,
        imm,
    }:
        @always_inline
        def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var q_device = TileTensor(
                cb_q.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size * seq_len,
                        Idx[num_heads],
                        Idx[depth],
                    )
                ),
            )
            var k_device = TileTensor(
                cb_k.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size * num_keys,
                        Idx[num_heads],
                        Idx[kv_depth],
                    )
                ),
            )
            var v_device = TileTensor(
                cb_v.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size * num_keys,
                        Idx[num_heads],
                        Idx[kv_depth],
                    )
                ),
            )
            var cache_device = TileTensor(
                cb_cache.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size,
                        num_keys,
                        Idx[cache_num_heads],
                        Idx[cache_depth],
                    )
                ),
            )
            var output_device = TileTensor(
                cb_o.offset_ptr(iteration),
                row_major(
                    Coord(
                        batch_size * seq_len,
                        Idx[num_heads],
                        Idx[kv_depth],
                    )
                ),
            )

            flare_mla_prefill[rank=q_device.rank](
                output_device,
                q_device,
                k_device,
                v_device,
                cache_device,
                CausalMask(),
                input_row_offsets_device,
                cache_row_offsets_device,
                scale,
                ctx,
                q_max_seq_len=seq_len,
            )

        bencher_iter_custom(b, _kernel_launch, ctx)

    def compute_flops() {imm} -> Int:
        return 4 * batch_size * num_heads * seq_len * num_keys * depth

    m.bench_function(
        bench_func,
        BenchId(
            "mla_prefill",
            # fmt: off
        input_id=String(
            "qkv_type=", qkv_type,
            "/num_heads=", num_heads,
            "/seq_len=", seq_len,
            "/num_keys=", num_keys,
            "/batch_size=", batch_size,
            "/kv_depth=", kv_depth,
            "/cache_depth=", cache_depth,
            "/cache_num_heads=", cache_num_heads,
            "/cache_busting=", cache_busting,
        ),
            # fmt: on
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_cache
    _ = cb_o


def bench_prefill_sparse[
    qkv_type: DType,
    output_type: DType,
    num_heads: Int,
    qk_depth: Int,
    v_depth: Int,
    topk: Int,
    b_topk: Int = 128,
    num_mbars: Int = 2,
    q_smem_depth: Int = 192,
    q_tmem_depth: Int = 384,
    page_size: Int = 128,
    cache_busting: Bool = True,
](mut m: Bench, s_q: Int, num_kv_tokens: Int, ctx: DeviceContext) raises:
    var scale = Float32(1.0) / sqrt(Float32(192.0))
    comptime kv_num_heads = 1
    comptime num_layers = 1
    comptime batch_size = 1

    var num_pages = ceildiv(num_kv_tokens, page_size)

    var q_elems = s_q * num_heads * qk_depth
    var out_elems = s_q * num_heads * v_depth
    var block_elems = (
        num_pages * num_layers * page_size * kv_num_heads * qk_depth
    )
    var total_indices = s_q * topk

    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_elems, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[output_type](
        out_elems, simd_size, ctx, cache_busting
    )
    cb_q.init_on_device(InitializationType.uniform_distribution, ctx)

    # KV blocks: host-init with random data, copy to device.
    var kv_host_alloc = alloc[Scalar[qkv_type]](
        {count = block_elems}
    ).into_managed()
    var kv_host = kv_host_alloc.unsafe_ptr()
    randn[qkv_type](
        kv_host, block_elems, mean=Float64(0.0), standard_deviation=Float64(0.5)
    )
    var blocks_device = ctx.enqueue_create_buffer[qkv_type](block_elems)
    ctx.enqueue_copy(blocks_device, kv_host)

    # Sequential LUT: page i → physical block i (batch_size=1).
    var lut_host_alloc = alloc[UInt32]({count = num_pages}).into_managed()
    var lut_host = lut_host_alloc.unsafe_ptr()
    for i in range(num_pages):
        lut_host[i] = UInt32(i)
    var lut_device = ctx.enqueue_create_buffer[.uint32](num_pages)
    ctx.enqueue_copy(lut_device, lut_host)

    var cache_lengths_host_alloc = alloc[UInt32](
        {count = batch_size}
    ).into_managed()
    var cache_lengths_host = cache_lengths_host_alloc.unsafe_ptr()
    cache_lengths_host[0] = UInt32(num_kv_tokens)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    # Physical indices: cycle through all valid (page, offset) pairs.
    var indices_host_alloc = alloc[UInt32](
        {count = total_indices}
    ).into_managed()
    var indices_host = indices_host_alloc.unsafe_ptr()
    for i in range(total_indices):
        var page_id = i % num_pages
        var tok_in_page = i % page_size
        indices_host[i] = UInt32(page_id * page_size + tok_in_page)
    var indices_device = ctx.enqueue_create_buffer[.uint32](total_indices)
    ctx.enqueue_copy(indices_device, indices_host)

    var topk_lengths_host_alloc = alloc[UInt32]({count = s_q}).into_managed()
    var topk_lengths_host = topk_lengths_host_alloc.unsafe_ptr()
    for i in range(s_q):
        topk_lengths_host[i] = UInt32(topk)
    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](s_q)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    dealloc(kv_host_alloc^)
    dealloc(lut_host_alloc^)
    dealloc(cache_lengths_host_alloc^)
    dealloc(indices_host_alloc^)
    dealloc(topk_lengths_host_alloc^)

    # Build PagedKVCacheCollection from device buffers.
    comptime kv_params = KVCacheStaticParams(
        num_heads=kv_num_heads, head_size=qk_depth, is_mla=True
    )
    comptime kv_block_layout = Layout.row_major[6]()
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lut_layout = Layout.row_major[2]()

    var kv_block_lt = LayoutTensor[qkv_type, kv_block_layout](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                num_pages, 1, num_layers, page_size, kv_num_heads, qk_depth
            )
        ),
    )
    var cache_lengths_lt = LayoutTensor[mut=False, .uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var lut_lt = LayoutTensor[mut=False, .uint32, lut_layout, _](
        lut_device.unsafe_ptr(),
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_pages)
        ),
    )

    var kv_collection = PagedKVCacheCollection[qkv_type, kv_params, page_size](
        kv_block_lt,
        cache_lengths_lt,
        lut_lt,
        UInt32(s_q),
        UInt32(num_kv_tokens),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var indices_tt = TileTensor(
        indices_device.unsafe_ptr(), row_major(total_indices)
    )
    var topk_lengths_tt = TileTensor(
        topk_lengths_device.unsafe_ptr(), row_major(s_q)
    )

    comptime config = MLASparseConfig[
        qkv_type, b_topk, num_mbars, q_smem_depth, q_tmem_depth
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=qk_depth,
        v_depth=v_depth,
        indices_stride=topk,
        group=num_heads,
    )

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {
        var cb_q,
        var cb_o,
        var kv_cache,
        var indices_tt,
        var topk_lengths_tt,
        var scale,
        imm,
    }:
        @always_inline
        def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var q_tt = TileTensor(
                cb_q.offset_ptr(iteration),
                row_major((s_q, Idx[num_heads], Idx[qk_depth])),
            )
            var out_tt = TileTensor(
                cb_o.offset_ptr(iteration),
                row_major((s_q, Idx[num_heads], Idx[v_depth])),
            )
            mla_prefill_sparse[
                config=config,
                group=num_heads,
                q_depth=qk_depth,
            ](
                out_tt,
                q_tt,
                kv_cache,
                indices_tt,
                topk_lengths_tt,
                Optional[ImmPointer[Float32, ImmutAnyOrigin]](None),
                scale,
                Int32(topk),
                ctx,
            )

        bencher_iter_custom(b, _kernel_launch, ctx)

    def compute_flops() {imm} -> Int:
        return 2 * s_q * topk * num_heads * (qk_depth + v_depth)

    m.bench_function(
        bench_func,
        BenchId(
            "mla_prefill_sparse",
            # fmt: off
            input_id=String(
                "qkv_type=", qkv_type,
                "/num_heads=", num_heads,
                "/s_q=", s_q,
                "/num_kv_tokens=", num_kv_tokens,
                "/topk=", topk,
                "/b_topk=", b_topk,
                "/num_mbars=", num_mbars,
                "/q_smem_depth=", q_smem_depth,
                "/cache_busting=", cache_busting,
            ),
            # fmt: on
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )

    ctx.synchronize()

    _ = blocks_device
    _ = lut_device
    _ = cache_lengths_device
    _ = indices_device
    _ = topk_lengths_device
    _ = cb_q
    _ = cb_o


@fieldwise_init
struct MLA_cfg(ImplicitlyCopyable, Writable):
    # params
    var qkv_type: DType
    var output_type: DType
    var depth: Int
    var prefill_depth: Int
    var num_heads: Int
    var group: Int
    var decoding_warp_split_k: Bool
    var cache_busting: Bool
    var kv_depth: Int
    var cache_depth: Int
    var cache_num_heads: Int
    # sparse prefill tuning params
    var sparse_topk: Int
    var sparse_b_topk: Int
    var sparse_num_mbars: Int
    var sparse_q_smem_depth: Int
    var sparse_q_tmem_depth: Int


def main() raises:
    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime output_type = get_defined_dtype["output_type", .bfloat16]()
    comptime depth = get_defined_int["depth", 576]()
    comptime prefill_depth = get_defined_int["prefill_depth", 192]()
    comptime num_heads = get_defined_int["num_heads", 128]()
    comptime group = get_defined_int["group", 128]()
    comptime decoding_warp_split_k = get_defined_bool[
        "decoding_warp_split_k", False
    ]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime kv_depth = get_defined_int["kv_depth", 128]()
    comptime cache_depth = get_defined_int["cache_depth", 576]()
    comptime cache_num_heads = get_defined_int["cache_num_heads", 1]()
    comptime sparse_topk = get_defined_int["sparse_topk", 2048]()
    comptime sparse_b_topk = get_defined_int["sparse_b_topk", 128]()
    comptime sparse_num_mbars = get_defined_int["sparse_num_mbars", 2]()
    comptime sparse_q_smem_depth = get_defined_int["sparse_q_smem_depth", 192]()
    comptime sparse_q_tmem_depth = get_defined_int["sparse_q_tmem_depth", 384]()

    var seq_len = Int(arg_parse("seq_len", 64))
    var num_keys = Int(arg_parse("num_keys", 64))
    var batch_size = Int(arg_parse("batch_size", 1))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    var mode = String(arg_parse("mode", "decode"))

    comptime cfg = MLA_cfg(
        qkv_type=qkv_type,
        output_type=output_type,
        depth=depth,
        prefill_depth=prefill_depth,
        num_heads=num_heads,
        group=group,
        decoding_warp_split_k=decoding_warp_split_k,
        cache_busting=cache_busting,
        kv_depth=kv_depth,
        cache_depth=cache_depth,
        cache_num_heads=cache_num_heads,
        sparse_topk=sparse_topk,
        sparse_b_topk=sparse_b_topk,
        sparse_num_mbars=sparse_num_mbars,
        sparse_q_smem_depth=sparse_q_smem_depth,
        sparse_q_tmem_depth=sparse_q_tmem_depth,
    )

    var m = Bench()
    with DeviceContext() as ctx:
        if mode != "sparse_prefill":
            bench_decode[
                cfg.qkv_type,
                cfg.output_type,
                cfg.depth,
                cfg.num_heads,
                cfg.group,
                cfg.decoding_warp_split_k,
                cfg.cache_busting,
            ](
                m,
                seq_len,
                num_keys,
                batch_size,
                num_partitions,
                mode,
                ctx,
            )

            bench_prefill[
                cfg.qkv_type,
                cfg.output_type,
                cfg.prefill_depth,
                cfg.num_heads,
                cfg.kv_depth,
                cfg.cache_depth,
                cfg.cache_num_heads,
                cache_busting=cfg.cache_busting,
            ](m, seq_len, num_keys, batch_size, ctx)

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            bench_prefill_sparse[
                cfg.qkv_type,
                cfg.output_type,
                num_heads=cfg.num_heads,
                qk_depth=cfg.depth,
                v_depth=cfg.depth - 64,
                topk=cfg.sparse_topk,
                b_topk=cfg.sparse_b_topk,
                num_mbars=cfg.sparse_num_mbars,
                q_smem_depth=cfg.sparse_q_smem_depth,
                q_tmem_depth=cfg.sparse_q_tmem_depth,
                cache_busting=cfg.cache_busting,
            ](m, seq_len, num_keys, ctx)

    m.dump_report()
