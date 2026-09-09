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

"""Benchmark for the SM100 sparse MLA decode over a paged FP8 latent cache.

Covers the speculative-decode verify: several query positions per sequence,
each attending its own top-k window. The measured region is the split-K decode
and the combine that reduces its partial outputs.

The two `q_dtype` values reach different kernels, so both belong in the sweep.
Cache length is the axis worth sweeping because the kernel reads `top_k` keys
per position regardless, so the curve flattens once the window saturates.
"""

from std.math import ceildiv
from std.random import rand, seed
from std.sys import get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils.index import IndexList

comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime Q_DEPTH = KV_LATENT_DIM + ROPE_DEPTH
comptime V_DEPTH = KV_LATENT_DIM
comptime NUM_LAYERS = 1


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Returns a stride that visits every position of a length-`n` ring.

    A stride sharing a factor with `n` revisits a subset, so the gather would
    touch far fewer pages than a real top-k does.

    Args:
        n: Ring length to be coprime with.

    Returns:
        A multiplier coprime with `n`.
    """
    if n <= 1:
        return 1
    for m in [3, 5, 7, 11]:
        if _gcd(m, n) == 1:
            return m
    return 13


def _run_name[
    q_dtype: DType, num_heads: Int, page_size: Int, top_k: Int
](batch_size: Int, seq_len: Int, cache_len: Int) -> String:
    # fmt: off
    return String(
        "mla_decode_sparse_fp8 : ",
        "q_dtype=", q_dtype, ", ",
        "num_heads=", num_heads, ", ",
        "page_size=", page_size, ", ",
        "top_k=", top_k, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len,
    )
    # fmt: on


def execute_mla_decode_sparse[
    q_dtype: DType,
    num_heads: Int,
    page_size: Int,
    top_k: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
) raises:
    """Benchmarks the sparse decode at one shape.

    Parameters:
        q_dtype: Dtype Q is handed to the kernel in.
        num_heads: Query heads on this device.
        page_size: KV cache page size.
        top_k: Sparse index count per query position.

    Args:
        ctx: Device context.
        m: Bench harness collecting results.
        batch_size: Number of sequences.
        seq_len: Query positions per sequence.
        cache_len: Cached tokens per sequence.
    """
    comptime kv_dtype = DType.float8_e4m3fn
    comptime scale = Float32(0.125)
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )

    var num_keys = cache_len + seq_len
    var total_q_tokens = batch_size * seq_len
    var pages_per_seq = ceildiv(num_keys, page_size)
    var total_pages = batch_size * pages_per_seq

    var block_shape = IndexList[6](
        total_pages, 1, NUM_LAYERS, page_size, kv_params.num_heads, Q_DEPTH
    )
    var blocks_device = ctx.enqueue_create_buffer[kv_dtype](
        block_shape.flattened_length()
    )
    with blocks_device.map_to_host() as blocks_host:
        rand(blocks_host.as_span())

    var lut_size = batch_size * pages_per_seq
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    with lookup_table_device.map_to_host() as lut_host:
        for i in range(lut_size):
            lut_host[i] = UInt32(i)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_len)

    var q_device = ctx.enqueue_create_buffer[q_dtype](
        total_q_tokens * num_heads * Q_DEPTH
    )
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var out_device = ctx.enqueue_create_buffer[.bfloat16](
        total_q_tokens * num_heads * V_DEPTH
    )

    # Physical slot ids, spread across the sequence's pages the way a real
    # top-k list is rather than clustered.
    var d_indices_device = ctx.enqueue_create_buffer[.int32](
        total_q_tokens * top_k
    )
    var mult = _coprime_multiplier(num_keys)
    with d_indices_device.map_to_host() as idx_host:
        for bi in range(batch_size):
            for s in range(seq_len):
                var g = bi * seq_len + s
                for i in range(top_k):
                    var t = (i * mult + 1 + s) % num_keys
                    var block_id = bi * pages_per_seq + t // page_size
                    idx_host[g * top_k + i] = Int32(
                        block_id * page_size + t % page_size
                    )

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    with row_offsets_device.map_to_host() as ro_host:
        for i in range(batch_size + 1):
            ro_host[i] = UInt32(i * seq_len)
    ctx.synchronize()

    comptime blocks_layout = Layout.row_major[6]()
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lut_layout = Layout.row_major[2]()

    var kv_collection = PagedKVCacheCollection[kv_dtype, kv_params, page_size](
        LayoutTensor[kv_dtype, blocks_layout, MutAnyOrigin](
            blocks_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[blocks_layout].row_major(block_shape),
        ),
        LayoutTensor[.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
        ),
        LayoutTensor[.uint32, lut_layout, ImmutAnyOrigin](
            lookup_table_device.unsafe_ptr().as_unsafe_any_origin(),
            RuntimeLayout[lut_layout].row_major(
                IndexList[2](batch_size, pages_per_seq)
            ),
        ),
        UInt32(seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(), row_major(batch_size + 1)
    )

    var mla_args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, cache_len, seq_len, ctx
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()
    var d_indices = rebind[MutPointer[Int32, MutAnyOrigin]](
        d_indices_device.unsafe_ptr()
    )

    @always_inline
    def kernel_launch(
        launch_ctx: DeviceContext,
    ) raises {
        mut out_tt,
        imm q_tt,
        imm kv_cache,
        imm row_offsets_tt,
        imm scalar_args_buf_tt,
        imm d_indices,
    }:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_dtype](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            launch_ctx,
            scalar_args_buf_tt,
            d_indices=d_indices,
            indices_stride=top_k,
        )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            _run_name[q_dtype, num_heads, page_size, top_k](
                batch_size, seq_len, cache_len
            )
        ),
    )

    _ = mla_args
    _ = blocks_device
    _ = lookup_table_device
    _ = cache_lengths_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device


def main() raises:
    comptime q_dtype = get_defined_dtype["q_dtype", .float8_e4m3fn]()
    comptime num_heads = get_defined_int["num_heads", 8]()
    comptime page_size = get_defined_int["page_size", 128]()
    comptime top_k = get_defined_int["top_k", 2048]()

    var batch_size = arg_parse("batch_size", 8)
    # 1 + num_speculative_tokens: the verify width a spec-decode step runs at.
    var seq_len = arg_parse("seq_len", 6)
    # 0 sweeps the cache axis, crossing the point where the top-k window
    # saturates and the work per position stops growing.
    var cache_len = arg_parse("cache_len", 0)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        if cache_len != 0:
            execute_mla_decode_sparse[q_dtype, num_heads, page_size, top_k](
                ctx, m, batch_size, seq_len, cache_len
            )
        else:
            for c in [512, 1024, 2048, 4096, 8192, 16384]:
                execute_mla_decode_sparse[q_dtype, num_heads, page_size, top_k](
                    ctx, m, batch_size, seq_len, c
                )

    m.dump_report()
