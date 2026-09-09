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

"""Correctness tests for MLA_SM100_Decode_Sparse_KV_BF16.

Covers bit-exact comparison against dense BF16 (`topk == cache_len`,
sequential page table); real sparse selection with NullMask and
CausalMask; multi-token Q (`q_max_seq_len > 1`); per-batch variable topk;
attention sinks; extra (always-attend) KV; and topk clamping when
`topk > cache_length + seq_len`.
"""

from std.math import ceildiv, exp
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
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
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from std.testing import assert_almost_equal, assert_equal
from std.utils.index import IndexList
from std.utils.numerics import isnan, min_or_neg_inf


# DeepSeek V3 MLA dims.
comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1
comptime KV_HEAD_SIZE = Q_DEPTH


def run_bit_exact_vs_dense[
    q_type: DType,
    num_heads: Int,
](ctx: DeviceContext, batch_size: Int, cache_len: Int) raises:
    """Bit-exact comparison: sparse BF16 KV vs dense BF16 KV.

    Sparse `d_indices` enumerates rows 0..cache_len-1 in the same logical
    order dense reads them (the page table is sequential, so logical row i
    maps to physical row i, and `kv_lut.get_tma_row(i) == i`).
    """
    comptime q_max_seq_len = 1
    var num_keys = cache_len + q_max_seq_len
    var topk = num_keys  # full sparse fan-out (= cache_len + 1 token)

    print(
        "BF16 sparse vs dense bit-exact: batch_size=",
        batch_size,
        " cache_len=",
        cache_len,
        " topk=",
        topk,
        " num_heads=",
        num_heads,
    )

    seed(0xC0FFEE)
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var pages_per_batch = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * pages_per_batch
    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    randn(
        blocks_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # Sequential page table: lookup[b * pages_per_batch + p] = b * pages_per_batch + p.
    # With this, dense `kv_lut.populate(batch_idx, base_row)` produces
    # physical_block = base_row // PAGE_SIZE + b * pages_per_batch, i.e. rows
    # 0..num_keys-1 in order. The sparse encoded index for token t is
    # `physical_block * PAGE_SIZE + tok_in_page`, so the sparse TMA row sequence
    # is the same.
    var lut_size = batch_size * pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for bi in range(batch_size):
        for p in range(pages_per_batch):
            lookup_table_host[bi * pages_per_batch + p] = UInt32(
                bi * pages_per_batch + p
            )

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    # Q: [batch_size * q_max_seq_len, num_heads, Q_DEPTH] (ragged with
    # row_offsets = i * q_max_seq_len).
    var q_size = batch_size * q_max_seq_len * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Sparse indices enumerate rows 0..num_keys-1 in order.
    # `physical_block * PAGE_SIZE + tok_in_page` with the identity page table
    # collapses to just `t` (since physical_block == logical_block).
    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            # logical token i; identity page table -> physical_block == logical_block.
            var page_idx = i // PAGE_SIZE
            var tok_in_page = i % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    # Device buffers (KV blocks are shared between the two runs).
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_dense = ctx.enqueue_create_buffer[q_type](
        batch_size * q_max_seq_len * num_heads * V_DEPTH
    )
    var out_sparse = ctx.enqueue_create_buffer[q_type](
        batch_size * q_max_seq_len * num_heads * V_DEPTH
    )

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    var total_q_tokens = batch_size * q_max_seq_len
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_dense_tt = TileTensor(
        out_dense.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var out_sparse_tt = TileTensor(
        out_sparse.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args_dense = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_dense_tt = mla_args_dense.gpu_tile_tensor()

    # sparse_max_topk shrink is now derived inside mla_decode_sm100_dispatch
    # from indices_stride, so it's no longer passed at host-side projection.
    var mla_args_sparse = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_sparse_tt = mla_args_sparse.gpu_tile_tensor()

    print("  Launching dense BF16 MLA decode (reference)...")
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
    ](
        out_dense_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_dense_tt,
    )
    ctx.synchronize()

    print("  Launching sparse BF16 MLA decode...")
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_sparse_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_sparse_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=topk,
    )
    ctx.synchronize()

    var out_dense_host = ctx.enqueue_create_host_buffer[q_type](
        batch_size * q_max_seq_len * num_heads * V_DEPTH
    )
    var out_sparse_host = ctx.enqueue_create_host_buffer[q_type](
        batch_size * q_max_seq_len * num_heads * V_DEPTH
    )
    ctx.enqueue_copy(out_dense_host, out_dense)
    ctx.enqueue_copy(out_sparse_host, out_sparse)
    ctx.synchronize()

    var mismatches = 0
    for i in range(batch_size * q_max_seq_len * num_heads * V_DEPTH):
        var d = out_dense_host[i]
        var s = out_sparse_host[i]
        if d != s:
            mismatches += 1
            if mismatches <= 5:
                print(
                    "  mismatch idx=",
                    i,
                    " dense=",
                    d,
                    " sparse=",
                    s,
                )

    if mismatches > 0:
        print("  FAILED: ", mismatches, "bit-exact mismatches")
        raise Error("sparse BF16 KV output does not match dense BF16")
    # Defensive assert (mismatch count would already have raised).
    assert_equal(mismatches, 0)
    print("  PASSED: outputs are bit-exact")

    _ = mla_args_dense
    _ = mla_args_sparse
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_dense
    _ = out_sparse
    _ = d_indices_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Helpers for sparse-with-tolerance tests
# ===-----------------------------------------------------------------------===#


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    if n <= 1:
        return 1
    if _gcd(3, n) == 1:
        return 3
    if _gcd(5, n) == 1:
        return 5
    if _gcd(7, n) == 1:
        return 7
    if _gcd(11, n) == 1:
        return 11
    return 13


def host_reference_varkeys[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """CPU reference for variable-length per-batch sparse attention."""
    var k_offsets = List(length=batch_size, fill=Int(0))
    var running = 0
    for b in range(batch_size):
        k_offsets[b] = running
        running += num_keys_per_batch[b] * depth

    for b in range(batch_size):
        var num_keys = num_keys_per_batch[b]
        var k_base_offset = k_offsets[b]
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth
            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = k_base_offset + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = k_base_offset + k * depth
                    acc += s_buf[k] * k_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^
    _ = k_offsets^


def host_reference_with_attn_sink[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    attn_sink_host: Pointer[Float32, _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """CPU reference with attn_sink correction (natural log domain)."""
    for b in range(batch_size):
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth
            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = b * num_keys * depth + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var attn_sink_val = Float64(attn_sink_host[h])
            if attn_sink_val > max_s:
                max_s = attn_sink_val

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            sum_exp += exp(attn_sink_val - max_s)

            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    acc += s_buf[k] * k_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^


# ===-----------------------------------------------------------------------===#
# Sparse selection (topk < cache_len), NullMask / CausalMask, spec decode.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_bf16[
    q_type: DType,
    num_heads: Int,
    use_causal: Bool = False,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
) raises:
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
        " q_max_seq_len:",
        q_max_seq_len,
        " causal:",
        use_causal,
    )

    var num_keys = cache_len + q_max_seq_len
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)

    var k_total = batch_size * num_keys * Q_DEPTH
    var k_host = ctx.enqueue_create_host_buffer[q_type](k_total)
    randn(
        k_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var tok_stride = kv_params.head_size  # 576 BF16 elems

    # Build shuffled page table so gather4 exercises scatter.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    # Fill KV cache from k_host (BF16 — no quantization).
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_host[k_base + d]

    # Select topk unique tokens PER QUERY TOKEN via deterministic permutation.
    #
    # The sparse decode kernel gathers d_indices per query token, keyed by the
    # global query-token index g = batch * q_max_seq_len + s. To actually
    # exercise that (and catch a regression to per-batch gather), give each
    # query token a DISTINCT top-k set: a per-token rotation `+ s` shifts the
    # selected positions per query token. Because `mult` is coprime to
    # num_keys and topk <= num_keys, each token's set stays internally unique
    # while differing across query tokens, so a kernel that read the wrong
    # token's indices would pick different keys and fail.
    var selected_tokens = List(length=total_q_tokens * topk, fill=Int(0))
    var mult = _coprime_multiplier(num_keys)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                selected_tokens[g * topk + i] = (i * mult + 1 + s) % num_keys

    # Build sparse reference K buffer [total_q_tokens, topk, Q_DEPTH] — one
    # selected-key set per query token.
    var k_sparse_ref_size = total_q_tokens * topk * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](k_sparse_ref_size)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                var t = selected_tokens[g * topk + i]
                var src_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
                var dst_base = g * topk * Q_DEPTH + i * Q_DEPTH
                for d in range(Q_DEPTH):
                    k_sparse_ref[dst_base + d] = k_host[src_base + d]

    var q_size = batch_size * q_max_seq_len * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)

    if use_causal:
        # Causal sparse reference: per (q_token s) only attend to selected
        # tokens whose logical position < cache_len + s + 1.
        for b in range(batch_size):
            for s in range(q_max_seq_len):
                var g = b * q_max_seq_len + s
                var causal_limit = cache_len + s + 1
                for h in range(num_heads):
                    var q_base = (
                        b * q_max_seq_len * num_heads * Q_DEPTH
                        + s * num_heads * Q_DEPTH
                        + h * Q_DEPTH
                    )
                    var max_s = Float64(min_or_neg_inf[.float32]())
                    var s_buf = List(length=topk, fill=Float64(0))
                    var valid = List(length=topk, fill=False)
                    for i in range(topk):
                        var tok = selected_tokens[g * topk + i]
                        if tok >= causal_limit:
                            valid[i] = False
                            s_buf[i] = Float64(min_or_neg_inf[.float32]())
                            continue
                        valid[i] = True
                        var k_base = g * topk * Q_DEPTH + i * Q_DEPTH
                        var dot = Float64(0)
                        for d in range(Q_DEPTH):
                            dot += (
                                q_host[q_base + d].cast[.float64]()
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        s_buf[i] = dot * Float64(scale)
                        if s_buf[i] > max_s:
                            max_s = s_buf[i]

                    var sum_exp = Float64(0)
                    for i in range(topk):
                        if not valid[i]:
                            s_buf[i] = Float64(0)
                            continue
                        s_buf[i] = exp(s_buf[i] - max_s)
                        sum_exp += s_buf[i]
                    if sum_exp > Float64(0):
                        for i in range(topk):
                            if valid[i]:
                                s_buf[i] = s_buf[i] / sum_exp

                    var o_base = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                    )
                    for d in range(V_DEPTH):
                        var acc = Float64(0)
                        for i in range(topk):
                            if not valid[i]:
                                continue
                            var k_base = g * topk * Q_DEPTH + i * Q_DEPTH
                            acc += (
                                s_buf[i]
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        ref_host[o_base + d] = acc.cast[q_type]()
                    _ = valid^
                    _ = s_buf^
    else:
        # Non-causal reference. Each query token attends to its OWN top-k set
        # (k_sparse_ref is laid out [total_q_tokens, topk, Q_DEPTH]), so we
        # cannot use the shared-key host_reference helper here.
        for b in range(batch_size):
            for s in range(q_max_seq_len):
                var g = b * q_max_seq_len + s
                for h in range(num_heads):
                    var q_base = (
                        b * q_max_seq_len * num_heads * Q_DEPTH
                        + s * num_heads * Q_DEPTH
                        + h * Q_DEPTH
                    )
                    var max_s = Float64(min_or_neg_inf[.float32]())
                    var s_buf = List(length=topk, fill=Float64(0))
                    for i in range(topk):
                        var k_base = g * topk * Q_DEPTH + i * Q_DEPTH
                        var dot = Float64(0)
                        for d in range(Q_DEPTH):
                            dot += (
                                q_host[q_base + d].cast[.float64]()
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        s_buf[i] = dot * Float64(scale)
                        if s_buf[i] > max_s:
                            max_s = s_buf[i]

                    var sum_exp = Float64(0)
                    for i in range(topk):
                        s_buf[i] = exp(s_buf[i] - max_s)
                        sum_exp += s_buf[i]
                    for i in range(topk):
                        s_buf[i] = s_buf[i] / sum_exp

                    var o_base = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                    )
                    for d in range(V_DEPTH):
                        var acc = Float64(0)
                        for i in range(topk):
                            var k_base = g * topk * Q_DEPTH + i * Q_DEPTH
                            acc += (
                                s_buf[i]
                                * k_sparse_ref[k_base + d].cast[.float64]()
                            )
                        ref_host[o_base + d] = acc.cast[q_type]()
                    _ = s_buf^

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    # Build d_indices, laid out [total_q_tokens, topk]. The sparse decode
    # kernel gathers per query token (global index g = batch * q_max_seq_len
    # + s), each row using that token's own selected_tokens set.
    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                var t = selected_tokens[g * topk + i]
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
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

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    var indices_stride = topk
    print(
        "  Launching MLA sparse KV_BF16 decode... topk=",
        topk,
        " num_keys=",
        num_keys,
    )

    comptime if use_causal:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
        )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(5e-3)
    var atol = Float64(5e-3)
    var max_err = Float64(0)
    var nan_count = 0
    var total_checked = 0
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[.float64]()
                    var actual_val = out_host[idx].cast[.float64]()
                    if isnan(actual_val):
                        nan_count += 1
                        if nan_count <= 5:
                            print(
                                "  NaN at b=",
                                b,
                                " s=",
                                s,
                                " h=",
                                h,
                                " d=",
                                d,
                            )
                        continue
                    var err = abs(actual_val - ref_val)
                    if err > max_err:
                        max_err = err
                    total_checked += 1
                    if err > 1e-2:
                        print(
                            "  large err b=",
                            b,
                            " s=",
                            s,
                            " h=",
                            h,
                            " d=",
                            d,
                            " got=",
                            actual_val,
                            " ref=",
                            ref_val,
                            " err=",
                            err,
                        )

    if nan_count > 0:
        print(
            "  FAILED: ",
            nan_count,
            "NaN values in output (max_err over non-NaN:",
            max_err,
            ")",
        )
        raise Error("NaN in kernel output")

    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[.float64]()
                    var actual_val = out_host[idx].cast[.float64]()
                    assert_almost_equal(
                        actual_val, ref_val, atol=atol, rtol=rtol
                    )

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Variable per-batch topk (has_variable_topk).
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_bf16_variable_topk[
    q_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH

    var k_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(k_host.as_span(), mean=0.0, standard_deviation=0.5)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_host[k_base + d]
        k_offset += nk * Q_DEPTH

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # d_indices: padded to max_topk per batch.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)

    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += topk_per_batch[bi] * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_host[src_base + d]
        sparse_ref_offset += topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(topk_per_batch[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )
    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA sparse KV_BF16 (variable topk)...")
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(5e-3)
    var atol = Float64(5e-3)
    var max_err = Float64(0)
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                if err > 1e-2:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print("  PASSED: max_err=", max_err)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Attention sink (has_attn_sink).
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_bf16_attn_sink[
    q_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
) raises:
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
    )

    comptime q_max_seq_len = 1
    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)
    var k_total = batch_size * num_keys * Q_DEPTH
    var k_host = ctx.enqueue_create_host_buffer[q_type](k_total)
    randn(k_host.as_span(), mean=0.0, standard_deviation=0.5)

    var tok_stride = kv_params.head_size

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_host[k_base + d]

    # Select topk tokens via deterministic permutation.
    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        batch_size * topk * Q_DEPTH
    )
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            var dst = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst + d] = k_host[src + d]

    # attn_sink per head (natural log domain).
    var attn_sink_host = ctx.enqueue_create_host_buffer[.float32](num_heads)
    for h in range(num_heads):
        attn_sink_host[h] = Float32(
            -1.0 + 3.0 * Float32(h) / Float32(num_heads)
        )

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.3)

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_with_attn_sink[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        attn_sink_host.unsafe_ptr(),
        batch_size,
        num_heads,
        topk,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    var attn_sink_device = ctx.enqueue_create_buffer[.float32](num_heads)
    ctx.enqueue_copy(attn_sink_device, attn_sink_host)
    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )
    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    # Build d_indices.
    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA sparse KV_BF16 (attn_sink)...")
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=topk,
        attn_sink_ptr=rebind[MutPointer[Float32, origin=MutAnyOrigin]](
            attn_sink_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # attn_sink injects an extra exp(attn_sink - max_s) into the denominator;
    # BF16 quantization of Q and K combined with this extra non-linear term
    # produces ~1-1.5e-2 scatter at attn_sink magnitudes near +2.0.
    var max_err = Float64(0)
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                if err > 3e-2:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=2e-2, rtol=2e-2)

    print("  PASSED: max_err=", max_err)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = attn_sink_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Extra (always-attend) KV (has_extra_kv).
# Two paged KV caches (main + extra), both BF16, SWIZZLE_128B.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_bf16_extra_kv[
    q_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var max_extra_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        if extra_topk_per_batch[bi] > max_extra_topk:
            max_extra_topk = extra_topk_per_batch[bi]

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " extra_cache_len=",
            extra_cache_lengths[i],
            " extra_topk=",
            extra_topk_per_batch[i],
        )

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1
    var tok_stride = kv_params.head_size
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- ORIGINAL cache ---
    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)
    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH
    var k_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(k_host.as_span(), mean=0.0, standard_deviation=0.5)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_host[k_base + d]
        k_offset += nk * Q_DEPTH

    # Build the EXTRA cache state first — its lookup table is needed to
    # interleave [orig, extra] per batch into k_combined_ref below.
    # --- EXTRA cache ---
    var max_extra_cache_len = 0
    var extra_total_pages = 0
    var extra_num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = extra_cache_lengths[i]
        if cl > max_extra_cache_len:
            max_extra_cache_len = cl
        var nk = cl + q_max_seq_len
        extra_num_keys_list.append(nk)
        extra_total_pages += ceildiv(nk, PAGE_SIZE)
    var max_extra_num_keys = max_extra_cache_len + q_max_seq_len
    var max_extra_pages_per_batch = ceildiv(max_extra_num_keys, PAGE_SIZE)

    var extra_block_shape = IndexList[6](
        extra_total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var extra_block_elems = (
        extra_total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var extra_blocks_host = ctx.enqueue_create_host_buffer[q_type](
        extra_block_elems
    )
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[q_type](0)
    var extra_lut_size = batch_size * max_extra_pages_per_batch
    var extra_lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](
        extra_lut_size
    )
    for i in range(extra_lut_size):
        extra_lookup_table_host[i] = UInt32(0)
    var extra_page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(extra_num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            extra_lookup_table_host[
                bi * max_extra_pages_per_batch + p
            ] = UInt32(extra_page_offset + shuffled_p)
        extra_page_offset += np_bi

    var extra_cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size
    )
    for i in range(batch_size):
        extra_cache_lengths_host[i] = UInt32(extra_cache_lengths[i])

    var extra_total_k_elems = 0
    for bi in range(batch_size):
        extra_total_k_elems += extra_num_keys_list[bi] * Q_DEPTH
    var extra_k_host = ctx.enqueue_create_host_buffer[q_type](
        extra_total_k_elems
    )
    randn(extra_k_host.as_span(), mean=0.0, standard_deviation=0.5)

    var ek_offset = 0
    for bi in range(batch_size):
        var enk = extra_num_keys_list[bi]
        for t in range(enk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = ek_offset + t * Q_DEPTH
            for d in range(Q_DEPTH):
                extra_blocks_host[base + d] = extra_k_host[k_base + d]
        ek_offset += enk * Q_DEPTH

    # Build d_indices, extra_d_indices, and k_combined_ref together,
    # interleaving [orig | extra] per batch so the host reference layout
    # matches what the kernel sees (orig tokens first, then extras, per
    # batch).
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var extra_total_indices = batch_size * max_extra_topk
    var extra_h_indices = ctx.enqueue_create_host_buffer[.int32](
        extra_total_indices
    )
    for i in range(extra_total_indices):
        extra_h_indices[i] = Int32(0)

    var combined_ref_elems = 0
    for bi in range(batch_size):
        combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * Q_DEPTH
    var k_combined_ref = ctx.enqueue_create_host_buffer[q_type](
        combined_ref_elems
    )

    var combined_ref_offset = 0
    var k_offset_src = 0
    var ek_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = k_host[src_base + d]
        combined_ref_offset += topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

        var enk = extra_num_keys_list[bi]
        var extra_topk_bi = extra_topk_per_batch[bi]
        var mult_e = _coprime_multiplier(enk)
        for i in range(extra_topk_bi):
            var t = (i * mult_e + 1) % enk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            extra_h_indices[bi * max_extra_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = ek_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = extra_k_host[src_base + d]
        combined_ref_offset += extra_topk_bi * Q_DEPTH
        ek_offset_src += enk * Q_DEPTH

    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_combined_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        combined_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var extra_blocks_device = ctx.enqueue_create_buffer[q_type](
        extra_block_elems
    )
    ctx.enqueue_copy(extra_blocks_device, extra_blocks_host)
    var extra_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    ctx.enqueue_copy(extra_cache_lengths_device, extra_cache_lengths_host)
    var extra_lookup_table_device = ctx.enqueue_create_buffer[.uint32](
        extra_lut_size
    )
    ctx.enqueue_copy(extra_lookup_table_device, extra_lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var extra_d_indices_device = ctx.enqueue_create_buffer[.int32](
        extra_total_indices
    )
    ctx.enqueue_copy(extra_d_indices_device, extra_h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    var extra_topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](
        batch_size
    )
    for bi in range(batch_size):
        extra_topk_lengths_host[bi] = Int32(extra_topk_per_batch[bi])
    var extra_topk_lengths_device = ctx.enqueue_create_buffer[.int32](
        batch_size
    )
    ctx.enqueue_copy(extra_topk_lengths_device, extra_topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )
    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var extra_blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        extra_blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(extra_block_shape),
    )
    var extra_cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        extra_cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var extra_lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        extra_lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_extra_pages_per_batch)
        ),
    )
    var extra_kv_collection = PagedKVCacheCollection[
        q_type, kv_params, PAGE_SIZE
    ](
        LayoutTensor[q_type, Layout.row_major[6]()](
            extra_blocks_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                extra_blocks_lt.runtime_layout.shape.value,
                extra_blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            extra_cache_lengths_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[cl_layout](
                extra_cache_lengths_lt.runtime_layout.shape.value,
                extra_cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            extra_lookup_table_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[lt_layout_2d](
                extra_lookup_table_lt.runtime_layout.shape.value,
                extra_lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_extra_cache_len),
    )
    var extra_kv_cache = extra_kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA sparse KV_BF16 (extra KV)...")
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
        extra_k=extra_kv_cache,
        extra_d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            extra_d_indices_device.unsafe_ptr()
        ),
        extra_indices_stride=max_extra_topk,
        extra_topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            extra_topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var max_err = Float64(0)
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                if err > 1e-2:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=5e-3, rtol=5e-3)

    print("  PASSED: max_err=", max_err)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = extra_blocks_device
    _ = extra_cache_lengths_device
    _ = extra_lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = extra_d_indices_device
    _ = topk_lengths_device
    _ = extra_topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Topk clamping: cover the case where topk_per_batch[b] > actual_tokens[b].
# `topk_lengths` carries the UNCLAMPED user-supplied topk; the kernel clamps
# to `min(topk, cache_length + seq_len)` and reads only that many valid
# indices.  Indices beyond `effective_topk` are filled with -1 to catch OOB
# reads.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_bf16_topk_clamping[
    q_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Topk clamping with BF16 KV layout.

    When topk_per_batch[b] > actual_tokens[b] (= cache_length + seq_len),
    the kernel should clamp to actual_tokens. Indices beyond effective_topk
    are filled with -1 to catch OOB reads.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var effective_topk_list = List[Int]()
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        var actual_tokens = cache_lengths[bi] + q_max_seq_len
        var eff_topk = topk_per_batch[bi]
        if eff_topk > actual_tokens:
            eff_topk = actual_tokens
        effective_topk_list.append(eff_topk)

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " actual_tokens=",
            cache_lengths[i] + q_max_seq_len,
            " effective_topk=",
            effective_topk_list[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[q_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH
    var k_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # Fill KV cache from k_host (BF16 — no quantization).
    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_host[k_base + d]
        k_offset += nk * Q_DEPTH

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # d_indices: first effective_topk entries valid; rest are -1 sentinel.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(-1)

    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += effective_topk_list[bi] * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var eff_topk = effective_topk_list[bi]
        var mult = _coprime_multiplier(nk) if nk > 1 else 1
        for i in range(eff_topk):
            var t: Int
            if nk == 1:
                t = 0
            else:
                t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_host[src_base + d]
        sparse_ref_offset += eff_topk * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(effective_topk_list[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Uploads.
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])  # UNCLAMPED
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )
    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6]()](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=False,
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=False, half_sms=sm_count // 2
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        sm_count,
    )
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = max_topk
    print("  Launching MLA sparse KV_BF16 (topk clamping)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=indices_stride,
        topk_lengths=rebind[MutPointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(5e-3)
    var atol = Float64(5e-3)
    var max_err = Float64(0)
    var nan_count = 0
    var total_checked = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                if isnan(actual_val):
                    nan_count += 1
                    if nan_count <= 5:
                        print("  NaN at b=", b, " h=", h, " d=", d)
                    continue
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-2:
                    print(b, h, d, actual_val, ref_val, err)

    if nan_count > 0:
        print(
            "  FAILED: ",
            nan_count,
            "NaN values in output (max_err over non-NaN:",
            max_err,
            ")",
        )
        raise Error("NaN in kernel output")

    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[.float64]()
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            # Bit-exact vs dense (topk == cache_len).
            run_bit_exact_vs_dense[.bfloat16, 128](
                ctx, batch_size=1, cache_len=2048
            )

            # Sparse selection (topk < cache_len).
            # bs=1, h=64, cl=256, topk=16.
            run_test_sparse_kv_bf16[.bfloat16, 64](
                "sparse_kv_bf16_b1_h64_cl256_topk16",
                1,
                256,
                ctx,
                topk=16,
            )
            # bs=1, h=128, cl=512, topk=64.
            run_test_sparse_kv_bf16[.bfloat16, 128](
                "sparse_kv_bf16_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )
            # bs=4, h=64, cl=2048, topk=128 (split-K likely).
            run_test_sparse_kv_bf16[.bfloat16, 64](
                "sparse_kv_bf16_b4_h64_cl2048_topk128",
                4,
                2048,
                ctx,
                topk=128,
            )

            # CausalMask.
            run_test_sparse_kv_bf16[.bfloat16, 64, use_causal=True](
                "sparse_kv_bf16_causal_b1_h64_cl256_topk64",
                1,
                256,
                ctx,
                topk=64,
            )

            # Speculative decode (q_max_seq_len > 1).
            run_test_sparse_kv_bf16[.bfloat16, 64](
                "sparse_kv_bf16_b1_h64_cl256_topk64_seq4",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )
            run_test_sparse_kv_bf16[.bfloat16, 64](
                "sparse_kv_bf16_b1_h64_cl256_topk64_seq2",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=2,
            )
            run_test_sparse_kv_bf16[.bfloat16, 64](
                "sparse_kv_bf16_b2_h64_cl256_topk64_seq8",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=8,
            )

            # Variable per-batch topk.
            var vt_cls: List[Int] = [256, 384, 128, 512]
            var vt_topk: List[Int] = [64, 128, 32, 64]
            run_test_sparse_kv_bf16_variable_topk[.bfloat16, 64](
                "sparse_kv_bf16_variable_topk_b4_h64",
                vt_cls,
                vt_topk,
                ctx,
            )

            # Attention sink.
            run_test_sparse_kv_bf16_attn_sink[.bfloat16, 64](
                "sparse_kv_bf16_attn_sink_b1_h64_cl256_topk64",
                1,
                256,
                ctx,
                topk=64,
            )
            run_test_sparse_kv_bf16_attn_sink[.bfloat16, 128](
                "sparse_kv_bf16_attn_sink_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Extra (always-attend) KV.
            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [64]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_kv_bf16_extra_kv[.bfloat16, 64](
                "sparse_kv_bf16_extra_kv_b1_h64_topk64_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            var ek_cls_2: List[Int] = [256, 384]
            var ek_topk_2: List[Int] = [64, 64]
            var ek_ecls_2: List[Int] = [64, 128]
            var ek_etopk_2: List[Int] = [64, 64]
            run_test_sparse_kv_bf16_extra_kv[.bfloat16, 64](
                "sparse_kv_bf16_extra_kv_b2_h64_variable",
                ek_cls_2,
                ek_topk_2,
                ek_ecls_2,
                ek_etopk_2,
                ctx,
            )

            # =====================================================
            # Topk clamping (small caches).
            # =====================================================

            # First decode: cache_length=0, actual=1, topk=64 -> clamp to 1.
            var tc_cls_1: List[Int] = [0]
            var tc_topk_1: List[Int] = [64]
            run_test_sparse_kv_bf16_topk_clamping[.bfloat16, 64](
                "sparse_kv_bf16_topk_clamp_first_exec_b1_h64",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, actual=6, topk=64 -> clamp to 6.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_kv_bf16_topk_clamping[.bfloat16, 64](
                "sparse_kv_bf16_topk_clamp_small_cache_b1_h64",
                tc_cls_2,
                tc_topk_2,
                ctx,
            )

            # Mixed batch: cl=0 and cl=256.
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_kv_bf16_topk_clamping[.bfloat16, 64](
                "sparse_kv_bf16_topk_clamp_mixed_b2_h64",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )
