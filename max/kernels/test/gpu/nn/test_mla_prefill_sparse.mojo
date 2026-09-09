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

"""Numerical E2E test for mla_prefill_sparse kernel (BF16).

This kernel performs sparse MLA prefill attention over a subset of KV tokens
selected by an external scoring phase (the indexer). DSv3.2 only uses the
"absorbed" / latent shape: qk_depth = kv_lora_rank(512) + qk_rope_head_dim(64)
= 576, v_depth = 512.

Memory layout:
  - KV:     Paged KV cache (PagedKVCacheCollection) with BF16 data.
            head_size = qk_depth.  V is the first v_depth columns.
  - Q:      [total_q_tokens, num_heads, qk_depth]   BF16
  - output: [total_q_tokens, num_heads, v_depth]    BF16
  - indices:       [total_q_tokens, indices_stride] uint32 (PER-QUERY)
  - topk_lengths:  [total_q_tokens]                 uint32 (PER-QUERY)

Host reference (per query row):
    Score = Q @ KV_sel^T * scale
    O     = softmax(Score) @ V_sel
"""

from std.math import ceildiv, exp2, sqrt
from std.math.constants import log2e
from std.memory import UnsafePointer, alloc
from std.random import randn
from std.sys import has_nvidia_gpu_accelerator, size_of

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext
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
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import DynamicInt
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse import mla_prefill_sparse
from std.utils.index import Index, IndexList
from std.utils.numerics import min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# DSv3.2 absorbed dims (latent space).
comptime KV_LORA_RANK = 512
comptime QK_ROPE_HEAD_DIM = 64
comptime QK_DEPTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
comptime V_DEPTH = KV_LORA_RANK  # 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1

# softmax scale = 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim) * mscale^2,
# which for DSv3.2 with mscale=1 is 1 / sqrt(128 + 64) = 1 / sqrt(192).
# Even though the kernel operates over 576 latent dims, scale uses the
# pre-absorption per-head depth (192).
comptime SOFTMAX_SCALE_BASE_DIM = 192


# ===-----------------------------------------------------------------------===#
# Helpers
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
    """Find a multiplier coprime to n for deterministic token selection."""
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


# ===-----------------------------------------------------------------------===#
# Host-side reference
# Per-query indices: each (b, s) row selects its own `topk` KV tokens.
# Score[h, k]    = Q[b,s,h,:] @ KV_sel[b,s,k,:]^T * scale
# O[b,s,h,d]     = softmax(Score)[h,:] @ V_sel[b,s,:,d]
# ===-----------------------------------------------------------------------===#


def host_reference[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    kv_sparse_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    topk: Int,
    qk_depth: Int,
    v_depth: Int,
    scale: Float32,
    valid_topk: Int = -1,
    sink_values: List[Float32] = [],
):
    """Compute reference MLA sparse prefill output on host.

    Q:               [B*seq_len, num_heads, qk_depth]
    KV_sparse:       [B*seq_len, topk, qk_depth]
    V_sparse_per_q = KV_sparse[bs, :, :v_depth]

    For each (b, s, h):
      score[k]    = Q[b,s,h,:] @ KV[b*seq_len+s,k,:]^T * scale
      prob[k]     = softmax(score)[k]
      O[b,s,h,d]  = sum_k prob[k] * V[b*seq_len+s,k,d]

    Implementation note: uses the kernel's log2-domain softmax path
    (`exp2(P*scale*log2e - mi)` and accumulate `li`) instead of the
    natural-exp form, so the host fp64 reference and the bf16 kernel
    follow the same algebraic chain.  This shaves off the
    natural-vs-log2 rounding drift from `max_err` so the diagnostic
    reflects only BF16 precision and MMA accumulator order — not the
    representation choice in the reference itself.
    """
    var scale_log2e = Float64(scale) * Float64(log2e)
    # `topk` is the kv_sparse stride (= indices_stride); `valid_topk` is
    # the per-query effective key count.  When `valid_topk == -1`, treat
    # all `topk` keys as valid (= dense full-topk).
    var n_valid = topk if valid_topk == -1 else valid_topk
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                var bs = b * seq_len + s
                var q_base = bs * num_heads * qk_depth + h * qk_depth

                var mi = Float64(min_or_neg_inf[.float32]())
                var s_buf = alloc[Float64](n_valid)

                for k in range(n_valid):
                    var kv_base = (bs * topk + k) * qk_depth
                    var dot = Float64(0)
                    for d in range(qk_depth):
                        dot += (
                            q_ptr[q_base + d].cast[.float64]()
                            * kv_sparse_ptr[kv_base + d].cast[.float64]()
                        )
                    # Match kernel's `cur_pi_max *= scale_log2e` step.
                    s_buf[k] = dot * scale_log2e
                    if s_buf[k] > mi:
                        mi = s_buf[k]

                # Softmax in log2 domain (matches kernel's exp2 path).
                # s_buf[k] = exp2(P_k * scale_log2e - mi); li accumulates
                # the sum; final softmax = s_buf[k] / li.
                var li = Float64(0)
                for k in range(n_valid):
                    s_buf[k] = exp2(s_buf[k] - mi)
                    li += s_buf[k]
                # Attention sink: a per-head virtual logit that contributes
                # `exp2(sink_h * log2e - mi)` to the softmax normalizer but
                # has no V (no numerator term). Matches the kernel epilogue's
                # `output_scale = 1 / (li + exp2(attn_sink_val - mi))` with
                # `attn_sink_val = sink_h * log2e`. Empty `sink_values` == no
                # sink (normalizer stays `li`).
                #
                # Guard the exp2 argument: production uses a finite-huge sink
                # (-1e38 => arg ~ -1.4e38), and host fp64 `exp2` on such an
                # extreme argument is undefined (it crashes / returns garbage
                # here), whereas the GPU `ex2.approx.f32` correctly underflows
                # to 0. Clamp: below -1000 the term is < 1e-301 (== 0 to any
                # precision that matters), so add exactly 0 to match the GPU.
                if len(sink_values) > 0:
                    var sink_arg = Float64(sink_values[h]) * Float64(log2e) - mi
                    if sink_arg > Float64(-1000.0):
                        li += exp2(sink_arg)
                for k in range(n_valid):
                    s_buf[k] = s_buf[k] / li

                # O = P @ V (V = first v_depth columns of KV)
                var o_base = bs * num_heads * v_depth + h * v_depth
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(n_valid):
                        var kv_base = (bs * topk + k) * qk_depth
                        acc += (
                            s_buf[k]
                            * kv_sparse_ptr[kv_base + d].cast[.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()

                s_buf.free()


# ===-----------------------------------------------------------------------===#
# Core test function
# ===-----------------------------------------------------------------------===#


def run_test_prefill_sparse[
    q_type: DType,
    num_heads: Int,
    topk: Int,
](
    name: StringLiteral,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    ctx: DeviceContext,
    *,
    valid_topk: Int = topk,
    topk_lengths_override: Int = -1,
    qkv_std: Float64 = 0.5,
    q_std: Float64 = -1.0,
    atol: Float64 = 0.02,
    sink_values: List[Float32] = [],
    num_layers: Int = NUM_LAYERS,
    layer_idx: Int = 0,
) raises:
    """Test the sparse MLA prefill kernel with a paged KV cache, per-query
    indices, and the absorbed DSv3.2 dims (qk_depth=576, v_depth=512).

    `num_layers` / `layer_idx` exercise the per-layer paged-cache addressing:
    with `num_layers > 1` and `layer_idx > 0` the K/V gather must fold the
    `num_layers` block stride into every physical row (via `get_tma_row`).
    Every layer is seeded with distinct random data and only `layer_idx`
    holds the real KV, so a gather that lands in the wrong layer decorrelates
    the output and trips the cosine / mean-error / tail gates below. Defaults
    (`num_layers=1`, `layer_idx=0`) keep the single-layer cases byte-identical.

    `sink_values` (optional) is a per-query-head attention sink: an empty
    list means no sink (kernel gets `None`); a list of length `num_heads`
    is copied to a device buffer of exactly `num_heads` Float32 and passed
    to the kernel, and the fp64 oracle adds the matching sink term. The
    exact-`num_heads` buffer means a broken padded-row sink guard reading
    `sink[num_heads..63]` is a real OOB (catchable under compute-sanitizer).

    `topk` here is the indices buffer stride (= the indexer's `index_topk`
    in DSv3.2 deployment).  `valid_topk` is the per-query effective count;
    when `valid_topk < topk`, positions `[valid_topk..topk)` in the
    indices buffer are filled with sentinel `0xFFFFFFFF` (= -1 in int32),
    and `topk_lengths[i]` is set to `valid_topk`.  The kernel's
    k-valid mask should poison those positions in softmax.

    `topk_lengths_override` (default -1 = disabled) DECOUPLES the value
    written to `topk_lengths[i]` from `valid_topk`.  In DSv3.2/GLM
    deployment the indexer broadcasts `index_topk` (e.g. 2048) into
    `topk_lengths` for EVERY query token, regardless of the token's real
    candidate count (early tokens have far fewer valid keys); the unused
    index slots carry the `0xFFFFFFFF` sentinel.  Setting
    `topk_lengths_override = topk` reproduces that regime: the kernel runs
    `ceildiv(topk, B_TOPK)` k-blocks (e.g. 32 at topk=2048), the tail
    blocks are entirely sentinel, and masking is driven SOLELY by the
    `idx >= 0` value check (the `abs_pos < top_k_length` term is vacuous
    when `top_k_length == topk`).  Real indices + the fp64 host reference
    still track `valid_topk`, so the reference expects only the first
    `valid_topk` keys to contribute.
    """
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " seq_len:",
        seq_len,
        " num_heads:",
        num_heads,
        " num_kv_tokens:",
        num_kv_tokens,
        " topk:",
        topk,
    )

    var scale = Float32(1.0) / sqrt(Float32(SOFTMAX_SCALE_BASE_DIM))
    comptime group = num_heads
    var total_q_tokens = batch_size * seq_len

    # -----------------------------------------------------------------------
    # KV cache parameters
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=QK_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

    var total_pages = batch_size * ceildiv(num_kv_tokens, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_kv_tokens, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        num_layers,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * num_layers
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Generate random BF16 KV data: [batch_size * num_kv_tokens, qk_depth]
    # -----------------------------------------------------------------------
    var kv_total = batch_size * num_kv_tokens * QK_DEPTH
    var kv_host = alloc[Scalar[q_type]](kv_total)
    randn[q_type](kv_host, kv_total, mean=0.0, standard_deviation=qkv_std)

    # -----------------------------------------------------------------------
    # Build shuffled page mapping (coprime permutation).
    # -----------------------------------------------------------------------
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = alloc[UInt32](lut_size)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_kv_tokens, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    # -----------------------------------------------------------------------
    # Fill KV cache blocks from random data with paged layout.
    # -----------------------------------------------------------------------
    var blocks_host = alloc[Scalar[q_type]](block_elems)
    # Multi-layer: seed every layer with distinct random data so a gather
    # into the wrong layer (the num_layers>1 addressing bug) reads garbage
    # and trips the cosine / tail gates. Layer `layer_idx` is overwritten
    # with the real KV below. Single-layer stays zero-initialized (NFC).
    if num_layers > 1:
        randn[q_type](
            blocks_host, block_elems, mean=0.0, standard_deviation=1.0
        )
    else:
        for i in range(block_elems):
            blocks_host[i] = Scalar[q_type](0)

    var page_stride_elems = (
        kv_dim2
        * num_layers
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    # Distance (in elements) between consecutive layers within one block.
    var layer_stride_elems = (
        PAGE_SIZE * kv_params.num_heads * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                block_id * page_stride_elems
                + layer_idx * layer_stride_elems
                + tok_in_page * QK_DEPTH
            )
            var src_base = (bi * num_kv_tokens + t) * QK_DEPTH
            for d in range(QK_DEPTH):
                blocks_host[base + d] = kv_host[src_base + d]

    # -----------------------------------------------------------------------
    # Q tensor: [total_q_tokens, num_heads, qk_depth]
    # -----------------------------------------------------------------------
    var q_elems = total_q_tokens * num_heads * QK_DEPTH
    # `q_std < 0` => use `qkv_std` for Q too. Decoupling Q-std from KV-std
    # lets a probe make scores PEAKED (large q_std => big Q.K spread =>
    # sharp softmax + frequent O-rescale) while keeping OUTPUT magnitude
    # SMALL (small kv_std => small V), so the scale-calibrated error gates
    # stay valid (bf16 ULP ~ output_magnitude * 2^-8).
    var q_std_eff = qkv_std if q_std < 0.0 else q_std
    var q_host = alloc[Scalar[q_type]](q_elems)
    randn[q_type](q_host, q_elems, mean=0.0, standard_deviation=q_std_eff)

    # -----------------------------------------------------------------------
    # Per-query token selection: each (b, s) row picks its own topk tokens.
    # We rotate the starting point by `s` so different queries see different
    # selections (catches per-query stride bugs in the kernel).
    # selected_tokens[bs * topk + i] = which physical-row in batch `b` to use
    # -----------------------------------------------------------------------
    var selected_tokens = alloc[Int](total_q_tokens * topk)
    var sel_mult = _coprime_multiplier(num_kv_tokens)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            var rotation = s % num_kv_tokens
            for i in range(topk):
                selected_tokens[bs * topk + i] = (
                    (rotation + i) * sel_mult + 1
                ) % num_kv_tokens

    # -----------------------------------------------------------------------
    # Build sparse KV ref: [total_q_tokens, topk, qk_depth]
    # Gather selected rows from the full KV buffer per query.
    # -----------------------------------------------------------------------
    var kv_sparse_size = total_q_tokens * topk * QK_DEPTH
    var kv_sparse = alloc[Scalar[q_type]](kv_sparse_size)

    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                var t = selected_tokens[bs * topk + i]
                var src_base = (bi * num_kv_tokens + t) * QK_DEPTH
                var dst_base = (bs * topk + i) * QK_DEPTH
                for d in range(QK_DEPTH):
                    kv_sparse[dst_base + d] = kv_host[src_base + d]

    # -----------------------------------------------------------------------
    # Compute host reference output
    # -----------------------------------------------------------------------
    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var ref_host = alloc[Scalar[q_type]](out_elems)
    # Host ref iterates over only the *valid* per-query keys (the kernel
    # masks the rest to -inf so they contribute zero).  `kv_sparse` rows
    # `[valid_topk..topk)` exist in memory but are never read by the
    # kernel (mask-out) or by host ref (loop bound = valid_topk).
    host_reference[q_type](
        q_host,
        kv_sparse,
        ref_host,
        batch_size,
        seq_len,
        num_heads,
        topk,
        QK_DEPTH,
        V_DEPTH,
        scale,
        valid_topk=valid_topk,
        sink_values=sink_values,
    )

    # -----------------------------------------------------------------------
    # Copy data to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_host = alloc[UInt32](batch_size)
    for bi in range(batch_size):
        cache_lengths_host[bi] = UInt32(num_kv_tokens)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_elems)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_elems)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build per-query gather4 indices.
    # indices[bs * topk + i] = physical_block * PAGE_SIZE + tok_in_page.
    # -----------------------------------------------------------------------
    var total_indices = total_q_tokens * topk
    var h_indices = alloc[UInt32](total_indices)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                if i < valid_topk:
                    var t = selected_tokens[bs * topk + i]
                    var page_idx = t // PAGE_SIZE
                    var tok_in_page = t % PAGE_SIZE
                    var block_id = Int(
                        lookup_table_host[bi * max_pages_per_batch + page_idx]
                    )
                    h_indices[bs * topk + i] = UInt32(
                        block_id * PAGE_SIZE + tok_in_page
                    )
                else:
                    # Padding sentinel: 0xFFFFFFFF cast to int32 inside
                    # the kernel = -1, which fails the `idx >= 0` check
                    # in the k-valid producer and gets masked out.
                    h_indices[bs * topk + i] = UInt32(0xFFFFFFFF)

    var indices_device = ctx.enqueue_create_buffer[.uint32](total_indices)
    ctx.enqueue_copy(indices_device, h_indices)

    # topk_lengths is per-query (not per-batch): the kernel reads
    # topk_lengths[seq_idx] for seq_idx in [0, total_q_tokens).
    # `topk_lengths_override` (when >= 0) decouples the reported length
    # from `valid_topk` so we can reproduce the production regime where
    # `index_topk` is broadcast constant across tokens while the real
    # candidate count (`valid_topk`) is smaller.
    var reported_topk_length = (
        valid_topk if topk_lengths_override < 0 else topk_lengths_override
    )
    var h_topk_lengths = alloc[UInt32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(reported_topk_length)

    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    ctx.enqueue_copy(topk_lengths_device, h_topk_lengths)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
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
        UInt32(seq_len),
        UInt32(num_kv_tokens),
    )

    var kv_cache = kv_collection.get_key_cache(layer_idx)

    # -----------------------------------------------------------------------
    # Build TileTensors for Q, output, indices, and topk_lengths.
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[QK_DEPTH])),
    )

    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var indices_tt = TileTensor(
        indices_device,
        row_major(total_indices),
    )

    var topk_lengths_tt = TileTensor(
        topk_lengths_device,
        row_major(total_q_tokens),
    )

    # -----------------------------------------------------------------------
    # Call mla_prefill_sparse
    # -----------------------------------------------------------------------
    print("  Launching mla_prefill_sparse...")

    # Mirror the dispatch policy in mla_prefill.mojo: head128 → 2SM
    # (cta_group=2, B_TOPK=128); head64 → single-CTA WS (cta_group=1,
    # B_TOPK=64).
    comptime cta_group = 2 if num_heads == 128 else 1
    comptime b_topk = 128 if num_heads == 128 else 64
    comptime config = MLASparseConfig[
        q_type, b_topk_=b_topk, cta_group_=cta_group
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=topk,
        group=num_heads,
    )

    # Optional attention sink (one Float32 per query head). Empty list ->
    # `None` (kernel skips the `exp2(sink - mi)` softmax term). Non-empty ->
    # a device buffer of EXACTLY `num_heads` Float32 so a broken sub-64
    # padded-row guard reading sink[num_heads..63] is a real OOB.
    var attn_sink_ptr = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](None)
    var sink_len = len(sink_values) if len(sink_values) > 0 else 1
    var sink_device = ctx.enqueue_create_buffer[.float32](sink_len)
    if len(sink_values) > 0:
        var sink_host = alloc[Float32](len(sink_values))
        for i in range(len(sink_values)):
            sink_host[i] = sink_values[i]
        ctx.enqueue_copy(sink_device, sink_host)
        ctx.synchronize()
        sink_host.free()
        attn_sink_ptr = Optional[UnsafePointer[Float32, ImmutAnyOrigin]](
            sink_device.unsafe_ptr().bitcast[Float32]().as_unsafe_any_origin()
        )

    mla_prefill_sparse[
        config=config,
        group=group,
        q_depth=QK_DEPTH,
    ](
        out_tt,
        q_tt,
        kv_cache,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        scale,
        Int32(topk),
        ctx,
    )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output against host reference
    # -----------------------------------------------------------------------
    var out_host = alloc[Scalar[q_type]](out_elems)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # atol sits just above the measured bf16 rounding floor: max_err
    # <= 0.002 on every non-sink shape (sharp/diffuse, ragged/full); the
    # -inf-sink cases sit ~10x higher (<= 0.0176, stable — the RNG is
    # default-seeded so draws are deterministic). Earlier revisions used
    # 0.18-0.45 "noise floor" values that were in fact calibrated to real
    # kernel defects (a dropped key-half and a mis-strided K read) —
    # never widen this to match an observed baseline without a
    # first-principles error budget.
    var max_err = Float64(0)
    var max_err_low_d = Float64(0)
    var max_err_high_d = Float64(0)
    var max_err_low_h = Float64(0)
    var max_err_high_h = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
    var nonzero_low_depth = 0
    var nonzero_high_depth = 0
    var nonzero_low_head = 0
    var nonzero_high_head = 0
    var total_checked = 0
    # Non-finite counters + finite-only secondary-metric accumulators. The
    # pure max-abs check below silently ignores NaN (abs(NaN-ref)=NaN and
    # NaN>atol is False), so these are what actually catch a NaN/garbage
    # kernel.
    var nan_actual = 0
    var inf_actual = 0
    var nan_ref = 0
    var inf_ref = 0
    var sum_abs_err = Float64(0)
    var dot_ar = Float64(0)
    var norm_a = Float64(0)
    var norm_r = Float64(0)
    var n_finite = 0
    var n_err_gt_1em2 = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[.float64]()
                    var actual_val = out_host[idx].cast[.float64]()
                    var err = abs(actual_val - ref_val)
                    if actual_val != actual_val:
                        nan_actual += 1
                    elif abs(actual_val) > 1.0e300:
                        inf_actual += 1
                    elif ref_val == ref_val and abs(ref_val) <= 1.0e300:
                        sum_abs_err += err
                        if err > 1.0e-2:
                            n_err_gt_1em2 += 1
                        dot_ar += actual_val * ref_val
                        norm_a += actual_val * actual_val
                        norm_r += ref_val * ref_val
                        n_finite += 1
                    if ref_val != ref_val:
                        nan_ref += 1
                    elif abs(ref_val) > 1.0e300:
                        inf_ref += 1
                    if err > max_err:
                        max_err = err
                    if d < 256:
                        if err > max_err_low_d:
                            max_err_low_d = err
                    else:
                        if err > max_err_high_d:
                            max_err_high_d = err
                    if h < 64:
                        if err > max_err_low_h:
                            max_err_low_h = err
                    else:
                        if err > max_err_high_h:
                            max_err_high_h = err
                    if abs(actual_val) > max_actual:
                        max_actual = abs(actual_val)
                    if abs(actual_val) > 1e-6:
                        num_nonzero += 1
                        if d < 256:
                            nonzero_low_depth += 1
                        else:
                            nonzero_high_depth += 1
                        if h < 64:
                            nonzero_low_head += 1
                        else:
                            nonzero_high_head += 1
                    total_checked += 1
    print(
        "  DIAG: max_err=",
        max_err,
        " max_abs_actual=",
        max_actual,
        " num_nonzero=",
        num_nonzero,
        "/",
        total_checked,
    )
    print(
        "  max_err by depth: low(0..255)=",
        max_err_low_d,
        " high(256..511)=",
        max_err_high_d,
    )
    print(
        "  max_err by head: low(0..63)=",
        max_err_low_h,
        " high(64..127)=",
        max_err_high_h,
    )
    print("  Sample out vs ref for seq=0:")
    for h in [0, 32, 64, 96]:
        if h >= num_heads:
            continue
        var base = h * V_DEPTH
        for d in [0, 64, 128, 192]:
            var idx = base + d
            print(
                "    h=",
                h,
                " d=",
                d,
                " out=",
                out_host[idx].cast[.float64](),
                " ref=",
                ref_host[idx].cast[.float64](),
            )

    # Fail hard on ANY non-finite output/reference. This is the check the
    # pure max-abs test lacked: abs(NaN - ref) = NaN and `NaN > atol` is
    # False, so a NaN-producing kernel used to report PASSED.
    if nan_actual != 0 or inf_actual != 0 or nan_ref != 0 or inf_ref != 0:
        raise Error(
            "non-finite values detected: nan_actual="
            + String(nan_actual)
            + " inf_actual="
            + String(inf_actual)
            + " nan_ref="
            + String(nan_ref)
            + " inf_ref="
            + String(inf_ref)
        )

    # Tight secondary bounds the outlier-driven max-abs cannot mask:
    #   mean_abs_err catches a broadly-wrong / mis-scaled output whose few
    #     large elements still fit under `atol`;
    #   cosine catches a decorrelated / all-wrong output.
    # First-principles error budget: the fp64 oracle consumes the same bf16
    # inputs as the kernel, so the honest disagreement floor is bf16
    # rounding + MMA-accumulation order — well under 1% relative (cosine
    # measured 0.999997 on every head count). Do NOT loosen these gates to
    # match an observed baseline: a low cosine here means a real kernel
    # defect (a hardcoded S-store key-half offset once hid at exactly
    # cosine ~= sqrt(1/2), and a mis-strided cta_group=2 K read at ~0.91,
    # both behind gates fitted to "measured" values).
    comptime COS_MIN: Float64 = 0.99
    comptime MEAN_ERR_MAX: Float64 = 0.01
    var mean_abs_err = sum_abs_err / Float64(max(n_finite, 1))
    if mean_abs_err > MEAN_ERR_MAX:
        raise Error(
            "mean_abs_err exceeded tolerance: "
            + String(mean_abs_err)
            + " > "
            + String(MEAN_ERR_MAX)
        )
    var cos_denom = sqrt(norm_a) * sqrt(norm_r)
    var cosine: Float64
    if norm_a < 1e-12 and norm_r < 1e-12:
        # Both ~zero (e.g. an all-invalid query row): trivially aligned.
        cosine = 1.0
    elif cos_denom < 1e-12:
        # One side ~zero but not the other: a real mismatch.
        cosine = 0.0
    else:
        cosine = dot_ar / cos_denom
    if cosine < COS_MIN:
        raise Error(
            "cosine similarity below tolerance: "
            + String(cosine)
            + " < "
            + String(COS_MIN)
        )

    # Real assertion — without this the test would print "PASSED" even
    # if the kernel regressed.
    # Error-tail gate: the post-fix error distribution is a cliff — every
    # non-sink case has ZERO elements with |err| > 1e-2, and the -inf-sink
    # cases have <= 0.11%. A growing tail means a partial defect (a subset
    # of rows/blocks wrong) that max_err, mean_abs_err, and the global
    # cosine can each individually miss.
    var tail_frac = Float64(n_err_gt_1em2) / Float64(max(n_finite, 1))
    if tail_frac >= 0.0015:
        raise Error(
            "error-tail exceeded bound: "
            + String(n_err_gt_1em2)
            + " elements (|err| > 1e-2) = "
            + String(tail_frac * 100)
            + "% >= 0.15%"
        )

    if max_err > atol:
        raise Error(
            "max_err exceeded tolerance: "
            + String(max_err)
            + " > atol "
            + String(atol)
        )
    print(
        "  PASSED: max_err=",
        max_err,
        " mean_abs_err=",
        mean_abs_err,
        " cosine=",
        cosine,
        " n_err_gt_1e-2=",
        n_err_gt_1em2,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = indices_device
    _ = topk_lengths_device
    _ = sink_device

    blocks_host.free()
    kv_host.free()
    lookup_table_host.free()
    cache_lengths_host.free()
    q_host.free()
    kv_sparse.free()
    selected_tokens.free()
    ref_host.free()
    out_host.free()
    h_indices.free()
    h_topk_lengths.free()


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # ===============================================================
            # Production DSv3.2/GLM regime: constant broadcast topk_lengths.
            # ---------------------------------------------------------------
            # The indexer broadcasts index_topk (=2048) into topk_lengths for
            # EVERY query token (sparse_mla.py:897), DECOUPLED from the token's
            # real candidate count; the unused index slots carry the -1
            # (0xFFFFFFFF) sentinel. So the kernel runs ceildiv(2048, B_TOPK=64)
            # = 32 k-blocks for every token, masking is driven SOLELY by the
            # `idx >= 0` value check (the `abs_pos < top_k_length` term is
            # vacuous at 2048), and early tokens have long all-sentinel tails
            # that the K loader skips via `skip_tma`.  Every OTHER test couples
            # topk_lengths[i] = valid_topk, so this decoupled 32-block regime
            # was previously uncovered.
            #
            # These cases also run at q_std=8 (PEAKED softmax): production
            # RoPE'd data produces a sharply-peaked softmax where one key
            # dominates and the running-max jumps between blocks, so the
            # online-softmax O-rescale (`should_scale_o`) fires on nearly every
            # block. The pre-existing tests all used std=0.5 (diffuse: weights
            # ~1/topk, running-max barely moves, O-rescale almost never fires),
            # leaving the peaked deep-fold path effectively untested. kv_std
            # stays 0.5 so the OUTPUT magnitude stays small and the
            # scale-calibrated error gates remain valid.
            #
            # The attention sink is the EXACT production value: -1.0e38
            # broadcast to every head (sparse_mla.py:905), i.e. a numerically
            # inert "disabled" sink. The kernel's ex2.approx.f32 underflows
            # exp2(-1.4e38 - mi) to 0 (verified on B200), matching the fp64
            # host oracle's clamped sink term.
            var sink_prod_h8 = List[Float32]()
            for _ in range(8):
                sink_prod_h8.append(Float32(-1.0e38))
            var sink_prod_h64 = List[Float32]()
            for _ in range(64):
                sink_prod_h64.append(Float32(-1.0e38))

            # h8 (GLM TP8) late-token: 32 fully-valid blocks, peaked + sink.
            run_test_prefill_sparse[.bfloat16, 8, 2048](
                "b1_s8_h8_len2048_valid2048_peaked_sink",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
                q_std=8.0,
                sink_values=sink_prod_h8,
            )
            # h8 early-token: 2 valid blocks + 30 all-sentinel skip_tma blocks.
            run_test_prefill_sparse[.bfloat16, 8, 2048](
                "b1_s8_h8_len2048_valid96_peaked_sink",
                1,
                8,
                256,
                ctx,
                valid_topk=96,
                topk_lengths_override=2048,
                q_std=8.0,
                sink_values=sink_prod_h8,
            )
            # h8 all-invalid: all 32 blocks sentinel (skip_tma throughout) =>
            # O=0 via the have_valid_indices vote at constant length 2048.
            run_test_prefill_sparse[.bfloat16, 8, 2048](
                "b1_s8_h8_len2048_valid0",
                1,
                8,
                256,
                ctx,
                valid_topk=0,
                topk_lengths_override=2048,
            )
            # h8 diffuse deep: 32-block fold with a flat softmax (complements
            # the peaked cases above; O-rescale rarely fires here).
            run_test_prefill_sparse[.bfloat16, 8, 2048](
                "b1_s8_h8_len2048_valid2048_diffuse",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            # h8 large cache: valid keys scattered across an 8k-row cache =>
            # large physical gather4 indices (128 pages), peaked.
            run_test_prefill_sparse[.bfloat16, 8, 2048](
                "b1_s8_h8_len2048_bigcache8k_peaked",
                1,
                8,
                8192,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
                q_std=8.0,
            )
            # h64 (GLM unsharded) late-token: 32 fully-valid blocks, peaked +
            # sink — confirms the landed 64-head path shares the h8 result.
            run_test_prefill_sparse[.bfloat16, 64, 2048](
                "b1_s8_h64_len2048_valid2048_peaked_sink",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
                q_std=8.0,
                sink_values=sink_prod_h64,
            )
            # h64 early-token: 2 valid blocks + 30 all-sentinel skip_tma tail.
            run_test_prefill_sparse[.bfloat16, 64, 2048](
                "b1_s8_h64_len2048_valid96_peaked_sink",
                1,
                8,
                256,
                ctx,
                valid_topk=96,
                topk_lengths_override=2048,
                q_std=8.0,
                sink_values=sink_prod_h64,
            )

            # `topk` (= indices_stride) must be a multiple of B_TOPK=128
            # since the kernel reads B_TOPK indices per k-block per
            # query unconditionally (matches phase1.cuh:628's
            # `KU_ASSERT(params.topk % B_TOPK == 0)`).  Per-query
            # `valid_topk < topk` is now supported via the k-valid
            # mask (see padded test cases below).

            # Single k-block: topk == B_TOPK.
            run_test_prefill_sparse[.bfloat16, 128, 128](
                "b1_s32_h128_kv512_topk128",
                1,
                32,
                512,
                ctx,
            )

            # Multi-batch, single k-block.
            run_test_prefill_sparse[.bfloat16, 128, 128](
                "b4_s16_h128_kv256_topk128",
                4,
                16,
                256,
                ctx,
            )

            # Multi-block: topk=256 = 2 * B_TOPK exercises the cross-block
            # online-softmax state (mi/li updates between k iters).
            run_test_prefill_sparse[.bfloat16, 128, 256](
                "b1_s32_h128_kv512_topk256",
                1,
                32,
                512,
                ctx,
            )

            # Production-flavored shape, scaled down so the Float64 host
            # reference completes in a tractable amount of time. Real
            # DSv3.2 uses topk=2048; the kernel paths exercised here
            # (multi-block, multi-warpgroup pipeline, full epilogue) are
            # the same.
            run_test_prefill_sparse[.bfloat16, 128, 256](
                "b1_s64_h128_kv1024_topk256_prodlike",
                1,
                64,
                1024,
                ctx,
            )

            # Padded-index cases: `valid_topk < topk`.  The last
            # `topk - valid_topk` indices per query are sentinel
            # 0xFFFFFFFF; the k-valid mask must drop them from softmax.
            #
            # Single-block padded: B_TOPK=128 with the last 64 indices
            # masked out — exercises the producer's `abs_pos <
            # top_k_length` check on positions inside one k-block.
            run_test_prefill_sparse[.bfloat16, 128, 128](
                "b1_s32_h128_kv512_topk128_valid64",
                1,
                32,
                512,
                ctx,
                valid_topk=64,
            )

            # Multi-block padded: indices_stride=256, second k-block
            # entirely padded — exercises the all-invalid k-block
            # fast-path in load_k's `skip_tma` and the producer's
            # whole-block mask=0 case.
            run_test_prefill_sparse[.bfloat16, 128, 256](
                "b1_s32_h128_kv512_topk256_valid128",
                1,
                32,
                512,
                ctx,
                valid_topk=128,
            )

            # Multi-block padded with partial second block:
            # valid_topk=192 covers the full first block + 64 keys of
            # the second block — the mask must fire mid-block.
            run_test_prefill_sparse[.bfloat16, 128, 256](
                "b1_s32_h128_kv512_topk256_valid192",
                1,
                32,
                512,
                ctx,
                valid_topk=192,
            )

            # ---------------------------------------------------------------
            # head64 path (GLM): cta_group=1, B_TOPK=64, single-CTA
            # warp-specialized packed-TMEM MMA.  `topk` must be a multiple
            # of B_TOPK=64.  Matrix: {exact 1-block, exact 2-block, ragged
            # < 64, large 16+ blocks} x num_q_rows {1, small, large}.
            # ---------------------------------------------------------------

            # Exact 1 k-block, num_q_rows=1 (smallest grid).
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b1_s1_h64_kv256_topk64",
                1,
                1,
                256,
                ctx,
            )

            # Exact 1 k-block, small num_q_rows.
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b1_s32_h64_kv256_topk64",
                1,
                32,
                256,
                ctx,
            )

            # Multi-batch, exact 1 k-block (num_q_rows = 4*16 = 64).
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b4_s16_h64_kv256_topk64",
                4,
                16,
                256,
                ctx,
            )

            # Exact 2 k-blocks (cross-block online-softmax state).
            run_test_prefill_sparse[.bfloat16, 64, 128](
                "b1_s32_h64_kv512_topk128",
                1,
                32,
                512,
                ctx,
            )

            # Ragged tail < B_TOPK: topk=64, valid_topk=40 — k-valid mask
            # poisons positions [40..64) inside the single block.
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b1_s32_h64_kv256_topk64_valid40",
                1,
                32,
                256,
                ctx,
                valid_topk=40,
            )

            # Multi-block ragged: 2 blocks, valid_topk=96 fires mid second
            # block.
            run_test_prefill_sparse[.bfloat16, 64, 128](
                "b1_s32_h64_kv512_topk128_valid96",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )

            # Prime valid_topk (37) in a single block: the mask boundary
            # lands at a non-aligned offset inside [0, 64).
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b1_s32_h64_kv256_topk64_valid37",
                1,
                32,
                256,
                ctx,
                valid_topk=37,
                atol=0.45,
            )

            # Prime valid_topk (97) across 2 blocks: full first block plus a
            # non-aligned 33-key tail in the second.
            run_test_prefill_sparse[.bfloat16, 64, 128](
                "b1_s32_h64_kv512_topk128_valid97",
                1,
                32,
                512,
                ctx,
                valid_topk=97,
                atol=0.32,
            )

            # Prime num_q_rows (13): exercises a partial query tile.
            run_test_prefill_sparse[.bfloat16, 64, 64](
                "b1_s13_h64_kv256_topk64",
                1,
                13,
                256,
                ctx,
                atol=0.45,
            )

            # Large: 16 k-blocks (topk=1024) exercises the deep k loop.
            run_test_prefill_sparse[.bfloat16, 64, 1024](
                "b1_s8_h64_kv1024_topk1024",
                1,
                8,
                1024,
                ctx,
            )

            # ---------------------------------------------------------------
            # Sub-64-head paths (GLM 5.2, tensor-parallel sharded): 64 heads
            # / TP{8,4,2} = {8,16,32} heads per device.  These reuse the same
            # single-CTA WS datapath (cta_group=1, B_TOPK=64) as head64 but
            # run a 64-row MMA M-tile padded from num_q_heads real rows; only
            # the real rows are loaded (Q) and stored (O), and the padded Q
            # rows are zeroed.  `topk` must be a multiple of B_TOPK=64.
            # Coverage matrix per head count: {exact 1-block, multi-block,
            # ragged < 64, ragged mid-block} x num_q_rows {1, small, multi}.
            # ---------------------------------------------------------------

            # --- 8 heads (GLM TP=8) ---
            # Exact 1 k-block, num_q_rows=1 (smallest grid).
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s1_h8_kv256_topk64",
                1,
                1,
                256,
                ctx,
            )

            # Exact 1 k-block, small num_q_rows.
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64",
                1,
                32,
                256,
                ctx,
            )

            # Multi-batch, exact 1 k-block (num_q_rows = 4*16 = 64).
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b4_s16_h8_kv256_topk64",
                4,
                16,
                256,
                ctx,
            )

            # Exact 2 k-blocks (cross-block online-softmax state).
            run_test_prefill_sparse[.bfloat16, 8, 128](
                "b1_s32_h8_kv512_topk128",
                1,
                32,
                512,
                ctx,
            )

            # Multi-block: 4 k-blocks (topk=256).
            run_test_prefill_sparse[.bfloat16, 8, 256](
                "b1_s32_h8_kv1024_topk256",
                1,
                32,
                1024,
                ctx,
            )

            # Ragged tail < B_TOPK: topk=64, valid_topk=40 — k-valid mask
            # poisons positions [40..64) inside the single block.
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64_valid40",
                1,
                32,
                256,
                ctx,
                valid_topk=40,
            )

            # Multi-block ragged: 2 blocks, valid_topk=96 fires mid second
            # block.
            run_test_prefill_sparse[.bfloat16, 8, 128](
                "b1_s32_h8_kv512_topk128_valid96",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )

            # --- 16 heads (GLM TP=4) ---
            # Exact 1 k-block, small num_q_rows.
            run_test_prefill_sparse[.bfloat16, 16, 64](
                "b1_s32_h16_kv256_topk64",
                1,
                32,
                256,
                ctx,
            )

            # Multi-block: 4 k-blocks (topk=256).
            run_test_prefill_sparse[.bfloat16, 16, 256](
                "b1_s32_h16_kv1024_topk256",
                1,
                32,
                1024,
                ctx,
            )

            # Multi-block ragged: 2 blocks, valid_topk=96.
            run_test_prefill_sparse[.bfloat16, 16, 128](
                "b1_s32_h16_kv512_topk128_valid96",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )

            # --- 8 heads: topk boundary, grid extremes, all-invalid, deep,
            #     and attention-sink corner cases ---

            # topk-1 boundary: valid_topk = topk-1 (63) inside one B_TOPK=64
            # block — the k-valid mask poisons exactly the last position.
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64_valid63",
                1,
                32,
                256,
                ctx,
                valid_topk=63,
            )

            # (1, 64): single batch, 64 query rows (grid = 64 CTAs).
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s64_h8_kv256_topk64",
                1,
                64,
                256,
                ctx,
            )

            # All-invalid query: valid_topk=0 — every index is a sentinel, so
            # the epilogue's have_valid_indices vote is false and the real row
            # resets to real_mi=-inf -> O=0. Confirms the zeroed padded rows
            # don't flip the vote for a fully-invalid real row at nqh<64.
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64_valid0",
                1,
                32,
                256,
                ctx,
                valid_topk=0,
            )

            # Deep: 16 k-blocks (topk=1024) at 8 heads — the deep
            # online-softmax k loop under the sub-64 padded M-tile.
            run_test_prefill_sparse[.bfloat16, 8, 1024](
                "b1_s8_h8_kv1024_topk1024",
                1,
                8,
                1024,
                ctx,
            )

            # Attention sink (finite): per-head sink logit added to the
            # softmax normalizer (down-weights all real tokens). The device
            # sink buffer is exactly 8 Float32, so a broken sub-64 padded-row
            # guard would OOB-read sink[8..63].
            var sink_bf16_h8_finite: List[Float32] = [
                0.5,
                1.0,
                1.5,
                2.0,
                0.5,
                1.0,
                1.5,
                2.0,
            ]
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64_sink_finite",
                1,
                32,
                256,
                ctx,
                sink_values=sink_bf16_h8_finite,
            )

            # Attention sink (-inf): numerically inert (exp2(-inf)=0 -> same as
            # no sink), so the oracle still matches. Its purpose is proving the
            # padded-row sink guard reads no OOB: run under compute-sanitizer
            # to confirm no read of sink[8..63].
            var sink_bf16_h8_neginf = List[Float32]()
            for _ in range(8):
                sink_bf16_h8_neginf.append(Float32(min_or_neg_inf[.float32]()))
            run_test_prefill_sparse[.bfloat16, 8, 64](
                "b1_s32_h8_kv256_topk64_sink_neginf",
                1,
                32,
                256,
                ctx,
                sink_values=sink_bf16_h8_neginf,
            )

            # --- 32 heads (GLM TP=2) ---
            # Exact 1 k-block.
            run_test_prefill_sparse[.bfloat16, 32, 64](
                "b1_s32_h32_kv256_topk64",
                1,
                32,
                256,
                ctx,
            )

            # Multi-block: 4 k-blocks (topk=256).
            run_test_prefill_sparse[.bfloat16, 32, 256](
                "b1_s32_h32_kv1024_topk256",
                1,
                32,
                1024,
                ctx,
            )

            # Multi-block ragged: 2 blocks, valid_topk=96.
            run_test_prefill_sparse[.bfloat16, 32, 128](
                "b1_s32_h32_kv512_topk128_valid96",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )

            # --- 24 heads (Part C: non-power-of-2 multiple of 8) ---
            # Proves the sub-64 mechanism generalizes to ANY multiple of 8 in
            # (0, 64], not just {8, 16, 32}. 24 % 8 == 0 satisfies the
            # SWIZZLE_128B 8-row core-matrix constraint the assert enforces.
            run_test_prefill_sparse[.bfloat16, 24, 64](
                "b1_s32_h24_kv256_topk64",
                1,
                32,
                256,
                ctx,
            )

            # ===============================================================
            # Multi-layer paged-cache addressing (num_layers>1, layer_idx>0).
            # ---------------------------------------------------------------
            # A paged MLA KV cache is a 6D [blocks, 1, num_layers, page,
            # heads, dim] tensor; real serving has ~61 layers. The K/V gather
            # must fold the `num_layers` block stride into every physical row
            # (via `get_tma_row`). Every layer here is seeded with distinct
            # data and only `layer_idx` holds the real KV, so a gather that
            # drops the num_layers factor lands in the wrong layer and the
            # cosine / tail / mean-error gates fire. num_kv_tokens=512 spans 4
            # physical blocks so blocks > 0 (which the pre-fix path mis-reads
            # into layer 0) are exercised. Single-layer cases above stay NFC.
            #
            # `layer_idx=1` is the guard (was RED before the get_tma_row fix
            # in load_k / v_tma_gather4_load / load_{k,v}_fp8_tma);
            # `layer_idx=0` at num_layers=2 is the control that the
            # multi-layer path is still correct at the base layer.
            run_test_prefill_sparse[.bfloat16, 8, 128](
                "b1_s32_h8_kv512_topk128_L2_layer1",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
            run_test_prefill_sparse[.bfloat16, 8, 128](
                "b1_s32_h8_kv512_topk128_L2_layer0_control",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=0,
            )
            run_test_prefill_sparse[.bfloat16, 64, 128](
                "b1_s32_h64_kv512_topk128_L2_layer1",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
        else:
            pass
