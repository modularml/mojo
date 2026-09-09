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

"""Numerical E2E test for mla_prefill_sparse_fp8 kernel (FP8 KV cache).

This kernel performs sparse MLA prefill attention over a subset of KV tokens
selected by an external scoring phase (the indexer). DSv3.2 only uses the
"absorbed" / latent shape: qk_depth = kv_lora_rank(512) + qk_rope_head_dim(64)
= 576, v_depth = 512.

Memory layout:
  - KV:     Paged KV cache (PagedKVCacheCollection) with FP8 (float8_e4m3fn) data.
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
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_kv_fp8 import (
    mla_prefill_sparse_fp8,
)
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
# FP8 test helpers
# ===-----------------------------------------------------------------------===#

# FP8 e4m3fn max representable magnitude.
comptime FP8_E4M3_MAX = Float32(448.0)


def run_test_prefill_sparse_fp8[
    num_heads: Int,
    topk: Int,
    scale_block_size: Int,
](
    name: StringLiteral,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    ctx: DeviceContext,
    *,
    valid_topk: Int = topk,
    atol: Float64 = 0.02,
    sink_values: List[Float32] = [],
    num_layers: Int = NUM_LAYERS,
    layer_idx: Int = 0,
    topk_lengths_override: Int = -1,
) raises:
    """FP8 KV-cache variant of run_test_prefill_sparse.

    `topk_lengths_override` (default -1 = disabled) decouples the reported
    per-token length from `valid_topk`, exactly like the BF16 harness. The
    DSv3.2 indexer broadcasts a constant `index_topk` (e.g. 2048) into
    `topk_lengths` regardless of the real valid count, so
    `topk_lengths_override = topk` reproduces that regime (long all-sentinel
    tails at low `valid_topk`).

    `num_layers` / `layer_idx` mirror the BF16 harness and guard the same
    per-layer paged-cache addressing for the FP8 K/V (data) gather. The FP8
    per-row scales are indexed in `get_tma_row` space (row =
    `block_id * num_layers * PAGE_SIZE + tok_in_page`), matching the kernel's
    `scales_ptr[get_tma_row(raw) * scales_per_token]`.

    `sink_values` threads an optional per-head attention sink exactly like
    the BF16 harness: empty -> `None`; length-`num_heads` -> a device buffer
    of exactly `num_heads` Float32 plus the matching fp64-oracle sink term.

    The KV cache is quantized to FP8 with tensorwise scaling (one Float32
    scale per KV token when ``scale_block_size == QK_DEPTH``).  The host
    reference is computed from the dequantized BF16 values so both kernel
    and reference see the same quantization noise.

    Tolerance is 0.45 — looser than the BF16 test's 0.18 to account for
    FP8 quantization error on top of BF16 rounding noise (padded cases
    with sentinel indices can reach ~0.40).
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
        " scale_block_size:",
        scale_block_size,
    )

    var scale = Float32(1.0) / sqrt(Float32(SOFTMAX_SCALE_BASE_DIM))
    comptime group = num_heads
    var total_q_tokens = batch_size * seq_len

    # -----------------------------------------------------------------------
    # KV cache parameters — same dims as BF16, but FP8 dtype.
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=QK_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

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

    # -----------------------------------------------------------------------
    # Page lookup table (same coprime shuffle as BF16 test).
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
    # Generate random BF16 KV data, quantize to FP8, record scales.
    # scales_host is indexed in `get_tma_row` space:
    #   phys_row = block_id * num_layers * PAGE_SIZE + tok_in_page
    # matching the kernel's `scales_ptr[get_tma_row(raw) * scales_per_token]`.
    # -----------------------------------------------------------------------
    var kv_total = batch_size * num_kv_tokens * QK_DEPTH
    var kv_bf16_host = alloc[BFloat16](kv_total)
    randn[.bfloat16](kv_bf16_host, kv_total, mean=0.0, standard_deviation=0.5)

    # scale_block_size == 0 => no-scale mode (unit scale): FP8 latents are
    # quantized/dequantized with s == 1.0 (no per-block scaling), matching the
    # scale-less MLA latent cache the kernel reads at scale_block_size == 0.
    # One "block" of width QK_DEPTH covers the whole token; scales_host holds a
    # single unused-by-kernel 1.0 per token (the kernel skips the read).
    comptime no_scale = scale_block_size == 0
    comptime eff_block = QK_DEPTH if no_scale else scale_block_size
    comptime scales_per_token = ceildiv(
        QK_DEPTH, eff_block
    )  # 1 for tensorwise/no-scale
    var total_phys_rows = total_pages * num_layers * PAGE_SIZE
    var scales_host = alloc[Float32](total_phys_rows * scales_per_token)
    for i in range(total_phys_rows * scales_per_token):
        scales_host[i] = Float32(1.0)

    var blocks_fp8_host = alloc[Float8_e4m3fn](block_elems)
    # Multi-layer: seed every layer with a distinct nonzero pattern so a
    # gather into the wrong layer (the num_layers>1 addressing bug) reads it
    # and decorrelates the output. Layer `layer_idx` is overwritten with the
    # real quantized KV below. Single-layer stays zero-initialized (NFC).
    if num_layers > 1:
        for i in range(block_elems):
            blocks_fp8_host[i] = Float8_e4m3fn(
                Float32(Int(i % 17) - 8) * Float32(0.5)
            )
    else:
        for i in range(block_elems):
            blocks_fp8_host[i] = Float8_e4m3fn(0)

    # dequantized BF16 values for host reference.
    var kv_dequant_host = alloc[BFloat16](kv_total)

    for bi in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var phys_row = block_id * num_layers * PAGE_SIZE + tok_in_page
            var src_base = (bi * num_kv_tokens + t) * QK_DEPTH

            # Compute block scales and quantize.
            comptime for blk in range(scales_per_token):
                comptime blk_start = blk * eff_block
                comptime blk_end = min(blk_start + eff_block, QK_DEPTH)
                var abs_max = Float32(0.0)
                for d in range(blk_start, blk_end):
                    var v = abs(kv_bf16_host[src_base + d].cast[.float32]())
                    if v > abs_max:
                        abs_max = v
                # No-scale mode forces s == 1.0 (unit scale, no dequant).
                var s = Float32(1.0) if no_scale else (
                    abs_max / FP8_E4M3_MAX if abs_max
                    > Float32(0.0) else Float32(1.0)
                )
                scales_host[phys_row * scales_per_token + blk] = s

                var base = (
                    block_id * page_stride_elems
                    + layer_idx * layer_stride_elems
                    + tok_in_page * QK_DEPTH
                )
                for d in range(blk_start, blk_end):
                    var q_val = kv_bf16_host[src_base + d].cast[.float32]() / s
                    # Clamp to FP8 range to prevent overflow → NaN (FP8 e4m3fn
                    # overflows to NaN, not infinity, for values > 448).
                    var q_val_clamped = min(
                        max(q_val, -FP8_E4M3_MAX), FP8_E4M3_MAX
                    )
                    var fp8_val = q_val_clamped.cast[.float8_e4m3fn]()
                    blocks_fp8_host[base + d] = fp8_val
                    kv_dequant_host[src_base + d] = (
                        fp8_val.cast[.float32]() * s
                    ).cast[.bfloat16]()

    # -----------------------------------------------------------------------
    # Q tensor: [total_q_tokens, num_heads, qk_depth]  (BF16)
    # -----------------------------------------------------------------------
    var q_elems = total_q_tokens * num_heads * QK_DEPTH
    var q_host = alloc[BFloat16](q_elems)
    randn[.bfloat16](q_host, q_elems, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Per-query token selection (same coprime rotation).
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
    # Sparse KV ref from DEQUANTIZED values.
    # -----------------------------------------------------------------------
    var kv_sparse_size = total_q_tokens * topk * QK_DEPTH
    var kv_sparse = alloc[BFloat16](kv_sparse_size)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                var t = selected_tokens[bs * topk + i]
                var src_base = (bi * num_kv_tokens + t) * QK_DEPTH
                var dst_base = (bs * topk + i) * QK_DEPTH
                for d in range(QK_DEPTH):
                    kv_sparse[dst_base + d] = kv_dequant_host[src_base + d]

    # -----------------------------------------------------------------------
    # Host reference output.
    # -----------------------------------------------------------------------
    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var ref_host = alloc[BFloat16](out_elems)
    host_reference[.bfloat16](
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
    # Copy data to device.
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[.float8_e4m3fn](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_fp8_host)

    var cache_lengths_host = alloc[UInt32](batch_size)
    for bi in range(batch_size):
        cache_lengths_host[bi] = UInt32(num_kv_tokens)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[.bfloat16](q_elems)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[.bfloat16](out_elems)

    var scales_device = ctx.enqueue_create_buffer[.float32](
        total_phys_rows * scales_per_token
    )
    ctx.enqueue_copy(scales_device, scales_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build per-query gather4 indices.
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
                    # Sentinel: cast to int32 = -1, which fails the
                    # `raw >= 0` check in load_k_scales_to_smem (scale→0)
                    # and is treated as OOB by the TMA gather4 (writes 0).
                    # WG3's kv_valid_producer masks scores ≥ valid_topk
                    # to -inf via topk_lengths.
                    h_indices[bs * topk + i] = UInt32(0xFFFFFFFF)

    var indices_device = ctx.enqueue_create_buffer[.uint32](total_indices)
    ctx.enqueue_copy(indices_device, h_indices)

    # Reported length: `valid_topk` by default, or a constant override
    # (decoupled from valid_topk) to reproduce the indexer's broadcast regime.
    var reported_len = (
        valid_topk if topk_lengths_override < 0 else topk_lengths_override
    )
    var h_topk_lengths = alloc[UInt32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(reported_len)

    var topk_lengths_device = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    ctx.enqueue_copy(topk_lengths_device, h_topk_lengths)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build FP8 PagedKVCacheCollection.
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[.float8_e4m3fn, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn, kv_params, PAGE_SIZE
    ](
        LayoutTensor[.float8_e4m3fn, Layout.row_major[6]()](
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
    # Build TileTensors.
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
    # Call mla_prefill_sparse_fp8.
    # -----------------------------------------------------------------------
    print("  Launching mla_prefill_sparse_fp8...")

    comptime cta_group = 2 if num_heads == 128 else 1
    comptime b_topk = 128 if num_heads == 128 else 64
    comptime config = MLASparseConfig[
        DType.bfloat16, b_topk_=b_topk, cta_group_=cta_group
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        indices_stride=topk,
        group=num_heads,
    )

    # Optional attention sink (one Float32 per query head). Empty -> `None`;
    # non-empty -> device buffer of exactly `num_heads` Float32 (a broken
    # sub-64 padded-row guard reading sink[num_heads..63] is a real OOB).
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

    mla_prefill_sparse_fp8[
        config=config,
        group=group,
        q_depth=QK_DEPTH,
        scale_block_size=scale_block_size,
    ](
        out_tt,
        q_tt,
        kv_cache,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        scales_device.unsafe_ptr().bitcast[Float32]().as_unsafe_any_origin(),
        scale,
        Int32(topk),
        ctx,
    )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output.
    # -----------------------------------------------------------------------
    var out_host = alloc[BFloat16](out_elems)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # atol sits just above the measured floor (max_err <= 0.002 non-sink,
    # <= 0.0176 for the -inf-sink cases; stable — default-seeded RNG):
    # FP8 quantization noise is common-mode here because the host
    # reference consumes the same quantize-then-dequantized KV values the
    # kernel does, so the honest disagreement floor is ~bf16 rounding,
    # same as the BF16 test. Earlier revisions used 0.32-0.55 values
    # calibrated to real kernel defects (a gather4 staging-pitch race on
    # top of the BF16 defects) — never widen this to match an observed
    # baseline.
    var max_err = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
    var total_checked = 0
    # Non-finite counters + finite-only secondary-metric accumulators (see
    # the checks after the loop). The pure max-abs check alone silently
    # ignores NaN: abs(NaN-ref)=NaN and NaN>atol is False.
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
                    if err > max_err:
                        max_err = err
                    if abs(actual_val) > max_actual:
                        max_actual = abs(actual_val)
                    if abs(actual_val) > 1e-6:
                        num_nonzero += 1
                    total_checked += 1
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

    # Fail hard on ANY non-finite output/reference. This is the check the
    # pure max-abs test lacked: abs(NaN - ref) = NaN and `NaN > atol` is
    # False, so a NaN-producing FP8 kernel used to report PASSED (the FP8
    # V-dequant coverage-gap bug this suite now guards against).
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
    # First-principles error budget: the oracle consumes the SAME
    # quantize-then-dequantized KV values as the kernel, so FP8 quant noise
    # is common-mode and the honest disagreement floor is ~bf16 rounding
    # (cosine measured 0.999997). A low cosine here means a real kernel
    # defect — a gather4 staging-pitch race once capped this metric at
    # ~0.4 behind gates fitted to the "measured" baseline. Do NOT re-fit
    # these to an observed baseline.
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
    _ = scales_device
    _ = sink_device

    blocks_fp8_host.free()
    kv_bf16_host.free()
    kv_dequant_host.free()
    lookup_table_host.free()
    cache_lengths_host.free()
    q_host.free()
    kv_sparse.free()
    selected_tokens.free()
    ref_host.free()
    out_host.free()
    h_indices.free()
    h_topk_lengths.free()
    scales_host.free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            run_test_prefill_sparse_fp8[128, 128, QK_DEPTH](
                "b1_s32_h128_kv512_topk128_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
            )

            # Multi-batch FP8: mirrors the BF16 b4 shape.
            run_test_prefill_sparse_fp8[128, 128, QK_DEPTH](
                "b4_s16_h128_kv256_topk128_fp8_tensorwise",
                4,
                16,
                256,
                ctx,
            )

            # Multi-block FP8: topk=256 = 2 × B_TOPK, exercises cross-block
            # softmax state update in the FP8 path.
            run_test_prefill_sparse_fp8[128, 256, QK_DEPTH](
                "b1_s32_h128_kv512_topk256_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
            )

            # Production-flavored FP8 shape (scaled down).
            run_test_prefill_sparse_fp8[128, 256, QK_DEPTH](
                "b1_s64_h128_kv1024_topk256_fp8_tensorwise_prodlike",
                1,
                64,
                1024,
                ctx,
            )

            # Padded FP8: single-block, half masked out.
            run_test_prefill_sparse_fp8[128, 128, QK_DEPTH](
                "b1_s32_h128_kv512_topk128_valid64_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                valid_topk=64,
            )

            # Padded FP8: multi-block, second block entirely masked.
            run_test_prefill_sparse_fp8[128, 256, QK_DEPTH](
                "b1_s32_h128_kv512_topk256_valid128_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                valid_topk=128,
            )

            # Padded FP8: multi-block, partial second block masked.
            run_test_prefill_sparse_fp8[128, 256, QK_DEPTH](
                "b1_s32_h128_kv512_topk256_valid192_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                valid_topk=192,
            )

            # ---------------------------------------------------------------
            # FP8 head64 path (GLM): cta_group=1, B_TOPK=64.  convert_v is
            # parametrized on B_TOPK (ROWS_PER_KH), so V dequant stays inside
            # the 32-row key-half.
            # ---------------------------------------------------------------

            # Exact 1 k-block (topk == B_TOPK == 64).
            run_test_prefill_sparse_fp8[64, 64, QK_DEPTH](
                "b1_s32_h64_kv256_topk64_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
            )

            # Exact 2 k-blocks (cross-block online-softmax state in FP8).
            run_test_prefill_sparse_fp8[64, 128, QK_DEPTH](
                "b1_s32_h64_kv512_topk128_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
            )

            # Ragged tail < B_TOPK: topk=64, valid_topk=40 poisons
            # positions [40..64) inside the single block.
            run_test_prefill_sparse_fp8[64, 64, QK_DEPTH](
                "b1_s32_h64_kv256_topk64_valid40_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                valid_topk=40,
            )

            # Prime valid_topk (37): non-aligned mask boundary, FP8 path.
            run_test_prefill_sparse_fp8[64, 64, QK_DEPTH](
                "b1_s32_h64_kv256_topk64_valid37_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                valid_topk=37,
                atol=0.55,
            )

            # Large: 16 k-blocks (topk=1024) exercises the deep k loop with
            # the FP8 K/V dequant on every block.
            run_test_prefill_sparse_fp8[64, 1024, QK_DEPTH](
                "b1_s8_h64_kv1024_topk1024_fp8_tensorwise",
                1,
                8,
                1024,
                ctx,
            )

            # ---------------------------------------------------------------
            # FP8 sub-64-head paths (GLM TP{8,4,2} = {8,16,32}).  The Q load
            # is shared with the BF16 path via `_load_q_prologue` (Q is BF16 in
            # both); only the FP8 K/V dequant differs.  h8 gets full coverage;
            # h16/h32/h24 a representative multi-block case each.
            # ---------------------------------------------------------------

            # h8: exact 1 k-block, smallest grid (num_q_rows=1).
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s1_h8_kv256_topk64_fp8_tensorwise",
                1,
                1,
                256,
                ctx,
            )

            # h8: exact 1 k-block, small num_q_rows.
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s32_h8_kv256_topk64_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
            )

            # h8: multi-batch, exact 1 k-block (num_q_rows = 4*16 = 64).
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b4_s16_h8_kv256_topk64_fp8_tensorwise",
                4,
                16,
                256,
                ctx,
            )

            # h8: exact 2 k-blocks (cross-block online-softmax state in FP8).
            run_test_prefill_sparse_fp8[8, 128, QK_DEPTH](
                "b1_s32_h8_kv512_topk128_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
            )

            # h8: multi-block, 4 k-blocks (topk=256).
            run_test_prefill_sparse_fp8[8, 256, QK_DEPTH](
                "b1_s32_h8_kv1024_topk256_fp8_tensorwise",
                1,
                32,
                1024,
                ctx,
            )

            # h8: ragged tail < B_TOPK (valid_topk=40 poisons [40..64)).
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s32_h8_kv256_topk64_valid40_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                valid_topk=40,
            )

            # h8: multi-block ragged (valid_topk=96 fires mid 2nd block).
            run_test_prefill_sparse_fp8[8, 128, QK_DEPTH](
                "b1_s32_h8_kv512_topk128_valid96_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                valid_topk=96,
            )

            # h8: all-invalid query (valid_topk=0) -> O=0.
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s32_h8_kv256_topk64_valid0_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                valid_topk=0,
            )

            # h8: deep 16 k-blocks (topk=1024).
            run_test_prefill_sparse_fp8[8, 1024, QK_DEPTH](
                "b1_s8_h8_kv1024_topk1024_fp8_tensorwise",
                1,
                8,
                1024,
                ctx,
            )

            # h8: attention sink (finite) — device buffer exactly 8 Float32.
            var sink_fp8_h8_finite: List[Float32] = [
                0.5,
                1.0,
                1.5,
                2.0,
                0.5,
                1.0,
                1.5,
                2.0,
            ]
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s32_h8_kv256_topk64_sink_finite_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                sink_values=sink_fp8_h8_finite,
            )

            # h8: attention sink (-inf) — inert; proves the padded-row guard
            # (run under compute-sanitizer: no OOB read of sink[8..63]).
            var sink_fp8_h8_neginf = List[Float32]()
            for _ in range(8):
                sink_fp8_h8_neginf.append(Float32(min_or_neg_inf[.float32]()))
            run_test_prefill_sparse_fp8[8, 64, QK_DEPTH](
                "b1_s32_h8_kv256_topk64_sink_neginf_fp8_tensorwise",
                1,
                32,
                256,
                ctx,
                sink_values=sink_fp8_h8_neginf,
            )

            # h16: representative multi-block.
            run_test_prefill_sparse_fp8[16, 256, QK_DEPTH](
                "b1_s32_h16_kv1024_topk256_fp8_tensorwise",
                1,
                32,
                1024,
                ctx,
            )

            # h32: representative multi-block.
            run_test_prefill_sparse_fp8[32, 256, QK_DEPTH](
                "b1_s32_h32_kv1024_topk256_fp8_tensorwise",
                1,
                32,
                1024,
                ctx,
            )

            # h24 (Part C): non-power-of-2 multiple of 8 generalization.
            run_test_prefill_sparse_fp8[24, 64, QK_DEPTH](
                "b1_s32_h24_kv256_topk64_fp8_tensorwise",
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
            run_test_prefill_sparse_fp8[8, 128, QK_DEPTH](
                "b1_s32_h8_kv512_topk128_L2_layer1_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
            run_test_prefill_sparse_fp8[8, 128, QK_DEPTH](
                "b1_s32_h8_kv512_topk128_L2_layer0_control_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=0,
            )
            run_test_prefill_sparse_fp8[64, 128, QK_DEPTH](
                "b1_s32_h64_kv512_topk128_L2_layer1_fp8_tensorwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )

            # ===============================================================
            # FP8 blockwise (cache-native) scaling: scale_block_size=32.
            #
            # The MLA FP8 latent cache stores ceildiv(qk_depth=576, 32)=18
            # Float32 scales per token over the full latent row. K consumes
            # all 18; V (the 512-wide nope part) consumes the first
            # ceildiv(v_depth=512, 32)=16 blocks of that SAME per-token
            # vector. Both index the HBM buffer with stride 18 (the cache
            # stride). The harness quantizes per 32-col block and lays scales
            # out exactly this way, so these cases are the discriminator for
            # the blockwise gather/convert fixes (V HBM stride + the
            # cta_group=2 V-convert latent-column mapping).
            # ===============================================================

            # h128 (cta_group=2), single k-block. THE guard for the V-convert
            # latent-column mapping: at cta_group=2 V is column-split across
            # CTAs, so block_idx must be derived from the absolute latent col,
            # not the within-CTA smem col. RED without the convert_v fix.
            run_test_prefill_sparse_fp8[128, 128, 32](
                "b1_s32_h128_kv512_topk128_fp8_blockwise",
                1,
                32,
                512,
                ctx,
            )

            # h128 (cta_group=2), 2 k-blocks: blockwise across the cross-block
            # online-softmax fold.
            run_test_prefill_sparse_fp8[128, 256, 32](
                "b1_s32_h128_kv512_topk256_fp8_blockwise",
                1,
                32,
                512,
                ctx,
            )

            # h128 blockwise, padded: second block fully sentinel — exercises
            # the -1-sentinel scale→0 path in the blockwise K/V scale gather.
            run_test_prefill_sparse_fp8[128, 256, 32](
                "b1_s32_h128_kv512_topk256_valid128_fp8_blockwise",
                1,
                32,
                512,
                ctx,
                valid_topk=128,
            )

            # h64 (cta_group=1): guards the V HBM-stride fix (V now reads the
            # cache stride 18, not V's own 16). K stride was already correct.
            run_test_prefill_sparse_fp8[64, 128, 32](
                "b1_s32_h64_kv512_topk128_fp8_blockwise",
                1,
                32,
                512,
                ctx,
            )

            # h8 (sub-64, cta_group=1) blockwise.
            run_test_prefill_sparse_fp8[8, 128, 32](
                "b1_s32_h8_kv512_topk128_fp8_blockwise",
                1,
                32,
                512,
                ctx,
            )

            # Multilayer blockwise: combines the per-layer paged-cache row
            # addressing (get_tma_row fold) with the blockwise scale stride.
            # layer_idx=1 is the guard; layer_idx=0 is the control.
            run_test_prefill_sparse_fp8[128, 128, 32](
                "b1_s32_h128_kv512_topk128_L2_layer1_fp8_blockwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
            run_test_prefill_sparse_fp8[128, 128, 32](
                "b1_s32_h128_kv512_topk128_L2_layer0_control_fp8_blockwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=0,
            )
            run_test_prefill_sparse_fp8[64, 128, 32](
                "b1_s32_h64_kv512_topk128_L2_layer1_fp8_blockwise",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )

            # Production-regime blockwise: topk=2048 (32 k-blocks) with the
            # indexer's constant-length broadcast (topk_lengths_override=2048).
            # Covers heads {8, 64, 128} — the GLM/DSv3.2 TP shards.
            run_test_prefill_sparse_fp8[8, 2048, 32](
                "b1_s8_h8_len2048_valid2048_fp8_blockwise",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            run_test_prefill_sparse_fp8[64, 2048, 32](
                "b1_s8_h64_len2048_valid2048_fp8_blockwise",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            run_test_prefill_sparse_fp8[128, 2048, 32](
                "b1_s8_h128_len2048_valid2048_fp8_blockwise",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            # Constant length 2048 with only 96 valid: long all-sentinel tail
            # decoupled from the reported length (the indexer regime) under
            # blockwise scaling.
            run_test_prefill_sparse_fp8[8, 2048, 32](
                "b1_s8_h8_len2048_valid96_override2048_fp8_blockwise",
                1,
                8,
                2048,
                ctx,
                valid_topk=96,
                topk_lengths_override=2048,
            )

            # ===============================================================
            # FP8 no-scale (unit-scale) reads: scale_block_size=0.
            #
            # Today's DSv3.2 MLA latent FP8 cache stores NO dequant scales
            # (scale_dtype=int8 placeholder => quantization disabled), so the
            # sparse-decode kernel reads it at unit scale. The graph flip
            # routes FP8-cache sparse prefill to this same no-scale kernel
            # path. The harness quantizes the latents at s=1.0 and the oracle
            # dequantizes at s=1.0, so these are genuine discriminators: a
            # kernel that applied any scale (or read uninitialized scale SMEM)
            # would diverge from the unit-scale reference. Covers the GLM/
            # DSv3.2 TP shards {8, 16-not-needed, 32-not-needed, 64, 128}.
            # ===============================================================

            # h128 (cta_group=2): exercises the no-scale convert_v path (the
            # scale gather + latent-col scale lookup are skipped).
            run_test_prefill_sparse_fp8[128, 128, 0](
                "b1_s32_h128_kv512_topk128_fp8_noscale",
                1,
                32,
                512,
                ctx,
            )
            # h128 multi-block (cross-block online-softmax fold, no-scale).
            run_test_prefill_sparse_fp8[128, 256, 0](
                "b1_s32_h128_kv512_topk256_fp8_noscale",
                1,
                32,
                512,
                ctx,
            )
            # h128 padded: second block fully sentinel (data-gather skip).
            run_test_prefill_sparse_fp8[128, 256, 0](
                "b1_s32_h128_kv512_topk256_valid128_fp8_noscale",
                1,
                32,
                512,
                ctx,
                valid_topk=128,
            )
            # h64 (cta_group=1) no-scale.
            run_test_prefill_sparse_fp8[64, 128, 0](
                "b1_s32_h64_kv512_topk128_fp8_noscale",
                1,
                32,
                512,
                ctx,
            )
            # h8 (sub-64) no-scale.
            run_test_prefill_sparse_fp8[8, 128, 0](
                "b1_s32_h8_kv512_topk128_fp8_noscale",
                1,
                32,
                512,
                ctx,
            )
            # Multilayer no-scale: per-layer paged row addressing under the
            # no-scale read. layer_idx=1 is the guard; layer_idx=0 the control.
            run_test_prefill_sparse_fp8[128, 128, 0](
                "b1_s32_h128_kv512_topk128_L2_layer1_fp8_noscale",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
            run_test_prefill_sparse_fp8[128, 128, 0](
                "b1_s32_h128_kv512_topk128_L2_layer0_control_fp8_noscale",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=0,
            )
            run_test_prefill_sparse_fp8[64, 128, 0](
                "b1_s32_h64_kv512_topk128_L2_layer1_fp8_noscale",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )
            run_test_prefill_sparse_fp8[8, 128, 0](
                "b1_s32_h8_kv512_topk128_L2_layer1_fp8_noscale",
                1,
                32,
                512,
                ctx,
                num_layers=2,
                layer_idx=1,
            )

            # Production-regime no-scale: topk=2048 (32 k-blocks) with the
            # indexer's constant-length broadcast, heads {8, 64, 128}. This is
            # the shape the graph flip actually routes for DSv3.2 prefill.
            run_test_prefill_sparse_fp8[8, 2048, 0](
                "b1_s8_h8_len2048_valid2048_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            run_test_prefill_sparse_fp8[64, 2048, 0](
                "b1_s8_h64_len2048_valid2048_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
            run_test_prefill_sparse_fp8[128, 2048, 0](
                "b1_s8_h128_len2048_valid2048_fp8_noscale",
                1,
                8,
                2048,
                ctx,
                valid_topk=2048,
                topk_lengths_override=2048,
            )
        else:
            pass
