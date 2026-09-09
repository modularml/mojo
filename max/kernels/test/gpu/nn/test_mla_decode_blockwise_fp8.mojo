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

"""Test MLA decode with blockwise FP8 quantization (scale_block_size > 0).

This test exercises the blockwise scaling path in the SM100 MLA decode kernel
with comprehensive per-block and per-token scale variation.  Key aspects:

  - Non-uniform per-block scales: each block of channels within a token uses
    a different power-of-2 scale, verifying the kernel applies the correct
    scale to the correct block of channels.
  - Non-uniform per-token scales: different tokens carry different scales,
    verifying row-level indexing in _load_scales_for_tile.
  - Multiple quantization granularities (32, 64, 128).
  - Variable cache lengths across batch entries including page boundaries.
  - Edge cases: very short cache, exact page boundary, single token.

Scale values are restricted to power-of-2 values (0.25, 0.5, 1.0, 2.0, 4.0,
8.0, etc.) because the e8m0 format only stores 8-bit biased exponents and
these values are exact.
"""

from std.math import ceildiv
from std.random import randn, seed
from std.sys import argv, has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
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
from std.memory import alloc
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import NullMask
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_almost_equal
from max.gpu.host.info import B200, _is_sm10x_gpu
from std.utils.index import Index, IndexList


# ===-----------------------------------------------------------------------===#
# Test parameters matching DeepSeek-V2-Lite
# ===-----------------------------------------------------------------------===#

comptime DEPTH = 576  # Q/K head dimension
comptime V_DEPTH = 512  # Output head dimension (depth - 64)
comptime NUM_LAYERS = 1  # Single layer for testing
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


def _palette_scale(index: Int) -> Float32:
    """Pick a scale from the palette by index (wrapping)."""
    if index % 9 == 0:
        return 0.25
    if index % 9 == 1:
        return 0.5
    if index % 9 == 2:
        return 1.0
    if index % 9 == 3:
        return 2.0
    if index % 9 == 4:
        return 4.0
    if index % 9 == 5:
        return 8.0
    if index % 9 == 6:
        return 16.0
    if index % 9 == 7:
        return 0.125
    return 0.0625


# ===-----------------------------------------------------------------------===#
# Core test: paged KV cache with blockwise FP8 scaling + numerical verification
# ===-----------------------------------------------------------------------===#


def run_test_blockwise_fp8[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
    page_size: Int,
    quant_granularity: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    comptime head_dim_gran = ceildiv(DEPTH, quant_granularity)

    var batch_size = len(cache_lengths)
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " q_type:",
        q_type,
        " kv_type:",
        kv_type,
        " page_size:",
        page_size,
        " quant_granularity:",
        quant_granularity,
        " scales_per_token:",
        head_dim_gran,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    # All seq_lens = 1 (standard decode)
    comptime q_max_seq_len = 1
    var total_q_tokens = batch_size  # Each batch has 1 query token

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        # Allocate pages like production: ceildiv(cache_len + 1, page_size).
        # The +1 accounts for the new token that is written at position
        # cache_len by the fused_rope_rmsnorm kernel BEFORE attention runs.
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, page_size)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache blocks on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)

    # Generate FP8 data: generate bf16 first, then cast to fp8.
    # Use small stddev to keep values in FP8 range.
    var blocks_bf16 = ctx.enqueue_create_host_buffer[q_type](block_elems)
    randn(
        blocks_bf16.as_span(),
        mean=0.0,
        standard_deviation=0.3,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    # -----------------------------------------------------------------------
    # Step 1b: Create the scales tensor on host with NON-UNIFORM per-block,
    #          per-token scales
    # -----------------------------------------------------------------------
    # Scales 6D shape: [total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, num_heads, head_dim_gran]
    var scales_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        head_dim_gran,
    )
    var scales_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * head_dim_gran
    )

    var scales_host = ctx.enqueue_create_host_buffer[.float32](scales_elems)

    # Assign non-uniform per-block, per-token scales.
    # scale(token_global, block_idx) = palette[(token_global * 7 + block_idx) % 9]
    # The factor 7 is coprime to 9, ensuring all palette entries are exercised
    # even for small token counts.
    var scale_tok_stride = kv_params.num_heads * head_dim_gran
    var scale_page_stride = (
        kv_dim2 * NUM_LAYERS * page_size * kv_params.num_heads * head_dim_gran
    )
    var global_tok = 0
    var cur_page = 0
    for bi in range(batch_size):
        var cl = cache_lengths[bi]
        # Match production: allocate pages for cache_len + 1 tokens.
        var num_keys_i = cl + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, page_size)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * page_size
            if valid_toks > page_size:
                valid_toks = page_size
            for tok_in_page in range(page_size):
                for h in range(kv_params.num_heads):
                    for blk in range(head_dim_gran):
                        var offset = (
                            cur_page * scale_page_stride
                            + tok_in_page * scale_tok_stride
                            + h * head_dim_gran
                            + blk
                        )
                        if tok_in_page < valid_toks:
                            # Non-uniform: varies by token AND block
                            var tok_global = (
                                global_tok + pg * page_size + tok_in_page
                            )
                            scales_host[offset] = _palette_scale(
                                tok_global * 7 + blk
                            )
                        else:
                            # Neutral scale for unused token slots
                            scales_host[offset] = 1.0
            cur_page += 1
        global_tok += num_keys_i

    # -----------------------------------------------------------------------
    # Step 1c: Build lookup table and zero out unused token slots
    # -----------------------------------------------------------------------

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Match production: max_pages_per_batch covers cache_len + 1 tokens.
    var max_pages_per_batch = ceildiv(max_cache_len + q_max_seq_len, page_size)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    # Zero-init: unused per-batch slots must be 0.
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        # Match production: include the page for the new token.
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, page_size)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond num_keys).
    # The new token at position cache_len has real randn data (written by
    # the fused_rope_rmsnorm kernel in production). Tokens beyond num_keys
    # are zeroed so TMA reads produce zeros that get masked out.
    var _tok_stride = kv_params.num_heads * kv_params.head_size
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )
    cur_page = 0
    for bi in range(batch_size):
        var cl = cache_lengths[bi]
        var num_keys_i = cl + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, page_size)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * page_size
            if valid_toks > page_size:
                valid_toks = page_size
            # Zero out tokens [valid_toks, PAGE_SIZE) in this page
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (page_size - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -----------------------------------------------------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -----------------------------------------------------------------------
    # Step 4: Output tensor
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    ctx.enqueue_copy(scales_device, scales_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection on device
    # -----------------------------------------------------------------------

    # Blocks LayoutTensor
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    # Scales LayoutTensor (6D, same layout pattern as blocks)
    var scales_lt = LayoutTensor[.float32, Layout.row_major[6]()](
        scales_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(scales_shape),
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

    # Build PagedKVCacheCollection with quantization_granularity and scales.
    var kv_collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=quant_granularity,
    ](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
        # Pass the scales tensor
        LayoutTensor[.float32, Layout.row_major[6]()](
            scales_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                scales_lt.runtime_layout.shape.value,
                scales_lt.runtime_layout.stride.value,
            ),
        ),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
    )

    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_tt = TileTensor(
        row_offsets_device,
        row_major(batch_size + 1),
    )

    # -----------------------------------------------------------------------
    # Step 7: Pre-compute scalar args and call the kernel
    # -----------------------------------------------------------------------
    var mla_args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size,
        max_cache_len,
        1,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA decode kernel (blockwise FP8)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, DEPTH),
        ragged=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        q_max_seq_len=q_max_seq_len,
    )

    ctx.synchronize()
    print("  Kernel completed successfully (no crash).")

    _ = mla_args

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch (with dequantization)...")

    # Tolerances: blockwise FP8 has more quantization noise than tensorwise.
    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
    var total_checked = 0
    var max_abs_err = Float64(0)

    # Run mha_gpu_naive per batch with dequantized K.
    #
    # The MLA kernel with _is_cache_length_accurate=False processes
    # num_keys = cache_len + q_max_seq_len tokens. The new token at
    # position cache_len has real data and a real scale (written by the
    # fused_rope_rmsnorm kernel in production, and by randn in this test).
    # The reference dequantizes all num_keys tokens from the paged blocks.
    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var ref_num_keys = cache_len + q_max_seq_len

        # Extract dequantized K for this batch from paged blocks + scales.
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = ctx.enqueue_create_host_buffer[q_type](k_b_size)

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, page_size)

        for tok in range(ref_num_keys):
            var page_idx, tok_in_page = divmod(tok, page_size)
            var physical_page = page_base_b + page_idx

            for h in range(KV_NUM_HEADS):
                for d in range(DEPTH):
                    var src_offset = (
                        physical_page
                        * kv_dim2
                        * NUM_LAYERS
                        * page_size
                        * kv_params.num_heads
                        * kv_params.head_size
                        + tok_in_page
                        * kv_params.num_heads
                        * kv_params.head_size
                        + h * kv_params.head_size
                        + d
                    )
                    var fp8_val = blocks_host[src_offset].cast[.float32]()

                    var block_idx = d // quant_granularity
                    var scale_offset = (
                        physical_page * scale_page_stride
                        + tok_in_page * scale_tok_stride
                        + h * head_dim_gran
                        + block_idx
                    )
                    var block_scale = scales_host[scale_offset]

                    var dequant_val = (fp8_val * block_scale).cast[q_type]()

                    var dst_offset = tok * KV_NUM_HEADS * DEPTH + h * DEPTH + d
                    k_b_host[dst_offset] = dequant_val

        # Q for this batch: [1, 1, num_heads, depth]
        var q_b_size = 1 * num_heads * DEPTH
        var q_b_host = ctx.enqueue_create_host_buffer[q_type](q_b_size)
        for i in range(q_b_size):
            q_b_host[i] = q_host[b * num_heads * DEPTH + i]

        # Reference output: [1, 1, num_heads, depth] (full depth)
        var ref_b_size = 1 * num_heads * DEPTH
        var ref_b_host = ctx.enqueue_create_host_buffer[q_type](ref_b_size)

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[q_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        ctx.synchronize()

        # Build 4D LayoutTensors: batch_size=1
        comptime layout_4d = Layout.row_major[4]()
        var q_b_lt = LayoutTensor[q_type, layout_4d](
            q_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(Index(1, 1, num_heads, DEPTH)),
        )
        var k_b_lt = LayoutTensor[q_type, layout_4d](
            k_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(
                Index(1, ref_num_keys, KV_NUM_HEADS, DEPTH)
            ),
        )
        var ref_b_lt = LayoutTensor[q_type, layout_4d](
            ref_b_device.unsafe_ptr(),
            RuntimeLayout[layout_4d].row_major(Index(1, 1, num_heads, DEPTH)),
        )

        # Run mha_gpu_naive: batch_size=1, num_keys=ref_num_keys
        # K passed as both K and V (MLA: V = K[:,:,:512])
        # Note: K is already dequantized bf16
        mha_gpu_naive(
            q_b_lt,
            k_b_lt,
            k_b_lt,
            NullMask(),
            ref_b_lt,
            scale,
            1,  # batch_size
            1,  # seq_len
            ref_num_keys,  # num_keys
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims (depth-64) per head
        # ref layout: [1, 1, num_heads, depth]
        # actual layout: [total_tokens, num_heads, V_DEPTH]
        var out_offset = b * num_heads * V_DEPTH
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host[d + DEPTH * h].cast[.float64]()
                var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                    DType.float64
                ]()
                var abs_err = abs(actual - expect)
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if abs_err > 1e-1:
                    print(b, h, d, actual, expect)
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

        total_checked += num_heads * V_DEPTH

        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    _ = blocks_device
    _ = scales_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===-----------------------------------------------------------------------===#
# Benchmark: blockwise FP8 paged KV cache (no numerical verification)
# ===-----------------------------------------------------------------------===#


def run_bench_blockwise_fp8[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
    page_size: Int,
    quant_granularity: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    comptime head_dim_gran = ceildiv(DEPTH, quant_granularity)

    var batch_size = len(cache_lengths)
    print(
        "bench:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " quant_granularity:",
        quant_granularity,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    comptime q_max_seq_len = 1
    var total_q_tokens = batch_size

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, page_size)

    comptime scale = Float32(0.125)

    # Step 1: Paged KV cache blocks on host
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    var blocks_bf16 = ctx.enqueue_create_host_buffer[q_type](block_elems)
    randn(
        blocks_bf16.as_span(),
        mean=0.0,
        standard_deviation=0.3,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    # Scales tensor
    var scales_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        head_dim_gran,
    )
    var scales_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * head_dim_gran
    )
    # Use uniform scale=1.0 for benchmark (perf is scale-value independent)
    var scales_host = ctx.enqueue_create_host_buffer[.float32](scales_elems)
    scales_host.enqueue_fill(Float32(1.0))

    var _tok_stride = kv_params.num_heads * kv_params.head_size
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * page_size
        * kv_params.num_heads
        * kv_params.head_size
    )

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = ceildiv(max_cache_len + q_max_seq_len, page_size)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    # Zero-init: unused per-batch slots must be 0.
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, page_size)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, page_size)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * page_size
            if valid_toks > page_size:
                valid_toks = page_size
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (page_size - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # Step 2: Q tensor
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Step 3: input_row_offsets
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # Step 4: Output tensor
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)

    # Step 5: Copy to device
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    ctx.enqueue_copy(scales_device, scales_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # Step 6: Build LayoutTensors and PagedKVCacheCollection
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    var scales_lt = LayoutTensor[.float32, Layout.row_major[6]()](
        scales_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(scales_shape),
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
        kv_type,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=quant_granularity,
    ](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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
        LayoutTensor[.float32, Layout.row_major[6]()](
            scales_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                scales_lt.runtime_layout.shape.value,
                scales_lt.runtime_layout.stride.value,
            ),
        ),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH])),
    )

    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_tt = TileTensor(
        row_offsets_device,
        row_major(batch_size + 1),
    )

    # Step 7: Pre-compute scalar args and benchmark
    var mla_args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size,
        max_cache_len,
        1,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {
        var out_tt,
        var q_tt,
        var kv_cache,
        var row_offsets_tt,
        var scalar_args_buf_tt,
        imm,
    }:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            q_max_seq_len=q_max_seq_len,
        )

    comptime nrun = 200

    # Warmup
    kernel_launch(ctx)

    var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
        nrun
    )
    var mstime = nstime / 1000000
    print("  ", nrun, "runs avg", mstime, "ms")
    print()

    _ = mla_args
    _ = blocks_device
    _ = scales_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def make_uniform(count: Int, value: Int) -> List[Int]:
    """Create a list of `count` identical cache lengths."""
    var result = List[Int]()
    for _ in range(count):
        result.append(value)
    return result^


def run_bench_uniform[
    num_heads: Int,
    page_size: Int,
    quant_granularity: Int,
](
    config_name: StringLiteral,
    count: Int,
    value: Int,
    ctx: DeviceContext,
) raises:
    """Run blockwise FP8 benchmark with uniform cache lengths."""
    run_bench_blockwise_fp8[
        DType.bfloat16,
        DType.float8_e4m3fn,
        num_heads,
        page_size,
        quant_granularity,
    ](config_name, make_uniform(count, value), ctx)


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def main() raises:
    seed(42)

    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            if is_benchmark():
                # -----------------------------------------------------------
                # Benchmark mode: time kernel execution, no verification
                # DeepSeek-V3/R1: 128 heads, depth=576
                # Blockwise FP8 (granularity=64)
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Blockwise FP8 BENCHMARK (B200)")
                print("=" * 72)
                print()

                # Batch size 1: single long context
                run_bench_uniform[128, 128, 64]("bs1_32k_bw64", 1, 32768, ctx)

                # Batch size 1: medium context
                run_bench_uniform[128, 128, 64]("bs1_4k_bw64", 1, 4096, ctx)

                # Batch size 2: variable lengths
                var b2: List[Int] = [4096, 32768]
                run_bench_blockwise_fp8[
                    DType.bfloat16,
                    DType.float8_e4m3fn,
                    128,
                    128,
                    64,
                ]("bs2_4k_32k_bw64", b2, ctx)

                # Batch size 4: mixed lengths
                var b4: List[Int] = [1024, 4096, 16384, 32768]
                run_bench_blockwise_fp8[
                    DType.bfloat16,
                    DType.float8_e4m3fn,
                    128,
                    128,
                    64,
                ]("bs4_mixed_bw64", b4, ctx)

                # Batch size 8: mixed lengths (production-like)
                var b8: List[Int] = [
                    128,
                    512,
                    1024,
                    4096,
                    8192,
                    16384,
                    24576,
                    32768,
                ]
                run_bench_blockwise_fp8[
                    DType.bfloat16,
                    DType.float8_e4m3fn,
                    128,
                    128,
                    64,
                ]("bs8_mixed_bw64", b8, ctx)

                # Batch size 8: all long (worst case)
                run_bench_uniform[128, 128, 64](
                    "bs8_all32k_bw64", 8, 32768, ctx
                )

                print("=" * 72)
                print("BENCHMARK COMPLETE")
                print("=" * 72)

            else:
                # -----------------------------------------------------------
                # Correctness mode: numerical verification against reference
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Blockwise FP8 Scaling Test (B200)")
                print("=" * 72)
                print()

                # ===========================================================
                # Group 1: Per-block scale differentiation (granularity=64)
                # Most critical: verifies correct scale-to-channel mapping.
                # ===========================================================
                print(
                    "--- Group 1: Per-block scale differentiation"
                    " (granularity=64, page_size=128) ---"
                )

                # Short cache, fits in one page
                var cl_1a: List[Int] = [30, 50]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("perblock_short_g64", cl_1a, ctx)

                # Medium cache spanning multiple pages, including
                # page-aligned lengths.
                # 640 = 5*128: exactly 5 full pages, new token on page 6.
                var cl_1b: List[Int] = [256, 640]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("perblock_medium_g64", cl_1b, ctx)

                # Large cache with split-K (page-aligned)
                var cl_1c: List[Int] = [2048]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("perblock_large_g64", cl_1c, ctx)

                # ===========================================================
                # Group 2: Different quantization granularities
                # ===========================================================
                print("--- Group 2: Different quantization granularities ---")

                # granularity=128: 5 scales per token
                var cl_2a: List[Int] = [50, 300]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 128
                ]("perblock_g128", cl_2a, ctx)

                # granularity=32: 18 scales per token (finest granularity)
                var cl_2b: List[Int] = [50, 300]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 32
                ]("perblock_g32", cl_2b, ctx)

                # ===========================================================
                # Group 3: Short cache + large batch (split_page_size=64)
                # ===========================================================
                print(
                    "--- Group 3: Short cache, large batch"
                    " (split_page_size=64) ---"
                )

                var cl_3a = List[Int]()
                for _ in range(32):
                    cl_3a.append(64)
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("short_cache_large_batch_g64", cl_3a, ctx)

                # ===========================================================
                # Group 4: Variable cache lengths (edge cases)
                # ===========================================================
                print("--- Group 4: Variable cache lengths and edge cases ---")

                var cl_4a: List[Int] = [1, 2]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("edge_very_short", cl_4a, ctx)

                var cl_4b: List[Int] = [127, 255]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("edge_near_page_boundary", cl_4b, ctx)

                var cl_4c: List[Int] = [129, 257]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("edge_just_over_page", cl_4c, ctx)

                var cl_4d: List[Int] = [1, 2048]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("edge_disparity_short_long", cl_4d, ctx)

                var cl_4e: List[Int] = [4096]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 16, 128, 64
                ]("edge_single_batch_4k", cl_4e, ctx)

                # ===========================================================
                # Group 5: Higher head counts
                # ===========================================================
                print("--- Group 5: Higher head counts ---")

                var cl_5a: List[Int] = [30, 512]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 64, 128, 64
                ]("heads64_g64", cl_5a, ctx)

                var cl_5b: List[Int] = [30, 512]
                run_test_blockwise_fp8[
                    DType.bfloat16, DType.float8_e4m3fn, 128, 128, 64
                ]("heads128_g64", cl_5b, ctx)

                print("=" * 72)
                print("ALL BLOCKWISE FP8 TESTS PASSED")
                print("=" * 72)
        else:
            print("Skipping: requires B200 GPU")
